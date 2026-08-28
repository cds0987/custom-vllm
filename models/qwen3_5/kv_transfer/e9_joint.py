"""e9_joint -- KIEN TRUC 2 LOP (user chot 2026-08-28).

    4B + LoRA  --(cache)-->  mapper  -->  27B  --> CE(gold)
       ^_______________ gradient chay nguoc het duong _______|

Khac han moi vong train truoc (e6v3_ce.py): o do 4B chi la NGUON tinh mot
lan roi spill cache ra dia, 27B train mapper doc lai tu dia -- 4B khong bao
gio hoc gi. O day 4B CUNG HOC: LoRA cua no bi ep "doc ho cho 27B". Sau khi
train, merge LoRA vao 4B -> luc serve KHONG co module thua tren duong nong.

DIEU KIEN VAT LY (do that, probe_joint_lora.py, 6 luot):
  - hai model cung tren GPU = 16,16 GiB tinh, con trong 5,54 GiB
  - thu pham: prefill 4B CO GRAD (+3,34 GiB ngay o T=256)
  - gradient checkpointing BI LOAI VE NGUYEN TAC: transformers ep
    use_cache=False khi bat GC, ma forward 4B o day ton tai CHINH DE sinh
    cache -> hai thu loai tru nhau
  - TBPTT mo duoc cong: T-w token dau no_grad, w token cuoi co grad

HAI CAI BAY DA XU LY (cai thu 2 la bug that neu bo qua):
 1. Template 27B: build_template_from_meta can "bo xuong" shape cho DUNG do
    dai T. Chay teacher prefill moi buoc thi qua dat. Giai: suy meta tu mot
    meta goc (chi doi chieu T) + TU KIEM bang meta THAT o vai do dai
    (--verify-meta). Sai thi dung ngay, khong doan.
 2. CAT TRAI khi tokenize. Moi prompt trong bo nay dat CAU HOI O CUOI
    (musr: narrative 1000-1500 token roi moi hoi). truncation mac dinh cua
    tokenizer cat PHAI = cat mat cau hoi -> item thanh vo nghia ma van chay
    tron, khong bao loi. Bat buoc truncation_side='left'.

Loss: CE(gold) co trong so, KHONG co KL/aux/dense -- ba thanh phan do can
teacher logits tien tinh (B1), ma o che do joint thi cache thay doi moi buoc
nen khong tien tinh duoc. v3.1 da do: CE-gold la thanh phan quyet dinh.
"""

import argparse
import copy
import gc
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# PHAI patch TRUOC khi nap model (hoc phi 3 probe: 5.15 update_recurrent_state
# lam .copy_() IN-PLACE len state -> vo autograd khi state mang grad)
try:
    from transformers.cache_utils import LinearAttentionLayer
    _orig_urs = LinearAttentionLayer.update_recurrent_state

    def _urs(self, recurrent_states, state_idx=0, **kw):
        cur = (self.recurrent_states.get(state_idx)
               if isinstance(self.recurrent_states, dict) else None)
        if cur is not None and (cur.requires_grad or recurrent_states.requires_grad):
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states
        return _orig_urs(self, recurrent_states, state_idx, **kw)

    LinearAttentionLayer.update_recurrent_state = _urs
    print("patched update_recurrent_state (rebind khi co grad)", flush=True)
except ImportError as e:
    print("PATCH_IMPORT_FAIL", e, flush=True)

e5 = _load("e5_train")
gd = _load("gen_data")

WARM_P = 5
FIRST_W = 3.0
GOLD_MAX = 64
GMAX = {"bfcl": 16, "needle": 12, "ifstruct": 96, "pbtable": 64,
        "gsm8k": 256, "bbh": 24, "musr": 8,
        "suite_rag": 16, "suite_mid": 16, "suite_math": 16, "suite_swe": 16}
GEN_LEN = {"bfcl": 24, "needle": 16, "ifstruct": 160, "pbtable": 120,
           "gsm8k": 320, "bbh": 48, "musr": 24,
           "suite_rag": 24, "suite_mid": 24, "suite_math": 24, "suite_swe": 24}
NEW_KINDS = {"gsm8k", "bbh", "musr", "suite_rag", "suite_mid",
             "suite_math", "suite_swe"}
SEED = 7


def gib():
    return torch.cuda.max_memory_allocated() / 2**30


# BAO DO THAT (probe_joint_lora, 42 cau hinh, 2026-08-28). gold = SO VI TRI
# feed vao 27B; moi vi tri giu them state GDN cho backward nen no dat hon
# ctx nhieu. Tuong CUNG o gold=64 voi MOI T (256..2048) — khong phai gioi han
# tron, la nguong cap phat.
#
#     T      gold toi da    peak GiB
#     256        48           20,75
#     512        48           20,92
#    1024        48           21,24
#    1536        16           20,97
#    2048        16           21,30
GOLD_ENVELOPE = [(1024, 48), (2048, 16)]


def gold_cap_for(t, hard_cap):
    for t_max, g in GOLD_ENVELOPE:
        if t <= t_max:
            return min(g, hard_cap)
    return min(GOLD_ENVELOPE[-1][1], hard_cap)


# ---------------------------------------------------------------- template --

def meta_for_len(base, t_base, t):
    """Suy 'bo xuong' template cho do dai t tu meta goc do o t_base.

    Chi co shape cua keys/values (chieu T) va vai truong int bang t_base la
    phu thuoc do dai; state GDN thi khong. Ham nay PHAI qua verify_meta()
    truoc khi tin -- neu co truong int nao trung t_base mot cach tinh co thi
    kiem tra se bat duoc."""
    m = copy.deepcopy(base)
    m["cache_ints"] = {k: (t if v == t_base else v)
                       for k, v in m["cache_ints"].items()}
    for lay in m["layers"]:
        lay["ints"] = {k: (t if v == t_base else v)
                       for k, v in lay["ints"].items()}
        if lay["kind"] == "a":
            for key in ("k", "v"):
                sh, dt = lay[key]
                lay[key] = (tuple(t if d == t_base else d for d in sh), dt)
    return m


def verify_meta(model_t, base, t_base, lens):
    """Doi chieu meta SUY RA voi meta THAT (chay prefill that o tung do dai).
    Sai mot li = template lech vi tri = cache vo nghia ma khong bao loi."""
    ok = True
    for t in lens:
        ids = torch.randint(1000, 5000, (1, t), device="cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_t, ids)
        real = e5.cache_meta(past)
        derived = meta_for_len(base, t_base, t)
        if real != derived:
            ok = False
            print(f"  VERIFY-META LECH o T={t}", flush=True)
            for k in real["cache_ints"]:
                if real["cache_ints"][k] != derived["cache_ints"].get(k):
                    print(f"    cache_ints[{k}]: that={real['cache_ints'][k]} "
                          f"suy={derived['cache_ints'].get(k)}", flush=True)
            for j, (r, d) in enumerate(zip(real["layers"], derived["layers"])):
                if r != d:
                    print(f"    layer {j}: that={r} suy={d}", flush=True)
                    break
        else:
            print(f"  verify-meta T={t}: KHOP", flush=True)
        del past, ids
        gc.collect()
        torch.cuda.empty_cache()
    return ok


# ------------------------------------------------------------------- data ---

def load_data(args, tok_s):
    e6 = _load("e6v3_ce")
    data = e6.build_data(tok_s, max_ctx=min(args.max_ctx, 2000))
    if args.data_file:
        extra = json.loads(Path(args.data_file).read_text())
        data["train"] += extra["train"]
        data["val"] += extra["val"]
    random.Random(SEED).shuffle(data["train"])
    random.Random(SEED).shuffle(data["val"])
    from collections import Counter
    for k in ("train", "val", "test"):
        print(f"{k}: {len(data[k])} — "
              f"{dict(Counter(x['kind'] for x in data[k]))}", flush=True)
    return data


# ------------------------------------------------------------------ train ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--max-ctx", type=int, default=2048,
                    help="user 2026-08-28 'tang tran 1024-2048'; da do: 2048 "
                         "chay tron voi tbptt=64 (gold tu dong ha ve 16 o do)")
    ap.add_argument("--tbptt", type=int, default=64,
                    help="do duoc: w=128 OOM o ctx>=1536; w=64 chay tron toi "
                         "2048 (probe_joint_lora luot 6)")
    ap.add_argument("--gold-cap", type=int, default=48,
                    help="tran token gold. gold = SO VI TRI feed vao 27B nen "
                         "an bo nho that (do trong probe: gold 256 khac han "
                         "gold 16). Cat theo cau hinh da do duoc.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="lr cua mapper")
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-n", type=int, default=40)
    ap.add_argument("--ce-floor", type=float, default=0.2)
    ap.add_argument("--init-mapper", default="",
                    help="warm-start tu mapper da co (v427_4k)")
    ap.add_argument("--out", default="/content/joint_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="joint_v1")
    ap.add_argument("--verify-meta", default="256,512,1024")
    ap.add_argument("--sanity", type=int, default=0,
                    help="chay N buoc roi dung + in VRAM/toc do")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    _api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    def hf_up(local, dest):
        """quy tac 11b: khong phong train dai khi duong upload chua song."""
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        try:
            _api.upload_file(path_or_fileobj=str(local),
                             path_in_repo=f"{args.hf_prefix}/{dest}",
                             repo_id=args.hf_repo)
            print(f"HF-UP {dest}", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL {dest}: {type(ex).__name__}: {ex}", flush=True)

    # ---- nap 27B (CPU-offload embed/lm_head: tiet kiem 4,85GiB da do) ----
    t0 = time.time()
    tok_t, model_t = e5.load_4bit_cpu_offload_io(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    print(f"27B nap xong {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]

    # meta goc + TU KIEM (bay 1)
    T_BASE = 512
    ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
    with torch.no_grad():
        p0 = e5.prefill_chunked(model_t, ids)
    base_meta = e5.cache_meta(p0)
    del p0, ids
    gc.collect()
    torch.cuda.empty_cache()
    lens = [int(x) for x in args.verify_meta.split(",") if x]
    if not verify_meta(model_t, base_meta, T_BASE, lens):
        print("META SUY RA KHONG KHOP META THAT -> DUNG (khong doan mo)",
              flush=True)
        return
    print("verify-meta: TAT CA KHOP", flush=True)

    # ---- nap 4B + LoRA ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    for p in model_s.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    model_s = get_peft_model(model_s, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    print(f"LoRA r={args.lora_r}: "
          f"{sum(p.numel() for p in lora_params)/1e6:.1f}M param", flush=True)

    # BAY 2: cat TRAI — moi prompt bo nay dat cau hoi O CUOI
    tok_s.truncation_side = "left"
    tok_t.truncation_side = "left"

    from transformers import AutoConfig
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.ckpt = True
    if args.init_mapper and Path(args.init_mapper).exists():
        mapper.load(args.init_mapper)
        print(f"warm-start mapper: {args.init_mapper}", flush=True)

    data = load_data(args, tok_s)

    def enc(it):
        """(cut, warm, gold_ids, feed). Cat TRAI (bay 2)."""
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.max_ctx)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        # tran gold theo BAO DO THAT, phu thuoc do dai prompt (xem
        # GOLD_ENVELOPE): gold dai an bo nho hon ca ctx dai
        gm = min(GMAX.get(it["kind"], GOLD_MAX),
                 gold_cap_for(cut.shape[1], args.gold_cap))
        gold_ids = tok_t(it["gold"], add_special_tokens=False,
                         return_tensors="pt")["input_ids"][:, :gm].to("cuda")
        feed = torch.cat([warm, gold_ids[:, :-1]], 1)
        return cut, warm, gold_ids, feed

    def prefill_tbptt(ids, w):
        """T-w token dau no_grad, w token cuoi co grad (xem docstring dau
        file: GC bi loai ve nguyen tac nen day la duong duy nhat chan duoc
        activation cua 4B)."""
        past, cutp = None, max(0, ids.shape[1] - w)
        if cutp:
            with torch.no_grad():
                for s in range(0, cutp, 1024):
                    o = model_s(input_ids=ids[:, s:min(s + 1024, cutp)],
                                past_key_values=past, use_cache=True,
                                logits_to_keep=1)
                    past = o.past_key_values
        for s in range(cutp, ids.shape[1], 1024):
            o = model_s(input_ids=ids[:, s:s + 1024], past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = o.past_key_values
        return past

    def student_past(cut):
        src = prefill_tbptt(cut, args.tbptt)
        tpl = e5.build_template_from_meta(
            probe_t, meta_for_len(base_meta, T_BASE, cut.shape[1]))
        st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit(
        [{"params": mapper.params, "lr": args.lr},
         {"params": lora_params, "lr": args.lora_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    results = {"args": vars(args), "val": [], "train_loss": []}

    def save_results():
        (out / "results.json").write_text(json.dumps(results, indent=1))

    def save_ckpt(tag):
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        hf_up(out / f"mapper_{tag}.pt", f"mapper_{tag}.pt")
        for f in sorted((out / f"lora_{tag}").glob("*")):
            if f.is_file():
                hf_up(f, f"lora_{tag}/{f.name}")

    @torch.no_grad()
    def run_val(limit):
        from collections import defaultdict
        sc = defaultdict(list)
        for it in data["val"][:limit]:
            cut, warm, gold_ids, _ = enc(it)
            if gold_ids.shape[1] < 1:
                continue
            st = student_past(cut)
            o = model_t(input_ids=warm, past_key_values=st, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen = [int(inp)]
            for _ in range(GEN_LEN.get(it["kind"], 24) - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen.append(int(inp))
                if inp.item() == tok_t.eos_token_id:
                    break
            txt = tok_t.decode(gen)
            if it["kind"] in NEW_KINDS:
                hit = gd.score_item(it, txt)
            elif it["kind"] == "bfcl":
                hit = int(it["fn"] in txt)
            elif it["kind"] == "needle":
                hit = int(it["code"] in re.sub(r"\D", "", txt))
            else:
                hit = int(re.sub(r"\s+", " ", it["gold"])[:30]
                          in re.sub(r"\s+", " ", txt))
            sc[it["kind"]].append(hit)
            del st, cur, o
            torch.cuda.empty_cache()
        return {k: f"{sum(v)}/{len(v)}" for k, v in sc.items()}, \
               sum(sum(v) for v in sc.values())

    best = -1
    t_start = time.time()
    for step in range(1, args.steps + 1):
        it = data["train"][(step - 1) % len(data["train"])]
        try:
            cut, warm, gold_ids, feed = enc(it)
            if gold_ids.shape[1] < 2:
                continue
            st = student_past(cut)
            o = model_t(input_ids=feed, past_key_values=st, use_cache=True)
            logp = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
            nll = -logp.gather(2, gold_ids.unsqueeze(-1)).squeeze(-1)
            wts = torch.ones_like(nll)
            wts[:, 0] = FIRST_W          # token quyet dinh
            ce = (nll * wts).sum() / wts.sum()
            opt.zero_grad(set_to_none=True)
            ce.backward()
            opt.step()
            sched.step()
            cev = float(ce)
            del st, o, logp, nll, ce
        except torch.cuda.OutOfMemoryError:
            print(f"  buoc {step} OOM (kind={it['kind']}, "
                  f"len={len(it['prompt'])}) -> bo qua", flush=True)
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        gc.collect()
        torch.cuda.empty_cache()

        if step % 20 == 0:
            results["train_loss"].append([step, round(cev, 4)])
            print(f"buoc {step}/{args.steps} ce={cev:.4f} "
                  f"{(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={gib():.2f}GiB", flush=True)
        if args.sanity and step >= args.sanity:
            print(f"SANITY xong {step} buoc, peak={gib():.2f}GiB, "
                  f"{(time.time()-t_start)/step:.2f}s/buoc", flush=True)
            print("E9_SANITY_EXIT", flush=True)
            return
        if step % args.val_every == 0:
            vs, score = run_val(args.val_n)
            results["val"].append([step, score, vs])
            print(f"=== VAL buoc {step}: score={score} {vs}", flush=True)
            save_results()
            hf_up(out / "results.json", "results.json")
            save_ckpt("last")
            if score > best:
                best = score
                save_ckpt("best")
                print(f"    ky luc moi: {score}", flush=True)
            if cev < args.ce_floor:
                print(f"CE_FLOOR ({cev:.4f} < {args.ce_floor}) -> dung som",
                      flush=True)
                break

    vs, score = run_val(args.val_n)
    results["val"].append([args.steps, score, vs])
    print(f"=== VAL cuoi: score={score} {vs}", flush=True)
    save_results()
    hf_up(out / "results.json", "results.json")
    save_ckpt("last")
    if score > best:
        save_ckpt("best")
    print("E9_JOINT_EXIT", flush=True)


if __name__ == "__main__":
    main()
