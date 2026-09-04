"""sft_gsm -- SFT day du 1 epoch tren TOAN BO pool gsm8k co pseudo-gold.

VI SAO CHAY LAI (user 2026-09-04). Phep kiem probe_tok chi train 300 buoc tren
300 mau (dung 1 phan nho cua pool) va thay: CE giam 5 lan (0,96->0,19) ma do
chinh xac chuc nang chi +1,7 diem. Nhung do la mau qua nho de ket luan -- pool
that co 2.483 mau gsm8k da co CoT do 9B tu sinh va GIAI DUNG (joint_v1/
train_items.json), gap 6,5 lan cai dang dung (384 trong train_items_gsm.json).
Bai hoc EBA trong chinh chien dich nay: pool 170 -> dao dong vo nghia; pool
2.000 -> tin hieu on dinh, McNemar p<0,0001. Nen phai thu 1 epoch DAY DU truoc
khi ket luan "SFT khong an thua".

Kien truc GIU NGUYEN ban da chung minh (tok_rank=0 -- kenh danh tinh token da
duoc do la vo ich, xem probe_tok.json).

LUU CHECKPOINT DINH KY: Colab recycle bat cu luc nao (da dinh 3 lan trong cac
phien gan day), 1 epoch ~2 gio nen bat buoc phai luu duoc dang do. Moi lan luu
gop thanh DUNG 1 COMMIT (rate-limit HF 60 commit/gio da tung dinh).

    python -u sft_gsm.py --epochs 1 --data-file /content/train_items.json
"""
import argparse
import gc
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
gd = _load("gen_data")
e5.patch_recurrent_rebind()
WARM_P = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--init-dir", default="/content/gsm_grpo_v1c")
    ap.add_argument("--data-file", default="/content/train_items.json",
                    help="pool DAY DU (joint_v1/train_items.json co 2.882 gsm8k)")
    ap.add_argument("--pseudo-gold", default="/content/pseudo_gold_gsm2.json")
    ap.add_argument("--only-pseudo", type=int, default=1,
                    help="1 = CHI dung item co CoT 9B tu sinh va giai DUNG "
                         "(khong day mapper hoc theo loi giai sai)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="0 = tinh tu epochs")
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--save-every", type=int, default=400)
    ap.add_argument("--gold-cap", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--tbptt", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/content/sft_gsm_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="sft_gsm_v1")
    ap.add_argument("--sanity", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    _api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    # ---- du lieu: pool DAY DU ----------------------------------------------
    data = json.loads(Path(args.data_file).read_text())
    items = [it for it in data["train"] if it.get("kind") == "gsm8k"]
    pg = json.loads(Path(args.pseudo_gold).read_text()) \
        if Path(args.pseudo_gold).exists() else {}
    n_rep = 0
    for it in items:
        g = pg.get(it.get("id", ""))
        if g and g.get("gold"):
            it["gold"] = g["gold"]
            it["pseudo"] = True
            n_rep += 1
    if args.only_pseudo:
        items = [it for it in items if it.get("pseudo")]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    ev_items, tr_items = items[:args.n_eval], items[args.n_eval:]
    steps = args.steps or int(args.epochs * len(tr_items))
    print(f"pool gsm8k: {len(items)} mau (pseudo-gold {n_rep}) -> "
          f"{len(tr_items)} train / {len(ev_items)} eval | {steps} buoc "
          f"({args.epochs} epoch)", flush=True)

    # ---- 9B + LoRA-9B -------------------------------------------------------
    t0 = time.time()
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    model_t = get_peft_model(model_t, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "o_proj", "in_proj_qkvz", "out_proj"],
        task_type="CAUSAL_LM"))
    model_t.train()
    p_lt = Path(args.init_dir) / "lorat_best"
    if p_lt.exists():
        set_peft_model_state_dict(model_t, load_file(str(p_lt / "adapter_model.safetensors")))
        print(f"warm-start LoRA-9B tu {p_lt}", flush=True)
    lora_t_params = [p for p in model_t.parameters() if p.requires_grad]
    tok_t.truncation_side = "left"
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    STOPS = e5.stop_ids(tok_t, model_t)

    # ---- 4B + LoRA-4B -------------------------------------------------------
    tok_s, model_s = e5.load_4bit(args.src_model)
    for p in model_s.parameters():
        p.requires_grad_(False)
    model_s = get_peft_model(model_s, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    p_l = Path(args.init_dir) / "lora_best"
    if p_l.exists():
        set_peft_model_state_dict(model_s, load_file(str(p_l / "adapter_model.safetensors")))
        print(f"warm-start LoRA-4B tu {p_l}", flush=True)
    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    tok_s.truncation_side = "left"
    theta_s = e5.e1.get_rope_theta(
        __import__("transformers").AutoConfig.from_pretrained(
            args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    # ---- mapper (tok_rank=0: kenh token da do la vo ich) --------------------
    mp = Path(args.init_dir) / "mapper_best.pt"
    _meta = torch.load(mp, map_location="cpu").get("_meta", {}) if mp.exists() else {}
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    if mp.exists():
        mapper.load(str(mp))
        print(f"warm-start mapper tu {mp} (gdn_terms={_meta.get('gdn_terms',1)})",
              flush=True)
    print(f"nap xong {time.time()-t0:.0f}s", flush=True)

    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _p = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_p)
    del _p, _ids
    torch.cuda.empty_cache()
    import copy as _copy

    def meta_for_len(t):
        m = _copy.deepcopy(base_meta)
        m["cache_ints"] = {k: (t if v == T_BASE else v)
                           for k, v in m["cache_ints"].items()}
        for lay in m["layers"]:
            lay["ints"] = {k: (t if v == T_BASE else v) for k, v in lay["ints"].items()}
            if lay["kind"] == "a":
                for key in ("k", "v"):
                    sh, dt = lay[key]
                    lay[key] = (tuple(t if d == T_BASE else d for d in sh), dt)
        return m

    def prefill_tbptt(ids, wnd):
        past, cutp = None, max(0, ids.shape[1] - wnd)
        if cutp:
            with torch.no_grad():
                for s in range(0, cutp, 1024):
                    o = model_s(input_ids=ids[:, s:min(s + 1024, cutp)],
                                past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = o.past_key_values
        for s in range(cutp, ids.shape[1], 1024):
            o = model_s(input_ids=ids[:, s:s + 1024], past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = o.past_key_values
        return past

    def enc(it):
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=2048)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        gold = tok_t(it["gold"], add_special_tokens=False,
                     return_tensors="pt")["input_ids"][:, :args.gold_cap].to("cuda")
        return cut, warm, gold

    def student_past(cut, grad=True):
        src = prefill_tbptt(cut, args.tbptt) if grad else e5.prefill_chunked(model_s, cut)
        tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
        st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit([{"params": mapper.params, "lr": args.lr},
                              {"params": lora_params, "lr": args.lora_lr},
                              {"params": lora_t_params, "lr": args.lora_lr}])

    @torch.no_grad()
    def evaluate():
        ce_sum, n_ce, hit = 0.0, 0, 0
        for it in ev_items:
            cut, warm, gold = enc(it)
            st = student_past(cut, grad=False)
            if gold.shape[1] > 1:
                feed = torch.cat([warm, gold[:, :-1]], 1)
                o = model_t(input_ids=feed, past_key_values=e5.clone_cache_struct(st),
                            use_cache=True)
                lp = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
                ce_sum += float(-lp.gather(2, gold.unsqueeze(-1)).squeeze(-1).mean())
                n_ce += 1
            o = model_t(input_ids=warm, past_key_values=st, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen = [int(inp)]
            for _ in range(args.gen_len - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen.append(int(inp))
                if int(inp) in STOPS:
                    break
            hit += int(gd.score_item(it, tok_t.decode(gen, skip_special_tokens=True)))
            del st, cur, o
            torch.cuda.empty_cache()
        return ce_sum / max(n_ce, 1), hit / max(len(ev_items), 1)

    results = {"args": vars(args), "eval": [], "train": []}

    def save_all(tag):
        """Luu + upload trong DUNG 1 COMMIT (rate-limit 60/gio)."""
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        model_t.save_pretrained(str(out / f"lorat_{tag}"))
        (out / "results.json").write_text(json.dumps(results, indent=1))
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(f"{args.hf_prefix}/mapper_{tag}.pt",
                                  str(out / f"mapper_{tag}.pt")),
               CommitOperationAdd(f"{args.hf_prefix}/results.json",
                                  str(out / "results.json"))]
        for sub in (f"lora_{tag}", f"lorat_{tag}"):
            for f in sorted((out / sub).glob("*")):
                if f.is_file():
                    ops.append(CommitOperationAdd(f"{args.hf_prefix}/{sub}/{f.name}", str(f)))
        try:
            _api.create_commit(repo_id=args.hf_repo, operations=ops,
                               commit_message=f"{args.hf_prefix} {tag}")
            print(f"HF-UP {tag} ({len(ops)} file, 1 commit)", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL {tag}: {type(ex).__name__}: {ex}", flush=True)

    ce0, acc0 = evaluate()
    print(f"TRUOC train: CE={ce0:.4f} acc={acc0*100:.1f}%", flush=True)
    results["eval"].append({"step": 0, "ce": ce0, "acc": acc0})

    best = 1e9
    t_start = time.time()
    for step in range(1, steps + 1):
        it = tr_items[(step - 1) % len(tr_items)]
        cut, warm, gold = enc(it)
        if gold.shape[1] < 2:
            continue
        st = student_past(cut, grad=True)
        feed = torch.cat([warm, gold[:, :-1]], 1)
        o = model_t(input_ids=feed, past_key_values=st, use_cache=True)
        lp = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
        loss = -lp.gather(2, gold.unsqueeze(-1)).squeeze(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        del st, o, lp, loss
        gc.collect()
        torch.cuda.empty_cache()

        if step % 50 == 0:
            results["train"].append([step, round(lv, 4)])
            print(f"buoc {step}/{steps} CE={lv:.4f} "
                  f"{(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                  f"con ~{(steps-step)*(time.time()-t_start)/step/60:.0f} phut",
                  flush=True)
        if args.sanity and step >= args.sanity:
            print("SFT_GSM_SANITY_EXIT", flush=True)
            return
        if step % args.eval_every == 0 or step == steps:
            ce, acc = evaluate()
            results["eval"].append({"step": step, "ce": ce, "acc": acc})
            print(f"=== EVAL buoc {step}: CE={ce:.4f} acc={acc*100:.1f}% "
                  f"(dau: CE={ce0:.4f} acc={acc0*100:.1f}%)", flush=True)
            save_all("last")
            if ce < best:
                best = ce
                save_all("best")
        elif step % args.save_every == 0:
            save_all("last")

    ce1, acc1 = evaluate()
    print(f"\nSAU {steps} buoc ({args.epochs} epoch): CE={ce1:.4f} acc={acc1*100:.1f}%")
    print(f"thay doi: CE {ce0:.4f} -> {ce1:.4f} ({ce1-ce0:+.4f}) | "
          f"acc {acc0*100:.1f}% -> {acc1*100:.1f}% ({(acc1-acc0)*100:+.1f} diem)",
          flush=True)
    results["eval"].append({"step": steps, "ce": ce1, "acc": acc1})
    save_all("last")
    print("SFT_GSM_EXIT", flush=True)


if __name__ == "__main__":
    main()
