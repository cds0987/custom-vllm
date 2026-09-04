"""probe_tok_channel -- PHEP KIEM 40 PHUT: kenh danh tinh token (EAGLE) co
mang thong tin that khong?

BOI CANH (ra soat 2026-09-04). Oracle ablation da do: attention THAT + GDN
mapped = 26,7% gsm8k, trong khi 9B tu lam 86,7%. Tuc la rieng phep dich GDN
hien tai da mat 60 diem, VA tran cua moi cong viec reward/RL dang lam chi la
26,7%. Ra soat kien truc mapper thay no thieu 4 thu, trong do #1 la KHONG BIET
token nao o vi tri nao -- du ta co san thong tin do mien phi.

PHEP KIEM NAY (mot-bien, tat ca phan con lai giu nguyen):
    A) tok_rank=0  -> mapper y het hien tai      (doi chung)
    B) tok_rank=64 -> them kenh embedding token  (bien duy nhat)
Cung seed, cung du lieu, cung so buoc, cung lr. Chi so doc:
  - CE tren gold (thap hon = cache mapped tai tao duoc hanh vi 9B tot hon)
  - do chinh xac gsm8k tren tap giu rieng (thi nghiem CHUC NANG -- theo luat
    error-placement, CE mot minh KHONG du de ket luan)

Chay:
    python -u probe_tok_channel.py --tok-rank 0  --steps 300 --tag base
    python -u probe_tok_channel.py --tok-rank 64 --steps 300 --tag tok
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
    ap.add_argument("--init-dir", default="/content/gsm_grpo_v1c",
                    help="warm-start CHUNG cho ca hai nhanh (checkpoint tot "
                         "nhat hien co) -- diem xuat phat phai y het nhau")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--tok-rank", type=int, default=0,
                    help="0 = doi chung (mapper hien tai); >0 = bat kenh token")
    ap.add_argument("--gsm-data", default="/content/train_items_gsm.json")
    ap.add_argument("--pseudo-gold", default="/content/pseudo_gold_gsm2.json")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--gold-cap", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=200)
    ap.add_argument("--tbptt", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/content/logs/probe_tok.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    # ---- du lieu: dung CHINH pool gsm8k + pseudo-gold 9B tu sinh ------------
    data = json.loads(Path(args.gsm_data).read_text())
    items = [it for it in data["train"] if it.get("kind") == "gsm8k"]
    pg = json.loads(Path(args.pseudo_gold).read_text()) \
        if Path(args.pseudo_gold).exists() else {}
    n_rep = 0
    for it in items:
        g = pg.get(it.get("id", ""))
        if g and g.get("gold"):
            it["gold"] = g["gold"]
            n_rep += 1
    rng = random.Random(args.seed)
    rng.shuffle(items)
    ev_items, tr_items = items[:args.n_eval], items[args.n_eval:args.n_eval + args.n_train]
    print(f"du lieu: {len(tr_items)} train / {len(ev_items)} eval "
          f"(pseudo-gold {n_rep})", flush=True)

    # ---- nap 9B + LoRA-9B ---------------------------------------------------
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
    # BANG EMBEDDING cua 9B -- nguon "danh tinh token" chinh xac, tra bang,
    # KHONG ton attention/GDN (khac han bridge-token phai prefill that).
    emb_layer = model_t.get_input_embeddings()
    d_model = emb_layer.weight.shape[1]
    print(f"d_model 9B = {d_model}", flush=True)

    # ---- nap 4B + LoRA-4B ---------------------------------------------------
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

    # ---- mapper: BIEN DUY NHAT la tok_rank ---------------------------------
    mp = Path(args.init_dir) / "mapper_best.pt"
    _meta = torch.load(mp, map_location="cpu").get("_meta", {}) if mp.exists() else {}
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1),
                       tok_rank=args.tok_rank, d_model=d_model)
    if mp.exists():
        mapper.load(str(mp))   # checkpoint cu khong co kenh -> kenh giu ZERO
    print(f"mapper: tok_rank={args.tok_rank} (0 = doi chung) | "
          f"nap {time.time()-t0:.0f}s", flush=True)

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
        """emb = embedding 9B cua CHINH cac token trong `cut` -- kenh danh tinh
        token. tok_rank=0 thi mapper bo qua emb (hanh vi y het ban cu)."""
        src = prefill_tbptt(cut, args.tbptt) if grad else e5.prefill_chunked(model_s, cut)
        tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
        emb = emb_layer(cut) if args.tok_rank else None
        st = e5.build_student_past(tpl, src, mapper, emb)
        del tpl
        return st

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit([{"params": mapper.params, "lr": args.lr},
                              {"params": lora_params, "lr": 5e-5},
                              {"params": lora_t_params, "lr": 5e-5}])

    @torch.no_grad()
    def evaluate():
        """Hai chi so: CE tren gold + do chinh xac gsm8k THAT (chuc nang)."""
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

    res = {"args": vars(args), "eval": []}
    ce0, acc0 = evaluate()
    print(f"[{args.tag}] TRUOC train: CE={ce0:.4f} acc={acc0*100:.1f}%", flush=True)
    res["eval"].append({"step": 0, "ce": ce0, "acc": acc0})

    t_start = time.time()
    for step in range(1, args.steps + 1):
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
        del st, o, lp
        gc.collect()
        torch.cuda.empty_cache()
        if step % 25 == 0:
            print(f"[{args.tag}] buoc {step}/{args.steps} CE={float(loss):.4f} "
                  f"{(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

    ce1, acc1 = evaluate()
    print(f"\n[{args.tag}] SAU {args.steps} buoc: CE={ce1:.4f} acc={acc1*100:.1f}%")
    print(f"[{args.tag}] thay doi: CE {ce0:.4f} -> {ce1:.4f} ({ce1-ce0:+.4f}) | "
          f"acc {acc0*100:.1f}% -> {acc1*100:.1f}% ({(acc1-acc0)*100:+.1f})",
          flush=True)
    res["eval"].append({"step": args.steps, "ce": ce1, "acc": acc1})

    out = Path(args.out)
    all_res = json.loads(out.read_text()) if out.exists() else {}
    all_res[args.tag] = res
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_res, indent=1))
    print(f"da ghi {out}", flush=True)
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(out), path_in_repo=f"evalbig/{out.name}",
                repo_id="gunnybd01/qwen35-kv-mapper-4b-27b")
            print("HF-UP", out.name, flush=True)
        except Exception as ex:
            print("HF-UP FAIL", type(ex).__name__, str(ex)[:80], flush=True)
    print("PROBE_TOK_EXIT", flush=True)


if __name__ == "__main__":
    main()
