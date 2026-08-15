"""E8 — compatibility LoRA: day 2B VIET cache theo he 9B (user-approved 2026-08-15).

E7 verdict: {0.8B,2B} attention thang hang (CCA ~0.96) nhung GDN lac he
(CCA ~0.23) — tile 0/5 moi cap. Loi 1 (deep-innovation #5): thay vi dich cache
sau khi ghi, sua NGUOI VIET — gan LoRA vao cac linear trong khoi GDN cua 2B,
train de tile(cache_2B) hoat dong trong 9B.

Khac E5 (mapper ngoai, nguon spill disk, khong grad qua nguon): o day gradient
phai chay XUYEN QUA forward cua 2B (LoRA o trong) → hai model DONG TRU
(2B bnb ~1.8GB + 9B bnb ~6.5GB, vua L4), KHONG gradient-checkpointing
(use_cache=True can giu grad tren cache — checkpointing tat cache).

Loss = functional KL (logits 9B doc cache-tile vs 9B self) + aux state-MSE
warm-start decay (luat error-placement: MSE khong du, KL la chinh).
CONV_WARM token dau suffix bo khoi loss (conv_states zero-fill nhu E5/E7).

Gate E8a chay truoc trong cung script: 2B self needle @800/@2000 — tran
thong tin cua cache 2B; baseline tile 0-train do lai truoc step 0.

Run:  python e8_compat.py --steps 300 --out /content/lora_e8
OOM?  ha --l-ctx 160 (mac dinh 256).
"""

import argparse
import gc
import importlib.util
import json
import random
import time
from pathlib import Path

spec7 = importlib.util.spec_from_file_location(
    "e7_matrix", Path(__file__).parent / "e7_matrix.py")
e7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(e7)
e5 = e7.e5
e2 = e7.e2

CONV_WARM = 4
T2 = 32
EVAL_EVERY = 50
N_EVAL = 5


def make_train_ids(tok, idx, l_ctx):
    """Train sample: 40% co needle+question (data mix E6), con lai van ban tho."""
    rng = random.Random(4000 + idx)
    ids = e2.token_stream(tok, l_ctx + 64, seed=4000 + idx)
    if rng.random() < 0.4:
        name = rng.choice(e2.NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        half = len(ids) // 2
        text = (tok.decode(ids[:half])
                + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                + tok.decode(ids[half:half + l_ctx // 4])
                + "\n" + e2.build_q(name) + " " + code)
        ids = tok(text)["input_ids"]
    return ids[:l_ctx]


def needle_trials(tok, n, ctx_tok, seed0):
    rng = random.Random(seed0)
    out = []
    for ti in range(n):
        name = rng.choice(e2.NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = e2.token_stream(tok, ctx_tok, seed=seed0 + ti)
        half = ctx_tok // 2
        ctx = (tok.decode(ids[:half])
               + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
               + tok.decode(ids[half:]) + "\n" + e2.build_q(name))
        out.append((name, code, ctx))
    return out


def greedy_hit(model, tok, past, inp, code):
    import re
    import torch
    cur, gen = past, []
    with torch.no_grad():
        for _ in range(10):
            o = model(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
    return int(code in re.sub(r"\D", "", tok.decode(gen)))


def gate_2b_self(model_s, tok, results):
    """E8a: tran thong tin — 2B tu doc tu tra loi."""
    import torch
    for ctx_tok, tag in ((800, "self_800"), (2000, "self_2000")):
        hit = 0
        for name, code, ctx in needle_trials(tok, N_EVAL, ctx_tok, 8100):
            enc = tok(ctx, return_tensors="pt").to("cuda")
            with torch.no_grad():
                past = model_s(input_ids=enc["input_ids"][:, :-1],
                               use_cache=True, logits_to_keep=1).past_key_values
            hit += greedy_hit(model_s, tok, past, enc["input_ids"][:, -1:], code)
            del past
            torch.cuda.empty_cache()
        results["gate"][tag] = f"{hit}/{N_EVAL}"
        print(f"GATE 2B {tag}: {hit}/{N_EVAL}")


def tile_eval(model_s, model_t, tok, trials):
    """2B (LoRA hien tai) prefill → tile → 9B tra loi. Ca hai dong tru."""
    import torch
    hit = 0
    for name, code, ctx in trials:
        enc = tok(ctx, return_tensors="pt").to("cuda")
        pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
        with torch.no_grad():
            src = model_s(input_ids=pre, use_cache=True,
                          logits_to_keep=1).past_key_values
            tpl = model_t(input_ids=pre, use_cache=True,
                          logits_to_keep=1).past_key_values
            tiled = e7.tile_into(tpl, src)
        assert tiled is not None, "ty le head khong nguyen?!"
        hit += greedy_hit(model_t, tok, tiled, last, code)
        del src, tpl, tiled
        torch.cuda.empty_cache()
    return hit


def attach_lora(model_s, r, alpha):
    """LoRA vao moi linear trong khoi GDN (linear_attn) cua 2B."""
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    targets = []
    for name, mod in model_s.named_modules():
        cls = type(mod).__name__
        if isinstance(mod, nn.Linear) or "Linear4bit" in cls:
            if "linear_attn" in name or "LinearAttention" in name:
                targets.append(name)
    assert targets, "khong tim thay linear nao trong khoi GDN — kiem tra ten module"
    print(f"LoRA targets: {len(targets)} linear (vd {targets[:3]})")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules=targets, bias="none")
    return get_peft_model(model_s, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l-ctx", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--out", default="/content/lora_e8")
    ap.add_argument("--results", default="/content/logs/e8_results.json")
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()

    import copy
    import torch
    import torch.nn.functional as F

    results = {"gate": {}, "baseline_tile": None, "evals": [], "config": vars(args)}

    def save_results():
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w") as fh:
            json.dump(results, fh, indent=1)

    # ---- E8a: gate tran thong tin (2B mot minh) ----
    tok, model_s = e5.load_4bit(args.src_model)
    gate_2b_self(model_s, tok, results)
    save_results()
    if args.gate_only:
        print("E8_GATE_DONE")
        return

    # ---- E8b: dong tru 2B(+LoRA) + 9B, train compatibility ----
    model_s = attach_lora(model_s, args.lora_r, args.lora_alpha)
    model_s.eval()   # dropout off; LoRA van requires_grad
    _, model_t = e5.load_4bit(args.tgt_model)
    print(f"VRAM sau khi nap ca hai: "
          f"{torch.cuda.memory_allocated()/2**30:.1f}GiB alloc")

    trials = needle_trials(tok, N_EVAL, 800, 8200)
    base = tile_eval(model_s, model_t, tok, trials)
    results["baseline_tile"] = f"{base}/{N_EVAL}"
    print(f"BASELINE tile 2B->9B (truoc train): {base}/{N_EVAL}")
    save_results()

    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    n_par = sum(p.numel() for p in lora_params)
    print(f"LoRA params: {n_par/1e6:.1f}M")
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(lora_params, lr=args.lr)
    except Exception:
        opt = torch.optim.Adam(lora_params, lr=args.lr, foreach=False)

    best = base
    t0 = time.time()
    for step in range(args.steps):
        gc.collect()
        torch.cuda.empty_cache()
        ids = torch.tensor([make_train_ids(tok, step, args.l_ctx)],
                           device="cuda")
        ctx_ids, suffix = ids[:, :-T2], ids[:, -T2:]

        with torch.no_grad():
            tch_past = model_t(input_ids=ctx_ids, use_cache=True,
                               logits_to_keep=1).past_key_values
            tch_ext = copy.deepcopy(tch_past)
            tch_logp = torch.log_softmax(
                model_t(input_ids=suffix, past_key_values=tch_ext,
                        use_cache=True).logits[:, CONV_WARM:].float(), -1)
            del tch_ext
            torch.cuda.empty_cache()

        # 2B forward CO GRAD — cache mang grad_fn ve LoRA
        src_past = model_s(input_ids=ctx_ids, use_cache=True,
                           logits_to_keep=1).past_key_values
        if step == 0:
            g_ok = any("LinearAttention" in type(l).__name__
                       and l.recurrent_states.requires_grad
                       for l in src_past.layers)
            assert g_ok, ("cache 2B KHONG mang grad — transformers da detach "
                          "GDN state; can hook thay the")
        tpl = copy.deepcopy(tch_past)
        student_past = e7.tile_into(tpl, src_past)
        lam = max(0.0, 1.0 - step / (0.3 * args.steps))
        aux = e5.aux_mse(student_past, tch_past)
        out = model_t(input_ids=suffix, past_key_values=student_past,
                      use_cache=True)
        stu_logp = torch.log_softmax(out.logits[:, CONV_WARM:].float(), -1)
        kl = F.kl_div(stu_logp, tch_logp, log_target=True,
                      reduction="batchmean")
        loss = kl + lam * aux
        opt.zero_grad()
        loss.backward()
        if step == 0:
            gnorm = sum(float(p.grad.abs().sum()) for p in lora_params
                        if p.grad is not None)
            assert gnorm > 0, "grad LoRA = 0 — do thi dut o dau do"
            print(f"grad-flow OK (|g| sum {gnorm:.2e})")
        opt.step()
        klv = float(kl)
        del src_past, tpl, student_past, out, stu_logp, tch_logp, tch_past
        del kl, loss, aux
        torch.cuda.empty_cache()

        if step % 10 == 0:
            print(f"step {step}/{args.steps} KL {klv:.4f} lam {lam:.2f} "
                  f"({time.time()-t0:.0f}s)")
        if step % EVAL_EVERY == EVAL_EVERY - 1 or step == args.steps - 1:
            hit = tile_eval(model_s, model_t, tok, trials)
            results["evals"].append({"step": step, "kl": round(klv, 4),
                                     "needle": f"{hit}/{N_EVAL}"})
            print(f"EVAL step {step}: needle tile {hit}/{N_EVAL} (best {best})")
            save_results()
            if hit >= best:
                best = hit
                model_s.save_pretrained(args.out)

    results["best"] = f"{best}/{N_EVAL}"
    save_results()
    print("===== E8 KET QUA =====")
    print(f"gate {results['gate']} | baseline {results['baseline_tile']} | "
          f"best {best}/{N_EVAL}")
    print("E8_DONE")


if __name__ == "__main__":
    main()
