"""E8 v2 — QLoRA-style compatibility train 2B->9B (user duyet a+b, 2026-08-15).

Don Unsloth ap dung: LoRA phu MOI linear (r=64, alpha=2r, no dropout), bf16
compute, Adam8bit, warmup+cosine, va cat moi forward thua — loss chinh la
STATE-ALIGNMENT truc tiep (tile(cache_2B) khop cache_9B, normalized MSE
per-layer) nen moi buoc chi can: 1 fwd 9B no-grad (teacher) + 1 fwd/bwd 2B.
Batch 4. So voi v1 (KL qua suffix: 3 fwd 9B + 1 fwd/bwd 2B, batch 1, 7s/buoc)
ky vong nhanh hon nhieu lan tren moi don vi tin hieu.

Luat error-placement: MSE khong du tin — gate van la needle tile moi 75 buoc.
Chap nhan code ban/bug, uu tien toc do (user chot).

Run: python e8v2_qlora.py --steps 600
"""

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path

spec8 = importlib.util.spec_from_file_location(
    "e8_compat", Path(__file__).parent / "e8_compat.py")
e8 = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(e8)
e7, e5, e2 = e8.e7, e8.e5, e8.e2

EVAL_EVERY = 75
N_EVAL = 5
ATTN_W = 0.25


def attach_lora_all(model_s, r, alpha):
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    targets = []
    for name, mod in model_s.named_modules():
        if (isinstance(mod, nn.Linear) or "Linear4bit" in type(mod).__name__) \
                and "lm_head" not in name and name.startswith("model.layers"):
            targets.append(name)
    print(f"LoRA ALL-linear targets: {len(targets)}")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                     target_modules=targets, bias="none")
    return get_peft_model(model_s, cfg)


def batch_ids(tok, step, batch, l_ctx):
    import torch
    rows = [e8.make_train_ids(tok, step * batch + j, l_ctx)
            for j in range(batch)]
    n = min(len(r) for r in rows)
    return torch.tensor([r[:n] for r in rows], device="cuda")


def state_loss(src_past, tch_past):
    """tile(2B states) vs 9B states, normalized MSE per layer."""
    import torch
    attn_s, gdn_s = e5.split_layers(src_past)
    attn_t, gdn_t = e5.split_layers(tch_past)
    ks, kt = sorted(attn_s), sorted(attn_t)
    gs, gt = sorted(gdn_s), sorted(gdn_t)
    amap = e5.depth_map(len(ks), len(kt))
    gmap = e5.depth_map(len(gs), len(gt))
    ra = attn_t[kt[0]].keys.shape[1] // attn_s[ks[0]].keys.shape[1]
    rg = (gdn_t[gt[0]].recurrent_states.shape[1]
          // gdn_s[gs[0]].recurrent_states.shape[1])
    loss = 0.0
    for j, it in enumerate(kt):
        s = attn_s[ks[amap[j]]]
        t = attn_t[it]
        for xs, xt in ((s.keys, t.keys), (s.values, t.values)):
            xs = xs.float().repeat_interleave(ra, dim=1)
            xt = xt.float()
            loss = loss + ATTN_W * ((xs - xt).pow(2).mean()
                                    / (xt.pow(2).mean() + 1e-8))
    for j, it in enumerate(gt):
        xs = gdn_s[gs[gmap[j]]].recurrent_states.float().repeat_interleave(
            rg, dim=1)
        xt = gdn_t[it].recurrent_states.float()
        loss = loss + (xs - xt).pow(2).mean() / (xt.pow(2).mean() + 1e-8)
    return loss / (len(kt) * 2 * ATTN_W + len(gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--l-ctx", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--out", default="/content/lora_e8v2")
    ap.add_argument("--results", default="/content/logs/e8v2_results.json")
    args = ap.parse_args()

    import torch

    results = {"evals": [], "config": vars(args)}

    def save_results():
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w") as fh:
            json.dump(results, fh, indent=1)

    tok, model_s = e5.load_4bit(args.src_model)
    model_s = attach_lora_all(model_s, args.lora_r, args.lora_r * 2)
    model_s.eval()
    _, model_t = e5.load_4bit(args.tgt_model)
    print(f"VRAM: {torch.cuda.memory_allocated()/2**30:.1f}GiB")

    trials = e8.needle_trials(tok, N_EVAL, 800, 8200)
    base = e8.tile_eval(model_s, model_t, tok, trials)
    results["baseline_tile"] = f"{base}/{N_EVAL}"
    print(f"BASELINE tile (LoRA init=identity): {base}/{N_EVAL}")
    save_results()

    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params)/1e6:.1f}M")
    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit(lora_params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, 1.0)
        * 0.5 * (1 + math.cos(math.pi * s / args.steps)))

    best = base
    t0 = time.time()
    for step in range(args.steps):
        ids = batch_ids(tok, step, args.batch, args.l_ctx)
        with torch.no_grad():
            tch_past = model_t(input_ids=ids, use_cache=True,
                               logits_to_keep=1).past_key_values
        src_past = model_s(input_ids=ids, use_cache=True,
                           logits_to_keep=1).past_key_values
        loss = state_loss(src_past, tch_past)
        opt.zero_grad()
        loss.backward()
        if step == 0:
            gn = sum(float(p.grad.abs().sum()) for p in lora_params
                     if p.grad is not None)
            assert gn > 0, "grad = 0 — do thi dut"
            print(f"grad-flow OK ({gn:.2e})")
        opt.step()
        sched.step()
        lv = float(loss)
        del src_past, tch_past, loss
        if step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"step {step}/{args.steps} nMSE {lv:.4f} "
                  f"lr {sched.get_last_lr()[0]:.1e} ({time.time()-t0:.0f}s)")
        if step % EVAL_EVERY == EVAL_EVERY - 1 or step == args.steps - 1:
            hit = e8.tile_eval(model_s, model_t, tok, trials)
            results["evals"].append({"step": step, "nmse": round(lv, 4),
                                     "needle": f"{hit}/{N_EVAL}"})
            print(f"EVAL step {step}: needle {hit}/{N_EVAL} (best {best})")
            save_results()
            if hit >= best:
                best = hit
                model_s.save_pretrained(args.out)

    results["best"] = f"{best}/{N_EVAL}"
    save_results()
    print("===== E8V2 KET QUA =====")
    print(f"baseline {results['baseline_tile']} | best {best}/{N_EVAL}")
    print("E8V2_DONE")


if __name__ == "__main__":
    main()
