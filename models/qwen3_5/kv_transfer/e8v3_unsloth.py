"""E8 v3 — dung DUNG bai Unsloth cho Qwen3.5 (khao sat docs 2026-08-15).

Khao sat unsloth.ai/docs/models/qwen3.5/fine-tune chot:
  - Unsloth ho tro Qwen3.5 chinh thuc, co KERNEL TRITON MAMBA/GDN rieng
    (chinh la "fast path" ma transformers bao thieu → v2 thuong 13s/buoc).
  - Unsloth KHUYEN CAO KHONG QLoRA 4-bit tren Qwen3.5 (lech luong tu) —
    trung bai hoc bnb-noise E6c cua ta → student 2B chay BF16 (5GB).
  - Cong thuc: FastLanguageModel + get_peft_model(target_modules=attn+MLP),
    ta THEM cac linear GDN (in_proj_qkv/in_proj_z/in_proj_ba/out_proj).
  - use_gradient_checkpointing=False: GC tat use_cache ma loss cua ta
    SONG bang cache states co grad; VRAM khong phai nut that (2B bf16 ~4GB
    + 9B teacher bnb ~7GB < 22GB).

Loss/eval giu nguyen v2 (state-alignment nMSE, gate needle tile).
Teacher 9B van bnb-4bit no-grad (bf16 9B ~18GB khong vua khi dong tru).

Run: python e8v3_unsloth.py --steps 600
"""

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path

from unsloth import FastLanguageModel   # PHAI import truoc transformers

spec = importlib.util.spec_from_file_location(
    "e8v2_qlora", Path(__file__).parent / "e8v2_qlora.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
e8, e5 = v2.e8, v2.e5

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_ba", "out_proj"]


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
    ap.add_argument("--out", default="/content/lora_e8v3")
    ap.add_argument("--results", default="/content/logs/e8v3_results.json")
    args = ap.parse_args()

    import torch

    results = {"evals": [], "config": vars(args)}

    def save_results():
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w") as fh:
            json.dump(results, fh, indent=1)

    model_s, tok = FastLanguageModel.from_pretrained(
        model_name=args.src_model, max_seq_length=2048,
        load_in_4bit=False, load_in_16bit=True)
    model_s = FastLanguageModel.get_peft_model(
        model_s, r=args.lora_r, lora_alpha=args.lora_r * 2,
        target_modules=TARGETS, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing=False, random_state=3407)
    FastLanguageModel.for_training(model_s)
    _, model_t = e5.load_4bit(args.tgt_model)
    print(f"VRAM: {torch.cuda.memory_allocated()/2**30:.1f}GiB")

    trials = e8.needle_trials(tok, v2.N_EVAL, 800, 8200)
    base = e8.tile_eval(model_s, model_t, tok, trials)
    results["baseline_tile"] = f"{base}/{v2.N_EVAL}"
    print(f"BASELINE tile (student bf16, LoRA identity): {base}/{v2.N_EVAL}")
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
        ids = v2.batch_ids(tok, step, args.batch, args.l_ctx)
        with torch.no_grad():
            tch_past = model_t(input_ids=ids, use_cache=True,
                               logits_to_keep=1).past_key_values
        src_past = model_s(input_ids=ids, use_cache=True,
                           logits_to_keep=1).past_key_values
        loss = v2.state_loss(src_past, tch_past)
        opt.zero_grad()
        loss.backward()
        if step == 0:
            gn = sum(float(p.grad.abs().sum()) for p in lora_params
                     if p.grad is not None)
            assert gn > 0, ("grad = 0 — kernel unsloth detach cache states; "
                            "fallback: bo unsloth, dung bf16 thuong")
            print(f"grad-flow OK ({gn:.2e})")
        opt.step()
        sched.step()
        lv = float(loss.detach())
        del src_past, tch_past, loss
        if step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"step {step}/{args.steps} nMSE {lv:.4f} "
                  f"lr {sched.get_last_lr()[0]:.1e} ({time.time()-t0:.0f}s)")
        if step % v2.EVAL_EVERY == v2.EVAL_EVERY - 1 or step == args.steps - 1:
            hit = e8.tile_eval(model_s, model_t, tok, trials)
            results["evals"].append({"step": step, "nmse": round(lv, 4),
                                     "needle": f"{hit}/{v2.N_EVAL}"})
            print(f"EVAL step {step}: needle {hit}/{v2.N_EVAL} (best {best})")
            save_results()
            if hit >= best:
                best = hit
                model_s.save_pretrained(args.out)

    results["best"] = f"{best}/{v2.N_EVAL}"
    save_results()
    print("===== E8V3 KET QUA =====")
    print(f"baseline {results['baseline_tile']} | best {best}/{v2.N_EVAL}")
    print("E8V3_DONE")


if __name__ == "__main__":
    main()
