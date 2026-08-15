"""E6b — root-cause & fix for the greedy-hit crack (user order: fix tận gốc).

E6 found: 9B reading a raw-copied 4B cache holds NLL parity on BFCL (2.497 vs
2.461) but greedy function-call hit drops 19/20 -> 6/20. Three suspects, each
isolated here on the same 20 BFCL items:

  S1 fp16 spill roundtrip  -> compare fp16-spilled vs bf16-spilled copies
  S2 bnb-4bit model noise  -> (noted; this run keeps bnb for both — if S1+S3
                              fixes recover the hits, S2 is exonerated)
  S3 error near the generation point -> SUFFIX RE-PREFILL: transplant the
     cache only up to T-W, let 9B natively read the last W tokens (heals
     conv states exactly, last-W KV exactly, and updates GDN state natively
     from the copied state at T-W). W in {0, 32, 128}; cost ~W/T of prefill.

Phase A: 4B spills, per item, caches cut at T-W for each W (GDN state is
recurrent — the state at T-W must be captured during prefill, it cannot be
rewound from the final state). Phase B: 9B evaluates:
  self | copy_fp16_W0 (E6 repro) | copy_bf16_W0 | copy_bf16_W32 | copy_bf16_W128
"""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

spec6 = importlib.util.spec_from_file_location(
    "e6_suite", Path(__file__).parent / "e6_suite.py")
e6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(e6)
e5 = e6.e5

WINDOWS = (0, 32, 128)


def spill_cache_dtype(past, path, dtype):
    import torch
    d = []
    for l in past.layers:
        if "LinearAttention" in type(l).__name__:
            d.append(("g", l.recurrent_states.to(dtype).cpu(),
                      l.conv_states.to(dtype).cpu()))
        else:
            d.append(("a", l.keys.to(dtype).cpu(), l.values.to(dtype).cpu()))
    torch.save(d, path)


def copy_into(tpl, src):
    """Overwrite template layers (truncated to src length) with src tensors."""
    for lt, ls in zip(tpl.layers, src.layers):
        if hasattr(ls, "keys"):
            lt.keys = ls.keys.clone()
            lt.values = ls.values.clone()
        else:
            lt.recurrent_states = ls.recurrent_states.clone()
            lt.conv_states = ls.conv_states.clone()
    return tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--work", default="/content/e6b")
    ap.add_argument("--results", default="/content/logs/e6b_results.json")
    args = ap.parse_args()

    import re
    import statistics
    import torch

    wd = Path(args.work)
    wd.mkdir(parents=True, exist_ok=True)
    benches = e6.load_benches()
    items = benches.get("bfcl", [])
    assert items, "BFCL khong tai duoc"
    print(f"{len(items)} BFCL items")

    # ---- phase A: 4B — spill cut-point caches (bf16 + fp16 at W=0) ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    with torch.no_grad():
        for bi, (prompt, _, _) in enumerate(items):
            enc = tok_s(prompt, return_tensors="pt", truncation=True,
                        max_length=2048).to("cuda")
            pre = enc["input_ids"][:, :-1]
            T = pre.shape[1]
            for W in WINDOWS:
                cut = max(T - W, 8)
                past = model_s(input_ids=pre[:, :cut], use_cache=True,
                               logits_to_keep=1).past_key_values
                spill_cache_dtype(past, wd / f"b{bi}_W{W}_bf16.pt",
                                  torch.bfloat16)
                if W == 0:
                    spill_cache_dtype(past, wd / f"b{bi}_W0_fp16.pt",
                                      torch.float16)
                del past
                torch.cuda.empty_cache()
            if (bi + 1) % 5 == 0:
                print(f"A {bi+1}/{len(items)}")
    del model_s
    gc.collect(); torch.cuda.empty_cache()
    print("PHASE_A_DONE")

    # ---- phase B: 9B — 5 conditions ----
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    CONDS = ["self", "copy_fp16_W0", "copy_bf16_W0", "copy_bf16_W32",
             "copy_bf16_W128"]
    res = {c: {"hit": 0, "nll": []} for c in CONDS}
    with torch.no_grad():
        for bi, (prompt, gold, fn_name) in enumerate(items):
            enc = tok_t(prompt, return_tensors="pt", truncation=True,
                        max_length=2048).to("cuda")
            pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
            T = pre.shape[1]
            for cond in CONDS:
                if cond == "self":
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                else:
                    _, dt, wtag = cond.split("_")
                    W = int(wtag[1:])
                    cut = max(T - W, 8)
                    src = e5.load_cache(wd / f"b{bi}_W{W}_{dt}.pt")
                    tpl = model_t(input_ids=pre[:, :cut], use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = copy_into(tpl, src)
                    del src
                    if W > 0:   # 9B natively re-reads the last W tokens
                        past = model_t(input_ids=pre[:, cut:],
                                       past_key_values=past, use_cache=True,
                                       logits_to_keep=1).past_key_values
                nll, hit = e6.bench_metrics(model_t, tok_t, prompt, gold,
                                            fn_name, past, last)
                res[cond]["hit"] += hit or 0
                res[cond]["nll"].append(nll)
                del past
                torch.cuda.empty_cache()
            print(f"B {bi+1}/{len(items)}: " +
                  " ".join(f"{c}:{res[c]['hit']}" for c in CONDS))

    print("\n===== E6B KET QUA =====")
    for c in CONDS:
        print(f"{c:15s} hit {res[c]['hit']}/{len(items)}  "
              f"NLL {statistics.mean(res[c]['nll']):.3f}")
    with open(args.results, "w") as fh:
        json.dump(res, fh, indent=1)
    print("E6B_DONE")


if __name__ == "__main__":
    main()
