"""E3 — re-measure the 9B-pure metric suite through the 4B->9B cross path.

User order 2026-08-15: "đo lại tất cả chỉ số này bằng cross 4b-9b". Coverage:
  - quality gate  -> multi-fact QA + continuation-NLL at 30K (production ctx)
  - TTFT cold 30K -> measured ledger: 4B chunked prefill + transplant + 1 step
  - decode conc1  -> 64 greedy tokens on copied cache vs self cache (parity
                     check: decode must not care where the cache came from)
Under-load numbers (KV capacity / 12 sessions / tasks-hr) need the vLLM
injection path (Phase C); the CO-RESIDENCY COST side is measured separately
by the launch chain (two vLLM servers splitting the L4).

Prefill is chunked (4096) so 30K fits transformers on L4 (logits_to_keep=1
per chunk; full-seq lm_head OOMed E2 at 8K).
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e2_suite", Path(__file__).parent / "e2_suite.py")
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)

LENGTH = 30000
N_QA = 2
N_CONT = 2
CONT_TOKENS = 200
CHUNK = 4096
DECODE_TOKENS = 64


def chunked_prefill(model, ids, past=None):
    """Prefill in CHUNK-sized steps; returns (cache, seconds)."""
    import torch
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for i in range(0, ids.shape[1], CHUNK):
            out = model(input_ids=ids[:, i:i + CHUNK], past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = out.past_key_values
    torch.cuda.synchronize()
    return past, time.time() - t0


def make_trials(tok, seed=0):
    import random
    rng = random.Random(seed)
    trials = []
    for qi in range(N_QA):
        stream = e2.token_stream(tok, LENGTH, seed=100 * len(trials) + 7)
        facts = [(rng.choice(e2.NAMES) + f"{i}", "".join(
            rng.choice("0123456789") for _ in range(6))) for i in range(3)]
        parts, prev = [], 0
        for (name, code), frac in zip(facts, e2.FACT_FRACS):
            cut = int(LENGTH * frac)
            parts.append(tok.decode(stream[prev:cut]))
            parts.append(f"\nIMPORTANT: The secret code for project {name} is {code}.\n")
            prev = cut
        parts.append(tok.decode(stream[prev:]))
        ask = facts[qi % 3]
        trials.append({"kind": "qa", "ctx": "".join(parts) + "\n" + e2.build_q(ask[0]),
                       "name": ask[0], "gold": ask[1]})
    for ci in range(N_CONT):
        stream = e2.token_stream(tok, LENGTH + CONT_TOKENS,
                                 seed=100 * len(trials) + 7)
        trials.append({"kind": "cont", "ctx": tok.decode(stream[:LENGTH]),
                       "cont_ids": stream[LENGTH:]})
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--cache-dir", default="/content/e3_caches")
    ap.add_argument("--out", default="/content/logs/e3_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import copy as _c
    import re
    import statistics
    import torch

    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- stage A: 4B chunked prefill @30K, spill ----
    tok_s, model_s = e2.load_model(args.src_model)
    trials = make_trials(tok_s, args.seed)
    for ti, tr in enumerate(trials):
        enc = tok_s(tr["ctx"], return_tensors="pt").to("cuda")
        tr["n_ctx"] = enc["input_ids"].shape[1]
        past, dt = chunked_prefill(model_s, enc["input_ids"][:, :-1])
        tr["t_4b_prefill"] = dt
        e2.spill(past, f"{args.cache_dir}/t{ti}.npz")
        del past
        torch.cuda.empty_cache()
        print(f"A {ti+1}/{len(trials)} {tr['kind']} ({tr['n_ctx']} tok): "
              f"4B prefill {dt:.2f}s")
    del model_s
    torch.cuda.empty_cache()

    # ---- stage B: 9B ----
    tok_t, model_t = e2.load_model(args.tgt_model)
    results = []
    with torch.no_grad():
        for ti, tr in enumerate(trials):
            enc = tok_t(tr["ctx"], return_tensors="pt").to("cuda")
            row = {k: tr[k] for k in ("kind", "n_ctx", "t_4b_prefill")}
            # one 9B prefill; deepcopy BEFORE any measurement mutates it
            past_self, row["t_9b_prefill"] = chunked_prefill(
                model_t, enc["input_ids"][:, :-1])
            past_copy = _c.deepcopy(past_self)
            row["t_transplant"] = e2.overwrite_from(
                past_copy, f"{args.cache_dir}/t{ti}.npz")
            last = enc["input_ids"][:, -1:]

            for cond, base in (("self", past_self), ("copy", past_copy),
                               ("no_ctx", None)):
                if cond == "no_ctx":
                    if tr["kind"] == "qa":
                        q = tok_t(e2.build_q(tr["name"]),
                                  return_tensors="pt").to("cuda")
                        o0 = model_t(input_ids=q["input_ids"][:, :-1],
                                     use_cache=True, logits_to_keep=1)
                        past, step = o0.past_key_values, q["input_ids"][:, -1:]
                    else:
                        past, step = None, None
                else:
                    past, step = base, last

                if tr["kind"] == "qa":
                    torch.cuda.synchronize(); t0 = time.time()
                    out = model_t(input_ids=step, past_key_values=past,
                                  use_cache=True)
                    torch.cuda.synchronize()
                    row[f"{cond}_t_first"] = time.time() - t0
                    gold = tok_t(" " + tr["gold"], add_special_tokens=False,
                                 return_tensors="pt")["input_ids"].to("cuda")
                    logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                    cur, nll, gen = out.past_key_values, 0.0, []
                    # greedy: gold NLL first, then DECODE_TOKENS for tok/s
                    torch.cuda.synchronize(); td = time.time()
                    for gi in range(DECODE_TOKENS):
                        if gi < gold.shape[1]:
                            nll += -float(logp[0, gold[0, gi]])
                        nxt = logp.argmax(-1, keepdim=True)
                        gen.append(int(nxt))
                        o = model_t(input_ids=nxt, past_key_values=cur,
                                    use_cache=True)
                        cur = o.past_key_values
                        logp = torch.log_softmax(o.logits[:, -1, :].float(), -1)
                    torch.cuda.synchronize()
                    row[f"{cond}_decode_tps"] = DECODE_TOKENS / (time.time() - td)
                    row[f"{cond}_hit"] = int(
                        tr["gold"] in re.sub(r"\D", "", tok_t.decode(gen[:10])))
                    row[f"{cond}_nll"] = nll / gold.shape[1]
                else:
                    cont = torch.tensor([tr["cont_ids"]], device="cuda")
                    if cond == "no_ctx":
                        out = model_t(input_ids=cont)
                        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
                        row[f"{cond}_nll"] = float(-lp.gather(
                            2, cont[:, 1:].unsqueeze(-1)).mean())
                    else:
                        full = torch.cat([step, cont], 1)
                        out = model_t(input_ids=full, past_key_values=past,
                                      use_cache=True)
                        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
                        row[f"{cond}_nll"] = float(-lp.gather(
                            2, cont.unsqueeze(-1)).mean())
                del past
                torch.cuda.empty_cache()
            del past_self, past_copy
            torch.cuda.empty_cache()
            results.append(row)
            print(f"B {ti+1}/{len(trials)} {tr['kind']}: " + " ".join(
                f"{c}:" + (f"hit={row.get(c + '_hit')}" if tr["kind"] == "qa"
                           else f"nll={row[c + '_nll']:.3f}")
                for c in ("self", "copy", "no_ctx")))
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1)

    # ---- summary ----
    qa = [r for r in results if r["kind"] == "qa"]
    co = [r for r in results if r["kind"] == "cont"]
    print("\n===== E3 KET QUA (30K) =====")
    for c in ("self", "copy", "no_ctx"):
        line = f"{c:7s} QA {sum(r[f'{c}_hit'] for r in qa)}/{len(qa)}"
        line += f"  cont-NLL {statistics.mean(r[f'{c}_nll'] for r in co):.3f}"
        if f"{c}_decode_tps" in qa[0]:
            line += (f"  decode {statistics.mean(r[f'{c}_decode_tps'] for r in qa):.1f} tok/s"
                     f"  first-step {statistics.mean(r[f'{c}_t_first'] for r in qa):.3f}s")
        print(line)
    t4b = statistics.mean(r["t_4b_prefill"] for r in results)
    t9b = statistics.mean(r["t_9b_prefill"] for r in results)
    ttr = statistics.mean(r["t_transplant"] for r in results)
    print(f"TTFT lanh 30K (transformers): 9B self {t9b:.2f}s vs "
          f"4B {t4b:.2f}s + transplant {ttr:.3f}s = {t4b + ttr:.2f}s "
          f"(x{t9b / (t4b + ttr):.2f})")
    print("E3_DONE")


if __name__ == "__main__":
    main()
