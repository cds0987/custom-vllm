"""E2 — harder evals for raw-copy cache transfer 4B->9B + cascade TTFT ledger.

E1 showed raw copy retains needle 12/12 @1.5K ctx (ridge mapper: 0/12 — dead).
Before trusting the cascade (4B prefills, 9B decodes), E2 asks the harder
questions:

  1) UNDERSTANDING, not just retrieval: continuation NLL over 200 real
     FineWeb-Edu tokens under the copied cache (self vs copy vs no_ctx floor).
  2) Serving-relevant lengths: 2K / 8K / 16K, multi-fact QA (3 facts at
     20/50/80% of context, rotating question).
  3) The real end-to-end TTFT ledger per length: 4B prefill + transplant vs
     9B self prefill (component-timed; models can't co-reside in bf16 on L4,
     so the cascade number is a sum of measured parts, stated as such).

Single GPU pass, models sequential:
  stage A: 4B bf16 prefills every trial context, caches spilled to disk
           (fp16 npz, /content/e2_caches/) — Colab RAM can't hold them all.
  stage B: 9B bf16 per trial: self_prefill / copy(4B) / no_ctx.

Run (background, PID file, per repo rule 10):
  python e2_suite.py --out /content/logs/e2_results.json
"""

import argparse
import json
import os
import time

LENGTHS = (2000, 8000, 16000)
N_QA = 3          # multi-fact QA trials per length
N_CONT = 2        # continuation-NLL trials per length
CONT_TOKENS = 200
FACT_FRACS = (0.20, 0.50, 0.80)

NAMES = ["aurora", "falcon", "meridian", "obsidian", "harbor",
         "juniper", "cascade", "vertex", "quartz", "ember"]


def token_stream(tok, n_tokens, seed):
    """Deterministic stream of real FineWeb-Edu tokens (concatenated docs)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ids, skip = [], seed
    for ex in ds:
        if skip > 0:            # deterministic per-trial offset into the corpus
            skip -= 1
            continue
        ids += tok(ex["text"] + "\n\n", add_special_tokens=False)["input_ids"]
        if len(ids) >= n_tokens:
            return ids[:n_tokens]
    raise RuntimeError("corpus exhausted")


def build_q(name):
    return (f"Question: What is the secret code for project {name}?\n"
            f"Answer: The secret code for project {name} is")


def make_trials(tok, seed=0):
    """Trial list shared verbatim by both models (same tokenizer, asserted)."""
    import random
    rng = random.Random(seed)
    trials = []
    for L in LENGTHS:
        for qi in range(N_QA):
            stream = token_stream(tok, L, seed=100 * len(trials) + 7)
            facts = [(rng.choice(NAMES) + f"{i}", "".join(
                rng.choice("0123456789") for _ in range(6))) for i in range(3)]
            parts, prev = [], 0
            for (name, code), frac in zip(facts, FACT_FRACS):
                cut = int(L * frac)
                parts.append(tok.decode(stream[prev:cut]))
                parts.append(f"\nIMPORTANT: The secret code for project {name} is {code}.\n")
                prev = cut
            parts.append(tok.decode(stream[prev:]))
            ask = facts[qi % 3]
            ctx = "".join(parts) + "\n" + build_q(ask[0])
            trials.append({"kind": "qa", "L": L, "ctx": ctx,
                           "name": ask[0], "gold": ask[1]})
        for ci in range(N_CONT):
            stream = token_stream(tok, L + CONT_TOKENS,
                                  seed=100 * len(trials) + 7)
            trials.append({"kind": "cont", "L": L,
                           "ctx": tok.decode(stream[:L]),
                           "cont_ids": stream[L:]})
    return trials


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    return tok, model


def cache_layers(past):
    attn, gdn = {}, {}
    for i, layer in enumerate(past.layers):
        (gdn if "LinearAttention" in type(layer).__name__ else attn)[i] = layer
    return attn, gdn


def spill(past, path):
    """Cache -> disk npz fp16 (attn K/V + GDN recurrent/conv)."""
    import numpy as np
    import torch
    attn, gdn = cache_layers(past)
    d = {}
    for i, l in attn.items():
        d[f"K_{i}"] = l.keys[0].to(torch.float16).cpu().numpy()
        d[f"V_{i}"] = l.values[0].to(torch.float16).cpu().numpy()
    for i, l in gdn.items():
        d[f"R_{i}"] = l.recurrent_states[0].to(torch.float16).cpu().numpy()
        d[f"C_{i}"] = l.conv_states[0].to(torch.float16).cpu().numpy()
    np.savez(path, **d)


def overwrite_from(past, path):
    """Overwrite a template cache's tensors with the spilled 4B cache.
    Returns overwrite-only seconds (the honest transplant cost — disk load is
    an artifact of models-not-co-resident, excluded and reported separately)."""
    import numpy as np
    import torch
    z = np.load(path)
    attn, gdn = cache_layers(past)
    loaded = {k: torch.from_numpy(z[k]).cuda() for k in z.files}
    torch.cuda.synchronize(); t0 = time.time()
    hit = 0
    for i, l in attn.items():
        l.keys[0] = loaded[f"K_{i}"].to(l.keys.dtype); hit += 1
        l.values[0] = loaded[f"V_{i}"].to(l.values.dtype); hit += 1
    for i, l in gdn.items():
        l.recurrent_states[0] = loaded[f"R_{i}"].to(l.recurrent_states.dtype); hit += 1
        l.conv_states[0] = loaded[f"C_{i}"].to(l.conv_states.dtype); hit += 1
    torch.cuda.synchronize()
    if hit == 0:
        raise RuntimeError("overwrite_from hit NOTHING — layout mismatch")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--cache-dir", default="/content/e2_caches")
    ap.add_argument("--out", default="/content/logs/e2_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import re
    import statistics
    import torch

    os.makedirs(args.cache_dir, exist_ok=True)

    # ---- stage A: 4B prefills everything, spill to disk ----
    tok_s, model_s = load_model(args.src_model)
    trials = make_trials(tok_s, args.seed)
    print(f"{len(trials)} trials: " +
          " ".join(f"{t['kind']}@{t['L']}" for t in trials))
    with torch.no_grad():
        for ti, tr in enumerate(trials):
            enc = tok_s(tr["ctx"], return_tensors="pt").to("cuda")
            tr["n_ctx"] = enc["input_ids"].shape[1]
            torch.cuda.synchronize(); t0 = time.time()
            out = model_s(input_ids=enc["input_ids"][:, :-1], use_cache=True)
            torch.cuda.synchronize()
            tr["t_4b_prefill"] = time.time() - t0
            spill(out.past_key_values, f"{args.cache_dir}/t{ti}.npz")
            del out
            torch.cuda.empty_cache()
            print(f"A {ti+1}/{len(trials)} {tr['kind']}@{tr['L']} "
                  f"({tr['n_ctx']} tok): 4B prefill {tr['t_4b_prefill']:.2f}s")
    del model_s
    torch.cuda.empty_cache()

    # ---- stage B: 9B, three conditions per trial ----
    tok_t, model_t = load_model(args.tgt_model)
    assert tok_t(trials[0]["ctx"])["input_ids"] == \
        tok_s(trials[0]["ctx"])["input_ids"], "tokenizer mismatch"
    results = []
    with torch.no_grad():
        for ti, tr in enumerate(trials):
            enc = tok_t(tr["ctx"], return_tensors="pt").to("cuda")
            row = {k: tr[k] for k in ("kind", "L", "n_ctx", "t_4b_prefill")}
            for cond in ("self_prefill", "copy", "no_ctx"):
                torch.cuda.synchronize(); t0 = time.time()
                if cond == "no_ctx":
                    if tr["kind"] == "qa":     # question only
                        q = tok_t(build_q(tr["name"]), return_tensors="pt").to("cuda")
                        o0 = model_t(input_ids=q["input_ids"][:, :-1], use_cache=True)
                        past, last = o0.past_key_values, q["input_ids"][:, -1:]
                    else:                       # bare continuation, no cache
                        past, last = None, None
                else:
                    o0 = model_t(input_ids=enc["input_ids"][:, :-1], use_cache=True)
                    past, last = o0.past_key_values, enc["input_ids"][:, -1:]
                    if cond == "copy":
                        row["t_transplant"] = overwrite_from(
                            past, f"{args.cache_dir}/t{ti}.npz")
                torch.cuda.synchronize()
                t_cache = time.time() - t0

                if tr["kind"] == "qa":
                    torch.cuda.synchronize(); t0 = time.time()
                    out = model_t(input_ids=last, past_key_values=past, use_cache=True)
                    torch.cuda.synchronize()
                    row[f"{cond}_t_step"] = time.time() - t0
                    gold = tok_t(" " + tr["gold"], add_special_tokens=False,
                                 return_tensors="pt")["input_ids"].to("cuda")
                    logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                    cur, nll, gen = out.past_key_values, 0.0, []
                    for gi in range(gold.shape[1]):
                        nll += -float(logp[0, gold[0, gi]])
                        nxt = logp.argmax(-1, keepdim=True)
                        gen.append(int(nxt))
                        o = model_t(input_ids=nxt, past_key_values=cur, use_cache=True)
                        cur = o.past_key_values
                        logp = torch.log_softmax(o.logits[:, -1, :].float(), -1)
                    row[f"{cond}_hit"] = int(tr["gold"] in
                                             re.sub(r"\D", "", tok_t.decode(gen)))
                    row[f"{cond}_nll"] = nll / gold.shape[1]
                else:                           # continuation NLL, one forward
                    cont = torch.tensor([tr["cont_ids"]], device="cuda")
                    if cond == "no_ctx":
                        out = model_t(input_ids=cont)
                        logits = out.logits[:, :-1]
                        targets = cont[:, 1:]
                    else:
                        full = torch.cat([last, cont], 1)
                        out = model_t(input_ids=full, past_key_values=past,
                                      use_cache=True)
                        logits = out.logits[:, :-1]
                        targets = cont
                    lp = torch.log_softmax(logits.float(), -1)
                    row[f"{cond}_nll"] = float(-lp.gather(
                        2, targets.unsqueeze(-1)).mean())
                row[f"{cond}_t_cache"] = t_cache
                del past
                torch.cuda.empty_cache()
            results.append(row)
            print(f"B {ti+1}/{len(trials)} {tr['kind']}@{tr['L']}: " + " ".join(
                f"{c}:" + (f"hit={row.get(c + '_hit')}" if tr["kind"] == "qa"
                           else f"nll={row[c + '_nll']:.3f}")
                for c in ("self_prefill", "copy", "no_ctx")))
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1)

    # ---- summary ----
    print("\n===== E2 KET QUA =====")
    for L in LENGTHS:
        qa = [r for r in results if r["kind"] == "qa" and r["L"] == L]
        co = [r for r in results if r["kind"] == "cont" and r["L"] == L]
        for c in ("self_prefill", "copy", "no_ctx"):
            hits = sum(r[f"{c}_hit"] for r in qa)
            cnll = statistics.mean(r[f"{c}_nll"] for r in co)
            print(f"L={L:5d} {c:13s} QA {hits}/{len(qa)}  cont-NLL {cnll:.3f}")
        t4b = statistics.mean(r["t_4b_prefill"] for r in qa + co)
        t9b = statistics.mean(r["self_prefill_t_cache"] for r in qa + co)
        ttr = statistics.mean(r["t_transplant"] for r in qa + co)
        print(f"L={L:5d} TTFT: 9B self {t9b:.2f}s vs 4B {t4b:.2f}s + "
              f"transplant {ttr:.3f}s = {t4b + ttr:.2f}s (x{t9b / (t4b + ttr):.2f})")
    print("E2_DONE")


if __name__ == "__main__":
    main()
