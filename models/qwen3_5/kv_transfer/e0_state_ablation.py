"""E0 — where does context memory live in hybrid Qwen3.5: GDN state or KV?

Decides the kv_transfer direction (see MANIFEST.md):
- If zeroing GDN state after prefill barely hurts -> attention-only cross-model
  KV transfer (paper's recipe, matched-KV verified) is already viable.
- If it destroys context use -> Phase B (GDN-state mapping research) is
  mandatory before any transfer works.

Protocol (needle-in-context + NLL):
  For each of N trials: build a ~1.5K-token context with an embedded fact
  ("The secret code for <name> is <6 digits>"), prefill once, then generate
  the answer to "What is the secret code for <name>?" under three cache
  conditions: (a) intact; (b) GDN state zeroed, attention KV kept;
  (c) attention KV zeroed, GDN state kept. Report needle accuracy per
  condition + mean NLL of the gold digits.

GPU job (~20-30 min on L4). Run via:
  nohup python models/qwen3_5/kv_transfer/e0_state_ablation.py \
      --model /content/models/champion9b --trials 10 > /content/logs/e0.log 2>&1 &

Note: transformers (not vLLM), batch 1 — __main__ guard kept per repo rule.
"""

import argparse
import copy
import random
import re


def zero_cache_parts(past, *, zero_gdn: bool, zero_attn: bool, verbose=False):
    """Zero tensors inside a transformers hybrid cache, walking the object
    RECURSIVELY (tensors live in past.layers[i].<attr>, not at top level —
    the first version scanned only top-level attrs, zeroed nothing, and made
    all conditions identical; hard-fail guard below prevents that recurrence).
    """
    import torch
    hit = []

    def classify(name):
        n = name.lower()
        if any(s in n for s in ("recurrent", "ssm", "conv")):
            return "gdn"
        if any(s in n for s in ("key", "value")):
            return "attn"
        return None

    def visit(obj, prefix, depth):
        if depth > 4 or obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for i, o in enumerate(obj):
                visit(o, f"{prefix}[{i}]", depth + 1)
            return
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(obj, attr)
            except Exception:
                continue
            kind = classify(attr)
            if torch.is_tensor(val):
                if (zero_gdn and kind == "gdn") or (zero_attn and kind == "attn"):
                    val.zero_(); hit.append(f"{prefix}.{attr}")
            elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
                if (zero_gdn and kind == "gdn") or (zero_attn and kind == "attn"):
                    for t in val:
                        t.zero_()
                    hit.append(f"{prefix}.{attr}[*]")
            elif attr == "layers":
                visit(val, f"{prefix}.layers", depth + 1)

    visit(past, "past", 0)
    if verbose:
        names = sorted({h.split('.')[-1] for h in hit})
        print(f"  zeroed {len(hit)} tensors; attr names: {names}")
        print("  cache type:", type(past).__name__,
              "| layer types:", sorted({type(l).__name__ for l in getattr(past, 'layers', [])}))
    if not hit:
        raise RuntimeError(
            "zero_cache_parts hit NOTHING — cache layout unknown; measurement "
            "would be invalid (all conditions identical). Inspect cache attrs.")
    return past


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--filler-tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    FILLER = ("The study of distributed systems involves consensus, replication, "
              "partitioning, and fault tolerance across unreliable networks. ")

    results = {c: {"hits": 0, "nll": []} for c in ("intact", "no_gdn", "no_attn")}

    for trial in range(args.trials):
        name = rng.choice(["aurora", "falcon", "meridian", "obsidian", "harbor",
                           "juniper", "cascade", "vertex", "quartz", "ember"])
        code = "".join(rng.choice("0123456789") for _ in range(6))
        filler = (FILLER * 200)
        filler_ids = tok(filler, add_special_tokens=False)["input_ids"][:args.filler_tokens]
        pre = tok.decode(filler_ids[: args.filler_tokens // 2])
        post = tok.decode(filler_ids[args.filler_tokens // 2:])
        context = (f"{pre}\nIMPORTANT: The secret code for project {name} is {code}.\n"
                   f"{post}\nQuestion: What is the secret code for project {name}?\n"
                   f"Answer: The secret code for project {name} is")
        enc = tok(context, return_tensors="pt").to("cuda")
        gold_ids = tok(" " + code, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")

        with torch.no_grad():
            base_out = model(**enc, use_cache=True)

        for cond in ("intact", "no_gdn", "no_attn"):
            with torch.no_grad():
                past = copy.deepcopy(base_out.past_key_values)
                if cond == "no_gdn":
                    zero_cache_parts(past, zero_gdn=True, zero_attn=False,
                                     verbose=(trial == 0))
                elif cond == "no_attn":
                    zero_cache_parts(past, zero_gdn=False, zero_attn=True,
                                     verbose=(trial == 0))
                # NLL of gold digits under teacher forcing
                nll, cur = 0.0, past
                logits = base_out.logits[:, -1:, :] if cond == "intact" else None
                # recompute last-token logits against the modified cache:
                step_in = enc["input_ids"][:, -1:]
                out = model(input_ids=step_in, past_key_values=cur, use_cache=True)
                # NOTE: feeding the last token again double-counts it in the
                # cache; acceptable for a relative probe (same for all conds).
                cur = out.past_key_values
                logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                gen = []
                for gi in range(gold_ids.shape[1]):
                    tgt = gold_ids[0, gi]
                    nll += -float(logp[0, tgt])
                    nxt = logp.argmax(-1, keepdim=True)
                    gen.append(int(nxt))
                    out = model(input_ids=nxt, past_key_values=cur, use_cache=True)
                    cur = out.past_key_values
                    logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                text = tok.decode(gen)
                ok = code in re.sub(r"\D", "", text) or code in text
                results[cond]["hits"] += int(ok)
                results[cond]["nll"].append(nll / gold_ids.shape[1])
        print(f"trial {trial+1}/{args.trials}: " + " | ".join(
            f"{c}:{'HIT' if results[c]['hits'] > sum(1 for _ in range(trial)) - (trial - results[c]['hits']) and False else results[c]['hits']}"
            for c in results))

    print("\n===== E0 KET QUA =====")
    for cond, r in results.items():
        import statistics
        print(f"{cond:8s}  needle {r['hits']}/{args.trials}   "
              f"NLL/token {statistics.mean(r['nll']):.3f}")
    print("Dien giai: no_gdn ~ intact => context song o attention KV (transfer kha thi ngay);"
          " no_gdn sup => can Phase B (GDN mapping).")


if __name__ == "__main__":
    main()
