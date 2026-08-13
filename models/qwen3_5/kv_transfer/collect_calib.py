"""Calibration collector: dump paired per-layer K/V (and GDN state) for the
ridge mapper. GPU job (Colab L4) — models run SEQUENTIALLY, never together
(27B alone is 18.6GB; L4 cannot hold both).

    # run once per model on the SAME texts / seed:
    python models/qwen3_5/kv_transfer/collect_calib.py \
        --model /content/models/champion9b --out /content/calib_9b.npz \
        --n-seqs 200 --seq-len 1024 --stride 4
    python models/qwen3_5/kv_transfer/collect_calib.py \
        --model /content/models/frame27b  --out /content/calib_27b.npz ...

Then fit on CPU:  ridge_mapper.KVMapper(k_sources=4).fit(load(9b), load(27b))

Design notes:
- Uses transformers AutoModelForCausalLM (compressed-tensors checkpoints load
  and dequantize on the fly; slow but calibration is a one-shot batch-1 job).
- Hooks the ATTENTION layers only (layer_types == "full_attention"): captures
  the k_proj/v_proj outputs AFTER RoPE is applied by the model — i.e. keys as
  cached. Positions are saved so the mapper can strip RoPE (sec 3.3).
- Token subsampling stride-4 (paper: N ~= 128K tokens from 500x1024; on L4 we
  start at 200x1024/stride4 ~= 51K tokens — paper's Appendix C shows N=200
  sequences is already within noise of production).
- --dump-gdn-state additionally saves each linear_attention layer's recurrent
  state at end-of-sequence (ONE sample per sequence per layer) for the Phase B
  GDN-state mapping research. NOTE 9B has 32 v-heads vs 27B 48 — do not expect
  the attention-style per-head fit to apply; see MANIFEST.md.
- Calibration texts: FineWeb-Edu sample via datasets if available, else
  local fallback files. Domain matters ~5pp (paper App. C) — keep consistent.

LLM() spawn trap does not apply (transformers, not vLLM), but keep the
__main__ guard anyway per repo rule.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-gdn-state", action="store_true")
    ap.add_argument("--texts", default=None, help="jsonl with {'text': ...}; default: FineWeb-Edu sample")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    cfg = model.config.get_text_config()
    layer_types = list(cfg.layer_types)
    attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    print(f"attention layers: {attn_layers}")

    # ---- calibration texts, deterministic ----
    rng = np.random.default_rng(args.seed)
    if args.texts:
        texts = [json.loads(l)["text"] for l in open(args.texts, encoding="utf-8")]
    else:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        texts = []
        for ex in ds:
            if len(ex["text"]) > 4000:
                texts.append(ex["text"])
            if len(texts) >= args.n_seqs:
                break

    # ---- hooks on attention self_attn to capture per-layer K/V as cached ----
    store = {("K", i): [] for i in attn_layers} | {("V", i): [] for i in attn_layers}
    gdn_store = {}

    def grab(i):
        def hook(module, inp, out):
            # transformers attention returns (attn_out, attn_weights, past_kv...) —
            # we instead read from the cache object after forward; simplest robust
            # route: use output_hidden_states-free forward with use_cache=True and
            # read past_key_values per layer (done below, no hook needed).
            pass
        return hook

    keep = slice(None, None, args.stride)
    K_acc = {i: [] for i in attn_layers}
    V_acc = {i: [] for i in attn_layers}
    pos_acc = []

    with torch.no_grad():
        for si in range(min(args.n_seqs, len(texts))):
            enc = tok(texts[si], return_tensors="pt", truncation=True,
                      max_length=args.seq_len).to("cuda")
            T = enc["input_ids"].shape[1]
            outp = model(**enc, use_cache=True)
            past = outp.past_key_values
            for i in attn_layers:
                k, v = past[i][0], past[i][1]   # (1, n_kv, T, dh)
                K_acc[i].append(k[0].permute(1, 0, 2)[keep].to(torch.float32).cpu().numpy())
                V_acc[i].append(v[0].permute(1, 0, 2)[keep].to(torch.float32).cpu().numpy())
            pos_acc.append(np.arange(T)[keep])
            if args.dump_gdn_state:
                # hybrid cache: recurrent states for linear_attention layers
                st = getattr(past, "recurrent_states", None) or getattr(past, "ssm_states", None)
                if st is not None:
                    for i, t in enumerate(layer_types):
                        if t == "linear_attention" and st[i] is not None:
                            gdn_store.setdefault(i, []).append(
                                st[i][0].to(torch.float32).cpu().numpy())
            if (si + 1) % 20 == 0:
                print(f"{si+1}/{args.n_seqs} seqs")

    out = {"rope_theta": float(cfg.rope_theta),
           "positions": np.concatenate(pos_acc)}
    for i in attn_layers:
        out[f"K_{i}"] = np.concatenate(K_acc[i], axis=0)
        out[f"V_{i}"] = np.concatenate(V_acc[i], axis=0)
    for i, chunks in gdn_store.items():
        out[f"GDN_{i}"] = np.stack(chunks, axis=0)
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}: " +
          ", ".join(f"{k}:{v.shape}" for k, v in out.items() if hasattr(v, 'shape')))


if __name__ == "__main__":
    main()
