"""E1 — cross-model cache transfer 4B->9B: fit, transplant, measure speedup.

Three stages, each a separate nohup job on L4 (models never co-resident):

  1) collect   (GPU, per model ~20-40'):
     python e1_pipeline.py collect --model Qwen/Qwen3.5-4B --out /content/calib_4b.npz
     python e1_pipeline.py collect --model Qwen/Qwen3.5-9B --out /content/calib_9b.npz
     Dumps, on the SAME deterministic texts: per-attention-layer K/V (token-
     subsampled) and per-GDN-layer end-of-sequence recurrent_states + conv_states.

  2) fit       (CPU, minutes):
     python e1_pipeline.py fit --src /content/calib_4b.npz --tgt /content/calib_9b.npz \
         --out /content/mapper_4b_9b.npz
     Attention: per-head ridge, layer-1:1 first (both models: 8 attn layers at
     interval 4) with optional top-k. GDN: per-layer per-head COLUMN-wise ridge
     (each of the 128 d_v columns of a 128-dim key-projected state is one
     sample => 200 seqs x 128 cols = 25.6K samples of dim 128) + conv_states
     copied (shapes identical). Also stores nothing for the "copy" baseline —
     that's applied at eval time.

  3) eval      (GPU, ~30'):
     python e1_pipeline.py eval --mapper /content/mapper_4b_9b.npz \
         --src-model Qwen/Qwen3.5-4B --tgt-model Qwen/Qwen3.5-9B
     Needle-in-context protocol from e0: contexts prefis prefilled by the 4B
     (timed), cache mapped (timed), injected into the 9B, which answers with
     NO prefill (timed). Compared against: 9B self-prefill (quality upper
     bound + latency baseline) and raw copy (cheap baseline).
     Reports: needle accuracy, NLL, and the speedup ledger.

Notes: bf16 originals (champion's ct config trips transformers, STATUS bug);
9B bf16 ~18.4GB fits L4 at batch 1. Kill jobs by PID file (pgrep self-match
trap, see STATUS 2026-08-14).
"""

import argparse
import json
import time


# --------------------------------------------------------------- shared bits

FILLER = ("The study of distributed systems involves consensus, replication, "
          "partitioning, and fault tolerance across unreliable networks. ")

NAMES = ["aurora", "falcon", "meridian", "obsidian", "harbor",
         "juniper", "cascade", "vertex", "quartz", "ember"]


def calib_texts(n_seqs, seed=0):
    """Deterministic calibration texts. FineWeb-Edu if datasets is present,
    else a repeatable synthetic mix (domain matters ~5pp — keep consistent
    between collect runs; the fallback is deterministic so both models see
    byte-identical text either way)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        texts = []
        for ex in ds:
            if len(ex["text"]) > 4000:
                texts.append(ex["text"][:8000])
            if len(texts) >= n_seqs:
                break
        if len(texts) >= n_seqs:
            return texts
    except Exception as e:
        print(f"[calib_texts] datasets unavailable ({type(e).__name__}) — synthetic fallback")
    import random
    rng = random.Random(seed)
    base = FILLER.split()
    return [" ".join(rng.choice(base) for _ in range(1400)) for _ in range(n_seqs)]


def get_rope_theta(cfg) -> float:
    """transformers moved rope_theta around across versions — try every home."""
    for probe in (lambda: cfg.rope_theta,
                  lambda: cfg.rope_parameters["rope_theta"],
                  lambda: cfg.rope_scaling["rope_theta"],
                  lambda: cfg.rotary_emb_base):
        try:
            v = probe()
            if v:
                return float(v)
        except (AttributeError, KeyError, TypeError):
            continue
    raise RuntimeError(f"cannot find rope_theta in {type(cfg).__name__}")


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    return tok, model


def cache_layers(past):
    """Split DynamicCache layers into (attn_idx->layer, gdn_idx->layer)."""
    attn, gdn = {}, {}
    for i, layer in enumerate(past.layers):
        tname = type(layer).__name__
        if "LinearAttention" in tname:
            gdn[i] = layer
        else:
            attn[i] = layer
    return attn, gdn


# ------------------------------------------------------------------ collect

def cmd_collect(args):
    import numpy as np
    import torch
    tok, model = load_model(args.model)
    cfg = model.config.get_text_config()
    texts = calib_texts(args.n_seqs, args.seed)
    keep = slice(None, None, args.stride)

    K_acc, V_acc, pos_acc = {}, {}, []
    GDN_R, GDN_C = {}, {}
    t0 = time.time()
    with torch.no_grad():
        for si, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.seq_len).to("cuda")
            out = model(**enc, use_cache=True)
            attn, gdn = cache_layers(out.past_key_values)
            for i, layer in attn.items():
                k = layer.keys[0].permute(1, 0, 2)[keep]      # (T', n_kv, dh)
                v = layer.values[0].permute(1, 0, 2)[keep]
                K_acc.setdefault(i, []).append(k.to(torch.float32).cpu().numpy())
                V_acc.setdefault(i, []).append(v.to(torch.float32).cpu().numpy())
            pos_acc.append(np.arange(enc["input_ids"].shape[1])[keep])
            for i, layer in gdn.items():
                GDN_R.setdefault(i, []).append(
                    layer.recurrent_states[0].to(torch.float32).cpu().numpy())
                GDN_C.setdefault(i, []).append(
                    layer.conv_states[0].to(torch.float32).cpu().numpy())
            if (si + 1) % 20 == 0:
                print(f"{si+1}/{len(texts)} seqs  ({time.time()-t0:.0f}s)")

    out_d = {"rope_theta": get_rope_theta(cfg),
             "positions": np.concatenate(pos_acc)}
    for i in K_acc:
        out_d[f"K_{i}"] = np.concatenate(K_acc[i], 0)
        out_d[f"V_{i}"] = np.concatenate(V_acc[i], 0)
    for i in GDN_R:
        out_d[f"GDNR_{i}"] = np.stack(GDN_R[i], 0).astype(np.float16)
        out_d[f"GDNC_{i}"] = np.stack(GDN_C[i], 0).astype(np.float16)
    np.savez_compressed(args.out, **out_d)
    print(f"saved {args.out} in {time.time()-t0:.0f}s; keys: {len(out_d)}")


# ---------------------------------------------------------------------- fit

def _layer_ids(d, prefix):
    return sorted(int(k.split("_")[1]) for k in d.files if k.startswith(prefix))


def cmd_fit(args):
    import numpy as np
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ridge_mapper", Path(__file__).parent / "ridge_mapper.py")
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)

    src = np.load(args.src)
    tgt = np.load(args.tgt)
    aid_s, aid_t = _layer_ids(src, "K_"), _layer_ids(tgt, "K_")
    gid_s, gid_t = _layer_ids(src, "GDNR_"), _layer_ids(tgt, "GDNR_")
    assert len(aid_s) == len(aid_t) and len(gid_s) == len(gid_t), \
        f"layer maps differ: attn {aid_s}/{aid_t} gdn {gid_s}/{gid_t}"
    print(f"attn layers 1:1 {list(zip(aid_s, aid_t))}")
    out = {"attn_src": np.array(aid_s), "attn_tgt": np.array(aid_t),
           "gdn_src": np.array(gid_s), "gdn_tgt": np.array(gid_t),
           "theta_src": src["rope_theta"], "theta_tgt": tgt["rope_theta"]}

    pos = src["positions"]
    # Recover sequence ids from position resets (collect stored per-seq
    # positions starting at 0) so we can hold out whole sequences — held-out
    # R2 is the number that matters; train R2 self-flatters.
    seq_id = np.cumsum(np.concatenate([[1], (np.diff(pos) <= 0).astype(int)])) - 1
    n_seq = int(seq_id.max()) + 1
    n_val = min(args.holdout, max(1, n_seq // 5))
    tr_rows, va_rows = seq_id < (n_seq - n_val), seq_id >= (n_seq - n_val)
    print(f"holdout: {n_seq} seqs -> train {n_seq - n_val} / val {n_val} "
          f"({tr_rows.sum()}/{va_rows.sum()} rows)")

    val_attn, val_gdn = [], []
    # attention: per-head ridge, layer-1:1 (identical layouts) in stripped space
    for ls, lt in zip(aid_s, aid_t):
        Ks, Kt = src[f"K_{ls}"], tgt[f"K_{lt}"]         # (N, 4, 256)
        Vs, Vt = src[f"V_{ls}"], tgt[f"V_{lt}"]
        Ks_st = np.stack([rm.strip_rope(Ks[:, h], pos, float(src["rope_theta"]))
                          for h in range(Ks.shape[1])], 1)
        Kt_st = np.stack([rm.strip_rope(Kt[:, h], pos, float(tgt["rope_theta"]))
                          for h in range(Kt.shape[1])], 1)
        XK = Ks_st.reshape(len(Ks_st), -1)
        XV = Vs.reshape(len(Vs), -1)
        r2k, r2v = [], []
        for h in range(Kt.shape[1]):
            Wk, bk = rm.fit_ridge(XK[tr_rows], Kt_st[tr_rows, h])
            Wv, bv = rm.fit_ridge(XV[tr_rows], Vt[tr_rows, h])
            out[f"AK_W_{lt}_{h}"], out[f"AK_b_{lt}_{h}"] = Wk, bk
            out[f"AV_W_{lt}_{h}"], out[f"AV_b_{lt}_{h}"] = Wv, bv
            r2k.append(rm.r2_score(Kt_st[va_rows, h], rm.apply_ridge(XK[va_rows], Wk, bk)))
            r2v.append(rm.r2_score(Vt[va_rows, h], rm.apply_ridge(XV[va_rows], Wv, bv)))
        val_attn += r2k + r2v
        print(f"attn L{ls}->L{lt}: heldout R2(K) mean {np.mean(r2k):.3f} "
              f"(min {np.min(r2k):.3f})  R2(V) mean {np.mean(r2v):.3f} "
              f"(min {np.min(r2v):.3f})")

    # GDN recurrent state: per-layer per-head column-wise ridge.
    # state: (n_heads=32, dk=128, dv=128); columns over dv are samples.
    for ls, lt in zip(gid_s, gid_t):
        Ss = src[f"GDNR_{ls}"].astype(np.float64)   # (S, 32, 128, 128)
        St = tgt[f"GDNR_{lt}"].astype(np.float64)
        S, H, DK, DV = Ss.shape
        s_tr, s_va = slice(0, S - n_val), slice(S - n_val, S)
        r2s = []
        for h in range(H):
            X = Ss[:, h].transpose(0, 2, 1).reshape(S * DV, DK)   # samples: (seq, col)
            Y = St[:, h].transpose(0, 2, 1).reshape(S * DV, DK)
            ntr = (S - n_val) * DV
            # Scale-adaptive lambda: fixed lam=1.0 crushed small-magnitude heads
            # (identity gate caught it: R2 0.047 on quiet heads). lam relative to
            # mean diag(X^T X) makes shrinkage uniform across head scales.
            lam_h = args.gdn_lam * ntr * float(np.mean(X[:ntr] ** 2)) + 1e-12
            W, b = rm.fit_ridge(X[:ntr], Y[:ntr], lam=lam_h)
            out[f"G_W_{lt}_{h}"], out[f"G_b_{lt}_{h}"] = W, b
            r2s.append(rm.r2_score(Y[ntr:], rm.apply_ridge(X[ntr:], W, b)))
        val_gdn += r2s
        print(f"gdn L{ls}->L{lt}: heldout R2 mean {np.mean(r2s):.3f} (min {np.min(r2s):.3f})")

    ma, mg = float(np.mean(val_attn)), float(np.mean(val_gdn))
    print(f"\nTONG KET heldout R2: attention {ma:.4f}  gdn {mg:.4f}")
    if args.expect_identity and (ma < 0.99 or mg < 0.99):
        raise SystemExit(f"IDENTITY GATE FAIL: R2 attn {ma:.4f} / gdn {mg:.4f} "
                         "< 0.99 — plumbing sai, dung lai truoc khi ton GPU eval")
    np.savez_compressed(args.out, **out)
    print(f"saved mapper {args.out}")


# --------------------------------------------------------------------- eval

def transplant(past_src, mapper, mode):
    """Build a 9B cache from the 4B cache. mode: 'ridge' | 'copy'."""
    import numpy as np
    import torch
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ridge_mapper", Path(__file__).parent / "ridge_mapper.py")
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)

    attn, gdn = cache_layers(past_src)
    theta_s, theta_t = float(mapper["theta_src"]), float(mapper["theta_tgt"])
    dev = "cuda"
    # E0 lesson: verify the manipulation actually happened (checksum guard).
    def _cksum():
        s = sum(float(l.keys.float().abs().sum() + l.values.float().abs().sum())
                for l in attn.values())
        s += sum(float(l.recurrent_states.float().abs().sum()) for l in gdn.values())
        return s
    before = _cksum()
    for (i, layer), lt in zip(sorted(attn.items()),
                              [int(x) for x in mapper["attn_tgt"]]):
        if mode == "copy":
            continue
        k = layer.keys[0].permute(1, 0, 2).to(torch.float32).cpu().numpy()  # (T,4,256)
        v = layer.values[0].permute(1, 0, 2).to(torch.float32).cpu().numpy()
        T = k.shape[0]
        pos = np.arange(T)
        k_st = np.stack([rm.strip_rope(k[:, h], pos, theta_s)
                         for h in range(k.shape[1])], 1)
        XK = k_st.reshape(T, -1)
        XV = v.reshape(T, -1)
        new_k, new_v = [], []
        h = 0
        while f"AK_W_{lt}_{h}" in mapper.files:
            kk = rm.apply_ridge(XK, mapper[f"AK_W_{lt}_{h}"], mapper[f"AK_b_{lt}_{h}"])
            new_k.append(rm.apply_rope(kk, pos, theta_t))
            new_v.append(rm.apply_ridge(XV, mapper[f"AV_W_{lt}_{h}"], mapper[f"AV_b_{lt}_{h}"]))
            h += 1
        knew = torch.tensor(np.stack(new_k, 0), dtype=layer.keys.dtype, device=dev)  # (H,T,dh)
        vnew = torch.tensor(np.stack(new_v, 0), dtype=layer.values.dtype, device=dev)
        layer.keys[0] = knew
        layer.values[0] = vnew

    for (i, layer), lt in zip(sorted(gdn.items()),
                              [int(x) for x in mapper["gdn_tgt"]]):
        if mode == "copy":
            continue
        S = layer.recurrent_states[0].to(torch.float32).cpu().numpy()  # (32,128,128)
        H, DK, DV = S.shape
        newS = np.empty_like(S)
        for h in range(H):
            X = S[h].T                                   # (DV, DK) columns as rows
            Y = rm.apply_ridge(X, mapper[f"G_W_{lt}_{h}"], mapper[f"G_b_{lt}_{h}"])
            newS[h] = Y.T
        layer.recurrent_states[0] = torch.tensor(newS, dtype=layer.recurrent_states.dtype,
                                                 device=dev)
        # conv_states copied as-is (identical shapes)
    if mode == "ridge" and _cksum() == before:
        raise RuntimeError("transplant no-op: ridge mode changed NOTHING in the "
                           "cache — mapper/layout mismatch, measurement invalid")
    return past_src


def cmd_eval(args):
    import numpy as np
    import torch
    import random, re, statistics
    mapper = np.load(args.mapper)
    rng = random.Random(args.seed)

    FRACS = (0.25, 0.5, 0.75)          # needle position sweep: catches RoPE bugs
    trials = []
    for ti in range(args.trials):
        name = rng.choice(NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        trials.append((name, code, FRACS[ti % len(FRACS)]))

    def build_ctx(tok, name, code, frac):
        ids = tok(FILLER * 200, add_special_tokens=False)["input_ids"][:args.filler_tokens]
        cut = int(args.filler_tokens * frac)
        pre, post = tok.decode(ids[:cut]), tok.decode(ids[cut:])
        return (f"{pre}\nIMPORTANT: The secret code for project {name} is {code}.\n"
                f"{post}\n{build_q(name)}")

    def build_q(name):
        return (f"Question: What is the secret code for project {name}?\n"
                f"Answer: The secret code for project {name} is")

    # ---- stage 1: 4B prefills all contexts (on [:-1]), save caches ----
    tok_s, model_s = load_model(args.src_model)
    src_caches, t_src_prefill = [], []
    with torch.no_grad():
        for name, code, frac in trials:
            enc = tok_s(build_ctx(tok_s, name, code, frac), return_tensors="pt").to("cuda")
            torch.cuda.synchronize(); t0 = time.time()
            out = model_s(input_ids=enc["input_ids"][:, :-1], use_cache=True)
            torch.cuda.synchronize(); t_src_prefill.append(time.time() - t0)
            src_caches.append(out.past_key_values)
    del model_s
    torch.cuda.empty_cache()
    print(f"4B prefill mean {statistics.mean(t_src_prefill):.3f}s")

    # ---- stage 2: 9B answers under 4 conditions ----
    # Symmetric protocol: EVERY condition builds its cache from tokens [:-1],
    # then feeds the final token — latency and cache length are comparable.
    # no_ctx = floor control (question only, no context at all).
    tok_t, model_t = load_model(args.tgt_model)
    CONDS = ("self_prefill", "ridge", "copy", "no_ctx")
    res = {c: {"hits": 0, "nll": [], "lat": [], "byfrac": {}} for c in CONDS}
    with torch.no_grad():
        for ti, (name, code, frac) in enumerate(trials):
            ctx = build_ctx(tok_t, name, code, frac)
            enc = tok_t(ctx, return_tensors="pt").to("cuda")
            if ti == 0:
                ids_s = tok_s(ctx)["input_ids"]
                assert ids_s == enc["input_ids"][0].tolist(), \
                    "tokenizer 4B != 9B tren cung text — cache khong the ghep"
            gold = tok_t(" " + code, add_special_tokens=False,
                         return_tensors="pt")["input_ids"].to("cuda")
            for cond in CONDS:
                torch.cuda.synchronize(); t0 = time.time()
                if cond == "self_prefill":
                    o0 = model_t(input_ids=enc["input_ids"][:, :-1], use_cache=True)
                    past, last = o0.past_key_values, enc["input_ids"][:, -1:]
                elif cond == "no_ctx":
                    q = tok_t(build_q(name), return_tensors="pt").to("cuda")
                    o0 = model_t(input_ids=q["input_ids"][:, :-1], use_cache=True)
                    past, last = o0.past_key_values, q["input_ids"][:, -1:]
                else:
                    import copy as _c
                    past = transplant(_c.deepcopy(src_caches[ti]), mapper, mode=cond)
                    last = enc["input_ids"][:, -1:]
                out = model_t(input_ids=last, past_key_values=past, use_cache=True)
                torch.cuda.synchronize()
                lat = time.time() - t0
                logp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
                cur, nll, gen = out.past_key_values, 0.0, []
                for gi in range(gold.shape[1]):
                    nll += -float(logp[0, gold[0, gi]])
                    nxt = logp.argmax(-1, keepdim=True)
                    gen.append(int(nxt))
                    o = model_t(input_ids=nxt, past_key_values=cur, use_cache=True)
                    cur = o.past_key_values
                    logp = torch.log_softmax(o.logits[:, -1, :].float(), -1)
                ok = code in re.sub(r"\D", "", tok_t.decode(gen))
                res[cond]["hits"] += int(ok)
                res[cond]["nll"].append(nll / gold.shape[1])
                res[cond]["lat"].append(lat)
                bf = res[cond]["byfrac"].setdefault(frac, [0, 0])
                bf[0] += int(ok); bf[1] += 1
            print(f"trial {ti+1}/{len(trials)} (needle@{frac:.0%}): " +
                  " ".join(f"{c}:{res[c]['hits']}" for c in CONDS))

    print("\n===== E1 KET QUA =====")
    print(f"4B prefill:            {statistics.mean(t_src_prefill):.3f}s")
    base_hits = max(res["self_prefill"]["hits"], 1)
    for c in CONDS:
        r = res[c]
        frs = " ".join(f"@{f:.0%}:{v[0]}/{v[1]}" for f, v in sorted(r["byfrac"].items()))
        print(f"{c:13s} needle {r['hits']}/{len(trials)}  "
              f"retention {100 * r['hits'] / base_hits:.0f}%  "
              f"NLL {statistics.mean(r['nll']):.3f}  "
              f"lat {statistics.mean(r['lat']):.3f}s  [{frs}]")
    t4b = statistics.mean(t_src_prefill)
    tself = statistics.mean(res["self_prefill"]["lat"])
    tridge = statistics.mean(res["ridge"]["lat"])
    print(f"\nTTFT ledger: 9B self {tself:.3f}s  vs  4B prefill {t4b:.3f}s "
          f"+ transplant&step {tridge:.3f}s = {t4b + tridge:.3f}s "
          f"(x{tself / (t4b + tridge):.2f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True); c.add_argument("--out", required=True)
    c.add_argument("--n-seqs", type=int, default=200)
    c.add_argument("--seq-len", type=int, default=1024)
    c.add_argument("--stride", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit")
    f.add_argument("--src", required=True); f.add_argument("--tgt", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--gdn-lam", type=float, default=1e-3,
                   help="ridge GDN tuong doi theo mean diag(X^T X)")
    f.add_argument("--holdout", type=int, default=40,
                   help="so seq giu lai de cham R2 (khong dung de fit)")
    f.add_argument("--expect-identity", action="store_true",
                   help="cong sanity: src==tgt thi R2 heldout phai ~1.0, khong thi fail")
    e = sub.add_parser("eval")
    e.add_argument("--mapper", required=True)
    e.add_argument("--src-model", required=True); e.add_argument("--tgt-model", required=True)
    e.add_argument("--trials", type=int, default=10)
    e.add_argument("--filler-tokens", type=int, default=1500)
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
