"""E4 — cache-distribution statistics: pick the mapping algorithm from DATA.

User directive 2026-08-15: no algorithm proposals by guessing — measure the
distributions first, let the numbers choose. This job answers, per layer/head:

  1) SHAPE of each cache distribution: per-dim variance profile, kurtosis
     (heavy tails => MSE-based fits misbehave), effective rank of K/V/GDN.
  2) GEOMETRY between source and target spaces (the E1 postmortem made
     precise): fit ridge A-hat and measure ||A-hat - I||_F / ||A-hat||_F and
     its singular-value range (all ~1 => the true map is a rotation, and
     shrinkage — not misalignment — is what killed E1's ridge).
  3) Held-out FUNCTIONAL-PROXY comparison of every lightweight closed-form
     candidate on the same split:
        identity            h_t = h_s              (0 params)
        procrustes          h_t = h_s R, R = UV^T  (rotation only, NO shrink)
        scaled-procrustes   h_t = s h_s R          (+1 scalar)
        ridge               h_t = h_s A-hat        (paper baseline, 1 layer)
        concat-ridge        h_t = [h_s^L1..h_s^Lk] W  (paper FULL recipe:
                            concatenate ALL source layers -> giant feature;
                            user 2026-08-15 flagged E1 silently used the 1:1
                            degenerate form — measure both, let numbers talk)
  4) Cross-capacity: canonical correlations between spaces (how much linearly
     shared structure exists at all — the go/no-go number for any linear map,
     and the honest ceiling estimate for mismatched pairs like x->27B).

Small-N is fine for statistics (64 texts x 1024 tok); outputs a compact JSON.
Models run sequentially (never co-resident). 27B is optional via bitsandbytes
4-bit (bf16 does not fit an L4).

  python e4_stats.py --models Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
      --out /content/logs/e4_stats.json
"""

import argparse
import json
import time

N_TEXTS = 64
SEQ_LEN = 1024
STRIDE = 4
HOLDOUT_FRAC = 0.2
TOPK_CCA = 64


def collect(model_name, texts, four_bit=False):
    """Return {"attn": {layer: {"K": arr(N,H,D), "V": ...}}, "gdn": {...}}."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = dict(device_map="cuda")
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    K, V, G = {}, {}, {}
    with torch.no_grad():
        for si, text in enumerate(texts):
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=SEQ_LEN).to("cuda")
            out = model(**enc, use_cache=True, logits_to_keep=1)
            for i, layer in enumerate(out.past_key_values.layers):
                if "LinearAttention" in type(layer).__name__:
                    G.setdefault(i, []).append(
                        layer.recurrent_states[0].float().cpu().numpy())
                else:
                    k = layer.keys[0].permute(1, 0, 2)[::STRIDE]
                    v = layer.values[0].permute(1, 0, 2)[::STRIDE]
                    K.setdefault(i, []).append(k.float().cpu().numpy())
                    V.setdefault(i, []).append(v.float().cpu().numpy())
            if (si + 1) % 16 == 0:
                print(f"  {model_name}: {si+1}/{len(texts)}")
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect(); torch.cuda.empty_cache()
    return ({i: {"K": np.concatenate(K[i]), "V": np.concatenate(V[i])} for i in K},
            {i: np.stack(G[i]) for i in G})


def dist_stats(X):
    """Per-matrix distribution shape numbers. X: (N, D)."""
    import numpy as np
    Xc = X - X.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    z = Xc / (Xc.std(0) + 1e-9)
    return {"eff_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
            "kurtosis_med": float(np.median((z ** 4).mean(0) - 3)),
            "norm_cv": float(np.linalg.norm(X, axis=1).std()
                             / (np.linalg.norm(X, axis=1).mean() + 1e-9))}


def fit_candidates(X, Y):
    """Held-out R2 for each closed-form candidate + geometry of A-hat.
    X, Y: (N, D) paired samples, SAME D (matched spaces)."""
    import numpy as np
    n = len(X)
    ntr = int(n * (1 - HOLDOUT_FRAC))
    Xtr, Ytr, Xva, Yva = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my

    def r2(pred):
        ss = ((Yva - pred) ** 2).sum()
        return float(1 - ss / (((Yva - Yva.mean(0)) ** 2).sum() + 1e-12))

    out = {"identity": r2(Xva)}
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt
    out["procrustes"] = r2((Xva - mx) @ R + my)
    s = S.sum() / ((Xc ** 2).sum() + 1e-12)
    out["scaled_procrustes"] = r2(s * ((Xva - mx) @ R) + my)
    lam = 1e-3 * ntr * float((Xc ** 2).mean())
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    out["ridge"] = r2((Xva - mx) @ A + my)
    svA = np.linalg.svd(A, compute_uv=False)
    out["A_dist_I"] = float(np.linalg.norm(A - np.eye(len(A))) / (np.linalg.norm(A) + 1e-12))
    out["A_sv_min"], out["A_sv_max"] = float(svA.min()), float(svA.max())
    return out


def cca(X, Y, k=TOPK_CCA):
    """Top-k canonical correlations — linear shared structure, any dims."""
    import numpy as np
    n = len(X)
    ntr = int(n * (1 - HOLDOUT_FRAC))

    def whiten(Z):
        Zc = Z[:ntr] - Z[:ntr].mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        keep = S > S[0] * 1e-6
        return Vt[keep].T / S[keep], Z[:ntr].mean(0)

    Wx, mx = whiten(X)
    Wy, my = whiten(Y)
    C = ((X[:ntr] - mx) @ Wx).T @ ((Y[:ntr] - my) @ Wy)
    rho = np.linalg.svd(C, compute_uv=False)
    return [float(r) for r in np.clip(rho, 0, 1)[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"])
    ap.add_argument("--four-bit", nargs="*", default=[],
                    help="model names to load in bnb-4bit (27B khong vua bf16)")
    ap.add_argument("--out", default="/content/logs/e4_stats.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    from transformers import AutoTokenizer
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "e2_suite", Path(__file__).parent / "e2_suite.py")
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)

    tok0 = AutoTokenizer.from_pretrained(args.models[0])
    stream = e2.token_stream(tok0, N_TEXTS * SEQ_LEN + N_TEXTS, seed=7)
    texts = [tok0.decode(stream[i * SEQ_LEN:(i + 1) * SEQ_LEN])
             for i in range(N_TEXTS)]

    packs = {}
    for m in args.models:
        print(f"== collect {m}")
        try:
            packs[m] = collect(m, texts, four_bit=(m in args.four_bit))
        except Exception as e:
            print(f"BO QUA {m}: {type(e).__name__}: {e}")
    args.models = [m for m in args.models if m in packs]

    report = {"models": args.models, "n_texts": N_TEXTS}
    # ---- per-model distribution shape ----
    for m, (attn, gdn) in packs.items():
        rep = {}
        for i, kv in attn.items():
            flat = kv["K"].reshape(len(kv["K"]), -1)
            rep[f"attnK_{i}"] = dist_stats(flat)
        for i, S in gdn.items():
            H = S.shape[1]
            cols = S.transpose(0, 1, 3, 2).reshape(len(S) * 1, H, -1)
            rep[f"gdn_{i}"] = dist_stats(S.reshape(len(S), -1))
        report[f"dist::{m}"] = rep
        print(f"{m}: dist stats done ({len(rep)} layers)")

    # ---- pairwise geometry ----
    for a in range(len(args.models)):
        for b in range(a + 1, len(args.models)):
            ma, mb = args.models[a], args.models[b]
            (attn_a, gdn_a), (attn_b, gdn_b) = packs[ma], packs[mb]
            ids_a, ids_b = sorted(attn_a), sorted(attn_b)
            pair_rep = {}
            # align attention layers by relative depth
            for j, ia in enumerate(ids_a):
                ib = ids_b[min(int(round(j * (len(ids_b) - 1)
                                         / max(len(ids_a) - 1, 1))),
                               len(ids_b) - 1)]
                Xa = attn_a[ia]["K"].reshape(len(attn_a[ia]["K"]), -1)
                Xb = attn_b[ib]["K"].reshape(len(attn_b[ib]["K"]), -1)
                n = min(len(Xa), len(Xb))
                entry = {"cca_mean_top64": float(np.mean(cca(Xa[:n], Xb[:n])))}
                if Xa.shape[1] == Xb.shape[1]:
                    entry.update(fit_candidates(Xa[:n], Xb[:n]))
                # paper FULL recipe: concat ALL source layers -> ridge
                Xcat = np.concatenate(
                    [attn_a[i]["K"].reshape(len(attn_a[i]["K"]), -1)[:n]
                     for i in ids_a], axis=1)
                ntr = int(n * (1 - HOLDOUT_FRAC))
                Xc = Xcat[:ntr] - Xcat[:ntr].mean(0)
                Yc = Xb[:ntr] - Xb[:ntr].mean(0)
                lam = 1e-3 * ntr * float((Xc ** 2).mean())
                W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]),
                                    Xc.T @ Yc)
                pred = (Xcat[ntr:] - Xcat[:ntr].mean(0)) @ W + Xb[:ntr].mean(0)
                ss = ((Xb[ntr:] - pred) ** 2).sum()
                entry["concat_ridge"] = float(
                    1 - ss / (((Xb[ntr:] - Xb[ntr:].mean(0)) ** 2).sum() + 1e-12))
                pair_rep[f"K_L{ia}->L{ib}"] = entry
                print(f"{ma}->{mb} K L{ia}->L{ib}: {entry}")
            # GDN: column samples per matched head where head counts allow
            ga, gb = sorted(gdn_a), sorted(gdn_b)
            for j in (0, len(ga) // 2, len(ga) - 1):
                ia = ga[j]
                ib = gb[min(int(round(j * (len(gb) - 1) / max(len(ga) - 1, 1))),
                            len(gb) - 1)]
                Sa, Sb = gdn_a[ia], gdn_b[ib]         # (N,H,dk,dv)
                Ha, Hb = Sa.shape[1], Sb.shape[1]
                Xa = Sa[:, 0].transpose(0, 2, 1).reshape(-1, Sa.shape[2])
                Xb = Sb[:, 0].transpose(0, 2, 1).reshape(-1, Sb.shape[2])
                n = min(len(Xa), len(Xb))
                entry = {"heads": f"{Ha}vs{Hb}",
                         "cca_mean_top64": float(np.mean(cca(Xa[:n], Xb[:n])))}
                if Sa.shape[2] == Sb.shape[2]:
                    entry.update(fit_candidates(Xa[:n], Xb[:n]))
                pair_rep[f"GDN_L{ia}->L{ib}_h0"] = entry
                print(f"{ma}->{mb} GDN L{ia}->L{ib}: {entry}")
            report[f"pair::{ma}->{mb}"] = pair_rep

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", args.out)
    print("E4_STATS_DONE")


if __name__ == "__main__":
    main()
