"""E7 — full pair matrix sweep (user order 2026-08-15):
2->4, 4->9, 2->9, 2->27, 4->27, 0.8->4, 0.8->9, 0.8->27.

Config census says the family has THREE cache-shape clans:
  {0.8B, 2B}: 24 layers, kv 2x256, GDN 16x128
  {4B,  9B}: 32 layers, kv 4x256, GDN 32x128   (raw copy proven 100%)
  {27B}:     64 layers, kv 4x256, GDN 48x128
Cross-clan pairs from the small clan have INTEGER head ratios (kv x2,
GDN x2 or x3), so a zero-train TILE transplant exists: repeat each source
head, depth-map layers by relative position. 4->27 has non-integer GDN
(32->48) — no tile; its trained-mapper answer comes from E6.

Per pair this sweep measures:
  - needle retention (5 trials, ~800-tok ctx) under tile-transplant vs
    self vs no_ctx on the target
  - CCA (linear shared structure) for attention-K and GDN mid-layers
Models are never co-resident: phase per SOURCE model spills caches+samples,
phase per TARGET model evaluates. 27B runs bnb-4bit.
"""

import argparse
import gc
import importlib.util
import json
import random
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)
e2 = e5.e2
spec4 = importlib.util.spec_from_file_location(
    "e4_stats", Path(__file__).parent / "e4_stats.py")
e4 = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(e4)

SIZES = ["0.8B", "2B", "4B", "9B", "27B"]
PAIRS = [("2B", "4B"), ("4B", "9B"), ("2B", "9B"), ("2B", "27B"),
         ("4B", "27B"), ("0.8B", "4B"), ("0.8B", "9B"), ("0.8B", "27B")]
SOURCES = sorted({s for s, _ in PAIRS}, key=SIZES.index)
TARGETS = sorted({t for _, t in PAIRS}, key=SIZES.index)
N_TRIALS = 5
CTX_TOK = 800
N_STAT = 24          # texts for CCA samples
STAT_LEN = 512


def model_name(size):
    return f"Qwen/Qwen3.5-{size}"


def make_trials(tok):
    rng = random.Random(0)
    trials = []
    for ti in range(N_TRIALS):
        name = rng.choice(e2.NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = e2.token_stream(tok, CTX_TOK, seed=700 + ti)
        half = CTX_TOK // 2
        ctx = (tok.decode(ids[:half])
               + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
               + tok.decode(ids[half:]) + "\n" + e2.build_q(name))
        trials.append((name, code, ctx))
    return trials


def tile_into(tpl_past, src_past):
    """Zero-train transplant: depth-map layers, repeat heads by integer ratio.
    Returns None if any ratio is non-integer (pair has no tile form)."""
    import torch
    attn_s, gdn_s = e5.split_layers(src_past)
    attn_t, gdn_t = e5.split_layers(tpl_past)
    ks, kt = sorted(attn_s), sorted(attn_t)
    gs, gt = sorted(gdn_s), sorted(gdn_t)
    Hs_a = attn_s[ks[0]].keys.shape[1]
    Ht_a = attn_t[kt[0]].keys.shape[1]
    Hs_g = gdn_s[gs[0]].recurrent_states.shape[1]
    Ht_g = gdn_t[gt[0]].recurrent_states.shape[1]
    if Ht_a % Hs_a or Ht_g % Hs_g:
        return None
    ra, rg = Ht_a // Hs_a, Ht_g // Hs_g
    amap = e5.depth_map(len(ks), len(kt))
    for j, it in enumerate(kt):
        src = attn_s[ks[amap[j]]]
        attn_t[it].keys = src.keys.repeat_interleave(ra, dim=1)
        attn_t[it].values = src.values.repeat_interleave(ra, dim=1)
    gmap = e5.depth_map(len(gs), len(gt))
    for j, it in enumerate(gt):
        src = gdn_s[gs[gmap[j]]]
        gdn_t[it].recurrent_states = src.recurrent_states.repeat_interleave(rg, dim=1)
        if gdn_t[it].conv_states.shape == src.conv_states.shape:
            gdn_t[it].conv_states = src.conv_states.clone()
        else:
            gdn_t[it].conv_states = torch.zeros_like(gdn_t[it].conv_states)
    return tpl_past


def spill_stats(model, tok, path):
    """Mid-layer K rows + GDN states over shared stat texts."""
    import numpy as np
    import torch
    stream = e2.token_stream(tok, N_STAT * STAT_LEN + N_STAT, seed=5)
    texts = [tok.decode(stream[i * STAT_LEN:(i + 1) * STAT_LEN])
             for i in range(N_STAT)]
    Ks, Gs = [], []
    with torch.no_grad():
        for text in texts:
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=STAT_LEN).to("cuda")
            past = model(**enc, use_cache=True, logits_to_keep=1).past_key_values
            attn, gdn = e5.split_layers(past)
            ai = sorted(attn)[len(attn) // 2]
            gi = sorted(gdn)[len(gdn) // 2]
            k = attn[ai].keys[0].permute(1, 0, 2)[::4]
            Ks.append(k.reshape(len(k), -1).float().cpu().numpy())
            Gs.append(gdn[gi].recurrent_states[0].float().cpu().numpy())
            del past
            torch.cuda.empty_cache()
    np.savez(path, K=np.concatenate(Ks), G=np.stack(Gs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/e7")
    ap.add_argument("--results", default="/content/logs/e7_results.json")
    args = ap.parse_args()

    import numpy as np
    import torch

    wd = Path(args.work)
    wd.mkdir(parents=True, exist_ok=True)

    # ---- phase 1: each SOURCE spills needle caches + stat samples ----
    trials = None
    for size in SOURCES:
        if (wd / f"stats_{size}.npz").exists():
            continue
        tok, model = e5.load_4bit(model_name(size))
        if trials is None:
            trials = make_trials(tok)
        with torch.no_grad():
            for ti, (_, _, ctx) in enumerate(trials):
                enc = tok(ctx, return_tensors="pt").to("cuda")
                past = model(input_ids=enc["input_ids"][:, :-1],
                             use_cache=True, logits_to_keep=1).past_key_values
                e5.spill_cache(past, wd / f"nc_{size}_{ti}.pt")
                del past
                torch.cuda.empty_cache()
        spill_stats(model, tok, wd / f"stats_{size}.npz")
        del model
        gc.collect(); torch.cuda.empty_cache()
        print(f"SOURCE {size} spilled")

    # ---- phase 2: each TARGET evaluates its pairs ----
    results = {}
    for tsize in TARGETS:
        tok, model = e5.load_4bit(model_name(tsize))
        if trials is None:
            trials = make_trials(tok)
        pairs_here = [s for s, t in PAIRS if t == tsize]
        res = {f"{s}->{tsize}": 0 for s in pairs_here}
        res["self"], res["no_ctx"] = 0, 0
        na = set()
        import re
        with torch.no_grad():
            # stat samples for the target too (for CCA)
            if not (wd / f"stats_{tsize}.npz").exists():
                spill_stats(model, tok, wd / f"stats_{tsize}.npz")
            for ti, (name, code, ctx) in enumerate(trials):
                enc = tok(ctx, return_tensors="pt").to("cuda")
                pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]

                def answer(past, inp):
                    cur, gen = past, []
                    for _ in range(10):
                        o = model(input_ids=inp, past_key_values=cur,
                                  use_cache=True)
                        cur = o.past_key_values
                        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                        gen.append(int(inp))
                    return int(code in re.sub(r"\D", "", tok.decode(gen)))

                tpl = model(input_ids=pre, use_cache=True,
                            logits_to_keep=1).past_key_values
                res["self"] += answer(tpl, last)
                q = tok(e2.build_q(name), return_tensors="pt").to("cuda")
                p0 = model(input_ids=q["input_ids"][:, :-1], use_cache=True,
                           logits_to_keep=1).past_key_values
                res["no_ctx"] += answer(p0, q["input_ids"][:, -1:])
                for s in pairs_here:
                    key = f"{s}->{tsize}"
                    if key in na:
                        continue
                    src = e5.load_cache(wd / f"nc_{s}_{ti}.pt")
                    tpl2 = model(input_ids=pre, use_cache=True,
                                 logits_to_keep=1).past_key_values
                    tiled = tile_into(tpl2, src)
                    if tiled is None:
                        na.add(key)
                        res[key] = "N/A (ty le head khong nguyen)"
                        continue
                    res[key] += answer(tiled, last)
                    del src, tpl2
                    torch.cuda.empty_cache()
                del tpl
                torch.cuda.empty_cache()
            print(f"TARGET {tsize}: {res}")
        results[tsize] = res
        del model
        gc.collect(); torch.cuda.empty_cache()

    # ---- phase 3: CCA + variance-explained (heldout ridge R2) per pair ----
    def ridge_r2(X, Y):
        """Heldout variance of the TARGET explained by a linear map from the
        source (dims may differ). The user-requested 'variance explained'."""
        n = min(len(X), len(Y))
        X, Y = X[:n].astype(np.float64), Y[:n].astype(np.float64)
        ntr = int(n * 0.8)
        mx, my = X[:ntr].mean(0), Y[:ntr].mean(0)
        Xc, Yc = X[:ntr] - mx, Y[:ntr] - my
        lam = 1e-3 * ntr * float((Xc ** 2).mean()) + 1e-12
        W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
        pred = (X[ntr:] - mx) @ W + my
        ss = ((Y[ntr:] - pred) ** 2).sum()
        return float(1 - ss / (((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum() + 1e-12))

    cca_out = {}
    for s, t in PAIRS:
        zs = np.load(wd / f"stats_{s}.npz")
        zt = np.load(wd / f"stats_{t}.npz")
        n = min(len(zs["K"]), len(zt["K"]))
        rhos = e4.cca(zs["K"][:n], zt["K"][:n])
        cca_k = float(np.mean(rhos))
        shared_var_k = float(np.mean([r * r for r in rhos]))
        r2_k = ridge_r2(zs["K"], zt["K"])
        Gs, Gt = zs["G"], zt["G"]
        m = min(len(Gs), len(Gt))
        Xs = Gs[:m, 0].transpose(0, 2, 1).reshape(-1, Gs.shape[2])
        Xt = Gt[:m, 0].transpose(0, 2, 1).reshape(-1, Gt.shape[2])
        rhog = e4.cca(Xs, Xt)
        cca_g = float(np.mean(rhog))
        r2_g = ridge_r2(Xs, Xt)
        cca_out[f"{s}->{t}"] = {
            "cca_attnK": round(cca_k, 3), "cca_gdn": round(cca_g, 3),
            "var_explained_attnK": round(r2_k, 3),
            "var_explained_gdn": round(r2_g, 3),
            "shared_var_cca_top64": round(shared_var_k, 3)}
        print(f"{s}->{t}: CCA attnK {cca_k:.3f} gdn {cca_g:.3f} | "
              f"VAR-EXPLAINED attnK {r2_k:.3f} gdn {r2_g:.3f}")

    with open(args.results, "w") as fh:
        json.dump({"needle": results, "cca": cca_out,
                   "n_trials": N_TRIALS}, fh, indent=1)
    print("saved", args.results)
    print("E7_ALL_DONE")


if __name__ == "__main__":
    main()
