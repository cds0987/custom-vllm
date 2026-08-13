"""Closed-form per-head ridge mapper for cross-model KV cache transfer.

Implementation of arXiv:2608.03893 (NVIDIA, 2026-08) — no official code exists
(checked paper text + web, 2026-08-14), so this is written from the paper's
equations:

  W* = (X^T X + lambda*I)^-1 X^T Y          (eq. 4, lambda=0.01)
  X, Y centered before solving; bias b = mean(Y) - mean(X) @ W*
  X = concat of top-k source layers' per-head features   (eq. 5)
  Keys are mapped in RoPE-STRIPPED content space:
      K_target = (K_src @ R_src(t)^-1 @ W_K + b_K) @ R_tgt(t)   (sec 3.3)
  Values carry no position -> mapped directly.

Qwen3.5 9B<->27B specifics (configs fetched 2026-08-14):
  - ATTENTION layers are matched-KV: both have num_key_value_heads=4,
    head_dim=256 — same precondition as the paper's best pair (Qwen3 14B->32B,
    97.6% retention). 9B has 8 attention layers (interval 4 of 32), 27B has 16.
  - GDN/linear-attention layers (3/4 of all layers) carry recurrent state with
    MISMATCHED value heads (9B: 32, 27B: 48) — outside the paper's scope
    (their declared future work). See MANIFEST.md for the phase plan.

Everything here is plain NumPy and runs on CPU. Calibration collection (paired
K/V from both models on the same texts) happens on GPU via collect_calib.py.
"""

import numpy as np

LAMBDA_DEFAULT = 0.01


# ---------------------------------------------------------------- ridge core

def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float = LAMBDA_DEFAULT):
    """Closed-form centered ridge. X: (N, Din), Y: (N, Dout).
    Returns (W (Din, Dout), b (Dout,))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_mean, y_mean = X.mean(0), Y.mean(0)
    Xc, Yc = X - x_mean, Y - y_mean
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Yc)
    b = y_mean - x_mean @ W
    return W, b


def apply_ridge(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ W + b


def r2_score(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    Y_true = np.asarray(Y_true, dtype=np.float64)
    ss_res = float(((Y_true - Y_pred) ** 2).sum())
    ss_tot = float(((Y_true - Y_true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ------------------------------------------------------------- RoPE stripping

def rope_rotation(positions: np.ndarray, head_dim: int, theta: float = 1e6):
    """cos/sin tables for the standard rotate-half RoPE at given positions.
    Qwen3.5 uses rope_theta=1e6 (pass the model's actual value).
    Returns (cos, sin) each (T, head_dim//2)."""
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) / half)
    ang = np.outer(positions.astype(np.float64), inv_freq)
    return np.cos(ang), np.sin(ang)


def _rotate(k: np.ndarray, cos: np.ndarray, sin: np.ndarray, inverse: bool):
    """Apply (or invert) rotate-half RoPE. k: (T, head_dim)."""
    half = k.shape[-1] // 2
    k1, k2 = k[..., :half], k[..., half:]
    if inverse:
        sin = -sin
    out = np.empty_like(k)
    out[..., :half] = k1 * cos - k2 * sin
    out[..., half:] = k2 * cos + k1 * sin
    return out


def strip_rope(k: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    cos, sin = rope_rotation(positions, k.shape[-1], theta)
    return _rotate(k, cos, sin, inverse=True)


def apply_rope(k: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    cos, sin = rope_rotation(positions, k.shape[-1], theta)
    return _rotate(k, cos, sin, inverse=False)


# ------------------------------------------------- top-k source-layer choice

def select_topk_sources(src_feats: dict, tgt_feats: dict, k: int, lam: float = LAMBDA_DEFAULT):
    """Paper sec 3.2: for each target layer pick the k source layers with the
    highest single-source head-averaged R^2 (screening with plain ridge fits).

    src_feats/tgt_feats: {layer_idx: (N, n_heads*head_dim)} token-level,
    RoPE-stripped for keys. Returns {tgt_layer: [src_layers ranked]}.
    """
    choice = {}
    for lt, Y in tgt_feats.items():
        scored = []
        for ls, X in src_feats.items():
            W, b = fit_ridge(X, Y, lam)
            scored.append((r2_score(Y, apply_ridge(X, W, b)), ls))
        scored.sort(reverse=True)
        choice[lt] = [ls for _, ls in scored[:k]]
    return choice


# ----------------------------------------------------------- full K/V mapper

class KVMapper:
    """Per-(target layer, head, K|V) ridge maps, fit from calibration dumps.

    calib format (what collect_calib.py produces), per model:
      {"K": {layer: (N, n_kv, dh)}, "V": {layer: (N, n_kv, dh)},
       "positions": (N,), "rope_theta": float}
    Keys must be stored AS THE MODEL USES THEM (rotated); stripping happens here.
    """

    def __init__(self, k_sources: int = 4, lam: float = LAMBDA_DEFAULT):
        self.k_sources = k_sources
        self.lam = lam
        self.maps = {}      # {("K"|"V", tgt_layer, head): (W, b)}
        self.sources = {}   # {tgt_layer: [src_layers]}

    @staticmethod
    def _flat(feats, layer):
        a = feats[layer]                      # (N, n_kv, dh)
        return a.reshape(a.shape[0], -1)      # (N, n_kv*dh)

    def fit(self, src, tgt):
        thetas, thetat = src["rope_theta"], tgt["rope_theta"]
        pos_s, pos_t = src["positions"], tgt["positions"]

        def stripped(d, pos, theta):
            return {l: np.stack([strip_rope(d["K"][l][:, h, :], pos, theta)
                                 for h in range(d["K"][l].shape[1])], axis=1)
                    for l in d["K"]}

        ks_s, ks_t = stripped(src, pos_s, thetas), stripped(tgt, pos_t, thetat)
        # screening features: whole-layer flatten of stripped K + V averaged
        screen_src = {l: np.concatenate([self._flat(ks_s, l), self._flat(src["V"], l)], 1)
                      for l in ks_s}
        screen_tgt = {l: np.concatenate([self._flat(ks_t, l), self._flat(tgt["V"], l)], 1)
                      for l in ks_t}
        self.sources = select_topk_sources(screen_src, screen_tgt, self.k_sources, self.lam)

        for lt, srcs in self.sources.items():
            XK = np.concatenate([self._flat(ks_s, ls) for ls in srcs], axis=1)
            XV = np.concatenate([self._flat(src["V"], ls) for ls in srcs], axis=1)
            n_heads_t = tgt["K"][lt].shape[1]
            for h in range(n_heads_t):
                self.maps[("K", lt, h)] = fit_ridge(XK, ks_t[lt][:, h, :], self.lam)
                self.maps[("V", lt, h)] = fit_ridge(XV, tgt["V"][lt][:, h, :], self.lam)
        return self

    def transform(self, src_cache, positions, theta_src, theta_tgt):
        """src_cache: {"K": {l: (T, n_kv, dh)}, "V": ...} -> target-format cache."""
        ks_stripped = {l: np.stack([strip_rope(src_cache["K"][l][:, h, :], positions, theta_src)
                                    for h in range(src_cache["K"][l].shape[1])], 1)
                       for l in src_cache["K"]}
        out = {"K": {}, "V": {}}
        for lt, srcs in self.sources.items():
            XK = np.concatenate([self._flat(ks_stripped, ls) for ls in srcs], 1)
            XV = np.concatenate([self._flat(src_cache["V"], ls) for ls in srcs], 1)
            heads_k, heads_v = [], []
            h = 0
            while ("K", lt, h) in self.maps:
                Wk, bk = self.maps[("K", lt, h)]
                Wv, bv = self.maps[("V", lt, h)]
                heads_k.append(apply_rope(apply_ridge(XK, Wk, bk), positions, theta_tgt))
                heads_v.append(apply_ridge(XV, Wv, bv))
                h += 1
            out["K"][lt] = np.stack(heads_k, axis=1)
            out["V"][lt] = np.stack(heads_v, axis=1)
        return out
