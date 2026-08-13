"""CPU unit tests for models/qwen3_5/kv_transfer/ridge_mapper.py.

No GPU, no model: synthetic data with a KNOWN linear relationship, so the
mapper must recover it (paper's premise, in miniature).

Run: python tests/test_kv_transfer_mapper.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MOD = REPO / "models" / "qwen3_5" / "kv_transfer" / "ridge_mapper.py"
spec = importlib.util.spec_from_file_location("ridge_mapper", MOD)
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


class TestRidge(unittest.TestCase):
    def test_recovers_linear_map(self):
        rng = np.random.default_rng(0)
        W_true = rng.normal(size=(32, 16))
        b_true = rng.normal(size=16)
        X = rng.normal(size=(2000, 32))
        Y = X @ W_true + b_true + rng.normal(scale=1e-3, size=(2000, 16))
        W, b = rm.fit_ridge(X, Y)
        self.assertLess(np.abs(W - W_true).max(), 1e-2)
        self.assertLess(np.abs(b - b_true).max(), 1e-2)
        self.assertGreater(rm.r2_score(Y, rm.apply_ridge(X, W, b)), 0.999)

    def test_rope_strip_roundtrip(self):
        rng = np.random.default_rng(1)
        k = rng.normal(size=(50, 256))
        pos = np.arange(50)
        back = rm.apply_rope(rm.strip_rope(k, pos, 1e6), pos, 1e6)
        self.assertLess(np.abs(k - back).max(), 1e-9)

    def test_rope_strip_makes_fit_position_free(self):
        # same content at different positions: stripped keys identical
        rng = np.random.default_rng(2)
        content = rng.normal(size=(1, 128))
        k_pos5 = rm.apply_rope(content, np.array([5]), 1e6)
        k_pos99 = rm.apply_rope(content, np.array([99]), 1e6)
        s5 = rm.strip_rope(k_pos5, np.array([5]), 1e6)
        s99 = rm.strip_rope(k_pos99, np.array([99]), 1e6)
        self.assertLess(np.abs(s5 - s99).max(), 1e-9)


class TestKVMapper(unittest.TestCase):
    def _make_pair(self, rng, n=800, src_layers=4, tgt_layers=6, n_kv=2, dh=32):
        # ground truth: each target layer is a linear function of 2 source layers
        theta = 1e6
        pos = np.arange(n) % 512
        src = {"K": {}, "V": {}, "positions": pos, "rope_theta": theta}
        tgt = {"K": {}, "V": {}, "positions": pos, "rope_theta": theta}
        base = {l: rng.normal(size=(n, n_kv, dh)) for l in range(src_layers)}
        baseV = {l: rng.normal(size=(n, n_kv, dh)) for l in range(src_layers)}
        for l in range(src_layers):
            src["K"][l] = np.stack([rm.apply_rope(base[l][:, h, :], pos, theta)
                                    for h in range(n_kv)], 1)
            src["V"][l] = baseV[l]
        self.mixes = {}
        for lt in range(tgt_layers):
            l1, l2 = lt % src_layers, (lt + 1) % src_layers
            Wk = rng.normal(size=(2 * n_kv * dh, n_kv * dh)) / np.sqrt(2 * n_kv * dh)
            Wv = rng.normal(size=(2 * n_kv * dh, n_kv * dh)) / np.sqrt(2 * n_kv * dh)
            XK = np.concatenate([base[l1].reshape(n, -1), base[l2].reshape(n, -1)], 1)
            XV = np.concatenate([baseV[l1].reshape(n, -1), baseV[l2].reshape(n, -1)], 1)
            k_content = (XK @ Wk).reshape(n, n_kv, dh)
            tgt["K"][lt] = np.stack([rm.apply_rope(k_content[:, h, :], pos, theta)
                                     for h in range(n_kv)], 1)
            tgt["V"][lt] = (XV @ Wv).reshape(n, n_kv, dh)
        return src, tgt

    def test_fit_and_transform_high_r2(self):
        rng = np.random.default_rng(3)
        src, tgt = self._make_pair(rng)
        mapper = rm.KVMapper(k_sources=2).fit(src, tgt)
        out = mapper.transform({"K": src["K"], "V": src["V"]},
                               src["positions"], 1e6, 1e6)
        for lt in tgt["K"]:
            r2v = rm.r2_score(tgt["V"][lt].reshape(len(tgt["V"][lt]), -1),
                              out["V"][lt].reshape(len(out["V"][lt]), -1))
            self.assertGreater(r2v, 0.95, f"V layer {lt} r2={r2v}")
            k_true_s = np.stack([rm.strip_rope(tgt["K"][lt][:, h, :], tgt["positions"], 1e6)
                                 for h in range(tgt["K"][lt].shape[1])], 1)
            k_pred_s = np.stack([rm.strip_rope(out["K"][lt][:, h, :], tgt["positions"], 1e6)
                                 for h in range(out["K"][lt].shape[1])], 1)
            r2k = rm.r2_score(k_true_s.reshape(len(k_true_s), -1),
                              k_pred_s.reshape(len(k_pred_s), -1))
            self.assertGreater(r2k, 0.95, f"K layer {lt} r2={r2k}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
