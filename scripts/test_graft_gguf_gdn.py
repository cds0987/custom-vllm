#!/usr/bin/env python3
"""
Local (CPU-only, no real model/GPU) test suite for scripts/graft_gguf_gdn.py.

Builds a small synthetic "champion" checkpoint (safetensors + a minimal but
schema-correct compressed-tensors config.json, two layers -- one GDN, one
full-attention) and a small synthetic GGUF (Q4_K-encoded GDN tensors, tiled
the way llama.cpp actually stores Qwen3.5's V-head-carrying GDN tensors --
see scripts/patch_gguf_qwen35_transforms.py), then runs the real CLI
end-to-end and checks its output. The Q4_K encoder is reused from
scripts/test_gguf2marlin.py (`encode_q4k_naive`) via importlib, same as that
file's own approach: an independent-of-gguf2marlin encoder, not code
under test authoring its own fixtures from its own internals.

Covers (per the task's checklist):
  (a) int8 relative RMS error < 1e-2 for every grafted tensor.
  (b) config.json comes out correct: a new int8 config_group targeting
      exactly the grafted modules, and the `ignore` list narrowed so the
      grafted modules are no longer covered while unrelated GDN submodules
      (conv1d, norm) and other layers stay covered.
  (c) every non-grafted tensor (other layer's weights, lm_head, the GDN
      layer's conv1d/norm) is copied bit-for-bit identical.
  (d) a merge-pair scheme mismatch (one of a pair grafted, the other not,
      because its GGUF tensor is missing) fails loudly with a clear message
      -- both via a direct unit test of `validate_merge_pairs` and via a
      real CLI run.
  (e) running the CLI twice on the same inputs produces byte-identical
      output (deterministic).

Run: python scripts/test_graft_gguf_gdn.py
Exits 0 and prints "ALL TESTS PASSED" on success, exits 1 otherwise.
"""

import filecmp
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
GRAFT_PATH = SCRIPTS_DIR / "graft_gguf_gdn.py"

spec = importlib.util.spec_from_file_location("graft_gguf_gdn", GRAFT_PATH)
graft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graft)

spec2 = importlib.util.spec_from_file_location("test_gguf2marlin", SCRIPTS_DIR / "test_gguf2marlin.py")
t_g2m = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(t_g2m)  # reuse encode_q4k_naive; also runs that file's own module-level nothing (guarded by __main__)

import gguf
from safetensors.torch import save_file
from safetensors import safe_open

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ==========================================================================
# Independent forward "tile" transform (HF grouped order -> llama.cpp's
# on-disk tiled order) -- the mathematical inverse of graft_gguf_gdn's
# untile_v_heads, but written from scratch here (different reshape axis
# order), not by calling the module under test.
# ==========================================================================


def tile_v_heads_test(t: np.ndarray, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> np.ndarray:
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    t2 = t.reshape(new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(t2.transpose(perm)).reshape(shape)


def test_tile_untile_are_inverses():
    print("\n-- tile_v_heads_test / untile_v_heads round-trip (sanity for the fixture builder) --")
    rng = np.random.default_rng(0)
    num_k, num_v_per_k, head_dim = 2, 3, 4
    w = rng.normal(size=(num_k * num_v_per_k * head_dim, 8)).astype(np.float32)
    tiled = tile_v_heads_test(w, 0, num_k, num_v_per_k, head_dim)
    back = graft.untile_v_heads(tiled, 0, num_k, num_v_per_k, head_dim)
    check("tile then untile recovers the original", np.allclose(w, back), f"max diff={np.abs(w-back).max()}")


# ==========================================================================
# Fixture builders.
# ==========================================================================

HIDDEN = 256  # in_features -- must be a multiple of 256 (Q4_K superblock size)
NUM_K, NUM_V = 2, 4  # linear_num_key_heads / linear_num_value_heads
HEAD_K, HEAD_V = 4, 4  # linear_key_head_dim / linear_value_head_dim
NUM_V_PER_K = NUM_V // NUM_K  # 2 -- reorder=True since NUM_K != NUM_V
QK_DIM = 2 * NUM_K * HEAD_K  # in_proj_qkv's Q+K rows (untouched by tiling)
V_DIM = NUM_V * HEAD_V  # in_proj_qkv's V rows / in_proj_z's full output


def build_true_gdn_weights(rng):
    """HF-orientation (out_features, in_features) ground-truth weights for
    the four GDN in_proj_* modules of one layer, as they'd sit in the
    champion's own (fp16, ignore-listed) checkpoint."""
    return {
        "in_proj_qkv": rng.normal(0, 0.05, size=(QK_DIM + V_DIM, HIDDEN)).astype(np.float32),
        "in_proj_z": rng.normal(0, 0.05, size=(V_DIM, HIDDEN)).astype(np.float32),
        "in_proj_b": rng.normal(0, 0.05, size=(NUM_V, HIDDEN)).astype(np.float32),
        "in_proj_a": rng.normal(0, 0.05, size=(NUM_V, HIDDEN)).astype(np.float32),
    }


def tile_for_gguf(suffix, w):
    if suffix == "in_proj_qkv":
        return np.concatenate([w[:QK_DIM], tile_v_heads_test(w[QK_DIM:], 0, NUM_K, NUM_V_PER_K, HEAD_V)], axis=0)
    if suffix == "in_proj_z":
        return tile_v_heads_test(w, 0, NUM_K, NUM_V_PER_K, HEAD_V)
    return tile_v_heads_test(w, 0, NUM_K, NUM_V_PER_K, 1)  # in_proj_b / in_proj_a


def build_test_frame(frame_dir: Path, true_weights: dict, *, omit_layer1_gdn=False) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for suffix, w in true_weights.items():
        tensors[f"model.layers.0.linear_attn.{suffix}.weight"] = torch.from_numpy(w).to(torch.float16)
    tensors["model.layers.0.linear_attn.conv1d.weight"] = torch.zeros(V_DIM + QK_DIM, 1, 4, dtype=torch.float16)
    tensors["model.layers.0.linear_attn.norm.weight"] = torch.ones(V_DIM, dtype=torch.float16)
    tensors["model.layers.1.self_attn.q_proj.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16)
    tensors["model.layers.1.self_attn.k_proj.weight"] = torch.randn(HIDDEN, HIDDEN, dtype=torch.float16)
    tensors["lm_head.weight"] = torch.randn(37, HIDDEN, dtype=torch.float16)
    save_file(tensors, frame_dir / "model.safetensors", metadata={"format": "pt"})

    cfg = {
        "model_type": "qwen35",
        "num_hidden_layers": 2,
        "linear_num_key_heads": NUM_K,
        "linear_num_value_heads": NUM_V,
        "linear_key_head_dim": HEAD_K,
        "linear_value_head_dim": HEAD_V,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_default": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "group_size": 32,
                        "strategy": "group",
                        "symmetric": True,
                        "type": "int",
                        "dynamic": False,
                    },
                }
            },
            "ignore": ["lm_head", "re:.*linear_attn.*"],
        },
    }
    (frame_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (frame_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def build_test_gguf(gguf_path: Path, true_weights: dict, *, omit_suffix=None) -> None:
    writer = gguf.GGUFWriter(str(gguf_path), "qwen35")
    writer.add_block_count(1)
    for suffix, w in true_weights.items():
        if suffix == omit_suffix:
            continue
        ggml_suffix = graft.GGML_SUFFIX_FOR_HF_SUFFIX[suffix]
        tiled = tile_for_gguf(suffix, w)
        writer.add_tensor(f"blk.0.{ggml_suffix}", t_g2m.encode_q4k_naive(tiled), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ==========================================================================
# (d) direct unit test of validate_merge_pairs / narrow_ignore_list.
# ==========================================================================


def test_validate_merge_pairs_unit():
    print("\n-- validate_merge_pairs: direct unit test --")
    ok = {(0, "in_proj_qkv"): True, (0, "in_proj_z"): True, (0, "in_proj_b"): False, (0, "in_proj_a"): False}
    try:
        graft.validate_merge_pairs(ok)
        check("matched pair (both True) and matched pair (both False) do not raise", True)
    except graft.GraftConfigError:
        check("matched pair (both True) and matched pair (both False) do not raise", False)

    mismatched = {(0, "in_proj_qkv"): True, (0, "in_proj_z"): False, (0, "in_proj_b"): True, (0, "in_proj_a"): True}
    try:
        graft.validate_merge_pairs(mismatched)
        check("mismatched qkv/z pair raises GraftConfigError", False, "no exception raised")
    except graft.GraftConfigError as e:
        check("mismatched qkv/z pair raises GraftConfigError", True)
        check("error message names the mismatched pair", "in_proj_qkv" in str(e) or "qkv" in str(e), str(e))


def test_narrow_ignore_list_unit():
    print("\n-- narrow_ignore_list: direct unit test --")
    ignore = ["lm_head", "re:.*linear_attn.*"]
    grafted = [
        "model.layers.0.linear_attn.in_proj_qkv",
        "model.layers.0.linear_attn.in_proj_z",
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
    ]
    probes = [
        "lm_head",
        "model.layers.0.linear_attn.conv1d",
        "model.layers.0.linear_attn.norm",
        "model.layers.1.linear_attn.in_proj_qkv",  # different layer, must stay ignored
    ]
    new_ignore = graft.narrow_ignore_list(ignore, grafted, probes)
    check("lm_head entry untouched", "lm_head" in new_ignore, str(new_ignore))
    for name in grafted:
        matched = any(graft._is_equal_or_regex_match(name, e) for e in new_ignore)
        check(f"grafted module no longer ignored: {name}", not matched, str(new_ignore))
    for name in probes:
        matched = any(graft._is_equal_or_regex_match(name, e) for e in new_ignore)
        check(f"probe still ignored: {name}", matched, str(new_ignore))

    # A literal (non-regex) ignore entry that exactly matches a grafted name
    # should be dropped outright.
    lit_ignore = ["model.layers.0.linear_attn.in_proj_qkv", "lm_head"]
    new_lit = graft.narrow_ignore_list(lit_ignore, ["model.layers.0.linear_attn.in_proj_qkv"], ["lm_head"])
    check("literal exact-match entry dropped", new_lit == ["lm_head"], str(new_lit))


# ==========================================================================
# End-to-end CLI tests.
# ==========================================================================


def run_cli(frame, gguf_path, out_dir, group_size=32):
    return subprocess.run(
        [sys.executable, str(GRAFT_PATH), "--frame", str(frame), "--gguf", str(gguf_path),
         "--out", str(out_dir), "--group-size", str(group_size)],
        capture_output=True, text=True,
    )


def test_end_to_end():
    print("\n-- end-to-end CLI run --")
    tmp = Path(tempfile.mkdtemp(prefix="graft_gdn_test_"))
    try:
        rng = np.random.default_rng(42)
        true_weights = build_true_gdn_weights(rng)
        frame_dir = tmp / "frame"
        gguf_path = tmp / "src.gguf"
        out_dir = tmp / "out"
        build_test_frame(frame_dir, true_weights)
        build_test_gguf(gguf_path, true_weights)

        result = run_cli(frame_dir, gguf_path, out_dir)
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        check("CLI exits 0", result.returncode == 0, f"returncode={result.returncode}")
        if result.returncode != 0:
            return out_dir, frame_dir, true_weights

        # (a) RMS error.
        manifest = json.loads((out_dir / "graft_manifest.json").read_text())
        check("manifest lists all 4 grafted modules", len(manifest["grafted_modules"]) == 4, str(manifest["grafted_modules"]))
        for entry in manifest["tensors"]:
            check(f"rel_rms_error < 1e-2 for {entry['module']}", entry["rel_rms_error"] < 1e-2, str(entry))

        # (b) config.json.
        cfg = json.loads((out_dir / "config.json").read_text())
        qc = cfg["quantization_config"]
        new_group = qc["config_groups"].get("graft_gdn_int8")
        check("new config_group present", new_group is not None, str(qc["config_groups"].keys()))
        if new_group is not None:
            check("new group targets exactly the 4 grafted modules",
                  set(new_group["targets"]) == {f"model.layers.0.linear_attn.{s}" for s in graft.GDN_SUFFIXES},
                  str(new_group["targets"]))
            check("new group num_bits == 8", new_group["weights"]["num_bits"] == 8, str(new_group["weights"]))
            check("new group group_size == 32", new_group["weights"]["group_size"] == 32, str(new_group["weights"]))
        for suffix in graft.GDN_SUFFIXES:
            name = f"model.layers.0.linear_attn.{suffix}"
            matched = any(graft._is_equal_or_regex_match(name, e) for e in qc["ignore"])
            check(f"{name} no longer in ignore", not matched, str(qc["ignore"]))
        for probe in ("model.layers.0.linear_attn.conv1d", "model.layers.0.linear_attn.norm", "lm_head"):
            matched = any(graft._is_equal_or_regex_match(probe, e) for e in qc["ignore"])
            check(f"{probe} still in ignore", matched, str(qc["ignore"]))

        with safe_open(out_dir / "model.safetensors", framework="pt") as f:
            keys = set(f.keys())
            base = "model.layers.0.linear_attn.in_proj_qkv"
            check(f"{base}.weight_packed present", base + ".weight_packed" in keys)
            check(f"{base}.weight absent (replaced)", base + ".weight" not in keys)
            wp = f.get_tensor(base + ".weight_packed")
            ws = f.get_tensor(base + ".weight_scale")
            wsh = f.get_tensor(base + ".weight_shape")
            n, k = QK_DIM + V_DIM, HIDDEN
            check("weight_packed shape (N, K/4)", tuple(wp.shape) == (n, k // 4), str(wp.shape))
            check("weight_scale shape (N, K/group_size)", tuple(ws.shape) == (n, k // 32), str(ws.shape))
            check("weight_shape == [N, K]", wsh.tolist() == [n, k], str(wsh.tolist()))

            # (a, deeper) grafted weight, once dequantized, recovers the true
            # pre-tile HF weight (not just "low error vs. its own tiled
            # reference") -- this is the real test that untile_module's
            # V-head inversion is correct, not merely that quantization
            # noise is small.
            k_dim = HIDDEN
            n_dim = QK_DIM + V_DIM
            deq = graft.g2m.dequant_gptq_for_verification(wp.numpy().T, ws.numpy().T, k_dim, n_dim, 32, bits=8)
            rel_rms_vs_true = float(np.sqrt(((deq - true_weights["in_proj_qkv"].T) ** 2).mean())) / \
                float(np.sqrt((true_weights["in_proj_qkv"].T ** 2).mean()))
            # Bound is loose on purpose: this compares against the TRUE
            # pre-Q4_K float, so it also carries the naive Q4_K encoder's own
            # ~5-15% RTN quantization noise (see gguf2marlin.py's DECISION 2
            # CORRECTION note) on top of the ~0.5% int8-regraft noise already
            # checked above -- 0.25 is well below where a real row-permutation
            # bug would land (rows would be essentially decorrelated, driving
            # this towards sqrt(2)~1.4x the raw signal RMS), so this still
            # catches an untile mistake while tolerating expected Q4_K noise.
            check("grafted in_proj_qkv dequant matches TRUE pre-tile HF weight (untile correctness)",
                  rel_rms_vs_true < 0.25, f"rel_rms_vs_true={rel_rms_vs_true}")

            # Discriminating power check: if the script's untile were simply
            # absent/wrong, the grafted output would sit much closer to the
            # STILL-TILED reference than to the true (untiled) weight -- so
            # the tiled-vs-true distance must be clearly larger than the
            # measured grafted-vs-true distance, confirming this test would
            # actually catch a broken/missing untile step.
            tiled_true = tile_for_gguf("in_proj_qkv", true_weights["in_proj_qkv"])
            rel_rms_vs_tiled = float(np.sqrt(((deq - tiled_true.T) ** 2).mean())) / \
                float(np.sqrt((tiled_true.T ** 2).mean()))
            check("grafted output is far closer to the untiled truth than to the still-tiled truth",
                  rel_rms_vs_true < 0.5 * rel_rms_vs_tiled,
                  f"rel_rms_vs_true={rel_rms_vs_true}, rel_rms_vs_tiled={rel_rms_vs_tiled}")

            # (c) non-grafted tensors copied bit-identical.
            with safe_open(frame_dir / "model.safetensors", framework="pt") as f_in:
                for key in ("model.layers.0.linear_attn.conv1d.weight", "model.layers.0.linear_attn.norm.weight",
                            "model.layers.1.self_attn.q_proj.weight", "lm_head.weight"):
                    check(f"{key} copied bit-identical", torch.equal(f.get_tensor(key), f_in.get_tensor(key)))

        return out_dir, frame_dir, true_weights
    finally:
        pass  # caller (test_determinism) reuses tmp inputs; cleaned there


def test_determinism(tmp_frame_gguf):
    frame_dir, gguf_path = tmp_frame_gguf
    print("\n-- determinism: two runs produce identical output --")
    out1 = frame_dir.parent / "out1"
    out2 = frame_dir.parent / "out2"
    r1 = run_cli(frame_dir, gguf_path, out1)
    r2 = run_cli(frame_dir, gguf_path, out2)
    check("both runs exit 0", r1.returncode == 0 and r2.returncode == 0)
    if r1.returncode != 0 or r2.returncode != 0:
        return
    check("config.json identical", (out1 / "config.json").read_text() == (out2 / "config.json").read_text())
    check("model.safetensors identical", filecmp.cmp(out1 / "model.safetensors", out2 / "model.safetensors", shallow=False))


def test_mismatched_pair_fails():
    print("\n-- end-to-end CLI run with a missing GGUF tensor (mismatched merge pair) --")
    tmp = Path(tempfile.mkdtemp(prefix="graft_gdn_mismatch_"))
    try:
        rng = np.random.default_rng(7)
        true_weights = build_true_gdn_weights(rng)
        frame_dir = tmp / "frame"
        gguf_path = tmp / "src.gguf"
        out_dir = tmp / "out"
        build_test_frame(frame_dir, true_weights)
        # Omit in_proj_z from the GGUF -- in_proj_qkv will graft, its pair
        # partner in_proj_z cannot (no source tensor) -> mismatched pair.
        build_test_gguf(gguf_path, true_weights, omit_suffix="in_proj_z")

        result = run_cli(frame_dir, gguf_path, out_dir)
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        check("CLI exits non-zero on mismatched merge pair", result.returncode != 0, f"returncode={result.returncode}")
        check("stderr names the mismatch clearly",
              "merge pair" in result.stderr.lower() or "mismatch" in result.stderr.lower(),
              result.stderr[-1000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_tile_untile_are_inverses()
    test_validate_merge_pairs_unit()
    test_narrow_ignore_list_unit()

    out_dir, frame_dir, true_weights = test_end_to_end()
    test_determinism((frame_dir, frame_dir.parent / "src.gguf"))
    test_mismatched_pair_fails()

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed:")
        for name in FAILURES:
            print(f"    - {name}")
        print("=" * 72)
        return 1
    print("ALL TESTS PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
