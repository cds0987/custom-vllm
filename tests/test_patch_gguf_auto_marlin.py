#!/usr/bin/env python3
"""
Local (CPU-only, no real vLLM/GPU install) test suite for
scripts/patch_gguf_auto_marlin.py. Two independent halves, matching the
task's "nghiem thu local" requirement:

(A) test_patch_application(): downloads the real vllm-gguf-plugin sdist
    from PyPI (same one scripts/setup_env.sh builds from -- this needs
    network access, exactly like setup_env.sh itself), extracts it into a
    scratch copy, and runs patch_gguf_auto_marlin.py against that copy
    (via its --site-packages override) instead of a real site-packages
    install. Verifies the marker text lands in plugin.py and that running
    the patch a second time is a no-op (idempotent), same contract as
    every other scripts/patch_*.py in this repo.

(B) test_transcode_cache_logic(): the injected HELPERS code (the actual
    text patch_gguf_auto_marlin.py splices into plugin.py -- not a
    reimplementation of it) is exec()'d into a throwaway namespace and its
    entry point, _custom_vllm_maybe_auto_marlin(), is exercised directly
    against synthetic GGUF files built the same way
    scripts/test_gguf2marlin.py does (gguf.GGUFWriter +
    gguf.quants.quantize()) -- covers: disabled by default (env unset),
    cache MISS -> transcode -> cache HIT on a second call, and the
    K-quant refuse-by-default / ALLOW_K-opt-in policy. This calls the
    REAL scripts/gguf2marlin.py as a subprocess (via
    CUSTOM_VLLM_GGUF2MARLIN_PATH) -- no vLLM install, no GPU, no real
    model download, matching the task's "khong GPU, khong tai model that"
    constraint.

Run: python scripts/test_patch_gguf_auto_marlin.py
(Part A requires network access to fetch the sdist from PyPI; part B does
not. If the network is unavailable, part A is skipped with a clear notice
rather than failing the whole suite -- the SAME code path is verified
either way via part B's exec() of the identical injected text.)
"""

import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPTS_DIR.parent
PATCH_PATH = REPO_DIR / "models" / "qwen3_5" / "engine" / "vllm" / "patches" / "patch_gguf_auto_marlin.py"
GGUF2MARLIN_PATH = REPO_DIR / "models" / "qwen3_5" / "load" / "legacy_gguf2marlin.py"

spec = importlib.util.spec_from_file_location("patch_gguf_auto_marlin", PATCH_PATH)
pam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pam)

import gguf
from gguf.quants import quantize as gguf_quantize

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ==========================================================================
# Part A: apply the patch to a real (downloaded) sdist copy.
# ==========================================================================


def test_patch_application():
    print("\n-- patch application against a real vllm-gguf-plugin sdist --")
    tmp = Path(tempfile.mkdtemp(prefix="patch_auto_marlin_sdist_"))
    try:
        try:
            meta = json.loads(
                urllib.request.urlopen("https://pypi.org/pypi/vllm-gguf-plugin/json", timeout=15).read()
            )
            sdist_url = next(u["url"] for u in meta["urls"] if u["packagetype"] == "sdist")
            tar_path = tmp / "sdist.tar.gz"
            urllib.request.urlretrieve(sdist_url, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(tmp)
        except Exception as e:
            print(f"  SKIPPED (no network access to PyPI: {e})")
            return

        extracted = next(p for p in tmp.iterdir() if p.is_dir())
        plugin_py = extracted / "vllm_gguf_plugin" / "plugin.py"
        check("sdist contains vllm_gguf_plugin/plugin.py", plugin_py.is_file())
        if not plugin_py.is_file():
            return

        import subprocess

        r1 = subprocess.run(
            [sys.executable, str(PATCH_PATH), "--site-packages", str(extracted)],
            capture_output=True, text=True,
        )
        print(r1.stdout.strip())
        check("first apply exits 0", r1.returncode == 0, r1.stderr[-1000:])
        src_after_1 = plugin_py.read_text(encoding="utf-8")
        check("marker present after first apply", pam.PATCH_MARKER in src_after_1)
        check("helper function present after first apply",
              "_custom_vllm_maybe_auto_marlin" in src_after_1)

        r2 = subprocess.run(
            [sys.executable, str(PATCH_PATH), "--site-packages", str(extracted)],
            capture_output=True, text=True,
        )
        print(r2.stdout.strip())
        check("second apply exits 0 (idempotent)", r2.returncode == 0, r2.stderr[-1000:])
        check("second apply reports already-patched", "Already patched" in r2.stdout)
        src_after_2 = plugin_py.read_text(encoding="utf-8")
        check("file byte-identical after a second apply", src_after_1 == src_after_2)

        import ast
        try:
            ast.parse(src_after_2)
            check("patched plugin.py is syntactically valid Python", True)
        except SyntaxError as e:
            check("patched plugin.py is syntactically valid Python", False, str(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ==========================================================================
# Part B: exec() the injected HELPERS text and drive
# _custom_vllm_maybe_auto_marlin() directly against synthetic GGUF files.
# ==========================================================================


def _load_helpers_namespace():
    """exec()'s the EXACT text patch_gguf_auto_marlin.py would splice into
    plugin.py (via _build_helpers), baking CUSTOM_VLLM_GGUF2MARLIN_PATH's
    role through the real gguf2marlin.py on disk in this repo. Returns the
    namespace dict so tests can call its functions directly."""
    helpers_src = pam._build_helpers(str(GGUF2MARLIN_PATH))
    ns: dict = {"__name__": "patch_gguf_auto_marlin_helpers_under_test"}
    # `wraps`/`original_create_model_config` etc. from the surrounding
    # plugin.py module are not needed -- HELPERS is self-contained (only
    # imports stdlib + optionally vllm.logger, gracefully falling back to
    # print() when vllm isn't installed, exactly as in the real target file).
    exec(compile(helpers_src, "<injected-helpers>", "exec"), ns)
    return ns


def _build_toy_gguf(path: Path, *, kquant: bool):
    """A tiny checkpoint: all-Q4_0 (kquant=False, must auto-run) or with one
    Q4_K tensor mixed in (kquant=True, must be refused without ALLOW_K)."""
    rng = np.random.default_rng(5)
    hidden = 32
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(1)
    writer.add_tensor("token_embd.weight", rng.normal(0, 0.02, size=(8, hidden)).astype(np.float32))
    writer.add_tensor("blk.0.attn_norm.weight", np.ones((hidden,), dtype=np.float32))
    q_w = rng.normal(0, 0.05, size=(hidden, 64)).astype(np.float32)
    writer.add_tensor("blk.0.attn_q.weight", gguf_quantize(q_w, gguf.GGMLQuantizationType.Q4_0),
                       raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    if kquant:
        from test_gguf2marlin import encode_q4k_naive
        k_w = rng.normal(0, 0.05, size=(hidden, 256)).astype(np.float32)
        writer.add_tensor("blk.0.attn_k.weight", encode_q4k_naive(k_w), raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.add_tensor("output_norm.weight", np.ones((hidden,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_disabled_by_default():
    print("\n-- CUSTOM_VLLM_GGUF_AUTO_MARLIN unset -> always passes through --")
    tmp = Path(tempfile.mkdtemp(prefix="patch_auto_marlin_off_"))
    try:
        gguf_path = tmp / "toy.gguf"
        _build_toy_gguf(gguf_path, kquant=False)
        ns = _load_helpers_namespace()
        import os
        env_backup = dict(os.environ)
        try:
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN", None)
            result = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("disabled by default returns None (stock passthrough)", result is None)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_sibling_hf_config(gguf_path: Path) -> Path:
    """Bug C fixture helper: a real caller only gets past auto-marlin's
    config.json check by placing one next to the .gguf file (same
    convention vllm_gguf_plugin's own stock GGUF path already requires --
    see patch_gguf_auto_marlin.py's module docstring, "HF CONFIG"). Content
    just needs a real (non-GGUF-internal) `model_type` for
    scripts/gguf2marlin.py's --hf-config overlay + sanity check."""
    cfg_path = gguf_path.with_name("config.json")
    cfg_path.write_text(json.dumps({"model_type": "qwen3_5"}))
    return cfg_path


def test_safe_file_auto_transcodes_and_caches():
    print("\n-- all-Q4_0 file: auto-transcodes, then cache-hits on rerun --")
    tmp = Path(tempfile.mkdtemp(prefix="patch_auto_marlin_safe_"))
    try:
        gguf_path = tmp / "toy_safe.gguf"
        _build_toy_gguf(gguf_path, kquant=False)
        _write_sibling_hf_config(gguf_path)
        cache_root = tmp / "cache"
        ns = _load_helpers_namespace()
        import os
        env_backup = dict(os.environ)
        try:
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN"] = "1"
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            os.environ["CUSTOM_VLLM_MARLIN_CACHE_DIR"] = str(cache_root)
            os.environ["CUSTOM_VLLM_GGUF2MARLIN_PATH"] = str(GGUF2MARLIN_PATH)

            result1 = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("cache-miss run returns a directory path", result1 is not None, str(result1))
            if result1 is None:
                return
            out_dir = Path(result1)
            check("returned dir has model.safetensors", (out_dir / "model.safetensors").is_file())
            check("returned dir has config.json", (out_dir / "config.json").is_file())
            check("returned dir has manifest.json", (out_dir / "manifest.json").is_file())
            manifest = json.loads((out_dir / "manifest.json").read_text())
            check("manifest used int4 (no K-quants in this file)", manifest.get("k_quants_to") == "int4")
            check("returned dir is under the overridden cache root", str(cache_root) in str(out_dir))

            mtime_before = (out_dir / "model.safetensors").stat().st_mtime

            result2 = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("second call (cache hit) returns the SAME directory", result2 == result1, f"{result2} vs {result1}")
            mtime_after = (out_dir / "model.safetensors").stat().st_mtime
            check("cache hit did not re-run the transcoder (mtime unchanged)", mtime_before == mtime_after)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _toy_hf_config(model_type="qwen3_5", **extra):
    """A minimal but REAL-shaped upstream config.json body -- distinct from
    gguf2marlin.py's own generic fallback (which sets model_type to the raw
    GGUF architecture string, e.g. "qwen35") and from the GGUF-parser-only
    alias that string collides with. Used to prove Bug C's fix: when this
    is found and passed through --hf-config, the checkpoint's model_type
    is THIS value, not the "qwen35" fallback."""
    cfg = {"model_type": model_type, "architectures": ["Qwen3_5ForCausalLM"]}
    cfg.update(extra)
    return cfg


def test_sibling_hf_config_used_and_missing_config_refuses():
    print("\n-- Bug C: real config.json next to the .gguf is found and used; "
          "missing config.json refuses transcode --")
    tmp = Path(tempfile.mkdtemp(prefix="patch_auto_marlin_hfconfig_"))
    try:
        import os

        # ---- (1) no config.json anywhere -> refuse, fall back to stock ---
        gguf_path_missing = tmp / "missing_cfg" / "toy.gguf"
        gguf_path_missing.parent.mkdir(parents=True, exist_ok=True)
        _build_toy_gguf(gguf_path_missing, kquant=False)
        ns = _load_helpers_namespace()
        env_backup = dict(os.environ)
        try:
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN"] = "1"
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_HF_CONFIG", None)
            os.environ["CUSTOM_VLLM_MARLIN_CACHE_DIR"] = str(tmp / "cache_missing")
            os.environ["CUSTOM_VLLM_GGUF2MARLIN_PATH"] = str(GGUF2MARLIN_PATH)

            result = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path_missing))
            check("missing config.json -> refuses transcode (returns None)", result is None, str(result))
            check("nothing written to the cache dir on refusal",
                  not ns["_custom_vllm_am_cache_dir"](
                      ns["_custom_vllm_am_hash_file"](str(gguf_path_missing))
                  ).exists())
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

        # ---- (2) sibling config.json -> found and passed to gguf2marlin ---
        gguf_path_sibling = tmp / "sibling_cfg" / "toy.gguf"
        gguf_path_sibling.parent.mkdir(parents=True, exist_ok=True)
        _build_toy_gguf(gguf_path_sibling, kquant=False)
        (gguf_path_sibling.parent / "config.json").write_text(
            json.dumps(_toy_hf_config()), encoding="utf-8"
        )
        ns = _load_helpers_namespace()
        env_backup = dict(os.environ)
        try:
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN"] = "1"
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_HF_CONFIG", None)
            os.environ["CUSTOM_VLLM_MARLIN_CACHE_DIR"] = str(tmp / "cache_sibling")
            os.environ["CUSTOM_VLLM_GGUF2MARLIN_PATH"] = str(GGUF2MARLIN_PATH)

            result = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path_sibling))
            check("sibling config.json -> transcode succeeds", result is not None, str(result))
            if result is None:
                return
            out_cfg = json.loads((Path(result) / "config.json").read_text())
            check("output config.json model_type == real config's (qwen3_5)",
                  out_cfg.get("model_type") == "qwen3_5", str(out_cfg.get("model_type")))
            check("output config.json model_type is NOT the gguf-only fallback ('qwen35')",
                  out_cfg.get("model_type") != "qwen35", str(out_cfg.get("model_type")))
            check("output config.json carries architectures from the real config",
                  out_cfg.get("architectures") == ["Qwen3_5ForCausalLM"], str(out_cfg.get("architectures")))
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

        # ---- (3) CUSTOM_VLLM_GGUF_AUTO_MARLIN_HF_CONFIG env override -------
        gguf_path_env = tmp / "env_cfg" / "toy.gguf"
        gguf_path_env.parent.mkdir(parents=True, exist_ok=True)
        _build_toy_gguf(gguf_path_env, kquant=False)
        # deliberately NOT placing config.json next to the .gguf -- only the
        # env override should supply it.
        override_dir = tmp / "override_cfg_dir"
        override_dir.mkdir(parents=True, exist_ok=True)
        (override_dir / "config.json").write_text(
            json.dumps(_toy_hf_config(model_type="qwen3_5_moe")), encoding="utf-8"
        )
        ns = _load_helpers_namespace()
        env_backup = dict(os.environ)
        try:
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN"] = "1"
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            # override is a DIRECTORY containing config.json, not the file itself.
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN_HF_CONFIG"] = str(override_dir)
            os.environ["CUSTOM_VLLM_MARLIN_CACHE_DIR"] = str(tmp / "cache_env")
            os.environ["CUSTOM_VLLM_GGUF2MARLIN_PATH"] = str(GGUF2MARLIN_PATH)

            result = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path_env))
            check("env-override config.json -> transcode succeeds", result is not None, str(result))
            if result is None:
                return
            out_cfg = json.loads((Path(result) / "config.json").read_text())
            check("output config.json model_type == env-overridden config's (qwen3_5_moe)",
                  out_cfg.get("model_type") == "qwen3_5_moe", str(out_cfg.get("model_type")))
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kquant_refused_by_default_and_allowed_with_flag():
    print("\n-- Q4_K present: refused by default, allowed with ALLOW_K=1 --")
    tmp = Path(tempfile.mkdtemp(prefix="patch_auto_marlin_kquant_"))
    try:
        gguf_path = tmp / "toy_kquant.gguf"
        _build_toy_gguf(gguf_path, kquant=True)
        _write_sibling_hf_config(gguf_path)
        cache_root = tmp / "cache"
        ns = _load_helpers_namespace()
        import os
        env_backup = dict(os.environ)
        try:
            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN"] = "1"
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            os.environ["CUSTOM_VLLM_MARLIN_CACHE_DIR"] = str(cache_root)
            os.environ["CUSTOM_VLLM_GGUF2MARLIN_PATH"] = str(GGUF2MARLIN_PATH)

            refused = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("Q4_K file refused without ALLOW_K (returns None)", refused is None)
            check("nothing was written to the cache dir on refusal",
                  not ns["_custom_vllm_am_cache_dir"](ns["_custom_vllm_am_hash_file"](str(gguf_path))).exists())

            os.environ["CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K"] = "1"
            allowed = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("Q4_K file transcodes with ALLOW_K=1", allowed is not None, str(allowed))
            if allowed is None:
                return
            out_dir = Path(allowed)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            check("manifest used int8 (K-quant branch)", manifest.get("k_quants_to") == "int8")
            check("manifest has at least one int8 module", len(manifest.get("int8_modules", [])) >= 1)

            # Now unset ALLOW_K again: cache exists but must be REFUSED, not
            # silently reused -- see module docstring "consent check runs on
            # EVERY invocation".
            os.environ.pop("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K", None)
            re_refused = ns["_custom_vllm_maybe_auto_marlin"](str(gguf_path))
            check("cached int8 result is NOT silently reused once ALLOW_K is unset again",
                  re_refused is None, str(re_refused))
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_patch_application()
    test_disabled_by_default()
    test_safe_file_auto_transcodes_and_caches()
    test_sibling_hf_config_used_and_missing_config_refuses()
    test_kquant_refused_by_default_and_allowed_with_flag()

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
    sys.path.insert(0, str(SCRIPTS_DIR))  # for `from test_gguf2marlin import encode_q4k_naive`
    sys.exit(main())
