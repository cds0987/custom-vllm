#!/usr/bin/env python3
"""
"Auto Marlin": `vllm serve <local .gguf file>` transparently transcodes to a
GPTQ-Marlin checkpoint (via scripts/gguf2marlin.py) and caches it, so the
*Marlin* CUDA kernel serves the model instead of vllm-gguf-plugin's own
GGUF CUDA/Triton kernels -- with zero flag changes at the `vllm serve` call
site. Off by default (CUSTOM_VLLM_GGUF_AUTO_MARLIN unset => byte-identical
stock plugin behaviour); every other patch script in this repo, and every
`vllm serve` invocation that doesn't set the env var, is completely
unaffected by this file even being imported/patched.

HOOK POINT -- chosen by reading the REAL vllm-gguf-plugin source, not
guessed
-----------------------------------------------------------------------
Downloaded and extracted the vllm-gguf-plugin sdist from PyPI
(vllm_gguf_plugin-0.0.4.tar.gz, matching the wheel scripts/setup_env.sh
installs) into a scratch dir and read every file that touches "what is the
model string / when does the plugin first see it":

    vllm_gguf_plugin/plugin.py          <- chosen anchor, see below
    vllm_gguf_plugin/__init__.py        (just re-exports plugin.register)
    vllm_gguf_plugin/config_parser.py   (GGUFConfigParser.parse)
    vllm_gguf_plugin/loader.py          (GGUFModelLoader._prepare_weights)
    vllm_gguf_plugin/gguf_utils.py      (check_gguf_file / is_remote_gguf / ...)

`vllm_gguf_plugin/plugin.py`'s `_patch_engine_args()` monkeypatches
`EngineArgs.create_model_config` (vllm/engine/arg_utils.py) with a wrapper
that runs BEFORE vLLM builds anything: before `ModelConfig` exists, before
`GGUFConfigParser.parse()` reads any HF config, before
`GGUFModelLoader._prepare_weights()`/`download_gguf()`/`hf_hub_download()`
resolve a remote reference to a local file, and before the weight loader
class is even selected (`register_model_loader("gguf")(GGUFModelLoader)`
just registers a name -> class mapping at `register()` time; the loader
object plugin.py builds is not instantiated until engine-core startup,
long after `create_model_config` returns). Concretely, the wrapper's first
line inside the `if _is_gguf_reference(self.model):` branch is

    gguf_model = self.model

-- this is the EARLIEST point in the whole request path that (a) has the
raw model string exactly as the user typed it to `vllm serve <path>` and
(b) runs before any weight-loader machinery is constructed. This is why
the ANCHOR below targets that exact function body: it is the "chỗ plugin
nhận model path/config" the task asks for, verified by reading every
candidate site in the real sdist rather than assumed from the package name
alone. (`GGUFConfigParser.parse` and `GGUFModelLoader._prepare_weights`
both run LATER and duplicate/derive from what `create_model_config` already
decided -- patching either of those would be strictly later and would have
to re-parse the same model string a second time for no benefit.)

Once `_custom_vllm_maybe_auto_marlin()` (injected below, appended to the
patched `plugin.py`) succeeds, `self.model`/`self.model_weights` are
swapped to the transcoded checkpoint directory and `self.quantization` /
`self.load_format` / `self.config_format` are reset to "auto" (rather than
the plugin's own "gguf") -- from that point on this is fully indistinguishable
from `vllm serve <plain HF GPTQ checkpoint dir>` to the rest of vLLM: normal
HFConfigParser, normal safetensors weight loader, normal AutoGPTQConfig ->
Marlin kernel selection (auto-detected from config.json's
quantization_config, exactly as scripts/gguf2marlin.py's own docstring
documents for a manually-run transcode). No vLLM or plugin source beyond
this one wrapper function is touched.

QUALITY POLICY -- "khong bao gio am tham chuyen co mat mat" (never silently
switch to something lossy)
-----------------------------------------------------------------------
Before transcoding, the GGUF file's tensor types are scanned (cheap: only
`gguf.GGUFReader` tensor-type metadata, no dequant). Only a checkpoint made
ENTIRELY of {Q4_0, F16, F32} tensors runs automatically -- Q4_0 is
scripts/gguf2marlin.py's byte-exact int4 fast path (rel. RMS error == 0.0,
see that script's DECISION 2), F16/F32 are never quantized at all. Any
OTHER GGML type present (Q4_1, any K-quant, Q8_0, IQ*, ...) means the
checkpoint would need scripts/gguf2marlin.py's lossy generic re-fit path
(~5-15% RMS for the default int4 branch, ~0.5% for --k-quants-to int8 --
see that script's DECISION 1b/2) for at least one tensor, and auto-marlin
REFUSES by default: it logs a message naming the offending GGML types and
pointing at `CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K=1` (which switches the
transcoder to `--k-quants-to int8`, printing the same "not ppl-gate-tested,
~0.5% error" caveat every time it runs, cache hit or miss -- see below),
then returns None so `create_model_config` falls through to the STOCK
`_is_gguf_reference` handling: the model still serves, just via the
unmodified GGUF plugin path (whatever CUDA/Triton kernel that resolves to),
not a silently-degraded Marlin checkpoint. Refusing to transcode is a
deliberately SAFE fallback, not a hard error -- it costs performance
(no Marlin speedup), never correctness.

The consent check runs on EVERY invocation, cache hit or miss: a manifest
whose `int8_modules` list is non-empty (i.e. was produced with
`--k-quants-to int8`, which only ever happens when ALLOW_K was set at
transcode time) is treated as requiring ALLOW_K on every subsequent serve
too, not just the one that created it -- otherwise a user could set
ALLOW_K=1 once, get a cached lossy checkpoint, unset ALLOW_K, and have a
LATER `vllm serve` of the same file silently reuse that lossy cache with no
warning at all. Re-checking the manifest's own recorded scheme on cache hit
closes that gap.

MTP / unmapped tensors: scripts/gguf2marlin.py's own generic architecture
mapper already passes through any tensor it cannot resolve to a modern HF
name (this repo's Qwen3.5 MTP head is one such case) as
`unmapped.<ggml_name>` fp16 -- see that script's DECISION 3/LIMITATIONS.
This hook does not special-case MTP tensors further; it just surfaces
`manifest.json`'s `unmapped` and `fp16_kept` lists in its own log output
after a successful transcode ("log ro" -- log clearly), so an operator
sees exactly which tensors did not make it into the Marlin/GPTQ path
without having to open manifest.json by hand.

CACHE
-----
`~/.cache/custom_vllm/marlin/<sha256(gguf file bytes)>/` (override the root
via CUSTOM_VLLM_MARLIN_CACHE_DIR). Contains exactly what
scripts/gguf2marlin.py writes for any manual run: model.safetensors,
config.json, manifest.json (per-tensor target scheme + rel. RMS error --
see that script's "manifest.json" section). A cache hit is
`config.json`+`model.safetensors`+`manifest.json` all present AND (per the
policy above) the consent check re-passing.

TRANSCODER SCRIPT LOCATION -- an installed-package hook cannot assume this
private repo is on PYTHONPATH at `vllm serve` time (it may not even be
checked out on the serving box). Resolved, in order: (1)
`CUSTOM_VLLM_GGUF2MARLIN_PATH` env var (explicit override -- what this
patch script's own local test suite uses, pointing at a scratch copy); (2)
the absolute path this patch script itself resolves `gguf2marlin.py` to at
PATCH-APPLY time (baked into the injected code as a literal string -- valid
whenever setup_env.sh's own repo checkout is still present on the serving
box, which is the normal case for this project's Colab runbook); (3) a
handful of common relative locations (`scripts/gguf2marlin.py` under the
current working directory, and under this file's own package-adjacent
paths). If none resolve to an existing file, auto-marlin logs an error and
refuses (same safe fallback as the policy refusal above) -- it never
crashes `vllm serve` itself.

REMOTE / MULTI-FILE GGUF REFERENCES -- LIMITATION
--------------------------------------------------
Only two model-string forms are handled: (a) an existing local `.gguf`
file, and (b) a plain `repo_id/filename.gguf` HF Hub reference (resolved
with `huggingface_hub.hf_hub_download`, the same call
`GGUFModelLoader._prepare_weights` would make later -- doing it here too is
safe and cheap because `hf_hub_download` is itself locally cached).
`repo_id:quant_type` and `local_dir:quant_type` references (vllm-gguf-
plugin's own multi-file quant-type-selector syntax, `gguf_utils.py`'s
`is_remote_gguf`/`is_local_gguf_quant`) are NOT resolved here -- doing so
correctly requires the plugin's own repo-file-listing/glob logic
(`weight_utils.py: download_gguf`/`resolve_local_gguf`), and a repo can
legitimately contain several same-named tensors across shard files, which
this hook's single-file hashing model does not account for. Auto-marlin
logs "unsupported reference form, passing through" and returns None for
these -- same safe stock fallback, not a crash.

Must be listed LAST in setup_env.sh's patch loop (after
patch_vllm_gdn_quant_load): it has no anchor/ordering dependency on any
other patch (it touches a different file, vllm_gguf_plugin/plugin.py,
that no other patch in this repo modifies), so "last" is purely a
convention -- new hooks are appended, not inserted, so re-running
setup_env.sh after adding one never reorders anything already applied.
"""

import glob
import sysconfig
from pathlib import Path

PATCH_MARKER = "# --- custom_vllm: gguf auto-marlin transcode hook ---"

ANCHOR = '''    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
        return original_create_model_config(self, *args, **kwargs)
'''

PATCH = '''    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            ''' + PATCH_MARKER + '''
            _marlin_dir = _custom_vllm_maybe_auto_marlin(gguf_model)
            if _marlin_dir is not None:
                # Transcoded checkpoint is a plain HF-style GPTQ directory
                # (model.safetensors + config.json with quantization_config
                # -- see scripts/gguf2marlin.py). Route it through vLLM's
                # STOCK (non-gguf) config/load path so AutoGPTQConfig picks
                # the Marlin kernel by normal auto-detection, exactly like
                # `vllm serve <plain gptq checkpoint dir>`.
                self.model = _marlin_dir
                self.model_weights = _marlin_dir
                if self.quantization in (None, "gguf"):
                    self.quantization = None
                if self.load_format in ("auto", "gguf"):
                    self.load_format = "auto"
                if self.config_format in ("auto", "gguf"):
                    self.config_format = "auto"
                if self.served_model_name is None:
                    self.served_model_name = [gguf_model]
                return original_create_model_config(self, *args, **kwargs)
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
        return original_create_model_config(self, *args, **kwargs)
'''


def _build_helpers(baked_gguf2marlin_path: str) -> str:
    """Renders the injected module-level helper code. Factored into a
    function (rather than a bare f-string at import time) so this patch
    script's own local test suite can `exec()` the SAME text into a throwaway
    namespace and unit-test the cache/policy logic directly, without a real
    vllm_gguf_plugin install -- see scripts/test_patch_gguf_auto_marlin.py.
    `baked_gguf2marlin_path` is resolved once, at patch-apply time (see
    bottom of this file), from THIS repo checkout -- see module docstring,
    "TRANSCODER SCRIPT LOCATION".
    """
    return f'''

{PATCH_MARKER}
import hashlib as _custom_vllm_am_hashlib
import json as _custom_vllm_am_json
import os as _custom_vllm_am_os
import subprocess as _custom_vllm_am_subprocess
import sys as _custom_vllm_am_sys
from pathlib import Path as _CustomVllmAmPath

# Types scripts/gguf2marlin.py transcodes losslessly / never lossily: Q4_0
# is its byte-exact int4 fast path (rel. RMS == 0.0); F16/F32 are never
# quantized. Anything else present in the file means at least one tensor
# would need the lossy generic re-fit path -- see module docstring,
# "QUALITY POLICY".
_CUSTOM_VLLM_AM_SAFE_TYPE_NAMES = frozenset({{"Q4_0", "F16", "F32"}})

_CUSTOM_VLLM_AM_BAKED_GGUF2MARLIN_PATH = {baked_gguf2marlin_path!r}


def _custom_vllm_am_enabled():
    return _custom_vllm_am_os.environ.get("CUSTOM_VLLM_GGUF_AUTO_MARLIN") == "1"


def _custom_vllm_am_allow_k():
    return _custom_vllm_am_os.environ.get("CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K") == "1"


def _custom_vllm_am_log(msg):
    try:
        from vllm.logger import init_logger
        init_logger("custom_vllm.gguf_auto_marlin").info(msg)
    except Exception:
        print(f"[gguf-auto-marlin] {{msg}}")


def _custom_vllm_am_warn(msg):
    try:
        from vllm.logger import init_logger
        init_logger("custom_vllm.gguf_auto_marlin").warning(msg)
    except Exception:
        print(f"[gguf-auto-marlin] WARNING: {{msg}}")


def _custom_vllm_am_resolve_local_file(model_ref):
    """model string -> local .gguf file path, or None (+ log) if this hook
    does not know how to resolve that reference form. See module docstring,
    "REMOTE / MULTI-FILE GGUF REFERENCES".
    """
    if _custom_vllm_am_os.path.isfile(model_ref) and str(model_ref).endswith(".gguf"):
        return str(model_ref)
    if ":" in model_ref:
        _custom_vllm_am_log(
            f"auto-marlin: {{model_ref!r}} is a repo:quant_type / "
            "local_dir:quant_type reference (multi-file quant-type "
            "selector) -- not resolved by this hook, passing through to "
            "the stock GGUF loader. See patch_gguf_auto_marlin.py's "
            "module docstring."
        )
        return None
    if "/" in model_ref and model_ref.endswith(".gguf"):
        try:
            from huggingface_hub import hf_hub_download
            repo_id, filename = model_ref.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            _custom_vllm_am_warn(
                f"auto-marlin: failed to resolve/download {{model_ref!r}} "
                f"from the HF Hub ({{e}}); passing through to the stock "
                "GGUF loader."
            )
            return None
    return None


def _custom_vllm_am_hash_file(path):
    h = _custom_vllm_am_hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _custom_vllm_am_cache_dir(file_hash):
    root = _custom_vllm_am_os.environ.get("CUSTOM_VLLM_MARLIN_CACHE_DIR")
    base = _CustomVllmAmPath(root) if root else _CustomVllmAmPath.home() / ".cache" / "custom_vllm" / "marlin"
    return base / file_hash


def _custom_vllm_am_scan_gguf_types(path):
    """Set of GGML type NAMES present in the file's tensors -- cheap header
    scan only (gguf.GGUFReader mmaps; no dequant)."""
    import gguf
    reader = gguf.GGUFReader(path)
    return {{gguf.GGMLQuantizationType(t.tensor_type).name for t in reader.tensors}}


def _custom_vllm_am_find_transcoder_script():
    """See module docstring, "TRANSCODER SCRIPT LOCATION"."""
    env_override = _custom_vllm_am_os.environ.get("CUSTOM_VLLM_GGUF2MARLIN_PATH")
    candidates = [
        env_override,
        _CUSTOM_VLLM_AM_BAKED_GGUF2MARLIN_PATH,
        "scripts/gguf2marlin.py",
        str(_CustomVllmAmPath.cwd() / "custom_vllm" / "scripts" / "gguf2marlin.py"),
    ]
    for c in candidates:
        if c and _custom_vllm_am_os.path.isfile(c):
            return c
    return None


def _custom_vllm_am_log_manifest_summary(manifest):
    quantized = manifest.get("quantized_modules", [])
    int8_mods = manifest.get("int8_modules", [])
    fp16_kept = manifest.get("fp16_kept", [])
    unmapped = manifest.get("unmapped", [])
    tensors = manifest.get("tensors", [])
    worst = sorted(tensors, key=lambda e: -e.get("rel_rms_error", 0.0))[:3]
    _custom_vllm_am_log(
        f"auto-marlin: {{len(quantized)}} modules quantized "
        f"({{len(int8_mods)}} int8, {{len(quantized) - len(int8_mods)}} int4), "
        f"{{len(fp16_kept)}} kept fp16, {{len(unmapped)}} unmapped. "
        f"worst rel. RMS error: "
        + ", ".join(f"{{e['module']}}={{e['rel_rms_error']:.4f}}" for e in worst)
    )
    if fp16_kept:
        _custom_vllm_am_log(f"auto-marlin: fp16-passthrough modules: {{fp16_kept[:20]}}"
                             + (" ..." if len(fp16_kept) > 20 else ""))
    if unmapped:
        _custom_vllm_am_log(
            f"auto-marlin: UNMAPPED tensors (no generic HF name resolved, "
            f"e.g. MTP heads on hybrid architectures -- kept fp16 under "
            f"'unmapped.*', will NOT be visible to vLLM's stock loader "
            f"under any real parameter name): {{unmapped[:20]}}"
            + (" ..." if len(unmapped) > 20 else "")
        )
    if int8_mods:
        _custom_vllm_am_warn(
            "auto-marlin: this checkpoint used the --k-quants-to int8 "
            "branch (K-quant tensors re-encoded to GPTQ int8) -- NOT "
            "verified against a real perplexity gate; measured relative "
            "RMS error vs. the source GGUF's own dequant is typically "
            "~0.5% (see scripts/gguf2marlin.py's DECISION 1b / "
            "scripts/test_gguf2marlin.py), not a guarantee on downstream "
            "quality."
        )


def _custom_vllm_am_refuse(unsafe_types):
    _custom_vllm_am_warn(
        "auto-marlin: this GGUF contains tensor type(s) "
        f"{{sorted(unsafe_types)}} outside the always-safe set "
        f"{{sorted(_CUSTOM_VLLM_AM_SAFE_TYPE_NAMES)}} (Q4_0 is byte-exact, "
        "F16/F32 are never quantized). Auto-transcoding a K-quant/other "
        "lossy type to Marlin by default would silently trade accuracy "
        "for speed, which this project's campaign policy forbids. "
        "REFUSING to auto-transcode; serving via the stock GGUF plugin "
        "path instead (no Marlin speedup, but no silent accuracy loss "
        "either). To opt in to the lossy int8 K-quant branch anyway "
        "(scripts/gguf2marlin.py --k-quants-to int8, measured ~0.5% "
        "relative RMS error, NOT verified against a real perplexity "
        "gate), set CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K=1."
    )


def _custom_vllm_am_transcode(local_path, out_dir, k_quants_to):
    script = _custom_vllm_am_find_transcoder_script()
    if script is None:
        _custom_vllm_am_warn(
            "auto-marlin: could not locate scripts/gguf2marlin.py (checked "
            "CUSTOM_VLLM_GGUF2MARLIN_PATH, the path baked in at patch time, "
            "and common relative locations -- see patch_gguf_auto_marlin.py "
            "module docstring). Passing through to the stock GGUF loader."
        )
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _custom_vllm_am_sys.executable, script, local_path, str(out_dir),
        "--group-size", "32", "--k-quants-to", k_quants_to,
    ]
    _custom_vllm_am_log(f"auto-marlin: cache miss -- transcoding via {{' '.join(cmd)}}")
    result = _custom_vllm_am_subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _custom_vllm_am_warn(
            f"auto-marlin: transcode failed (exit {{result.returncode}}); "
            f"passing through to the stock GGUF loader. stderr tail:\\n"
            + "\\n".join(result.stderr.strip().splitlines()[-20:])
        )
        return False
    return True


def _custom_vllm_am_ready(cache_dir):
    return (
        (cache_dir / "model.safetensors").is_file()
        and (cache_dir / "config.json").is_file()
        and (cache_dir / "manifest.json").is_file()
    )


def _custom_vllm_maybe_auto_marlin(model_ref):
    """Entry point called from create_model_config. Returns a local
    directory path to serve as a plain (non-GGUF) GPTQ checkpoint, or None
    to fall through to stock GGUF handling. Never raises -- every failure
    mode logs and returns None (see module docstring, "QUALITY POLICY").
    """
    if not _custom_vllm_am_enabled():
        return None

    local_path = _custom_vllm_am_resolve_local_file(str(model_ref))
    if local_path is None:
        return None

    try:
        file_hash = _custom_vllm_am_hash_file(local_path)
    except OSError as e:
        _custom_vllm_am_warn(f"auto-marlin: failed to hash {{local_path!r}} ({{e}}); passing through.")
        return None
    cache_dir = _custom_vllm_am_cache_dir(file_hash)

    allow_k = _custom_vllm_am_allow_k()

    if _custom_vllm_am_ready(cache_dir):
        try:
            manifest = _custom_vllm_am_json.loads((cache_dir / "manifest.json").read_text())
        except Exception:
            manifest = {{}}
        # Re-check consent EVERY run, even on a cache hit -- see module
        # docstring, "The consent check runs on EVERY invocation".
        if manifest.get("int8_modules") and not allow_k:
            _custom_vllm_am_warn(
                "auto-marlin: cached checkpoint for this file used the "
                "int8 K-quant branch (requires "
                "CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K=1), which is not set "
                "for this run. Refusing to reuse it silently; passing "
                "through to the stock GGUF loader. Set "
                "CUSTOM_VLLM_GGUF_AUTO_MARLIN_ALLOW_K=1 to reuse the cache."
            )
            return None
        _custom_vllm_am_log(f"auto-marlin: cache hit -- {{cache_dir}}")
        _custom_vllm_am_log_manifest_summary(manifest)
        return str(cache_dir)

    try:
        types_present = _custom_vllm_am_scan_gguf_types(local_path)
    except Exception as e:
        _custom_vllm_am_warn(f"auto-marlin: failed to scan {{local_path!r}} ({{e}}); passing through.")
        return None
    unsafe_types = types_present - _CUSTOM_VLLM_AM_SAFE_TYPE_NAMES

    if unsafe_types and not allow_k:
        _custom_vllm_am_refuse(unsafe_types)
        return None

    k_quants_to = "int8" if unsafe_types else "int4"
    if not _custom_vllm_am_transcode(local_path, cache_dir, k_quants_to):
        return None

    try:
        manifest = _custom_vllm_am_json.loads((cache_dir / "manifest.json").read_text())
    except Exception:
        manifest = {{}}
    _custom_vllm_am_log(f"auto-marlin: transcode complete -- {{cache_dir}}")
    _custom_vllm_am_log_manifest_summary(manifest)
    return str(cache_dir)
'''


def patch(path, marker, anchor, replacement, *, append=None):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if marker in src:
        print(f"Already patched: {path}")
        return
    if anchor not in src:
        raise SystemExit(
            f"Anchor not found in {path}; vllm-gguf-plugin's plugin.py may "
            "have changed (this patch was written against vllm_gguf_plugin "
            "0.0.4's create_model_config wrapper -- see this file's module "
            "docstring, 'HOOK POINT')."
        )
    src = src.replace(anchor, replacement, 1)
    if append:
        src += append
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")


def find_target_path(site_packages_override=None):
    site_packages = site_packages_override or sysconfig.get_paths()["purelib"]
    matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/plugin.py")
    if not matches:
        raise SystemExit(
            f"vllm_gguf_plugin/plugin.py not found under {site_packages}; "
            "is vllm-gguf-plugin installed?"
        )
    return matches[0]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--site-packages", default=None,
        help="override the site-packages root to patch under (for local "
             "testing against an extracted sdist copy instead of a real "
             "install -- see scripts/test_patch_gguf_auto_marlin.py)",
    )
    args = ap.parse_args()

    target = find_target_path(args.site_packages)
    baked_path = str((Path(__file__).resolve().parent / "gguf2marlin.py"))
    patch(target, PATCH_MARKER, ANCHOR, PATCH, append=_build_helpers(baked_path))
