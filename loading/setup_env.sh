#!/usr/bin/env bash
# Install vllm + vllm-gguf-plugin and apply every patch, from scratch.
# Split out of serve_test.sh so a recycled Colab runtime can be rebuilt in one
# command, and so serving options can be varied without redoing the setup.
set -e

# `uv pip` resolves and installs in parallel and reuses a global wheel cache;
# on this workload it consistently beats pip's serial download+install on the
# vllm dependency tree (torch alone is ~2.5 GB). Costs ~15s to bootstrap and
# falls back to pip if anything about it misbehaves, so there is no downside.
PIP="pip install -q"
if pip install -q uv 2>/dev/null && command -v uv >/dev/null 2>&1; then
  PIP="uv pip install --system -q"
  echo "=== Using uv for installs (parallel resolver + shared wheel cache) ==="
else
  echo "=== uv unavailable; falling back to pip ==="
fi

echo "=== Installing vllm (GHIM 0.27.1 — toan bo so do Phase C/production tren ban nay;"
echo "    2026-08-26 runtime moi keo 0.28.0/torch 2.13 khong ghim -> drift ngam) ==="
$PIP "vllm==${VLLM_VERSION:-0.27.1}" requests

# vLLM 0.27.1 (2026-08-12) pulls a torchaudio built against a different CUDA
# minor than the torch it installs (torch cu13.0 vs torchaudio cu12.8).
# transformers/loss/loss_rnnt.py hard-imports torchaudio, so `import
# transformers` dies before anything else runs. torchaudio is not needed for
# text-only serving -- drop it.
# BUT: removing it also drops torchvision, and vllm's qwen3_5.py imports the
# multimodal branch (Qwen2VLImageProcessor) even for text-only use, so
# torchvision must be put back WITHOUT letting pip drag torchaudio along.
echo "=== Fixing torchaudio/torchvision CUDA-build mismatch (see STATUS.md drift 2026-08-12) ==="
pip uninstall -q -y torchaudio torchvision || true
$PIP --no-deps torchvision || echo "WARNING: torchvision reinstall failed; qwen3_5.py import may break"

# hf_transfer: Rust-backed parallel chunk downloader. The frame (8 GB) + GGUF
# (5.8 GB) dominate rebuild wall-clock and huggingface_hub's default single
# stream leaves most of Colab's bandwidth on the table. Costs seconds to
# install, saves minutes on every download.
echo "=== Installing hf_transfer (parallel HF downloads) ==="
$PIP hf_transfer || echo "WARNING: hf_transfer install failed; downloads stay single-stream"
export HF_HUB_ENABLE_HF_TRANSFER=1

# llmcompressor + datasets are only needed by the quantize_*.py tools and by
# eval_quality_swebench.py's lazy `from datasets import load_dataset`. They are
# a multi-minute install that also drags huggingface_hub backwards (see the
# re-pin below), so a serve/bench-only session should skip them entirely.
# Opt in with CUSTOM_VLLM_TOOLS=1.
if [ "${CUSTOM_VLLM_TOOLS:-0}" = "1" ]; then
  echo "=== Installing llmcompressor + datasets (CUSTOM_VLLM_TOOLS=1) ==="
  $PIP llmcompressor datasets || echo "WARNING: llmcompressor/datasets install failed (non-fatal for serving-only sessions)"
else
  echo "=== Skipping llmcompressor + datasets (set CUSTOM_VLLM_TOOLS=1 if you need quantize_*.py or eval_quality_swebench.py) ==="
fi
# datasets pins an older huggingface_hub (observed: downgraded to 1.23.0,
# which predates the `ResolvedRevision` symbol) and silently overwrites the
# newer huggingface_hub vllm/vllm_gguf_plugin need -- vllm_gguf_plugin's
# loader.py does `from huggingface_hub import ResolvedRevision, ...` at
# import time, so any server load of ANY model (not just GGUF) then dies
# with ImportError before it even reaches gguf-specific code. Force the
# newer huggingface_hub back after datasets has had its say; harmless if
# datasets didn't touch it.
$PIP -U huggingface_hub || echo "WARNING: huggingface_hub re-pin failed"

echo "=== Installing vllm-gguf-plugin (GGUF moved out-of-tree as of vllm 0.26) ==="
# once for the dependencies (gguf, ...), then package-only so the in-place
# patches below always apply to pristine sources
$PIP vllm-gguf-plugin

# The prebuilt wheel's _C_gguf.abi3.so was, on some torch builds, compiled
# against a different torch ABI (ImportError: undefined symbol
# torch_exception_get_what_without_backtrace on torch 2.11.0+cu128), silently
# dropping every GGUF matmul to the Triton fallback -- no GEMV path, 3.9x cost
# at conc1 (TEST 8). Rebuilding from sdist fixes it, but that compile is the
# single most expensive step in this script (minutes of nvcc), and as of
# vllm 0.27.1 the published wheel imports fine. So: TEST FIRST, BUILD ONLY IF
# BROKEN. Observed 2026-08-12 on a fresh Colab runtime: the sdist build failed,
# fell back to the wheel, and the wheel's _C_gguf imported cleanly anyway --
# i.e. the whole compile was pure waste.
echo "=== Checking whether the prebuilt wheel's _C_gguf matches this torch ABI ==="
if python -c "from vllm_gguf_plugin import _C_gguf" 2>/dev/null; then
  echo "_C_gguf import OK from prebuilt wheel — skipping sdist rebuild (saves minutes)"
  SKIP_SDIST=1
else
  echo "_C_gguf broken in prebuilt wheel — rebuilding from sdist against this torch"
  SKIP_SDIST=0
fi

if [ "$SKIP_SDIST" = "0" ]; then
SDIST_URL=$(python - <<'EOF'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/vllm-gguf-plugin/json"))
print(next(u["url"] for u in d["urls"] if u["packagetype"] == "sdist"))
EOF
)
SDIST_OK=0
curl -sL "$SDIST_URL" -o /tmp/vllm_gguf_plugin_sdist.tar.gz && SDIST_OK=1
if [ "$SDIST_OK" = "1" ] && TORCH_CUDA_ARCH_LIST="8.9" pip install -q --no-build-isolation --force-reinstall --no-deps /tmp/vllm_gguf_plugin_sdist.tar.gz; then
  echo "Built plugin from sdist (sm_89)"
else
  echo "WARNING: sdist build failed; falling back to prebuilt wheel (Triton-only kernels)"
  pip install -q --force-reinstall --no-deps vllm-gguf-plugin
fi
fi   # SKIP_SDIST
python - <<'EOF'
try:
    from vllm_gguf_plugin import _C_gguf  # noqa: F401
    print("_C_gguf import OK — CUDA kernels active")
except ImportError as e:
    print(f"_C_gguf NOT available ({e}); Triton fallback in use")
EOF

# Patches live NEXT TO the thing they patch (see models/_template/MANIFEST.md):
# engine-specific ones under models/qwen3_5/engine/vllm/patches/, engine-neutral
# ones (transformers) under models/qwen3_5/utils/. Run engine-neutral first —
# vLLM's config parsing goes through transformers.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES_VLLM="$REPO_DIR/models/qwen3_5/engine/vllm/patches"
PATCHES_NEUTRAL="$REPO_DIR/models/qwen3_5/utils"

echo "=== patch_transformers_qwen35 (engine-neutral) ==="
python "$PATCHES_NEUTRAL/patch_transformers_qwen35.py"

for s in \
  patch_gguf_plugin \
  patch_gguf_tensor_mapping \
  patch_vllm_qwen35_embed \
  patch_gguf_weight_type_loader \
  patch_gguf_conv1d_shape \
  patch_vllm_qwen35_registry \
  patch_gguf_drop_mrope \
  patch_vllm_qwen35_hybrid \
  patch_gguf_empty_suffix \
  patch_gguf_qwen35_transforms \
  patch_gguf_prefer_dequant \
  patch_gguf_hybrid_dispatch \
  patch_fla_ada_shmem \
  patch_gguf_override_signature \
  patch_gguf_repack_q6k \
  patch_gguf_dequant_buffer \
  patch_gguf_threeway_dispatch \
  patch_vllm_gdn_quant_load \
  patch_gguf_auto_marlin
do
  echo "=== $s ==="
  python "$PATCHES_VLLM/$s.py"
done

echo "=== Ensuring CUDA runtime libs are on the loader path ==="
CUDART_PATH=$(find / -name "libcudart.so.13*" 2>/dev/null | head -n1)
if [ -n "$CUDART_PATH" ]; then
  echo "export LD_LIBRARY_PATH=$(dirname "$CUDART_PATH"):\$LD_LIBRARY_PATH" > /tmp/vllm_env.sh
  echo "Found $CUDART_PATH — source /tmp/vllm_env.sh before running vllm"
else
  echo "WARNING: libcudart.so.13 not found"
fi

echo "=== Setup complete ==="
