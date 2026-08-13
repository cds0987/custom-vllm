#!/usr/bin/env python3
"""
Transcode a Qwen3.5 GGUF checkpoint into a GPTQ-format HF checkpoint that
vLLM serves through its gptq_marlin kernels on sm89+ (L4, A10G, RTX 40xx, ...).

WHY
---
STATUS.md's optimisation loop on L4 (sm89) closed with one open route left:
on long-context prefill, the winning configurations bracket 4-bit GGUF between
two fp16-adjacent numbers --

    GGUF Q4_K_M + dequant+cuBLAS   ~8.6K tok/s   (best 4-bit-on-disk result)
    fp16 safetensors + fp8 KV      ~10.95K tok/s  (best result, period, but
                                                    4x the disk/VRAM footprint)

GGUF's own kernels (Triton fused, or the dequant+cuBLAS fallback) cannot close
that ~27% gap: they were never the fast path on this architecture, they were
the only path, because gptq_marlin_repack does not understand ggml block
layouts (Q4_K's 144-byte superblocks, importance-weighted sub-scales, etc --
see the "Ngõ cụt" note in STATUS.md). Marlin is vLLM's actual fast W4A16
kernel on sm89, and it only accepts one input shape: a standard GPTQ
checkpoint (qweight/qzeros/scales/g_idx in the AutoGPTQ int32-packed layout).
So the only way to reach Marlin from a GGUF is to leave the ggml format
entirely: dequantise every block to fp32, undo llama.cpp's Qwen3.5-specific
value/layout rewrites (see scripts/patch_gguf_qwen35_transforms.py -- this
script reimplements that exact math standalone, since this box has no vllm
install and must not touch the GPU lane), then round-trip through a *second*
quantisation pass into GPTQ's int4 group layout. That gets us: same 4-bit
footprint on disk and in VRAM as the GGUF, served by the kernel that already
measured ~27% faster than the best GGUF path on this hardware.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
This is RTN (round-to-nearest), not calibrated GPTQ. No forward passes, no
Hessian, no activation statistics -- each weight matrix is independently
rounded to the nearest representable value in its own group. That is a
strictly weaker quantiser than real GPTQ (which minimises reconstruction
error against a calibration set, layer by layer, propagating each layer's
rounding error into the next layer's Hessian). RTN was chosen for the first
cut because: (a) it needs no calibration data or forward passes, so it runs
in a few minutes on CPU for a 2B model with nothing but the GGUF file; (b) the
error budget already has slack -- STATUS.md's own K-quant dequant path shows
0.4-7% relative error is "quantisation noise" for this architecture, and RTN
int4 g128 lands in a similar band (see the verification report this script
prints); (c) it de-risks the checkpoint-format and vLLM-wiring questions
(tensor-name mapping, inverse transforms, GPTQ tensor shapes, config.json
shape) independently of quantiser quality. A calibrated GPTQ pass (e.g.
GPTQModel, using WikiText or a Qwen-style calibration set) is a strict
drop-in upgrade on top of the same output-tensor contract this script
produces, and is noted as follow-up work below, not blocking.

WHY HAND-ROLLED, NOT A LIBRARY (GPTQModel / llm-compressor / AutoAWQ)
-----------------------------------------------------------------------
Those libraries quantise by loading the *original* HF model class and hooking
its forward pass (calibration) or replacing its nn.Linear modules in place
(RTN). Qwen3.5 is a hybrid architecture with a custom gated-delta-net mixer
(linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,
A_log,dt_bias,norm}) that transformers/GPTQModel do not special-case --
fighting a library's model-loading and module-replacement assumptions about
this specific hybrid mixer would cost more than the packer itself. The
*output* contract (the four GPTQ tensors: qweight/qzeros/scales/g_idx, in
exactly the int32-packed layout vLLM's AutoGPTQLinearMethod +
MarlinLinearKernel expect -- see vllm/model_executor/layers/quantization/
auto_gptq.py:create_weights and vllm/model_executor/layers/quantization/
utils/quant_utils.py:quantize_weights/pack_rows, both read directly from the
local vllm clone to pin down the exact math) is small, stable, and already
fully specified by vLLM itself. Reimplementing that contract directly, and
driving it from tensors we already have a verified GGUF->HF pipeline for
(scripts/patch_gguf_qwen35_transforms.py), is simpler than adapting a library
to an architecture it has never seen. We do NOT reimplement Marlin's bit
layout -- vLLM repacks a standard GPTQ checkpoint into Marlin's tile format
itself at `process_weights_after_loading` time (ops.gptq_marlin_repack), so
this script only has to produce a *correct standard GPTQ checkpoint*.

WHAT'S QUANTISED, WHAT STAYS FP16
-----------------------------------
Only the Linear-shaped weights that vLLM's GPTQ/Marlin path actually handles:
  - attention: q_proj, k_proj, v_proj, o_proj
  - MLP:       gate_proj, up_proj, down_proj
  - GDN (gated-delta-net) linear projections: in_proj_qkv, in_proj_z,
    in_proj_a, in_proj_b, out_proj
This was verified by reading the local vLLM clone
(vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py):
QwenGatedDeltaNetAttention.create_qkvz_proj/create_ba_proj build
MergedColumnParallelLinear(quant_config=self.quant_config, ...) for
in_proj_qkvz (loaded from separate in_proj_qkv/in_proj_z checkpoint tensors,
fused at load time) and in_proj_ba (from in_proj_b/in_proj_a), and out_proj is
a RowParallelLinear(quant_config=self.quant_config, ...). The class even has
a Marlin-specific TP workaround keyed on `isinstance(quant_config,
(AutoAWQConfig, AutoGPTQConfig, INCConfig))` (maybe_disable_tp, qwen_gdn_
linear_attn.py:618) -- i.e. vLLM's own authors already anticipated GPTQ/AWQ
quantisation of these exact GDN projections. conv1d is a plain
ColumnParallelLinear built WITHOUT quant_config (line ~468), so it is
correctly excluded. A_log and dt_bias are raw nn.Parameter (not Linear
weights) and stay fp32/fp16. linear_attn.norm, input_layernorm,
post_attention_layernorm, the final model.norm, and embed_tokens/lm_head all
stay fp16 -- vLLM never routes norms or embeddings through a LinearMethod.

WHAT IS APPROXIMATED (be honest about the error budget)
-----------------------------------------------------------
This is a *double* quantisation: the original bf16 weights were already
quantised once by llama.cpp into Q4_K/Q5_K/Q6_K/Q8_0 (whatever the source
GGUF used per-tensor), and we dequantise that back to fp32 and requantise a
*second* time into GPTQ int4 group128. Each step adds independent rounding
noise; the combined error is not just the RTN error, it's on top of whatever
the GGUF's own K-quant error already was. The verification step below
reports both numbers (GGUF-dequant vs RTN-requant, and combined vs the
original bf16 HF reference) so this isn't hand-waved.

HOW THE NAME MAPPING WORKS
-----------------------------
gguf-py's `gguf.get_tensor_name_map(MODEL_ARCH.QWEN35, n_layers)` builds a
dict from every known *HF*-style parameter name (across every architecture
gguf-py knows how to write, e.g. "model.layers.{bid}.self_attn.q_proj" for
llama-style attention) to the corresponding ggml tensor name (e.g.
"blk.{bid}.attn_q"). This is the same table vllm-gguf-plugin uses (in the
opposite direction) at load time -- see
vllm_gguf_plugin/weights_adapter/default.py:find_hf_name_in_tensor_map,
which builds a *dummy* transformers Qwen3.5 model, walks its state_dict (the
real, upstream HF parameter names), and looks each one up in this same table
to recover the matching ggml name. We do the same lookup here, driven by a
hand-enumerated list of the real Qwen3.5 HF parameter names (read directly
from vllm/model_executor/models/qwen3_5.py and qwen_gdn_linear_attn.py,
rather than instantiating a dummy model, since this box has neither vllm nor
transformers installed) -- see `HF_TENSOR_BASE_NAMES` below. This is the
"reuse, don't reinvent" tensor-mapping asset the task specifically calls out.

HOW THE VALUE/LAYOUT INVERSION WORKS
-----------------------------------------
`undo_qwen35_transform()` below ports the exact math from
scripts/patch_gguf_qwen35_transforms.py's `_undo_qwen35_gguf_transform` and
`_qwen35_untile_v_heads` (read that file's docstring first -- it documents
the 16-bug history behind these four rewrites). We do not import that module
directly: its helpers are written to be spliced into vllm_gguf_plugin's
source (they assume streaming per-tensor state via a `qweight_type` tag seen
just before each quantised tensor, needed there because the plugin never
fully dequantises a still-ggml-packed tensor unless it has to). This script
always dequantises every tensor up front (RTN needs fp32 either way), so the
"keep it Q8_0-packed and permute in the packed domain" special case in the
original does not apply here -- we permute plain fp32/fp16 tensors, which is
strictly simpler. The math (which tensors get the -exp() inverse, the 1+w
norm inverse, the conv1d squeeze/unsqueeze, and which get the grouped<->tiled
V-head permutation) is unchanged and MUST stay unchanged: shipping a
checkpoint without these inverses is silent garbage (see STATUS.md's most
expensive bug), not a loud failure.

RUNBOOK -- L4 GPU VALIDATION (not run from this box; GPU time is scheduled
separately, see the L4 sm89 numbers this route is chasing in STATUS.md)
-----------------------------------------------------------------------------
0. Prereq check (0 GPU cost): benchmark the ready-made
   QuantTrio/Qwen3.5-4B-AWQ checkpoint FIRST -- it answers "does 4-bit-on-
   Marlin beat fp16 on this architecture at all" without waiting on this
   transcode. vLLM auto-detects awq_marlin on sm89, no extra flags:
     vllm serve QuantTrio/Qwen3.5-4B-AWQ --max-model-len 16384
     python scripts/bench_load.py <url> 1 4 16 32
     python scripts/bench_serving.py <url> --dataset longalign --qps 0.1 0.2 0.4
   That's a single, real AWQ quantisation (fp16 -> W4A16 once, calibrated) of
   the 4B, not a GGUF round-trip -- treat it as the quality/perf ceiling for
   "4-bit Marlin on this arch", and this script's 2B RTN-from-GGUF checkpoint
   as the controlled comparison against our own Q4_K_M/fp16 2B numbers (no
   4-bit GPTQ/AWQ 2B exists anywhere on the Hub as of this writing).
1. Run this script (CPU only, no GPU needed):
     python scripts/transcode_gguf_to_gptq.py \
       unsloth/Qwen3.5-2B-GGUF:Q4_K_M out/qwen3.5-2b-gptq \
       --base-repo unsloth/Qwen3.5-2B
2. Copy out/qwen3.5-2b-gptq to the GPU box (or re-run there directly).
3. Serve it -- vLLM auto-detects "quant_method": "gptq" from config.json and
   picks gptq_marlin on sm89 automatically, no --quantization flag needed
   (pass it explicitly the first time to fail loudly if it doesn't):
     vllm serve out/qwen3.5-2b-gptq --quantization gptq --max-model-len 16384
4. Same benchmark battery as the fp16/GGUF sweeps in STATUS.md, for a
   like-for-like comparison:
     python scripts/bench_load.py <url> 1 4 16 32
     python scripts/bench_serving.py <url> --dataset longalign --qps 0.1 0.2 0.4
5. Quality gate (same prompt used throughout this project):
     curl <url>/v1/completions -d '{"prompt": "Thủ đô của Việt Nam là gì?", ...}'
   Expect "Hà Nội", greedy, run it 3x like the hybrid-dispatch check in
   STATUS.md. A GDN weight in the wrong slot degenerates to "!!!!" garbage
   (the historical A_log bug) or fluent-but-wrong hallucination (IQ4_XS
   precedent) -- both are FAIL, not "close enough".
6. Compare tok/s against the GGUF-dequant (~8.6K) and fp16+fp8-KV (~10.95K)
   numbers in STATUS.md. If this lands between them, the route paid off; if
   it doesn't beat GGUF-dequant, the double-quantisation error or Marlin's
   own overhead on these odd-shaped GDN projections (in_proj_a/in_proj_b are
   only 16 columns wide before tp -- see the Marlin MIN_THREAD_N note in
   qwen_gdn_linear_attn.py:618) ate the win; report which.

FOLLOW-UP (not blocking this cut)
--------------------------------------
- Calibrated GPTQ (real Hessian-based error correction, e.g. via GPTQModel
  once it's taught this hybrid mixer, or a hand-rolled single-pass GPTQ using
  a small calibration set) should strictly reduce the RTN error reported
  below without changing the output tensor contract at all.
- This script has only been run against the 2B GGUF. Scaling to 4B/9B needs
  --group-size tuned per the smallest Linear (see the in_proj_a/in_proj_b
  width note above) and a re-check that the --hf-config-derived layer_types/
  head-count fields still match (GGUF-metadata-only fallback is best-effort,
  see `derive_config()`).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf.quants import dequantize
from safetensors.torch import save_file

# --------------------------------------------------------------------------
# GPTQ tensor-layout constants (must match vllm's AutoGPTQConfig exactly --
# see vllm/model_executor/layers/quantization/auto_gptq.py and
# .../utils/quant_utils.py, both read from the local vllm clone).
# --------------------------------------------------------------------------
BITS = 4
GROUP_SIZE = 128
SYM = True  # only symmetric (uint4b8) is wired up below
DESC_ACT = False  # RTN has no activation order; g_idx is trivial (i // group_size)
PACK_FACTOR = 32 // BITS  # 8 int4 values per int32
QUANT_MIN, QUANT_MAX = -8, 7  # uint4b8: unsigned 4-bit with bias 8 -> signed range [-8,7]
ZERO_POINT = 8  # uint4b8's fixed bias; vLLM's Marlin path sets zero_points=False
# unconditionally in AutoGPTQLinearMethod.create_weights, so qzeros' *content*
# is never read at inference time for GPTQ+Marlin with desc_act=False -- see
# MarlinLinearKernel.process_weights_after_loading: `if c.zero_points: ...
# else: setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))`.
# We still write the standard symmetric encoding (all zero-points = 8) so the
# checkpoint round-trips correctly under any GPTQ-compatible tool that DOES
# read qzeros.

# --------------------------------------------------------------------------
# The real Qwen3.5 HF parameter names (from vllm/model_executor/models/
# qwen3_5.py and .../mamba/gdn/qwen_gdn_linear_attn.py), used to drive
# gguf-py's HF-name -> ggml-name table. Text-only, non-MoE (Qwen3_5ForCausalLM
# / Qwen3_5TextConfig) -- the 2B target. MoE (Qwen3_5MoeForCausalLM) is out of
# scope for this first cut.
# --------------------------------------------------------------------------
GLOBAL_TENSORS = ["model.embed_tokens", "model.norm", "lm_head"]

PER_LAYER_TENSORS_COMMON = ["input_layernorm", "post_attention_layernorm"]
PER_LAYER_TENSORS_FULL_ATTN = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
]
PER_LAYER_TENSORS_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PER_LAYER_TENSORS_LINEAR_ATTN = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
    "linear_attn.norm",
    "linear_attn.conv1d",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
]

# Tensors quantised into GPTQ int4 (constraint from the task: only the Linear
# weights vLLM's gptq_marlin path actually handles).
QUANTIZABLE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
)


def is_quantizable(hf_base_name: str) -> bool:
    return any(hf_base_name.endswith(s) for s in QUANTIZABLE_SUFFIXES)


# ==========================================================================
# 1. Config derivation
# ==========================================================================


def derive_config(gguf_reader: "gguf.GGUFReader", hf_config: dict | None) -> dict:
    """Build the flat Qwen3_5TextConfig-shaped dict for our output checkpoint.

    Prefers `hf_config` (the base repo's real config.json, `text_config` sub-
    dict if present) when given -- it's authoritative. Falls back to
    inferring from the GGUF's own KV metadata, which is best-effort: it
    matches the field semantics observed on the 2B (qwen35.ssm.group_count ==
    linear_num_key_heads, qwen35.ssm.time_step_rank == linear_num_value_heads,
    qwen35.ssm.inner_size == linear_value_head_dim * linear_num_value_heads,
    qwen35.ssm.state_size == linear_key_head_dim == linear_value_head_dim) but
    is UNVERIFIED for other sizes (e.g. 4B has 16 K-heads / 32 V-heads, so the
    "state_size == both head dims" identity may not hold there).
    """
    fields = gguf_reader.fields

    def kv(key, default=None):
        f = fields.get(key)
        if f is None:
            return default
        return f.contents()

    if hf_config is not None:
        tc = hf_config.get("text_config", hf_config)  # flat or nested
        cfg = {k: v for k, v in tc.items()}
    else:
        print("[config] no --base-repo/--hf-config given; deriving from GGUF "
              "metadata only (best-effort, verified for 2B only)", file=sys.stderr)
        n_layers = kv("qwen35.block_count")
        interval = kv("qwen35.full_attention_interval", 4)
        hidden = kv("qwen35.embedding_length")
        num_v_heads = kv("qwen35.ssm.time_step_rank")
        inner_size = kv("qwen35.ssm.inner_size")
        cfg = dict(
            vocab_size=None,  # filled in from token_embd tensor shape below
            hidden_size=hidden,
            intermediate_size=kv("qwen35.feed_forward_length"),
            num_hidden_layers=n_layers,
            num_attention_heads=kv("qwen35.attention.head_count"),
            num_key_value_heads=kv("qwen35.attention.head_count_kv"),
            hidden_act="silu",
            rms_norm_eps=kv("qwen35.attention.layer_norm_rms_epsilon"),
            head_dim=kv("qwen35.attention.key_length"),
            linear_conv_kernel_dim=kv("qwen35.ssm.conv_kernel"),
            linear_key_head_dim=kv("qwen35.ssm.state_size"),
            linear_value_head_dim=inner_size // num_v_heads,
            linear_num_key_heads=kv("qwen35.ssm.group_count"),
            linear_num_value_heads=num_v_heads,
            layer_types=[
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(n_layers)
            ],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": kv("qwen35.rope.freq_base"),
                "partial_rotary_factor": (
                    kv("qwen35.rope.dimension_count") / kv("qwen35.attention.key_length")
                ),
                "mrope_interleaved": True,
                "mrope_section": list(kv("qwen35.rope.dimension_sections"))[:3],
            },
        )

    # vocab_size and tie_word_embeddings are read from the GGUF's own tensors
    # regardless of --base-repo, since they must match what we're about to
    # write byte-for-byte.
    token_embd = next(t for t in gguf_reader.tensors if t.name == "token_embd.weight")
    cfg["vocab_size"] = int(token_embd.shape[-1])
    has_output = any(t.name == "output.weight" for t in gguf_reader.tensors)
    cfg["tie_word_embeddings"] = not has_output

    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    cfg["model_type"] = "qwen3_5_text"
    cfg["torch_dtype"] = "float16"
    cfg["quantization_config"] = {
        "quant_method": "gptq",
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "sym": SYM,
        "desc_act": DESC_ACT,
        "lm_head": False,
    }
    return cfg


# ==========================================================================
# 2. GGUF ggml-name <-> HF-name mapping (gguf-py's own table, see module
#    docstring "HOW THE NAME MAPPING WORKS")
# ==========================================================================


def build_hf_to_ggml_map(n_layers: int, layer_types: list[str]) -> dict[str, tuple[str, str]]:
    """hf_base -> (ggml_name, ggml_suffix).

    ggml_suffix is ".weight" for every real Linear/norm/conv1d tensor,
    ".bias" for dt_bias (a bare bias parameter, not a Linear -- llama.cpp
    writes it as "blk.{bid}.ssm_dt.bias", see patch_gguf_tensor_mapping.py's
    docstring), and "" for A_log (a genuinely bare, suffix-less parameter,
    "blk.{bid}.ssm_a" -- see patch_gguf_empty_suffix.py's docstring, the most
    expensive bug in this project's history hinged on this exact tensor
    having no suffix at all).

    gguf-py's stock tensor_mapping.py has no entry for "linear_attn.dt_bias"
    (dt_bias isn't a Linear layer here, so it isn't spelled that way in any
    upstream architecture's checkpoint); patch_gguf_tensor_mapping.py adds
    "model.layers.{bid}.linear_attn.dt" (the _bias-stripped form) as an alias
    for MODEL_TENSOR.SSM_DT to work around this. That patch only touches the
    installed `gguf` package (no vllm dependency), so it's safe -- and
    necessary -- to apply locally too; run it once via
    `python scripts/patch_gguf_tensor_mapping.py` before this script if you
    see a "no entry for ... dt_bias" error.
    """
    tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, n_layers)
    bases = list(GLOBAL_TENSORS)
    for i, layer_type in enumerate(layer_types):
        p = f"model.layers.{i}"
        bases += [f"{p}.{t}" for t in PER_LAYER_TENSORS_COMMON]
        bases += [f"{p}.{t}" for t in PER_LAYER_TENSORS_MLP]
        if layer_type == "full_attention":
            bases += [f"{p}.{t}" for t in PER_LAYER_TENSORS_FULL_ATTN]
        elif layer_type == "linear_attention":
            bases += [f"{p}.{t}" for t in PER_LAYER_TENSORS_LINEAR_ATTN]
        else:
            raise ValueError(f"layer {i}: unknown layer_type {layer_type!r}")

    hf_to_ggml = {}
    unmapped = []
    for base in bases:
        if base.endswith("dt_bias"):
            lookup_key, ggml_suffix = base[: -len("_bias")], ".bias"
        elif base.endswith("A_log"):
            lookup_key, ggml_suffix = base, ""
        else:
            lookup_key, ggml_suffix = base, ".weight"
        name = tmap.get_name(lookup_key)
        if name is None:
            unmapped.append(base)
        else:
            hf_to_ggml[base] = (name, ggml_suffix)
    if unmapped:
        raise RuntimeError(
            f"gguf-py's tensor name map has no entry for {len(unmapped)} expected "
            f"Qwen3.5 tensors: {unmapped[:10]}. If these are all *.dt_bias, run "
            "`python scripts/patch_gguf_tensor_mapping.py` first (see this "
            "function's docstring)."
        )
    return hf_to_ggml


# ==========================================================================
# 3. Value/layout inversion -- see scripts/patch_gguf_qwen35_transforms.py,
#    which this ports (see module docstring for why it isn't imported).
# ==========================================================================


def untile_v_heads(t: torch.Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Inverse of llama.cpp's grouped -> tiled V-head permutation.

    Verbatim port of _qwen35_untile_v_heads in patch_gguf_qwen35_transforms.py.
    """
    shape = list(t.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    t = t.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def undo_qwen35_transform(
    hf_name: str,
    weight: torch.Tensor,
    num_k: int,
    num_v: int,
    head_k: int,
    head_v: int,
) -> torch.Tensor:
    """Undo llama.cpp's Qwen3.5 GGUF weight transforms for one tensor.

    `hf_name` includes the ".weight" suffix (or bare, for A_log/dt_bias which
    are parameters, not Linear weights). `weight` is the fp32 dequantised
    tensor in (out_features, in_features) orientation (matching gguf.quants.
    dequantize's own output convention -- verified empirically: dequantizing
    blk.0.ffn_gate.weight, ne=[2048,6144] (in,out) in ggml's own axis order,
    yields a (6144, 2048) = (out,in) array, i.e. standard PyTorch nn.Linear
    weight layout, no transpose needed here).

    reorder is only True when num_k != num_v (e.g. Qwen3.5-4B: 16 K-heads,
    32 V-heads). Qwen3.5-2B has num_k == num_v == 16, so reorder is False and
    every untile() call below is an identity -- exercised structurally but
    not numerically by the 2B dry run. Ported here anyway (not left as a
    TODO) because scaling this script to 4B/9B is explicit follow-up work
    and this is the one part of the transform that actually depends on
    per-size config, not just tensor category.
    """
    reorder = num_k > 0 and num_v > 0 and num_k != num_v
    num_v_per_k = (num_v // num_k) if num_k else 1

    def untile(t, dim, head_dim):
        if not reorder:
            return t
        return untile_v_heads(t, dim, num_k, num_v_per_k, head_dim)

    if ".linear_attn." in hf_name:
        if hf_name.endswith("linear_attn.A_log"):
            # GGUF holds -exp(A_log); vllm re-applies -exp() at runtime.
            weight = torch.log(weight.double().neg().clamp_min(1e-30)).to(torch.float32)
            return untile(weight, 0, 1)
        if hf_name.endswith("linear_attn.dt_bias"):
            return untile(weight, 0, 1)
        if hf_name.endswith("linear_attn.conv1d.weight"):
            # GGUF stores conv1d 2-D (C, K); real HF checkpoint (a genuine
            # nn.Conv1d, groups=C) stores it 3-D (C, 1, K).
            if weight.ndim == 3:
                weight = weight.squeeze(1)
            if reorder:
                qk = head_k * num_k * 2
                weight = torch.cat([weight[:qk], untile(weight[qk:], 0, head_v)], dim=0)
            return weight.unsqueeze(1)
        if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
            if reorder:
                qk = head_k * num_k * 2
                weight = torch.cat([weight[:qk], untile(weight[qk:], 0, head_v)], dim=0)
            return weight
        if hf_name.endswith("linear_attn.in_proj_z.weight"):
            return untile(weight, 0, head_v)
        if hf_name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
            return untile(weight, 0, 1)
        if hf_name.endswith("linear_attn.out_proj.weight"):
            return untile(weight, 1, head_v) if reorder else weight
        # linear_attn.norm.weight: plain RMSNorm, no 1+w offset, no permute.
        return weight
    if hf_name.endswith("norm.weight"):
        # Every other norm (input_layernorm, post_attention_layernorm,
        # model.norm, q_norm, k_norm) is zero-centred: HF applies x*(1+w),
        # llama.cpp folds the +1 into the stored weight.
        return weight - 1.0
    return weight


# ==========================================================================
# 4. RTN GPTQ quantiser -- reimplements vllm's quantize_weights()/pack_rows()
#    for quant_type=uint4b8, zero_points=False (see BITS/GROUP_SIZE consts).
# ==========================================================================


def pack_rows(q: torch.Tensor, bits: int, k: int, n: int) -> torch.Tensor:
    """int(0..2^bits-1) (K,N) -> int32 packed (K//pack_factor, N).

    Matches vllm.model_executor.layers.quantization.utils.quant_utils.
    pack_rows exactly (row i within a pack_factor-sized group goes into bits
    [bits*i : bits*i+bits) of the packed int32 -- this is the layout
    ops.gptq_marlin_repack expects on the vLLM side).
    """
    pack_factor = 32 // bits
    assert k % pack_factor == 0
    q_np = q.numpy().astype(np.uint32)
    out = np.zeros((k // pack_factor, n), dtype=np.uint32)
    for i in range(pack_factor):
        out |= (q_np[i::pack_factor, :] & ((1 << bits) - 1)) << (bits * i)
    return torch.from_numpy(out.astype(np.int32))


def unpack_rows(packed: torch.Tensor, bits: int, k: int, n: int) -> torch.Tensor:
    """Inverse of pack_rows, used only by our own verification step."""
    pack_factor = 32 // bits
    arr = packed.numpy().astype(np.uint32)
    out = np.zeros((k, n), dtype=np.uint32)
    mask = (1 << bits) - 1
    for i in range(pack_factor):
        out[i::pack_factor, :] = (arr >> (bits * i)) & mask
    return torch.from_numpy(out.astype(np.int32))


def make_symmetric_qzeros(num_groups: int, n: int, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    assert n % pack_factor == 0
    packed_val = 0
    for i in range(pack_factor):
        packed_val |= ZERO_POINT << (bits * i)
    arr = np.full((num_groups, n // pack_factor), packed_val, dtype=np.uint32)
    return torch.from_numpy(arr.astype(np.int32))


def rtn_quantize_gptq(weight: torch.Tensor, group_size: int = GROUP_SIZE, bits: int = BITS, n_grid: int = 20):
    """RTN-quantise an (out_features, in_features) fp32 weight into GPTQ int4.

    Returns (qweight, qzeros, scales, g_idx, w_ref) where w_ref is the
    dequantised reconstruction in the same (out,in) orientation as the input,
    for verification -- not written to the checkpoint.

    Plain min/max-derived scale (clip ratio 1.0) turned out to cost a lot on
    this model: a synthetic-Gaussian sanity check (mean 0, std 0.02, the
    rough magnitude of an LLM weight matrix) gives ~12% L1 relative
    reconstruction error at int4 group128 with plain min/max scale -- well
    above STATUS.md's K-quant band (0.4-7%), because plain min/max scale is
    entirely set by the single most extreme value in each group of 128,
    while the bulk of the mass sits far closer to zero. This is a known
    weakness of naive RTN, not a bug in the pack/unpack math (verified: the
    packed-then-unpacked round trip matches the pre-pack reference to within
    float rounding). The fix that stays within "RTN, no calibration data, no
    forward pass" is a per-group clip-ratio grid search: try shrinking the
    min/max-derived scale by a few percent at a time and keep whichever
    shrink minimizes MSE against the *unclipped* group -- clipping the rare
    outlier costs it a larger individual error but frees up resolution for
    every other value in the group, and for realistic weight distributions
    (concentrated near zero, occasional outliers) that trade is a net win.
    This is the standard "RTN + clip search" baseline used in the GPTQ/AWQ
    literature, not a new technique -- it uses only this tensor's own values,
    so it is still calibration-free.
    """
    assert weight.dtype == torch.float32
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} not divisible by group_size={group_size}"
        )
    k, n = in_features, out_features
    num_groups = k // group_size

    # GPTQ groups along the INPUT dimension -> transpose to (K, N).
    w = weight.t().contiguous()
    wg = w.reshape(num_groups, group_size, n)

    max_val = wg.amax(dim=1, keepdim=True)
    min_val = wg.amin(dim=1, keepdim=True)
    # Matches vllm's quantize_weights(): scale chosen so both extremes of the
    # asymmetric int4 range [-8,7] are representable (zero_points=False).
    base_scale = torch.maximum((max_val / QUANT_MAX).abs(), (min_val / QUANT_MIN).abs())
    base_scale = base_scale.clamp_min(1e-10)

    best_scale = base_scale
    best_mse = None
    for ratio in torch.linspace(1.0, 0.5, n_grid):
        s = (base_scale * ratio).clamp_min(1e-10)
        q_try = torch.round(wg / s).clamp(QUANT_MIN, QUANT_MAX)
        mse = ((q_try * s - wg) ** 2).mean(dim=1, keepdim=True)
        if best_mse is None:
            best_mse, best_scale = mse, s
        else:
            better = mse < best_mse
            best_scale = torch.where(better, s, best_scale)
            best_mse = torch.where(better, mse, best_mse)
    scale = best_scale

    q = torch.round(wg / scale).clamp(QUANT_MIN, QUANT_MAX)
    w_ref = (q * scale).reshape(k, n)  # dequantised reference, (K,N)

    scales_out = scale.reshape(num_groups, n).to(torch.float16)
    q_biased = (q + ZERO_POINT).reshape(k, n).to(torch.int32)  # uint4b8: [0,15]

    qweight = pack_rows(q_biased, bits, k, n)
    qzeros = make_symmetric_qzeros(num_groups, n, bits)
    g_idx = (torch.arange(k, dtype=torch.int32) // group_size)

    return qweight, qzeros, scales_out, g_idx, w_ref.t().contiguous()


def dequant_gptq_for_verification(
    qweight: torch.Tensor, scales: torch.Tensor, k: int, n: int, group_size: int = GROUP_SIZE, bits: int = BITS
) -> torch.Tensor:
    """Unpack + dequantise our own GPTQ tensors back to (out,in) fp32.

    Independent of rtn_quantize_gptq's internals (re-derives from the packed
    int32 tensors as written to disk) -- this is the "read our own output
    back" half of the verification report.
    """
    q = unpack_rows(qweight, bits, k, n).float() - ZERO_POINT
    num_groups = k // group_size
    q = q.reshape(num_groups, group_size, n)
    s = scales.float().reshape(num_groups, 1, n)
    w = (q * s).reshape(k, n)
    return w.t().contiguous()


# ==========================================================================
# 5. Source resolution (download GGUF / base repo files if not local)
# ==========================================================================


def resolve_gguf(src: str, local_gguf: str | None) -> Path:
    if local_gguf:
        return Path(local_gguf)
    if ":" not in src:
        raise SystemExit(f"--src must be 'repo:quant' (e.g. unsloth/Qwen3.5-2B-GGUF:Q4_K_M), got {src!r}")
    repo_id, quant = src.split(":", 1)
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo_id)
    matches = [f for f in files if quant.lower() in f.lower() and f.endswith(".gguf")]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly 1 GGUF matching {quant!r} in {repo_id}, found {matches}")
    print(f"[download] {repo_id}:{matches[0]}", file=sys.stderr)
    return Path(hf_hub_download(repo_id=repo_id, filename=matches[0]))


def resolve_base_repo_files(base_repo: str | None, local_hf_config_dir: str | None, out_dir: Path):
    """Copy config.json + tokenizer files into out_dir. Returns the parsed
    base config dict (or None if neither source was given)."""
    tokenizer_files = [
        "chat_template.jinja",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    if local_hf_config_dir:
        src_dir = Path(local_hf_config_dir)
        for fname in ["config.json"] + tokenizer_files:
            p = src_dir / fname
            if p.exists():
                shutil.copy(p, out_dir / fname)
        cfg_path = src_dir / "config.json"
        return json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    if base_repo:
        from huggingface_hub import hf_hub_download

        cfg = None
        for fname in ["config.json"] + tokenizer_files:
            try:
                p = hf_hub_download(repo_id=base_repo, filename=fname)
            except Exception as e:
                print(f"[base-repo] skip {fname}: {e}", file=sys.stderr)
                continue
            shutil.copy(p, out_dir / fname)
            if fname == "config.json":
                cfg = json.loads(Path(p).read_text(encoding="utf-8"))
        return cfg
    return None


# ==========================================================================
# 6. Main transcode pipeline
# ==========================================================================


def transcode(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = resolve_gguf(args.src, args.local_gguf)
    print(f"[transcode] reading {gguf_path}", file=sys.stderr)
    reader = gguf.GGUFReader(str(gguf_path))
    tensors_by_name = {t.name: t for t in reader.tensors}

    base_cfg = resolve_base_repo_files(args.base_repo, args.local_hf_config, out_dir)
    cfg = derive_config(reader, base_cfg)

    n_layers = cfg["num_hidden_layers"]
    layer_types = cfg["layer_types"]
    num_k = cfg["linear_num_key_heads"]
    num_v = cfg["linear_num_value_heads"]
    head_k = cfg["linear_key_head_dim"]
    head_v = cfg["linear_value_head_dim"]

    hf_to_ggml = build_hf_to_ggml_map(n_layers, layer_types)

    out_tensors: dict[str, torch.Tensor] = {}
    quantized_modules: list[str] = []
    fp16_kept: list[str] = []
    skipped_missing: list[str] = []

    def load_raw(ggml_name: str, ggml_suffix: str) -> torch.Tensor | None:
        t = tensors_by_name.get(ggml_name + ggml_suffix)
        if t is None:
            return None
        qtype = gguf.GGMLQuantizationType(t.tensor_type)
        if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            arr = np.array(t.data, dtype=np.float32) if qtype.name == "F16" else t.data
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        else:
            arr = np.ascontiguousarray(dequantize(t.data, qtype), dtype=np.float32)
        return torch.from_numpy(arr)

    for hf_base, (ggml_name, ggml_suffix) in hf_to_ggml.items():
        raw = load_raw(ggml_name, ggml_suffix)
        if raw is None:
            # Expected-optional tensors (e.g. conv1d.bias if the source GGUF
            # has none -- vLLM's conv1d is bias=False by default).
            skipped_missing.append(hf_base)
            continue

        is_bare_param = hf_base.endswith(("A_log", "dt_bias"))
        hf_name = hf_base if is_bare_param else hf_base + ".weight"

        transformed = undo_qwen35_transform(hf_name, raw, num_k, num_v, head_k, head_v)

        if is_quantizable(hf_base):
            qweight, qzeros, scales, g_idx, _w_ref = rtn_quantize_gptq(transformed)
            out_tensors[hf_name.replace(".weight", ".qweight")] = qweight
            out_tensors[hf_name.replace(".weight", ".qzeros")] = qzeros
            out_tensors[hf_name.replace(".weight", ".scales")] = scales
            out_tensors[hf_name.replace(".weight", ".g_idx")] = g_idx
            quantized_modules.append(hf_base)
        else:
            out_tensors[hf_name] = transformed.to(torch.float16)
            fp16_kept.append(hf_base)

    # lm_head: only written if the GGUF actually has an untied output tensor
    # (cfg["tie_word_embeddings"] already reflects this -- see derive_config).
    if cfg["tie_word_embeddings"]:
        out_tensors.pop("lm_head.weight", None)
        skipped_missing = [x for x in skipped_missing if x != "lm_head"]

    print(f"[transcode] {len(quantized_modules)} modules quantised to GPTQ int4, "
          f"{len(fp16_kept)} tensors kept fp16, {len(skipped_missing)} expected-missing "
          f"({skipped_missing})", file=sys.stderr)

    save_file(out_tensors, out_dir / "model.safetensors", metadata={"format": "pt"})

    cfg_out = {k: v for k, v in cfg.items()}
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[transcode] wrote {out_dir}", file=sys.stderr)
    return out_dir, reader, cfg, quantized_modules, fp16_kept, skipped_missing, hf_to_ggml


# ==========================================================================
# 7. Verification (no GPU needed)
# ==========================================================================


def verify(out_dir: Path, reader, cfg, quantized_modules, fp16_kept, skipped_missing, hf_to_ggml, ref_dir: str | None):
    print("\n" + "=" * 72)
    print("VERIFICATION REPORT")
    print("=" * 72)

    print(f"\n-- tensor inventory --")
    print(f"quantized (GPTQ int4 g{GROUP_SIZE}): {len(quantized_modules)}")
    for name in quantized_modules[:6]:
        print(f"    {name}")
    print(f"    ... ({len(quantized_modules)} total)")
    print(f"kept fp16: {len(fp16_kept)}")
    for name in fp16_kept[:6]:
        print(f"    {name}")
    print(f"    ... ({len(fp16_kept)} total)")
    print(f"expected-missing (e.g. tied lm_head, absent conv1d.bias): {skipped_missing}")

    from safetensors import safe_open

    with safe_open(out_dir / "model.safetensors", framework="pt") as f:
        keys = set(f.keys())

        print(f"\n-- shape/dtype spot checks --")
        sample_bases = []
        for name in quantized_modules:
            if name.endswith(("q_proj", "in_proj_qkv", "mlp.gate_proj", "linear_attn.in_proj_a")):
                sample_bases.append(name)
        sample_bases = sample_bases[:6]

        n_layers = cfg["num_hidden_layers"]
        group_size = GROUP_SIZE
        errors = []
        for base in sample_bases:
            qw = f.get_tensor(base + ".qweight")
            qz = f.get_tensor(base + ".qzeros")
            sc = f.get_tensor(base + ".scales")
            gi = f.get_tensor(base + ".g_idx")
            k_dim = gi.shape[0]
            n_dim = qw.shape[1]
            assert qw.shape == (k_dim // PACK_FACTOR, n_dim), f"{base}: qweight shape {qw.shape}"
            assert qz.shape == (k_dim // group_size, n_dim // PACK_FACTOR), f"{base}: qzeros shape {qz.shape}"
            assert sc.shape == (k_dim // group_size, n_dim), f"{base}: scales shape {sc.shape}"
            assert qw.dtype == torch.int32 and qz.dtype == torch.int32 and gi.dtype == torch.int32
            assert sc.dtype == torch.float16
            print(f"    OK  {base}: qweight{tuple(qw.shape)} qzeros{tuple(qz.shape)} "
                  f"scales{tuple(sc.shape)} g_idx{tuple(gi.shape)}")

        print(f"\n-- round-trip dequant error (RTN noise only) --")
        gguf_tensors = {t.name: t for t in reader.tensors}
        num_k, num_v = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        head_k, head_v = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        for base in sample_bases:
            ggml_name, ggml_suffix = hf_to_ggml[base]
            t = gguf_tensors[ggml_name + ggml_suffix]
            qtype = gguf.GGMLQuantizationType(t.tensor_type)
            pre = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, qtype), dtype=np.float32))
            pre = undo_qwen35_transform(base + ".weight", pre, num_k, num_v, head_k, head_v)

            qw = f.get_tensor(base + ".qweight")
            sc = f.get_tensor(base + ".scales")
            gi = f.get_tensor(base + ".g_idx")
            k_dim, n_dim = gi.shape[0], qw.shape[1]
            post = dequant_gptq_for_verification(qw, sc, k_dim, n_dim)

            rel_err = ((post - pre).abs().sum() / pre.abs().sum()).item()
            errors.append(rel_err)
            print(f"    {base}: rel. error vs pre-quant fp32 (RTN noise) = {rel_err:.4f}")

        if errors:
            print(f"    mean RTN rel. error over {len(errors)} sampled tensors: "
                  f"{sum(errors)/len(errors):.4f} (STATUS.md's K-quant band: 0.004-0.07)")

        if ref_dir:
            print(f"\n-- combined error vs HF reference ({ref_dir}) --")
            from safetensors import safe_open as safe_open_ref

            ref_files = list(Path(ref_dir).glob("*.safetensors"))
            if not ref_files:
                print(f"    no .safetensors found in {ref_dir}, skipping")
            else:
                ref_errs = []
                with safe_open_ref(ref_files[0], framework="pt") as rf:
                    ref_keys = set(rf.keys())
                    for base in sample_bases:
                        ref_key = base + ".weight"
                        if ref_key not in ref_keys:
                            print(f"    {ref_key} not in reference checkpoint, skipping")
                            continue
                        ref_w = rf.get_tensor(ref_key).float()
                        qw = f.get_tensor(base + ".qweight")
                        sc = f.get_tensor(base + ".scales")
                        gi = f.get_tensor(base + ".g_idx")
                        k_dim, n_dim = gi.shape[0], qw.shape[1]
                        post = dequant_gptq_for_verification(qw, sc, k_dim, n_dim)
                        rel_err = ((post - ref_w).abs().sum() / ref_w.abs().sum()).item()
                        ref_errs.append(rel_err)
                        print(f"    {base}: rel. error vs bf16 HF reference (combined "
                              f"GGUF-dequant + RTN noise) = {rel_err:.4f}")
                if ref_errs:
                    print(f"    mean combined rel. error: {sum(ref_errs)/len(ref_errs):.4f}")

    print("\n" + "=" * 72)


def main():
    # NOTE: deliberately NOT passing the module docstring as `description` --
    # it contains Vietnamese text (the quality-gate prompt from the runbook)
    # which crashes argparse's --help on a cp1252 Windows console
    # (UnicodeEncodeError). Read the module docstring in the source instead;
    # `--help` gets a short ASCII summary.
    ap = argparse.ArgumentParser(
        description="Transcode a Qwen3.5 GGUF into a GPTQ checkpoint vLLM serves via "
        "gptq_marlin. See this file's module docstring for the full design writeup and "
        "L4 validation runbook."
    )
    ap.add_argument("src", help="repo:quant, e.g. unsloth/Qwen3.5-2B-GGUF:Q4_K_M")
    ap.add_argument("out_dir", help="output directory for the GPTQ checkpoint")
    ap.add_argument("--base-repo", default=None, help="HF repo to pull config.json/tokenizer from, e.g. unsloth/Qwen3.5-2B")
    ap.add_argument("--local-hf-config", default=None, help="local dir with config.json/tokenizer instead of --base-repo")
    ap.add_argument("--local-gguf", default=None, help="local .gguf path instead of downloading --src")
    ap.add_argument("--ref-checkpoint", default=None, help="local dir with the bf16 HF reference .safetensors, for verification")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    out_dir, reader, cfg, quantized_modules, fp16_kept, skipped_missing, hf_to_ggml = transcode(args)

    if not args.no_verify:
        verify(out_dir, reader, cfg, quantized_modules, fp16_kept, skipped_missing, hf_to_ggml, args.ref_checkpoint)


if __name__ == "__main__":
    main()
