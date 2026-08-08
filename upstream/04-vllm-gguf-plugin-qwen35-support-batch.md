# [Bug] Qwen3.5 GGUF support: six correctness/compatibility gaps (naming, vision config, multimodality detection, weight naming, tensor layout, RoPE config)

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** High (each individually blocks Qwen3.5 GGUF loading or serving; none is reachable in isolation from the others, since Qwen3.5 support requires clearing all of them)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26, transformers 5.13.1
**Local fixes:** `scripts/patch_gguf_plugin.py` (items 1, 2, 3, 5, 6 below), `scripts/patch_gguf_conv1d_shape.py` (item 4), `scripts/patch_gguf_drop_mrope.py` (item 7)
**Duplicate check:**
- vllm-project/vllm#38122 / PR #38140 (unmerged, open): covers items 1 and 2 below, but against the pre-plugin in-tree loader `vllm/model_executor/model_loader/gguf_loader.py`, which GGUF support has since moved out of (vllm 0.26 relocated it to vllm-gguf-plugin — see our commit history). The equivalent bugs exist independently in vllm-gguf-plugin's own copy of this logic (`weights_adapter/default.py`), which #38122/#38140 do not touch. This draft's items 1–2 are therefore the plugin-side counterpart of that issue/PR, not a duplicate of it; items 3–7 are not covered by #38122/#38140 at all.
- vllm-project/vllm#36456: a *different* failure in the same feature area (crash in `maybe_override_with_speculators` reading a local GGUF file path directly instead of the supplied `--hf-config-path`), upstream of everything in this draft. Not overlapping — noted for completeness since it's part of the same "load a local Qwen3.5 GGUF" user journey.
- vllm-project/vllm-gguf-plugin#80: tracking issue for Qwen3.5/3.6 MoE support in general, no upstream fix merged. This draft (plus reports 01–03, 05) is offered as the concrete fix list for that tracking issue.

## Overview

Getting a text-only Qwen3.5 GGUF checkpoint to load and serve through vllm-gguf-plugin requires clearing seven distinct bugs across the plugin (this draft covers six of them plus one already reported separately). They are bundled here because none is independently useful — fixing any one still leaves the checkpoint unloadable until the rest are also fixed — but each is a small, self-contained change against a specific anchor.

## 1. `model_type` naming mismatch: `qwen3_5` (HF) vs `qwen35` (gguf package)

`build_name_map()` matches the HF config's `model_type` string directly against `gguf.MODEL_ARCH_NAMES` values. HF's `model_type` for Qwen3.5 is `"qwen3_5"` (underscore); the `gguf` package's architecture name is `"qwen35"` (no separator). The exact-match lookup fails even though the architecture itself is fully present in the `gguf` package:

```
RuntimeError: Unknown gguf model_type: qwen3_5
```

Same shape as the plugin's existing `gemma3_text` → `gemma3` normalization a few lines above. Fix: add `qwen3_5` → `qwen35` and `qwen3_5_moe` → `qwen35moe` to that same normalization block.

## 2. Vision config layer count: `num_hidden_layers` vs `depth`

For multimodal configs, the plugin reads `config.vision_config.num_hidden_layers` to build the vision tensor name map. `Qwen3_5VisionConfig` (like other Qwen-VL vision configs) exposes this as `depth` instead:

```
AttributeError: 'Qwen3_5VisionConfig' object has no attribute 'num_hidden_layers'
```

Fix: `getattr(config.vision_config, "num_hidden_layers", getattr(config.vision_config, "depth", None))`.

## 3. `is_multimodal` decided from config presence, not from the architecture vllm actually resolved

`build_name_map()` decides "is this multimodal?" purely from `config.vision_config is not None`, and always builds its meta-device dummy model with `AutoModelForCausalLM.from_config(config, ...)` (the *composite*, vision+text config). Both halves are wrong for a text-only GGUF of what is architecturally a multimodal-capable model family:

- vllm resolves the concrete *architecture* — `Qwen3_5ForCausalLM` for a text-only GGUF (see the companion vllm-core report on architecture registration) — and builds only the language tower. Deriving multimodality from the config alone instead produces `model.language_model.layers.*` name-map entries that don't exist in the built model:
  ```
  There is no module or parameter named 'language_model'
  ```
- Passing the composite config to `AutoModelForCausalLM` resolves to the text-only class, which then expects a `Qwen3_5TextConfig` and crashes reading `config.vocab_size` (see the separate transformers report on this attribute gap) — the plugin's own already-computed `text_config` (`config.get_text_config()`) is the correct thing to hand it instead.

Fix: clear `is_multimodal` whenever every architecture vllm resolved is a plain `*ForCausalLM`; use `AutoModelForImageTextToText.from_config(config, ...)` for the genuinely multimodal case (where `AutoModelForCausalLM` would otherwise resolve the composite config to the wrong class) and `AutoModelForCausalLM.from_config(text_config, ...)` for the text-only case.

## 4. `conv1d` weight stored 2-D, vllm's `MambaMixer2` expects 3-D

GGUF stores the SSM depthwise conv1d kernel as `(channels, kernel_size)`. vllm allocates it with `torch.nn.Conv1d`'s canonical `(channels, 1, kernel_size)` layout (`groups == channels`, so the per-group input-channel dim is 1). The sharded weight loader in `mamba_mixer2` assigns into a parameter slice without reconciling rank:

```
RuntimeError: The expanded size of the tensor (1) must match the existing size (2048)
at non-singleton dimension 1. Target sizes: [2048, 1, 4]. Tensor sizes: [2048, 4]
```

Not Qwen3.5-specific — this hits any hybrid GGUF model using `MambaMixer2` (qwen3next included). Fix belongs in the adapter's `transform_weight()` hook, the same place gemma3's adapter already fixes up its own GGUF-vs-HF discrepancies: `weight.unsqueeze(1)` when `hf_name.endswith("conv1d.weight") and weight.ndim == 2`.

## 5. `_bias` suffix stripping missing (only `_weight` is handled)

The plugin's suffix-splitting special-cases a trailing `_weight` (no dot) on top of the normal `.weight`/`.bias` split — e.g. `foo_weight` → base `foo`, suffix `weight`. Qwen3.5's SSM dt-bias tensor is named `...linear_attn.dt_bias` (trailing `_bias`, no dot), which falls through unstripped and never matches any `gguf` tensor template (the template itself is also missing — see the companion gguf-py report), causing `"Failed to map GGUF parameters"` for every `dt_bias` tensor. Fix: add the symmetric `elif base_name.endswith("_bias"): base_name = base_name[:-5]; suffix = "bias"`.

## 6. Vision-merger tensors absent in text-only quants are not allowlisted

A text-only Qwen3.5 GGUF (no separate mmproj file loaded, matching llama.cpp's own convention of shipping vision weights as a separate `mmproj-*.gguf`) genuinely has no `model.visual.merger.{norm,linear_fc1,linear_fc2}` tensors. This is expected-absent, not a bug — but the plugin's mapping-completeness check doesn't know that and raises `"Failed to map GGUF parameters"`. Fix: add these three parameter names to `sideload_params` (the same allowlist mechanism already used for MoE expert weights), gated on `is_multimodal and model_type == "qwen35"`.

## 7. M-RoPE left enabled on a text-only GGUF, triggering an unrelated assertion

A text-only GGUF still carries the multimodal text config's M-RoPE settings (`rope_scaling["mrope_section"]`). `ModelConfig.uses_mrope` is derived purely from that config field, but the text-only class vllm builds (`Qwen3_5ForCausalLM`) does not declare `SupportsMRoPE` — correctly, since M-RoPE only exists to give image/video tokens separate temporal/height/width position components, and with no vision tower every token is text (t == h == w == position), which is exactly standard 1-D RoPE:

```
File "vllm/v1/worker/gpu/mm/rope.py", line 144, in get_rope_state
    assert isinstance(model, SupportsMRoPE)
AssertionError
```

Fix, applied in `GGUFConfigParser.parse()` right after the architecture is pinned to the text-only class: when that architecture is a plain `*ForCausalLM`, strip `mrope_section`/`mrope_interleaved` out of `rope_scaling` (collapsing to `None` if nothing meaningful remains). This mirrors what llama.cpp itself does when loading the language tower of a Qwen3.5 GGUF without an mmproj file — it is a correction of the config to what the loaded model actually is, not an approximation.

## Weight-type routing note

A seventh item from the same investigation — GGUF weight-type tag parameters being routed through the fused-shard weight loader path instead of their own `_store()` — is significant and independent enough (it has its own crash signature and its own silent-miscompute failure mode) that we filed it separately: see the companion report "Tuple shard-ids collapse per-shard GGUF weight types" (`scripts/patch_gguf_weight_type_loader.py`).

## Reproduction (all seven fixes applied together)

```bash
vllm serve unsloth/Qwen3.5-2B-GGUF:Q4_K_M \
  --hf-config-path Qwen/Qwen3.5-2B --tokenizer Qwen/Qwen3.5-2B
```

Loads and serves without any of the errors above. Combined with reports 01–03 (which fix weight *values*, not loading mechanics), output is coherent and numerically matches the HF reference within quantization noise.
