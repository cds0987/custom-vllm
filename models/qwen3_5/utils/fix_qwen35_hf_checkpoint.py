"""
Post-process an HF-saved Qwen3.5 text-only checkpoint so vLLM can serve it.

Two independent bugs surfaced serving the TEST 10 AWQ checkpoint
(scripts/quantize_awq_2b.py's output, a plain safetensors save via
AutoModelForCausalLM + llm-compressor's oneshot()) that do NOT affect GGUF
checkpoints, because the GGUF loading path already has its own patches
(patch_vllm_qwen35_registry.py, patch_gguf_drop_mrope.py) covering the
GGUF-specific config parser. Any OTHER non-GGUF Qwen3.5 checkpoint saved
directly from HF Transformers (AWQ via llm-compressor, a plain fp16 copy,
a future GPTQ export, etc.) will hit the same two bugs, so this is a
standalone fixer rather than folded into the existing GGUF-only patches.

Bug 1 -- "language_model" prefix mismatch (RuntimeError at weight load):

    ValueError: There is no module or parameter named 'language_model' in
    Qwen3_5Model. The available parameters belonging to (Qwen3_5Model) are:
    {'layers.0.input_layernorm.weight', ...}

  HF's own Qwen/Qwen3.5-2B implementation saves every backbone weight under
  a "model.language_model.*" prefix (shared naming convention with the
  multimodal Qwen3_5ForConditionalGeneration sibling, present even in the
  text-only checkpoint). vLLM's Qwen3_5Model.hf_to_vllm_mapper (see
  vllm/model_executor/models/qwen3_5.py) has no rule to strip this prefix
  for the plain ForCausalLM path -- only the ConditionalGeneration wrapper
  class handles it, by construction (it literally has a `self.language_model
  = Qwen3_5ForCausalLM(...)` submodule to receive those keys). Loading the
  same checkpoint through the text-only registration (which
  patch_vllm_qwen35_registry.py makes vLLM prefer, correctly, for a
  text-only config) leaves the checkpoint's "language_model." segment with
  nothing on the vLLM side to consume it.

  Fix: strip "model.language_model." -> "model." from every safetensors key.

Bug 2 -- M-RoPE assertion (AssertionError at first generation step):

    File "vllm/v1/worker/gpu_model_runner.py", line 1640, in
    _init_mrope_positions
        assert supports_mrope(model), "M-RoPE support is not implemented."

  Same root cause patch_gguf_drop_mrope.py already documents and fixes for
  GGUF checkpoints: Qwen3.5's config.json ships rope_parameters with
  mrope_section/mrope_interleaved even for the text-only variant (inherited
  from the unified multimodal/text config design), and
  ModelConfig.uses_mrope reads that unconditionally regardless of which
  architecture class actually gets instantiated. The text-only
  Qwen3_5ForCausalLM class does not implement SupportsMRoPE (correctly --
  there is no vision tower, so every token is t==h==w==position, which is
  exactly plain 1-D RoPE), so this trips vLLM's own consistency check.
  patch_gguf_drop_mrope.py fixes this INSIDE GGUFConfigParser.parse(),
  which never runs for a plain HF config.json load -- hence this separate
  fixer.

  Fix: delete mrope_section/mrope_interleaved from config.json's
  rope_parameters, exactly like patch_gguf_drop_mrope.py's docstring
  reasons through for the GGUF case.

Bug 3 -- stale model.safetensors.index.json (silent, only bites downstream
tooling that trusts the index instead of re-reading the shard):

  Checkpoints saved with `save_pretrained` sometimes ship a
  model.safetensors.index.json ("weight_map": {tensor_key: shard_filename})
  even for a single shard (RedHatAI/Qwen3.5-9B-quantized.w4a16 does this).
  fix_safetensors rewrites keys INSIDE the .safetensors binaries but, until
  this fix, never touched the index -- so the index's weight_map kept the
  OLD "model.language_model.*" keys after a fix, silently out of sync with
  the tensors it claims to describe. Nothing in vLLM's own loader reads this
  file for the champion (it's a raw safetensors read via safe_open, index
  ignored), so this never surfaced serving through `vllm serve` directly --
  but scripts/graft_gguf_gdn.py's `load_frame_weight_map` PREFERS the index
  when present precisely because sharded checkpoints need it, and got zero
  `model.layers.N.linear_attn.in_proj_qkv.weight` matches back from a frame
  that, per fix_safetensors, was already fixed (GraftError: "no ... keys
  found in --frame; nothing to graft") -- the tensors were right, the index
  pointing at them was not.

  Fix: rewrite the same STRIP_PREFIX -> REPLACEMENT substitution across
  model.safetensors.index.json's `weight_map` keys (the shard-filename
  values are untouched -- shard filenames were never prefixed).

Bug 4 -- mrope_section/mrope_interleaved can live under `text_config`, not
just the config root (silent, only matters after Bug 1's rename):

  Bug 2's fix only ever checked config['rope_parameters']. That is correct
  for a flat, text-only-saved config (this project's own quantize_*.py
  outputs, saved via plain AutoModelForCausalLM -- no text_config nesting).
  But RedHatAI's checkpoint (and any other Qwen3_5ForConditionalGeneration-
  shaped multimodal config) nests the text backbone's fields under
  config['text_config'], including rope_parameters -- fix_config silently
  no-op'd ("no rope_parameters block, nothing to drop") while
  config['text_config']['rope_parameters'] still had mrope_section/
  mrope_interleaved sitting in it. This didn't bite RedHatAI's own normal
  `vllm serve <repo>` (that goes through the ConditionalGeneration
  registration, which legitimately implements SupportsMRoPE since the
  checkpoint really does carry a vision tower) -- but Bug 1's prefix-strip
  repoints loading at the text-only class (same mechanism
  patch_vllm_qwen35_registry.py already documents forcing for a text-only
  serving target), which does NOT implement SupportsMRoPE, so leaving
  text_config's mrope keys in place risks the exact
  `AssertionError: M-RoPE support is not implemented.` Bug 2 was written to
  prevent -- just one level deeper in the config tree than Bug 2 checked.

  Fix: check config['rope_parameters'] AND config['text_config']
  ['rope_parameters'] (when text_config exists), drop mrope_section/
  mrope_interleaved from whichever is present.

Usage (idempotent -- safe to run more than once on the same checkpoint dir;
every fix here is a no-op once its target is already gone):

    python scripts/fix_qwen35_hf_checkpoint.py /content/qwen35-2b-awq

Rewrites model*.safetensors (single- or multi-shard) and their
model.safetensors.index.json if present, plus config.json (root and/or
text_config), all in place. Does not touch tokenizer/chat_template files.
"""

import argparse
import glob
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

STRIP_PREFIX = "model.language_model."
REPLACEMENT = "model."


def fix_safetensors(checkpoint_dir: str) -> None:
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shard_paths:
        print(f"no .safetensors files found under {checkpoint_dir}")
        return
    for path in shard_paths:
        tensors = {}
        renamed = 0
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                new_key = key.replace(STRIP_PREFIX, REPLACEMENT)
                if new_key != key:
                    renamed += 1
                tensors[new_key] = f.get_tensor(key)
        if renamed == 0:
            print(f"{path}: no '{STRIP_PREFIX}' keys found, already fixed")
            continue
        tmp_path = path + ".fixing"
        save_file(tensors, tmp_path)
        os.replace(tmp_path, path)
        print(f"{path}: renamed {renamed}/{len(tensors)} keys")


def fix_index(checkpoint_dir: str) -> None:
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"no model.safetensors.index.json found under {checkpoint_dir}")
        return
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index.get("weight_map", {})
    new_weight_map = {}
    renamed = 0
    for key, shard in weight_map.items():
        new_key = key.replace(STRIP_PREFIX, REPLACEMENT)
        if new_key != key:
            renamed += 1
        new_weight_map[new_key] = shard
    if renamed == 0:
        print(f"{index_path}: no '{STRIP_PREFIX}' keys found, already fixed")
        return
    index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"{index_path}: renamed {renamed}/{len(weight_map)} keys")


def _drop_mrope(rope_parameters: dict, label: str) -> list[str]:
    dropped = [
        k
        for k in ("mrope_section", "mrope_interleaved")
        if rope_parameters.pop(k, None) is not None
    ]
    if dropped:
        print(f"config.json: dropped {dropped} from {label}")
    else:
        print(f"config.json: {label}: mrope_section/mrope_interleaved already absent")
    return dropped


def fix_config(checkpoint_dir: str) -> None:
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"no config.json found under {checkpoint_dir}")
        return
    with open(config_path) as f:
        config = json.load(f)

    any_dropped = False
    root_rope = config.get("rope_parameters")
    if root_rope:
        any_dropped = bool(_drop_mrope(root_rope, "rope_parameters")) or any_dropped
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_rope = text_config.get("rope_parameters")
        if text_rope:
            any_dropped = bool(_drop_mrope(text_rope, "text_config.rope_parameters")) or any_dropped
    if not root_rope and not (isinstance(text_config, dict) and text_config.get("rope_parameters")):
        print("config.json: no rope_parameters block (root or text_config), nothing to drop")
        return

    if not any_dropped:
        return
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", help="path to the HF-saved checkpoint directory")
    args = ap.parse_args()

    fix_safetensors(args.checkpoint_dir)
    fix_index(args.checkpoint_dir)
    fix_config(args.checkpoint_dir)
    print("DONE_FIX_QWEN35_HF_CHECKPOINT")
