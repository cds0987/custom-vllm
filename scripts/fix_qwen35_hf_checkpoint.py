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

Usage (idempotent -- safe to run more than once on the same checkpoint dir;
Bug 1's rename is a no-op on keys that no longer have the prefix, and Bug 2's
config edit is a no-op once the keys are already gone):

    python scripts/fix_qwen35_hf_checkpoint.py /content/qwen35-2b-awq

Only rewrites model.safetensors (single-shard checkpoints; if the checkpoint
is sharded across multiple *.safetensors files, all of them are covered) and
config.json in place. Does not touch tokenizer/chat_template files.
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


def fix_config(checkpoint_dir: str) -> None:
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"no config.json found under {checkpoint_dir}")
        return
    with open(config_path) as f:
        config = json.load(f)
    rope_parameters = config.get("rope_parameters")
    if not rope_parameters:
        print("config.json: no rope_parameters block, nothing to drop")
        return
    dropped = [
        k
        for k in ("mrope_section", "mrope_interleaved")
        if rope_parameters.pop(k, None) is not None
    ]
    if not dropped:
        print("config.json: mrope_section/mrope_interleaved already absent")
        return
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"config.json: dropped {dropped} from rope_parameters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", help="path to the HF-saved checkpoint directory")
    args = ap.parse_args()

    fix_safetensors(args.checkpoint_dir)
    fix_config(args.checkpoint_dir)
    print("DONE_FIX_QWEN35_HF_CHECKPOINT")
