"""
vllm's text-only Qwen3.5 class is not marked as a hybrid model.

Qwen3.5 interleaves gated-delta-net (SSM) layers with full-attention layers --
Qwen3_5DecoderLayer literally subclasses Qwen3NextDecoderLayer -- and
Qwen3NextForCausalLM declares IsHybrid for exactly that reason. But
Qwen3_5ForCausalLMBase lists only (nn.Module, HasInnerState, SupportsEagle3,
SupportsLoRA, SupportsPP); IsHybrid and the three mamba-state classmethods
were put on the multimodal Qwen3_5ForConditionalGeneration instead.

is_hybrid(model) is just getattr(model, "is_hybrid", False), so without it
vllm never derives cache_config.mamba_block_size and KV-cache spec collection
trips over its own assertion:

    File "vllm/model_executor/layers/mamba/abstract.py", line 47,
      in get_kv_cache_spec
        assert mamba_block_size is not None
    AssertionError

The bug was unreachable until Qwen3_5ForCausalLM became selectable (see
patch_vllm_qwen35_registry.py). The state helpers read nothing but
hf_text_config, so the text-only class can share the ones already written for
the multimodal class verbatim.
"""

import glob
import sysconfig

PATCH_MARKER = "# --- custom_vllm: text-only Qwen3.5 is hybrid (SSM + full attention) too ---"

BASES_ANCHOR = """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
"""
BASES_PATCH = """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
):
"""

# The mamba-state classmethods live on Qwen3_5ForConditionalGeneration, which is
# defined after Qwen3_5ForCausalLMBase, so they are shared at module scope.
TAIL_PATCH = f'''

{PATCH_MARKER}
for _name in (
    "get_mamba_state_dtype_from_config",
    "get_mamba_state_shape_from_config",
    "get_mamba_state_copy_func",
):
    setattr(
        Qwen3_5ForCausalLMBase,
        _name,
        classmethod(getattr(Qwen3_5ForConditionalGeneration, _name).__func__),
    )
del _name
'''

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm/model_executor/models/qwen3_5.py")
if not matches:
    raise SystemExit(f"vllm/model_executor/models/qwen3_5.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif BASES_ANCHOR not in src:
    raise SystemExit(f"Class-bases anchor not found in {path}; vllm source may have changed")
else:
    src = src.replace(BASES_ANCHOR, BASES_PATCH, 1) + TAIL_PATCH
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
