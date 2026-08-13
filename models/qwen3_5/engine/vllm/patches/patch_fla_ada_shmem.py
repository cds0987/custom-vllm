"""
Opt-in: stop the fla/GDN kernels from silently downgrading block sizes on Ada.

vllm/model_executor/layers/fla/ops/utils.py gates several autotune spaces on
check_shared_mem(), whose default path compares the device's shared memory
against Backend.DEFAULT = 102400 bytes. Ada-class GPUs (L4, RTX 40xx) have
101376 bytes — 1 KiB short — so check_shared_mem() returns False and
chunk_o.py / cumsum.py fall back to the smaller BKV_LIST/BS_LIST bins meant
for pre-Ampere hardware, even though a Backend.ADA = 101376 enum entry exists
and is referenced nowhere.

On Qwen3.5 this touches 75% of the layers (the gated-delta-net path), and the
measured picture fits: prefill lands at ~36% MFU on an L4 and scaled ~2x
below the FLOPs prediction from 2B to 9B, with the matmul path already ruled
out (dequant/cuBLAS and fp16 moved throughput; attention was shown to be a
<=5% end-to-end factor at 1-in-4 layers).

Gated behind CUSTOM_VLLM_FLA_ADA_SHMEM=1 rather than unconditional: enabling
the larger tiles admits autotune configs that are close to the 99 KiB limit,
and a config that exceeds it fails with "out of resource: shared memory"
rather than running slowly (upstream PR #43047 adds a shmem-aware pruner for
exactly this class of problem; this patch is the minimal unlock while that
is unmerged). If serving crashes with an OOR error under this flag, turn it
off and report the shape — that is the pruner's job, not this patch's.

The change: when the caller does not name an arch (arch="none"), compare
against the real device limit instead of the DEFAULT constant, so the check
answers "does this GPU have its own full shared memory" rather than "does
this GPU have at least 100 KiB".
"""

import glob
import os
import sysconfig

PATCH_MARKER = "# --- custom_vllm: Ada has 101376B shared mem, 1KiB under DEFAULT; don't downgrade ---"

ANCHOR = '''@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
'''

PATCH = f'''{PATCH_MARKER}
_CUSTOM_VLLM_FLA_ADA_SHMEM = (
    os.environ.get("CUSTOM_VLLM_FLA_ADA_SHMEM", "0") == "1"
)


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        if _CUSTOM_VLLM_FLA_ADA_SHMEM and arch == "none":
            # An unnamed-arch probe means "is a reasonably modern GPU"; Ada
            # (101376B) fails the 102400B DEFAULT by 1 KiB and gets pre-Ampere
            # block sizes. Compare against Ada's own limit instead.
            return max_shared_memory >= Backend.ADA.value
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
'''

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm/model_executor/layers/fla/ops/utils.py")
if not matches:
    # Not every vllm build ships this file (it was added for the fla/GDN
    # kernels and its path has moved across versions). Absence here just
    # means this opt-in unlock has nothing to patch on this build -- that is
    # not an error, so skip cleanly instead of aborting the rest of
    # setup_env.sh (which runs under `set -e`). Contrast with the
    # anchor-mismatch case below: if the file DOES exist but the anchor
    # text is gone, the source really did change underneath us and that
    # stays fatal.
    print(
        f"vllm fla/ops/utils.py not found under {site_packages} -- "
        "skipping (patch not applicable on this vllm build)"
    )
    raise SystemExit(0)
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {path}; vllm source may have changed")
else:
    src = src.replace(ANCHOR, PATCH, 1)
    if "\nimport os\n" not in src:
        src = src.replace("import functools\n", "import functools\nimport os\n", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
