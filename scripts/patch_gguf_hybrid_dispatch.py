"""
Opt-in: route GGUF matmuls by shape instead of by a single global flag —
dequant + cuBLAS for prefill-sized batches, fused Triton kernels for decode.

Depends on patch_gguf_prefer_dequant.py having already run (it must be listed
earlier in setup_env.sh). That patch adds a single unconditional switch,
CUSTOM_VLLM_GGUF_DEQUANT, which sends *every* matmul through dequant+cuBLAS
when set. On L4 (sm89) that is the wrong call half the time: measured on a
Qwen3.5-2B GGUF Q4_K_M server, tok/s at concurrency 1 / 4 / 16 / 32:

                        fused Triton          dequant + cuBLAS
    L4   (sm89)    33.7  / 131  / 489  / 872   25.6 / 96.7 / 362 / 674

Decode (small batches) is 1.2-1.3x faster on the fused kernels. But the same
plugin, same hardware, measured on 12k-token LongAlign prompts (prefill-heavy,
open-loop Poisson load), inverts:

                        fused Triton      dequant + cuBLAS
    L4   (sm89)         ~3,500 tok/s       ~8,580 tok/s      (2.5x)

Prefill is bandwidth-bound on weight reads; cuBLAS's tiled GEMM amortises
that far better than the fused kernels' per-block decode. And a faster
prefill isn't just a throughput number — it directly shrinks the ITL stall
that concurrent decode users feel while a big prompt is being chunked through
the batch ahead of them (see STATUS.md's chunk-size findings).

Neither existing flag captures this: CUSTOM_VLLM_GGUF_DEQUANT=1 is unconditional
and gives up the decode win to get the prefill win. This patch adds a second,
independent flag, CUSTOM_VLLM_GGUF_HYBRID, that inspects the actual batch size
at each call and picks the winning kernel per call instead of per process:

    CUSTOM_VLLM_GGUF_HYBRID=1                    enable, default threshold 1024
    CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD=<int>       override the row-count cutoff

x.shape[0] is the row count of the activation entering this matmul — under
vLLM v1's chunked prefill, one engine step concatenates every token across
every sequence in that step into a single x, so shape[0] is exactly "how much
work is this step," not "how many requests." A pure-decode step at
--max-num-seqs 384 gives x.shape[0] up to ~384 (one row per in-flight
sequence, at most). Prefill chunks are governed by --max-num-batched-tokens,
which in every config measured here is >=2048 (and typically much larger to
keep the GPU saturated). 1024 sits cleanly in the gap between those two
regimes: comfortably above worst-case decode batches, comfortably below the
smallest realistic prefill chunk.

CUSTOM_VLLM_GGUF_HYBRID and CUSTOM_VLLM_GGUF_DEQUANT are independent switches
touching the same function. Hybrid's check runs first and returns early for
qualifying calls, so if DEQUANT=1 is also set, its unconditional branch never
executes for anything hybrid already routed to dequant — but for calls hybrid
declines (small x.shape[0], or qweight_type outside DEQUANT_TYPES), DEQUANT=1
still forces the dequant path underneath. In short: DEQUANT=1 makes HYBRID
moot (it subsumes and outperforms it for decode on sm75, and just wastes the
decode win on sm89); the two are not meant to be combined, only either/or.
"""

import glob
import sysconfig

PREFER_DEQUANT_MARKER = (
    "# --- custom_vllm: optional dequant + cuBLAS instead of the fused Triton kernels ---"
)

# Anchor 1: the flag block patch_gguf_prefer_dequant.py inserts right before
# the function definition. Only present once that patch has run.
FLAG_ANCHOR = (
    f"{PREFER_DEQUANT_MARKER}\n"
    "import os as _custom_vllm_os\n\n"
    "_CUSTOM_VLLM_PREFER_DEQUANT = (\n"
    '    _custom_vllm_os.environ.get("CUSTOM_VLLM_GGUF_DEQUANT", "0") == "1"\n'
    ")\n\n\n"
)

PATCH_MARKER = "# --- custom_vllm: hybrid kernel dispatch by matmul shape (prefill vs decode) ---"

FLAG_PATCH = (
    FLAG_ANCHOR
    + f"{PATCH_MARKER}\n"
    "_CUSTOM_VLLM_GGUF_HYBRID = (\n"
    '    _custom_vllm_os.environ.get("CUSTOM_VLLM_GGUF_HYBRID", "0") == "1"\n'
    ")\n"
    "_CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD = int(\n"
    '    _custom_vllm_os.environ.get("CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD", "1024")\n'
    ")\n\n\n"
)

# Anchor 2: the early-return block patch_gguf_prefer_dequant.py inserts at the
# top of _fused_mul_mat_gguf's body. Only present once that patch has run.
BODY_ANCHOR = (
    f"    {PREFER_DEQUANT_MARKER}\n"
    "    if _CUSTOM_VLLM_PREFER_DEQUANT and qweight_type in DEQUANT_TYPES:\n"
    "        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]\n"
    "        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)\n"
    "        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)\n"
    "        return x @ weight.T\n"
)

BODY_PATCH = (
    f"    {PATCH_MARKER}\n"
    "    if (\n"
    "        _CUSTOM_VLLM_GGUF_HYBRID\n"
    "        and qweight_type in DEQUANT_TYPES\n"
    "        and x.shape[0] >= _CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD\n"
    "    ):\n"
    "        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]\n"
    "        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)\n"
    "        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)\n"
    "        return x @ weight.T\n"
    + BODY_ANCHOR
)

site_packages = sysconfig.get_paths()["purelib"]
matches = glob.glob(f"{site_packages}/vllm_gguf_plugin/quantization/linear.py")
if not matches:
    raise SystemExit(f"vllm_gguf_plugin/quantization/linear.py not found under {site_packages}")
path = matches[0]

with open(path, encoding="utf-8") as f:
    src = f.read()

if PATCH_MARKER in src:
    print(f"Already patched: {path}")
else:
    for anchor in (FLAG_ANCHOR, BODY_ANCHOR):
        if anchor not in src:
            raise SystemExit(
                f"Anchor not found in {path}; is patch_gguf_prefer_dequant.py applied "
                "and listed earlier in setup_env.sh?"
            )
    src = src.replace(FLAG_ANCHOR, FLAG_PATCH, 1).replace(BODY_ANCHOR, BODY_PATCH, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched: {path}")
