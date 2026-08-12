# [PR proposal] Dispatch GGUF matmuls by shape — fused Triton kernels for decode, dequant+cuBLAS for prefill

## Tóm tắt cho người không chuyên

Có hai cách để nhân các ma trận đã nén 4-bit trong lúc chạy mô hình, và cách
nào nhanh hơn phụ thuộc vào việc máy đang "đọc một câu hỏi dài" (prefill) hay
"sinh từng chữ trả lời" (decode) — thử nghiệm cho thấy chênh lệch tới 2,5 lần
tuỳ tình huống, và không có cách nào thắng cả hai trường hợp cùng lúc. Đề
xuất này là thêm một cờ tuỳ chọn để phần mềm tự chọn đúng cách nhân theo độ
lớn của phép tính đang chạy, thay vì bắt người vận hành phải đoán và cấu
hình cứng một cách cho toàn bộ máy chủ. Đã đo và xác nhận cách chọn tự động
này đạt gần như tốt nhất ở cả hai tình huống cùng lúc.

**Target repo:** vllm-project/vllm-gguf-plugin
**Type:** Performance enhancement (PR proposal, not a bug report)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26
**Local implementation:** `scripts/patch_gguf_prefer_dequant.py` (prerequisite: adds the unconditional `CUSTOM_VLLM_GGUF_DEQUANT` env-gated dequant+cuBLAS path) + `scripts/patch_gguf_hybrid_dispatch.py` (this proposal: adds a second, independent flag that picks the kernel per call by activation shape instead of per process)
**Duplicate check:** No existing issue or PR found proposing shape-based kernel dispatch in vllm-gguf-plugin. vllm-project/vllm#36802 and PR #43047 concern a different problem in the same neighborhood (Triton *shared-memory* `OutOfResources` crashes on non-H100 GPUs for FLA/gated-delta-net kernels, addressed by an autotune config pruner) — not a speed regression, and not in vllm-gguf-plugin's matmul dispatch. No overlap; noted for context since both concern Triton kernel behavior on non-flagship GPUs for hybrid/SSM architectures.

## Problem

`_fused_mul_mat_gguf()` in `vllm_gguf_plugin/quantization/linear.py` picks between three strategies per matmul: `mmvq` for small batches, `mmq` for larger ones, and dequant+cuBLAS only for quant types the first two don't cover. K-quants (the most common GGUF quantization family) are members of both `MMVQ_QUANT_TYPES` and `MMQ_QUANT_TYPES`, so the dequant+cuBLAS path is structurally unreachable for them, even on hardware where it's dramatically faster.

Without llama.cpp's hand-tuned CUDA kernels available, the plugin's "fused" ops fall back to Triton implementations (`ggml_mul_mat_a8` → `ggml_mul_mat_a8_triton`), whose per-call overhead turns out to be *pathological* on Turing (sm75) specifically — not slow in some generic sense, but disproportionately slow relative to what the hardware is otherwise capable of.

## Measurements establishing the problem

Single-token decode, Qwen3.5-2B, Q4_K_M, tok/s at concurrency 1/4/16/32:

| | fused Triton | dequant + cuBLAS |
|---|---|---|
| T4 (sm75) | 0.92 / 3.4 / 12 / 21 | 13.7 / 52.4 / 193 / 340 |
| L4 (sm89) | 33.7 / 131 / 489 / 872 | 25.6 / 96.7 / 362 / 674 |

The reversal is the finding: dequant+cuBLAS wins decode by **15–17×** on T4, but *loses* decode by **1.2–1.3×** on L4. Going from T4 to L4 speeds the fused kernels up ~36× while the hardware itself is only ~2× faster (comparable memory bandwidth/compute ratio) — the fused kernels are not intrinsically slow, they are specifically broken on sm75. A single global "prefer dequant" flag (`CUSTOM_VLLM_GGUF_DEQUANT`, already proposed for K-quants in a separate, prerequisite patch) is therefore a per-GPU-architecture tuning decision an operator has to get right, and it's the *wrong* setting for one of the two architectures we tested for the single workload type (decode) it was designed for.

The picture inverts again once workload shape changes. Same hardware (L4/sm89), long-context prefill (12k-token LongAlign-10k prompts, open-loop Poisson load):

| | fused Triton | dequant + cuBLAS |
|---|---|---|
| sustained tok/s | ~3,500 | ~8,580 (2.5×) |

So on the *same GPU*, decode wants fused and prefill wants dequant. No single flag setting is correct for both workload types simultaneously on sm89.

## Proposal

Route each matmul call by its actual shape instead of by a single global switch. `x.shape[0]` — the row count of the activation entering `_fused_mul_mat_gguf` — is exactly "how much work is this call," not "how many requests are in flight": under vLLM v1's chunked prefill, one engine step concatenates every token across every sequence in that step into a single `x`, so a pure-decode step (bounded by `--max-num-seqs`, e.g. ≤384) and a prefill chunk (bounded by `--max-num-batched-tokens`, typically ≥2048) occupy disjoint, well-separated ranges of `x.shape[0]`.

```python
_CUSTOM_VLLM_GGUF_HYBRID = os.environ.get("CUSTOM_VLLM_GGUF_HYBRID", "0") == "1"
_CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD = int(
    os.environ.get("CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD", "1024")
)

def _fused_mul_mat_gguf(...):
    if (
        _CUSTOM_VLLM_GGUF_HYBRID
        and qweight_type in DEQUANT_TYPES
        and x.shape[0] >= _CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD
    ):
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        return x @ weight.T
    # ... existing mmvq / mmq / dequant selection unchanged below
```

1024 sits in the gap between the two regimes: comfortably above worst-case decode batch size, comfortably below the smallest realistic prefill chunk size. This is one new, independent, opt-in flag (`CUSTOM_VLLM_GGUF_HYBRID`, threshold overridable via `CUSTOM_VLLM_GGUF_HYBRID_THRESHOLD`) — it does not change default behavior, and it composes cleanly with the existing/prerequisite `CUSTOM_VLLM_GGUF_DEQUANT` flag: hybrid's check runs first and returns early for calls it claims; for calls it declines (small `x.shape[0]`, or a quant type outside `DEQUANT_TYPES`), `CUSTOM_VLLM_GGUF_DEQUANT=1` still forces dequant underneath if also set. The two flags are not meant to be combined in practice (`DEQUANT=1` alone already subsumes and outperforms hybrid for T4-style hardware, and wastes hybrid's decode win on L4-style hardware) — hybrid is the "one flag replaces per-workload manual tuning" answer specifically for sm89-class GPUs where fused-vs-dequant genuinely depends on call shape, not just device.

## Measured results with the hybrid dispatch enabled

L4 (sm89), Qwen3.5-2B, Q4_K_M:

| | fused | dequant | **hybrid** |
|---|---|---|---|
| decode tok/s @ conc32 | 872 | 674 | **852** (98% of fused-only peak) |
| long-context sustained tok/s | ~3,500 | ~8,580 | **~8,858** |
| ITL p95, long-context | — | — | **0.036 – 0.089 s** |

The hybrid path is not merely "average of the two" — it beats both single-mode configurations on long-context throughput (8,858 vs. 8,580 dequant-only), because prefill chunks and decode steps within the same server process are now each routed to their locally-optimal kernel, rather than the whole process being pinned to one choice. Decode throughput is retained at 98% of the fused-only ceiling. Quality gate: greedy decoding produced the deterministic reference completion 3/3 times with zero degenerate-token incidents across 45 requests under the hybrid path.

## Why this is a PR proposal, not a bug report

Neither existing path (fused-only, or the unconditional `CUSTOM_VLLM_GGUF_DEQUANT` flag) is *wrong* — each is a legitimate choice for a specific hardware/workload combination, and the plugin already documents dequant as opt-in for a reason. This proposal is additive: a second, independent, off-by-default flag that removes the need for an operator to know in advance whether their traffic is decode-heavy or prefill-heavy (or to run two separately-tuned server processes) on hardware where that distinction actually matters.

## Files

- `scripts/patch_gguf_prefer_dequant.py` — prerequisite; adds the unconditional dequant+cuBLAS path and its env flag.
- `scripts/patch_gguf_hybrid_dispatch.py` — this proposal; adds the shape-based dispatch on top.

Both are anchor-based source patches against the installed `vllm_gguf_plugin/quantization/linear.py`, applied in sequence (hybrid's anchors are the exact text prefer_dequant inserts, so it must run second).
