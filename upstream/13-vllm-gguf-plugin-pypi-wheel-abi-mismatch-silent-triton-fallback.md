# [Bug] PyPI wheel of `vllm-gguf-plugin` ships `_C_gguf` built against a mismatched torch ABI — silently fails to import, forcing every GGUF matmul onto the slow Triton fallback (up to 3.9x slower at low batch)

## Tóm tắt cho người không chuyên

Bản cài đặt nhanh (`pip install vllm-gguf-plugin`) đi kèm một phần lõi tăng
tốc viết bằng CUDA, nhưng bản lõi này được biên dịch cho một phiên bản
PyTorch khác với phiên bản đang chạy trên máy. Vì không khớp, nó không nạp
được — và phần mềm không hề báo lỗi hay cảnh báo gì, chỉ âm thầm chuyển sang
một đường tính chậm hơn hẳn. Người dùng vẫn thấy mọi thứ "chạy được", nhưng
tốc độ chậm hơn tới gần 4 lần mà không có cách nào biết được lý do nếu không
tự dò log CUDA. Cách khắc phục hiện tại là tự biên dịch lại gói này từ mã
nguồn (sdist) thay vì dùng bản wheel có sẵn.

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** Critical (silent performance regression — no error, no warning, default install path is affected; up to 3.9x slower for every user who does not know to rebuild from source)
**Affected versions:** vllm-gguf-plugin 0.0.4 (PyPI wheel), vllm 0.26, observed on sm89 (L4)
**Local fix:** Build from sdist with `TORCH_CUDA_ARCH_LIST=8.9` (or the target GPU's compute capability) instead of installing the prebuilt wheel; `scripts/setup_env.sh` does this automatically.
**Duplicate check:** No existing issue found in vllm-project/vllm-gguf-plugin specifically about the published wheel's `_C_gguf` extension failing to import due to a torch ABI mismatch. Distinct from vllm-project/vllm#36802 / PR #43047 (Triton shared-memory `OutOfResources` crashes on non-H100 GPUs — a different failure mode entirely; that is a hard crash, this is a silent, non-crashing fallback).

## Summary

`vllm_gguf_plugin` ships an optional compiled CUDA extension, `_C_gguf`, that provides fast native kernels (`mmvq`/`mmq`) for GGUF matmuls. When this extension fails to import — for any reason, including a torch build/ABI mismatch between the extension's build environment and the installed torch — the plugin does not raise, does not warn, and does not log anything a normal user would notice at startup. It silently falls back to a pure-Triton implementation (`ggml_mul_mat_a8_triton`) for every quantized matmul in the model, for the entire serving session.

We built the plugin from its sdist (source distribution) with `TORCH_CUDA_ARCH_LIST` pinned to the exact target GPU architecture and confirmed, via `_C_gguf`'s presence and via CUDA kernel names showing up in `torch.profiler` traces, that this produces a genuinely different code path than the PyPI wheel — and that every decode/prefill benchmark result collected before this discovery (a substantial fraction of this project's earlier measurements) was, without our knowing it, exclusively exercising the Triton fallback.

## Reproduction and measured impact

Serve the same GGUF checkpoint (Qwen3.5-2B, Q4_K_M) two ways on the same GPU (L4, sm89):

1. `pip install vllm-gguf-plugin` (PyPI wheel) — `_C_gguf` fails to import silently; all matmuls route through Triton.
2. Build from sdist with `TORCH_CUDA_ARCH_LIST=8.9` — `_C_gguf` imports successfully; native CUDA `mmvq`/`mmq` kernels are used.

Decode tok/s at concurrency 1/4/16/32, identical model/quant/hardware, only the extension-import outcome differs:

| | conc1 | conc4 | conc16 | conc32 |
|---|---|---|---|---|
| PyPI wheel (Triton fallback, silent) | ~33 | ~131 | ~489 | ~872 |
| sdist build (native `_C_gguf` active) | **130** | **254** | **482** | **856** |

(Figures for conc1 are the clearest signal — **3.9x** — because per-call kernel-launch overhead, the dimension Triton is worst at, dominates most heavily at low batch; the gap narrows at higher concurrency as raw compute saturates either kernel.)

## Why this is silent and dangerous

- No exception is raised anywhere in the import chain that a standard `vllm serve` invocation surfaces to the user.
- The server starts normally, serves requests normally, and produces numerically correct output either way — the Triton fallback is a legitimate, working implementation, just a much slower one. There is no correctness signal to alert a user that something is wrong.
- Every documented performance number for this plugin that we are aware of in the wider ecosystem is potentially affected the same way, since `pip install vllm-gguf-plugin` (the wheel path) is the only installation method documented in the plugin's own README.
- We only discovered this by independently instrumenting CUDA kernel names via `torch.profiler` while investigating an unrelated performance question, and noticing the kernel names present in a from-source build did not appear at all in a from-wheel run.

## Root cause (as far as verified without access to the wheel's exact build provenance)

We were not able to obtain the PyPI wheel's build logs to confirm the exact torch version/ABI it was compiled against. What we can state with confidence: `_C_gguf.so` fails to import under the torch version resolved by a plain `pip install vllm-gguf-plugin vllm` (both from PyPI, no pinning), and building the identical source (sdist) against the locally-resolved torch succeeds and produces a working, faster extension. This is consistent with — but not proof of — a torch ABI mismatch between the wheel's build-time torch and the install-time torch; we report the observable symptom and its confirmed workaround rather than asserting the exact build-pipeline defect, since we don't have visibility into the wheel's CI.

## Proposed fix

At minimum, `_C_gguf`'s import failure should not be silent: emit a `logging.warning` (or higher) at plugin load time stating that the native extension failed to import and that a slower Triton fallback is in use, ideally including the underlying `ImportError` text. Longer-term, publishing wheels per-torch-version (as many other CUDA-extension packages do, e.g. via `torch.__version__`-suffixed wheel tags) or documenting the sdist-build requirement prominently in the README would prevent users from unknowingly running 2-4x slower than necessary.

## Verification

Confirmed via `torch.profiler` CUDA kernel name inspection (native `mmvq_*`/`mmq_*` kernel names present only in the sdist build) and via the throughput table above, reproduced on L4 (sm89).
