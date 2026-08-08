# Trạng thái dự án

Handoff giữa các phiên. Đọc file này trước khi làm gì khác.

## Mục tiêu

Chạy và tối ưu GGUF của Qwen3.5 trên vLLM. Bản 2B dùng để lặp thử nghiệm nhanh;
đích cuối là biết cách tối ưu GGUF nói chung và scale lên các family lớn hơn.

## Phần đúng đắn — ĐÃ XONG

Qwen3.5 GGUF chạy được trên vLLM (cả 4B và 2B) sau **16 bug** được vá ở 4 lớp.
Mọi patch nằm trong `scripts/`, anchor-based, idempotent, docstring đủ chất lượng
gửi upstream. Dựng lại môi trường bằng một lệnh: `bash scripts/setup_env.sh`.

Bug đắt nhất: `find_hf_name_in_tensor_map()` nối `gguf_name + "." + suffix` vô điều kiện,
nên tham số không có hậu tố `.weight`/`.bias` bị đăng ký thành `blk.N.ssm_a.` (thừa dấu chấm).
Nó **im lặng** vì tên HF vẫn nằm trong `values()` nên kiểm tra "Failed to map GGUF parameters"
vẫn qua. Hậu quả: `A_log` của cả 24 lớp gated-delta-net giữ nguyên 0 → `-exp(0) = -1`
→ sụp về token 0 (`!!!!`). Ảnh hưởng mọi kiến trúc SSM/hybrid.
Xem `scripts/patch_gguf_empty_suffix.py`.

Trọng số đã đối chiếu số học với checkpoint HF gốc: `A_log`, `dt_bias`, `conv1d`,
các norm khớp tuyệt đối; phần lượng tử sai số 0.004–0.07 đúng mức nhiễu quant.

## Phần hiệu năng — số liệu đã đo trên T4

Prompt ngắn, 64 token output, closed-loop, concurrency 1/4/16/32:

| cấu hình | 1 | 4 | 16 | 32 |
|---|---|---|---|---|
| GGUF Q4_K_M, kernel Triton fused | 0.92 | 3.44 | 12.1 | 20.5 |
| GGUF + `CUSTOM_VLLM_GGUF_DEQUANT=1` | 13.7 | 52.4 | 193.2 | 339.8 |
| fp16 safetensors (`Qwen/Qwen3.5-2B`) | 55.1 | 77.3 | 277.2 | 832.4 |

**Luôn đặt `CUSTOM_VLLM_GGUF_DEQUANT=1`**, không có nó thì số liệu vô nghĩa.
K-quant thuộc cả `MMVQ_QUANT_TYPES` lẫn `MMQ_QUANT_TYPES` nên nhánh `DEQUANT_TYPES`
có sẵn trong plugin nhưng không bao giờ tới được. Xem `scripts/patch_gguf_prefer_dequant.py`.

Prompt dài (LongAlign-10k, 12k token, 128 output, open-loop Poisson):
server bão hoà ở **~0.09 QPS**, prefill trần **~1000 tok/s**.

| | 0.1 QPS | 0.2 QPS | 0.4 QPS |
|---|---|---|---|
| chunk mặc định — ITL p95 | 1.83 s | 1.61 s | 1.77 s |
| chunk 512 — ITL p95 | **0.65 s** | **0.66 s** | **0.65 s** |
| chunk mặc định — TTFT p50 | 9.8 s | 12.4 s | 26.0 s |
| chunk 512 — TTFT p50 | 13.6 s | 15.4 s | 30.6 s |

Quy luật quan sát được: `ITL p95 ≈ max_num_batched_tokens / prefill_tok_per_s`.

## Bẫy đo lường — đã mắc, đừng lặp lại

1. Giết server bằng danh sách PID từ `nvidia-smi --query-compute-apps` là **sai**.
   Chỉ EngineCore giữ VRAM; APIServer sống sót và giữ cổng 8000, nên server mới
   chết với `OSError: [Errno 98] Address already in use` còn benchmark thì âm thầm
   bắn vào server **cũ**. Dùng `pkill -9 -f "vllm serve"; pkill -9 -f VLLM::EngineCore`.
2. `pgrep -fc "bench_serving.py"` khớp cả chính câu lệnh chạy nó. Dùng `"[b]ench_serving"`.
3. Trước mỗi sweep phải kiểm tra hai điều: `non-default args` trong log server có đúng
   cờ đang thử, và `vllm:time_to_first_token_seconds_count` trên `/metrics` bằng 0.
4. Runtime Colab bị recycle khoảng mỗi giờ. Dựng lại bằng `setup_env.sh`, đừng than.

## Ngõ cụt trên T4 (sm75) — đừng phí giờ GPU

`--kv-cache-dtype fp8*` (chặn cứng, cần SM89+), KV bfloat16 (cần SM80+),
backend FlashInfer, cascade attention, TurboQuant KV, chỉnh `--block-size`,
chỉnh `--mamba-cache-dtype` (state của gated-delta-net là O(1) theo request,
~25 MB, bị KV attention 512 MB/request ở 16k áp đảo).

## Trên L4 (sm89) thì khác hẳn

Toàn bộ danh sách trên **sống lại**: fp8 KV cache chạy được, bfloat16 được,
FlashAttention 2 được, FlashInfer được, kernel Marlin cho AWQ/GPTQ đúng sân,
VRAM 24 GB thay vì 15 GB. Ba hướng chính cho L4:
1. `--kv-cache-dtype fp8_e4m3` / `fp8_e5m2` — giảm nửa KV
2. FlashAttention 2 thay TRITON_ATTN
3. Chuyển mã GGUF → AWQ/GPTQ rồi chạy Marlin — con đường duy nhất vượt fp16
   mà vẫn giữ 4-bit cả trên đĩa lẫn trong VRAM

## Công cụ đo

- `scripts/setup_env.sh` — dựng lại môi trường từ đầu, một lệnh
- `scripts/bench_load.py` — closed-loop, quét concurrency, prompt ngắn
- `scripts/bench_swebench.py` — SWE-bench, hai chế độ single / concurrent fairness
- `scripts/bench_serving.py` — **open-loop Poisson theo QPS**, LongAlign-10k,
  prompt dài, cờ bão hoà, chỉ số công bằng, có subcommand `compare`

Closed-loop giấu điểm bão hoà vì tải chào giảm theo khi server chậm đi;
open-loop mới tìm được điểm gãy.

## Việc còn dở

- Cụm T4 đang chạy vòng tối ưu đa tầng: `int8_per_token_head` KV, chunk 256,
  tắt prefix caching, rồi đào sâu xuống Python hot path và kernel Triton.
- Chưa benchmark 22 định dạng GGUF của repo 2B.
- Nhánh unquantized (BF16 GGUF) của plugin hỏng: nạp 0.03 GiB rồi chết ở
  `split(size=(s72, 0))`. Đây là bug thứ 16, chưa sửa.
- Chưa gửi issue/PR lên vllm và vllm-gguf-plugin.
