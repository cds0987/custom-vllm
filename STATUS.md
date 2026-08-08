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

## Kết quả L4 (sm89) — vòng lặp tối ưu đã đóng

L4 tự chọn FlashAttention 2 (T4 kẹt TRITON_ATTN). fp8 KV cache chạy được,
miễn phí về tốc độ, +79% sức chứa (1.36M → 2.44M token) — nhưng loại trừ
lẫn nhau với FA2 (bật fp8 KV thì rơi về FlashInfer, vẫn tốt trên sm89).

**Phát hiện trung tâm — chọn kernel theo pha, không có đường thắng tuyệt đối:**

| workload | thắng | số liệu |
|---|---|---|
| decode (prompt ngắn) | Triton fused | 872 vs 674 tok/s @conc32 (1.3×) |
| prefill (prompt 12k) | dequant+cuBLAS | 8.580 vs 3.500 tổng tok/s (2.5×) |
| prefill, trần tuyệt đối | fp16 thuần | 10.950 vs 8.580 (+27%) |

Trên T4, dequant thắng decode 15-17×; trên L4 nó THUA decode 1.2-1.3×.
Kernel Triton fused không chậm bẩm sinh — chúng bệnh trên sm75 (T4→L4
nhanh lên 36× trong khi phần cứng chỉ hơn ~2×). Khuyến nghị:
sm75 → `CUSTOM_VLLM_GGUF_DEQUANT=1`; sm89+ → decode để mặc định,
workload prefill-nặng thì bật. Patch lai route theo `x.shape[0]` là
việc đáng làm nhất tiếp theo.

**Điểm gãy concurrency (prompt ngắn, GGUF fused):** đỉnh 1.922 tok/s
@conc 384; 512 đi ngang. fp16 @conc 32 là 1.227.

**Kết quả phục vụ prompt dài (12k token, LongAlign, open-loop Poisson):**

| cấu hình | trần bền | ghi chú |
|---|---|---|
| GGUF fused | ~3.500 tổng tok/s | bão hoà ngay 0.3 QPS |
| GGUF dequant | ~8.580 | 0.83 QPS, TTFT p95 10.4s |
| **fp16 + fp8 KV** | **10.950** | **1.15 QPS, 300s sạch, 0 lỗi, TTFT p50 2s** |

Cấu hình thắng: `Qwen/Qwen3.5-2B --max-num-seqs 384 --kv-cache-dtype fp8_e4m3
--max-model-len 16384 --max-num-batched-tokens 16384`, chạy 1.0 QPS cho biên
an toàn (9.653 tok/s), 1.15 QPS là mép (10.950). GPU 100% suốt — trần compute
vật lý, không phải cấu hình. So T4: 0.09 QPS → 1.15 QPS, hơn 12×.

**Bẫy benchmark đã mắc và rút lại:** burst 15-request ở 8 QPS hiện 10.8K
tok/s nhưng là hàng đợi đang phình — bài 240s cùng mức cho TTFT 69s và
1.351 request rơi. Với hệ bão hoà, con số bền là ACHIEVED, không phải
offered; chỉ bài duration dài mới phân biệt được phục vụ thật với ảo giác.

Hướng còn mở: chuyển mã GGUF → AWQ/GPTQ chạy Marlin — con đường duy nhất
vượt fp16 mà vẫn giữ 4-bit cả trên đĩa lẫn trong VRAM. Nghiên cứu xong
(cần dequant + requant RTN/GPTQ, `gptq_marlin_repack` không ăn trực tiếp
block GGUF, ước ~1-2 ngày code), chưa dựng.

## Bảng định dạng GGUF trên L4 (decode, kernel fused, conc 32)

| format | tok/s | weights | chất lượng spot-check |
|---|---|---|---|
| Q4_0 | 1158 | 1.69 GiB | PASS (sơ đồ thô, chưa stress-test) |
| UD-Q4_K_XL | 904 | 1.88 GiB | PASS sạch |
| Q4_K_M | 872 | 1.82 GiB | PASS sạch — chuẩn bảo thủ |
| Q8_0 | 839 | 2.90 GiB | PASS |
| IQ4_XS | 822 | 1.63 GiB | SOFT-FAIL: bịa "Hội An là thủ đô" |
| UD-Q2_K_XL | 805 | 1.26 GiB | PASS — 2-bit vẫn đứng vững |
| Q5_K_M | 708 | 2.05 GiB | PASS |
| Q6_K | 704 | 2.28 GiB | PASS |
| BF16 GGUF | — | hỏng | bug #16 tái hiện y hệt trên sm89 |

Hai bài học của bảng:
1. Tốc độ không đơn điệu theo bit — chi phí giải mã layout quyết định ngang
   số bit (Q4_0 phẳng thắng, Q6_K block 210 byte lệch chuẩn thua cả Q8_0).
2. Chất lượng không đơn điệu theo bit — SƠ ĐỒ quan trọng hơn bit: 2-bit trộn
   khéo (UD giữ layer nhạy cảm) thắng 4-bit trộn vụng (IQ4_XS ảo giác).

## Khuyến nghị chọn cấu hình theo workload (L4)

- Chat ngắn / latency thấp: Q4_K_M, DEQUANT unset, fp8_e4m3 KV.
  Điểm vận hành 128-256 user (mỗi user 7-12 tok/s), trần 1.922 @conc384.
- Cần decode 4-bit nhanh nhất: Q4_0 (1158) — nhớ chưa stress-test chất lượng.
- Long-context 4-bit: Q4_K_M + DEQUANT=1 — ca duy nhất bật dequant. ~8.580 tok/s.
- Long-context cần vượt 10K: fp16 safetensors + fp8 KV — 10.950 bền 300s.
- Tránh: BF16 GGUF (hỏng), IQ4_XS (rủi ro ảo giác), UD-Q2_K_XL chỉ khi
  VRAM cực hạn hẹp.
- Thắng miễn phí mọi nơi trên L4: fp8_e4m3 KV cache.

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
