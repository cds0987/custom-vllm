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
5. **Prefix caching thổi phồng số prefill** (user chỉ ra 2026-08-08). vLLM 0.26 bật
   automatic prefix caching MẶC ĐỊNH. `bench_serving.py` cycle qua 500 prompt
   LongAlign; lượt đo nào vượt 500 request là lặp nguyên văn prompt cũ → prefill
   ăn cache gần như miễn phí → tổng tok/s ảo. Còn thêm hậu tố `(#i)` vào CUỐI
   prompt (bench_load cũ) không chống được gì vì cache khớp theo TIỀN TỐ block.
   Đã sửa: bench_serving gắn tag `[req N]` duy nhất vào ĐẦU mỗi prompt,
   bench_load chuyển `(#i)` lên đầu. Quy tắc: server benchmark luôn dựng với
   `--no-enable-prefix-caching`; báo cáo phải ghi rõ caching ON/OFF và số prompt
   distinct so với tổng request. **Mọi số long-context đo trước fix này (kể cả
   champion 10.950 tok/s) mang dấu hỏi cho tới khi tái kiểm với caching off** —
   mức độ thổi phồng phụ thuộc bao nhiêu request vượt ngưỡng 500 ở từng level.
   Test câu ngoài dataset bằng cách "kéo dài 1 câu" cũng vô hiệu vì cùng lý do.

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
block GGUF, ước ~1-2 ngày code).

**Cập nhật:** `scripts/transcode_gguf_to_gptq.py` đã dựng xong và pass dry-run
CPU-only trên `unsloth/Qwen3.5-2B-GGUF:Q4_K_M` (không đụng Colab/GPU — máy dev
này không có GPU). RTN W4 group128 sym vào đúng layout GPTQ vllm cần
(qweight/qzeros/scales/g_idx), quantise q/k/v/o_proj, gate/up/down_proj, và
5 phép chiếu tuyến tính GDN (in_proj_qkv/z/a/b, out_proj — xác nhận từ
`qwen_gdn_linear_attn.py` rằng vllm đã có sẵn đường GPTQ/AWQ cho các layer
này); conv1d/A_log/dt_bias/norm/embedding giữ fp16 đúng theo research trước.
Sai số RTN đo trên tensor thật ~10-13% L1 (cao hơn dải K-quant gốc 0.4-7%,
vì min/max scale RTN nhạy outlier hơn K-quant per-superblock; đã thêm clip-
ratio grid search tự-hiệu-chỉnh trên chính tensor, không cần calibration
data, giảm được ~2 điểm %). Đã tìm thấy sẵn AWQ 4-bit cho 4B/9B trên Hub
(QuantTrio, cyankiwi, mssfj) — không có bản 2B nào, transcode 2B của ta vẫn
là checkpoint 4-bit-trên-Marlin duy nhất cho size này. Việc còn lại cần GPU
L4 thật: serve + `bench_load`/`bench_serving`/quality gate, xem transcode
này có lọt vào khoảng 8.6K–10.95K tok/s hay không — xem docstring của script
để có runbook đầy đủ.

## Kiểm chứng scale 9B trên L4 (unsloth/Qwen3.5-9B-GGUF:Q4_K_M, hybrid)

| chỉ số | 2B | 9B đo được | dự đoán |
|---|---|---|---|
| weights đĩa / VRAM | 1.82 GiB | 5.29 / 8.5 GiB | 6–8 |
| decode conc32 | 852 | **268** | 294 theo byte-ratio (lệch 9%) ✓ |
| KV (fp8) | 2.44M tok | 437K tok (~36 phiên 12k) | — |
| long-ctx tổng | 8.858 | **≥1.068** (floor, chưa chạy duration ở mép) | 1.900–2.200 ✗ |

Hai bài học:
1. **Decode scale theo BYTE weights thật, không theo số tham số.** Q4_K_M 9B
   chỉ nặng 2.9× bản 2B (không phải 4.5×) vì tỷ trọng embedding giảm khi
   model lớn — dùng đúng proxy thì định luật bandwidth khớp trong 9%.
2. **Prefill scale kém hơn tuyến tính theo FLOPs** — 1.068–1.600 so với
   dự đoán ~2.000. Nghi phạm: hiệu suất kernel ở tensor lớn; và điểm đo
   sạch duy nhất (0.1 QPS) có thể chưa chạm trần thật. Cần một bài
   duration ở mép 0.15–0.2 QPS trước khi chốt số production.

Ước tính production 9B/L4 (sửa lại từ số đo): ~5–8 tài liệu 12k/phút,
chat ~32 user × 8 tok/s. Quality gate PASS (lưu ý model reasoning cần
max_tokens ≥300 mới thấy câu trả lời sau chuỗi suy nghĩ).

## Profile kernel trên L4 (torch.profiler, eager, 2B — % CUDA time)

| bucket | GGUF decode | GGUF prefill | fp16 decode |
|---|---|---|---|
| matmul/GEMM | **86.6%** | **47.9%** | **77.6%** |
| norm/elementwise | 6.6% | 30.9% (eager, chưa fusion) | 7.6% |
| GDN/fla | 4.7% | 2.1% | 11.7% |
| attention | 0.4% | 8.7% | 0.6% |
| tổng CUDA ms (cùng workload) | 2161 | 2865 | **1570** |

Ba kết luận:
1. **GDN không phải nút thắt** (2–12%) — bác giả thuyết đồng thuận của cả
   9 nhánh nghiên cứu đọc-code. Kernel GDN viết tốt; patch Ada shmem vì
   thế chỉ đáng vài phần nghìn tổng, đã hạ cấp.
2. **Kernel GGUF fused tốn NHIỀU CUDA time hơn fp16 làm cùng việc**
   (1871 vs 1219 ms GEMM): 4-bit hiện chỉ thắng ở kinh tế băng thông
   (1.82 vs 4.25 GiB), thua ở hiệu suất compute. Marlin là mảnh ghép
   đúng: int4 với kernel tensor-core — ăn cả hai đầu.
3. **q6_k_gemm_kernel một mình chiếm 34.3% decode** — Q4_K_M là scheme
   trộn, vài tensor giữ Q6_K, và kernel Q6_K chậm nhất họ (khớp format
   sweep: model Q6_K 704 < Q8_0 839 dù nhẹ hơn). Sinh ra patch repack
   Q6_K→Q8_0 lúc nạp.
Lưu ý đo: key_averages() đếm trùng custom op (op cha báo self-time =
tổng kernel con) — phải lọc hàng thuần-GPU; đối soát 2161/2160 ms khớp.

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

## Patch lai kernel-dispatch — ĐÃ ĐO, ăn cả hai đầu

`scripts/patch_gguf_hybrid_dispatch.py` (`CUSTOM_VLLM_GGUF_HYBRID=1`,
ngưỡng `x.shape[0] >= 1024`): decode đi fused, prefill đi dequant+cuBLAS.
Đo trên L4, Q4_K_M:

| | fused | dequant | **hybrid** |
|---|---|---|---|
| decode conc32 | 872 | 674 | **852** (giữ 98%) |
| long-ctx tổng tok/s | ~3.500 | ~8.580 | **~8.858** |
| ITL p95 long-ctx | — | — | **0.036–0.089 s** |

Một cờ duy nhất thay cho khuyến nghị "bật/tắt DEQUANT theo workload".
Quality gate PASS (greedy 3 lần "Hà Nội", degen=0 cả 45 request).

## Cô lập chat khỏi prefill tài liệu — hai instance thắng, MPS chưa

MPS SM-pinning KHÔNG hoạt động với vLLM hiện tại: `CUDA_MPS_*` tới
APIServer nhưng không truyền vào EngineCore (spawn multiprocessing) —
muốn dùng phải patch worker-bootstrap của vLLM.

Fallback đã đo — hai instance thường (mỗi cái gpu-mem-util 0.40,
driver time-slicing) so với một instance trộn lẫn, phía chat khi phía
tài liệu chịu tải:

| | 1 instance trộn | 2 instance |
|---|---|---|
| TTFT max | 1.985 s | **0.327 s** (6×) |
| ITL max | 1.257 s | **0.083 s** (15×) |

Khuyến nghị production: tách chat/tài liệu ra hai instance trên cùng
card — rẻ (weights 2B chỉ 1.8 GiB×2), không cần MPS, đuôi trễ sập 6–15×.

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
