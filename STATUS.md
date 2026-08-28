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
   EngineCore mồ côi có thể thoát lưới `pkill -f` (lệch title/argv) — khi đó
   `kill -9 <pid>` đích danh theo nvidia-smi. Trên Colab, `subprocess.run(
   capture_output=True)` với script bash có background job sẽ TREO vô hạn
   (con thừa kế pipe) — dùng `subprocess.Popen` không chặn.
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
   distinct so với tổng request. Test câu ngoài dataset bằng cách "kéo dài
   1 câu" cũng vô hiệu vì cùng lý do.
   **Tái kiểm 2026-08-08 (TEST R): điểm champion KHÔNG bị thổi phồng** — ở
   1.0-1.15 QPS × 300s mỗi lượt chỉ bắn 279-325 request < 500 prompt, lỗi
   chưa từng kích hoạt tại điểm đó; đo lại với caching off + tag fix cho
   11.211,6 tok/s (cao hơn nhờ gộp chiến thắng TEST 0b). Các mức QPS cao
   (>500 request/lượt) trước fix vẫn không đáng tin.

## Ngõ cụt trên T4 (sm75) — đừng phí giờ GPU

`--kv-cache-dtype fp8*` (chặn cứng, cần SM89+), KV bfloat16 (cần SM80+),
backend FlashInfer, cascade attention, TurboQuant KV, chỉnh `--block-size`,
chỉnh `--mamba-cache-dtype` (state của gated-delta-net là O(1) theo request,
~25 MB, bị KV attention 512 MB/request ở 16k áp đảo).

**KV dưới 8-bit: ngõ cụt trên MỌI GPU Ada, ba nguồn độc lập đồng quy
(2026-08-09):** TRT-LLM (FP4 KV = Blackwell-only, int4 KV không tồn tại),
turboquant-vllm (TQ4 tự đo trên RTX 4090/sm89: decode chậm hơn SDPA 8-9×,
TPOT +201%, chính tác giả kết luận "architectural mismatch"), CUTLASS
(int2b_t chỉ là khai báo kiểu, không kernel nào dùng). fp8 KV là sàn cuối.

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
| fp16 + fp8 KV | 10.950 | 1.15 QPS, 300s sạch, 0 lỗi, TTFT p50 2s |
| **GGUF hybrid + fp8 KV, bỏ fp32 SSM** | **11.211,6** | **TEST R 2026-08-08: caching off, tag fix, 325 req/500 prompt distinct, quality 3/3** |

**CHAMPION TOÀN DỰ ÁN 2B (TEST 10, 2026-08-09): AWQ W4A16 tự tạo + Marlin.**
Quantize bằng `scripts/quantize_awq_2b.py` (llm-compressor, AWQModifier +
W4A16, ignore lm_head + linear_attn, calibration ultrachat 256 mẫu), sửa
checkpoint bằng `scripts/fix_qwen35_hf_checkpoint.py` (strip prefix
`model.language_model.` + bỏ mrope_section — HF Qwen3.5 nào không-GGUF cũng
cần), serve thường với fp8 KV. Thắng hoặc hoà MỌI trục so GGUF champion:
- decode conc32: **1859 tok/s** (2.17× GGUF 856, 1.22× fp8-B 1524)
- long-ctx prefill: 10.637 vs 10.528 (+1%)
- **TTFT p50: 2.59s vs 4.91s (−47%, tốt nhất mọi cấu hình từng đo)**
- SWE-bench ppl: 3.724 vs 3.558, ratio 1.047 → PASS (<1.10)
- 5 câu dò: parity mọi chế độ (kiểm bằng control kép — xem mục phương pháp)
Checkpoint 3.1GB nằm Colab-local, rebuild sau recycle bằng script trên.
GGUF ba-đường vẫn là lựa chọn khi bắt buộc format GGUF (llama.cpp interop).

**Ba phát hiện probe-validity (2026-08-09) — quy tắc chấm cổng chất lượng:**
1. Câu "kể 3 trái cây" sập loop NGAY Ở BF16 9B — điểm hút của base model.
2. 2B dưới thinking-mode loop trên Nguyễn Du/địa lý ở MỌI precision.
3. 2B chế-độ-thẳng ảo giác tự tin câu recall đa-fact ở MỌI format quant
   (AWQ bịa "Núi Phú Quốc", GGUF bịa "Núi Bà Đen" — cùng bệnh khác triệu chứng).
→ Quy tắc: model nhỏ chấm theo CONTROL cùng-chế-độ cùng-gốc, không chấm
thang đúng/sai tuyệt đối; mọi probe mới phải control-validate trước khi
thành cổng. Giao thức chuẩn: 5 biến thể/probe + control full-precision.

**TEST 11a (sweep max-num-batched-tokens, long-ctx 16K):** 16384 và 8192
hoà throughput/TTFT nhưng 8192 cho ITL p95 tốt gấp đôi (0.73 vs 1.41) —
đề xuất mặc định mới 8192. 2048 = điểm trade-off ITL (TTFT tệ, bão hoà nhẹ).
512 SẬP ở 16k context (2.437 tok/s, 185/279 chết) — bài học: chunk scale
theo context/prefill_rate, khuyến nghị 512 thời T4 không mang sang L4 16k.

**Champion GGUF (TEST 8d, 2026-08-09):** GGUF Q4_K_M +
`CUSTOM_VLLM_GGUF_HYBRID=1` + `CUSTOM_VLLM_GGUF_TRITON_MID=1` (dispatch
3 đường: M nhỏ → CUDA mmvq của llama.cpp, M trung → Triton mmq, M>=1024 →
dequant+cuBLAS) + `--kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching`,
KHÔNG mang `--mamba-ssm-cache-dtype float32` (TEST 0b: bỏ cờ = +6.9%
decode, sạch 3/3; TRT-LLM đúng hướng, biên độ ~7% chứ không ~20%).
YÊU CẦU: plugin build từ sdist với TORCH_CUDA_ARCH_LIST=8.9 (setup_env.sh
tự làm) — wheel PyPI lệch ABI torch nên _C_gguf không load và mọi thứ rơi
về Triton (TEST 8 phát hiện; đó là lý do mọi số GGUF cũ đều là Triton).
Số: decode 130/254/482/856 @conc1/4/16/32 (3.9× conc1 so champion cũ),
prefill 10.527,9 tok/s @1.0 QPS, TTFT p50 4.91s, quality 3/3. Bài mở:
crossover mmvq_safe ở conc4 (CUDA mmq thuần đạt 330 vs 254 — còn headroom). 4-bit trên VRAM giờ VƯỢT
kỷ lục fp16 cũ — lưu ý so sánh chéo cấu hình: 10.950 cũ là fp16 safetensors,
11.211,6 mới là GGUF hybrid; fp16 + bỏ-fp32-SSM chưa đo lại (có thể còn cao
hơn nữa — bài mở). 1.0 QPS là biên an toàn (10.496, chưa bão hoà, TTFT p50
5s); 1.15 QPS là mép bão hoà (TTFT p50 13.4s). GPU 100% suốt — trần compute
vật lý. So T4: 0.09 QPS → 1.15 QPS, hơn 12×.

**Bẫy benchmark đã mắc và rút lại:** burst 15-request ở 8 QPS hiện 10.8K
tok/s nhưng là hàng đợi đang phình — bài 240s cùng mức cho TTFT 69s và
1.351 request rơi. Với hệ bão hoà, con số bền là ACHIEVED, không phải
offered; chỉ bài duration dài mới phân biệt được phục vụ thật với ảo giác.

**fp8 W8A8 đã được cứu (test 1b, 2026-08-08):** `--quantization fp8_per_tensor`
với ignore=`["in_proj_ba"]` (tối thiểu, theo ModelOpt) HOẶC
`["linear_attn","lm_head"]` (bảo thủ, theo unsloth) — cả hai sạch 3/3 hai câu
dò, decode 1524-1526 tok/s @conc32 = 99.5% fp8 thô (1534), gấp ~1.8× GGUF
hybrid. Toàn bộ độ nhạy fp8 của Qwen3.5 nằm ở đường GDN (`in_proj_ba`);
khuyến nghị dùng danh sách tối thiểu. Lưu ý: `--quantization fp8` KHÔNG nhận
`--quantization-config` — phải dùng `fp8_per_tensor` (online, calibration-free).
Long-context của fp8 chưa phân thắng bại với champion (quét rate đang chạy).

**`CUSTOM_VLLM_GGUF_REPACK=1` — ngõ cụt, khoá cứng lại (2026-08-09):** repack
Q6_K→Q8_0 lúc nạp (nhắm q6_k_gemm_kernel, 34.3% CUDA time decode — xem mục
"Profile kernel trên L4" bên dưới) crash trên L4 khi chồng lên ba-đường
champion (`HYBRID=1`+`TRITON_MID=1`): `RuntimeError: mat1 and mat2 shapes
cannot be multiplied (16384x6144 and 7936x2048)` ở bước warmup đầu tiên
(prefill). Đã root-cause bằng số học độc lập với plugin: `down_proj` của
Qwen3.5-2B (hidden=2048, intermediate=6144) là đúng tensor Q6_K theo quy ước
Q4_K_M của llama.cpp. Repack đúng byte cho K=6144 dạng Q8_0 (6144÷32×34 =
6528 byte), nhưng đọc lại 6528 byte đó bằng layout Q6_K (÷210×256) ra đúng
**7936** — khớp chính xác cạnh mat2 trong lỗi. Đây là bằng chứng dữ liệu
(qweight bytes) đã repack đúng nhưng TAG (`qweight_type`) tại điểm tiêu thụ
matmul vẫn đọc kiểu gốc Q6_K — desync tag/data, cùng họ với bug embedding đã
vá trước đó (docstring `patch_gguf_repack_q6k.py`, mục "Embedding/lm_head
exclusion"). Không dò ra được đúng dòng gây desync bằng đọc tĩnh mã plugin
(nghi ba điểm trong `quantization/params.py`/`linear.py`: vật liệu hoá lại
`GGUFWeightTypeParameter`, `apply()` đọc type từ hai thuộc tính khác nhau
tuỳ có shard hay không, và `_create_padded_weight_param` có thể thay hẳn
Parameter `qweight` mà không đụng `qweight_type`) — máy dev này không có GPU
nên không dò runtime được (xem mục "Cập nhật" ở trên). Vì desync có thể xảy
ra ở BẤT KỲ nơi nào đọc `qweight_type` (kể cả kernel CUDA/Triton mmq/mmvq mà
patch này nhắm tới tăng tốc — chúng không tự kiểm tra shape như nhánh dequant
nên có thể âm thầm tính sai thay vì crash), quyết định: khoá cứng
`CUSTOM_VLLM_GGUF_REPACK` — biến này giờ luôn raise `RuntimeError` rõ ràng
lúc khởi động plugin, bất kể có bật `HYBRID`/`DEQUANT` hay không. Xem
`scripts/patch_gguf_repack_q6k.py` mục docstring "HARD GUARD" để có toàn bộ
phép tính và điều kiện cần để mở khoá lại (audit từng nơi đọc `qweight_type`
hoặc thêm assertion xác nhận byte width khớp `GGML_QUANT_SIZES[qweight_type]`
tại mọi điểm tiêu thụ, chạy thật trên GPU để xác nhận không bao giờ kích hoạt).

**AWQ-Marlin xác nhận trên sm89 (test 2):** QuantTrio/Qwen3.5-4B-AWQ serve
sạch sau patch chữ ký (`patch_gguf_override_signature.py`), log chọn
`MarlinLinearKernel` tự động. Decode 757 tok/s @conc32 cho model 4B — per-byte
THẮNG rõ Triton GGUF fused (2B được 852). Tín hiệu mạnh cho bài GPTQ 2B
transcode: nếu chất lượng sống sót nhiễu RTN, decode 2B qua Marlin có thể
vượt xa 852. Quality PASS (model reasoning, trả lời trong chain-of-thought,
cần >200 token).

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

## Thang bit thấp 9B (TEST 9, 2026-08-09) — Q4_K_M là SÀN, dưới nữa mất cả hai

Chạy trên champion 3 đường (CUDA _C_gguf active), cổng 5 câu dò + control:

| bậc | đĩa | decode conc32 | KV (16k) | phán quyết cuối |
|---|---|---|---|---|
| Q4_K_M | 5.68GB | **274.3** | 435K tok | **PASS sạch — sàn 9B** |
| UD-Q3_K_XL | 5.05GB | 235.7 | — | FAIL (còn chút mơ hồ, xem dưới) |
| Q3_K_M | 4.67GB | (bỏ đo) | — | FAIL thật (địa lý non-term, control pass câu này) |
| UD-Q2_K_XL | 4.12GB | 219.1 | 588K tok | FAIL thật (bộ 5-biến-thể IF: 1/5 vs control 4/5) |
| UD-IQ2_M | 3.65GB | 166.5 | — | FAIL thật (Nguyễn Du + địa lý non-term) |

Ba định luật rút được:
1. **Dưới Q4_K_M không mua được gì trên stack này**: decode GIẢM đơn điệu
   khi bit giảm (274→236→219→167) — layout/kernel thắng byte count; IMATRIX
   còn tệ hơn vì chỉ route mmvq/dequant (mất lane Triton-mid). Động cơ duy
   nhất của 2-bit là KV headroom (588K vs 435K) — không đủ bù chất lượng.
2. **Chữ ký lỗi bit-thấp của model reasoning là NON-TERMINATION, không phải
   sai fact**: kiến thức + số học sống tới tận 2-bit; thứ chết đầu tiên là
   khả năng CHỐT đáp án sau chuỗi suy luận (lặp tự-kiểm-chứng, tự-sửa-đổi
   bất tận). Cổng chất lượng cho family này bắt buộc phải chứa probe
   termination/format-ràng-buộc, không chỉ câu hỏi kiến thức.
3. **Mọi câu dò phải control-validate ở full precision trước khi làm cổng**:
   câu "kể 3 trái cây" sập vòng lặp NGAY Ở BF16 (điểm hút thoái hoá của
   base model) — suýt tạo án oan cho cả thang. Giao thức chuẩn từ nay:
   5 biến thể/probe + control bf16, bậc chỉ FAIL nếu kém RÕ RỆT so control.
Mơ hồ còn lại: UD-Q3_K_XL rớt duy nhất câu trái cây (câu đã bị vô hiệu) —
chưa re-grade bằng bộ 5-biến-thể vì đã đủ trả lời câu hỏi chiến dịch;
nếu cần 3-bit thì chạy lại bộ 5-biến-thể trước khi dùng.
Khuyến nghị production 9B: **ở lại Q4_K_M**. Đường "9B nhỏ hơn nữa" chuyển
sang AWQ/GPTQ mixed-precision (giữ lớp nhạy bit cao có kiểm soát).

## TEST 12 (2026-08-09): W4A16 9B thắng — AWQ/Marlin hạ GGUF ở CẢ HAI cỡ

`RedHatAI/Qwen3.5-9B-quantized.w4a16` (có sẵn Hub, không cần tự quantize),
fp8 KV, caching off, so với sàn GGUF Q4_K_M 9B (TEST 9 bậc 1):
- decode conc 1/4/16/32: 29.5/107.0/362.7/**562.6** tok/s — **2.05×** @conc32
- long-ctx 0.1 QPS: 1.433 vs 1.416 tok/s — hoà (+1.2%)
- 5 câu dò: PASS (địa lý loop = bệnh nền đã biết, chấm theo control)
- SWE-bench ppl: **5.158 vs 5.648 GGUF — ratio 0.913, candidate TỐT HƠN baseline**
Vận hành: checkpoint compressed-tensors quảng cáo max_position_embeddings
262144 → PHẢI set `--max-model-len 16384` tường minh kẻo KV-too-small;
mamba-cache ceiling ở 9B: max_num_seqs ≤ số block log báo (299 @0.90 util).

**KẾT LUẬN CHIẾN DỊCH — công thức chuẩn cho family Qwen3.5 trên L4:**
| cỡ | khuyến nghị #1 | decode conc32 | ghi chú |
|---|---|---|---|
| 2B | AWQ W4A16 tự tạo + Marlin | 1859 | quantize_awq_2b.py, TTFT p50 2.59s |
| 9B | RedHatAI W4A16 + Marlin | 563 | tải Hub, ppl còn tốt hơn GGUF |
| mọi cỡ, bắt buộc GGUF | Q4_K_M + dispatch 3 đường | 856 (2B) / 274 (9B) | sàn chất lượng là Q4_K_M, đừng xuống thấp hơn |
Cộng: fp8 KV luôn bật (sm89+), chunk 8192, prefix-caching off khi benchmark,
không cờ fp32 SSM, plugin build sdist sm_89. Dưới 4-bit weights: mất cả
chất lượng lẫn tốc độ (thang TEST 9) — 4-bit LÀ đáy thực dụng trên Ada.
Spot-check FP8-dynamic 9B (RedHatAI): 467.6 tok/s @conc32 — thắng GGUF
(1.7×) nhưng thua W4A16 (563); mamba-cache block ceiling 172 ở footprint
8-bit (max_num_seqs 128) — quy tắc: LUÔN đọc số block trong log trước khi
đặt max_num_seqs cho 9B. Không có trục nào W4A16 thua → không chạy full gate.

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

## TASK F (2026-08-09): kịch bản shared-prefix 32K/65K — ĐẬU, đúng như thiết kế

Cấu hình: 9B RedHatAI W4A16, `--max-model-len 32768 --enable-prefix-caching
--mamba-cache-mode align`, fp8 KV. **Bẫy cấu hình mới**: trong build vLLM 0.26
này prefix caching KHÔNG bật mặc định (log "Mamba cache mode is set to 'none'
when prefix caching is disabled" bắn ra dù không truyền cờ tắt) — phải truyền
`--enable-prefix-caching` tường minh, đừng tin mặc định.

- **Correctness (cổng quyết định)**: cùng prompt (prefix 30K + suffix), temp=0,
  output cache-hit **byte-identical** với output cold — ở CẢ 32K và 65K.
  Hybrid GDN + APC align-mode đúng trên checkpoint này, không lỗi state-restore
  dù mode vẫn gắn nhãn experimental.
- **TTFT warm** (N=10, suffix unique 2000-2500 tok): p50=1,42s p95=1,43s
  (cold 13,98s → warm 2,55s, 5,5×). Tốt hơn dự phóng 1,7-2s.
- **Tải**: conc8 wall 10,86s / conc16 14,79s / conc32 24,51s — sạch, 0 preemption.
- **Hit rate prefix cache**: 95,3% (2,80M/2,94M token) — prefix 30K chung áp đảo.
- **VRAM/KV**: 19,3/23,0 GiB; KV 316.469 token, `kv_cache_max_concurrency`≈9,66
  phiên 32K ĐẦY — nhưng với prefix cache chung, mỗi request chỉ tốn KV cho
  ~2-2,5K suffix riêng, nên số request đồng thời thực tế cao hơn nhiều.
- **65K**: nạp sạch (max-num-seqs 128), prefix 60K warm-up 26,93s
  (~2.228 tok/s hiệu dụng — nhanh hơn trần 1.433 đã lập, ghi nhận chưa điều tra),
  cache-hit 1,81s, output identical. KV 378.019 token, ≈5,77 phiên đầy.

Kết luận: front-load system-prompt/skills một lần cho pool phiên biến workload
từ prefill-bound thành decode/cache-bound — TTFT warm ~1,4s bất kể context nền
30-60K. Đây là pattern production hợp lệ trên L4. Không có bug upstream để báo.

## TASK G (2026-08-09): GPTQ nén cả GDN — NHANH NHẤT chiến dịch, chất lượng WARN

Chuỗi: quantize_gptq_9b.py --quantize-gdn (in_proj_qkv+z int4 g32,
in_proj_b+a int8 g128 — mỗi cặp merge đồng nhất scheme) → fix checkpoint
(922/923 keys) → serve với patch_vllm_gdn_quant_load (anchor đã sửa 777c084).

- Quantize: con số "13 phút" báo lần đầu là SAI (số đọc nhầm, log gốc mất
  theo runtime recycle). Đo lại bằng timestamp ở run G2a: **~297s/layer ổn
  định × 33 layer ≈ 160-165 phút** — khớp dự kiến 90-120+ ban đầu. Chi phí
  là calibration per-sample (256 forward/layer, ~1s/it, GPU util chỉ ~12% —
  latency-bound chứ không compute-bound; int4 hay int8 không đổi giá).
  Muốn nhanh hơn ở vòng sau: giảm --num-samples 256→128 là chia đôi giờ,
  đổi bằng rủi ro chất lượng chưa kiểm. Checkpoint **7,5GB** (không phải
  5,5-6 — cặp b/a nằm int8). Đã xóa base bf16 19GB lấy chỗ (phải tải lại
  ở G2 — giữ base trên disk nếu còn vòng quantize kế tiếp).
- **Loader patch chạy thật**: log "Using MarlinLinearKernel for
  CompressedTensorsWNA16" trên cả shard GDN, không RuntimeError/Assert.
  Lần đầu tiên checkpoint Qwen3.5 nén-toàn-phần (cả 75% layer GDN) nạp
  và serve được trên vLLM.
- **Speed: conc1 37,71 (+27,8% vs 29,5) | conc32 768,47 (+36,5% vs 563)** —
  vượt xa mốc thắng 620. Cấu hình 9B nhanh nhất chiến dịch.
- **Quality: ppl SWE-bench 5,9646 vs champion 5,158 → ratio 1,156 = WARN**
  (băng: PASS<1,10 / WARN<1,25 / FAIL>=1,25). Chưa đủ chuẩn thay champion
  theo luật "không bypass chất lượng". Caveat: so với scalar cũ trong
  STATUS.md, chưa chạy compare đối chứng cùng runtime.
- Bẫy môi trường mới: kernel session mới của Colab không tự source
  /tmp/vllm_env.sh → ImportError libcudart.so.13; phải export lại tay.
- TASK G2a (2026-08-09, uniform int8 g128 cả 4 in_proj, checkpoint 8,0GB,
  baseline RedHatAI tái tạo TƯƠI cùng runtime + compare chuẩn, 99 đề):

      config                        conc1   conc32   ppl      ratio    verdict
      RedHatAI W4A16 (champion)     29,5    563      5,169    1,00     PASS
      G  (int4 qkv/z + int8 b/a)    37,7    768      5,9646   1,1541   WARN
      G2a (int8 toàn bộ GDN)        35,4    738      5,9618   1,1534   WARN

  **Giả thuyết "int4 là thủ phạm" SAI**: int8 không cứu được gì (ratio
  1,1534 vs 1,1541 — không phân biệt thống kê) mà còn chậm hơn G. Kết luận
  mạnh: chi phí ~15% ppl nằm ở việc CHẠM VÀO GDN in_proj bằng GPTQ, không
  phụ thuộc bit-width/group-size trong dải đã thử. Champion GIỮ NGUYÊN
  RedHatAI W4A16. G/G2a là "chế độ turbo" hợp lệ nếu use case chấp nhận
  đổi ~15% ppl lấy +31-36% conc32 — quyết định thuộc người vận hành.
  - Manh mối nhánh chất lượng còn mở: GGUF Q4_K_M (llama.cpp tự nén, KHÔNG
    calibration) nén cả GDN mà vẫn qua cổng chất lượng (TEST 9) → nghi vấn
    chuyển sang chính calibration/Hessian của GPTQ trên input linear_attn,
    không phải bản thân việc nén GDN. Phương án cấy GDN từ GGUF→int8 Marlin
    (transcoder TASK I/K) là nước thử kế tiếp nếu đánh tiếp nhánh này.
  - Vận hành: giữa run bị Colab FULL recycle (mất /content + HF cache);
    khôi phục = re-clone + setup_env.sh + pip install llmcompressor datasets
    (hai gói này KHÔNG nằm trong setup_env.sh) + export lại LD_LIBRARY_PATH.

## TASK M (2026-08-10): GRAFT GGUF→GDN — **CHAMPION MỚI**, thắng cả hai trục

Ý tưởng: thân RedHatAI W4A16 (attention/MLP calibration xịn) + tim GDN lấy
từ GGUF Q4_K_M của unsloth (lưới RTN llama.cpp, KHÔNG calibration) chuyển
int8 g32 (~0,55% RMS), khâu bằng scripts/graft_gguf_gdn.py, nạp qua
patch_vllm_gdn_quant_load.

    metric        champion cũ (RedHatAI)   GRAFT (mới)
    conc1         29,5                     33,06  (+12%)
    conc32        563                      610,45 (+8,4%)
    ppl (99 đề)   5,1645 (baseline tươi)   5,0672 → ratio 0,9812 **PASS**

- Ứng viên KHÔNG đánh đổi: nhanh hơn Ở CẢ HAI mức tải VÀ ppl còn nhỉnh hơn
  baseline → thắng tuyệt đối (Pareto), phong champion theo đúng luật.
- **Giả thuyết cốt lõi xác nhận sạch**: vấn đề chất lượng của G/G2a
  (ratio ~1,15) là đặc thù của GPTQ-Hessian trên in_proj GDN, không phải
  của việc nén GDN — lưới RTN của llama.cpp nén cùng tensor mà ppl 0,98.
- Turbo mode (G, 768 tok/s, WARN) vẫn là lựa chọn riêng cho ai cần tối đa
  throughput và chấp nhận ppl.
- 4 bug thật tìm/sửa trong quá trình exec (commit 62c08a9..9b730eb):
  (1) dtype scale phải đọc từ config frame (RedHatAI là bf16, không fp16);
  (2) KHÔNG chạy fix_qwen35_hf_checkpoint lên RedHatAI — checkpoint đó
  multimodal thật (visual.*), nạp qua Qwen3_5ForConditionalGeneration và
  CẦN prefix language_model nguyên bản; fixer chỉ dành cho output
  AutoModelForCausalLM của quantize_*.py nhà mình. Graft giờ tự detect
  convention; (3) tên tensor GGUF có đuôi ".weight" (fixture test cũ sai
  y hệt nên tự đồng thuận — bài học: fixture phải đối chiếu format thật);
  (4) **gotcha vLLM đáng nhớ**: find_matched_target khớp catch-all
  targets:["Linear"] bằng SUBSTRING TÊN CLASS ("MergedColumnParallelLinear"
  chứa "Linear") TRƯỚC khi bước reconcile fused-component kịp chạy → mọi
  config_group cho module fused phải target TÊN FUSED ("in_proj_qkvz",
  "in_proj_ba") để exact-match bước 1, nếu không sẽ âm thầm rơi về scheme
  mặc định rồi nổ merge-mismatch.
- Checkpoint: /content/redhatai_grafted3 (~8,7GB) + graft_manifest.json.
  Cần đẩy lên HF repo riêng để không mất theo runtime recycle.

## TASK F2/F2b (2026-08-09): SLA open-loop trên kịch bản shared-prefix 32K

Cùng cấu hình TASK F. Payload: suffix unique 2.000-2.500 tok sau prefix chung
30K, output 400 tok, Poisson open-loop. **Closed-loop (conc32 wall) đã đánh
lừa lần nữa**: batch-đồng-loạt cho 1,3 req/s ảo; Poisson thật kịch ở ~0,45.

| rate chào | đạt | TTFT p50 | TTFT p95 | >3s |
|---|---|---|---|---|
| 0,10 | 0,13 | 1,29s | 2,09s | 0% |
| **0,20** | **0,20** | **0,41s** | **1,79s** | **0%** |
| 0,30 | 0,31 | 0,42s | 3,65s | 6,6% |
| 0,50 | 0,45 | 2,73s | 7,06s | 48% |
| 1,0-2,0 | 0,45 | 10-23s | 21-45s | bão hòa, rớt hàng loạt |

- **SLA (TTFT p95 < 3s): 0,2 req/s sạch; 0,3 là mép mềm (6,6% vi phạm).**
- Chẩn đoán bằng /metrics mỗi 10s ở 0,3: `num_requests_waiting = 0` suốt
  180s, KV usage đỉnh 37% → KHÔNG phải hết KV, KHÔNG phải queue admission.
  Là **tranh chấp compute trong batch**: chunked-prefill của người mới
  (~2,25K tok/req) chen vào giữa các bước decode của người đang chạy;
  running leo 15-20 là TTFT/decode của tất cả chậm đi mà "waiting" vẫn 0.
- Đối chứng output ngắn (0,5 QPS, max_tokens 100): TTFT p95 7,06→5,69s,
  vi phạm 48%→29%, e2e p95 74,5→24,9s. Giúp đáng kể nhưng KHÔNG cứu được
  SLA ở 0,5 — chi phí serialize prefill suffix là thật, output ngắn không xóa.
- TASK F2c (2026-08-10, đã chạy trên champion, cùng kịch bản): tune
  `--max-num-batched-tokens` THẮNG một nấc thật:

      variant       rate 0,3: p50/p95/vi phạm   rate 0,5: p95/vi phạm
      8192 (mặc định)  1,37/2,99s/5%              7,40s/18%
      2048             1,65/2,69s/0%              5,98s/17%
      **1088 (sàn)**   **1,35/2,51s/0%**          5,84s/15%
      8192+lpt2048     1,64/2,85s/3%              7,28s/18%

  - **Config đề nghị cho kịch bản shared-prefix: --max-num-batched-tokens
    1088.** SLA crossing chính thức: **0,3 req/s sạch** (trước là 0,2).
  - SÀN CỨNG mới phát hiện: align mode assert `block_size <= max_num_batched_tokens`
    và mamba block_size của model này = **1056** → 1024/512 KHÔNG khởi động
    được (AssertionError lúc init, không phải kết quả hiệu năng);
    `--long-prefill-token-threshold` cũng phải >= 1056.
  - Nỗi lo "chunk nhỏ → cold TTFT tăng" KHÔNG xảy ra: prefix 30K đã cache
    nóng, thứ bị chunk chỉ là suffix 2-2,5K (2-3 chunk) — thuế chunk không
    đáng kể. lpt đơn lẻ gần như vô tác dụng — ngân sách batched-tokens mới
    là đòn bẩy thật.
  - Rate 0,5 vẫn KHÔNG mở được bằng đòn này (p95 5,84s): đúng chẩn đoán
    F2b — vấn đề của 0,5 là decode-residency/tồn đọng concurrency, không
    phải prefill chen. Đòn còn lại cho 0,5: rút ngắn completion, cap
    concurrency/admission, hoặc chấp nhận SLA 0,3.
- Lưu ý bench: script F2 nằm Colab-local (taskF2_sla_sweep.py), payload
  prefix-chung + suffix-unique, KHÔNG tag chống-cache (cache là chủ đích).

## TASK C2 (2026-08-09): CPU KV offload — mua context/độ phủ prefix, KHÔNG mua số phiên

Cú pháp (đọc từ factory.py + cpu/spec.py, đã smoke sạch trên 9B W4A16):

    --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
      "kv_connector_extra_config": {"cpu_bytes_to_use": 21474836480}}'

- `cpu_bytes_to_use` BẮT BUỘC (thiếu là raise); extras: `eviction_policy` (lru),
  `store_threshold`, `blocks_per_chunk`/`block_size` (loại trừ nhau).
- Model hybrid BẮT BUỘC `--enable-prefix-caching` (assert trong
  build_offloading_config.py: "Hybrid models need --enable-prefix-caching").
- Buffer CPU cấp phát TRƯỚC lúc khởi động (RSS 37GB với budget 20GB) — không lazy.
- Metrics /metrics đã nối đủ (kv_offload_total_bytes, cpu_cache_usage_perc...);
  chỉ nhảy khi GPU KV thật sự bị đòi chỗ.
- **Trần kiến trúc** (đọc code, chưa sweep thực nghiệm): OffloadingConnector chỉ
  quản tier KV attention; trạng thái mamba/conv nằm pool riêng
  (single_type_kv_cache_manager.py) và request đang chạy vẫn cần 1 slot mamba
  sống — nên offload KHÔNG nâng được số phiên chạy đồng thời quá trần block
  mamba. `align` chỉ đồng bộ evict mamba-state với evict KV-block dưới prefix
  caching (phiên idle/finished resume qua prefix cache), không phải cách chạy
  thêm phiên live. Mamba-state offload thật sự thì chưa ai ship.

## TASK H (2026-08-10): bench skills-pack + câu hỏi thật — kịch bản production ĐẬU ĐẸP

Champion RedHatAI W4A16, --enable-prefix-caching --max-model-len 32768,
prefix skills ~28.760 token thật (ước 23,2K từ --dry-run — đếm bằng tokenizer
served mới chuẩn), 74 câu tuyển từ public-test, bench_skills.py:

    conc   TTFT p50   TTFT p95   decode/user   throughput tổng
      1     0,21s      0,26s      27,1 tok/s      26,8
      8     0,45s      0,88s      19,1           140,8
     16     0,70s      1,73s      15,7           228,7
     32     1,27s      3,10s      11,2           286,5

- **74/74 câu sạch ở mọi mức, 0 lỗi.** Warm-up cold một lần: 10,52s.
- **Prefix cache hit 99,04%** (đọc thẳng /metrics) — TTFT 0,21s vs cold
  10,52s = 50×. Câu hỏi thật (ngắn hơn nhiều so với suffix synthetic 2-2,5K
  của F2/F2b) cho TTFT tốt hơn hẳn dự phóng worst-case.
- Ngưỡng "max conc giữ TTFT p95<3s" cho kịch bản này: giữa 16 và 32, sát 32.
- Bug đã sửa sau run: vllm 0.26 đổi tên metric thành *_total →
  bench_skills.py trước đó báo cache_hit_rate=None oan (đã vá alias).
- Bẫy vận hành ghi lại: (a) --max-model-len phải >= prefix thật (8192 →
  400 Bad Request); (b) kênh transfer base64: PHẢI verify sha256 từng chunk
  ngay lúc paste — 2 lần hỏng chunk im lặng đã bị manifest bắt được.

### Re-bench trên CHAMPION GRAFT (2026-08-10, config production đầy đủ
### mnbt1088 + prefix caching + align + fp8 KV) — bảng chung cuộc:

    conc   TTFT p50 (cũ→mới)   p95 (cũ→mới)    throughput (cũ→mới)
      1    0,21 → 0,217        0,26 → 0,28      26,8 → 31,2   (+16%)
      8    0,45 → 0,315        0,88 → 0,76     140,8 → 191,4  (+36%)
     16    0,70 → 0,488        1,73 → 1,62     228,7 → 273,7  (+20%)
     32    1,27 → 0,959        3,10 → 3,03     286,5 → 370,6  (+29%)

- 74/74 sạch mọi mức, hit rate 99,37%. Vượt kỳ vọng +8-12% (bench_load
  đo prompt ngắn đánh giá thấp lợi ích byte-giảm ở decode dài context).
- conc32 p95 3,03s — sát trần 3s; "max conc trong SLA" giờ ~28-32.
- Kết quả: out_skills/bench_skills_champion_graft.jsonl (Colab-local).

## TASK Q2 (2026-08-12): workload agent nhiều phiên — trần là 1 PHIÊN trong SLA 3s

champion, max-model-len 65536, production flags, prefix tổng hợp **30K chia
sẻ** (bản chạy đầu dùng prefix ~1K là ĐO SAI KỊCH BẢN, đã bỏ).

    TTFT p95 (s)      sessions=1   4      8       16
    nhẹ  (3 lượt)     1,59        4,16   8,41    13,89
    thường (5 lượt)   2,40        6,41   12,48   73,17
    nặng (10 lượt)    3,19        8,60   15,27   165,05

- **Chỉ sessions=1 giữ được p95<3s** (kịch bản nặng còn trượt luôn ở 1).
- Hit rate v2 lành mạnh (63-95%), khác hẳn v1 prefix tí hon (5-39%) →
  xác nhận hit thấp ở v1 là do prefix nhỏ, không phải bệnh lý.
- **CHẨN ĐOÁN Ở sessions=16 (bằng /metrics, không suy luận)**:
  `num_preemptions_total` = **0 tuyệt đối** → LOẠI TRỪ vòng xoáy
  evict-reprefill. `kv_cache_usage_perc` đỉnh **97,2%**;
  `num_requests_waiting_by_reason{capacity}` = 15 (deferred = 0).
  ⇒ **Hàng đợi do HẾT DUNG LƯỢNG KV** — thuốc là giảm max-model-len /
  giảm số phiên / thêm VRAM, KHÔNG phải chỉnh mnbt.
- **NHƯNG chẩn đoán này KHÔNG giải thích sessions=4**: dung lượng lý
  thuyết là 6,17 phiên @65536, nên 4 phiên còn DƯỚI trần mà TTFT p95 đã
  4,16-8,60s. Cơ chế ở vùng dưới-trần chưa xác định — đây mới là vùng
  đáng tối ưu (16 phiên vượt trần số học thì hỏng là đương nhiên).
- pct_time_prefill_reconnect tăng mạnh theo tải: nhẹ 9→22%, thường
  20,6→42%, nặng 21,4→**57,4%**. Ở tải cao, hơn nửa thời gian phiên là
  chờ/tính lại prefill.
### Q2b (đo TỪNG mức riêng lẻ, không chồng pha) — ba kết quả quyết định:

**(1) PREFIX CHUNG ĐƯỢC CHIA SẺ THẬT — kiến trúc được xác nhận bằng số:**

    sessions   KV usage đỉnh   ≈ token      mô hình "30K chung + 16K/phiên"
    1          11,5%           46.530       46.000  ✓
    4          24,0%           97.107       94.000  ✓
    8          39,8%           160.632      158.000 ✓
    16         70,4%           284.850      286.000 ✓

Khớp 4/4 điểm. Con số 97% hoảng hồn ở lượt trước là **artifact chồng pha**
(phiên mức trước chưa giải phóng khi mức sau bắt đầu) — không phải bệnh.
⇒ Giả định nền của cả dự án ĐÚNG.

**(2) VÙNG DƯỚI TRẦN: bộ lập lịch RESERVE theo worst-case max-model-len.**
sessions=4: KV usage chỉ 24% (còn 76% trống) mà `waiting_by_reason{capacity}`
đã = 2, preemption = 0. ⇒ scheduler từ chối nhận request mới dù cache dư
dả — vì nó phải chừa chỗ cho MỖI request có thể phình tới max-model-len.
Điều này giải thích chính con số "6,17× @65536" = 404.613 / 65.536.
**⇒ ĐÒN TỐI ƯU MỚI, rẻ và rõ: hạ max-model-len xuống ĐÚNG nhu cầu thật**
(prefix 30K + lượt × kết quả tool + biên ≈ 40K) thay vì 65536 → dung
lượng đồng thời tăng ~1,6× mà KHÔNG mất gì. Chưa đo — xem Q2c.

**(3) THƯỚC ĐO ĐÚNG: điểm vận hành là 8 PHIÊN, không phải 1.**

    sessions   tasks/hr     thời gian 1 tác vụ   TTFT p95
    1          81,0         44,4s                2,4s
    4          271,7        52,4s                6,4s
    **8**      **328,1 ⬅ đỉnh**  82,8s           12,5s
    16         189,2 (sụt)  293,4s (+254%)       73,2s

Với vòng lặp agent, user vốn đã chờ tool 0,5-5s/lượt nên TTFT từng lượt
ít quan trọng; cái đáng chọn là thông lượng tác vụ. **16 phiên là lỗ vốn
rõ ràng** (tasks/hr −42%, thời gian mỗi tác vụ ×3,5) — dấu hiệu kinh điển
của việc vượt điểm bão hoà và đốt GPU vào hàng đợi.

## TASK Q2c (2026-08-13): quét max-model-len — GIẢ THUYẾT "+1,6× capacity" BỊ BÁC Ở ĐIỂM VẬN HÀNH

Runtime mới A (e27f78d34f93, vLLM 0.27.1), bootstrap 5,9 phút bằng champion prebuilt
(sanity PASS: output mạch lạc, 36,3 tok/s conc1 — bản HF nguyên vẹn). Cả 3 điểm đo
CÙNG runtime, prefix tổng hợp 30K (thực tokenize ~24,5K), 5 lượt, sessions 4/8/16:

    tasks/hr          mml=40960        mml=49152     mml=65536
    sessions=4        (invalid)        197,1         197,4
    sessions=8        (invalid)        312,3         **329,5 ⬅ đỉnh giữ nguyên**
    sessions=16       (invalid)        **207,7 (+9,4%)**  189,8
    TTFT p95 @16      —                53,2s (−23%)  69,4s

- **Tái lập Q2b xuất sắc**: 65536 hôm nay 329,5/189,8 vs hôm qua 328,1/189,2
  (khác runtime, lệch <0,5%) — bench này đáng tin.
- **mml=40960 HỎNG workload**: 6/28 phiên 400 Bad Request ở lượt cuối. Nguyên nhân
  đo được: prefix "30K ước lượng" = ~24,5K token thật, nhưng context phình
  2,2-5,5K token THẬT mỗi lượt (turn1 24.533 → turn4 36.977; lượt 5 ≈ 41K > 40960).
  Nhu cầu thật của workload 5 lượt ≈ 42-43K. Bẫy: ước lượng theo từ đánh giá
  THẤP hơn tokenizer thật đáng kể ở cả hai chiều.
- **Giả thuyết Q2b(2) "hạ mml → +1,6× dung lượng, không mất gì" BỊ BÁC ở điểm
  vận hành**: tại 8 phiên, 49152 KÉM hơn 5,2% (312,3 vs 329,5) dù admission
  tăng 6,17→8,23 chỗ. Lý giải: ở 8 phiên, nghẽn thật là GPU compute chứ không
  phải admission — scheduler bắt chờ chỉ đổi thứ tự việc, GPU vẫn bận. Hàng
  đợi capacity ≠ GPU lãng phí.
- Hạ mml CHỈ có ích ở vùng quá bão hòa: 16 phiên +9,4% tasks/hr, TTFT p95 −23%.
- **KHUYẾN NGHỊ CHỐT: giữ 65536 cho điểm vận hành 8 phiên** (an toàn overflow,
  throughput cao nhất); nếu buộc chạy 16 phiên thì 49152 tốt hơn. Không có
  free lunch từ mml.

## TASK Q3 (2026-08-13): stress điểm vận hành 8 phiên — KHÔNG CÓ BỆNH LÝ

Cùng runtime/config với Q2c (mml=65536, baseline 329,5 tasks/hr):

- **Q3a abandon 25%**: server DỪNG generate NGAY khi client đóng kết nối. Test quyết
  định: 1 request max_tokens=3000, client đóng sau vài token → `num_requests_running`
  về 0 trong ≤3s, giữ 0 suốt 55s. KHÔNG có token chạy lậu, không GPU waste.
  Bẫy metric: vLLM 0.27.1 KHÔNG đếm abort-do-disconnect vào `request_success_total`
  (abort=0 vĩnh viễn) — đừng dùng metric đó để kết luận; suy luận ban đầu của tôi từ
  metric này là SAI, test trực tiếp mới chuẩn.
- **Q3b tool-latency tail 10%:30s**: vô hại — 8/8 xong, 334,8 tasks/hr (≈baseline),
  hit 90,9%. Prefix SỐNG SÓT gap 30s dưới tải 8 phiên (khớp Q1 single-session 60s).
- **Q3c burst-sync (thundering herd)**: −18,6% tasks/hr (268,2), TTFT p95 +14%
  (14,2s), 8/8 hoàn thành — đau nhưng ổn định, không gãy.

## TASK Q4 (2026-08-13): chính sách tràn context — summarize-stub/truncate giữ 100% việc, error mất trắng

Ép tràn bằng --context-limit-tokens 26000 (8 phiên, mml=65536, overflow áp dụng
13 lượt ở hai policy ghi đè):

    policy            hoàn thành   tasks/hr        TTFT p95   hit rate
    (baseline)        8/8          329,5           12,5s      90,9%
    error             **0/8**      0 (mất trắng)   —          —
    truncate-oldest   8/8          234,8 (−28,7%)  30,3s ×2,4 82,7%
    summarize-stub    8/8          230,3 (−30,1%)  23,6s ×1,9 81,3%

- Cơ chế giá phải trả: ghi đè transcript → miss prefix-cache phần bị viết lại →
  re-prefill (TTFT p95 tăng vọt). Prefix hệ thống 30K vẫn cache (hit còn ~82%).
- **KHUYẾN NGHỊ: summarize-stub** (p95 tốt hơn truncate 22%, tasks/hr tương đương);
  overflow policy là LƯỚI AN TOÀN, không phải chiến lược — thiết kế đúng là chọn
  mml ≥ nhu cầu thật (Q2c: ~42-43K cho 5 lượt).
- Bẫy harness ghi lại (2 lượt đo đầu VÔ HIỆU vì trigger không nổ): 3 thang token
  trộn lẫn — yêu cầu 30K → phát 24.000 TỪ (est) ≈ 24,5K token thật; filler
  `tool%06d` tokenize 3-5 token/từ nên context THẬT phình 2,2-5,5K/lượt trong khi
  est chỉ +~1K; --context-limit-tokens so với EST chứ không phải token thật.
  Cần sửa harness: dùng cùng một thang (ưu tiên real qua usage) — TODO.

## KV-TRANSFER E0 (2026-08-14): CONTEXT SỐNG Ở CẢ GDN LẪN KV — transfer chỉ-attention KHÔNG ĐỦ

Nghiên cứu arXiv:2608.03893 (ridge mapper cross-model, không có code chính thức —
tự cài `models/qwen3_5/kv_transfer/`, 4/4 unit test). E0 needle-in-context trên
Qwen3.5-4B bf16 (10 trial, ablation cache sau prefill):

    nguyên vẹn 10/10 (NLL 0,069) | xóa GDN 0/10 (5,6) | xóa KV 0/10 (11,8)

- Phase B (GDN-state mapping — vùng paper bỏ ngỏ) là BẮT BUỘC với họ hybrid.
- Cặp ưu tiên: **4B↔9B** (matched TOÀN PHẦN kể cả GDN 16/32×128, lớp 1:1, cùng
  vừa 1 L4 → cascade 1-GPU). 27B để sau (GDN v-heads lệch 32/48).
- Bẫy đo bị bắt 2 lần: (1) zero cache không chạm tensor → 3 điều kiện trùng
  từng số lẻ (đã thêm hard-fail guard); (2) pgrep/pkill TỰ KHỚP chuỗi lệnh
  wrapper → "đang chạy" giả + tự giết launcher. Quy tắc mới: kill theo PID,
  pattern kiểu '[e]0...' cũng không an toàn trong bash -lc.
- Bug ghi nhận: champion config có quant group `group_size: 0` — vLLM tha,
  transformers/compressed-tensors decompress TỪ CHỐI → calib trên champion cần
  fix config hoặc dùng bf16 gốc (9B bf16 ~18,4GB vừa L4 cho batch-1).

## KV-TRANSFER E1 (2026-08-15): COPY NGUYÊN CACHE 4B→9B GIỮ 100% NEEDLE — RIDGE MAPPER LẠI PHÁ

Chuỗi full trên L4 (collect 4B+9B 200×1024 FineWeb-Edu → identity gate → fit →
eval 12 trial needle 1,5K tok, needle ở 25/50/75% context, protocol đối xứng
mọi điều kiện prefill [:-1] + nạp token cuối):

    điều kiện      needle   retention   NLL/tok   lat cache+step
    self_prefill   12/12    100%        0,011     0,889s (prefill 9B)
    **copy 4B**    **12/12  100%        0,043     0,080s**
    ridge mapper   0/12     0%          2,678     4,119s (numpy CPU)
    no_ctx (sàn)   0/12     0%          10,013    0,327s

    TTFT 1,5K ctx: 9B self 0,889s vs 4B prefill 0,622s + copy 0,080s = 0,702s (×1,27)

- **Phát hiện chính: cache 4B và 9B tương thích THÔ** — copy nguyên K/V + GDN
  state (shape trùng hệt) vào 9B, needle 12/12 ở cả 3 vị trí, NLL 0,043 gần
  self-prefill (0,011), chi phí transplant ~0. Nghi vấn Qwen3.5 4B/9B chia sẻ
  không gian biểu diễn cache (cùng KV 4×256, GDN 32×128 — có thể upscale từ
  cùng gốc). Paper không thử copy thô vì các cặp của họ lệch shape.
- **Ridge mapper (đúng bài paper) NGƯỢC LẠI phá sạch**: heldout R² attention
  0,726 / GDN 0,600 — 27-40% variance mất là đủ giết needle (0/12). Nghịch lý
  R²-vs-chức-năng: copy thô R² thấp hơn về hình học nhưng giữ đúng các hướng
  attention đọc; ridge "gần đúng đều" lại xóa tín hiệu. Bài học: R² KHÔNG phải
  proxy của retention — phải đo chức năng.
- Identity gate 9B→9B cứu 1 vòng GPU: bắt bug λ=1.0 đè chết head GDN biên độ
  nhỏ (R² 0,047) → sửa λ thích ứng scale (mean diag XᵀX, mặc định 1e-3) →
  attention R²=1,000, GDN 0,997+ khi identity. Guard no-op transplant + sàn
  no_ctx (0/12, NLL 10) xác nhận không có đường rò kết quả.
- Vận hành: runtime recycle nuốt calib 5,9GB×2 (bài học quy tắc 6 — artifact
  chỉ nằm runtime); dòng `!nohup` trong cell Colab từng phóng câm — chuyển
  chuẩn sang subprocess.Popen(start_new_session=True) + kiểm alive sau 10s.
- Ý nghĩa sản phẩm: cascade 4B-prefill→9B-decode trên 1 GPU khả thi với chi
  phí transplant ≈0; trần TTFT ~×2 ở 30K ctx (prefill 4B ≈44% FLOPs 9B). Cần
  E2 trước khi tin: eval khó hơn needle (QA/ppl continuation), ctx dài hơn,
  và đo trần 4B prefill thật trên vLLM.

## KV-TRANSFER E2 (2026-08-15): COPY 4B→9B GIỮ QA 9/9 TỚI 16K; TRẦN PREFILL 4B vLLM ~1,8× 9B

E2a (`e2_suite.py`, bf16 tuần tự, cache spill đĩa fp16): QA đa fact (3 mã ở
20/50/80% ctx, hỏi xoay) + cont-NLL 200 token FineWeb thật, L = 2K/8K/16K:

    L       QA self/copy/no_ctx   cont-NLL self/copy/no_ctx   TTFT tfm: self vs 4B+ghép
    2K      3/3  3/3  0/3         2,092  2,152  2,325         1,00s vs 0,80s (×1,25)
    8K      3/3  3/3  0/3         2,063  2,126  2,264         4,65s vs 3,59s (×1,30)
    16K     3/3  3/3  0/3         2,279  2,382  2,525         10,46s vs 8,30s (×1,26)

- **QA copy 9/9 mọi độ dài** — truy xuất không suy giảm theo ctx. **Cont-NLL:
  copy giữ 58-74% lợi ích ngữ cảnh** (self→no_ctx cách 0,20-0,25 NLL; copy chỉ
  mất 0,06-0,10) — "hiểu" giữ phần lớn, không trọn. Transplant 2-8ms (ghi đè
  tensor thuần, đo tách khỏi disk-spill).
- Tỷ lệ prefill 4B/9B trên transformers chỉ ~0,8 (GDN torch fallback nghẽn) →
  ×1,25-1,30. Số vLLM mới là thật (dưới).

E2b (`run.sh serve 4b` mới — frame RedHatAI W4A16, config như 9B; prefill_bench
cache-proof): **4B prefill 5514 / 5307 / 4771 tok/s @4K/16K/30K** (9B champion:
2789-2934) = **×1,8-1,9**; KV cache 752.239 token (9B: 560.380, +34%). Server
lên sau 511s (fresh env). → Cascade 4B-prefill→9B-decode trên số vLLM: TTFT 30K
10,45s → ~6,3s (**×1,66**); 4K: 1,38s → 0,73s (×1,9).

- Bug bắt giữa chừng: OOM lm_head khi prefill 9B ≥8K (logits full-seq 3,7GB)
  → `logits_to_keep=1` ở mọi forward prefill; expandable_segments bật.
- Trạng thái tích hợp: vLLM CHƯA có đường bơm cache ngoài (RFC #44223 mở) —
  cascade production cần connector hoặc hack engine = Phase C. Số trên là
  trần đo được từ thành phần, ghép tuần tự trên 1 L4.

## KV-TRANSFER E3 (2026-08-15): BỘ CHỈ SỐ 9B ĐO LẠI QUA CROSS @30K + BỨC TƯỜNG ĐỒNG TRÚ vLLM

E3A (`e3_bench.py`, chunked prefill 4096, 4 trial @30K, bf16):

    điều kiện   QA     cont-NLL   decode      first-step
    self        2/2    2,235      11,8 tok/s  0,139s
    **copy**    **2/2  2,467      11,8 tok/s  0,085s**
    no_ctx      0/2    2,732      12,9        —

- **Decode parity XÁC NHẬN**: 11,8 = 11,8 — decode không quan tâm cache từ đâu
  → mọi số decode/throughput 9B thuần (34-36 conc1, 390 conc32) GIỮ NGUYÊN cho
  cross. TTFT transformers ×1,24 (18,66→15,02s); số vLLM thật ×1,66.
- Retention "hiểu" theo chiều dài: 74% @2K → 69% @8K → 58% @16K → **53% @30K**
  (suy giảm đơn điệu — trần của copy thô, cần ghi khi bán).

E3B đồng trú 2 server trên 1 L4: **4 lần chết = 4 ràng buộc đo được**:
1. KV 9B ≥ mml tối thiểu (util 0,62 cho 0,86GiB < 0,9GiB cần cho mml 65536).
2. Hybrid GDN: max_num_seqs ≤ số Mamba block (mặc định 256 > 193 → chết).
3. CUDA graphs ăn ~6% util ẩn (0,62 hiệu dụng 0,55).
4. **BỨC TƯỜNG: vLLM 0.27.1 tính non-torch memory bằng NVML — cộng cả VRAM
   của process KHÁC vào chi phí mình** → server thứ 2 cần util ≥0,84 cho KV
   nhưng check khởi động chặn ≤0,38 (free 8,5GiB) — mâu thuẫn, đồng trú naive
   BẤT KHẢ THI. Lối thoát đã xác định: `--kv-cache-memory-bytes` (bỏ profiling,
   có trong EngineArgs) — chưa thử. Ngoài ra EngineArgs có `kv_transfer_config`
   (khung KVConnector) = cửa Phase C.
- Bẫy vận hành mới: killpg chuỗi launcher giết luôn server con (9B) — server
  sống lâu phải setsid tách khỏi chuỗi phóng, hoặc chỉ kill launcher theo PID.

### E3C (2026-08-15): ĐỒNG TRÚ THÀNH CÔNG — combo đúng và GIÁ đo được

Combo vượt tường (lần thử 6, lần ĐẦU flag chính thực sự được kiểm): server 2
đặt util THẤP (0,35 — qua check startup free-memory) + `--kv-cache-memory-bytes
1.5e9` (KV tường minh, né profiling NVML) + enforce-eager. Kết quả:

    9B (util 0,68, mns 32):  KV 208.005 token (−63% vs 560.380 solo)  smoke OK
    4B (0,35 + kv-bytes):    KV 74.159 token, eager
    GPU tổng: 20,8GB / 22GiB. Prefill 4B đồng trú: 2138/3291/3406 tok/s
    @4K/16K/30K (71% mức độc chiếm @30K; 45% @4K — mất vì chia compute).

Ý nghĩa: cascade 1-GPU chạy ĐƯỢC về cơ học nhưng giá đắt — TTFT 30K chỉ còn
~×1,15-1,2 (4B đồng trú 8,8s vs 9B solo ~10,5s) và 9B mất 63% KV (~12 phiên →
~4). Trên MỘT L4, đồng trú chỉ đáng khi tải ít phiên + cold-miss nhiều; giá
trị lớn của cascade nằm ở 2-GPU disaggregation hoặc tuning lại phần chia
(4B bớt KV, 9B util cao hơn — chưa đo, để Phase C quyết bằng số khi có connector).

## KV-TRANSFER E4 (2026-08-15): CHỌN THUẬT TOÁN BẰNG PHÂN PHỐI THẬT — 3 MODEL, 3 CẶP

`e4_stats.py`, 64 văn bản chung, 4B/9B bf16 + **27B bnb-4bit** (lần đầu 27B chạy
transformers trên L4). Kết quả tại /content/logs/e4_stats.json; số chốt:

**Hình dáng phân phối**: attention K gần Gaussian (kurtosis 0,0-0,2), eff-rank
154-307/1024 — đất tốt cho phương pháp tuyến tính. GDN: eff-rank ~50/128,
kurtosis tới 2,4 và phổ Â có sv_max tới **110** ở lớp sâu — đuôi nặng, MSE dễ
nổ (ridge R² âm tới −11 ở GDN L62 là bằng chứng sống).

**Cặp 4B→9B (attention)**: CCA top-64 = **0,977-0,980** — cấu trúc tuyến tính
chung gần tuyệt đối. Nhưng identity R² thô chỉ 0,12-0,14 mà copy vẫn 12/12
chức năng → CHỐT bằng số: hình học MSE ≠ chức năng attention. Không ứng viên
đóng-form nào cần thay copy. Phần "hiểu" mất @30K KHOANH VÙNG được: GDN sâu
(CCA tụt 0,90→0,71 @L30, mọi ứng viên ≤0,24) — muốn đòi lại phải vá đúng chỗ đó.

**x→27B attention**: CCA **0,93-0,97** — cấu trúc chung TỒN TẠI dày (go cho
27B). Ridge 1-lớp 0,52-0,73 (tốt nhất L31→L63). **Concat-ridge full-recipe
NVIDIA THUA ridge 1-lớp** (0,22-0,52) ở N=13K mẫu — nhiễu ước lượng 8192 chiều
nuốt hết lợi ích; muốn dùng phải tăng calib ≥5×. Identity âm nặng (−1,8) —
copy thô chết như dự đoán với cặp lệch khuôn.

**x→27B GDN (32vs48 head)**: L0 map được ngay (ridge/scaled-Procrustes ≈0,91);
**L giữa (L33) CCA 0,26-0,27 = BỨC TƯỜNG** — gần như không có cấu trúc tuyến
tính chung; L62 CCA 0,64 nhưng ridge nổ vì đuôi nặng. → Tuyến tính KHÔNG đủ
cho GDN giữa của 27B; đây là chỗ duy nhất bắt buộc phương pháp học phi tuyến.

**PHÁN QUYẾT THUẬT TOÁN (theo số, không đoán)**:
1. 4B→9B: giữ copy; nâng cấp duy nhất đáng làm = mapper CHỈ cho GDN sâu, train
   bằng functional loss (khớp đầu ra model đích, không khớp giá trị cache).
2. x→27B: kiến trúc chọn = mapper per-layer nhẹ — attention khởi tạo từ ridge
   (CCA 0,93+ chống lưng), GDN dùng MLP nhỏ; TRAIN BẰNG FUNCTIONAL LOSS;
   chuẩn hóa RMS state trước khi map (trị sv_max 110/kurtosis 2,4); calib
   ≥500 seq nếu dùng concat. Train một lần trên L4 (model đóng băng, mapper
   vài triệu tham số, grad checkpointing qua 27B bnb-4bit).
3. Đường loại bỏ đã đóng bằng số: copy thô →27B (identity −1,8), concat-ridge
   cỡ mẫu hiện tại, ridge-MSE thuần cho GDN sâu.

## KV-TRANSFER E5 v1 (2026-08-15): TRAIN MAPPER 4B→27B CHẠY ĐƯỢC TRÊN L4 — KL GIẢM 4-5×, NEEDLE CHƯA ĐẠT

`e5_train.py` — hạ tầng train functional-loss qua 27B đóng băng trên MỘT L4
(đóng góp chính, chưa ai làm): 2 pha (4B một mình spill cache đĩa → 27B một
mình train), backward chỉ qua suffix 32 token (cache map là INPUT của forward
→ autograd với tới mapper không cần backprop qua prefill), Adam8bit, ctx 512.
5 lần OOM = 5 lớp bản đồ bộ nhớ: bnb không nén embed/lm_head (27B ~18GB);
đồ thị backward GDN torch-fallback ~1,2GB/32tok; Adam state 270MB là giọt
tràn ly. Train 400 bước ổn định, 4,7s/bước, GPU 22,4GB không creep.

    KL: 134,7 (init) → 61 (step 50) → 35-45 (200) → dao động 18-45 (240-390)
    Eval needle 10 trial: self 10/10 | **mapped 0/10** | no_ctx 0/10

- **Phán quyết v1**: functional loss HỌC ĐƯỢC (KL ÷4-5) nhưng chưa qua ngưỡng
  truy xuất — mapped vẫn ở sàn. Khớp dự đoán E4: mid-GDN CCA 0,27 là tường
  thật; 35M tham số tuyến tính+song tuyến, 400 bước, văn bản trơn — chưa đủ.
- Nghi phạm chính theo thứ tự bằng chứng: (1) KL còn ~1/token — xa mức
  teacher, cần 5-10× bước + lr schedule; (2) data trơn không có mẫu truy
  xuất — hướng gradient không ưu tiên các chiều "tra fact" (needle-aware
  data); (3) tín hiệu cuối chuỗi quá thưa — cần khớp attention-output từng
  lớp (dense supervision); (4) 4B→27B một nhảy có thể quá xa — đường 2 chặng
  4B→9B (copy, đã chứng minh) + 9B→27B (học) chưa thử.
- Hàng E5 v2 (chờ user duyệt scale): steps 2000+, mapper MLP per-head,
  needle-aware calib, per-layer output matching, thử nguồn 9B.

## KV-TRANSFER E6+E7+E6b (2026-08-15): BENCHMARK THẬT + MA TRẬN 8 CẶP + MỔ VẾT NỨT GREEDY

**E6 (train hội tụ + BFCL 20 câu function-calling)**: mapper 4B→27B v2 (2000
bước, needle-mix 40%, dense supervision 16 hook, cosine, Adam8bit, cycle 600
mẫu — 2000 spill = 112GB từng làm ĐẦY ĐĨA, bài học mới quy tắc vận hành):
KL hội tụ 134,7 → đáy 0,632 (v1 chưa từng dưới 17). Nhưng:

    đường                self NLL/hit      xfer NLL/hit      no_ctx
    27B + mapper v2      2,597 / 14/20     7,042 / 0/20      7,355 / 0/20
    9B + copy thô 4B     2,461 / 19/20     **2,497 / 6/20**  7,091 / 0/20

- 27B-mapper: hội tụ TRONG MIỀN nhưng chết NGOÀI MIỀN (train văn xuôi 480 tok
  vs đề JSON 2K tok) — lỗi miền train, không phải lỗi cặp. ifstruct/parsebench
  schema không khớp field-picker (nợ v3).
- **9B-copy trên task thật: NLL parity ~99% nhưng greedy hit 6/20 vs 19/20** —
  vết nứt đầu tiên của copy: lệch logit nhỏ (vô hình với NLL) lật argmax trong
  chuỗi 24 token khi biên quyết định mỏng (chọn tên hàm). Echo trực tiếp phát
  hiện r=−0,20 của NVIDIA: error PLACEMENT > error magnitude.

**E6b (mổ tận gốc vết nứt, 5 điều kiện cùng 20 câu)**:

    self 19/20 | copy fp16 6/20 | copy bf16-spill 4/20
    | copy + 9B tự đọc 32 tok cuối: 2/20 | 128 tok cuối: 1/20 (NLL cũng tệ đi)

- Nhiễu fp16-spill: VÔ CAN. **Suffix re-prefill PHẢN TÁC DỤNG** — mỗi ranh
  giới 4B-cache/9B-native là điểm gãy nhất quán (key hai model lệch thang →
  softmax nhìn qua ranh giới bị méo; GDN state 4B + động lực học 9B = trạng
  thái lai). **Cache copy sống nhờ NHẤT QUÁN NỘI TẠI; trộn là phá** — đo trực
  tiếp bài học transition-point của DroidSpeak.

**E6c (bf16 toàn tuyến — nghi phạm cuối) — PHÁN QUYẾT VỤ GREEDY: hai thủ phạm,
mỗi bên một nửa**:

    self 19/20 (NLL 2,637) | copy bf16 9/20 (NLL 2,430!) | copy fp16 10/20

- bnb-4bit của harness gánh ~nửa vết nứt (bnb 4-6/20 → bf16 9-10/20). Nửa còn
  lại (10 vs 19) là GIỚI HẠN THẬT của cache 4B với quyết định biên mỏng —
  information bottleneck đúng nghĩa C2C mô tả.
- Nghịch lý error-placement lần 4, mạnh nhất: copy NLL 2,430 TỐT HƠN self
  2,637 mà greedy vẫn thua 9 vs 19 — cache copy "tự tin trung bình" hơn nhưng
  đặt sai lệch đúng các bước 51/49.
- Kết luận vận hành: production W4A16 Marlin (sạch hơn bnb nf4) kỳ vọng nằm
  giữa hai mức — PHẢI đo lại trong Phase C trên vLLM thật. Copy scope an toàn:
  chat/QA/RAG; function-calling nghiêm ngặt cần target-side polisher (mapper
  nhỏ train functional trên chính 9B — không phải suffix-repair đã bị bác)
  hoặc giới hạn task.

**E7 (ma trận 8 cặp, needle tile-transplant + CCA + variance-explained)**:

    cặp        needle   CCA-attnK  CCA-GDN   VarExpl-attnK  VarExpl-GDN
    4B→9B      5/5      0,996      0,916     0,531          0,051
    4B→27B     N/A(*)   0,992      0,785     0,234          −0,646
    2B→4B      0/5      0,979      0,232     0,488          −0,134
    2B→9B      0/5      0,977      0,232     0,451          −0,127
    2B→27B     0/5      0,962      0,246     0,267          −0,105
    0.8B→4B    0/5      0,976      0,231     0,416          −0,355
    0.8B→9B    0/5      0,974      0,234     0,381          −0,382
    0.8B→27B   0/5      0,960      0,244     0,201          −0,467
    (*) tỷ lệ GDN head 32→48 không nguyên, không có dạng tile; mapper = E6.

- **PAIRABILITY SCORE tìm thấy: CCA-GDN ≥ ~0,9 ⟺ copy/tile sống.** Attention
  thẳng hàng TOÀN HỌ (0,96+) — số phận mỗi cặp nằm 100% ở GDN.
- Họ {0.8B, 2B}: GDN CCA ~0,23 với mọi model lớn — chế độ mã hóa khác hẳn
  (16 v-head không phải bản thu nhỏ); không có đường tắt, muốn dùng làm
  prefill-servant phải functional-mapper hoặc compatibility-finetuning.
- 4B→27B GDN CCA 0,785 = cặp-học-được sáng nhất ngoài họ (chọn cặp E5/E6 đúng).
- Variance-explained PHẢN chức năng lần 3 (4B→9B chỉ 0,53/0,05 mà copy 100%).

**E8 v1 (compatibility LoRA 2B→9B — deep-innovation #5, user duyệt "làm đi"
2026-08-15, `e8_compat.py`, e8_results.json)**:

    gate 2B self needle: @800 5/5, @2000 5/5   ← trần thông tin SÁNG
    baseline tile 2B→9B (0-train): 0/5          ← tái lập E7
    train 300 bước LoRA r=16 (90 linear khối GDN, 5,9M param, lr 2e-4,
      L_CTX 256, loss KL-functional qua tile + aux state-MSE decay 30%):
      KL 231 → ~126 (÷1,8, nhiễu 81-183), needle tile 0/5 Ở CẢ 6 MỐC EVAL.

- Gate quan trọng nhất: chính 2B nhớ needle hoàn hảo → cache 2B CÓ thông tin;
  thất bại là thuần "phương ngữ" GDN (CCA 0,23), KHÔNG phải bottleneck.
- Hạ tầng mới chạy được: 2B(bnb+LoRA) + 9B(bnb) ĐỒNG TRÚ 10GB/23GB, gradient
  chảy xuyên forward 2B (guard grad-flow OK |g|=2,5e6), ~7s/bước.
- KL giảm ÷1,8 rồi đi ngang từ ~step 100 (E5 mapper 27B từng ÷4-5): 5,9M
  tham số LoRA chỉ-GDN chưa đủ lực uốn phương ngữ. KẾT QUẢ ÂM CÓ KIỂM SOÁT.
- V2 đề xuất (CHỜ USER DUYỆT): (a) r=64 + target mọi linear (phương ngữ có
  thể hình thành trước khối GDN) + lr 5e-4 + 600 bước; (b) đổi loss chính
  sang state-alignment RMS-norm MSE (tín hiệu dày per-layer, KL làm gate);
  (c) nếu cả hai thua → chốt "phương ngữ nhóm nhỏ không sửa được bằng adapter
  nhẹ", 4B là prefill-helper chính thức duy nhất.

**E8 v2+v3 (user duyệt a+b, lệnh "theo bài Unsloth, khảo sát kỹ") — ĐÓNG,
phương án (c) kích hoạt (2026-08-15, `e8v2_qlora.py`/`e8v3_unsloth.py`)**:

    v2 (bnb student, LoRA r=64 MỌI linear 186 module 67,3M, loss
      state-alignment nMSE, batch 4): 13s/bước — kill sớm (torch-fallback).
    v3 (đúng bài Unsloth: student BF16 — docs Unsloth "KHÔNG QLoRA 4-bit
      trên Qwen3.5", trùng E6c; kernel Triton fla+causal-conv1d build 13ph,
      fast path BẬT thật): grad-flow OK, GPU 100%, vẫn 13,8s/bước.
      nMSE 10,52 → 1,06 @20 (học xong THANG ĐO) → mài 0,9855→0,9587→0,9588
      (đứng im) — needle 0/5 @74 và @149 → DỪNG SỚM đúng cam kết
      (tiết kiệm ~1,5h GPU).

- Khảo sát Unsloth (docs Qwen3.5): FastLanguageModel wrapper KHÔNG dùng được
  cho bài cache-loss — forward patch trả past_key_values=None trong training
  + target_modules bị filter vision/language + trả PROCESSOR đa phương thức
  (tok(text) chết ở image loader — bug đã vá). Tinh túy giữ được: bf16
  student + kernel fla + LoRA rộng.
- Chẩn đoán tốc độ sai 2 lần liên tiếp (bnb → torch-fallback → hóa ra
  compute thật, GPU 100%): quy tắc 5 thắng — chỉ số đo mới tin.
- **PHÁN QUYẾT E8 (3 đòn tấn công đủ mạnh đều thua)**: gate trần thông tin
  SÁNG (2B self 5/5+5/5) nhưng nMSE kẹt ~0,96 = LoRA 67M chỉ chiếm được ~4%
  cấu trúc state 9B — khớp tường CCA-GDN 0,23. **Phương ngữ GDN nhóm nhỏ
  {0.8B,2B} KHÔNG sửa được bằng adapter nhẹ + alignment/functional loss cỡ
  vài trăm bước.** Ô ma trận đóng bằng kết quả âm 3 lớp; 4B giữ vai
  prefill-helper chính thức duy nhất của 9B. Muốn mở lại cần vũ khí khác
  hẳn: finetune sâu (full GDN blocks) hoặc weight-derived conjugation (#1).

**E6 v3 (CE-GOLD mapper 4B→27B trên miền thật — user chốt "KL chưa đủ, cần
CE bảo đảm đầu ra + bộ SE/BFCL/validation", 2026-08-23→24, `e6v3_ce.py`)**:

    Loss: CE(gold) + 0,3·KL + warm-MSE + 0,05·dense; CE_FLOOR 0,2 (Unsloth).
    Data THẬT: BFCL 285 + ifstruct 60 (pseudo-gold 27B + validator) +
      ParseBench-table 40 (bảng→trích hàng k) + needle 200; val 50; test
      niêm phong = 20 BFCL E6 + 10 needle@2K.
    KẾT CỤC: CE train 8,7 → 0,008 (THUỘC LÒNG 585 cache) nhưng VAL 0 TOÀN
      TUYẾN cả 4 mốc (149/299/449/599) → tự dừng stale-3. TEST NIÊM PHONG:
      BFCL self 20/20 | mapped 0/20 | no_ctx 0/20. (needle2k VÔ HIỆU —
      TRAIN_MAX cắt mất câu hỏi, self cũng 0: artifact, đã ghi nợ.)

- **PHÁN QUYẾT v3 (lần 3, cùng một câu trả lời qua 3 hàm loss: KL thuần /
  KL hội tụ sâu / CE-gold)**: mapper 35M tuyến-tính/song-tuyến-tính GHI NHỚ
  được nhưng KHÔNG TỔNG QUÁT HÓA được phép dịch cross-shape từ vài trăm mẫu
  — vấn đề nằm ở LỚP HÀM, không phải hàm loss hay miền data nữa.
- Trận chiến môi trường runtime mới (py3.13 + datasets mới + transformers
  5.15): 7 bug fix nối tiếp — zombie PID qua mặt kill -0; BFCL "Trailing
  data" (đổi hf_hub_download + tự parse); cache states bọc dict {0:tensor}
  (shim _get/_set_like trong e5); update_recurrent_state .copy_() in-place
  phá autograd fla (monkey-patch rebind khi có grad); needle_items gọi
  token_stream per-item seed lớn = skip vạn doc (treo 20ph); dense-hook
  không công tắc → OOM @VAL (16×160MB/prefill); cell launch còn logic kill
  giết nhầm chuỗi khỏe. Hạ tầng chống chịu mới: checkpoint .last/50 bước +
  resume + bash retry 8 vòng — CHẠY THẬT (sống sót 2 OOM + 1 kill nhầm).
- Ba lối còn lại cho 27B: (a) weight-derived conjugation #1 (không học);
  (b) hai chặng 4B→9B(copy)→27B (chặng học ngắn hơn); (c) đóng ô 27B cho
  adapter, dồn lực Phase C. Chờ user chọn. → BỊ LẬT BỞI v3.1 (dưới).

**E6 v3.1 — ĐỘT PHÁ (2026-08-24): tìm ra và sửa lỗi loss → mapper 4B→27B
SỐNG THẬT. BFCL niêm phong: MAPPED 16/20** (`e6v3_ce.py` giao thức mới):

    Nguyên nhân gốc (user truy "kiểm tra trên train đi" + "loss CE chưa
    đúng?"): CONV_WARM=4 thừa kế E5 BỎ 4 TOKEN ĐẦU CỦA GOLD khỏi CE/KL —
    token quyết định (đầu tên hàm) KHÔNG BAO GIỜ được dạy; CE 0,008 của v3.0
    = chỉ giỏi phần đuôi khi được mớm. Falsification train-check xác nhận:
    v3.0 chỉ 1/30 NGAY TRÊN MẪU TRAIN.
    Fix v3.1: cache cắt T-5, warm conv bằng 5 token CUỐI PROMPT (token thật)
    → CE chấm TRỌN 100% gold, token đầu trọng số ×3; giao thức nhất quán
    train/val/test.
    Kết quả: val needle 0→1→8/10 (@449, cache chưa gặp — truyền được NỘI
    DUNG qua tường GDN); TEST NIÊM PHONG BFCL: self 20/20 | MAPPED 16/20 |
    no_ctx 0/20; train-check bfcl 7/15 needle 8/9.

    Bảng mốc chấm đúng (baselines đo 2026-08-24, e6v3_baselines.json):
    27B-self 20/20 & nk 10/10 | 4B-SELF 11/20 & nk 10/10 (thanh kinh tế) |
    v3.1 MAPPED 16/20 | v3.0 4/20 | mapper-init 0/20 | no_ctx 0/20.
    → MAPPED 16/20 VƯỢT thanh kinh tế 4B-self (11/20): cascade 4B-prefill →
    27B-decode có giá trị thật cho function-calling.

- Nợ đo: needle@2K niêm phong vẫn vô hiệu (TRAIN_MAX cắt câu hỏi) — cần
  chạy lại baselines script với mapper v3.1 max_len 4096 (~15ph GPU).
- Nợ upload (quy tắc 6d): mapper_v31.pt (best@449) + .last + 2 json — CHỜ
  TOKEN HF WRITE của user, runtime có thể recycle bất cứ lúc nào.
- Bài học phương pháp: phán quyết "lớp hàm không tổng quát hóa" của v3.0
  là SAI — chết bởi 1 dòng loss kế thừa; hai câu truy vấn của user (train-
  check + nghi loss) đã cứu cả mặt trận. Error-placement lần 6: CE trung
  bình che token quyết định.

**E6 v3.2 — SCALE-UP (user duyệt, 2026-08-24→25): BFCL 17/20, needle@2K lộ
vách đá độ dài** (`e6v3_ce.py`, data ×2,5, 2000 bước, e6v32_results.json):

    Data: BFCL 465 + ifstruct 135 (gold 96) + pbtable 110 (gold 64) +
    needle 350 (700/950); val 55 (gồm needle 1500); test niêm phong y cũ
    + needle@2K maxlen 4096 (LẦN ĐẦU chấm đúng).
    Val curve (best@1249): bfcl 3→6→9→10/15, needle 14-15/15 (CẢ nhóm
    1500 = 1,6× miền train); dừng đúng kỷ luật stale-3 @1999.
    TEST NIÊM PHONG: BFCL self 20/20 | MAPPED 17/20 | no_ctx 0.
    needle@2K: self 10/10 | MAPPED 1/10 — VÁCH ĐÁ giữa 1500 và 2000
    (retention-length law #6 hiện bằng số: GDN state tích lũy theo độ dài,
    cache 2K ngoài phân phối train ≤950).
    TRAIN-CHECK: bfcl 11/11, needle 12/12 (hoàn hảo — học quy luật thật).
    ifstruct/pbtable van 0 moi noi: sinh-cấu-trúc-dài chưa học được (gold
    96/64 không đủ chữa — cần mổ output, nợ v3.3).

- Trận OOM 22GB (transformers 5.15 GDN fp32 nặng ×2): phong bì train 1024;
  hạ tầng chống chịu hoàn chỉnh MỚI: skip-mẫu-OOM tự học (attempt.txt +
  skip.json), step toàn cục bền qua restart (gstep.txt + fast-forward
  cosine), val kiểu ngưỡng (bắt kịp mốc bị crash nhảy), phòng thủ chủ động
  ctx>850→gold 32. Sống qua ~10 lần ngã, không mất tiến độ.
- Scope 4B→27B sau v3.2: function-calling 85% trần + truy xuất tới ~1500
  token = DÙNG ĐƯỢC; >2K cần train ctx dài (GPU >22GB hoặc tiết kiệm bộ
  nhớ sâu hơn — hướng v3.3/Phase C).

**E6 v3.3 — TỐC ĐỘ + CTX DÀI + CHÍNH XÁC (user duyệt, 2026-08-24→25):
BFCL 18/20 + needle@2K 10/10 TUYỆT ĐỐI — vách đá độ dài SỤP** (commit
12cb470/7900c8b, e6v33_results.json):

    Kỹ thuật mới: (tốc độ) clone_cache_struct thay 2×deepcopy ~600MB/bước;
    Phase B1 tiền tính teacher 1 lần/item (top-64 logp + dense caps 4/16
    layer fp16 ra đĩa) → vòng train hết teacher feed-forward; aux chỉ khi
    λ>0. (ctx) checkpoint map_attn (recompute trong backward); needle
    curriculum train 700/950/1200/1600/2000. (chính xác) BFCL +parallel
    +multiple (655 bfcl train); trọng số token-xương ×2; val dump text.
    SANITY trước train (20 bước, toàn item ~2000 tok): 4,05 s/bước,
    peak 20,32/22,5 GiB → ctx-2000 KHẢ THI trên L4 (v3.2 OOM từ 1536);
    --gdn-bf16 loại bằng đo (chậm hơn 20%, tiết kiệm 0,05 GiB).
    Vận hành thật: ~2,2-2,4 s/bước (nhanh ~2× v3.2), 0 RETRY, 0 OOM-skip
    toàn chiến dịch — lần đầu một run E-series đi hết không ngã lần nào.
    Data: 1330 train / 70 val (needle bucket 700+1500+2000) / test y cũ.
    Val curve: score 12→29→29→32→36→37→40 (bfcl 6→9→11→12→16→17→19/25;
    needle 20/20 TỪ MỐC 2, @2000 = 5/5 suốt 6 mốc); ifstruct 1/15 lần
    đầu có điểm @1749. CE_FLOOR <0,2 kích hoạt @1750 → dừng sớm đúng
    kỷ luật, best-by-val giữ checkpoint score 40.
    TEST NIÊM PHONG: BFCL self 20/20 | MAPPED 18/20 (v3.2: 17, v3.1: 16,
    4B-self: 11) | no_ctx 0.
    needle@2K NIÊM PHONG: self 10/10 | MAPPED 10/10 | no_ctx 0 —
    v3.2 chỉ 1/10. Retention-length law xác nhận chiều THUẬN: vách đá
    là artifact của phân phối train, KHÔNG phải giới hạn của GDN state
    hay của mapper — cho state dài vào train là đọc được state dài.

- Mổ ifstruct/pbtable bằng val dumps (nợ v3.2 trả xong): cả hai chết cùng
  MỘT bệnh = repetition collapse khi sinh dài (>~30 token) — pbtable ra
  ĐÚNG khung `<tr><td>` (token-xương có tác dụng) nhưng lặp một ô vô hạn;
  ifstruct trôi vào <think> rồi kẹt vòng. BFCL (24 tok) và needle (16 tok)
  không dính vì sinh ngắn. Error-placement lần 7: bệnh ở decode-time tích
  lũy, không phải "không hiểu đề". Nghi vấn bổ sung: pseudo-gold ifstruct
  là output 27B tự sinh (nhiều bản mở đầu <think>) — có thể CHÍNH teacher
  cũng không qua validator trong 96 token → trần suite ≈ 0 từ đề bài.
  Việc v3.4: đo 27B-self trên val ifstruct + repetition penalty/chấm 30
  token đầu khi eval, trước khi đổ lỗi cho mapper.
- Scope 4B→27B sau v3.3: function-calling 90% trần + truy xuất ≤2000 token
  TUYỆT ĐỐI = cascade có giá trị sản phẩm rõ; cửa còn lại là Phase C
  (KVConnector vLLM + đo trên W4A16 Marlin thật) và sinh-dài (v3.4).
- NỢ UPLOAD (quy tắc 6d): mapper_v33.pt (+v32) vẫn CHỈ nằm trên runtime —
  Colab Secrets HF_TOKEN chưa set tại thời điểm chốt (đã nhắc user 3 lần).
  → THÀNH HỌC PHÍ LẦN 2: runtime recycle 2026-08-25 nuốt cả v33 lẫn v32.
  Hệ quả: token flow đổi (user chốt: .env root repo, dùng trực tiếp) +
  script TỰ upload mỗi mốc val (--hf-repo) — xem E6 v3.4.

**E6 v3.4-long — TĂNG SEQUENCE LENGTH (user duyệt 2026-08-25): TÁI LẬP
18/20 BFCL + needle NIÊM PHONG 15/15 TUYỆT ĐỐI (10@2K + 5@4K)** (commit
edf9232→3ed64d4, e6v34_results.json, HF v34/):

    Ky thuat (triet ly Unsloth may do lai): prefill_chunked (transient
    doc lap T); TEMPLATE-XUONG — B1 luu cache_meta (shape/int, khong
    tensor), train dung lai template bang zeros khi khong can aux →
    BO HAN teacher prefill moi buoc; tpl-check 8/8 item @4K:
    max|dlogit| 0.0000, argmax-agree 1.000 (dung tung bit).
    LADDER (L4 22GiB): 4096 = 4,86s/buoc peak 21,26 GiB spill 155MB/item
    → GO; 8192/16384 OOM = TRAN PHAN CUNG L4 (khong phai code).
    16K tren L4: KHONG — can GPU lon hon (A100 40GB) neu muon.
    Van hanh: ~1,4 s/buoc khi full template-path (nhanh 40% hon v3.3 du
    data dai gap doi); 3 OOM dau (combo prefill-that+aux+student o item
    dai) → fix 3ed64d4: item >2500 di template tu buoc 0, skip.json xoa
    1 lan; sau do 0 nga. AUTO-UPLOAD HF moi moc val chay that (best +
    .last + results) — recycle giua train gio chi mat ≤250 buoc.
    Data: 1330 train (needle buckets 700..4000) / 73 val / test 35 =
    20 BFCL cu + 10 needle@2K cu + 5 needle@4K MOI (niem phong).
    Val: 18→27→32→30→36→39→39→38 (needle 23/23 TUYET DOI tu moc 1249 —
    moi bucket 700/1500/2000/4000); CE_FLOOR dung @2016.
    TEST NIEM PHONG: BFCL self 20/20 | MAPPED 18/20 (= v3.3, tai lap
    sau khi mat sach checkpoint — cong thuc la that, khong phai may);
    needle self 15/15 | MAPPED 15/15 | no_ctx 0 — gom TRON 5 de @4K
    chua tung gap.

- Ket luan v3.4: (1) retention-length law xac nhan chieu thuan LAN 2 —
  train toi dau doc toi do, 4K = tran L4 chu khong phai tran phuong phap;
  (2) template-xuong la phat kien tai su dung duoc cho Phase C (dung
  template khong can teacher = KVConnector khong can chay 27B prefill);
  (3) ifstruct/pbtable van 0 (benh sinh-dai decode-time, thuoc o v3.5);
  (4) chuoi tu dong hoa tron ven: recon → chon nac → train → cuu ho →
  niem phong khong cham tay.

**E6 v3.5 — MỔ BỆNH SINH-DÀI (user chốt "C, v3.5", 2026-08-25): TRẦN
TEACHER XÁC NHẬN — suite ifstruct/pbtable không đo được mapper** (25 item
val × 4 điều kiện, mapper v34 best, HF v35/e6v35_decode.json):

    suite      | 27B-SELF | self+rp1.3 | MAPPED | map+rp1.3
    ifstruct   |   1/15   |    0/15    |  0/15  |   0/15
    pbtable    |   2/10   |    0/10    |  0/10  |   0/10

- **Phán quyết 1 — trần teacher**: chính 27B tự đọc prompt cũng chỉ đạt
  4-8% dưới cùng giao thức (GEN_LEN 160/120, validator/head-match).
  Mapper 0 vs teacher 1-2 = mapper BÁM SÁT TRẦN, không phải hỏng riêng.
  Món nợ "ifstruct/pbtable 0 điểm" từ v3.2 là nợ của ĐỀ (pseudo-gold
  nhiễm <think>, GEN_LEN quá ngắn cho câu trả lời thật, validator đòi
  fmt+key trong cửa sổ hẹp), không phải của model/mapper.
- **Phán quyết 2 — repetition-penalty 1.3 PHẢN TÁC DỤNG**: self-rp 0 <
  self 1-2. Lý do nhìn từ dumps cũ: output hợp lệ của các suite này vốn
  LẶP CẤU TRÚC (bảng lặp <td>, JSON lặp dấu ") — penalty đè chết đúng
  token-xương. "Rep-penalty phá vòng lặp" nghe hợp lý nhưng đo ra sai —
  đo-hơn-suy-luận thêm một lần.
- Hành động: LOẠI ifstruct/pbtable khỏi thang điểm chính thức của mapper
  (giữ file kết quả để tái kiểm); scope sản phẩm cascade giữ nguyên =
  function-calling 90% trần + retrieval ≤4K tuyệt đối. Muốn đo sinh-dài
  tử tế cần suite mới: gold THẬT (không pseudo), GEN_LEN đủ dài, chấm
  khớp nội dung — để ngỏ, không chặn Phase C.

## PHASE C (2026-08-25→26): cross-model KV vào vLLM serving — C2a→C2b-7, verdict interim

Thiết kế `docs/phase-c-design.md`: 2 vLLM + LMCache MP, vá key-namespace
(`model_name="qwen35-shared"` trong lmcache_mp_connector), 1-GPU tuần tự
qua kho lmcache sống xuyên restart. Kết quả HF `c2b/`..`c2b7/`.

**C2a (3/3 PASS)**: block size 4B = 9B = 1056; boot sạch; lmcache external
0.5.4 import OK.

**Chuỗi phân xử C2b (7 lượt, mỗi lượt một biến)**:

    Lượt | Biến               | Hit?      | Cross needle | Chữ ký
    1    | fp8 KV             | chỉ 30K   | 0/2 miền hit | RÁC thuần
    2    | bf16, L1 8GB       | KHÔNG     | (giả sạch)   | kho tràn evict
    3    | bf16, L1 20GB      | toàn tuyến| 0/6→ttft×16  | gần đúng rồi đứt
    4    | + align rem 3-5    | toàn tuyến| 1/4          | ca sạch ĐẦU TIÊN
    5    | + pad sạch, rem=2  | toàn tuyến| 1/4          | 4-6 số đầu ĐÚNG cả 4 ca
    6    | consumer 9B GỐC    | toàn tuyến| 1/4          | GIỐNG HỆT từng ký tự
    7    | producer 4B bf16   | toàn tuyến| 1/4          | GIỐNG HỆT

**Đã chốt bằng đo**: (a) cơ chế vận chuyển HOÀN CHỈNH — hit mọi độ dài,
TTFT 30K 11-24s → ~1s (×12-16); (b) fp8-scale là tầng lỗi thật (rác vs
gần-đúng) — bf16 KV là điều kiện; (c) block-align cần để hit; (d) LOẠI:
suffix-re-prefill (rem=2), champion-graft (stock giống hệt), producer
W4A16 (bf16 giống hệt). **Bất biến còn lại**: mọi biến thể đều lấy đúng
4-6 chữ số đầu rồi degenerate lặp.

**Giả thuyết tầng lỗi thật (khớp mọi bằng chứng, CHƯA kiểm)**: trang
GDN-state KHÔNG được truyền/áp — chỉ attention KV sang được. Khớp E0
(context sống ở CẢ GDN lẫn KV — thiếu một là chết) + E7 (attention thẳng
hàng toàn họ → retrieval tức thời 1-2 token đầu vẫn chạy bằng attention)
+ bất biến qua mọi dtype/checkpoint. Chẩn đoán kế (chưa chạy): instrument
key/nhóm object GDN trong lmcache store — xem nhóm `--separate-object-
groups` có ghi/đọc trang mamba dưới key nào, consumer có lookup trúng.

**Giá trị đã giao được của Phase C interim**: đường ống sản phẩm đầu
tiên chạy trọn (patch 1 dòng lmcache + EXTRA_FLAGS/KV_DTYPE trong run.sh
+ c2b_gates.py làm harness 3 cổng tái dùng); con số TTFT ×12-16 là trần
tốc độ thật khi chất lượng được giải.

**Bài học hạ tầng mới (đã trả giá)**: 2 tiến trình chia CUDA-IPC phải
CÙNG torch (lmcache chạy SAU run.sh setup); cổng 8080 Colab chiếm
(--http-port 8081); pkill cha không giết con giữ cổng → kill theo chủ
cổng (ss -tlnp + /proc/PID/cmdline); pkill -f 'vllm serve' TỰ KHỚP bash
chứa pattern → dùng '[e]'; 2-writer-1-log = null bytes (python con sống
sót kill cha vẫn giữ fd); L1 lmcache = pinned RAM cấp háo hức lúc boot.

## PHASE C VERDICT CUỐI (2026-08-26, sau lab-check + khám kho + C2b-8)

Ba lượt cuối (user chỉ đạo "kiểm ngoài vLLM trước"):

    Lab-check (transformers bf16, protocol E1, CÙNG bộ đề): self 4/4,
      copy-nguyên 3/4 MÃ TRỌN (ca 30K-2 degenerate '9346666').
    Khám kho (đầu dò OBJGRP-PROBE): plan chạy ĐỐI XỨNG 2 nhóm object
      76/76 (attention + mamba/GDN), cả store lẫn retrieve → giả thuyết
      "trang GDN rơi" BỊ BÁC.
    C2b-8 (vLLM consumer 9B bf16-WEIGHTS, 8K-only vì 30K không vừa L4
      bf16): self 2/2, cross 1/2 — ca 0 vẫn '4398' cụt, GIỐNG W4A16
      → giả thuyết "W4A16-consumer" CŨNG BỊ BÁC.

**Kết luận cuối**: đã loại HẾT các giả thuyết "một con bug rời rạc"
(fp8-scale*, block-align*, suffix-re-prefill, graft, producer-W4A16,
GDN-trang-rơi, consumer-W4A16). [* = hai cái này là điều kiện cần thật,
đã vá]. Cái còn lại là ĐỊNH LUẬT BIÊN MỎNG: cross-model cache đặt decode
đứng sát mép vực số học — mọi nhiễu nhỏ (khác kernel GDN vLLM vs fla,
roundtrip trang, lượng tử hóa) lật các ca cận biên. Bằng chứng chuỗi:
cùng ca 0, lab-fla ra mã trọn còn vLLM-kernel ra 4 số; junk '<|1|1|'
sau mã xuất hiện Ở CẢ LAB — biên vốn mỏng sẵn (E6c đã đo nửa vết nứt
là lượng tử hóa, E2 đã đo retention giảm theo độ dài trên đề khó).

**Hệ quả sản phẩm**: (1) exact-retrieval dài qua cross-serving = CHƯA
bán được, cần gia cố biên (polisher phía consumer — đúng công nghệ
mapper/calibration đã có, E6-series); (2) scope AN TOÀN từ E2-E3 vẫn
đứng: chat/QA/RAG ngữ nghĩa (không đòi khớp chuỗi chính xác) — chưa đo
lại trên serving, là bài đo kế tiếp hợp lý; (3) transport stack đã
ĐÚNG và ĐỦ (bf16 KV + block-align + L1 đủ + cùng-torch): TTFT ×12-16
là trần thật, kho vận hành chuẩn — mọi thứ sẵn sàng cho ngày biên được
gia cố; (4) 4B→27B batch (transformers, mapper v3.4) KHÔNG bị ảnh
hưởng — vẫn là đường sản phẩm sạch nhất hiện tại.

## C2b-N (2026-08-26, user: "3/4 quá ít samples, thử 2-300") — N=240 @8K

Hạ tầng: gen-n 240 prompt aligned (rem 2-4), self-baseline 1 lượt +
10 wave × 24 (4B bf16 kv_producer ghi kho → xả → 9B champion kv_consumer
đọc; kho L1 20GB chỉ chứa ~30 vở 8K nên phải wave). Kết quả HF c2bN/.

    Self (champion tự đọc):  240/240 = 100,0%   p50 4,33s
    Cross (vở 4B qua kho):   137/240 = 57,1%    p50 1,58s (×2,7 @8K)
    95% CI cross: ~[51%, 63%]

- Tỷ lệ THẬT của cross-serving exact-retrieval @8K là **~57%** — các mẫu
  nhỏ trước đó (1/4=25%, lab 3/4=75%) đều là nhiễu quanh dải này.
  KHÔNG phải vách đá về 0, cũng không gần-hoàn-hảo: đúng hình dạng của
  hiện tượng BIÊN MỎNG xác suất — mỗi ca có biên riêng, ~57% ca sống
  qua nhiễu kernel/roundtrip, phần còn lại rơi.
- Kèm tốc độ ×2,7 ở 8K (×12-16 ở 30K đã đo): bức tranh kinh tế rõ —
  polisher cần kéo 57% → ~95%+ để bán exact-retrieval; hoặc bán ngay
  cho workload chấp nhận fallback (miss thì tự prefill lại — hybrid
  "thử cross trước, fail thì cold" vẫn lời lớn về TTFT trung bình).
- kv_role kv_producer/kv_consumer chạy sạch (lần đầu dùng, 10 wave 0 lỗi
  role); toàn chuỗi 2h tự động không ngã.

## MAPPER 4B→27B — CROSS trên benchmark ngoài: KẾT QUẢ CUỐI (2026-08-28)

Chuỗi đầy đủ user yêu cầu ("thử thuần 27B, sau đó apply training và re-test,
để thấy tổng quát khả năng của mapper"). Mapper train xong @ctx4096
(warm-start v34, dừng sớm CE_FLOOR @1075, val curve 27→29→34→38→39).

**Test NIÊM PHONG (miền ĐÃ train) — tái lập kỷ lục:**

    bfcl:     self 20/20 | mapped 18/20 (90%) | no_ctx 0/20
    needle2K: self 15/15 | mapped 15/15 (100%) | no_ctx 0/15

**Benchmark NGOÀI (miền HOÀN TOÀN MỚI) — sụp đổ, kèm đối chứng 4B:**

    bo       n    27B-self   cross-mapper   giữ được   4B-self
    bbh    182      53,8%          6,0%        11%      34,6%
    gsm8k   50*     80,0%          0,0%         0%      81,5%
    musr   198      58,1%          1,5%       2,6%       4,5%
    (* gsm8k cross cắt ở 50 mẫu sau 0/46 — CI trên ~7,7%, không đáng
       tốn thêm 2,4h GPU cho kết quả đã biết)

**ĐỐI CHỨNG 4B-self là chìa khóa giải thích** (phép đo quyết định, rẻ ~1h):
- gsm8k: 4B-self **81,5%** — CAO HƠN cả 27B (80,0%)! Thông tin để giải
  toán NẰM SẴN trong cache 4B, model 4B tự đọc cache của mình làm được
  163/200. Nhưng cache đó đi qua mapper sang 27B thì còn **0/50**.
  → Lỗi HOÀN TOÀN ở khâu DỊCH của mapper, không phải giới hạn model nguồn.
- bbh: 4B-self 34,6% vs cross 6,0% — cùng kết luận.
- musr: 4B-self chỉ 4,5% (4B không làm được bài suy luận truyện dài) →
  riêng bộ này cache nguồn vốn nghèo, cross 1,5% không nói lên nhiều.

**Chất lượng sinh (self rác 0,2% → mọi rác đều do mapper):**

    chỉ số            self     cross    chênh
    đúng             58,8%      3,3%   -55,6%
    sinh rác          0,2%     15,6%   +15,3%
    bị cắt            4,9%     19,8%   +14,9%
    không ra đáp án   6,5%     55,1%   +48,6%
    lặp trigram TB    0,015     0,186   +0,172

**Phân loại 243 ca self-đúng→cross-sai** (bench_analyze chỉ bắt được
nhánh 1; nhánh 2 phải ĐỌC TAY vì văn bản hoàn toàn sạch):
1. **SINH RÁC / degenerate: 53 ca (22%)** — lặp vô hạn
   ("The probability of the event is 1/2." ×4), chuỗi số dài
   ("1000000...", rep=35), gsm8k rác tới 98%.
2. **LẠC ĐỀ nhưng MẠCH LẠC: phần lớn 78% còn lại** — nguy hiểm hơn vì
   trông như câu trả lời thật:
   - hỏi biểu thức boolean → "xác suất tung xúc xắc bằng 7 là..."
   - hỏi bảng chim cánh cụt → "chiều cao trung bình của người là 170cm"
   - hỏi án mạng (MuSR) → "bài hát The Troubadour của The Beatles"
   - hallu tên riêng 0,63/mẫu ở musr (self: 0,04) — bịa Beatles, Prada,
     United States… hoàn toàn không có trong đề.
3. musr thêm dấu hiệu: 94,4% "không ra đáp án", 39,9% bị cắt trước khi
   trả lời — cache không dẫn model tới định dạng câu trả lời.

**KẾT LUẬN KHOA HỌC**: mapper 35M **KHÔNG học "cách dịch cache" tổng
quát — chỉ học được ánh xạ CHO MIỀN CỤ THỂ đã train**. Bằng chứng:
90% BFCL / 100% needle trong miền, nhưng 0-6% ngoài miền, trong khi
đối chứng chứng minh thông tin CÓ sẵn trong cache nguồn (gsm8k 81,5%).
Đây là phán quyết lần 4 về giới hạn LỚP HÀM của mapper (xem E6 v3, v3.1),
lần này có thêm đối chứng 4B-self nên loại trừ được giả thuyết "model
nguồn yếu".

## KIẾN TRÚC 2 LỚP (LoRA-trên-4B + mapper) — CỔNG VRAM: MỞ, KHE HẸP (2026-08-28)

User chốt hướng: "1 cải tiến 2 lớp mapper, 1 gắn vào 4b để ép 4b cho việc
đọc cho 27B sau đó merge cái đó vào 4b, sau đó huấn luyện mapper dịch 4-27,
tổng quát toàn bộ dataset". Về sản phẩm đây là hướng đúng: sau khi merge
LoRA, lúc serve KHÔNG có module thừa trên đường nóng — ta bán một biến thể
"Qwen3.5-4B-prefill-for-27B".

Ràng buộc kỹ thuật: để LoRA trên 4B học được, gradient phải chảy từ output
27B QUA mapper VÀO 4B → cả hai model phải cùng trên GPU KÈM đồ thị autograd.
Đường tránh duy nhất (loss khớp-trạng-thái, chỉ cần 4B + cache 27B đọc từ
đĩa) đã CHẾT 4 LẦN trong dự án (E8 v3 nMSE 10,5→1,06 kẹt, needle 0/5; luật
error-placement) — không đi lại. Nên phải đo, `probe_joint_lora.py`.

**Kết quả 5 lượt đo (mọi lượt đều PYTORCH_CUDA_ALLOC_CONF=expandable_segments):**

    hai model cung tren GPU (27B 4bit CPU-offload + 4B 4bit + LoRA r=16)
      = 16,16 GiB tinh  ->  con TRONG 5,54 GiB

    chang                         T=256      T=512     T=1024
    nen (2 model)                 16,16      16,16      16,16
    sau prefill 4B CO GRAD        19,50      OOM        OOM
    sau template 27B              19,66       -          -
    sau mapper                    20,22       -          -
    sau forward 27B                OOM        -          -

**Thủ phạm được chỉ mặt: prefill 4B CÓ GRAD ăn +3,34 GiB ngay ở T=256**, và
vượt hẳn 5,54 GiB khả dụng ở T=512. Ngay cả ở T=256 — mức vô dụng cho mọi
bài thật (prompt bbh/gsm8k/musr dài tới 1512 token) — thì forward 27B sau đó
vẫn OOM với 1,45 GiB còn lại. Thiếu KHÔNG PHẢI một chút: để chạy T=2048 cần
thêm hàng chục GiB.

**3 đòn bẩy đã thử và kết cục:**
1. `expandable_segments:True` — bật ở mọi lượt, KHÔNG cứu (peak 21,65-21,76
   ổn định qua mọi cấu hình → không phải bài phân mảnh).
2. **Gradient checkpointing trên 4B — BẤT KHẢ THI VỀ NGUYÊN TẮC, không phải
   vì thiếu bộ nhớ**: transformers tắt cứng `use_cache=False` khi bật GC
   (log: "`use_cache=True` is incompatible with gradient checkpointing"), mà
   forward 4B ở đây TỒN TẠI CHÍNH ĐỂ sinh cache → hai thứ loại trừ nhau. Đây
   là cơ chế KHÁC hẳn lần đóng GC cho 27B hôm 2026-08-27 (lần đó GC chạy
   nhưng peak y hệt baseline). Bài học chung: GC và cache-as-output không
   sống chung.
3. CPU-offload embed/lm_head của 27B — có hiệu lực (27B tĩnh 12,7 GiB đúng
   như đo 2026-08-27), đã tính vào con số 16,16 ở trên.

**Bài học hạ tầng**: `ps -eo pid,etimes,stat,pcpu` KHÔNG có cột lệnh nên mọi
`grep` tên tiến trình đều rỗng — tưởng job chết trong khi nó đang chạy. Phải
có `args` trong format.

**ĐÒN BẨY THỨ 4 MỞ ĐƯỢC CỔNG: TBPTT (cắt lan truyền ngược theo thời gian).**
Vì GC bị loại về nguyên tắc, dùng cách chặn khác cho ĐÚNG thủ phạm đã chỉ mặt:
T−w token đầu chạy `no_grad` (chỉ lấy GIÁ TRỊ state), w token cuối chạy có
grad → bộ nhớ activation phụ thuộc w, không phụ thuộc T (`prefill_tbptt`).

    voi --tbptt 128:
    chang                        T=512     T=1024     T=2048
    nen (2 model)                16,16     16,18      16,18
    sau prefill 4B (TBPTT)       17,92     18,01      18,15   (+1,8-2,0 GiB;
                                                               truoc: +3,34 o T=256)
    sau template 27B             18,09     18,22      18,42
    sau mapper                   18,67     18,83      19,10
    sau forward 27B              20,99     21,34       OOM
    sau backward (peak)          21,18     21,50        -
    ket qua                        OK        OK        OOM
    t/buoc                        6,1s      6,0s         -
    grad toi LoRA / mapper       60 / 208/208  (day du — duong autograd THONG)

**PHÁN QUYẾT CỔNG: ĐI ĐƯỢC, nhưng khe rất hẹp** — trần ctx **1024**, còn
trống 0,09 GiB ở đỉnh backward tại T=1024 (T=2048 OOM). Kiểm chứng đường
autograd thông suốt: gradient tới đủ 60 tensor LoRA và 208/208 tham số mapper.

**3 cái giá phải trả, không được lấp liếm:**
1. **TBPTT là XẤP XỈ**: LoRA chỉ nhận gradient từ 128 vị trí CUỐI. Với GDN
   (hồi quy) đây là TBPTT kinh điển, chấp nhận được; với attention thì K/V
   của token cũ thành hằng số — LoRA không học được từ chúng.
2. **Trần ctx 1024 cắt cụt MuSR** (narrative 1000-1500 token). gsm8k (~200)
   và bbh (~200) thì vừa.
3. **6,0 s/bước** vs ~1,4 s/bước của mapper-đơn → 1000 bước ≈ 1,7h.

Trên A100 40GB cả 3 cái giá này biến mất; đây là giới hạn của L4, không phải
của ý tưởng.

**BAO DO DAY DU 42 CAU HINH (user: "tang tran 1024-2048")** — quet
ctx x gold x tbptt. Phat hien LON: **gold moi la rang buoc chinh, khong phai
ctx**. gold = SO VI TRI feed vao 27B; moi vi tri giu them state GDN cho
backward (48 lop GDN) nen no dat hon ctx nhieu lan.

    T=2048: gold 16 OK (w=64) | gold 64,128,256 OOM voi MOI w
    T=1536: gold 16 OK (w=64) | gold 64+ OOM voi MOI w
    T=1024: gold 24/32/48 OK  | gold 64+ OOM   (peak 20,80/20,95/21,24)
    T= 512: gold 24/32/48 OK  | gold 64+ OOM   (peak 20,48/20,62/20,92)
    T= 256: gold 24/32/48 OK  | gold 64+ OOM   (peak 20,32/20,46/20,75)
    tbptt : w=128 OOM tu T>=1536; w=64 chay tron toi 2048

Tuong CUNG o gold=64 voi MOI T (khong tuyen tinh — la nguong cap phat).
BAO DO CHOT: gold <= 48 khi ctx <= 1024, gold <= 16 khi ctx 1536-2048, w=64.

**Hau qua that cho data**: gold gsm8k la LOI GIAI DAY DU ~150-250 token ->
KHONG day duoc tron trong che do joint tren L4; bi cat con 48. Bien minh duy
nhat (va no that): luat bien mong noi cross-cache hong ngay o vai token DAU
— giam sat 48 token dau nham DUNG cho vo. Nhung day la MAT MAT, phai ghi
nhan, khong phai lua chon toi uu.

**GIAI THICH cau hoi user "sao hom truoc dat 8192 ma gio 2048 OOM":** hai
cau hinh KHAC NHAU, khong mau thuan. Do 27/08 la 27B MOT MINH (hai pha, 4B
da spill cache ra dia roi xa) -> nen tinh 12,81 GiB, con trong ~9,5. Bay gio
4B phai O LAI kem do thi autograd -> nen tinh 16,16 GiB, con trong 5,54.
Phep cong noi het: 18,11 (dinh 27B mot minh @8192) + 3,5 (trong so 4B) =
21,6 GiB — cham tran 22,3 TRUOC KHI tinh mot byte activation nao cua 4B.
8192 la tran cua kien truc 1 model thuong tru; 2048 la tran cua kien truc
2 model + gradient xuyen qua ca hai. Quay lai mapper-don thi 8192 co ngay.

**HAI BUG THAT bat duoc khi dung `e9_joint.py`:**
1. `ifstruct` KHONG co truong `gold` — trong e6v3_ce gold cua no do 27B TU
   SINH o buoc B0 (pseudo-gold); che do joint khong co B0 -> `tok(None)` nem
   "You need to specify either `text` or `text_target`". Loai 135 train +
   15 val, CO GHI SO LUONG (v3.5 da bo ifstruct khoi thang chinh nen khong
   tiec).
2. **CAT TRAI khi tokenize** — bug nay KHONG BAO LOI neu bo qua: moi prompt
   bo nay dat CAU HOI O CUOI (musr: narrative 1000-1500 token roi moi hoi).
   `truncation=True` mac dinh cat PHAI = an mat cau hoi, item thanh vo nghia
   ma van chay tron. Bat buoc `truncation_side='left'`.
Mot bay nua da chan truoc: template 27B suy tu meta goc PHAI tu kiem bang
meta THAT (`--verify-meta`) — da chay, KHOP o T=256/512/1024/2048.

**TRAIN THAT DA PHONG (2026-08-28)**: `e9_joint.py`, warm-start mapper
v427_4k, data 7768 train / 330 val (bfcl 655 + needle 380 + pbtable 110 +
gsm8k 2882 + bbh 2500 + musr 474 + suite 767), ctx 2048, tbptt 64, 2500
buoc x ~5,8 s/buoc ~ 4h, val moi 250 buoc, auto-upload HF `joint_v1/`.
Kiem ro ri: **0/6898 item trung 580 mau niem phong** (doi chieu chuoi
prompt, chay tren chinh runtime train). Sanity 30 buoc: ce 1,497 |
5,76 s/buoc | peak 20,91 GiB | 0 buoc OOM.

## JOINT 4->9 LUOT 1 + PSEUDO-GOLD (2026-08-28)

**Luot 1 (joint49e) XONG**: warm-start v49, ctx 4096, tbptt 128, gold-cap 256,
2000 buoc. BEST @1750 score 16. Val 40 mau (self do CUNG item):

    bo        self   @1750
    needle 4     4      4    <- cham tran
    bfcl   4     4      4    <- cham tran
    bbh   10     2      4
    musr   5     1      3
    gsm8k 13     7      0    <- hut hoan toan

**HAI BUG THAT bat duoc bang dem, khong bang doan:**
1. Nguong `gold_ids.shape[1] < 2` nem BO moi item gold DUNG 1 token. Dem bang
   tokenizer tren 6623 item: **musr 474/474 = 100%, bbh 634/2500 = 25,4%,
   tong 1108 = 16,7%** khong bao gio gop gradient. Dap an trac nghiem ("A")
   va dap an ngan bbh ("valid"/so) deu 1 token. => cot musr trong val luot 1
   KHONG phai cong cua huan luyen (mapper chua tung thay mot mau musr nao).
2. `continue` khi bo item nhay LUON qua khoi val cuoi vong -> moc val roi
   trung item bi bo la MAT HAN, im lang (moc 1500 bien mat khoi log). Thay
   bang co `skipped` + dem `n_skip`.
Vá thêm: cache cot `self` trong val (self KHONG doi theo buoc train — do that:
y het nhau o ca 5 moc — nhung dang tinh lai moi lan = vut 50% chi phi val);
bo val cuoi bi lap.

**PSEUDO-GOLD bang vLLM OFFLINE (user: "cho no hoc ca buoc reasoning cua
model large", "can map gan 9b nhat", "sao ko dung vllm mode offline")**

Dich hoc doi tu dap an NGUOI VIET sang **quy dao 9B TU DI** -> dich trung voi
thuoc do (retention = mapped/self). Chi giu dau ra CHAM DIEM DUNG; sai thi
giu gold tham chieu => khong mat mau, khong day mapper suy luan sai.

vLLM offline thay transformers: **577 tok/s vs 11,8 tok/s = nhanh 49 lan**
(transformers bnb-4bit batch1 eager vs vLLM 0.27.1 + bitsandbytes + continuous
batching + CUDA graph). 671.327 token trong **30,7 phut** thay vi ~9 gio.
Teacher dung `quantization=bitsandbytes` tren stock 9B de KHOP model dang lam
tran self. Unsloth FastLanguageModel da dong tu E8 (past=None khi training).

    bo       n mau   9B tu lam dung   ty le
    gsm8k     3000       2583         86,1%
    bbh       2600        797         30,7%
    musr       498        194         39,0%
    TONG      6098       3574         58,6%

**BA CON SO NAY LAT KET QUA VAL 40 MAU:**

    bo       val 40 mau noi   do that (n lon)
    gsm8k    54% (7/13)       86,1% (3000)
    bbh      20% (2/10)       30,7% (2600)
    musr     20% (1/5)        39,0%  (498)

Ca ba deu lech nang — khoang tin cay 95% cua 7/13 trai tu 25% den 81%. User
dung khi doi eval vai nghin mau. Hau qua: tran that cua gsm8k la 86% chu
khong phai 54%, nen khoang hut cua mapper (0/13) LON HON tuong.

Doi chieu ba model tren cung bai gsm8k: 4B 81,5% (200) | 27B 80,0% (200) |
**9B 86,1% (3000)**.

**Luot 2 (joint49p) DA PHONG**: pseudo-gold 3574 item + 2 bug da va + val-n
100, warm-start tu best cua luot 1, 2500 buoc.

**KHE HO TRAIN/SERVE (user hoi, da duyet do)**: mapper hoc tren stock 9B +
bnb NF4 nhung production chay **champion** (khung W4A16 + trong so GDN ghep
tu GGUF Q4_K_M) — khac CA luong tu LAN trong so. Bang chung cu MAU THUAN:
E6c do bnb ganh NUA vet nut (bf16 9/20 vs bnb 4-6/20); C2b-8 do bf16 va
W4A16 GIONG HET. Chua lan nao do tren mapper DA huan luyen. Da them
`ext_bench --tgt-quant {bnb,auto,bf16}` cho ca run_self lan run_cross (self
phai cung dang luong tu voi cross, neu khong retention la so sanh hai model
khac nhau). Duong thu 3 vua duoc chung minh kha thi trong luot pseudo-gold:
**vLLM nap duoc stock 9B voi bitsandbytes** (7,81 GiB, ctx 4096, FlashAttn 2,
KV 201.821 token) -> co the SERVE dung model da train, xoa sach khe ho, doi
lai toc do so voi Marlin W4A16 (chua do).


**CHUYEN HUONG SANG CAP 4->9 (user 2026-08-28: "khoan dung idea moi ve
mapper tren 4-27, ma dung cap 4-9 de dam bao chat luong da").** Da dung job
4->27 va do lai bao cho 4->9. Ly do vat ly: 9B bnb-4bit + CPU-offload chi
~4,4 GiB (27B: 12,7) -> nen 2 model 6,86 GiB, TRONG 14,88 (27B: 16,16 va
5,54). Bao do 4->9 (12 cau hinh, tbptt=128 — w=0 tuc lan truyen nguoc DAY DU
van OOM o moi ctx):

    ctx      gold 48            gold 256 (loi giai gsm8k tron ven)
    4096   11,25 GiB / 6,7s     15,70 GiB / 13,1s
    8192   12,68 GiB / 8,8s     17,04 GiB / 15,0s
   16384   15,36 GiB / 12,8s    19,71 GiB / 19,1s

**MOI cau hinh deu chay** — 4->9 lay lai duoc CA ctx dai LAN gold day du,
dung hai thu nhanh 27B phai hy sinh (o 27B: gold >=64 OOM voi MOI T, ctx
tran 2048). Day la ly do ky thuat vi sao huong user chot la dung.

**TRAIN JOINT 4->9 DA PHONG**: `--tgt-model Qwen/Qwen3.5-9B`, ctx 8192
(chon 8192 thay vi 16384 de con bien 4,5 GiB thay vi 1,8), tbptt 128,
gold-cap 256, warm-start mapper v49 (92% BFCL / 100% needle), 2000 buoc,
val moi 250, auto-upload HF `joint49/`. verify-meta KHOP o T=512 va 4096.
Buoc 20: ce 1,4754 | **4,79 s/buoc** | peak 12,05 GiB — nhanh HON ca nhanh
27B (5,76 s/buoc) du ctx gap 4 va gold gap 5.


**HIỆU CHỈNH NGAY SAU ĐÓ (user, 2026-08-28) — kết luận trên VƯỢT DỮ LIỆU.**
User hỏi: "có train mapper cho math/reasoning không? nếu chỉ train BFCL thì
fail task khác là đúng rồi". Kiểm `build_data()` trong `e6v3_ce.py`:

    train ~1330 item @max_ctx=4096:
      bfcl exec_simple 80 + simple 385 + parallel ~95 + multiple ~95 = ~655
      needle (curriculum 700..4000) .......................... 430
      ifstruct ............................................... 135
      pbtable ................................................ 110
    math: 0 | reasoning nhiều bước: 0 | QA văn xuôi tự nhiên: 0

Thêm một lệch nữa quan trọng không kém MIỀN: **chế độ sinh**. Mọi target
train là NGẮN-TRÍCH (GEN_LEN bfcl 24, needle 16 token) — mapper chưa bao giờ
bị ép giữ cache sống qua vài trăm token sinh. bbh/gsm8k/musr đòi đúng thế
(N_NEW 48/320/24 + chain-of-thought). Vậy cross hỏng có thể là "cache trôi
sau ~30 token sinh" chứ không phải "không dịch được ngữ nghĩa toán".

→ Thí nghiệm như đã chạy KHÔNG PHÂN GIẢI được 2 giả thuyết:
  H1 lớp hàm mapper (tuyến tính per-layer) quá yếu để tổng quát
  H2 miền train + chế độ sinh quá hẹp
Đối chứng 4B-self (gsm8k 81,5%) chỉ loại được H0 "model nguồn yếu", KHÔNG
tách được H1/H2. Câu "phán quyết lần 4 về LỚP HÀM" phải hạ xuống
**"chưa xác định — cần đo H2 trước"**. Ghi lại như một lần suy luận vượt
số đo (đúng thứ quy tắc 5 cấm).

Phép đo phân giải đã đề xuất (chờ user duyệt): train lại CÙNG mapper, CÙNG
siêu tham số, chỉ đổi DATA (thêm gsm8k/bbh train-split + suite_gen.py 4 họ
rag/mid/math/swe đã dựng sẵn từ 2026-08-26 nhưng chưa từng dùng để TRAIN) và
nâng GEN_LEN cho item suy luận; rồi đo lại ĐÚNG bộ bbh/gsm8k/musr này.
Cross tăng rõ → H2 (sửa được bằng data). Vẫn ~0 → khi đó H1 mới thành
phán quyết có cơ sở.

**Hàm ý sản phẩm**: cascade 4→27B dùng được TRONG miền đã train
(function-calling, retrieval needle ≤4K) — đó vẫn là sản phẩm thật với
số đo vững. Nhưng KHÔNG bán được như "tăng tốc đa dụng". Muốn tổng quát:
phải đa dạng hóa mạnh miền train (thêm reasoning/math/QA vào data), hoặc
đổi lớp hàm mapper (hiện chỉ là ánh xạ tuyến tính per-layer).

Kết quả thô + phân tích: HF `extbench_cross/` (cross 3 bộ, self4b 3 bộ,
bench_quality.json). Code: `ext_bench.py`, `bench_analyze.py`.

**2 bug hạ tầng vá trong lượt này**: (1) dict-wrap transformers 5.15
(`recurrent_states`/`keys` bọc `{0: tensor}` → phải qua `e5._get()`);
(2) OOM do nạp CẢ 4B (3,5GB) LẪN 27B (18GB) cùng lúc trên card 22GB →
đổi `run_cross` sang KIẾN TRÚC HAI PHA (4B spill cache ra đĩa → xả →
27B đọc lại), đúng như docstring `e5_train` đã cảnh báo từ đầu và
`cascade_427.py` đã làm.

## BENCHMARK NGOÀI — baseline 27B THUẦN (2026-08-27, user chốt bộ đề)

User: "test 4-27 trên 1 bộ dài vài nghìn samples với đủ thể loại câu hỏi...
thử thuần 27B, sau đó apply training và re-test, để thấy tổng quát khả năng
của mapper", sau đó chốt tiếp: bỏ CUDA (compute-eval), chỉ math+reasoning,
200 mẫu/bộ, và "ko chỉ xem kết quả, phải xem cách model suy luận, có tạo dữ
liệu rác ko, có hallu ko".

Bộ đề chốt (1138 mẫu chạy thật; code `ext_bench.py`, kết quả HF `extbench_self/`):

    bench     n     đúng    rác   lặp>0,6  cắt-hại  ko-đáp-án  hallu  sai-tính  ký-tự-TB
    bbh     182    53,8%   0,0%     0,0%     0,0%       0,0%   0,00      0,00        88
    gsm8k   200    80,0%   2,5%     0,0%     4,5%      21,5%   0,00      0,12       652
    musr    198    58,1%   0,0%     0,0%     8,6%       8,6%   0,04      0,00        85
    (musr toàn bộ 756 mẫu: 405 = 53,6% — bản 198 là tập con của bộ đề mới)

**Chất lượng sinh của 27B thuần (mốc đối chứng cho mapper)**:
- **KHÔNG degenerate**: lặp trigram >0,6 = 0,0% ở CẢ 3 bộ. Đây là mốc quan
  trọng nhất — chiến dịch Phase C đã cho thấy chữ ký hỏng của cache ngoại
  là "ra đúng vài ký tự đầu rồi lặp/trôi". Baseline sạch 0% nghĩa là khi
  chạy cross, MỌI tỷ lệ rác > 0 đều quy được cho mapper.
- **Không bịa tên** (hallu ~0,00-0,04/mẫu; soi 8 ca musr còn lại vẫn là
  báo động giả: 'Explicitly', 'Poor' — từ viết hoa đầu dòng).
- Sai chủ yếu là **"sai nhưng sạch"** (204 ca): suy luận đúng dạng, đúng
  trọng tâm, chọn nhầm đáp án — khác hẳn sinh bậy.
- gsm8k: 21,5% "không đáp án" là do model kết luận bằng câu chữ thay vì
  'Final Answer:'/\boxed (chấm vẫn bắt được qua số cuối); 0,12 phép tính
  sai/mẫu — có trượt số học lẻ tẻ nhưng phần lớn vẫn ra đúng kết quả.

**AIME và MATH-500 bị loại (có lý do đo được, không phải bỏ tùy tiện)**:
AIME 431s/bài dùng TRỌN 2560 token mà vẫn cắt giữa phần suy nghĩ → 0/2,
sẽ về ~0; một bộ đạt 0% KHÔNG đo được suy giảm của mapper (không có gì
để suy giảm). MATH-500 dừng theo lệnh user ("đã đủ để đánh giá") — tiết
kiệm ~6h GPU cho self và ~7h cho cross.

**Hạ tầng đo: 3 lần suýt báo cáo số liệu SAI trong cùng phiên** (ghi lại
vì đây là rủi ro hệ thống, không phải xui rủi):
1. MuSR 198/198 hit=0 — Qwen3.5 là model *thinking*, `max_new=8` khiến nó
   chưa thoát khỏi `<think>`. Vá: đóng sẵn `<think>\n\n</think>` trong prompt.
2. AIME 0/5 — ngân sách 1024 token không đủ cho suy luận; và chấm điểm
   `_strip_think` XOÁ mất `\boxed{}` nằm trong khối think.
3. `bench_analyze` báo "rác 7,1%, cắt 88,4%, hallu 0,15/mẫu" — cả 3 đều là
   BÁO ĐỘNG GIẢ (câu trả lời ngắn đúng ' B' bị cờ empty<3; musr chỉ 24
   token nên cắt sau khi đã trả lời = vô hại; tiêu đề markdown
   '**Initial State:**' bị tính là tên bịa).
→ Từ đó mỗi lớp đo có test chạy KHÔNG cần GPU: `test_ext_bench_scoring.py`
(14/14) và `test_bench_analyze.py` (9/9). Quy tắc rút ra: **hit=0 hàng
loạt phải nghi harness TRƯỚC khi kết luận năng lực model.**

**Bước kế**: train mapper 4→27B @ctx8192 (`--tgt-cpu-offload` đã tích hợp)
rồi `ext_bench.py cross --mapper <ckpt>` trên ĐÚNG 1138 mẫu này.
`bench_analyze --glob-b` trả lời câu hỏi then chốt mà tỷ lệ đúng/sai không
bao giờ cho thấy: trong các ca *self đúng → cross sai*, bao nhiêu % do
**sinh rác** (cache hỏng) và bao nhiêu % do **suy luận kém** (mất thông tin).

## MAPPER 4B→9B — TRAIN THẬT XONG (2026-08-27, max-ctx=16384)

User duyệt "phóng train thật cho 4→9, theo cách làm 4-27". Chạy nguyên
`e6v3_ce.py` (0 code mới ngoài `--hf-prefix` đã tách), max-ctx=16384 —
độ dài mà 27B từng OOM cứng, 9B đi thẳng không cần ladder-hạ. Runtime
recycle 1 lần ngay trước lúc phóng (đã bootstrap lại); phát hiện thêm
bài học hạ tầng: launch thiếu `python -u` khiến log "im lặng" ~1h dù
job chạy thật (block-buffered stdout) — đã ghi vào skill colab-mcp.

    step249 (val 1) : BFCL 20/25 | needle 28/29 | score 49
    step499 (val 2) : BFCL 17/25 | needle 28/29 | score 45 (giảm nhẹ, binh thuong)
    step749 (val 3) : BFCL 21/25 | needle 29/29 | score 50 -> best-by-val
    step984 (CE_FLOOR): BFCL 23/25 | needle 29/29 | score 54 -> BEST, DUNG SOM

Dừng sớm ở bước 984/2600 do CE trung bình 50 bước < CE_FLOOR=0,2 (đúng
kỷ luật Unsloth chống overfit, giống hệt cơ chế đã dùng cho 4→27).
Tổng thời gian job (kể cả Phase A/B0/B1 + bootstrap lại runtime): ~1h53.

**So với mapper 4→27B (v3.4, đã đóng)**: BFCL 23/25 (92%) so với 18/20
(90%) — nhỉnh hơn; needle 29/29 (100%) so với 15/15 (100%) — ngang
nhau nhưng đạt trực tiếp ở max-ctx cao hơn nhiều (16384 vs 4096, không
cần ladder hạ độ dài vì VRAM dư). ifstruct 2/15, pbtable 0/10 — vẫn
gần 0, khớp phán quyết v3.5 cũ (nợ của ĐỀ, ngay cả model tự đọc cũng
điểm thấp trên 2 suite này — không phải mapper yếu).

**Kết luận**: giả thuyết E7 (CCA-GDN 4→9 ≥0,9 dễ hơn 4→27's 0,785)
được xác nhận bằng số đo thật — mapper 4→9 hội tụ nhanh hơn (dừng ở
984 bước so với ~1750-2016 bước của các bản 4→27), điểm cao hơn, ở
độ dài context lớn hơn. Checkpoint tốt nhất `mapper_v49.pt` +
`.last` + toàn bộ data/pseudo_gold đã trên HF
`gunnybd01/qwen35-kv-mapper-4b-27b/v49/`.

**Bước kế chưa làm (cần user chọn)**: (a) tích hợp mapper vào vLLM
serving (Phase C3 — thay đường copy-nguyên đã đo 55-57% bằng mapper
này, kỳ vọng vượt xa); (b) đo mapper trên bộ `suite_gen.py`
(rag/mid/math/swe) để so trực tiếp với target "80-90% như normal
decode" user đề ra; (c) đóng gói `cascade_427.py`-style cho 4→9.

## MAPPER 4B→27B — CPU offload thủ công: kết quả cuối (2026-08-27)

Sau khi đường `accelerate` (device_map dict + `llm_int8_enable_fp32_cpu_
offload`) vướng lỗi "meta tensor" (mục trên), viết `load_4bit_cpu_
offload_io()` né hoàn toàn accelerate dispatch: nạp bình thường trên
GPU rồi tự tay `.to('cpu')` + hook cho `embed_tokens`/`lm_head`. Mất
4 lần relaunch để loại 2 bug TRONG PROBE (không phải trong đường
offload): quên monkeypatch GDN autograd (bug cũ tái phát), và tái sử
dụng 1 `past_key_values` qua nhiều `.backward()` gây "backward qua
graph lần 2" — sau khi mỗi lần gọi tự dựng cache riêng (đúng thực tế
train), kết quả sạch:

    steady_after_offload : 12,81 GiB (so baseline 17,66GiB — tiết
                            kiệm thật 4,85GiB, khớp đường accelerate
                            cũ 12,94GiB — xác nhận 2 lần)
    T=4096  : OK  peak 15,81GiB  t=2,38s
    T=8192  : OK  peak 18,11GiB  t=1,40s   <- baseline OOM CỨNG ở đây
    T=16384 : FAIL OOM (21,94GiB gần đầy — 4,85GiB tiết kiệm không đủ)

    Kiểm tra sống còn — 2 lần gọi ĐỘC LẬP liên tiếp @8192 (mô phỏng
    2 bước train kế tiếp, mỗi lần tự dựng cache từ đầu):
    E-repeat1: OK peak 18,11GiB t=1,39s
    E-repeat2: OK peak 18,11GiB t=1,42s

**KẾT LUẬN DỨT KHOÁT: CPU-offload thủ công THẮNG THẬT ở T=8192.**
Đây là độ dài baseline GPU-full OOM cứng (đo 2 lần, nhất quán). Tốc
độ 1,4s/lần gọi — với vài trăm-nghìn bước train, tổng thời gian vẫn
trong tầm vài giờ (ước ~1000 bước × 1,4s ≈ 23 phút phần forward+
backward thuần, chưa tính overhead khác — hoàn toàn thực tế). Gọi
lặp lại nhiều lần liên tiếp KHÔNG có vấn đề gì (2/2 thành công, thời
gian ổn định) — nghi ngờ trước đó về cache dequant bitsandbytes giữa
các lần gọi là SAI, lỗi nằm ở cách viết probe (chia sẻ 1 cache tái sử
dụng), không phải giới hạn thật của kỹ thuật offload.

T=16384 vẫn đóng — 4,85GiB tiết kiệm không đủ bù phần tăng bộ nhớ
attention (repeat_kv) ở độ dài đó; cần đòn bẩy khác (chưa thử) nếu
muốn đi xa hơn 8192.

**Khuyến nghị cuối cho nhánh 4→27B**: max-ctx hiện có 2 lựa chọn đã
đo — 4096 (an toàn tuyệt đối, không cần offload) hoặc **8192 (gần
gấp đôi, cần tích hợp `load_4bit_cpu_offload_io` vào `e5_train.py`'s
model loading path chính thức — hiện chỉ có trong probe, CHƯA nối
vào `e6v3_ce.py`)**. Việc tích hợp là thay 1 dòng gọi hàm load, không
phức tạp. Quyết định train 27B ở 8192 hay giữ 4096 — và có tích hợp
offload vào pipeline train chính hay không — cần user duyệt (GPU dài
hơi, khác phạm vi điều tra này).

## MAPPER 4B→27B — thử giải pháp mở rộng context (2026-08-27, user: "ko được kết luận sớm phải làm kỹ")

User chất vấn kết luận cũ (27B kẹt ở 4096 trên L4) — đúng quy tắc 4, đã
tranh luận và ĐO THẬT 2 đòn bẩy trên chính forward+backward thật (cache
có gradient tiêm vào, đúng cơ chế mapper), không suy luận. Đối chứng
baseline trước (tái lập đúng ladder cũ): 4096 OK 20,19GiB / 8192 OOM.

    Baseline           : 4096 OK 20,19GiB (54s) | 8192 OOM
    + gradient ckpt     : 4096 OK 20,18GiB (6s)  | 8192 OOM (giống hệt baseline)
    + CPU offload I/O   : load peak 12,94GiB (−4,72GiB so 17,66GiB)
                          | 4096 VÀ 8192 đều FAIL (lỗi khác — không phải OOM)
    + cả hai            : load 12,94GiB | 8192/16384 FAIL (cùng lỗi offload)

**Đòn bẩy 1 (gradient checkpointing trên backbone) — THUA THẬT, có cơ
chế rõ ràng, không phải bug**: peak VRAM giống hệt baseline (20,18 vs
20,19 GiB). Lý do: `gradient_checkpointing_enable()` của HF chỉ xả bớt
activation TÍCH LŨY QUA NHIỀU LỚP (giữa các decoder layer), trong khi
OOM ở đây xảy ra NGAY TRONG MỘT LỚP — do `repeat_kv` (mở rộng GQA)
tạo tensor tạm khổng lồ khi attention phải quét toàn bộ 8192 token
cache. Hai cơ chế lệch nhau hoàn toàn → đòn bẩy này đóng, có bằng
chứng cơ học, không phải "chưa thử kỹ".
- Tốc độ 6s vs 54s ở cùng T=4096 là do cache CUDA/cuDNN đã ấm từ lượt
  chạy trước trong cùng process, không phải hiệu ứng thật của gradient
  checkpointing (ghi chú tránh hiểu lầm khi đọc lại số này).

**Đòn bẩy 2 (CPU offload embed_tokens+lm_head, `llm_int8_enable_
fp32_cpu_offload=True`) — THẮNG MỘT NỬA, CHƯA ĐÓNG**: tiết kiệm bộ
nhớ tĩnh THẬT và tái lập được 2 lần (12,94GiB, giảm 4,72GiB so với
17,66GiB) — xác nhận đúng nghi ngờ ban đầu (embed/lm_head không bị
lượng tử hóa là thủ phạm chính). NHƯNG vướng lỗi tích hợp thật:
`NotImplementedError: Cannot copy out of meta tensor; no data!` — xảy
ra ở CẢ 4096 (đáng lẽ dư sức chứa) lẫn 8192, kèm cảnh báo "Some
parameters are on the meta device because they were offloaded to the
cpu". Chẩn đoán: cơ chế offload của `accelerate`/`bitsandbytes` dùng
hook nạp trọng số thật "vừa lúc" (lazy) trong đúng luồng forward đã
đăng ký — việc mapper tự ý sửa tensor trong `past_key_values` NGOÀI
luồng đó (clone + requires_grad_) phá vỡ hợp đồng lazy-load này. Đây
KHÔNG phải bằng chứng offload vô dụng — là bằng chứng đường ống
`device_map` thủ công + hook tự động không tương thích với cách mapper
thao túng cache. Đường thoát chưa thử: tự tay `.to('cpu')`/`.to('cuda')`
quanh 2 module đó SAU khi load bình thường (không qua accelerate
dispatch hook), viết forward thủ công — tốn công hơn 1 cờ nhưng khả
thi về nguyên lý (bộ nhớ đã xác nhận đủ rẻ).

**Kết luận (không phải "hết đường", là "cần đầu tư sâu hơn nếu muốn
theo tiếp")**: gradient checkpointing đóng hẳn. CPU offload còn cửa —
tiềm năng thật (4,72GiB) nhưng cần viết lại đường nạp trọng số thủ
công thay vì dựa vào cờ có sẵn, ước thêm 1-2h kỹ thuật + đo lại. Chưa
làm vì cần user xác nhận có đáng đầu tư tiếp hay ưu tiên nhánh 4→9
(đã duyệt, đang chờ phóng) trước.

## MAPPER 4B→9B — sanity XONG (2026-08-27, user chốt "tune thay copy nguyên")

User: copy-nguyên 4→9 "hên xui", đòi mapper functional-loss như 4→27
(v3.4). Tin tốt xác nhận bằng đo: `e6v3_ce.py` tổng quát theo
`--tgt-model` — đổi tham số 9B thay 27B, KHÔNG SỬA CODE, chạy trọn
20 bước sanity không lỗi ngay lần đầu (0 giả định ngầm 27B-only lộ ra).

    20 bước, 4.72 s/bước (gồm cả overhead khởi động lần đầu ~61s)
    peak VRAM 8.76 GiB (so với 27B bnb-4bit sát trần L4 22GB)
    CE bước 0: 7.089 (điểm khởi đầu bình thường, chưa đủ bước để thấy xu hướng)

- **VRAM rẻ hơn 27B rất nhiều** (8,76 GiB vs sát trần 22GB) — còn nhiều
  dư địa tăng `--max-ctx`, batch, hoặc bỏ `mapper.ckpt` (checkpoint
  autograd) để đổi tốc độ lấy VRAM, khác hẳn ràng buộc chật của 27B.
  Sau khi trừ ~61s khởi động lần đầu, tốc độ ổn định ước ~1,7-1,8
  s/bước — cùng dải với 27B (1,4-2,2 s/bước ở v3.3/v3.4) dù it hơn.
- Sửa 1 bug hạ tầng trước khi chạy: `hf_up()` từng **ghim cứng thư
  mục `"v34/"`** cho mọi target — nếu không tách, chiến dịch 4→9 sẽ
  đè lên kết quả mapper 4→27B trên HF. Đã thêm `--hf-prefix` (mặc
  định suy từ tên `--out`).
- Chưa đủ bước để đánh giá hội tụ (sanity chỉ đo tốc độ/VRAM, không
  đo chất lượng — đúng mục đích ban đầu). Theo E7 (CCA-GDN 4→9 ≥0,9,
  dễ hơn 4→27's 0,785), kỳ vọng hội tụ nhanh hơn — cần chạy train
  thật (vài trăm-nghìn bước, có val curve) để xác nhận.
- **Ladder XONG (2026-08-27): CẢ 3 MỐC 4096/8192/16384 CHẠY SẠCH,
  KHÔNG OOM** — khác hẳn 27B từng kẹt ở 4096 (8K/16K OOM phần cứng).

      4096  : 0,74 s/bước (template-path) | peak  9,11 GiB
      8192  : 1,07 s/bước (template-path) | peak 10,04 GiB
      16384 : 1,88 s/bước (template-path) | peak 11,92 GiB

  Số đo "template-path" đại diện ĐÚNG cho mọi bước train ở các độ dài
  này: điều kiện `cut.shape[1] > 2500` (đã có sẵn từ bản vá 27B, dòng
  993) buộc MỌI item train >2500 token luôn đi đường template dù đang
  ở pha λ>0 — tức tổ hợp nặng nhất (prefill thật + aux + student, nơi
  27B từng OOM 2 lần) không bao giờ xảy ra ở các độ dài này. Số đo
  không phải ngoại suy — đã đo trực tiếp.
- **Khuyến nghị max-ctx cho train thật: 16384** — còn dư ~11 GiB dưới
  trần L4 23GB, khớp đúng mục tiêu gốc user hỏi ("Unsloth hỗ trợ tới
  16K"). Đang CHỜ USER DUYỆT trước khi phóng (GPU dài hơi, khác sanity).
- **Bước kế (chờ user duyệt riêng — GPU dài hơi)**: phóng train thật
  4→9, tái dùng nguyên data BFCL/needle/ifstruct/pbtable cũ trước
  (đã kiểm chứng cho 27B), val mỗi 250 bước, auto-upload HF
  `v49/` (không đụng `v34/`). Nếu val lên nhanh → tích hợp thêm data
  4 họ đề mới (`suite_gen.py`: rag/mid-info/reasoning-math/swe) làm
  Giai đoạn B, rồi đo trên `c2suite.sh` (đã dựng, chưa phóng).

## C2c sem (2026-08-26, user duyệt bước 1 "scope ngữ nghĩa") — N=60 @8K

Câu hỏi: cross 4B→9B qua serving có sống ở bài **hiểu** (QA/RAG paraphrase,
không đòi khớp nguyên văn) hay chỉ hỏng ở bài trích chuỗi số? Harness
`gen-sem` (c2b_gates.py): filler wikitext THẬT, 1 câu fact tự nhiên giấu
giữa bài, câu hỏi paraphrase cuối bài, chấm bằng keyword (không phải
substring số) — tránh trùng bài "trích nguyên văn" của C2b-N.

    Self  (champion tự đọc):  54/60 = 90,0%   p50 4,42s
    Cross (vở 4B qua kho):    33/60 = 55,0%   p50 1,63s (×2,7 @8K)

- **Kết luận: scope ngữ nghĩa KHÔNG miễn nhiễm** — 55% gần với needle số
  (57,1% @N=240) hơn là với E2 transformers (9/9). Tức bài toán không
  phải "trích xuất chính xác riêng khó" mà đúng là **định luật biên
  mỏng áp dụng cho MỌI decode đầu tiên trên cache ngoại**, bất kể dạng
  câu hỏi — cache 4B mang đủ thông tin (soi lỗi: nhiều ca cross đúng
  nghĩa nhưng lệch từ, vd "barn" thay "stables") nhưng bước decode đầu
  bị nhiễu kernel/roundtrip lật ngẫu nhiên ~45% ca.
- Vá bug hạ tầng phát hiện giữa chiến dịch (bài học, xem thêm mục
  RUNTIME HYGIENE bên dưới): runtime mới kéo vllm 0.28.0 không ghim
  (drift ngầm) → ghim `vllm==0.27.1`; `run.sh serve` chết câm khi thiếu
  `/tmp/vllm_env.sh` do `set -e` + `source` fail → vá `|| true`; upload
  HF 401 hàng loạt do **gõ nhầm token khi nhúng cell** (37→36 ký tự,
  không phải lỗi code) → sửa bằng cách đọc token từ file + assert độ
  dài, không bao giờ gõ tay secret nữa; mọi `HfApi()` trong repo đã vá
  sang `HfApi(token=...)` tường minh.
- Kết quả + prompt gốc trên HF `c2c_sem/`. Quyết định: KHÔNG bán cross
  4→9 ở dạng "dùng thẳng" cho bất kỳ bài nào (số hay chữ); đường sản
  phẩm khả thi = polisher hoặc hybrid-fallback, HOẶC (hướng user chốt
  2026-08-26) train mapper functional-loss riêng cho 4→9 thay copy
  nguyên — xem mục MAPPER 4→9 (đang mở).

## SPEC DECODING NGRAM (2026-08-14): OFF MẶC ĐỊNH TRÊN L4 — đo 2 model × 2 mức tải

Profile `-spec` (ngram k=4, prompt-lookup 2-4) thêm vào run.sh; cùng runtime,
baseline đo lại tươi:

    9B                    baseline    ngram-spec
    skills conc1 decode   34,34       37,68 (+9,7%)
    skills conc8 thr      158,6       145,3 (−8,4%)
    agent-loop 8 phiên    278,6/hr    **179,4 (−36%)**, TTFT p95 50s
    KV capacity           404.613     350.907 (−13%)

    27B                   baseline    ngram-spec
    skills conc1 decode   15,83       19,49 (+23%)
    KV tokens             12.288      8.874 (−28%)
    prefix hit            95,3%       **54% (sập)**; TTFT p50 0,23→2,1s; conc2 thr −43%

Cơ chế: ở tải cao GPU đã bão hòa compute (P7) — verify draft đốt thêm compute,
draft trượt là công cốc; và spec chiếm VRAM đúng vào KV vốn là tài nguyên hiếm
nhất. Spec chỉ thắng ở single-stream sinh dài + VRAM dư (GPU to) — không phải L4.
Ghi vào hardware/l4.py. Đóng hướng bằng số đo, không phải suy luận.

## UTIL SWEEP (2026-08-14, lệnh user "nghiên cứu 98/100"): MẶC ĐỊNH MỚI 0.97 — ĐỈNH DỊCH 8→12 PHIÊN

    GPU_UTIL   KV tokens (mml 65536)   agent-loop 12 phiên
    0.85       404.613 (cũ)            —
    0.90       469.199 (+16%)          —
    0.95       534.735 (+32%)          —
    **0.97**   **560.380 (+38,5%)**    308,4 cold / **358,1 warm ⬅ đỉnh mới**
    0.98       573.677 (+2,4% nữa)     308,4 (trùng 0.97 đến số lẻ)
    1.00       CHẾT khi khởi động engine (đo thật, không phải lý thuyết)

- **Điểm vận hành mới: 12 phiên @0.97** — warm 358,1 tasks/hr (+27% vs 8 phiên
  cùng điều kiện; đỉnh cũ toàn chiến dịch 329). 16 phiên giờ 330,7 (×1,74 con
  số cũ 189,8) — không còn là vùng lỗ.
- Bài học đo: 358 vs 308 là hiệu ứng SERVER ẤM (prefix cache đầy sẵn, hit 90,3%
  vs 87,7%) — báo cả hai số, không chọn số đẹp.
- 0.98 không cho gì thêm (perf y hệt, margin mỏng hơn) → chốt 0.97; run.sh +
  adapter.py đã đổi mặc định.

## REFACTOR (2026-08-13): kiến trúc sản phẩm theo khuôn transformers — TEST THẬT PASS

User chốt qua 5 vòng thảo luận (lấy `out/transformers` làm tham chiếu):
architecture-centric, engine mang patches riêng, registry đệ quy, utils 2 tầng.

- Cấu trúc: `models/qwen3_5/{engine/vllm/(adapter.py+patches/×19), load/(gguf_to_marlin
  = graft cũ, pure_gguf, pytorch_tensor), hardware/l4.py, utils/}` + `models/qwen3_5_moe`
  (dummy thành thật) + `models/_template` (7 luật) + `sdk/ loading/ logging/ utils/
  bench/ bench/workload/ tests/`. 54 file git mv thuần (lịch sử nguyên vẹn).
- **Registry đệ quy** (`register.py` gốc + mỗi folder 1 cái): folder không có
  register.py = vô hình (patches/ ẩn khỏi bề mặt sản phẩm có chủ đích); đọc
  REGISTER/ADAPTER literal bằng ast, không import. `python register.py --flat`.
- **`run.sh` = 1 lệnh kiểu vLLM**: setup / serve 9b|27b (config đã tune nhúng sẵn) /
  status / logs / bench / eval / registry / stop — tất cả idempotent.
- **Notebook A = ĐÚNG 1 CELL**: clone + `bash run.sh serve 9b && status`. Đo thật:
  fresh-runtime → READY 370s (KV 404.613 tokens khớp lịch sử tuyệt đối), lần 2 (ấm)
  140s, smoke completion trả lời đúng. Cần gì thêm LỆNH vào cell, không thêm cell.
- Kiểm định: 9 file test local xanh, gồm test_structure.py mới (6 luật: cấm import
  chéo model, utils không import models, ADAPTER bắt buộc, registry sạch, patch đặt
  đúng chỗ, register.py đủ mặt).
- Sửa path kéo theo: setup_env patch loop, colab_bootstrap, serve_test, 8 test,
  baked path trong patch_gguf_auto_marlin, REPO_ROOT của workload scripts.

## CHIẾN DỊCH 27B — PHASE 1 (2026-08-13): Qwen3.5-27B LÊN SÓNG TRÊN L4, 15,8 tok/s

Khảo sát frame (RedHatAI KHÔNG có 27B w4a16):
- **`apolo13x/Qwen3.5-27B-quantized.w4a16`** (18,6GB, compressed-tensors W4A16 g128):
  ĐÃ QUANTIZE SẴN CẢ GDN (in_proj_qkv/z, out_proj đều weight_packed; chỉ in_proj_a/b
  bf16 trong ignore — 207 mục). KHÁC 9B RedHat (GDN bf16 → ta phải graft). Vì thế
  graft_gguf_gdn.py báo "no in_proj_qkv weights" (nó tìm `.weight` bf16) — với frame
  này KHÔNG CẦN graft. Cũng là frame duy nhất vừa L4.
- Loại: Qwen/Qwen3.5-27B-GPTQ-Int4 chính chủ (30,2GB), QuantTrio AWQ (21,9GB) —
  không vừa 22,5GB.
- GGUF nguồn có đủ nếu sau này cần graft chất lượng: unsloth 27B Q4_K_M (16,7GB),
  UD-Q4_K_XL (17,6GB). Tải frame+GGUF song song: 35GB / 1,7 phút (Xet).

Leo thang config để vừa VRAM (mỗi bước một lỗi thật, đều đo được):
1. util 0.85 + graphs mặc định (capture tới 512) → OOM khi profile (còn 95MB).
2. `--enforce-eager` + mnbt 512 + util 0.95 → CHẠY, KV 14.336 tok, decode **8,4 tok/s**
   (eager giết decode — overhead phóng kernel).
3. graphs nhỏ [1,2,4,8] + util 0.95 → KV còn 0,48GiB < 0,57 cần cho mml 8192.
4. util 0.97 → lỗi mới: **max_num_seqs mặc định 256 > 19 block Mamba cache** (bài học
   hybrid: mỗi seq decode cần 1 block Mamba; model to → block ít).
5. **CHỐT: mml 8192 + mnbt 512 + max-num-seqs 8 + graphs [1,2,4,8] + util 0.97
   + fp8 KV + align + prefix caching** → READY 140s, VRAM 19,9GB, KV 12.288 tok,
   decode **15,8 tok/s conc1** (+88% vs eager; sát trần băng thông ~16 = 300GB/s ÷ 18,6GB).

Sanity 27B PASS: toán/tiếng Việt/code đều mạch lạc.

**PHASE 2 — CỔNG CHẤT LƯỢNG PASS (cùng ngày): ppl 27B = 4,1484** (99 đề SWE-bench,
đúng bộ đề của 9B; chạy eval qua HTTP trên server đang sống, mnbt 512 chặn peak
logprob). So sánh: 9B champion 4,7637 / 9B bf16 5,13 → 27B W4A16 tốt hơn 9B champion
12,3% — đúng kỳ vọng model to hơn, frame cộng đồng KHÔNG hỏng. Không cần graft.
Bench mini (prefix 4K, cùng ngày): conc1 TTFT p50 **0,24s** (prefix cache hit 95,3%),
decode 15,8; conc2 thr 29,5 (scale gần tuyến tính); conc4 thr 36,6 nhưng TTFT p95
nổ lên 21,2s (KV 12K cạn → hàng đợi). **Điểm vận hành 27B trên L4: 1-2 người dùng,
chất lượng cao hơn 9B 12,3% ppl** — đúng vai "single-user quality tier" bên cạnh
9B "multi-session throughput tier". 27B ngày 1 ĐÓNG TRỌN VẸN.

## TASK R2b (2026-08-13): chunk=32 TRUNG TÍNH trên 0.27.1 — khai tử con số +7,6%

Cùng runtime, cùng seed, bench_skills synthetic 30K prefix, conc 1 và 32:

    chunk    conc1 decode/thr    conc32 decode/thr
    64       34,34 / 32,7        16,10 / 262,0
    32       34,33 / 32,7        16,09 / 261,8

GIỐNG HỆT (lệch <0,1%). Cộng với R2 (chunk16 identical): FLA_CHUNK_SIZE không còn
ảnh hưởng chế độ server-decode trên 0.27.1. Con số +7,6% của chunk=32 là di sản
0.26 + đo offline, KHÔNG ghi thành khuyến nghị. Serve config chuẩn giữ mặc định.

## TASK P7 (2026-08-13): CPU KHÔNG nghẽn — hoàn toàn GPU-bound

Đo trong lúc bench conc32 (tải nặng nhất): GPU util phẳng **100%** (17 mẫu, min=max=100),
tiến trình vLLM (API server + engine) ăn trung bình **8% một core**, đỉnh 65%, máy 12
core. Kết luận: không có đòn tối ưu nào nằm ở CPU trên L4 Colab; mọi cải thiện phải
đến từ GPU (kernel/quant/batching). Đóng P7.

## KIỂM ĐỊNH ĐÚNG ĐẮN CHẾ ĐỘ ĐỒNG THỜI (2026-08-13) — PASS với chú thích

8 prompt greedy (temp=0, seed=0): chạy riêng lẻ làm chuẩn → chạy đồng thời cả 8:
- Khớp tuyệt đối 3/8; 5/8 phân kỳ ở ký tự 135-764 (sâu trong generation, KHÔNG có
  cái nào hỏng từ đầu; mọi output mạch lạc, đúng chủ đề).
- Quyết định: chạy đồng thời 2 lần → conc-vs-conc cũng chỉ 3/8 khớp ⇒ phân kỳ do
  **dynamic batching đổi thành phần batch → đổi thứ tự reduction số học** — tính chất
  cố hữu của vLLM (không batch-invariant), KHÔNG phải bệnh của graft/Marlin/fp8 KV.
- Cổng đúng đắn đồng thời: **PASS** (không garbage, không hỏng cấu trúc). Audit top-5
  mục "concurrent correctness gate" đóng. Ai cần bitwise-repro phải tắt dynamic
  batching (không đáng ở production).

## TASK R2 (2026-08-12): chunk=16 — TRUNG TÍNH, đóng nhánh. 32 là điểm ngọt cục bộ

`FLA_CHUNK_SIZE` trong 0.27.1 vẫn ở `vllm/third_party/flash_linear_attention/
ops/utils.py:31` (chỉ đổi gốc import). Vá 64→16, đo trên champion cùng
runtime với bảng R1b, revert ngay sau đo:

    chunk        conc1 decode/thr    conc32 decode/thr
    64 (đối chứng)  34,10 / 33,6      14,39 / 365,5
    16              34,10 / 33,6      13,95 / 369,9

- conc1 GIỐNG HỆT (sai lệch <0,01%) — không tiếp nối đà +7,6% mà chunk=32
  từng cho ở TASK P2. conc32 trung tính (trong nhiễu).
- ⇒ **Suy luận "xu hướng chỉ về phía nhỏ hơn" của coordinator SAI.** 32
  là điểm ngọt CỤC BỘ (cân bằng số lần phóng kernel vs cường độ số học),
  không phải điểm trên một đường dốc đơn điệu. Đóng nhánh chunk nhỏ.
- CÒN NỢ: con số +7,6% của chunk=32 đo trên **0.26 + dữ liệu thật**; cần
  một lượt tái kiểm trên 0.27.1 + bench tổng hợp trước khi ghi thành
  khuyến nghị chính thức (rẻ, ~6 phút).

## TASK R1 (2026-08-12): fp8_per_tensor PHÁ TRẦN PREFILL +34-38% — "tường vật lý" SỤP

Đo prefill thuần (prompt ngẫu nhiên duy nhất, tắt prefix caching, max_tokens
nhỏ) trên vLLM 0.27.1, cùng runtime:

    độ dài   champion (graft int4)   bf16 gốc + fp8_per_tensor
    4K       2.921 tok/s             **3.973 (+36,0%)**
    16K      2.934 tok/s             **4.061 (+38,4%)**
    30K      2.789 tok/s             **3.754 (+34,6%)**

- Cấu hình fp8: `--quantization fp8_per_tensor` trên `Qwen/Qwen3.5-9B` bf16
  gốc, chỉ cần `ignore: ["in_proj_ba"]`. Online, KHÔNG cần calibrate lại.
- **Xác nhận đúng dự đoán của tranh luận vòng 1**: fp8 thua ở decode (nghẽn
  băng thông) nhưng THẮNG ở prefill (nghẽn compute) — hai chế độ khác nhau,
  không được suy diễn từ cái này sang cái kia. Sai lầm cũ của coordinator.
- **CHẤN ĐỘNG PHỤ — con số nền của "tường 1.433 tok/s" SAI**: champion đo
  ngay bây giờ cho **2.789-2.934 tok/s**, tức GẤP ĐÔI mốc 1.433 đã dùng
  suốt chiến dịch để lập luận "trần vật lý". Khớp với dị thường từng ghi ở
  TASK F (2.228 tok/s ở prefix 60K, khi đó bị coi là bất thường). ⇒ Mốc
  1.433 nhiều khả năng đo dưới cấu hình bị giới hạn (chunk/mnbt?) hoặc sai
  phương pháp. **Mọi con số dung lượng dựa trên 1.433 phải tính lại.**
### R1b — bảng đánh đổi đầy đủ (cùng runtime, cùng config production):

    metric                    champion (int4)   fp8_per_tensor   thắng
    prefill 4K/16K/30K        2921/2934/2789    3973/4061/3754   fp8 +35%
    decode conc1 (tok/s)      34,1              23,2             champion +47%
    decode conc8 (thr)        201,2             145,6            champion +38%
    decode conc16 (thr)       276,8             230,1            champion +20%
    decode conc32 (thr)       365,5             320,7            champion +14%
    TTFT p50/p95 @conc32      1,49s / 10,2s     1,18s / 7,89s    fp8 nhỉnh
    ppl (99 đề, đo tươi)      4,7637            5,4383 (1,142)   champion RÕ
    KV @32768                 367K tok, 11,21×  254K tok, 7,76×  champion +44%
    VRAM phục vụ              18.650 MiB        18.634 MiB       ngang

**PHÁN QUYẾT: fp8 KHÔNG phong champion — là công cụ CHUYÊN DỤNG.**
- fp8 chỉ thắng ở prefill thuần. Champion thắng decode, sức chứa phiên
  (+44%), và chất lượng (fp8 rơi vùng WARN 1,142 — sát FAIL).
- ⇒ **Hai cấu hình cho hai loại việc**: fp8 cho workload prefill-nặng
  (đọc tài liệu lạ một lượt, tóm tắt hàng loạt, ít phiên); champion cho
  chat/agent nhiều lượt nhiều phiên — tức workload chính của dự án.
- **Bài học đo lường (lại một lần nữa)**: ghi chép cũ nói fp8 "sạch 3/3"
  — nhưng đó là smoke test 2 câu trên model 2B. Ở 9B với ppl 99 đề, cái
  giá chất lượng lộ rõ. **Smoke test không bao giờ thay được cổng ppl.**
- `int8_per_channel_weight_only`: CRASH OOM lúc load (21,68/22,03 GiB
  trước khi cấp KV — đường quant online này giữ cả bản bf16 tạm thời).
  Bỏ qua: weight-only vẫn dequant về fp16 để compute nên không ăn được
  INT8 TOPS thật.
- CÂU HỎI MỞ đáng theo đuổi sau: có scheme **W4A8** nào (trọng số 4-bit +
  activation 8-bit) để ăn CẢ HAI đầu không? Chưa khảo sát.
- Danh mục quantization online của 0.27.1 có `int8_per_channel_weight_only`
  (áp được ngay, không cần calibrate) nhưng là weight-only → nhiều khả
  năng KHÔNG kích hoạt đường INT8 TOPS gấp đôi của Ada. W8A8 thật phải
  qua compressed-tensors + calibrate offline (~15-25 phút với recipe nhanh).

## DRIFT MÔI TRƯỜNG (2026-08-12): vLLM đã lên 0.27.1 — MỌI SỐ CŨ ĐO Ở 0.26

Tái thiết sau wipe #8 phát hiện `pip install vllm` giờ kéo **0.27.1**:
- **CẢNH BÁO PHƯƠNG PHÁP**: toàn bộ số đo của chiến dịch (champion 388,7
  tok/s, ppl 4,778, SLA 0,3 QPS, P1 522 tok/s...) đo trên **0.26**. So
  trực tiếp số mới với số cũ là KHÔNG HỢP LỆ — phải đo lại baseline trên
  0.27.1 trước mọi kết luận. (Đúng bài học "baseline phải đo tươi".)
- `torchaudio`/`torchvision` đi kèm lệch CUDA build (torch cu13.0 vs
  torchaudio cu12.8) → transformers hard-import torchaudio → crash lúc
  import. **Fix: `pip uninstall -y torchaudio torchvision`** (không cần
  cho serving text-only). Nên thêm vào setup_env.sh.
- **Hai bug của ta ĐÃ ĐƯỢC UPSTREAM SỬA** (patch thành no-op):
  (a) `patch_vllm_qwen35_hybrid` — 0.27.1 đã thêm `IsHybrid` vào base
  class Qwen3_5ForCausalLMBase; (b) `patch_gguf_override_signature` —
  plugin đã tự có fix. → Cập nhật hồ sơ upstream/: đánh dấu "đã sửa
  upstream, KHÔNG gửi".
- `patch_fla_ada_shmem` skip: đường dẫn
  `vllm/third_party/flash_linear_attention/ops/utils.py` không còn ở vị
  trí cũ → **phải dò lại vị trí `FLA_CHUNK_SIZE` trong 0.27.1** trước khi
  làm thí nghiệm chunk.
- 16/19 patch còn lại áp sạch; `_C_gguf` import OK (kernel CUDA thật).

## TRANH LUẬN VÒNG 1 (2026-08-12): phán quyết trên 3 "trần" đã đóng vội

Hai agent độc lập (CÔNG: docs/debate-attack.md · THỦ: docs/debate-defense.md).

### Trần 1 — prefill 1.433 tok/s "là vật lý" → **MỞ LẠI. Kết luận cũ SAI.**
- Phép tính cũ dùng 58% GEMM — con số này chỉ có trong tài liệu giảng dạy;
  **số đo thật là 47,9%**, và bản thân nó đo ở chế độ eager chưa fusion nên
  còn thổi phồng phần non-GEMM. Nền của kết luận "vật lý" là cát.
- **int8 trên Ada L4 THẬT SỰ gấp đôi fp16** (datasheet: 242 TOPS INT8 vs
  121 TFLOPS FP16 dense). Khác bản chất với thất bại fp8 trước đây: fp8
  thua ở DECODE (nghẽn băng thông, compute không giúp gì), còn **prefill
  nghẽn compute — đúng chế độ int8 có cửa**. MFU thực và chi phí quantize
  activation động thì CHƯA AI ĐO.
- `fp8_per_tensor` đã chứng minh chạy sạch trên stack này nhưng **chưa từng
  đo cho prefill/long-context** — một lượt đo còn thiếu, gần như miễn phí.
- Trần điện 72W: xác nhận CỨNG (TDP thiết kế + Colab không cho quyền root).
- ⇒ **Ưu tiên số 1 khi GPU trở lại: đo W8A8/int8 và fp8_per_tensor cho
  prefill.** Đây là viên đạn thật, không phải hy vọng suông.

### Trần 2 — chunk GDN ≤64 → **KHẢ THI nhưng VÔ NGHĨA. Đổi hướng.**
- Cả hai bên đồng ý: assert không phải ràng buộc toán học, cũng không phải
  giới hạn shared memory (chunk 128 chỉ cần ~36-40KB < 100KB/SM); chỉ là
  kernel `merge_*_to_128x128` chưa ai viết (~1 buổi việc cơ học).
- NHƯNG bằng chứng thực nghiệm đi NGƯỢC: TASK P2 cho chunk **32 thắng 64**.
  Xu hướng chỉ về phía NHỎ HƠN, không phải lớn hơn.
- ⇒ **Bỏ ý định viết kernel 128. Thay bằng thí nghiệm MIỄN PHÍ: thử
  chunk=16** (đã được hỗ trợ sẵn) — đúng chiều bằng chứng, tốn 10 phút.

### Trần 3 — cascade attention → **ĐÓNG, nhưng bằng lý do TỐT HƠN.**
- Bên thủ truy ra PR gốc (vllm #26130): lỗi là **TREO dưới tải nhiều
  request + prefix cache hit**, tác giả tự nhận chưa truy được nguyên nhân
  — tức đúng điều kiện production của ta.
- **Lỗ hổng quy trình phát hiện được**: cổng byte-identical của dự án
  (TASK F/N6) chỉ kiểm MỘT request tuần tự → **không thể bắt loại lỗi
  treo-dưới-tải-đồng-thời**. Cần bổ sung cổng kiểm ở chế độ đồng thời.
- Lợi ích thật cho kiến trúc lai: attention chỉ chiếm 0,4-0,6% thời gian
  decode (vì 75% layer là GDN) ⇒ cascade cứu được **dưới 1%** — quá nhỏ
  so với rủi ro treo.
- ⇒ Đóng vĩnh viễn, ghi rõ lý do là *lợi ích nhỏ + rủi ro treo*, KHÔNG
  phải "vì upstream hardcode" như lý do cũ.

## TASK Q1 (2026-08-11): đường cong NỐI LẠI sau khi tool chạy — cache SỐNG DAI

champion v2 + production flags, `--resume-probe`, gap 0,5/2/5/15/60s × 5 lần:

    gap tool    TTFT lượt kế tiếp
    0,5s        ~1,34s  (KHÔNG phải cache miss — xem dưới)
    2s          ~0,33s
    5s          ~0,33s
    15s         ~0,33s
    60s         ~0,34s

- **Cache KHÔNG bị thu hồi trong cửa sổ 60 giây** — TTFT phẳng lì bất kể
  nghỉ bao lâu. Đây là tin RẤT tốt cho workload agent: phiên ngủ đông chờ
  tool vẫn được nhớ, nối lại gần như miễn phí. (Chưa đo quá 60s.)
- Bất thường ở gap 0,5s là do **hàng đợi**, không phải cache: log cho thấy
  lượt trước còn đang decode ("Running: 1 reqs") khi lượt sau tới.
- **Bẫy cấu hình — GIẢ THUYẾT ĐÃ XÁC NHẬN 100%**: 400 Bad Request tái lập
  ở trial 0 và 3 mọi mức gap là do prefix ~28,7K + lịch sử tích lũy vượt
  `--max-model-len 32768` (chỉ ~4K dư địa). Chạy lại y hệt với **65536**:
  **25/25 lượt sạch, 0 lỗi**, TTFT vẫn phẳng 0,31-0,32s ở mọi gap.
  → **Luật cấu hình agent-vs-chat**: workload agent PHẢI có max-model-len
  lớn hơn nhiều vì prompt LỚN DẦN mỗi lượt (chat một lượt thì cố định).
  Công thức thô: prefix + số_lượt × cỡ_kết_quả_tool + biên an toàn.
- **Cái giá của context lớn**: max concurrency 9,66× @32K → **5,80× @64K**
  — gấp đôi context thì số phiên chứa được giảm gần một nửa. Đây là đánh
  đổi trung tâm khi quy hoạch hệ agent.
- Bench đã được vá để tự chặn trước và cảnh báo thay vì để server trả 400
  (`--max-model-len` trong bench_agent_loop.py, commit f573896).

## TASK P2 (2026-08-11): kernel GDN chunk size — lần đầu chạm 42% Amdahl, ăn nhẹ

`FLA_CHUNK_SIZE` là HẰNG SỐ module-level tại
`vllm/third_party/flash_linear_attention/ops/utils.py:31`, không có env var,
được ~7 file import lúc load → chỉ đổi được bằng cách vá nguồn TRƯỚC khi
tiến trình import vllm.

    chunk   conc1 decode/thr        conc32 decode/thr     conc32 p95
    64 (mặc định)  31,45 / 31,2     14,62 / 387,8         2,81s
    **32**  **33,85 / 33,6 (+7,6%)**  14,59 / 383,8 (~0%)  2,84s
    128     VỠ CỨNG: solve_tril.py `assert A.shape[-1] in [16,32,64]`

- chunk=32 thắng ~+7,6% ở conc1 (tải thấp/độ trễ đơn luồng), trung tính ở
  conc32 (đã bão hòa song song). Biên nhỏ nhưng THẬT — và là lần đầu tiên
  một tối ưu chạm được vào phần 42% non-GEMM của Amdahl.
- chunk=128 bị chặn bởi kernel Triton solve_tril (chỉ nhận 16/32/64) —
  trần kiến trúc, muốn vượt phải sửa kernel thật. **Ứng viên cho hàng đợi
  tấn công**: chunk lớn hơn = ít lần phóng kernel + cường độ số học cao
  hơn cho GDN; đáng điều tra xem sửa solve_tril khó tới đâu.
- Đã revert nguồn về bản gốc sau A/B.

## TASK P1 (2026-08-11): OFFLINE BATCH MODE — +34,8%, chế độ đúng cho automation

Champion v2, LLM class trong-tiến-trình (không HTTP/SSE/scheduler phục vụ),
cùng config production (prefix caching, fp8 KV, mnbt1088, 32K), 74 câu thật:

    đường            wall     completion tok   throughput
    server conc32    —        —                387,8 tok/s (N5c)
    offline 74 câu   135,4s   70.759           **522,6 (+34,8%)**
    offline 74×4     568,5s   284.280          500,0 (+29,0%) = trần thực

- Batch lớn hơn KHÔNG tăng thêm → trần compute thật ~500-520 tok/s; ở batch
  74 đã gần bão hòa, KV pool không phải nút thắt.
- **Quy đổi vận hành: ~0,55 câu/s ⇒ ~1.970 câu/giờ ⇒ ca đêm 16h ≈ 31.500
  tác vụ, ~28,8 triệu token sinh mới** (số đo thật, không phải ngoại suy).
- Khuyến nghị: workload automation theo lô (phân loại, tổng hợp, xử lý
  ticket) chạy bằng script Python gọi LLM.chat() trực tiếp — KHÔNG dựng
  server. Server chỉ dành cho traffic tương tác thời gian thực.

## TASK N5b (2026-08-11): lm_head int8 — ĐÓNG, giới hạn kiến trúc vLLM thật

scripts/graft_lm_head_int8.py (9e5caba) quantize lm_head 248320×4096
xuống int8 g32 thành công về toán (RMS 0,0055) — nhưng serve chết:
`There is no module or parameter named 'lm_head.weight_packed'`.
Nguyên nhân gốc đọc từ source: ParallelLMHead kế thừa
VocabParallelEmbedding (họ Embedding, chỉ có `weight` phẳng), KHÔNG phải
họ Linear mà compressed-tensors WNA16 nhắm — vLLM bản này không có code
path lượng tử hóa cho lm_head/embed dạng Embedding. Muốn làm phải sửa
vLLM (thêm quant method cho VocabParallelEmbedding) — ứng viên danh sách
upstream, không phải việc config. Đóng mục; byte lm_head fp16 (~2GB) là
chi phí decode không gỡ được trên stack hiện tại.

## TASK N5c (một nửa): cờ async-scheduling TỒN TẠI và ĐANG BẬT mặc định

`--async-scheduling/--no-async-scheduling`, default None = tự bật cho
backend hỗ trợ — log production xưa nay đã có "Asynchronous scheduling
is enabled", tức MỌI số đo production đều là async ON.

A/B TƯƠI (bench_skills 74 câu, champion v2, chỉ khác 1 cờ):

    conc   ON: p95/decode/thr          OFF: p95/decode/thr
     16    1,58s / 19,65 / 281,7       1,51s / 18,34 / 264,2
     32    2,81s / 14,62 / 387,8       2,73s / 13,79 / 360,5

ON thắng throughput +6,6-7,6% và decode/user +6-7%, trả giá TTFT nhỏ
(~30-40ms). **Verdict: giữ ON (đã là mặc định, không cần cờ).** Đóng
trọn chuỗi N5: a=ngram loại, b=lm_head bất khả (upstream), c=async ON.

## TASK N6 (2026-08-11): 128K context trên champion v2 — ĐẬU correctness, cascade KHÔNG khả dụng

- Rope native: max_position_embeddings=262144, theta=1e7, không scaling
  → 128K nằm trong native, không phải nới gì.
- Serve 131072 (production flags, **max-num-seqs phải hạ xuống 4** mới
  fit): KV pool 470.258 token, max concurrency 3,59×, VRAM 17,7/23,0GB.
- **CORRECTNESS GATE PASS**: prefix 120.036 token, temp=0 — output
  cache-hit BYTE-IDENTICAL với cold, ở 128K. Cold prefill 62,9s
  (~1.908 tok/s hiệu dụng); warm cùng prompt 3,06s.
- Warm TTFT suffix 4K: ~1,0s ổn định (run đầu 1,75s do compile).
  Decode per-user: conc1 28,9 / conc4 21,7 tok/s (−25% khi 4 luồng cùng
  quét KV 120K).
- **Cascade attention: KHÔNG KHẢ DỤNG trên stack này** — hai tầng khóa
  độc lập: (a) ModelConfig.disable_cascade_attn mặc định True (cờ
  --no-disable-cascade-attn để opt-in); (b) kể cả opt-in,
  FlashInferBackend.use_cascade_attention() HARDCODE return False
  (comment upstream "Cascade attention doesn't work, disable it for
  now", flashinfer.py ~1518-1524). FlashAttention backend có cascade
  thật nhưng không hợp lệ với fp8 KV (đã đóng ở A/B trước). → cột 128K
  không có phép chia-sẻ-đọc-prefix; theo dõi upstream khi họ mở lại.
- N6b (đã chạy): mns=4 KHÔNG phải trần — mns 16 và 32 đều nạp sạch ở
  131072, KV pool gần như không đổi (470K→461K, <2%), VRAM không tăng.
  Giới hạn bộ nhớ thật là KV pool theo gpu-memory-utilization, không
  phụ thuộc mns.
- Nhưng trần TRẢI NGHIỆM đến sớm hơn trần bộ nhớ: probe 16 request đồng
  thời (prefix 120K cached + suffix 3K riêng): TTFT p50 5,4s / p95 7,55s,
  decode per-user ~7,9 tok/s (vs 28,9 conc1 / 21,7 conc4). Quét KV 120K
  × nhiều luồng là chi phí chi phối.
- **Chốt kịch bản "132K tổng, prefix chung": bộ nhớ chịu 16-32; trải
  nghiệm tốt (TTFT<3s, decode >= tốc độ đọc) dừng ở ~6-10 đồng thời;
  kiểm soát ở CLIENT (bài học N1), server cứ để mns 32.**

## TASK N5a (2026-08-11): ngram speculative — LOẠI ở mọi mức tải

bench_skills 74 câu, graft int8, so {OFF, MTP (N2), ngram}:
- conc1: ngram p95 0,54s — CHẬM HƠN OFF (0,28s), không có lợi ích nào
  (khác MTP +29%). conc8: p95 2,99s vs OFF 0,76s. conc32: **SỤP — p95
  101,3s (33× OFF), throughput 239 vs 370,6, cache hit rơi 99%→53%**.
- Nguyên nhân từ log: ngram TẮT async scheduling ("Async scheduling not
  supported with ngram-based speculative decoding") + overhead
  prompt-lookup trên prefix 30K mỗi bước. Kỳ vọng "prefix chứa sẵn Đáp
  án:/JSON nên đoán trúng cao" KHÔNG cứu được chi phí matching.
- **Kết luận: loại ngram khỏi menu cho workload shared-prefix dài; MTP
  (bật tải thấp) vẫn là spec-decode duy nhất đáng dùng.**
- Vận hành: wipe TOÀN BỘ lần 4 ngay sau N5a — mất cả 2 checkpoint graft
  + skills_pack + môi trường. Rebuild ~40-60 phút (quy trình tất định).

## TASK N3 (2026-08-11): graft int4 — ứng viên soán ngôi, CHỜ XÁC NHẬN baseline tươi

graft_gguf_gdn.py --bits 4 (commit a1dea03, mặc định vẫn int8), graft
~4,6 phút CPU, RMS refit 9,4% (đúng tầm RTN int4 đã dự).

    checkpoint        ppl (99 đề)     bench_skills conc32 (74 câu thật)
    RedHatAI          5,1645 (CŨ*)    —
    graft int8 (đương kim) 5,0672      p50 0,96s / p95 3,03s / 370,6 tok/s
    **graft int4**    **4,7497 (ratio 0,92)** p50 0,61s / **p95 2,75s** / **388,7 (+4,9%)**

- int4 THẮNG int8 cả 4 chỉ số speed + ppl thấp hơn cả int8 (phản trực
  giác — giả thuyết: int4 g32 khớp nguyên scheme W4A16 của frame, không
  có biên mixed-precision int8/int4 tại Marlin; hoặc nhiễu mẫu 99 đề).
- XÁC NHẬN TƯƠI (2026-08-11, cùng runtime, cùng lệnh eval, 98/98 mỗi lượt):
  baseline 5,1322 / int8 5,0512 (0,9842) / **int4 4,7780 (0,9310)** —
  pattern int4 < int8 < baseline TÁI LẬP, không phải nhiễu baseline cũ
  (tươi vs cũ lệch <1%). **→ PHONG CHAMPION v2: graft int4**
  (`graft_gguf_gdn.py --bits 4`, RMS 9,4%, 9,1GB). Đầy đủ chỉ số:
  ppl ratio 0,931 · conc32 388,7 tok/s · TTFT p95 2,75s (dưới trần SLA)
  · TTFT p50 0,61s. Giải thích int4<int8 vẫn là giả thuyết mở
  (scheme thuần nhất với frame W4A16, không biên mixed-precision).
- Bug vận hành sửa tận gốc trong lượt rebuild: (a) `datasets` âm thầm
  downgrade huggingface_hub làm vllm_gguf_plugin chết từ import — fix
  re-pin trong setup_env.sh (bb8b894); (b) patch_gguf_override_signature
  giờ no-op khi plugin đã fix upstream thay vì giết setup (6699d2c);
  (c) doc trap "chạy fix_qwen35 lên output graft" trong suggested-run đã
  xóa — graft output phải serve NGUYÊN TRẠNG.

## TASK N2 (2026-08-11): MTP trên DỮ LIỆU THẬT — thắng tải thấp, cấm tải cao

Champion graft + production flags + `--speculative-config '{"method":
"qwen3_5_mtp", "num_speculative_tokens": 1}'` (model_mtp.safetensors trong
graft được nhận diện, share embed+lm_head). Thước chính: bench_skills 74 câu.

    conc   OFF thr/decode       MTP thr/decode        Δthr      TTFT p95 OFF→MTP
      1    31,2 / 31,45         40,2 / 41,39         +28,8%     0,28 → 0,81s
      8    191,4 / 25,32        223,0 / 30,38        +16,5%     0,76 → 4,64s (!)
     16    273,7 / 18,98        305,0 / 21,66        +11,4%     1,62 → 8,61s
     32    370,6 / 14,36        386,8 / 15,75        +4,4%      3,03 → 17,67s

- **Acceptance thực trên tiếng Việt/MCQ/JSON: 85,0%** (135.749/159.710,
  /metrics spec_decode) — dự đoán "phụ thuộc phân phối dữ liệu" đúng,
  và workload này đoán trúng cao.
- +33% conc1 của TASK D tái lập (+28,8% trên graft).
- **Nhưng TTFT p95 nổ theo concurrency** (4,6s ngay ở conc8): FlashInfer
  không hỗ trợ CUDAGraph FULL với spec-decode (log tự hạ về PIECEWISE) +
  draft forward mỗi bước cạnh tranh compute với prefill người mới.
- MTP + prefix caching chung sống ổn (không crash, hit 95,7% — nhích thấp
  hơn 99,37%, nghi do cách đếm draft-token, chưa đào).
- **Khuyến nghị vận hành: MTP là công tắc theo tải — bật khi conc<=4
  (decode +29%, TTFT vẫn <1s), TẮT từ conc8 trở lên (vi phạm SLA 3s).**
  Router tự động theo tải là việc tầng gateway nếu muốn ăn cả hai.

## TASK N1 (2026-08-10): admission cap phía server — THUA RÕ, đóng hướng này

Sweep --max-num-seqs {8,16,24} trên champion graft, client vẫn đổ conc32
(bench_skills 74 câu): baseline không-cap p95 3,03s / 370,6 tok/s;
mns=24 → p95 64,6s (p50 1,77s đẹp nhưng 8 request tràn cap chờ hàng đợi
chi phối đuôi); mns=16 → p95 57,4s; mns=8 → thảm họa (122,7s, 193 tok/s).
- **Kết luận: KHÔNG dùng max-num-seqs làm đòn SLA khi client đổ nguyên
  batch — cap server chỉ chuyển chi phí từ "cùng chậm" sang "đuôi chết
  đói", p95 luôn tệ hơn.** Kiểm soát SLA đúng chỗ = giới hạn concurrency
  phía CLIENT/load-balancer khớp sức chứa server.
- Artifact phương pháp ghi lại: sweep rate dùng --skip-warmup → mức rate
  ĐẦU mỗi lượt hứng chi phí nguội (thấy rate 0,4 "tệ hơn" 0,5 — đảo trực
  giác do thứ tự chạy, không phải số thật). Luật: sweep rate PHẢI warmup
  đầy đủ trước mức đầu tiên.
- Vận hành: runtime wipe lần 3 — champion graft DỰNG LẠI TỪ ĐẦU trong
  ~4,5 phút (frame + GGUF + graft, RMS khớp ~0,005) → quy trình tái tạo
  đã chứng minh rẻ và tất định; backup HF vẫn nên làm.

## Auto-marlin (TASK K/K2-K4) — trạng thái cuối 2026-08-10: cơ khí ĐẠT, đúng đắn CHƯA

Nghiệm thu Qwen3.5-2B GGUF "Q4_0" (thực chất trộn Q6_K/Q5_K/Q8_0/Q4_1 →
buộc ALLOW_K=1, nhánh int8 lossy chưa qua cổng ppl):
- ĐẠT: hook + config resolve + transcode (150 modules, tên GDN đúng sau K4)
  + serve qua MarlinLinearKernel + cache. 6 lớp bug đã bóc/sửa có test
  (A logging, B config upstream, C model_type, D naming GDN, transforms
  A_log/V-head/conv1d, norm 1+w + q/k_norm naming).
- TRƯỢT: completion rác ("The capital of France is notably purple...";
  "2+2?" → "4" rồi suy thoái ***). Tang vật: /content/marlin_transcode_evidence_K4
  (1,8GB) + cache hash cd70221b... Ba giả thuyết chưa phân định: (a) nhánh
  int8 K-quant chưa đủ chính xác cho GDN; (b) còn transform chưa đảo;
  (c) tokenizer/vocab mismatch base vs unsloth GGUF.
- Thí nghiệm phân định rẻ nhất (chưa chạy): thêm cờ transcode ép TOÀN BỘ
  linear_attn.* về fp16 → nếu mạch lạc = lỗi trong đường quant GDN; nếu
  vẫn rác = lỗi ngoài GDN (norm/embed/tokenizer). Một cờ local + một lượt
  serve.
- Lưu ý test-config: architectures phải đổi ForConditionalGeneration →
  ForCausalLM cho checkpoint text-only, và strip mrope như Bug 2 cũ.
- KHÔNG ảnh hưởng champion/graft (đường đó dùng frame RedHatAI + GGUF
  bóc tay, đã PASS ppl). Auto-marlin xếp trạng thái BETA cho tới khi
  phân định xong.

## TASK L (2026-08-10): calibration nhanh — 10-11× nhanh hơn, chất lượng WARN → chỉ làm "chế độ nháp"

`--calib-batch-size 8 --num-samples 128 --max-seq-len 1024` (không GDN):
- Wall time **~14-15 phút** vs ~160 phút chuẩn (10-11×) — đúng ước tính.
- ppl 5,9054 (98/99), ratio 1,1435 vs baseline RedHatAI tươi → **WARN**.
- Phán quyết: KHÔNG dùng cho checkpoint quality-gated; hợp vai trò nháp
  nhanh (kiểm pipeline/sanity) trước bản đầy đủ.
- Caveat đối chứng: chưa có số ppl recipe CHUẨN cùng script không-GDN
  (A2 bị kill từ trước) — so sánh 1-1 chuẩn-vs-nhanh cùng script còn thiếu.
- **NGHI VẤN MỞ đáng giá (confound với kết luận TASK G/G2a)**: fastcalib
  KHÔNG nén GDN mà ppl 5,905 ≈ G/G2a có nén GDN full-calib (5,962-5,965).
  Gợi ý phần lớn khoảng cách ~15% có thể đến từ RECIPE GPTQ của ta nói
  chung (bài calib/tham số) so với recipe RedHatAI, không riêng gì GDN.
  Kết luận "GPTQ-Hessian trên GDN là thủ phạm" cần hạ cấp thành giả thuyết
  chưa cô lập biến; graft M (0,98) vẫn đúng là đường thắng bất kể cách
  diễn giải. Muốn cô lập thật: chạy chuẩn 256/bs1/2048 KHÔNG GDN (~160p)
  rồi so ba chiều — để ngỏ, ưu tiên thấp.

## A/B attention backend (2026-08-10): KHÔNG CÓ trận đấu — FlashInfer là lựa chọn hợp lệ duy nhất

- vLLM 0.26 bỏ env `VLLM_ATTENTION_BACKEND` (log "Unknown ... variable",
  âm thầm bỏ qua — bẫy: hai lượt "A/B" đầu thực chất cùng một backend).
  Thay bằng cờ CLI `--attention-backend` (xem `vllm serve --help=AttentionConfig`).
- `--attention-backend FLASH_ATTN` + fp8_e4m3 KV → ValueError
  "kv_cache_dtype not supported" ngay lúc start: FLASH_ATTN không tương
  thích fp8 KV trong build này. Auto-selection xưa nay chỉ chào
  ['FLASHINFER', 'TRITON_ATTN'] là vì vậy.
- Kết luận: với config production (fp8 KV), FLASHINFER không chỉ thắng mà
  là lựa chọn duy nhất — mặc định hiện tại đã đúng, đóng mục này. Số
  FLASHINFER trên config mnbt1088: conc1 29,71 / conc32 567,5 / p95@0,3
  2,6-2,7s — khớp mọi số đã đo từ trước (vì luôn là backend này).

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
