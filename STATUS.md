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
