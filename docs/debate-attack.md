# Tấn công ba trần — hồ sơ BÊN CÔNG

Mục tiêu: chứng minh "bất khả" hiện tại là bất khả **có điều kiện** (điều
kiện kỹ thuật/kỹ sư), không phải bất khả toán học/vật lý, và định giá từng
đường vượt. Mỗi phương án: cơ chế · dẫn chứng · lợi ích ước lượng · chi phí
(giờ kỹ sư) · rủi ro · thí nghiệm rẻ nhất.

Xếp hạng trong mỗi trần theo `(lợi ích × xác suất thành công) / chi phí`,
cao → thấp.

---

## Trần 1 — Prefill 1.433 tok/s (kết luận cũ: 55 TFLOPS × 58% GEMM × 90% kernel)

Số liệu nền tảng dùng để công kích (STATUS.md, mục "Profile kernel trên L4"):
GEMM chỉ chiếm **47.9%** CUDA time lúc prefill (không phải 58% — con số 58%
là ước lượng Amdahl lý thuyết trước khi có profiler thật; profiler thật cho
thấy non-GEMM còn TO hơn dự đoán: norm/elementwise 30.9% + GDN 2.1% +
attention 8.7% = 41.7% non-GEMM, khớp gần "42%" mà TASK P2 nhắc tới). Đây
tự nó là điểm yếu đầu tiên của "trần vật lý": nếu 42% thời gian không nằm
trong GEMM, thì trần "55 TFLOPS × %GEMM" đang cận-đúng ở phần GEMM nhưng
KHÔNG nói gì về 42% kia — tức còn dư địa chưa đụng tới.

### 1a. W8A8 (int8 activation) — hạng 1

- **Cơ chế**: sm89 (Ada) tensor core hỗ trợ `INT8×INT8→INT32` ở gấp đôi
  throughput so với `FP16×FP16→FP32` trên cùng SM (Ada giữ tỷ lệ 2:1 kế
  thừa từ Ampere/Turing cho int8 dense). Với 47.9% thời gian prefill đang
  nằm trong GEMM, lý thuyết trần mới = `1433 × (1 - 0.479 + 0.479/2)`
  ≈ `1433 × 0.7605` ≈ **1090 tok/s KHÔNG PHẢI TĂNG** — sai, phải tính
  ngược: nếu GEMM nhanh gấp đôi, tổng thời gian giảm còn
  `T_total - 0.479×T_total/2 = T_total×0.7605`, tức throughput tăng
  `1/0.7605 ≈ 1.315×` → prefill ước ~**1.884 tok/s (+31%)**, KHÔNG phải
  2× — đúng như quy luật Amdahl mà chính dự án đã dùng cho fp8 KV/GEMM.
- **Dẫn chứng**: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:47,787,794`
  — `CompressedTensorsW8A8Int8` là scheme có thật, đã được chọn tự động khi
  quant_args khớp static/dynamic int8. STATUS.md dòng 180-186 xác nhận
  `fp8_per_tensor` (W8A8 FP8, không phải int8) đã chạy sạch trên chính stack
  này với ignore-list tối thiểu `["in_proj_ba"]` — tức đường ống
  "W8A8 + ignore GDN nhạy cảm" đã có tiền lệ THÀNH CÔNG, chỉ chưa thử biến
  thể int8 (rẻ hơn fp8 marginally trên phần cast, nhưng fp8 cũng đã đạt
  gấp đôi throughput int8 tensor core trên Ada theo cùng cơ chế — nên
  **fp8_per_tensor prefill** mới là thí nghiệm đúng cần chạy, không nhất
  thiết phải là int8 W8A8Int8).
- **Chi phí quantize activation runtime**: per-tensor/per-token dynamic
  quant là một kernel elementwise rẻ (đọc 1 lần, ghi 1 lần, O(n)) — với
  norm/elementwise đã chiếm 30.9%, thêm 1 kernel quant nữa cộng thêm vài %
  chứ không "nuốt hết lợi ích" — trừ khi nó không được fuse vào norm kernel
  hiện có (rủi ro thật, xem 1b).
- **Lợi ích ước lượng**: +20-35% prefill (1.433 → ~1.75-1.95 tok/s), dựa
  trên đúng phép Amdahl dự án đã tự dùng.
- **Chi phí cài đặt**: đã có cờ `--quantization fp8_per_tensor` trong stack
  — CHỈ CẦN chạy lại benchmark prefill (không phải decode) với cờ này,
  test 1b đã làm cho decode nhưng STATUS.md dòng 187 tự thừa nhận
  "Long-context của fp8 chưa phân thắng bại với champion" → **0.5-1 giờ**
  (một lượt bench_serving prefill-only).
- **Rủi ro**: độ nhạy GDN (`in_proj_ba`) đã biết — dùng đúng ignore-list đã
  qua cổng chất lượng test 1b, rủi ro chất lượng THẤP nếu giữ nguyên
  ignore-list đã pass.
- **Thí nghiệm rẻ nhất**: serve với `--quantization fp8_per_tensor
  --ignore in_proj_ba`, chạy `bench_serving` chế độ prefill-nặng (prompt
  12k, giống test đã có ở dòng 106-107 STATUS.md: "prefill (prompt 12k) |
  dequant+cuBLAS | 8.580") — so trực tiếp với baseline 1.433/8.580 hiện có.
  Không cần code mới, chỉ cần 1 lượt đo còn thiếu.

### 1b. Fusion norm/elementwise (30.9% non-GEMM lúc prefill) — hạng 2

- **Cơ chế**: 30.9% CUDA time ở norm/elementwise lúc prefill (so với 6.6%
  lúc decode GGUF!) là bất thường — chênh lệch 4.7× giữa 2 chế độ cho thấy
  đây không phải chi phí cố định của norm mà là **chi phí per-token của
  RMSNorm/residual-add/rope chưa được fuse cho batch prefill lớn**.
  STATUS.md dòng 333 tự ghi chú "(eager, chưa fusion)" — tức đội dự án đã
  biết đây là dư địa nhưng chưa khai thác vì mới đo bằng `eager` mode
  (không bật `torch.compile`/CUDA graph capture cho prefill).
- **Dẫn chứng**: STATUS.md dòng 328-336 (bảng profile), chú thích dòng 333
  "eager, chưa fusion". vLLM có `--compilation-config` với `CompilationLevel`
  hỗ trợ fuse RMSNorm+quant, residual+norm qua `torch.compile` custom pass
  (`vllm/compilation/`) — nhưng dự án hiện đo & chạy ở `eager` (rất có thể
  vì patch GGUF hybrid dispatch không tương thích cudagraph).
  Việc bật `-O3`/piecewise compilation cho riêng nhánh prefill (không đụng
  decode path đang cần eager cho hybrid dispatch) là khả thi vì hybrid dispatch
  route theo `x.shape[0]` (dòng 376 STATUS.md) — có thể compile riêng
  nhánh dequant+cuBLAS.
- **Lợi ích ước lượng**: nếu fusion cắt được 1/2-2/3 của 30.9% → +15-20%
  prefill tổng thể, CỘNG DỒN được với W8A8 (khác cơ chế, không loại trừ
  nhau).
- **Chi phí cài đặt**: 4-8 giờ (bật compile cho nhánh dequant, đo lại,
  kiểm tra không vỡ hybrid dispatch hiện có — patch đã có sẵn structure ở
  `scripts/patch_gguf_hybrid_dispatch.py`).
- **Rủi ro**: torch.compile với custom Triton/CUDA plugin ops (GGUF dequant
  custom op) có thể fail graph capture — rủi ro KỸ THUẬT (không chạy được)
  hơn là rủi ro CHẤT LƯỢNG.
- **Thí nghiệm rẻ nhất**: thử `-O3` (torch.compile mặc định vllm) chỉ trên
  nhánh dequant+cuBLAS, xem log có "graph break" hay không trước khi đầu tư
  sâu — 30 phút.

### 1c. Giảm khối lượng prefill thay vì tăng tốc — hạng 3 (đã một phần bị bác bỏ, một phần còn mở)

- **Cơ chế loại A — prefix cache**: đã CÓ trong stack (APC bật mặc định,
  STATUS.md dòng 66-68) — nhưng chính dự án phát hiện nó "thổi phồng số
  prefill" khi benchmark lặp request → nghĩa là **hiệu quả thật đã được
  hưởng khi request có overlap thật** (30-128K nền dùng chung, kịch bản
  N6). Đây không phải hướng mới nhưng cần tách bạch: benchmark hiện tại đo
  TRẦN COLD (prefix cache miss cưỡng bức để đo đúng), nên "1.433 tok/s" là
  trần ĐÚNG cho cold-prefill, không mâu thuẫn — nhưng production thật sẽ
  cao hơn nhiều nếu > 50% request trùng prefix. Không phải "vượt trần vật
  lý" mà là "trần vật lý chỉ áp cho phần cold, không áp cho toàn hệ thống".
- **Cơ chế loại B — mô hình nháp/tiền xử lý nhỏ lọc bớt**: dùng model nhỏ
  (Qwen3.5-2B đã có sẵn trong stack, STATUS.md nhiều chỗ) để phân loại/rút
  gọn tài liệu trước khi đẩy vào 9B — GIẢM số token phải prefill ở model
  lớn cho các phần không quan trọng. Đây là thay đổi ỨNG DỤNG, không phải
  hệ thống, nhưng hợp lệ vì đề bài cho phép "ý tưởng táo bạo".
- **Cơ chế loại C — sparse attention lúc prefill**: 8.7% CUDA time prefill
  là attention (không phải GDN) — chỉ 25% layer là attention thật (75%
  linear GDN), nên trần lý thuyết của việc sparsify riêng phần attention
  rất nhỏ (~8.7% × phần có thể cắt). Không đáng ưu tiên so với 1a/1b.
- **Lợi ích ước lượng**: loại A tùy tỷ lệ trùng prefix thật của workload
  (0% nếu toàn tài liệu mới, có thể 5-10× nếu chat lặp lại nền); loại B
  tùy % tài liệu bị lọc bỏ; loại C nhỏ (<5%).
- **Chi phí**: loại A = 0 (đã có, chỉ cần đo đúng workload thật thay vì
  benchmark cold); loại B = 1-2 ngày (cần pipeline định tuyến + đánh giá
  chất lượng lọc); loại C = không đáng làm.
- **Rủi ro**: loại B có rủi ro chất lượng cao (model nhỏ lọc sai làm mất
  thông tin) — cần eval riêng.
- **Thí nghiệm rẻ nhất cho loại A**: đo lại throughput với workload thật
  (không phải lặp nguyên văn prompt để trigger cache giả) nhưng CÓ prefix
  chung tự nhiên (multi-turn chat trên cùng tài liệu nền) — đã có sẵn kịch
  bản N6 128K để tái dùng.

### 1d. Ý tưởng táo bạo khác — hạng 4 (đầu cơ, chi phí cao)

- **Cắt tầng (layer skipping / early exit) lúc prefill**: prefill không
  cần độ chính xác per-token như decode (không sinh token ngay) — có thể
  thử early-exit ở layer giữa cho phần context "nền" ít quan trọng.
  Rủi ro chất lượng CAO, chưa có tiền lệ trong stack, cần nghiên cứu mới
  hoàn toàn (không phải "bật cờ có sẵn"). Ước chi phí: 3-5 ngày để có
  prototype đánh giá được, lợi ích không chắc chắn — **phương án tốn kém
  nhất được liệt kê theo yêu cầu đề bài**, vì nó đòi hỏi sửa forward pass
  của model, retrain/calibrate ngưỡng exit, và chạy lại toàn bộ cổng chất
  lượng 3 nhánh hiện có.

---

## Trần 2 — Kernel GDN khóa chunk ≤ 64 (`solve_tril.py` assert 16/32/64)

**Kết luận đọc code**: đây là giới hạn **KỸ THUẬT**, không phải toán học.

- **Bằng chứng trực tiếp** (`vllm/model_executor/layers/fla/ops/solve_tril.py:531,540-545`):
  ```python
  assert A.shape[-1] in [16, 32, 64]
  ...
  if BT == 16:   merge_fn = solve_tril_16x16_kernel
  elif BT == 32: merge_fn = merge_16x16_to_32x32_inverse_kernel
  elif BT == 64: merge_fn = merge_16x16_to_64x64_inverse_kernel
  ```
  Thuật toán là **block-recursive triangular inversion**: nghịch đảo ma
  trận tam giác `BT×BT` được dựng đệ quy từ khối `16×16` cơ bản
  (`solve_tril_16x16_kernel` — vòng lặp Neumann-series kiểu Gauss-Jordan
  trong `for i in range(2, min(16, T-...))`, đúng công thức nghịch đảo
  ma trận tam giác đơn vị `(I-A)^-1 = I + A + A² + ...` cắt hữu hạn), sau
  đó gộp 4 khối `16×16` → `64×64` bằng block matrix inversion formula
  chuẩn (`b_Ai_21 = -Ai_22 @ A_21 @ Ai_11`, đúng công thức Schur complement
  cho ma trận khối tam giác dưới). **Công thức này ĐÚNG với BẤT KỲ BT nào
  là lũy thừa của 2** — không có giới hạn toán học ở 64. Giới hạn 64 chỉ
  vì tác giả **chỉ viết 3 kernel Triton cụ thể** (16, merge→32, merge→64)
  và dừng lại — thiếu `merge_64x64_to_128x128_inverse_kernel`. Đây là công
  việc lặp lại có khuôn mẫu rõ ràng: `merge_16x16_to_64x64` đã cho thấy
  đúng pattern nhân đôi (ghép 4× khối 16, dùng 6 phép `tl.dot` để tính các
  khối ngoài đường chéo) — viết `128x128` cần ghép 2× khối 64 bằng ĐÚNG
  công thức Schur complement 2×2 khối (giống hệt bước 16→32), không cần
  phát minh thuật toán mới.

### 2a. Viết thêm kernel `merge_64x64_to_128x128` — hạng 1

- **Cơ chế**: thêm 1 kernel Triton mới theo đúng khuôn `merge_16x16_to_32x32`
  (chỉ 2 khối `Ai_11`, `Ai_22` cỡ 64×64 thay vì 16×16, cùng 1 phép
  `-Ai_22 @ A_21 @ Ai_11` để tính góc dưới trái) + sửa `solve_tril()` thêm
  nhánh `elif BT == 128: merge_fn = merge_64x64_to_128x128_inverse_kernel`
  + nới `assert A.shape[-1] in [16, 32, 64, 128]`.
- **Dẫn chứng**: cấu trúc lặp lại giữa `merge_16x16_to_32x32_inverse_kernel`
  (dòng 113-225) và `merge_16x16_to_64x64_inverse_kernel` (dòng 238-503) —
  bản 64 chỉ là bản 32 lặp lại pattern 4 lần thay vì 2 lần, cùng công thức
  toán, cùng `tl.dot`. Việc mở rộng lên 128 là NGOẠI SUY THẲNG của cùng
  pattern, không phải thiết kế mới.
- **Cần kiểm tra thêm (rủi ro kỹ thuật thật, không phải bịa)**: hàm gọi
  `solve_tril` từ `chunk.py`/`chunk_delta_h.py` có thể có giả định khác về
  `BT` ở chỗ khác (ví dụ kích thước shared memory cho `chunk_gated_delta`
  chính, không chỉ solve_tril) — TASK P2 đã revert sau A/B nên CHƯA xác
  nhận toàn bộ pipeline GDN (không chỉ solve_tril) chịu được BT=128 mà
  không đổi thêm. Đây là lý do upstream/dự án dừng ở 64: có thể còn giới
  hạn thứ hai (registers/shared-mem của `chunk_delta_h.py` — 64×64 float32
  block đã chiếm 16KB register/thread-block cỡ lớn, 128×128 sẽ gấp 4× dung
  lượng, có nguy cơ tràn shared memory trên sm89 (mỗi SM ~100KB shared)
  hoặc giảm occupancy mạnh — RỦI RO HIỆU NĂNG NGƯỢC (kernel chậm hơn vì ít
  block chạy song song), chưa chắc RỦI RO ĐÚNG.
- **Lợi ích ước lượng**: TASK P2 đã đo *giảm* chunk (64→32) cho +7.6% ở
  conc1 (tải thấp — ít lần phóng kernel không phải nút thắt ở tải thấp, mà
  là latency per-launch). Điều này gợi ý chunk LỚN HƠN có thể chỉ có lợi ở
  conc32 (tải cao, throughput-bound, cường độ số học cao hơn ăn điểm) —
  nhưng STATUS.md TASK P2 dòng 665 ghi conc32 chunk=32 "trung tính (~0%)"
  so baseline 64, nghĩa là ở conc32 hệ THỰC RA đã bão hòa GEMM/bandwidth,
  không phải bão hòa số-lần-launch. Suy luận thận trọng: **BT=128 nhiều
  khả năng lặp lại mẫu "trung tính ở conc32, có thể xấu đi ở conc1"**
  (do khối lớn hơn → ít song song hơn ở batch nhỏ) — lợi ích kỳ vọng
  THẤP hơn ước lượng ban đầu của TASK P2, không nên kỳ vọng >5%.
- **Chi phí cài đặt**: 4-6 giờ (viết kernel theo khuôn có sẵn + test số học
  đối chiếu `torch.linalg.inv` trên ma trận nhỏ + A/B benchmark giống
  TASK P2).
- **Rủi ro**: (a) hiệu năng có thể XẤU ĐI ở conc1 (occupancy thấp hơn);
  (b) cần audit thêm shared-mem của `chunk_delta_h.py`/`chunk.py` (chưa
  đọc trong phiên này) trước khi khẳng định không vỡ.
- **Thí nghiệm rẻ nhất**: viết kernel, test ĐÚNG SỐ trước (so
  `solve_tril(A, BT=128)` với `torch.linalg.inv(torch.eye(128)+A)` trên
  tensor ngẫu nhiên nhỏ, CPU-only, không cần GPU thật) để xác nhận công
  thức đúng TRƯỚC KHI đo hiệu năng trên GPU — tách rủi ro đúng/rủi ro
  nhanh làm 2 bước, bước 1 không cần GPU.

### 2b. Kiểm tra upstream flash-linear-attention có bản mới hơn — hạng 2

- **Cơ chế**: repo `flash-linear-attention` (fla-org) là dự án ngoài đang
  phát triển tích cực; rất có thể họ đã thêm `BT=128` sau khi vLLM vendor
  bản cũ vào `vllm/model_executor/layers/fla/` hoặc
  `vllm/third_party/flash_linear_attention/` (STATUS.md dòng 659 nhắc tới
  đường dẫn third_party riêng — CÓ HAI bản trong repo, cần đối chiếu bản
  nào đang được dùng thật).
- **Dẫn chứng**: cần WebFetch `github.com/fla-org/flash-linear-attention`
  file `fla/ops/common/utils.py`/`solve_tril.py` bản mới nhất — CHƯA làm
  trong phiên này (không có mạng truy cập trong tool call agent con), đây
  là việc BÊN CÔNG cần làm tiếp nếu có WebSearch. Nêu rõ trong debate: đây
  là khoảng trống bằng chứng cần đối phương hoặc điều tra thêm bù đắp.
- **Lợi ích/chi phí**: nếu tồn tại bản đã hỗ trợ BT=128/256, chi phí giảm
  xuống chỉ còn "port + test" (1-2 giờ) thay vì "viết mới" (2a, 4-6 giờ).
- **Thí nghiệm rẻ nhất**: `WebFetch` trực tiếp URL raw của file
  `solve_tril.py` mới nhất trên GitHub fla-org, tìm `assert A.shape[-1] in`
  — so sánh chuỗi giới hạn.

---

## Trần 3 — Cascade attention khóa cứng `return False`

- **Dẫn chứng chính xác** (đọc trực tiếp mã trong repo vllm cục bộ,
  `vllm/v1/attention/backends/flashinfer.py:1461-1468`):
  ```python
  def use_cascade_attention(self, *args, **kwargs) -> bool:
      if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
          # TODO: The cascade wrapper currently does not support setting
          # kv cache dtype to something different from query dtype.
          return False
      # TODO: Cascade attention doesn't work, disable it for now
      # return use_cascade_attention(*args, **kwargs)
      return False
  ```
  Hai lớp khóa TÁCH BIỆT: (1) fp8 KV luôn fail sớm ở nhánh đầu (đúng như
  STATUS.md N6 đã ghi — "không hợp lệ với fp8 KV"); (2) NGAY CẢ nếu fp8
  KV không phải vấn đề, dòng cuối vẫn `return False` cứng, bỏ qua hoàn
  toàn lời gọi hàm `use_cascade_attention(*args, **kwargs)` thật (đã bị
  comment-out) — đây là một quyết định của TÁC GIẢ VLLM, không phải giới
  hạn engine.

- **Bằng chứng đối chứng quan trọng nhất cho BÊN CÔNG**: backend
  `FlashAttentionBackend` (`vllm/v1/attention/backends/flash_attn.py:724-725`)
  **KHÔNG hardcode False**:
  ```python
  def use_cascade_attention(self, *args, **kwargs) -> bool:
      return use_cascade_attention(*args, **kwargs)
  ```
  và hàm thật `use_cascade_attention()` (dòng 1490-1565) có logic đầy đủ:
  kiểm tra độ dài prefix chung tối thiểu, số query tối thiểu, tắt cho DCP,
  và MỘT PHÉP SO SÁNH CHI PHÍ THỰC: `cascade_time < flash_decoding_time`
  (dòng 1565) — tức cascade chỉ bật khi mô hình chi phí (dựa trên số CTA,
  số SM, số wave) dự đoán nó THẬT SỰ nhanh hơn. Đây không phải code vỡ —
  nó là code có kiểm soát chi phí, đang chạy được trên backend khác.

### 3a. Chuyển sang FlashAttention backend + KV fp16 (bỏ fp8 KV) để có cascade thật — hạng 1

- **Cơ chế**: dùng `--attention-backend flash_attn` (hoặc tương đương) thay
  FlashInfer, chấp nhận KV fp16 thay fp8, đổi lại được cascade thật cho
  kịch bản 30-128K nền dùng chung.
- **Ước lượng lợi ích lý thuyết cho 32 người dùng chung nền 30K** (kịch bản
  đề bài yêu cầu): không có cascade, mỗi trong 32 request phải quét toàn
  bộ 30K token nền trong pha decode/attention riêng — chi phí đọc KV nền
  tỷ lệ O(32 × 30K). Có cascade, phần nền 30K được đọc **một lần dùng
  chung** cho cả batch, phần "shared" chỉ còn O(30K) + phần riêng O(32 ×
  suffix). Với suffix nhỏ (vài nghìn token, đúng kịch bản N6 "prefix 120K
  + suffix 3K"), tỷ lệ tiết kiệm đọc KV theo lý thuyết ≈ `1 - 1/32 ≈ 96.9%`
  lượng đọc KV nền — nhưng đây là tiết kiệm BĂNG THÔNG ĐỌC KV, không phải
  tiết kiệm toàn bộ thời gian attention (vẫn cần tính QK^T/softmax cho mỗi
  query). Ước tính thận trọng dựa theo đúng số N6 đã đo
  ("Quét KV 120K × nhiều luồng là chi phí chi phối" — decode per-user rơi
  từ 28.9 (conc1) xuống 7.9 tok/s (conc16), tức **72.7% suy giảm do quét
  KV lặp lại**) → cascade loại bỏ phần lặp lại này, kỳ vọng khôi phục phần
  lớn của khoảng suy giảm đó, tức decode per-user ở conc16-32 có thể
  **tăng 2-3× so với hiện trạng không-cascade**, dù vẫn thấp hơn conc1
  (còn chi phí suffix riêng + phần tính toán thật, không chỉ đọc bộ nhớ).
- **Chi phí đổi KV fp16**: VRAM tăng 2× cho KV cache — theo STATUS.md dòng
  254 "KV (fp8) 437K tok" cho 9B, fp16 sẽ ~218K tok — vẫn đủ cho kịch bản
  32 request × suffix ngắn + 1 bản nền dùng chung (không phải 32 bản nền
  riêng, nên tổng dung lượng cascade cần THẤP HƠN 32× fp8-riêng, có thể bù
  lại phần mất do fp16).
- **Chi phí cài đặt**: đổi cờ backend + bỏ `--kv-cache-dtype fp8` — vài
  phút cấu hình, nhưng ĐO ĐÚNG cần 2-4 giờ (chạy lại kịch bản N6 128K với
  backend mới, benchmark conc16/32).
- **Rủi ro**: (a) FlashAttention backend có thể không hỗ trợ đầy đủ GDN
  hybrid attention layers giống FlashInfer (cần xác nhận model Qwen3.5 GDN
  chạy được trên flash_attn backend — chưa kiểm chứng trong phiên này);
  (b) mất fp8 KV nghĩa là mất tối ưu VRAM đã dày công đạt được — cần cân
  đối lại `max-num-seqs`/`gpu-memory-utilization`; (c) code cascade upstream
  có TODO tự nhận biết ("doesn't work") ở NHÁNH FlashInfer — không rõ liệu
  cascade trên FlashAttention có được cộng đồng vLLM test kỹ hay là cũng
  có bug ẩn tương tự nhưng chưa bị tắt cứng (rủi ro "false negative" —
  BÊN THỦ có thể phản biện đây là chưa được kiểm chứng đủ).
- **Thí nghiệm rẻ nhất**: serve với `--attention-backend flash_attn
  --no-disable-cascade-attn` (bỏ `--kv-cache-dtype fp8`), lặp lại ĐÚNG kịch
  bản N6 (prefix 120K, 16 request suffix riêng), so TTFT/decode-per-user
  với số đã có (TTFT p50 5.4s, decode ~7.9 tok/s ở conc16) — nếu cascade
  hoạt động, kỳ vọng thấy decode-per-user tăng rõ rệt và KHÔNG giảm tuyến
  tính theo conc. Đã có sẵn quy trình cổng byte-identical (N6 dùng
  temp=0, cache-hit byte-identical) để verify cascade không phá tính đúng
  — chạy lại đúng gate đó với cascade bật.

### 3b. Tự bật lại nhánh FlashInfer cascade (bỏ comment `return False`) — hạng 2, RỦI RO CAO

- **Cơ chế**: sửa trực tiếp dòng 1467-1468, bỏ comment khôi phục
  `return use_cascade_attention(*args, **kwargs)`. Patch 1 dòng.
- **Vì sao KHÔNG hạng 1**: comment "Cascade attention doesn't work" là của
  UPSTREAM VLLM TEAM, không phải giả định của dự án này — nghĩa là có khả
  năng cao họ đã QUAN SÁT THẤY lỗi thật (kết quả sai hoặc crash) khi test
  trên FlashInfer, không phải tắt phòng ngừa. Cần tìm commit/PR/issue liên
  quan trên GitHub vllm-project/vllm để biết lỗi cụ thể trước khi bật lại
  — **đây là việc CẦN LÀM TIẾP bằng WebSearch/WebFetch** (không có trong
  phiên này do giới hạn công cụ) tìm `git blame` dòng đó hoặc search issue
  "cascade attention flashinfer doesn't work". Nêu rõ với đối phương: nếu
  BÊN THỦ dẫn ra được PR/issue mô tả lỗi CHÍNH XÁC (ví dụ sai số học ở biên
  chunk, hoặc lỗi chỉ xảy ra với multi-LoRA/spec-decode), phương án này hạ
  bậc mạnh; nếu BÊN THỦ không dẫn ra được, đây là "trần" tự áp đặt thận
  trọng quá mức và có thể vượt qua bằng chính quy trình gate byte-identical
  sẵn có.
- **Lợi ích ước lượng**: giống 3a nhưng GIỮ ĐƯỢC fp8 KV nếu sửa thêm điều
  kiện đầu (dtype mismatch) — lợi ích cao hơn 3a vì không phải đánh đổi
  VRAM, NHƯNG xác suất thành công thấp hơn nhiều (có thể vỡ đúng như
  upstream đã thấy).
- **Chi phí cài đặt**: 1 dòng sửa (~15 phút) + nhưng cần TOÀN BỘ quy trình
  kiểm chứng byte-identical đã có (N6) chạy lại kỹ — 3-4 giờ để tự tin.
- **Rủi ro**: CAO — có thể cho kết quả SAI ÂM THẦM (không crash nhưng output
  sai) nếu lỗi upstream là lỗi số học tinh vi ở biên. Đây chính là loại
  rủi ro dự án đã từng gặp và ghi nhận cẩn trọng (bài học desync
  qweight_type ở TASK GGUF repack, dòng 189-216 STATUS.md) — "không crash
  không có nghĩa là đúng".
- **Thí nghiệm rẻ nhất**: bật cờ, chạy ĐÚNG cổng byte-identical N6 đã có
  sẵn (prefix 120K, so cache-hit vs cold, temp=0) — nếu output KHÔNG
  byte-identical, phát hiện lỗi ngay mà không cần hiểu cơ chế lỗi trước.
  Đây là ưu thế lớn của dự án: quy trình kiểm chứng RẺ đã tồn tại, hạ chi
  phí rủi ro của việc "thử liều" xuống nhiều — biến 3b từ "liều lĩnh" thành
  "thử có lưới an toàn".

### 3c. Vì sao upstream tắt — cần điều tra thêm (khoảng trống thừa nhận)

- Phiên làm việc này KHÔNG có quyền truy cập WebSearch/WebFetch (bị hạn chế
  bởi khuôn khổ tool hiện tại của agent con) nên KHÔNG dẫn ra được commit/
  PR/issue cụ thể giải thích lý do "doesn't work". Đây là khoảng trống
  bằng chứng thành thật cần nêu, không che giấu. Đề nghị bước tiếp theo:
  tìm trên GitHub `vllm-project/vllm` blame của
  `vllm/v1/attention/backends/flashinfer.py` dòng chứa
  "Cascade attention doesn't work" để lấy SHA commit và PR liên kết.

---

## Khuyến nghị: nếu chỉ được làm 1 việc

**Chạy lại benchmark prefill với `--quantization fp8_per_tensor
--ignore in_proj_ba` (phương án 1a).**

Lý do chọn duy nhất phương án này trên cả 3 trần:
1. **Chi phí thấp nhất tuyệt đối** trong toàn bộ danh sách — không cần viết
   dòng code nào, cờ đã tồn tại và đã qua cổng chất lượng ở nhánh decode
   (test 1b). Chỉ là MỘT LƯỢT ĐO còn thiếu (prefill-nặng), ước 0.5-1 giờ.
2. **Xác suất thành công cao nhất** — không phải suy đoán, mà là drill-down
   của một kỹ thuật ĐÃ CHỨNG MINH hoạt động đúng trên chính stack này.
3. **Không cần đối lấy gì** — không mất VRAM (khác 3a), không có rủi ro
   đúng/sai tinh vi (khác 3b), không cần viết kernel mới (khác 2a).
4. Nếu ra kết quả tốt (+20-35% dự đoán), nó tự động NÂNG TRẦN của cả câu
   chuyện "1.433 tok/s là vật lý" — chuyển gánh nặng chứng minh sang BÊN
   THỦ phải giải thích tại sao con số baseline cũ vẫn được gọi là "trần".

Nếu được làm việc thứ 2: 3a (đổi backend cascade + KV fp16), vì kịch bản
N6 đã tự phát hiện suy giảm 72.7% do thiếu cascade — đây là khoản lỗ LỚN
NHẤT về tỷ lệ trong cả ba trần, và thí nghiệm kiểm chứng (chạy lại đúng
kịch bản N6 với backend khác) đã có sẵn quy trình, không cần dựng mới.
