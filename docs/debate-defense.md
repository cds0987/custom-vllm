# Thẩm định độc lập — ba trần

Vai trò: bên thủ. Điều tra độc lập, không đọc `docs/debate-attack.md`. Mỗi
kết luận có dẫn chứng file:dòng hoặc URL.

## Trần 1 — Prefill 1.433 tok/s (kết luận cũ: 55 TFLOPS × 58% GEMM × 90% = vật lý)

### (a) Nguồn của tỷ lệ Amdahl 58/42 — KHÔNG đáng tin như trình bày

Con số "58% GEMM / 42% non-GEMM" chỉ xuất hiện trong tài liệu diễn giải
(`docs/02-hanh-trinh-toi-uu.md:69`, `docs/05-hoc-cuda-vllm-serving.md:20`),
không có trong bảng profiling thật. Bảng profiling thật nằm ở
`STATUS.md:328-336` ("Profile kernel trên L4, torch.profiler, eager"):

| bucket | GGUF prefill |
|---|---|
| matmul/GEMM | **47,9%** |
| norm/elementwise | 30,9% (eager, chưa fusion) |
| GDN/fla | 2,1% |
| attention | 8,7% |

Đây là 47,9%, không phải 58%. Và chính STATUS.md tự cảnh báo phép đo này
là **eager mode** ("chưa fusion") — 30,9% norm/elementwise là mức thổi
phồng do không có kernel fusion (CUDA Graph / torch.compile sẽ gộp phần
lớn norm+elementwise vào ít kernel launch hơn, giảm % này đáng kể trong
runtime thật). STATUS.md cũng ghi nhận độc lập "lỗi bucketing 7,7pp" và
"artifact eager-mode" như câu hỏi đặt ra — xác nhận đúng: **con số 58/42
dùng trong kết luận "vật lý" là suy diễn từ tài liệu giảng dạy, không
khớp với số đo thật (47,9/52,1 eager), và số đo thật lại bị lệch theo
hướng phóng đại phần non-GEMM vì thiếu fusion.**

**Hệ quả**: nếu chạy lại profiling ở chế độ production thật (CUDA graph
piecewise, không eager) và GEMM chiếm >47,9%, thì phần "vật lý bất khả"
bị đánh giá thấp hơn thực tế — tức là **còn dư địa GEMM lớn hơn ước tính
cũ**, không phải ít hơn.

### (b) int8 trên Ada có thật gấp đôi fp16 không — CÓ, khác với case fp8

Tra cứu số liệu chính thức L4 (Ada AD104, sm89): FP16 Tensor Core 121
TFLOPS dense (242 với sparsity 2:4), INT8 Tensor Core 242 TOPS dense (485
với sparsity). Tỷ lệ 242/121 = đúng 2,0× — khớp với "55 TFLOPS thực" ×2 =
110 TFLOPS-tương-đương nếu chuyển sang int8 GEMM thuần. Đây LÀ đặc tính
kiến trúc thật của 4th-gen Tensor Core Ada, khác về bản chất với trường
hợp fp8 mà dự án đã kết luận không nhân đôi MFU:
- fp8 KV cache trong STATUS.md ("Kết quả L4") được đo trong **decode**,
  là workload băng-thông-giới-hạn (~300GB/s), không phải compute-giới-hạn
  — nên dù compute per-op nhanh gấp đôi, decode không hưởng lợi vì nút
  thắt là đọc KV từ VRAM, không phải GEMM. STATUS.md ghi đúng: fp8 KV
  "miễn phí về tốc độ" (không tăng, không giảm) — tức consistent với giả
  thuyết memory-bound, không phải bằng chứng "Ada không nhân đôi MFU".
- Prefill là compute-bound (batch token lớn, GEMM chiếm gần nửa CUDA
  time đo được). Đây chính là chế độ mà tensor-core 2× int8 CÓ cơ hội
  biểu hiện, khác hẳn ngữ cảnh decode của phát hiện fp8 cũ.
Kết luận: **không có bằng chứng int8 rơi vào "bẫy Ada" như fp8** — bẫy đó
là do workload memory-bound, không phải do kiến trúc int8/fp8 tự nó vô
dụng. Nhưng dự án CHƯA đo MFU thực tế int8 GEMM lớn (M~batch tokens) trên
sm89 — số 2× là trần lý thuyết trên tờ rơi, giống hệt cách "121 TFLOPS
trên tờ rơi" từng bị hạ xuống 55 TFLOPS thực đo vì power limit. Không có
lý do tiên nghiệm để MFU int8 = MFU fp16 (90% theo giả định cũ) — chưa
đo là chưa biết.

Nguồn: NVIDIA L4 Tensor Core GPU datasheet (121 TFLOPS FP16, 242 TOPS
INT8, dense) — https://www.nvidia.com/en-us/data-center/l4/ ,
Ada Lovelace whitepaper — https://images.nvidia.com/aem-dam/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf

### (c) Chi phí quantize activation lúc chạy — CHƯA ĐO, rủi ro thật

Prefill int8 GEMM cần per-token (hoặc per-tensor) scale cho activation
tính động mỗi bước forward (không giống weight-only AWQ/GPTQ đã quantize
tĩnh lúc build checkpoint). Chi phí này gồm: reduce max-abs theo hàng,
chia + làm tròn, tất cả trên tensor kích thước batch×hidden — là các
kernel elementwise/reduction thêm, đúng loại việc đã đo chiếm 30,9%
non-GEMM ở trên (dù con số đó là fp16, việc quantize hoạt động thêm
sẽ CỘNG vào bucket này, không giảm). Dự án CHƯA có số đo trực tiếp cho
overhead này trên prefill dài (12k token); ước lượng lạc quan "90% hiệu
suất kernel" trong tài liệu (`docs/02-hanh-trinh-toi-uu.md:71`) là giả
định, không phải số đo — cùng loại giả định thiếu cơ sở như (a).

### (d) Giới hạn điện 72W — RÀO CẢN CỨNG xác nhận, không nới được trên Colab

`docs/05-hoc-cuda-vllm-serving.md:17-20` ghi L4 bị khoá 72W (thiết kế
card, không phải setting mềm — L4 là card 1-slot low-profile, TDP gốc
72W theo spec NVIDIA, không phải giới hạn phần mềm áp đặt tuỳ ý).
`nvidia-smi -pl` sửa được power limit MỀM chỉ khi (i) có quyền root/
CAP_SYS_ADMIN trên driver và (ii) giá trị nằm trong khoảng [min, max]
mà **vendor card cho phép qua VBIOS** — với L4 passive-cooled datacenter
card, TDP 72W THƯỜNG LÀ CẢ min và max (không có headroom ép xung như card
gaming). Trên Colab cụ thể: người dùng không có quyền root với GPU
passthrough ảo hoá (thường chỉ có `compute` capability, không có
`sys_admin`), nên `nvidia-smi -pl` gần như chắc chắn trả lỗi permission
ngay cả khi VBIOS cho phép nới. Đây là rào cản KÉP: (1) khả năng cao
VBIOS L4 không có biên trên 72W để nới, (2) kể cả có, Colab không cấp
quyền. Dự án CHƯA thử lệnh thật để xác nhận thực nghiệm (không tìm thấy
log `nvidia-smi -pl` nào trong STATUS.md) — đây là câu hỏi rẻ, nên thử.

**Kết luận Trần 1: KHẢ THI NHƯNG ĐẮT ở phần (b)/(c) — int8 prefill GEMM có
cơ sở kiến trúc thật (2× tensor-core Ada, khác bẫy fp8), nhưng dự án chưa
đo MFU thực tế lẫn overhead quantize động, nên "1.433 tok/s = trần vật
lý" đang dựa trên phép nhân Amdahl KHÔNG khớp số đo thật (47,9% đo được,
lại còn bị lệch do eager mode) và một giả định 90% hiệu suất kernel chưa
kiểm chứng. Trần 72W là BẤT KHẢ xác nhận (thiết kế card + quyền Colab).
Kết luận cũ "1.433 = vật lý" bị đánh giá là chưa được chứng minh chặt —
có khả năng thật có headroom qua int8 GEMM prefill, mức độ chưa biết.**

## Trần 2 — Kernel GDN chunk ≤ 64 (`solve_tril.py:381`)

### Assert bảo vệ gì

`D:\Training\AI_Module\custom_vllm\out\tensorrt-llm\tensorrt_llm\_torch\modules\fla\solve_tril.py:381`:
```python
assert A.shape[-1] in [16, 32, 64]
```
Đọc toàn bộ file: thuật toán nghịch đảo ma trận tam giác dưới theo khối
dùng đệ quy khối 2×2 (block matrix inversion via Schur complement),
NHƯNG được **viết tay, unroll cứng** cho từng cỡ:
- `solve_tril_16x16_kernel` (dòng 25-69): giải khối nền 16×16 bằng vòng
  lặp tuần tự `for i in range(1, 16)` — đây LÀ ràng buộc toán học cơ sở
  (kích thước warp/tile 16 khớp instruction `tl.dot` MMA 16×16).
- `merge_16x16_to_32x32_inverse_kernel` (dòng 82-141): hợp 2 khối 16×16
  thành 32×32 bằng 1 phép nhân khối `Ai_21 = -Ai_22 @ A_21 @ Ai_11`.
- `merge_16x16_to_64x64_inverse_kernel` (dòng 154-356): hợp 4 khối 16×16
  thành 64×64 bằng 6 phép nhân khối tường minh viết tay (Ai_21, Ai_32,
  Ai_43, Ai_31, Ai_42, Ai_41 — công thức đệ quy Schur cho ma trận khối
  4×4 hạ tam giác).

Không có kernel `merge_..._to_128x128_...`. Hàm dispatcher `solve_tril`
(dòng 359-426) rẽ nhánh cứng `if BT == 32 ... else merge_16x16_to_64x64`
— cỡ 128 đơn giản **không có nhánh code**, không phải bị chặn bởi giới
hạn tài nguyên runtime (shared memory/register) mà bị chặn bởi **thiếu
implementation**: không ai viết hàm `merge_16x16_to_128x128_inverse_kernel`
(sẽ cần 8 khối 16×16, tổ hợp Schur đệ quy 3 tầng, ~28 block-dot thay vì
6 — tăng theo O(n²) số khối tam giác dưới).

### Số học shared memory cho chunk=128 — KHÔNG vượt giới hạn Ada

Kernel merge nạp các khối A (fp32, 16×16 = 1KB/khối) và Ai (fp32,
16×16 = 1KB/khối) qua `tl.load`/block_ptr — với BT=64 kernel hiện tại
nạp tối đa 6 khối A + 4 khối Ai/Ad đồng thời ≈ 10KB dữ liệu registers/
shared per block (Triton tự quản lý qua register file trước, tràn thì
mới xuống shared). Với BT=128 (8 khối 16×16), số khối A cần nạp đồng
thời cho tam giác dưới đầy đủ là C(8,2)=28 khối × 1KB = 28KB, cộng Ai 8
khối × 1KB = 8KB — tổng ~36-40KB nếu giữ tất cả trong flight cùng lúc
(có thể giảm bằng cách tính tuần tự theo tầng đệ quy thay vì nạp hết).
Giới hạn shared memory per SM của Ada (sm89, L4) là **100KB** (tối đa
99KB dùng được sau reserve, theo CUDA C Programming Guide bảng
"Technical Specifications per Compute Capability" cho compute capability
8.9). 36-40KB < 100KB — **không vượt giới hạn phần cứng**. Đây củng cố
kết luận: chunk=128 KHÔNG bị chặn bởi tài nguyên phần cứng, chỉ bị chặn
bởi công sức viết kernel (unroll tay 28 block-dot thay vì 6, dễ sai sót
đại số nhưng cơ học, không phải bất khả).

### Lợi ích lý thuyết của chunk lớn hơn — nhỏ, đã được đo thực nghiệm ở BT=32

STATUS.md TASK P2 (dòng 656-675) đã đo A/B **thật** cho chunk=32 vs 64:
chunk=32 thắng **+7,6%** ở conc1 (độ trễ đơn luồng), **~0%** ở conc32 (đã
bão hoà song song). Đây là bằng chứng thực nghiệm hướng NGƯỢC với trực
giác "chunk lớn hơn luôn tốt hơn vì ít lần phóng kernel": chunk NHỎ HƠN
(32 < 64) mới là chiều thắng đo được, vì lượng công việc non-parallel
(vòng lặp tuần tự trong `solve_tril_16x16_kernel`, dòng 59-63) tỷ lệ
thuận với BT, còn số lần phóng kernel giảm theo 1/BT — hai hiệu ứng
ngược chiều, và ở tải thấp phần tuần tự thắng. Ngoại suy sang BT=128:
nhiều khả năng lặp lại hoặc khuếch đại xu hướng của 32→64 (tệ hơn ở
conc thấp), KHÔNG chắc mang lại lợi ích ròng — pattern đo được mâu thuẫn
với giả thuyết "chunk lớn hơn = thắng" trong chính STATUS.md dòng 672-674
("Ứng viên cho hàng đợi tấn công... đáng điều tra"), tự nhận là suy đoán
chưa kiểm chứng, không phải kết luận.

**Kết luận Trần 2: KHẢ THI VÀ RẺ về mặt kỹ thuật (không có rào cản phần
cứng, chỉ cần viết thêm 1 kernel Triton theo đúng khuôn 3 kernel đã có,
~1 buổi code + kiểm thử số học đối chiếu `torch.linalg.inv`), nhưng
NGHI VẤN CAO về lợi ích — bằng chứng thực nghiệm sẵn có (32 vs 64) đi
NGƯỢC hướng "chunk lớn hơn thắng", nên khả năng cao chunk=128 sẽ tiếp
tục xu hướng đó (thắng nhẹ hoặc thua ở conc thấp, trung tính ở conc cao)
chứ không phải một chiến thắng lớn bị bỏ lỡ. Đây là trường hợp "chưa ai
làm vì lợi ích kỳ vọng thấp so với công viết kernel", không phải "bất
khả".**

## Trần 3 — Cascade attention khoá cứng trong FlashInfer backend

### (a) Lý do upstream tắt — xác nhận bằng chính PR gốc

`D:\Training\AI_Module\vllm\vllm\vllm\v1\attention\backends\flashinfer.py:1461-1468`:
```python
def use_cascade_attention(self, *args, **kwargs) -> bool:
    if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
        return False
    # TODO: Cascade attention doesn't work, disable it for now
    # return use_cascade_attention(*args, **kwargs)
    return False
```
`git log -S` xác định commit gốc: `f1fc2107a` — vllm PR **#26130**,
"[Bugfix] Disable cascade attention with FlashInfer" (Michael Goin,
2025-10-02). Nội dung PR (đọc trực tiếp qua `gh pr view 26130`):

> Using cascade attention with FlashInfer on Blackwell seems to break
> when prefix caching is hit. This eval **hangs** on the first batch of
> requests... `Avg prompt throughput: 0.0 tokens/s ... Running: 100 reqs`
> Adding `--no-enable-prefix-caching` or `--disable-cascade-attn` works
> fine. It happens with both the flashinfer and trtllm backends, with
> cudagraphs and eager. **We haven't tracked down the issue yet**, so
> for now we will disable cascade attention.

Lỗi thật là **treo (hang)** dưới tải nhiều request đồng thời + prefix
cache hit, không phải sai kết quả số học đơn lẻ — và tác giả upstream tự
nhận **chưa root-cause được**, nên đây không phải "một cấu hình hẹp đã
biết rõ" mà là bug mở, nguyên nhân không rõ, phạm vi ảnh hưởng không rõ.
Tên báo cáo là "Blackwell" nhưng PR tắt cascade cho toàn bộ FlashInfer
backend không phân biệt kiến trúc — có thể là biện pháp phòng ngừa rộng
tay hơn mức cần thiết, nhưng dự án **không có cách nào tự xác minh nó
không xảy ra trên Ada** vì chưa ai điều tra nguyên nhân gốc.

### (b) Rủi ro bật lại — cổng byte-identical của dự án KHÔNG bắt được

Đây là phát hiện quan trọng nhất: lỗi biểu hiện dưới **prefix caching +
nhiều request đồng thời (100 reqs trong report)**, KHÔNG phải sai kết
quả âm thầm ở một request đơn lẻ. Cổng chính của dự án
(`STATUS.md` TASK F, dòng 414-416: "cùng prompt... output cache-hit
byte-identical với output cold") chạy so sánh **1 request tại một thời
điểm** (cold vs warm, tuần tự) — đúng loại kịch bản KHÔNG chạm được bug
này, vì bug cần tranh chấp nhiều request cùng lúc đọc/ghi cascade
wrapper's shared prefix state. Một cổng single-request sẽ báo PASS dù
cascade có bug treo dưới tải — **rủi ro thật, không phải giả định**: nếu
tự bật `--no-disable-cascade-attn`, quy trình QA hiện tại của dự án
(TASK F/N6, luôn test 1 request cold/warm) sẽ không phát hiện được lỗi;
chỉ có các bài đã chạy nhiều request đồng thời + prefix caching (TASK H,
TASK F2, N6 conc4/conc16) mới có cơ hội chạm — và các bài đó CHƯA BAO GIỜ
chạy với cascade bật (bị chặn cứng False), nên dự án hiện tại 0 dữ liệu
thực nghiệm về việc nó treo hay không trên stack Qwen3.5/Ada.

### (c) Lợi ích thật cho kiến trúc lai 25% full-attention

Cascade attention tăng tốc bằng cách tính phần prefix DÙNG CHUNG một lần
(shared KV, ví dụ system prompt) thay vì mỗi request trong batch quét
lại toàn bộ prefix riêng — lợi ích tỷ lệ thuận với: (tỷ trọng tính toán
attention trong tổng decode) × (mức độ chia sẻ prefix giữa các request
đang chạy cùng batch). Với model lai Qwen3.5 GDN, theo chính STATUS.md
Profile kernel (dòng 328-336): bucket "attention" chỉ chiếm **0,4% CUDA
time ở decode** (0,6% cho fp16 decode) — vì chỉ 25% layer là full
attention, phần còn lại là GDN (không có KV cache attention để cascade
chia sẻ). Trần lý thuyết tiết kiệm của cascade với kiến trúc này ≈
0,4-0,6% tổng thời gian decode × (tỷ lệ request có prefix trùng nhau
trong cùng batch, ước lượng optimisitc 80-95% theo TASK H hit-rate
99% nhưng đó là **prefix cache thường** đã hoạt động, không phải
cascade). Cascade attention chỉ cộng thêm lợi ích BIÊN trên phần attention
compute vốn đã nhỏ — với 0,4-0,6% tổng ngân sách, dù cascade "tiết kiệm
100%" phần đó thì tổng speedup decode tối đa lý thuyết cũng chỉ **dưới
1%**. Đây khác hẳn với model attention thuần (Llama/Qwen dense) nơi
attention có thể chiếm 20-40% CUDA time và cascade có ý nghĩa thật.

**Kết luận Trần 3: BẤT KHẢ VỀ MẶT AN TOÀN QA hiện tại + KHẢ THI NHƯNG
KHÔNG ĐÁNG (đắt về rủi ro, rẻ về lợi ích)**. Rào cản không phải kỹ thuật
đơn thuần (code path tồn tại, chỉ bị `return False` chặn — 1 dòng để mở
lại) mà là: (1) bug upstream là HANG dưới tải, root cause chưa biết,
không giới hạn rõ ràng vào Blackwell; (2) quy trình kiểm chứng
byte-identical hiện tại của dự án là single-request nên KHÔNG có khả
năng phát hiện bug loại này; (3) ngay cả nếu bật thành công và không
treo, lợi ích lý thuyết cho kiến trúc 25%-full-attention của Qwen3.5 GDN
là <1% tổng decode — quá nhỏ để bù rủi ro treo production. **Đây là lý
do hợp lý tại sao "chưa ai bật lại" — không phải bỏ sót cơ hội, mà là
đánh giá lợi ích/rủi ro đúng đắn.**

## Câu hỏi còn mở — xếp theo mức độ rẻ (thí nghiệm mới trả lời được)

1. **(giây, không cần GPU)** Đọc `git log`/issue tracker vllm-project để
   xem có PR nào sau #26130 (2025-10-02) đã root-cause hoặc mở lại
   cascade cho FlashInfer trên Ada/Hopper — có thể upstream đã tiến
   triển từ đó tới nay (2026-08).
2. **(giây, 1 lệnh)** Thử `nvidia-smi -q -d POWER` rồi `nvidia-smi -pl
   <X>` trên phiên Colab L4 hiện tại để xác nhận thực nghiệm khoá cứng
   72W (permission denied hay giá trị max thực sự = 72W) thay vì suy
   luận từ spec sheet.
3. **(vài phút, không cần sửa kernel)** Re-run `torch.profiler` prefill
   ở chế độ **không eager** (CUDA graph piecewise, đúng production) để
   lấy lại tỷ lệ GEMM/non-GEMM thật, thay số 58/42 (tài liệu) hoặc 47,9/
   52,1 (eager, lệch) bằng số đúng chế độ chạy thật — sửa trực tiếp nền
   tảng của Trần 1.
4. **(10-30 phút)** Bật `--no-disable-cascade-attn` trên L4 với đúng
   kịch bản PR #26130 mô tả (FlashInfer + prefix caching + ~100 request
   đồng thời) để xem có treo trên Ada như trên Blackwell không — rẻ vì
   chỉ cần 1 cờ, nhưng PHẢI test đúng workload nhiều-request (không phải
   cổng single-request hiện có) để có ý nghĩa.
5. **(30-60 phút)** Micro-benchmark int8 GEMM thuần (cuBLAS/CUTLASS int8,
   M~batch-tokens tương tự prefill 12k) trên L4 để đo MFU thật so với
   fp16 — trả lời trực tiếp câu "int8 có thật gấp đôi thông lượng hiệu
   dụng hay chỉ gấp đôi trên tờ rơi", tách biệt khỏi chi phí quantize
   activation.
6. **(nửa ngày, cần code)** Viết `merge_16x16_to_128x128_inverse_kernel`
   theo đúng khuôn 3 kernel hiện có (đệ quy Schur 3 tầng, 28 block-dot),
   đối chiếu số học với `torch.linalg.inv`, rồi A/B chunk=128 vs 64/32 ở
   cả conc1 và conc32 — trả lời dứt điểm liệu xu hướng "chunk nhỏ thắng"
   đo được ở 32-vs-64 có tiếp diễn hay đảo chiều ở 128.
7. **(nhiều giờ, rủi ro cao)** Đo overhead quantize-activation động
   thật bằng cách viết một đường prefill int8 GEMM tối thiểu (dequant +
   cuBLAS int8 hoặc CUTLASS) cắm vào pipeline hiện có, so tổng thời gian
   với đường fp16 hiện tại trên đúng workload LongAlign-12k — đây là
   thí nghiệm đắt nhất nhưng là thí nghiệm DUY NHẤT trả lời trực tiếp
   "1.433 tok/s có thật là trần vật lý hay không".
