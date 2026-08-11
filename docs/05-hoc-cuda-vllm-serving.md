# 05 — Học CUDA → vLLM → Serving qua chính chiến dịch này

Đây là **lộ trình học 6 tầng**, từ bóng bán dẫn lên tới API. Mỗi tầng đều gắn
với một tình huống có thật trong dự án, kèm chỗ để bạn tự nhìn thấy nó.

---

## Tầng 1 — Phần cứng: GPU thực chất là gì

Một GPU gồm hàng nghìn nhân tính toán nhỏ, gom thành các cụm (**SM**), cộng
với bộ nhớ riêng (**VRAM**) nối bằng một đường ống có băng thông hữu hạn.

Số thật của chiếc L4 trong dự án:

| Thông số | Trị số | Ý nghĩa thực tế |
|---|---|---|
| VRAM | 23 GB | Chứa trọng số + cuốn sổ KV. Hết là hết ghế |
| Băng thông bộ nhớ | ~300 GB/s | **Trần của giai đoạn viết chữ** |
| Sức tính lý thuyết | 121 TFLOPS | Con số trên tờ rơi |
| **Sức tính thực đo** | **~55 TFLOPS** | Vì bị giới hạn điện 72W → xung nhịp chỉ ~50% |

**Tensor core** là mạch chuyên dụng nhân ma trận — nhanh hơn nhân thường hàng
chục lần, nhưng *kén ăn*: chỉ nhận dữ liệu xếp đúng khuôn và đúng kiểu số.
Đây là lý do sâu xa vì sao định dạng nén NF4 bị loại khỏi dự án: lưới số của
nó không đổ thẳng vào tensor core được.

> **Bạn đã gặp tầng này ở đâu:** vòng 3 của hành trình — giải thích vì sao đọc
> chỉ được 1.400 chữ/giây thay vì 5.600 như phép chia trên giấy.

**Tự nhìn thấy:** `nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu
--format=csv -l 1` trong lúc chạy — sẽ thấy xung nhịp bị ghìm và điện chạm trần.

## Tầng 2 — CUDA: cách ra lệnh cho GPU

**Kernel** là một hàm chạy đồng thời trên hàng nghìn luồng. Vài khái niệm đủ dùng:

- **thread / block / warp**: luồng đơn lẻ / nhóm luồng chia sẻ bộ nhớ nhanh /
  bó 32 luồng chạy *cùng một lệnh*. Nếu các luồng trong warp rẽ nhánh khác nhau,
  chúng phải chờ nhau → chậm. Đây là lý do "tra bảng từng phần tử" (kiểu NF4)
  đắt trên GPU còn "tính bằng công thức" (kiểu int4) thì rẻ.
- **shared memory**: bộ nhớ nhỏ, cực nhanh, dùng chung trong một block. Kernel
  giỏi = kernel biết chép dữ liệu vào đây một lần rồi dùng nhiều lần.
- **occupancy**: bao nhiêu phần trăm năng lực GPU thực sự có việc làm.

**Chi phí bị lãng quên: mỗi lần CPU phóng một kernel đều tốn thời gian.** Sinh
một chữ cần hàng trăm kernel nhỏ → CPU phóng lệnh trở thành nút thắt. Cách
chữa: **CUDA graph** — ghi lại cả chuỗi lệnh một lần rồi phát lại như một khối.
Trong dự án, riêng việc này đóng góp **+74%** ở tình huống một người dùng.

> **Bạn đã gặp:** khi bạn hỏi "hướng tối ưu graph để hạn chế số lần phóng lệnh
> từ CPU?" — câu trả lời là vLLM đã bật sẵn ở chế độ mạnh nhất.

**Tự nhìn thấy:** trong log khởi động vLLM có dòng ghi số hình dạng CUDA graph
đã ghi lại (capture).

## Tầng 3 — Thư viện kernel: ai làm việc gì

| Thư viện | Chuyên trị | Trong dự án |
|---|---|---|
| **cuBLAS** | Nhân ma trận chuẩn (16/32-bit) | Dùng cho giai đoạn đọc khi batch lớn |
| **Marlin** | Nhân ma trận **trọng số nén 4/8-bit** | ⭐ Kernel quyết định tốc độ của ta |
| **FlashAttention / FlashInfer** | Phép attention (tra sổ KV) | FlashInfer là lựa chọn hợp lệ duy nhất khi dùng KV nén fp8 |
| **Triton** | Ngôn ngữ viết kernel bằng Python | Đường dự phòng của GGUF; cũng là nơi cài kernel GDN |

**Vì sao Marlin nhanh** (đáng học vì nó gói gọn mọi nguyên lý ở tầng 1-2): xếp
sẵn trọng số đúng khuôn tensor core từ lúc nạp (khỏi xáo lúc chạy) · giải nén
bằng *phép tính* chứ không tra bảng, và đan cài việc giải nén vào lúc tensor
core đang bận · chia việc để đường ống bộ nhớ luôn đầy ở mọi cỡ batch.

## Tầng 4 — Mô hình: chuyện gì xảy ra bên trong một lượt trả lời

```
Câu hỏi → [tokenize] → prefill (đọc cả câu, tính KV cho từng chữ)
        → decode: sinh chữ 1 → sinh chữ 2 → ... (mỗi chữ đọc lại toàn bộ trọng số + tra KV)
        → [detokenize] → trả về
```

Điểm cần khắc cốt: **prefill là bài toán tính toán, decode là bài toán băng
thông**. Cùng một mô hình, hai nút thắt khác nhau, hai loại thuốc khác nhau.

**Kiến trúc lai (GDN)** của Qwen3.5: 75% số tầng không giữ cuốn sổ KV dày lên
mãi mà giữ một *trạng thái tóm tắt* cỡ cố định. Ưu: nhẹ bộ nhớ, chịu ngữ cảnh
dài. Nhược: **quá mới** → phần lớn bug trong file 03 sinh ra từ đây.

## Tầng 5 — vLLM: bộ máy phục vụ

Đây là tầng đáng học nhất nếu bạn muốn làm serving. Bốn cơ chế lõi:

**5.1. PagedAttention — quản bộ nhớ KV như hệ điều hành quản RAM.** Cuốn sổ KV
được cắt thành các *khối* cỡ cố định, cấp phát rời rạc. Nhờ vậy nhiều phiên
dùng chung khối giống nhau, và không bị phân mảnh.

**5.2. Continuous batching — không chờ cả lô.** Mỗi nhịp, hệ thống ghép những
ai đang cần phục vụ vào một mẻ; ai xong thì rời, ai mới đến chen vào ngay nhịp
sau. Đây là lý do một GPU phục vụ được nhiều người.

**5.3. Bộ xếp lịch có một "ngân sách chữ" mỗi nhịp** — và người đang được phục
vụ được ưu tiên trước người mới. Chính cơ chế này giải thích phát hiện đau đớn
nhất của dự án: khi khách đến liên tục, phần *đọc* của người mới **ăn vào ngân
sách** của phần *viết* cho người cũ → mọi người cùng chậm. Thuốc là hạ ngân
sách xuống 1.088 chữ mỗi nhịp.

**5.4. Prefix cache + cách thu hồi.** Các khối KV được băm theo nội dung; ai có
đoạn đầu giống nhau thì dùng chung khối. Khi hết chỗ, khối cũ nhất bị bỏ theo
luật LRU. Quan trọng: **vLLM thu hồi bằng cách "tính lại", không phải "cất
tạm"** — nghĩa là một phiên bị đá ra sẽ phải đọc lại từ đầu khi quay lại. Đây
chính là rủi ro lớn nhất của kịch bản agent gọi tool (phiên nằm im chờ tool).

**5.5. Cách vLLM nạp mô hình đã nén.** Có nhiều "phương pháp lượng tử hóa"
(compressed-tensors, GPTQ, AWQ...); mỗi phương pháp khai báo cách đọc trọng số
đóng gói và chọn kernel tương ứng. Khi ta ghép checkpoint lai, chính tầng này
là nơi mọi bug xảy ra (chọn nhầm quy tắc, không có đường nạp cho lớp Embedding...).

**Tự đọc mã nguồn** (đã clone sẵn tại `D:\Training\AI_Module\vllm\vllm\vllm`):

| Muốn hiểu | Đọc file |
|---|---|
| Xếp lịch, ngân sách chữ, thu hồi | `v1/core/sched/scheduler.py` |
| Quản khối KV, prefix cache, LRU | `v1/core/block_pool.py`, `v1/core/kv_cache_utils.py` |
| Nạp trọng số nén | `model_executor/layers/quantization/` |
| Chọn kernel attention | `v1/attention/backends/` |

## Tầng 6 — Serving: từ động cơ đến dịch vụ

**Các chỉ số phải phân biệt:**
- **TTFT** — chờ chữ đầu (người dùng cảm nhận rõ nhất)
- **ITL / tok/s mỗi người** — tốc độ chữ chạy (cần ≥ ~10 để nhanh hơn mắt đọc)
- **Throughput tổng** — năng suất hệ (dùng cho bài toán chi phí)
- ⚠️ Ba cái này **đánh đổi lẫn nhau**: nhồi thêm người thì tổng tăng, mỗi người
  chậm đi, đuôi TTFT xấu đi. Mọi quyết định vận hành là chọn điểm trên đường
  đánh đổi này.

**Đo cho đúng:** *closed-loop* (đổ cả rổ rồi đợi) cho số đẹp nhưng vô nghĩa;
*open-loop* (khách đến theo nhịp ngẫu nhiên) mới ra điểm gãy thật. Dự án này
lệch 6 lần vì lẫn hai kiểu.

**Kiểm soát tải đặt ở đâu:** đã chứng minh bằng thực nghiệm — chặn ở **máy chủ**
làm đuôi chờ nổ; phải chặn ở **cổng vào** (client/gateway) theo sức chứa thật.

**Vận hành:** khởi động lại thì cache trống → phải *làm nóng* trước khi nhận
khách (blue-green). Streaming SSE dễ bị proxy gom buffer làm hỏng trải nghiệm
dù server vô tội.

---

## Lộ trình tự học đề xuất (theo thứ tự)

1. **Tuần 1 — nhìn thấy phần cứng:** chạy `nvidia-smi` trong lúc benchmark;
   quan sát xung nhịp, điện, độ chiếm dụng. Hiểu vì sao "GPU 100%" không có
   nghĩa là đã tối ưu.
2. **Tuần 2 — hiểu hai giai đoạn:** chạy `bench_skills.py` với prompt ngắn rồi
   prompt dài; tự thấy prefill và decode phản ứng khác nhau.
3. **Tuần 3 — chạm vào bộ nhớ:** bật/tắt prefix caching, bật/tắt fp8 KV; đo
   TTFT và số phiên chứa được. Đây là bài học đắt giá nhất trong serving.
4. **Tuần 4 — đọc mã vLLM:** bắt đầu từ `scheduler.py`, tìm chỗ trừ "ngân sách
   chữ"; đối chiếu với hiện tượng đã đo ở vòng 6.
5. **Tuần 5 — nén mô hình:** chạy `graft_gguf_gdn.py` với `--bits 4` và `8`,
   so tốc độ và chất lượng. Tự trải nghiệm đánh đổi.
6. **Tuần 6 — vận hành:** dựng blue-green với `warmup_prefix.py`, đo open-loop
   bằng `bench_sla_prefix.py`, tự tìm điểm gãy của chính mình.

**Nguyên tắc học xuyên suốt:** mỗi khi đọc được một cơ chế, hãy tìm cách *nhìn
thấy nó bằng một phép đo*. Toàn bộ dự án này là chuỗi những lần làm đúng điều
đó — và ba lần đau nhất là những lần bỏ qua nó.
