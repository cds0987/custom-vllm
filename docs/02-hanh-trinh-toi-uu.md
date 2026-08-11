# 02 — Hành trình tối ưu: làm gì, theo thứ tự nào, và vì sao

File này kể lại theo đúng mạch logic: **mỗi bước sinh ra từ câu hỏi mà bước
trước để lại**. Đọc xong bạn sẽ hiểu vì sao thứ tự lại là như vậy.

---

## Vòng 0 — Làm cho nó *chạy được* đã

**Câu hỏi:** mô hình mới ra, thư viện phục vụ (vLLM) có chạy nổi không?

**Trả lời:** không. Qwen3.5 dùng kiến trúc lai quá mới, vLLM chưa đăng ký, cấu
hình sai khớp, tên tham số lệch nhau... Phải viết **~20 miếng vá** mới nạp nổi
mô hình.

**Bài học đặt nền cho cả dự án:** khi đứng ở biên giới công nghệ, phần lớn công
sức không phải "tối ưu" mà là "làm cho nó đừng vỡ". Mọi miếng vá đều lưu thành
script tự động (`scripts/setup_env.sh`) để dựng lại môi trường bằng một lệnh —
quyết định này về sau cứu dự án 6 lần khi máy ảo bị xóa trắng.

## Vòng 1 — Nén tới đâu thì gãy?

**Câu hỏi:** nén 4 bit đã tốt, vậy 3 bit, 2 bit thì sao? (bạn yêu cầu thử tới
2 bit nhưng cấm hy sinh chất lượng)

**Cách làm:** chạy thang bit từ cao xuống thấp, mỗi mức đo *cả tốc độ lẫn chất
lượng*, chất lượng chấm bằng bộ đề thật (SWE-bench) chứ không phải vài câu hỏi
mẫu.

**Kết quả — hai điều bất ngờ:**
1. Dưới 4 bit **mất chất lượng** (mô hình bắt đầu nói lảm nhảm không biết dừng).
2. Dưới 4 bit còn **mất cả tốc độ** (274 → 236 → 219 → 167 chữ/giây) — vì cách
   đóng gói 2-3 bit rắc rối hơn, bóc hộp tốn công hơn.

➡️ **4 bit là sàn.** Đóng vĩnh viễn hướng "nén sâu hơn". Từ đây, muốn nhanh hơn
phải tìm đường khác — dẫn thẳng tới vòng 2.

## Vòng 2 — Cùng 4 bit, nhưng đóng gói kiểu khác

**Câu hỏi:** nếu không nén sâu hơn được, thì cùng 4 bit có cách đóng gói nào
nhanh hơn không?

**Cách làm:** đấu loại các định dạng trên cùng một máy, cùng bài đo, cùng cổng
chất lượng.

| Định dạng (đều 4 bit) | Tốc độ | Chất lượng |
|---|---|---|
| GGUF Q4_K_M (sau khi ta vắt kiệt) | 274 | đạt |
| **W4A16 + kernel Marlin** | **563** | **tốt nhất** |
| AWQ tự nén | 616 | ✗ rớt |

**Vì sao chênh gấp đôi:** GGUF đóng gói kiểu "hộp trong hộp" phải bóc lúc dùng;
W4A16 xếp sẵn đúng khuôn GPU cần. *Cùng số bit, khác đường đi.*

➡️ **Đổi champion sang W4A16.** Nhưng vòng này để lại một câu hỏi lớn: tại sao
AWQ nhanh hơn mà chất lượng rớt? → dẫn tới vòng 5.

## Vòng 3 — Trần vật lý nằm ở đâu?

**Câu hỏi (của bạn):** card ghi 121 TFLOPS, mô hình cần 18 GFLOP mỗi chữ, vậy
sao chỉ đọc được 1.400 chữ/giây thay vì 5.600?

**Cách làm:** đo trực tiếp bằng phép nhân ma trận thuần, đọc xung nhịp và điện
năng lúc chạy.

**Trả lời — ba tầng hao hụt:**
1. Card L4 bị **giới hạn điện 72W** (thiết kế, không sửa được) → xung nhịp chỉ
   đạt ~50% → **55 TFLOPS thực**, không phải 121.
2. Chỉ **58% công việc** là phép nhân ma trận; 42% còn lại là các phép khác
   (chuẩn hóa, attention, GDN) — dù phép nhân nhanh vô hạn thì 42% kia vẫn đó.
3. Hiệu suất kernel ~90%.

`55 × 0,58 × 0,9 ≈ 1.400 chữ/giây` — khớp số đo.

➡️ **Trần đọc là vật lý, không phải phần mềm.** Kết luận này định hình mọi thứ
sau đó: nếu không thể đọc nhanh hơn, thì phải **đọc ít đi** → dẫn tới vòng 4.

## Vòng 4 — Đừng đọc lại thứ đã đọc (prefix caching)

**Câu hỏi (kịch bản của bạn):** phục vụ 32.000 chữ ngữ cảnh, trong đó bộ kỹ
năng ~30.000 chữ giống nhau cho mọi người, câu hỏi riêng chỉ ~2.500 chữ.

**Cách làm:** bật bộ nhớ đệm tiền tố, nhưng **cổng kiểm tra đầu tiên là tính
đúng đắn, không phải tốc độ** — cùng câu hỏi, so câu trả lời khi có cache và
khi không, phải **giống nhau từng byte**.

**Kết quả:**
- Đúng đắn: PASS (cả ở 32K, 64K và sau này 128K).
- Chờ chữ đầu: **10,5 giây → 0,2-1,4 giây**, tỷ lệ dùng lại **99,4%**.

➡️ Đây là đòn lớn nhất của cả dự án, và nó **không đến từ việc làm GPU nhanh
hơn** mà từ việc *bỏ bớt việc*. Nhưng nó đẻ ra câu hỏi mới: vậy giới hạn thật
khi khách đến liên tục là bao nhiêu? → vòng 6.

## Vòng 5 — Vì sao nén phần GDN lại làm hỏng chất lượng?

**Câu hỏi:** checkpoint tốt nhất thế giới (RedHatAI) chỉ nén 25% mô hình, để
nguyên 75% phần GDN ở 16 bit. Nén nốt phần đó có nhanh hơn không?

**Trở ngại:** không ai nén được — vì vLLM có bug, cứ gặp phần GDN đã nén là
sập lúc nạp. **Ta vá chính vLLM** (kèm chứng minh toán học rằng phép ghép mới
là chính xác từng bit).

**Thí nghiệm 1 (TASK G):** tự nén phần GDN bằng GPTQ → **768 chữ/giây (+36%)**
nhưng chất lượng **rớt 15%**. Không được phong.

**Thí nghiệm 2 (TASK G2):** nén nhẹ tay hơn (8 bit thay vì 4) → chất lượng
**y hệt** (không cải thiện gì) mà còn chậm hơn. ⟹ *Giả thuyết "nén sâu quá" là
SAI. Thủ phạm là cách GPTQ ước lượng độ quan trọng trên phần GDN.*

**Thí nghiệm 3 (TASK M) — nước cờ quyết định:** nếu GPTQ hỏng ở đó, hãy **mượn
phần GDN từ bản GGUF** (llama.cpp nén theo kiểu khác, đã chứng minh giữ chất
lượng), ghép vào khung W4A16.

| | Tốc độ | Chất lượng (thấp = tốt) |
|---|---|---|
| Champion cũ | 563 | 5,132 |
| Ghép (int8) | 610 | 5,051 |
| **Ghép (int4) = champion hiện tại** | **~611** | **4,778 — tốt hơn cả bản gốc** |

➡️ Thắng cả hai trục, không đánh đổi gì. Bài học: **khi một phương pháp hỏng ở
một chỗ cụ thể, hãy thay đúng chỗ đó thay vì bỏ cả phương pháp.**

## Vòng 6 — Cam kết dịch vụ thật là bao nhiêu?

**Câu hỏi:** phục vụ được bao nhiêu yêu cầu mỗi giây mà vẫn giữ "chờ dưới 3 giây"?

**Bẫy đã mắc:** đo kiểu "đổ cả rổ 32 yêu cầu cùng lúc" cho **1,3 yêu cầu/giây**.
Đo lại theo kiểu khách đến rải rác (Poisson) — **chỉ 0,2**. Số cũ lạc quan gấp
6 lần.

**Truy nguyên nhân bằng số liệu hệ thống:** hàng đợi luôn rỗng, bộ nhớ mới dùng
37% → không phải nghẽn hàng đợi, không phải hết bộ nhớ. Thủ phạm: **người mới
đến chen phần "đọc" vào giữa các nhịp "viết" của người đang được phục vụ**.

**Thuốc:** giới hạn mỗi nhịp GPU chỉ được nhai tối đa 1.088 chữ →
**cam kết nâng lên 0,3 yêu cầu/giây, 0% vi phạm**. (1.088 là *sàn cứng* — thấp
hơn thì mô hình không khởi động được.)

**Thuốc thử nhưng THẤT BẠI:** chặn bớt số người vào cửa ở phía máy chủ → tệ hơn
hẳn (đuôi chờ nổ lên 64 giây). Bài học: **kiểm soát tải phải làm ở cổng vào
(client/gateway), không phải ở máy chủ.**

## Vòng 7 — Các đòn phụ: cái nào ăn, cái nào không

| Đòn | Kết quả | Ghi chú |
|---|---|---|
| **Chạy theo lô, không qua server** | ✅ **+34,8%** (522 chữ/giây) | Chế độ đúng cho việc tự động hóa chạy đêm |
| **MTP (đoán trước 1 chữ)** | ✅ +29% khi vắng khách; ❌ phá cam kết khi đông | Dùng như công tắc theo tải. Tỷ lệ đoán trúng trên tiếng Việt: 85% |
| Đoán bằng cách dò lại prompt (ngram) | ❌ Loại | Sụp đổ khi đông (chờ 101 giây) |
| Nén nốt lớp phát chữ (lm_head) | ❌ Bất khả | vLLM không có đường nén cho loại lớp đó |
| Đổi kernel attention | ❌ Không có lựa chọn | Bản đang dùng là lựa chọn hợp lệ duy nhất |
| Chia sẻ lượt tra sổ (cascade) | ❌ Bị khóa | Chính vLLM tắt cứng: "chưa chạy được" |
| Tăng tốc khâu nén (calibration) | ⚠️ Nhanh 10× nhưng chất lượng WARN | Chỉ dùng làm "bản nháp" |

## Vòng 8 — Ngữ cảnh cực dài

**Câu hỏi (của bạn):** 64.000 hay 128.000 chữ thì sao?

**Kết quả:** cả hai **đậu cổng đúng đắn** (câu trả lời giống hệt từng byte).
Ở 128.000 chữ: chờ chữ đầu ~1 giây, nhưng số người phục vụ mượt giảm còn
**6-10** (bộ nhớ chịu được 32, nhưng tốc độ viết cho mỗi người rơi xuống ~8
chữ/giây khi đông).

➡️ **Giá của nền lớn tính bằng số ghế**: nền 30K cho ~30 người; nền 128K cho
~6-10 người.

## Đang làm — vòng 9: vòng lặp agent gọi tool

Lớp bài toán mới bạn giao: đọc yêu cầu → gọi công cụ → đọc kết quả → gọi tiếp →
trả lời, nhiều người cùng lúc. Đặc thù: **giữa các lượt, phiên nằm im chờ công
cụ chạy** trong khi vẫn chiếm bộ nhớ. Câu hỏi trung tâm: *sau 5 giây chờ, phiên
đó còn được nhớ hay phải làm lại từ đầu?* Đang đo.

---

## Sợi chỉ đỏ xuyên suốt

1. **Giảm byte phải khuân** (nén đúng cách) — vòng 1, 2, 5.
2. **Bỏ bớt việc phải làm** (nhớ phần dùng chung) — vòng 4.
3. **Đừng để việc này giẫm chân việc kia** (chia nhịp, xếp hàng) — vòng 6.
4. **Không tin số cho tới khi đo đúng cách** — bẫy cache, bẫy đổ-cả-rổ, bẫy
   dùng lại số cũ. Mỗi bẫy đều từng suýt đưa dự án đi sai đường.

➡️ Tiếp theo: [03-bug-va-cach-sua.md](03-bug-va-cach-sua.md)
