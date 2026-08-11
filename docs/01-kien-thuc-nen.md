# 01 — Kiến thức nền: máy đang làm gì khi "chạy AI"?

Không cần biết gì trước. Đọc xong file này bạn sẽ hiểu được mọi quyết định
trong file kế tiếp.

---

## 1. Mô hình AI, về mặt vật chất, là một đống số

Mô hình Qwen3.5-9B có **9 tỷ con số** (gọi là *trọng số*). Khi bạn hỏi một câu,
máy làm phép nhân giữa câu hỏi (đã chuyển thành số) và đống trọng số đó, rồi
ra chữ tiếp theo. Rồi lặp lại cho chữ sau. Từng chữ một.

Điểm mấu chốt: **mỗi chữ sinh ra, máy phải kéo gần như toàn bộ 9 tỷ số kia
từ bộ nhớ ra**. Không phải tính toán khó — mà là *khuân vác* nặng.

> **Ví von xuyên suốt tài liệu này:** GPU như một đầu bếp cực nhanh, nhưng kho
> nguyên liệu ở xa. Đầu bếp không bao giờ là nút thắt — **đường vận chuyển từ
> kho tới bếp mới là nút thắt**. Gần như mọi tối ưu trong dự án này là "làm
> nguyên liệu nhẹ đi" hoặc "bớt phải chạy đi lấy".

## 2. Hai giai đoạn hoàn toàn khác nhau: ĐỌC và VIẾT

Khi phục vụ một câu hỏi, máy làm hai việc có tính chất trái ngược:

| | **Prefill (đọc câu hỏi)** | **Decode (viết câu trả lời)** |
|---|---|---|
| Làm gì | Nhai toàn bộ câu hỏi một lượt | Sinh từng chữ, lần lượt |
| Bị chặn bởi | **Sức tính toán** của GPU | **Băng thông bộ nhớ** |
| Tốc độ đo được | ~1.400 chữ/giây | ~600 chữ/giây (chia cho mọi người) |
| Ví von | Đọc lướt cả cuốn sách | Viết tay từng chữ |

Hiểu sự khác biệt này là chìa khóa: hai giai đoạn nghẽn ở **hai chỗ khác nhau**,
nên thuốc chữa cũng khác nhau. Rất nhiều nhầm lẫn trong dự án đến từ việc đo
lẫn lộn hai thứ.

## 3. Nén mô hình (quantization) — giảm cân cho nguyên liệu

Mỗi trọng số vốn được ghi bằng **16 bit** — như viết "3,14159265". Nén xuống
**4 bit** là viết "3,14": mất chút chính xác, nhưng **file nhẹ đi 4 lần**, nên
mỗi chuyến khuân nhẹ đi 4 lần → viết chữ nhanh hơn nhiều.

Cái giá: nén ẩu thì mô hình "lú" đi. Cả dự án này xoay quanh câu hỏi:
**nén được sâu tới đâu mà chất lượng không sứt mẻ?**

Có nhiều *cách* nén, cùng 4 bit nhưng khác nhau xa về tốc độ:
- **GGUF** — định dạng của llama.cpp, chạy được mọi nơi (kể cả máy yếu, CPU),
  nhưng đóng gói phức tạp nên lúc dùng phải "bóc hộp" tốn công.
- **W4A16 / GPTQ / AWQ** — đóng gói sẵn theo đúng hình dạng mà GPU cần, gần
  như bốc là dùng được ngay. Nhanh gấp đôi GGUF trên cùng số bit.

## 4. Kernel và Marlin — "công thức thao tác" của GPU

GPU không tự biết làm gì; nó chạy các chương trình con gọi là **kernel**. Cùng
một phép nhân, kernel viết khéo hay vụng chênh nhau nhiều lần tốc độ.

**Marlin** là kernel chuyên trị "trọng số 4-bit nhân dữ liệu 16-bit" — nhanh
nhờ ba mánh: xếp sẵn nguyên liệu đúng khuôn từ lúc nạp, vừa nhân vừa bóc hộp
song song, và chia việc sao cho đường ống bộ nhớ luôn đầy. Trong dự án này,
**mục tiêu ngầm của mọi việc là đưa được nhiều phần mô hình nhất vào tay Marlin.**

## 5. KV cache — "cuốn sổ tra cứu" của cuộc hội thoại

Khi viết chữ thứ 100, mô hình cần "nhớ" 99 chữ trước. Nó không tính lại từ đầu
mà lưu sẵn một bảng ghi chú gọi là **KV cache**. Hệ quả:
- Hội thoại càng dài, cuốn sổ càng dày → **mỗi chữ mới càng tốn công tra**.
- Cuốn sổ chiếm bộ nhớ GPU → **giới hạn số người phục vụ cùng lúc**.

Trong dự án, ta lưu sổ này ở định dạng nén (fp8) — nhẹ đi một nửa, miễn phí.

## 6. Prefix caching — mánh quan trọng nhất của dự án

Nếu **mọi người dùng đều bắt đầu bằng cùng một đoạn văn bản** (bộ quy tắc,
kỹ năng, tài liệu — ở đây là ~30.000 chữ), thì thay vì bắt GPU đọc lại 30.000
chữ đó cho từng người, ta **đọc một lần rồi dùng chung**.

Số đo thật của dự án: chờ chữ đầu tiên giảm từ **10,5 giây → 0,2 giây**, tỷ lệ
dùng lại 99,4%. Đây là lý do một card giá rẻ phục vụ được ~30 người.

⚠️ Nhưng nhớ: dùng chung giúp **khỏi đọc lại**, chứ không giúp **khỏi tra sổ**
lúc viết. Nền càng dài, tốc độ viết càng giảm (600 → 390 → 150 chữ/giây khi
nền là 2K → 30K → 128K chữ).

## 7. Kiến trúc lai (hybrid GDN) — vì sao mô hình này đặc biệt

Mô hình thường có 100% tầng "attention" (loại phải tra sổ đầy đủ). Qwen3.5 chỉ
có **25% tầng attention, 75% tầng GDN** — loại giữ một bản tóm tắt cố định thay
vì cuốn sổ dày lên mãi.

Hệ quả tốt: tốn ít bộ nhớ hơn, chịu được ngữ cảnh dài hơn (ta chạy được
128.000 chữ trên card 23GB). Hệ quả xấu: **quá mới**, nên phần mềm phục vụ
chưa hỗ trợ đầy đủ — và đó chính là nguồn gốc của phần lớn bug trong file 03.

## 8. Vài chỉ số bạn sẽ gặp liên tục

| Chỉ số | Nghĩa là gì | Vì sao quan trọng |
|---|---|---|
| **tok/s** | số chữ sinh ra mỗi giây | thước đo năng suất |
| **TTFT** | thời gian tới chữ đầu tiên | người dùng cảm nhận "nhanh/chậm" ở đây |
| **p95** | 95% trường hợp nhanh hơn mức này | cam kết dịch vụ đo bằng đuôi, không đo trung bình |
| **conc / concurrency** | số người dùng cùng lúc | càng cao mỗi người càng chậm |
| **ppl (perplexity)** | mô hình "bối rối" cỡ nào — **thấp là tốt** | thước đo chất lượng sau khi nén |

## 9. Ba quy tắc đo lường mà dự án phải trả giá mới học được

1. **Đo trên dữ liệu thật, không đo trên câu mẫu tự chế** — nhiều kết luận đã
   bị lật khi đem ra bộ dữ liệu thật.
2. **Cẩn thận với cache khi benchmark** — nếu vô tình gửi lại prompt cũ, máy
   trả lời "nhanh giả" vì đã nhớ sẵn. Đã có kết quả bị thổi phồng vì lỗi này.
3. **Khách đến rải rác khác hẳn khách đến cùng lúc** — đo kiểu "đổ cả rổ" cho
   số đẹp gấp ba lần thực tế.

➡️ Tiếp theo: [02-hanh-trinh-toi-uu.md](02-hanh-trinh-toi-uu.md)
