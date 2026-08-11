# Đọc từ đâu — bản đồ tài liệu dự án

Bộ tài liệu này viết cho người **chưa biết gì về CUDA hay mô hình ngôn ngữ**.
Đọc theo đúng thứ tự dưới đây, mỗi file khoảng 10-15 phút.

| # | File | Trả lời câu hỏi gì |
|---|------|--------------------|
| 1 | [01-kien-thuc-nen.md](01-kien-thuc-nen.md) | Máy đang làm gì khi "chạy AI"? Vì sao nó chậm? Các khái niệm nền |
| 2 | [02-hanh-trinh-toi-uu.md](02-hanh-trinh-toi-uu.md) | Chúng ta đã làm gì, theo thứ tự nào, và **vì sao bước sau nối tiếp bước trước** |
| 3 | [03-bug-va-cach-sua.md](03-bug-va-cach-sua.md) | Những con bug đã gặp, giải thích bằng ngôn ngữ thường |
| 4 | [04-ket-qua-va-cach-dung.md](04-ket-qua-va-cach-dung.md) | Kết quả cuối, cấu hình khuyến nghị, cách chạy lại |
| 5 | [05-hoc-cuda-vllm-serving.md](05-hoc-cuda-vllm-serving.md) | **Học nghề**: CUDA → vLLM → serving, 6 tầng, kèm lộ trình tự học |
| 6 | [agent-loop-playbook.md](agent-loop-playbook.md) | Cẩm nang 20 kịch bản cho hệ agent gọi tool (nâng cao) |
| 7 | [routing-research.md](routing-research.md) | Thuật toán định tuyến của vLLM/Dynamo và 5 đề xuất cải tiến (nâng cao) |

Ngoài ra `STATUS.md` ở thư mục gốc là **sổ ghi chép kỹ thuật gốc** — chi tiết,
khô khan, dành cho người vận hành; bộ tài liệu này là bản kể lại dễ hiểu.

---

## Tóm tắt 2 phút — nếu bạn chỉ đọc một đoạn

**Bài toán:** chạy một mô hình AI 9 tỷ tham số (Qwen3.5-9B) trên **một card đồ
họa L4 giá rẻ**, phục vụ được càng nhiều người càng tốt, mà không làm mô hình
trả lời kém đi.

**Kết quả sau chiến dịch:**

| | Lúc bắt đầu | Bây giờ |
|---|---|---|
| Tốc độ (32 người dùng cùng lúc) | ~274 chữ/giây | **~390 chữ/giây** (đo trên dữ liệu thật) |
| Chất lượng | mốc chuẩn | **tốt hơn cả bản gốc chưa nén** |
| Chờ chữ đầu tiên | 10,5 giây | **0,2-0,6 giây** |
| Xử lý theo lô (chạy đêm) | chưa có | **~520 chữ/giây, ~31.500 việc/đêm** |
| Ngữ cảnh dài nhất | chưa kiểm | **128.000 chữ, đã kiểm đúng đắn** |

**Ba việc lớn nhất đã làm:**
1. **Đổi cách đóng gói mô hình** để card đọc nhanh hơn (nén 4-bit đúng cách).
2. **Dạy hệ thống nhớ phần dùng chung** (bộ nhớ đệm tiền tố) — nhờ vậy chờ 0,2s
   thay vì 10,5s.
3. **Vá 20+ lỗi** trong thư viện phục vụ, vì mô hình này quá mới, phần mềm
   chưa theo kịp.

**Một câu để nhớ:** phần lớn thời gian không phải "làm cho máy tính nhanh hơn"
mà là **giảm số byte máy phải khuân qua lại**, và **tránh làm lại việc đã làm**.
