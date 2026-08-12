# Quy tắc làm việc (user đặt, bắt buộc)

## Phê duyệt và phạm vi
1. Mọi thứ chạy trên GPU: đưa plan → user duyệt → mới chạy. Cấm tự làm.
2. Chỉ 1 GPU chính: notebook A = server `colab-mcp`. Notebook B (`colab-mcp-2`) và
   C (`colab-mcp-3`): không kết nối/dựng/chạy cho đến khi user ra lệnh đích danh.
3. Không subagent theo mặc định. Khi user cho phép: tối đa 2 đồng thời (Sonnet),
   việc dễ/cơ khí dùng Haiku 4.5.

## Kỷ luật khoa học
4. Cấm tự tuyên bố "hết đường" — phải tranh luận attack/defense (cần duyệt vì là
   subagent) hoặc trình bằng chứng đo được. Debate từng lật ngược "prefill wall" sai.
5. Số đo hơn suy luận — các suy luận nội suy đã sai nhiều lần. Đo trước khi kết luận.
6. Kết quả nào cũng vào `STATUS.md` + commit. Kết quả chỉ nằm trên runtime = sẽ mất.
6b. `TRANG-THAI.md` (root): Claude TỰ ĐỘNG cập nhật khi trạng thái thay đổi, không hỏi
    user; giữ ≤300 dòng, chi tiết cũ dồn sang `STATUS.md`.
6c. Mỗi quy trình chốt kết quả: cập nhật report HTML (Claude artifact, file
    `bao-cao-l4.html` trong scratchpad, URL artifact trong TRANG-THAI.md) và kiểm tra
    hiển thị đúng/đẹp (font Việt, light/dark, không tràn ngang).

## Vận hành
7. GPU đã duyệt chạy thì không ngồi chơi; bị chặn/lỗi báo ngay, không im lặng.
8. Báo cáo tiếng Việt, hướng người mới (user là beginner CUDA/LLM), tuần tự, logic.
9. Notebook giữ đúng 5 cell chuẩn (WHOAMI / BOOTSTRAP / SERVE / TASK / LOG). Việc mới
   ghi đè cell TASK — cấm tích lũy cell.
10. `run_code_cell` là blocking 30 phút — gom lệnh dài vào 1 cell và chờ trong lượt;
    không sleep/poll, không kết thúc lượt để đợi.

## An toàn
11. KHÔNG ghi HF token vào file/commit/log; nhắc user revoke sau chiến dịch.
12. Bản nháp `upstream/` không nộp ra ngoài khi chưa có user duyệt.
13. Commit/push khi user yêu cầu hoặc khi chốt kết quả; message mô tả đúng thực tế đo.
