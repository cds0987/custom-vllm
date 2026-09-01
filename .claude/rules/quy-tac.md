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
6d. (User chốt 2026-08-24) TRỪ KHI user nói khác: MỌI sản phẩm làm ra được — checkpoint,
    mapper/LoRA (KỂ CẢ kết quả âm — vẫn cần cho kiểm chứng lại), pseudo-gold, cache
    spill đắt tiền, data đã dựng — đều phải SAVE LÊN HUGGINGFACE trong CÙNG PHIÊN
    trước khi runtime có cơ hội recycle. Học phí: mapper E6v3 `.last` mất theo
    recycle → phải retrain ~2h chỉ để chạy một bài falsification. Không có token
    write tại thời điểm chốt → HỎI USER NGAY, không để sang hôm sau.

## Vận hành
7. GPU đã duyệt chạy thì không ngồi chơi; bị chặn/lỗi báo ngay, không im lặng.
8. Báo cáo tiếng Việt, hướng người mới (user là beginner CUDA/LLM), tuần tự, logic.
9. Notebook giữ ĐÚNG 1 CELL duy nhất chạy `bash run.sh <lệnh>` (kiểu vLLM). Cần gì
   THÊM LỆNH vào cell đó (mọi lệnh idempotent) — cấm thêm cell. `run.sh help` liệt kê.
10. Task LỚN trên Colab (serve, bench dài, download, soak): LUÔN chạy `nohup ... > log &`
    trong cell để cell trả về ngay — KHÔNG chờ đồng bộ trong lượt (user chốt 2026-08-14).
    Trong lúc Colab chạy nền thì làm việc local (research, code, docs). Kiểm tra sau
    bằng `bash run.sh status` / tail log. Task nhỏ (<1-2 phút) chạy thẳng được.

## Phạm vi tune-eval
14. (User chốt 2026-09-01, sau khi bắt lỗi thiếu sót lần 1) Khi user nói "tập trung
    vào bộ X, Y" (thu hẹp phạm vi cải tiến): PHẢI áp dụng NGAY vào cả train
    (`--drop-kinds`) LẪN vòng val/sanity — không chỉ ghi nhận rồi qua việc khác mà
    quên. Ghi rõ "mục tiêu hiện tại" thành một dòng riêng, dễ thấy, ở đầu mục
    "Trạng thái hiện tại" trong `TRANG-THAI.md` — mỗi khi mục tiêu đổi, sửa dòng đó
    NGAY LẬP TỨC (cùng lượt), không đợi đến khi tổng kết. Các bộ bị loại khỏi vòng
    tune-eval vẫn giữ nguyên số liệu cũ (không cần chạy lại) cho tới khi có lượt
    train mới đụng tới chúng.

## An toàn
11. (User sửa 2026-08-25) Token HF là account FREE thí nghiệm, user chấp nhận
    dùng trực tiếp: sống ở `.env` root repo (gitignored) + cell notebook —
    mọi môi trường đọc từ đó. Ranh giới còn lại TUYỆT ĐỐI: không commit/push
    token lên git (GitHub public + HF tự revoke token lộ = mất đường upload);
    kiểm `git check-ignore .env` trước khi đụng.
11b. (Học phí 2 lần — 2026-08-24 mất mapper E6v3 .last, 2026-08-25 mất
    mapper v3.3 kỷ lục + v3.2) KHÔNG phóng train dài khi đường upload chưa
    sống: script train phải tự upload best/.last mỗi mốc val (--hf-repo
    trong e6v3_ce.py), và cell launch phải dựng lại `.env` trên runtime
    (recycle xóa /content).
12. Bản nháp `upstream/` không nộp ra ngoài khi chưa có user duyệt.
13. Commit/push khi user yêu cầu hoặc khi chốt kết quả; message mô tả đúng thực tế đo.
