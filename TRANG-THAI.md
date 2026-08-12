# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi (chốt kết quả, xong việc, user đổi hướng) — KHÔNG cần hỏi user.
Giới hạn cứng: ≤300 dòng — chi tiết cũ dồn sang `STATUS.md`, ở đây chỉ giữ cái một
phiên mới cần biết chính xác và ngắn gọn.

Cập nhật: 2026-08-12.

## Trạng thái

- **Lệnh hiện hành (2026-08-13)**: dứt điểm L4 + Qwen3.5-9B trên DUY NHẤT notebook A,
  không subagent (chỉ được dùng để discuss/phân việc), tự làm tự fix. Xong hẳn 9B thì
  transfer sang Qwen3.5-27B-GGUF. Làm với code/workflow hiện tại — REFACTOR CHƯA ĐƯỢC
  CONFIRM (structure đã trình, chờ). Commit code + docs + upstream/ theo từng quy trình.
- Tầm nhìn chốt 2026-08-12: architecture-centric adapter (Qwen3.5 làm trung tâm, tầng
  ngoài proxy/loading/logging, SOLID). Structure hoàn chỉnh đã trình trong chat.
- Champion v2 đã build + upload `gunnybd01/qwen35-9b-champion` nhưng CHƯA sanity-check
  (sanity.py crash vì lỗi spawn; bản fix đã viết, chưa chạy).
- Notebook A: env cài rồi, chưa có champion. B: nơi build champion. C: chỉ WHOAMI.
  Cả 3 có thể đã bị recycle — coi như bootstrap lại từ đầu.
- vLLM 0.27.1 (drift từ 0.26, 2026-08-12): R0 xác nhận không regression; patch IsHybrid
  và override_signature đã thành no-op upstream.

## Việc local chưa xong (không GPU, vẫn chờ duyệt commit/push)

- [ ] Refactor repo theo 4 trục adapter (git mv thuần → sửa path → verify)
- [ ] Commit thay đổi treo: setup_env.sh (uv), colab_bootstrap.sh (CHAMPION_REPO),
      env_snapshot.py (mới), .mcp.json, CLAUDE.md + .claude/
- [ ] Vòng tranh luận 2: auto-marlin garbage, lm_head (cần duyệt subagent)
- [ ] Cổng kiểm định đúng đắn ở chế độ ĐỒNG THỜI (audit top-5)
- [ ] Bản nháp upstream #21: colab-mcp hardcode Scratchpad
- [ ] docs/scheduler-optimization.md có thể dở dang

## Hàng đợi GPU (một hàng, tuần tự trên A — đã được duyệt chạy 2026-08-13)

1. ✅ Dọn A về 5 cell chuẩn (WHOAMI/BOOTSTRAP/SERVE/TASK/LOG) + bootstrap 5,9 phút
   + sanity champion PASS (36,3 tok/s conc1, output mạch lạc — bản HF nguyên vẹn)
2. ✅ Q2c — KẾT QUẢ: giả thuyết "+1,6× capacity" BỊ BÁC ở điểm vận hành 8 phiên
   (49152 kém 65536 5,2%); chỉ có ích ở 16 phiên (+9,4%, TTFT p95 −23%);
   40960 hỏng workload (nhu cầu thật ~42-43K). CHỐT: giữ mml=65536 cho 8 phiên.
   Chi tiết: STATUS.md TASK Q2c.
3. ✅ Q3 — KHÔNG bệnh lý: server dừng generate ≤3s khi client bỏ đi (không GPU waste);
   tail 30s vô hại (prefix sống sót dưới tải); burst −18,6% nhưng ổn định.
4. ✅ Q4 — chốt policy tràn context: summarize-stub (giữ 8/8 việc, −30% tasks/hr,
   TTFT p95 ×1,9); error mất trắng 0/8. Lưới an toàn, không phải chiến lược.
   + Bẫy harness 3 thang token (est từ / token thật / yêu cầu) — TODO sửa.
5. ✅ R2b — chunk=32 GIỐNG HỆT 64 trên 0.27.1; con số +7,6% khai tử; giữ mặc định.
6. ✅ P7 — CPU không nghẽn: GPU 100% phẳng @ conc32, vLLM ăn 8% một core. Đóng.
6b. ✅ Cổng kiểm định đồng thời — PASS: không garbage; phân kỳ 5/8 là batch
    non-invariance cố hữu của vLLM (conc-vs-conc lặp lại cũng 3/8 khớp).
7. **TIẾP THEO: transfer sang Qwen3.5-27B-GGUF** (lệnh user 2026-08-13). Việc đầu:
   khảo sát frame W4A16 cho 27B trên HF (RedHatAI có không? nếu không → phương án
   GGUF-only qua plugin, hoặc tự quantize frame — 27B bf16 54GB KHÔNG vừa L4).
8. (Hoãn, chưa hủy) P5 ablation nguồn graft; P6 converter toàn-model — cân nhắc gộp
   vào chiến dịch 27B vì 27B chính là bài test thật của P6.

## Quyết định chờ user

- Nộp bản nháp `upstream/` ra ngoài? — Revoke HF token sau chiến dịch — Mở lại B/C khi nào
