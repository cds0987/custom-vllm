# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi (chốt kết quả, xong việc, user đổi hướng) — KHÔNG cần hỏi user.
Giới hạn cứng: ≤300 dòng — chi tiết cũ dồn sang `STATUS.md`, ở đây chỉ giữ cái một
phiên mới cần biết chính xác và ngắn gọn.

Cập nhật: 2026-08-12.

## Trạng thái

- **2026-08-13: REFACTOR HOÀN THÀNH + test thật trên Colab PASS.** Cấu trúc mới
  (khuôn transformers, user chốt qua 5 vòng thảo luận): `models/qwen3_5/{engine/vllm
  (+19 patches), load (3 đường), hardware, utils}` + `sdk/ loading/ logging/ utils/
  bench/ tests/`. **Registry đệ quy**: `python register.py --flat` (mỗi folder có
  register.py; không có = vô hình). **`run.sh` = 1 lệnh kiểu vLLM**; notebook A giờ
  ĐÚNG 1 CELL: clone + `bash run.sh serve 9b` → server READY 370s (fresh runtime,
  KV 404.613 khớp lịch sử), chạy lại 140s, smoke completion trả lời đúng.
  9 file test local xanh (26+14+16+29+8+6 case + 3 suite graft/marlin).
  Commit: 54cb272 (git mv 54 file) + 9a8f0e3 (run.sh/registry/manifest/paths).
- **2026-08-14**: (1) Spec decoding ngram ĐÓNG bằng số đo — OFF mặc định trên L4
  (9B −36% tasks/hr @8 phiên; 27B decode +23% nhưng KV −28%/prefix hit sập).
  (2) Util sweep (lệnh user): MẶC ĐỊNH MỚI gpu_util=0.97 — KV +38,5% (560.380),
  điểm vận hành dịch 8→12 phiên (358,1 warm / 308,4 cold), 16 phiên hết lỗ
  (330,7 = ×1,74 số cũ); 0.98 y hệt 0.97; 1.0 chết khi khởi động (đo thật).
  Chi tiết: STATUS.md. Hàng đợi tiếp: 4B/2B, soak test, HTML report.

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
7. ✅ 27B PHASE 1 — Qwen3.5-27B LÊN SÓNG trên L4: frame apolo13x w4a16 18,6GB (GDN
   đã quantize sẵn, KHÔNG cần graft), decode 15,8 tok/s conc1 (sát trần băng thông).
   Config chốt: mml 8192, mnbt 512, max-num-seqs 8, graphs [1,2,4,8], util 0.97,
   fp8 KV, align. Chi tiết leo thang 5 bước: STATUS.md "CHIẾN DỊCH 27B PHASE 1".
8. ✅ 27B PHASE 2 — CỔNG CHẤT LƯỢNG PASS: ppl 4,1484 (99 đề, cùng bộ với 9B) —
   tốt hơn 9B champion 12,3%, frame cộng đồng không hỏng, KHÔNG cần graft.
   Còn nợ nhỏ: bench mini single-user cho 27B.
9. (Hoãn, chưa hủy) P5 ablation nguồn graft 9B; P6 converter toàn-model.

## Quyết định chờ user

- Nộp bản nháp `upstream/` ra ngoài? — Revoke HF token sau chiến dịch — Mở lại B/C khi nào
