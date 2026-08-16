# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi — KHÔNG cần hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn
sang `STATUS.md`.

Cập nhật: 2026-08-16.

## Trạng thái hiện tại

- **Repo đã refactor xong** (khuôn transformers, user chốt): `models/qwen3_5/`
  (engine/vllm + 19 patches, load/{gguf,quantize,pytorch_tensor}, hardware/l4,
  utils) + `sdk/ loading/ logging/ utils/ bench/ tests/`. **Registry đệ quy**
  (`python register.py --flat`; folder không có register.py = vô hình).
  **`run.sh` = 1 lệnh kiểu vLLM**; notebook A = ĐÚNG 1 CELL, cần gì thêm lệnh.
  Đã kiểm sống trên Colab: fresh 370s / ấm 140s, smoke đúng.
- **Vận hành**: chỉ notebook A (server `colab-mcp`); B/C chờ lệnh đích danh;
  không subagent trừ khi user cho phép.
- **Config production 9B** (đo 2026-08-14): mml 65536, mnbt 1088,
  **gpu_util 0.97** (KV 560.380 = +38,5% vs 0.85), điểm vận hành **12 phiên**
  (358,1 tasks/hr warm / 308,4 cold; 16 phiên 330,7). 1.0 chết khi khởi động.
- **Config production 27B**: mml 8192, mnbt 512, seqs 8, graphs [1,2,4,8],
  util 0.97 — decode 15,8 tok/s, ppl 4,1484 (hơn 9B champion 12,3%), 1-2 user.
- **Spec decoding ngram: OFF mặc định trên L4** (đo 2 model × 2 mức tải:
  9B −36% tasks/hr @8 phiên; 27B KV −28%, prefix hit sập). Profile `-spec`
  vẫn còn trong run.sh cho GPU lớn.
- Chiến dịch 9B ĐÓNG SẠCH (Q2c/Q3/Q4/R2b/P7/cổng đồng thời — chi tiết STATUS.md).

## Nghiên cứu KV-transfer (lệnh user 2026-08-14, đang chạy)

- Paper 2608.03893: mapper tự cài ở `models/qwen3_5/kv_transfer/` (không có code
  chính thức). **E0 phán quyết: context sống ở CẢ GDN state lẫn KV** (needle
  10/10 → 0/10 khi xóa một trong hai) → Phase B GDN-mapping bắt buộc.
- **E1 XONG (2026-08-15): COPY NGUYÊN cache 4B→9B giữ 100% needle (12/12,
  NLL 0,043 vs self 0,011, transplant 0,08s); ridge mapper 0/12 (R² heldout
  0,73/0,60 — thiếu là chết); no_ctx sàn 0/12.** TTFT 1,5K: ×1,27; trần ×2 @30K.
  Kết luận: cặp Qwen3.5 4B/9B cache-compatible THÔ, không cần mapper; mapper chỉ
  còn cho cặp lệch shape (27B). R² không phải proxy retention. Chi tiết STATUS.md.
- Identity gate cứu 1 vòng GPU (bắt bug λ GDN); calib mất 1 lần do runtime
  recycle (đã thu lại); chuẩn phóng nền mới: subprocess.Popen thay !nohup.
- **E2+E3 XONG (2026-08-15)**: copy giữ QA 9/9 tới 16K và 2/2 @30K; retention
  "hiểu" giảm đơn điệu 74%@2K → 53%@30K; **decode parity 11,8=11,8 tok/s**
  (mọi số decode 9B thuần giữ nguyên cho cross); 4B prefill vLLM 4771-5514
  tok/s → cascade TTFT 30K ×1,66. Chi tiết STATUS mục E1/E2/E3.
- **E3B: đồng trú 2 vLLM server trên 1 L4 naive = BẤT KHẢ THI** (0.27.1 tính
  non-torch bằng NVML, cộng VRAM process khác vào mình — 4 ràng buộc ghi
  STATUS). Lối thoát chưa thử: `--kv-cache-memory-bytes`. EngineArgs có
  `kv_transfer_config` = khung KVConnector cho Phase C.
- **E4 XONG (2026-08-15)**: 27B chạy được transformers/bnb-4bit trên L4 (mới).
  Số chốt: 4B↔9B CCA attention 0,98 (copy giữ ngôi, thiếu hụt @30K khoanh vùng
  GDN sâu); **x→27B attention CCA 0,93-0,97 = GO**, GDN giữa CCA 0,27 = tường
  tuyến tính, GDN sâu đuôi nặng (sv_max 110); concat-ridge thua ridge 1-lớp ở
  N hiện tại. **Kiến trúc chốt: mapper per-layer + functional loss (E5)** —
  chi tiết STATUS mục E4. Target user chốt: 27B, không bỏ.
- **E3C XONG: đồng trú 2 server/1 L4 THÀNH CÔNG** (combo: util thấp 0,35 +
  --kv-cache-memory-bytes + eager). Giá: 9B KV −63% (208K token, ~4 phiên),
  4B prefill đồng trú 3406 tok/s @30K (71% solo) → TTFT chỉ còn ×1,15-1,2.
  1-GPU cascade = khả thi cơ học, đáng giá khi ít phiên + nhiều cold-miss;
  giá trị lớn ở 2-GPU. Chi tiết STATUS mục E3C.
- **E5→E6→E7→E6b XONG (2026-08-15, ngày thí nghiệm dài nhất dự án)** — số
  chốt trong STATUS. Tóm tắt: (1) 9B-copy trên BFCL: NLL parity 99% nhưng
  greedy hit 6/20 — vết nứt biên-mỏng; E6b loại nhiễu spill, phát hiện
  **suffix re-prefill phản tác dụng** (luật nhất quán nội tại của cache);
  **E6c chốt: bnb gánh nửa vết nứt (bf16 copy 9/20 vs bnb 4-6/20), nửa còn
  lại là information bottleneck thật của cache 4B ở biên mỏng** — copy NLL
  thậm chí TỐT hơn self (2,43 vs 2,64) mà greedy vẫn thua: error-placement
  lần 4. Scope copy an toàn: chat/QA/RAG; function-calling cần polisher
  (đo lại trên W4A16 Marlin thật trong Phase C). (2) 27B-mapper v2 hội tụ trong
  miền (KL 0,63) nhưng 0/20 ngoài miền — v3 cần miền train đa dạng. (3) Ma
  trận 8 cặp: **pairability = CCA-GDN ≥ 0,9**; attention thẳng hàng toàn họ;
  {0.8B,2B} GDN lạc hệ; 4B→27B là cặp học sáng nhất (0,785).
- Deep-innovation đề xuất (từ deep-research doc + số của ta): weight-derived
  conjugation GDN (#1), activation-splicing (#2), pairability score (#3 — đã
  có data), importance-weighted loss (#4), compatibility-finetuning 0.8/2B
  (#5), retention-length law (#6). **User chọn #5 (2026-08-15): E8 ĐANG CHẠY**.
- **E8 ĐÓNG TRỌN 3 PHIÊN BẢN (2026-08-15, compatibility LoRA 2B→9B — #5,
  phương án (c) kích hoạt)**: gate trần thông tin SÁNG (2B self needle
  5/5@800 + 5/5@2000 — cache 2B CÓ đủ thông tin) nhưng cả 3 đòn thua sạch:
  v1 (LoRA r=16 chỉ-GDN, KL-functional, 300 bước) 0/5×6 mốc; v2 (r=64 mọi
  linear, state-alignment) 13s/bước kill; v3 (đúng bài Unsloth: student bf16
  + kernel fla/causal-conv1d fast path BẬT) nMSE học xong thang đo 10,5→1,06
  rồi KẸT ~0,96 (chiếm ~4% cấu trúc state) — needle 0/5 @74/@149 → dừng sớm.
  **PHÁN QUYẾT: phương ngữ GDN nhóm nhỏ không sửa được bằng adapter nhẹ;
  4B là prefill-helper chính thức duy nhất của 9B.** Bài học Unsloth:
  FastLanguageModel không dùng được cho cache-loss (past=None khi training,
  trả processor đa phương thức); tinh túy giữ được = bf16 student (docs
  "KHÔNG QLoRA trên Qwen3.5" trùng E6c) + kernel Triton. Chi tiết STATUS E8.

## Hàng đợi (đã duyệt chuỗi 1→2→3 ngày 2026-08-14)

1. ✅ Spec decoding — đóng bằng số đo.
1b. ✅ Util sweep (lệnh user) — mặc định mới 0.97, đỉnh 12 phiên.
2. **TIẾP: lấp vùng trắng 4B rồi 2B** — profile `serve 4b`/`serve 2b` + bộ bench
   chuẩn (sanity → ppl gate → decode/prefill → agent-loop) → tier "throughput
   giá rẻ" của ma trận. RedHatAI có frame w4a16 cho 4B.
3. Soak test 3-4 giờ chạy nền (rò bộ nhớ? TTFT trôi?) — đặt cuối ngày.
4. Đóng gói `serve 9b-prefill` (fp8 specialist, số đã đo) + cập nhật HTML report.
5. (Hoãn) P5 ablation nguồn graft; P6 converter toàn-model; sửa harness 3 thang token.

## Nợ dài hạn / quyết định chờ user

- Chưa đo qua mạng thật (mọi số qua localhost); chưa soak nhiều giờ (mục 3).
- Nộp bản nháp `upstream/` ra ngoài? — Revoke HF token sau chiến dịch — Mở B/C khi nào.
