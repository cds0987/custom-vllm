# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi — KHÔNG cần hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn
sang `STATUS.md`.

Cập nhật: 2026-08-24.

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

- **E6 v3 ĐÓNG (2026-08-24, CE-gold mapper 4B→27B trên miền thật — user chốt
  "KL chưa đủ")**: loss CE(gold)+0,3KL+dense, data BFCL/ifstruct(pseudo-gold
  +validator)/ParseBench-table/needle (585 train/50 val/30 test niêm phong),
  3 chốt chặn (best-by-val, stale-3, CE_FLOOR 0,2). **CE train 8,7→0,008
  (thuộc lòng) nhưng VAL 0 TOÀN TUYẾN 4 mốc; test niêm phong BFCL: self
  20/20 | mapped 0/20** (needle2k vô hiệu — bug TRAIN_MAX cắt câu hỏi).
  PHÁN QUYẾT lần 3 qua 3 hàm loss: mapper 35M không tổng quát hóa được
  cross-shape — vấn đề ở LỚP HÀM. Trận runtime mới: 7 bug (zombie PID,
  datasets parser, cache dict 5.15, copy_ in-place phá autograd, seed-skip
  treo, hook OOM, cell kill nhầm) — hạ tầng .last+resume+retry sống sót
  thật. → PHÁN QUYẾT NÀY BỊ LẬT bởi v3.1 (dưới).
- **E6 v3.1 ĐỘT PHÁ (2026-08-24): mapper 4B→27B SỐNG — BFCL niêm phong
  MAPPED 16/20** (self 20/20, 4B-self 11/20, v3.0 4/20, init 0/20). Nguyên
  nhân v3.0 chết: CONV_WARM bỏ 4 token đầu GOLD khỏi loss — token quyết
  định không bao giờ được dạy (user truy ra bằng train-check 1/30 + nghi
  "loss CE chưa đúng"). Fix: cache cắt T-5 + warm conv bằng 5 token cuối
  prompt + CE trọn gold, token đầu ×3. Val needle 8/10 @449 (truyền NỘI
  DUNG qua tường GDN, cache chưa gặp); train-check bfcl 7/15 needle 8/9.
  **Cascade 4B→27B vượt thanh kinh tế — có giá trị sản phẩm thật.**
  ĐÃ UPLOAD HF (quy tắc 6d): `gunnybd01/qwen35-kv-mapper-4b-27b` (private) —
  mapper v31 best+last, 3 json kết quả, pseudo_gold, data.
- **E6 v3.2 XONG (2026-08-25, scale-up user duyệt)**: data ×2,5 + 2000 bước
  → **BFCL niêm phong MAPPED 17/20** (self 20, 4B-self 11); val needle
  15/15 kể cả cỡ 1500; train-check 11/11+12/12 (học thật). **needle@2K
  niêm phong: 1/10 — VÁCH ĐÁ độ dài giữa 1500-2000** (retention-length law:
  train ≤950, GDN state ngoài phân phối ở 2K). ifstruct/pbtable vẫn 0 (nợ
  mổ output). Hạ tầng chống chịu hoàn chỉnh (skip-OOM tự học + gstep bền +
  val ngưỡng). Scope: fn-calling + retrieval ≤1500 tok = dùng được. NỢ
  UPLOAD v32 lên HF — Colab Secrets HF_TOKEN CHƯA SET (user cần thêm).
  Chi tiết STATUS mục E6 v3.2.
- **E6 v3.3 CODE SẴN SÀNG (2026-08-24, user duyệt hướng: tốc độ + ctx dài +
  chính xác)**: (tốc độ) bỏ 2 deepcopy ~600MB/bước (`clone_cache_struct`),
  Phase B1 tiền tính teacher 1 lần/item (top-64 logp + dense caps 4 layer
  fp16 ra đĩa) → vòng train hết teacher feed-forward, aux chỉ tính khi
  λ>0; (ctx dài) checkpoint `map_attn` (mặc định BẬT), cờ `--gdn-bf16`
  thử nghiệm, needle curriculum train 700→2000 + val bucket @2000;
  (chính xác) BFCL +parallel/multiple (655 bfcl train), trọng số token-
  xương ×2 cho ifstruct/pbtable, val DUMP text sinh ra (mổ được 0 điểm).
  Dry-data pass local: 1330 train / 70 val / test niêm phong giữ nguyên.
  Sanity ctx-2000: 4,05 s/bước, peak 20,32 GiB → GO; `--gdn-bf16` loại.
- **E6 v3.3 XONG (2026-08-25): KỶ LỤC KÉP NIÊM PHONG — BFCL MAPPED 18/20
  (self 20; v3.2: 17) + needle@2K MAPPED 10/10 TUYỆT ĐỐI (v3.2: 1/10) —
  VÁCH ĐÁ ĐỘ DÀI SỤP.** Nguyên nhân thắng: needle curriculum train tới
  2000 (vách đá là artifact phân phối train, không phải giới hạn GDN);
  tốc độ ~2,2 s/bước (nhanh 2× v3.2 nhờ bỏ deepcopy + teacher tiền tính
  B1); 0 retry/0 OOM toàn run — lần đầu E-series đi hết không ngã. Val
  curve 12→40, CE_FLOOR dừng sớm @1750 đúng kỷ luật. ifstruct 1/15 lần
  đầu có điểm; mổ dumps: ifstruct/pbtable chết chung bệnh repetition
  collapse sinh dài (>~30 tok) + nghi trần teacher pseudo-gold — thuốc
  v3.4 ở decode-time. Scope cascade 4B→27B: fn-calling 90% trần +
  retrieval ≤2000 tuyệt đối. Chi tiết STATUS mục E6 v3.3.
  ~~NỢ UPLOAD KHẨN (6d)~~ → **ĐÃ THÀNH HỌC PHÍ LẦN 2 (2026-08-25):
  runtime recycle NUỐT SẠCH mapper_v33.pt (18/20+10/10) + mapper_v32.pt
  — 5 lần nhắc Colab Secrets HF_TOKEN không được thêm. Số liệu/code/công
  thức còn nguyên trong git; tái tạo = ~10h GPU (xác định). TỪ GIỜ:
  KHÔNG phóng train dài khi chưa có đường upload sống (điều kiện cứng
  trước v3.4 full).**
- **E6 v3.4-long XONG (2026-08-25): TÁI LẬP 18/20 BFCL + needle NIÊM
  PHONG 15/15 TUYỆT ĐỐI (10@2K + 5@4K chưa từng gặp).** Ladder phán: L4
  trần ở 4096 (8K/16K OOM phần cứng — 16K cần A100); tpl-check 8/8 đúng
  từng bit → template-XƯƠNG thay hẳn teacher prefill: ~1,4 s/bước (nhanh
  40% hơn v3.3 dù data dài ×2). Val needle 23/23 mọi bucket từ mốc 1249;
  CE_FLOOR dừng @2016. Tái lập 18/20 sau khi MẤT SẠCH checkpoint = công
  thức thật. **AUTO-UPLOAD HF mỗi mốc val chạy thật** — mapper_v34 (best,
  .last), results, pseudo_gold, data đều đã ở `gunnybd01/qwen35-kv-mapper-
  4b-27b/v34/` ngay trong phiên. Token flow mới: .env root repo (user
  chốt), cell launch tự dựng lại sau recycle. ifstruct/pbtable vẫn 0
  (bệnh sinh-dài decode-time — v3.5). Chi tiết STATUS mục E6 v3.4.
  Ngã rẽ kế: Phase C (KVConnector vLLM — template-xương tái dùng được)
  vs v3.5 (decode-time cho sinh dài) vs A100 cho 8K/16K.
- **E6 v3.5 XONG (2026-08-25, user chốt "C, v3.5")**: mổ sinh-dài bằng 4
  điều kiện → **TRẦN TEACHER: 27B-self cũng chỉ 1/15 + 2/10** dưới cùng
  giao thức — suite ifstruct/pbtable là nợ của ĐỀ, không phải mapper;
  rep-penalty 1.3 phản tác dụng (đè chết token-xương của output lặp cấu
  trúc hợp lệ). Hành động: loại 2 suite khỏi thang chính thức; scope
  cascade giữ nguyên (fn-calling 90% + retrieval ≤4K tuyệt đối). Chi
  tiết STATUS mục E6 v3.5. **Phase C design đã chốt**
  (docs/phase-c-design.md): 2 vLLM + LMCache MP + vá key-namespace;
  **C2a XONG (2026-08-25): 3/3 tiền đề PASS** — (1) block size 9B = 4B
  = **1056** (cùng dòng log "attention page >= mamba page"); (2) cả hai
  boot sạch flags production (9B 471s fresh / 4B 311s); (3)
  **LMCACHE_EXT_OK 0.5.4** — connector external import được, không rơi
  vào fallback builtin (vốn crash trên model lai). Kiến trúc 1-GPU cho
  C2b: TUẦN TỰ qua L2 POSIX (4B producer ghi đĩa → stop → 9B consumer
  đọc) — né hẳn bài đồng trú E3B.
- **C2b VERDICT INTERIM (2026-08-26, 7 lượt phân xử C2b→C2b-7, chi tiết
  bảng trong STATUS mục PHASE C; kết quả HF c2b/..c2b7/)**:
  (1) **Cơ chế vận chuyển HOÀN CHỈNH** — vá 1 dòng key lmcache
  (`qwen35-shared`), hit mọi độ dài, **TTFT 30K 11-24s → ~1s (×12-16)**;
  (2) đã chốt bằng đo: fp8-scale là tầng lỗi thật (bf16 KV bắt buộc),
  block-align cần để hit; ĐÃ LOẠI: suffix-re-prefill (rem=2),
  champion-graft (stock giống hệt), producer-W4A16 (bf16 giống hệt);
  (3) **bất biến qua MỌI biến thể: cross lấy đúng 4-6 chữ số đầu rồi
  degenerate** → giả thuyết tầng lỗi thật (chưa kiểm): **trang GDN-state
  KHÔNG được truyền/áp — chỉ attention KV sang** (khớp E0: thiếu 1 trong
  2 là chết; E7: attention thẳng hàng → token đầu vẫn chạy). Chẩn đoán
  kế: instrument key nhóm object GDN trong lmcache. 6 bài học hạ tầng
  ghi STATUS (cùng-torch CUDA-IPC, port 8080, kill theo chủ cổng, pkill
  tự khớp → '[e]', 2-writer null-bytes, L1 pinned háo hức).
- **LAB-CHECK (2026-08-26, user chỉ đạo "kiểm ngoài vLLM trước"; HF
  c2b_lab/)**: 4B→9B copy-NGUYÊN transformers bf16, protocol E1, trên
  ĐÚNG 4 prompt aligned đang fail ở vLLM → **self 4/4 | copy 3/4 mã
  TRỌN VẸN** (439814✓ 025150✓ 071412✓; ca 30K-2 ra '9346666' — đúng
  chữ ký degeneration). Hai tầng sự thật: (1) **vLLM làm MẤT THÊM
  thật** — lab 3/4 mã trọn vs vLLM 1/4 chỉ 4 số đầu, CÙNG đề → nghi
  án trang GDN rơi trong kho ĐỨNG VỮNG, giờ có mốc đối chứng định
  lượng; (2) biên phương pháp MỎNG hơn E1 trên đề filler tổng hợp 30K
  (E1 12/12 là FineWeb thật; continuation sau mã cũng lú nhẹ) — fail
  vLLM một phần là khuếch đại biên mỏng sẵn.
- **KHÁM KHO GDN (2026-08-26, đầu dò OBJGRP-PROBE tiêm vào
  _run_object_group_transfer_plan)**: **GIẢ THUYẾT TRANG-GDN-RƠI BỊ
  BÁC** — plan chạy ĐỐI XỨNG cho cả 2 nhóm object (group 0 attention +
  group 1 mamba/GDN: 76/76 lần, cặp đôi từng timestamp, cả store lẫn
  retrieve; 4 lần RETRIEVE = đủ 4 prompt). Kho giao đủ hai nửa vở —
  đo-hơn-suy-luận thêm một lần (kết quả âm quý). Nghi phạm còn lại
  DUY NHẤT sau khi đối chiếu lab-check: **trọng số consumer W4A16**
  — lab 3/4 dùng 9B bf16-weights; mọi lượt vLLM đều consumer W4A16
  (champion lẫn stock); E6c từng đo: student lượng tử hóa mất nửa biên.
- **C2b-8 + VERDICT CUỐI PHASE C (2026-08-26, HF c2b8/)**: consumer
  bf16-weights trong vLLM (8K-only, 30K không vừa L4 bf16) → cross VẪN
  1/2, ca 0 cụt '4398' giống hệt W4A16 → giả thuyết W4A16-consumer CŨNG
  bị bác. **Kết luận: không còn "một con bug" — là ĐỊNH LUẬT BIÊN MỎNG**:
  cross-cache decode đứng sát mép vực số học, nhiễu nhỏ nào (kernel GDN
  vLLM≠fla, roundtrip trang, lượng tử hóa) cũng lật ca cận biên (lab-fla
  cùng ca ra mã trọn; junk sau mã có Ở CẢ lab). Transport stack đã ĐÚNG
  (2 nhóm object giao đủ 76/76; bf16 KV + block-align + L1 + cùng-torch
  = điều kiện cần, đã vá; TTFT ×12-16 thật). Hệ quả: exact-retrieval
  serving cần POLISHER gia cố biên (công nghệ mapper sẵn có); scope ngữ
  nghĩa chat/QA/RAG (E2-E3) chưa đo trên serving = bài đo kế; 4B→27B
  batch không ảnh hưởng. Chi tiết STATUS mục PHASE C VERDICT CUỐI.
  GPU trống, không job nền — chờ user định hướng.

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
