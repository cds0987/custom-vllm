# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi — KHÔNG cần hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn
sang `STATUS.md`.

Cập nhật: 2026-08-28.

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
- **E3B: đồng trú 2 vLLM server 1-L4 naive = BẤT KHẢ THI** (0.27.1 cộng VRAM
  process khác vào mình) — giải xong ở E3C bên dưới.
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
- Deep-innovation 6 đề xuất (chi tiết STATUS); user chọn #5
  compatibility-finetuning → E8 dưới đây.
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

- **E6 v3 ĐÓNG (2026-08-24)**: CE-gold loss, val 0 toàn tuyến, test niêm
  phong mapped 0/20 → nghi lớp hàm. BỊ LẬT bởi v3.1. Chi tiết STATUS.
- **E6 v3.1 ĐỘT PHÁ (2026-08-24): mapper 4B→27B SỐNG — BFCL 16/20**.
  Nguyên nhân v3.0 chết: CONV_WARM bỏ token GOLD đầu khỏi loss. Fix: cache
  cắt T-5 + warm conv 5 token cuối + CE trọn gold. Chi tiết STATUS.
- **E6 v3.2 XONG (2026-08-25)**: BFCL 17/20 nhưng needle@2K vách đá 1/10
  (retention-length law) — vách đá này ĐÃ SỤP ở v3.3 bên dưới. Chi tiết
  STATUS mục E6 v3.2.
- **E6 v3.3 CODE (2026-08-24)**: bỏ deepcopy, B1 tiền tính teacher,
  checkpoint map_attn, needle curriculum → chi tiết STATUS. `--gdn-bf16` loại.
- **E6 v3.3 XONG (2026-08-25): BFCL 18/20 + needle@2K 10/10 — VÁCH ĐÁ ĐỘ
  DÀI SỤP** (needle curriculum tới 2000: vách đá là artifact phân phối
  train). ~2,2 s/bước, 0 retry. ifstruct/pbtable: repetition
  collapse sinh dài (>~30 tok) + nghi trần teacher pseudo-gold — thuốc
  v3.4 ở decode-time. Scope cascade 4B→27B: fn-calling 90% trần +
  retrieval ≤2000 tuyệt đối. Chi tiết STATUS mục E6 v3.3.
  ~~NỢ UPLOAD KHẨN (6d)~~ → **ĐÃ THÀNH HỌC PHÍ LẦN 2 (2026-08-25):
  runtime recycle NUỐT SẠCH mapper_v33.pt (18/20+10/10) + mapper_v32.pt
  — 5 lần nhắc Colab Secrets HF_TOKEN không được thêm. Số liệu/code/công
  thức còn nguyên trong git; tái tạo = ~10h GPU (xác định). TỪ GIỜ:
  KHÔNG phóng train dài khi chưa có đường upload sống (điều kiện cứng
  trước v3.4 full).**
- **E6 v3.4-long XONG (2026-08-25): 18/20 BFCL + needle NIÊM PHONG 15/15**
  (10@2K + 5@4K). Ladder: L4 trần 4096; template-XƯƠNG thay teacher
  prefill (~1,4 s/bước). Auto-upload HF `v34/` mỗi mốc val. Token flow:
  .env root repo. Chi tiết STATUS mục E6 v3.4.
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
- **C2b-N XONG (2026-08-26, user đòi N lớn; HF c2bN/)**: 240 prompt @8K,
  10 wave tự động 2h không ngã → **self 240/240 = 100% | cross 137/240
  = 57,1% (CI ~51-63%) | tốc độ ×2,7 @8K**. Tỷ lệ thật của biên-mỏng:
  không phải vách đá, không phải gần-hoàn-hảo — ~57% ca sống qua nhiễu.
  Kinh tế: hybrid "thử cross, miss thì cold-prefill" đã lời TTFT ngay;
  polisher cần kéo 57→95%+ cho exact-retrieval thuần. GPU trống, không
  job nền — chờ user định hướng (polisher / hybrid-fallback / cascade
  4→27 batch / kernel-diff bước 1).
- **C2c sem XONG (2026-08-26, bước 1 user duyệt)**: scope ngữ nghĩa
  (QA paraphrase trên văn bản wikitext thật, chấm keyword) trên serving
  **self 54/60=90% | cross 33/60=55%** — KHÔNG miễn nhiễm, sát mức needle
  số (57%). Kết luận: định luật biên mỏng áp cho mọi decode đầu trên
  cache ngoại, không riêng bài trích nguyên văn. HF `c2c_sem/`.
- **Hạ tầng vá cùng đợt**: ghim `vllm==0.27.1` trong `setup_env.sh`
  (runtime mới từng kéo 0.28.0 không ghim = drift ngầm âm thầm — MỌI
  số Phase C trước đó đo trên 0.27.1, phải cảnh giác runtime mới);
  `run.sh serve` từng chết câm khi thiếu `/tmp/vllm_env.sh` (set -e +
  source fail) → vá `|| true`; **học phí token**: gõ tay HF_TOKEN vào
  Colab cell làm mất 1 ký tự → 401 hàng loạt — từ nay đọc token từ
  file + assert độ dài, không gõ tay secret; mọi `HfApi()` trong repo
  đã vá `token=` tường minh.
- **User chốt hướng mới (2026-08-26)**: copy-nguyên 4→9 "hên xui" →
  chuyển sang **train mapper functional-loss cho 4→9** (tái dùng
  `e6v3_ce.py`, chỉ đổi `--tgt-model`; theo E7 cặp 4→9 CCA-GDN≥0,9 dễ
  hơn 4→27 CCA 0,785 — kỳ vọng hội tụ nhanh). Đã dựng thêm `suite_gen.py`
  (4 họ đề rag/mid-info/reasoning-math/swe, đích user "vài ngàn prompt,
  target 80-90% như normal decode") + `c2suite.sh` (chiến dịch lớn,
  resume-được theo wave) — CHƯA phóng, chờ mapper Giai đoạn A xong.
  Kế hoạch 2 giai đoạn: A) sanity mapper 4→9 với data cũ (0 code mới,
  ~1-2h) → tín hiệu đi/không; B) nếu khả quan, tích hợp data 4 họ mới
  vào train loop rồi đo trên `c2suite.sh`.
- **Giai đoạn A XONG (2026-08-27): sanity 4→9 chạy trọn KHÔNG SỬA
  CODE, 0 lỗi ngay lần đầu** — 20 bước, ~1,7-1,8 s/bước ổn định, peak
  VRAM chỉ **8,76 GiB** (27B bnb-4bit sát trần 22GB — 4→9 rẻ hơn
  nhiều, còn dư địa tăng ctx/tốc độ). Vá 1 bug trước khi chạy:
  `hf_up()` ghim cứng `"v34/"` → thêm `--hf-prefix` (né đè lên mapper
  4→27B). Sanity chỉ đo tốc độ/VRAM, CHƯA đo hội tụ — chờ user duyệt
  train thật (vài trăm-nghìn bước, val curve) trước khi phóng.
- **Ladder 4→9 XONG (2026-08-27): 4096/8192/16384 đều KHÔNG OOM**
  (27B từng kẹt ở 4096) — peak 9,1/10,0/11,9 GiB, còn dư ~11GiB dưới
  trần L4 tại 16384. Khuyến nghị max-ctx=16384 cho train thật (khớp
  mục tiêu gốc "Unsloth 16K"). Đang chờ user duyệt phóng train.
- **Mở context 27B (2026-08-27)**: gradient checkpointing ĐÓNG hẳn (peak
  y hệt baseline — OOM do 1 lớp `repeat_kv`, không phải tích lũy).
  **CPU-offload thủ công THẮNG**: steady 12,81GiB (−4,85GiB), T=8192 OK
  peak 18,11GiB t=1,4s, 2 lần gọi liên tiếp OK; T=16384 vẫn OOM.
  `load_4bit_cpu_offload_io` (né accelerate dispatch). Chi tiết STATUS.
- **MAPPER 4B→9B TRAIN THẬT XONG (2026-08-27, max-ctx=16384)**: dừng
  sớm đúng CE_FLOOR ở bước 984/2600 — **BEST: BFCL 23/25 (92%) |
  needle 29/29 (100%) | score 54**. Nhỉnh hơn mapper 4→27B (BFCL
  18/20, needle 15/15) mà đạt trực tiếp ở ctx 16384 (27B chỉ tới
  4096 trên L4) — xác nhận đúng E7 (cặp 4→9 dễ hơn 4→27). ifstruct/
  pbtable vẫn ~0 (nợ của ĐỀ, không phải mapper — khớp phán quyết
  v3.5 cũ). Checkpoint + data trên HF `v49/`. Bước kế chờ chọn: (a)
  tích hợp vLLM serving Phase C3, (b) đo trên `suite_gen.py`
  (rag/mid/math/swe), (c) đóng gói kiểu cascade_427 cho 4→9.
- **BENCHMARK NGOÀI — baseline 27B THUẦN XONG (2026-08-27)**: bộ đề user
  chốt (bỏ CUDA/AIME/MATH-500, giữ math+reasoning): **BBH 98/182=53,8% |
  GSM8K 160/200=80% | MuSR 115/198=58,1%** (toàn bộ 756: 53,6%).
  **Chất lượng sinh: rác 0,0% cả 3 bộ, hallu ~0** — mốc đối chứng then
  chốt: khi chạy cross, mọi tỷ lệ rác > 0 đều quy được cho mapper. Sai
  chủ yếu là "sai nhưng sạch" (suy luận đúng dạng, chọn nhầm đáp án).
  Kết quả + phân tích trên HF `extbench_self/`. Code: `ext_bench.py`,
  `bench_analyze.py` (soi rác/hallu/cắt/sai-tính) + 2 bộ test không cần
  GPU (14/14, 9/9) — dựng sau khi 3 LẦN suýt báo cáo số liệu sai vì lỗi
  harness (thinking-model, ngân sách token, báo động giả).
- **MAPPER 4→27B TRAIN @ctx4096 + CROSS: PHÁN QUYẾT (2026-08-28)** —
  chi tiết STATUS mục "CROSS trên benchmark ngoài: KẾT QUẢ CUỐI".
  Train: warm-start v34, dừng sớm CE_FLOOR @1075, val 27→29→34→38→39.
  **Test NIÊM PHONG (miền ĐÃ train) tái lập kỷ lục: bfcl 18/20, needle@2K
  15/15, no_ctx 0/20.** Nhưng **benchmark NGOÀI sụp đổ**: BBH 6,0% (self
  53,8%) | GSM8K 0% (self 80%) | MuSR 1,5% (self 58,1%).
  **ĐỐI CHỨNG 4B-self là chìa khóa**: gsm8k 4B-self **81,5%** — CAO HƠN
  27B! Thông tin CÓ SẴN trong cache 4B nhưng qua mapper còn 0% → lỗi ở
  khâu DỊCH, không phải giới hạn model nguồn. Chất lượng sinh: rác
  0,2%→15,6%, không-ra-đáp-án 6,5%→55,1%. Phân loại 243 ca self-đúng→
  cross-sai: 22% SINH RÁC, 78% LẠC ĐỀ NHƯNG MẠCH LẠC (nguy hiểm hơn —
  hỏi boolean trả lời xác suất, hỏi án mạng trả lời về Beatles; hallu
  tên riêng musr 0,63/mẫu vs self 0,04).
  **KẾT LUẬN: mapper 35M chỉ học ánh xạ CHO MIỀN CỤ THỂ, KHÔNG học
  "cách dịch cache" tổng quát** (phán quyết lần 4 về lớp hàm, lần này
  có đối chứng nên loại trừ được giả thuyết "model nguồn yếu").
  Hàm ý: cascade 4→27B bán được TRONG miền đã train (fn-calling,
  retrieval ≤4K) nhưng KHÔNG phải "tăng tốc đa dụng". Muốn tổng quát:
  đa dạng hóa mạnh miền train, hoặc đổi lớp hàm mapper. HF
  `extbench_cross/` + `v427_4k/`. Vá 2 bug: dict-wrap 5.15 (e5._get),
  OOM do nạp 2 model cùng lúc → run_cross HAI PHA.
- **KIẾN TRÚC 2 LỚP (user chốt 2026-08-28: LoRA gắn vào 4B ép nó "đọc hộ"
  27B → merge vào 4B → mapper dịch): CỔNG VRAM MỞ, khe hẹp.** 7 lượt probe
  (`probe_joint_lora.py`, 42 cấu hình). Hai model cùng GPU = 16,16GiB, trống
  5,54. Thủ phạm: prefill 4B CÓ GRAD (+3,34GiB @T=256). **GC bị loại VỀ
  NGUYÊN TẮC** (transformers ép `use_cache=False`, mà forward 4B tồn tại
  chính để sinh cache — cơ chế khác hẳn lần đóng GC cho 27B). **TBPTT mở
  được cổng**. Bao đo chốt: **gold ≤48 khi ctx ≤1024, gold ≤16 khi ctx
  1536-2048, w=64**; tường cứng gold=64 với MỌI T (gold = số vị trí feed vào
  27B, mỗi vị trí giữ state GDN cho backward → đắt hơn ctx nhiều). Giá:
  TBPTT là xấp xỉ (LoRA chỉ học từ w vị trí cuối), gold gsm8k 150-250 token
  bị cắt còn 48. Trả lời user "sao hôm trước 8192 mà giờ 2048 OOM": 8192 là
  trần 1-model-thường-trú (27B một mình 12,81GiB); giờ 4B phải ở lại kèm
  autograd → 18,11+3,5 = 21,6GiB đã chạm trần trước mọi activation.
- **TRAIN JOINT ĐANG CHẠY (2026-08-28)**: `e9_joint.py` + `gen_data.py`
  (7768 train/330 val: gsm8k 2882 + bbh 2500 + musr 474 + suite 767 + bfcl
  655 + needle 380 + pbtable 110), ctx 2048, tbptt 64, warm-start v427_4k,
  2500 bước ~4h, auto-upload HF `joint_v1/` mỗi mốc val. **Rò rỉ 0/6898** vs
  580 mẫu niêm phong (đối chiếu chuỗi). Sanity: ce 1,497 | 5,76s/bước |
  peak 20,91GiB | 0 OOM. Vá 2 bug: `ifstruct` không có `gold` (do B0 sinh,
  joint không có B0), và **cắt TRÁI khi tokenize** (mọi prompt bộ này đặt
  câu hỏi Ở CUỐI — cắt phải mặc định ăn mất câu hỏi mà KHÔNG báo lỗi).
- **USER LẬT KẾT LUẬN (2026-08-28): "có train mapper cho math/reasoning
  không? nếu chỉ train BFCL thì fail task khác là đúng rồi"** — ĐÚNG, đã
  kiểm code `build_data()`: train ~1330 item = bfcl ~655 + needle 430 +
  ifstruct 135 + pbtable 110. **Math 0, reasoning nhiều bước 0, QA văn
  xuôi 0.** Thêm nữa mọi target train đều NGẮN-TRÍCH (bfcl 24 tok,
  needle 16) trong khi bbh/gsm8k/musr đòi sinh dài suy luận (320 tok).
  → Thí nghiệm đã chạy KHÔNG tách được 2 giả thuyết: (H1) lớp hàm mapper
  yếu, (H2) miền train quá hẹp. Câu "phán quyết lần 4 về lớp hàm" là
  KẾT LUẬN VƯỢT DỮ LIỆU — hạ xuống "chưa xác định". Phép đo phân giải:
  train lại với data đa dạng (suite_gen.py 4 họ ĐÃ dựng sẵn, chưa dùng
  cho train + gsm8k/bbh train-split) rồi đo lại đúng benchmark đó.

## Hàng đợi (đã duyệt chuỗi 1→2→3 ngày 2026-08-14)

1. ✅ Spec decoding — đóng bằng số đo.
1b. ✅ Util sweep (lệnh user) — mặc định mới 0.97, đỉnh 12 phiên.
2. (Hoãn, KV-transfer chiếm ưu tiên từ 2026-08-14) profile `serve 4b`/`2b`.
3. Soak test 3-4 giờ chạy nền (rò bộ nhớ? TTFT trôi?) — đặt cuối ngày.
4. Đóng gói `serve 9b-prefill` (fp8 specialist, số đã đo) + cập nhật HTML report.
5. (Hoãn) P5 ablation nguồn graft; P6 converter toàn-model; sửa harness 3 thang token.

## Nợ dài hạn / quyết định chờ user

- Chưa đo qua mạng thật (mọi số qua localhost); chưa soak nhiều giờ (mục 3).
- Nộp bản nháp `upstream/` ra ngoài? — Revoke HF token sau chiến dịch — Mở B/C khi nào.
