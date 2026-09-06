# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật khi
trạng thái đổi — không hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn `STATUS.md`.

Cập nhật: 2026-09-05.

## Trạng thái hiện tại

- **🏆 CẢI TIẾN CÓ Ý NGHĨA THỐNG KÊ ĐẦU TIÊN CỦA CHIẾN DỊCH gsm8k
  (2026-09-06): ĐỊNH DẠNG CÓ CẤU TRÚC + SFT, không phải RL.** Hướng user
  chốt: bắt 9B tự sinh quỹ đạo `<think>` → `ENTITIES:` → `STEPS:` →
  `Final Answer:` (`gen_struct_gold.py`, lọc 2 tầng: parse được VÀ đáp số
  đúng → 2157 mẫu), rồi SFT cho mapper quen phân phối đó (`sft_struct.py`,
  1 epoch 2125 bước, parse 0% → 93,8%), rồi RL 4 reward phân rã trên chính
  định dạng đó (`gsm_struct.py` + `eba_grpo.py --task gsm8k_struct`).

  **Niêm phong 250 mẫu (gsm8k split `test`, rò rỉ 0/250, decode batch 1,
  `run_struct_sealed.sh`)**:

  | checkpoint | niêm phong 250 | nhánh chấm |
  |---|---|---|
  | **`sft_struct_v3` (SFT có cấu trúc)** | **55/250 = 22,0%** | final 54 |
  | `gsm_struct_rl_v2/best` (+RL) | 51/250 = 20,4% | final 51 |
  | `gsm_grpo_v1c` (kỷ lục cũ) | 29/250 = 11,6% | final 29 |
  | *trần self-9B* | *89,0%* | |

  **McNemar**: vs `gsm_grpo_v1c` — lệch 39-13, χ²=12,02, **p=0,0005 → CÓ ý
  nghĩa thống kê** (mọi lần trước chỉ tới p=0,11-0,16). vs
  `gsm_struct_rl_v2` — lệch 26-22, **p=0,665 → RL KHÔNG thêm gì**, dù hai bên
  bất đồng ở 48/250 mẫu. VAL nội bộ của RL trôi xuống đều suốt epoch (C 0,344
  bước 100 → 0,125 bước 508; `rel` đứng im 0,72-0,77). **Lần thứ tư RL bão
  hoà sớm trên val nội bộ.** Checkpoint + per-item trên HF.

  **Đọc cơ chế**: lỗi gốc đã chẩn đoán là "gán SAI số vào đúng thực thể";
  định dạng có cấu trúc BUỘC model viết ràng buộc số ra trước khi tính, nên
  phải cam kết sớm thay vì trôi. RL trên cùng định dạng không thêm tín hiệu.

- **🔭 ORACLE ABLATION TRÊN ĐỊNH DẠNG CÓ CẤU TRÚC (2026-09-06, n=100,
  `evalbig/oracle_struct100.json`) — BÁC giả thuyết "sắp chạm trần"**:

  | biến thể | tỷ lệ |
  |---|---|
  | self (9B tự prefill) | **93,0%** |
  | mapped (đường ống `sft_struct_v3`) | **22,0%** |
  | **attn THẬT + GDN-mapper** | **43,0%** |
  | attn-mapper + GDN THẬT | 3,0% |

  `mapped` 22,0% khớp ĐÚNG `sft_struct_v3` trên 250 mẫu — hai phép đo độc lập.
  **Dư địa cho phía huấn luyện là ~21 điểm (22 → 43), KHÔNG phải ~3 điểm** như
  Claude dự đoán. Sai vì lấy trần của oracle CŨ (26,7%, đo trên đường ống chưa
  có định dạng cấu trúc, lúc đó mapped=0%) áp cho hệ mới; định dạng cấu trúc
  nâng CẢ HAI đầu (0→22% và 26,7→43%). **Nút thắt lớn nhất giờ là ánh xạ
  ATTENTION** — và attention mapper chính là thứ được huấn luyện.
  `gdn_that` 3,0% tái lập độc lập lần 2: cắm GDN thật cạnh attn mapped làm 9B
  SUY BIẾN chứ không được cứu → **hai nửa cache phải nhất quán với nhau**.

- **⚠️ LỖI ĐO ĐẠC ĐÃ VÁ (2026-09-06) — cache 4B dùng chung nhầm giữa các
  checkpoint**: `eval_big.py` đặt tên thư mục spill chỉ là `lora` vs `base`,
  KHÔNG phân biệt LoRA-4B nào → checkpoint chạy sau âm thầm dùng lại cache 4B
  của checkpoint chạy trước = đo "mapper của A trên cache 4B của B", không lỗi
  không cảnh báo. Đúng khi so các biến thể mapper dùng CHUNG một LoRA-4B (giả
  định cũ), sai hẳn khi so checkpoint từ các lượt train khác nhau. Vá: băm nội
  dung file adapter 4B vào tên spill + `assert` dừng hẳn nếu không thấy file
  trọng số (sha1 của rỗng là hằng số → mọi checkpoint cùng tên). **Mọi số so
  sánh nhiều-checkpoint đo TRƯỚC 2026-09-06 đều cần soi lại bằng con mắt này.**
  Số 22,0% của `sft_struct_v3` đã chạy lại độc lập sau khi vá: **trùng khít
  55/250, cùng phân rã 54 final + 1 so_cuoi**.

- **⚡ TĂNG TỐC RL 2,25× — "lấy tốc độ vLLM ngay trong process" (2026-09-05)**.
  vLLM không cắm thẳng được (rollout bắt đầu từ **cache do mapper sinh**;
  LoRA-9B đổi mỗi bước; hết VRAM cho engine thứ hai). Nhưng tách được 3 nguồn
  tốc độ và đo riêng (`probe_decode_speed.py`, 9B bnb-4bit, decode 64 token):
  ms/bước decode **gần như KHÔNG đổi** từ 2→16 hàng (95,7 → 100,1) trong khi
  thông lượng ×7,7 (20,9 → 159,8 tok/s) — decode ở batch nhỏ bị chặn bởi băng
  thông đọc TRỌNG SỐ. Đó chính là continuous-batching của vLLM, lấy được
  nguyên vẹn mà không cần vLLM. Kernel Marlin qua transformers = **ngõ cụt**
  (giải nén ngược về bf16). Đồng bộ GPU→CPU mỗi token mất 9-10% → gom 1
  lần/16 token.

  **Kiến trúc mới (`--bsz`)**: mỗi bước xử lý B mẫu × K nhánh, lô chỉ gồm mẫu
  **cùng độ dài prompt CHÍNH XÁC** (không đệm — đệm phá attention 96%); B=4
  phủ 91,8% pool, phần lẻ chạy lô nhỏ hơn. Advantage chuẩn hoá RIÊNG trong
  nhóm K của TỪNG mẫu. `test_grpo_batch.py` 7/7.

  **Đoán sai 2 lần về chỗ OOM (logits → GDN forward), phải đo mới ra**: đỉnh
  VRAM nằm ở **backward của pha teacher-force** (pha 1 đỉnh 16,4 GiB / pha 2
  đỉnh 20,35 trên 22,03) → pha 1 gộp rộng (`@no_grad`), pha 2 chia miếng
  `--tf-chunk` và cộng dồn gradient (tổng loss đồng nhất). Đo dứt điểm:

  | cấu hình | s/bước | **s/mẫu** | đỉnh VRAM |
  |---|---|---|---|
  | bsz=1 k=2 (cũ) | 20,5 | 20,5 | 16,2 GiB |
  | bsz=2 k=2 tf=1 | 28,5 | 14,2 | 17,7 GiB |
  | **bsz=4 k=2 tf=1** | 36,5 | **9,1** | 20,4 GiB |
  | bsz=4 k=2 tf=2 | — | — | **44/48 miếng OOM** |

  → chốt **bsz=4, k=2, tf-chunk=1**; 1 epoch = 508 bước ≈ 5,1 giờ (trước
  10,7). Pha 2 KHÔNG rẻ theo hàng như pha 1 → k lớn vẫn đắt.
  **Bẫy đã chặn**: lưới an toàn bỏ-qua-miếng-khi-OOM cứu khỏi crash nhưng ở
  tf=2 làm 44/48 miếng bị bỏ → train "chạy xong" mà gần như không gradient
  (log đẹp, kết quả rỗng). Chốt: **dừng hẳn nếu >20% miếng OOM trong 20 bước
  đầu**. Sửa kèm: `--gsm-limit 0` (thiếu → pool bị cắt 2157→1200),
  `log_softmax(dtype=fp32)` thay `.float()` (bit-identical, bỏ 222MB/hàng).

- **EBA + GRPO (2026-09-04, chi tiết `STATUS.md`) — bài học về PROXY.**
  Sinh dữ liệu tổng hợp Entity-Binding-Arithmetic (ground-truth không qua
  model) rồi GRPO 2 pha: trên chính EBA, RL cải thiện **thật và mạnh** —
  C 0,310 → 0,630/0,650, McNemar **p<0,0001**. **NHƯNG đo trên gsm8k THẬT
  thì KHÔNG chuyển giao** (niêm phong 4,0%, kém hơn `joint49bb` 8,0%).
  → gộp `eba_grpo.py` thành 1 pipeline chung (`--task {eba,gsm8k,gsm8k_struct}`)
  rồi RL TRỰC TIẾP trên gsm8k thật (`gsm_grpo_v1c`, 400 bước, K=3): TRAIN
  10,0% / NIÊM PHONG 10,0% (train≈test, cao nhất chiến dịch **lúc đó**;
  nay đã bị `sft_struct_v3` 22,0% vượt với p=0,0005 — xem đầu file).
  McNemar hồi đó vs `joint49cc` p=0,149 và vs `eba_grpo_v2c` p=0,114 —
  **chưa đủ bằng chứng ở n=100**, đúng lý do sau này chuyển sang n=250.
  Học phí kỹ thuật: bug `continue` nhảy qua cả val/checkpoint khi reward
  đồng nhất trong nhóm K (vá ở `5a02e1e`); rate-limit HF 60 commit/giờ khi
  save nhiều file riêng lẻ → `save_ckpt()` gộp 1 commit/checkpoint.

- **🎯 MỤC TIÊU HIỆN TẠI (user chốt 2026-09-01): CHỈ `suite_swe` (đầy đủ) +
  `gsm8k`.** `joint49bb` (warm-start từ `joint49z`, drop hết các bộ khác kể
  cả ifstruct/pbtable) đã TRAIN XONG 1000 bước + NIÊM PHONG THẬT (123
  suite_swe + 100 gsm8k). Kết quả:

  | bộ | self 9B | mapped | ctx-BỎ | so joint49z |
  |---|---|---|---|---|
  | suite_swe | 99,2% | **77,2%** | 0,0% (sạch) | 56,1% → **77,2% (+21,1)** |
  | gsm8k | 89,0% | 8,0% | (bỏ qua, ctx=câu hỏi) | lần đầu đo đầy đủ |

  **`suite_swe` là bước nhảy lớn nhất chiến dịch** — thu hẹp phạm vi train
  (bỏ 7 bộ khác, không loãng tín hiệu) hiệu quả rõ rệt. `gsm8k` vẫn rất yếu
  (8%) dù đã sửa cắt gold đầu+đuôi (học phí joint49aa: cắt chỉ-đầu làm mất
  kết luận đáp án) — không còn kẹt cứng 0% nhưng quan hệ toán nhiều-bước
  có vẻ khó hơn hẳn với mapper hiện tại. Checkpoint `joint49bb/` đã lên HF.
  **`joint49bb` = checkpoint tham chiếu mới nhất** (thay `joint49z`).

  **PHÂN XỬ "học không nổi" vs "quá khớp" (2026-09-02)** — chấm `joint49bb`
  trên CHÍNH TẬP TRAIN (60 mẫu/bộ, `EVALBIG_ITEMS` mới thêm vào eval_big.py):

  | bộ | train | niêm phong | chênh |
  |---|---|---|---|
  | suite_swe | 93,3% | 77,2% | +16,1 (quá khớp NHẸ, tổng quát hoá thật) |
  | gsm8k | **8,3%** | 8,0% | **+0,3 → KHÔNG hề quá khớp** |

  **gsm8k sai y hệt trên chính dữ liệu đã train** → loại giả thuyết "thiếu/
  kém đa dạng dữ liệu". Đọc tay: mô hình lấy ĐÚNG thực thể nhưng **gán SAI
  con số** ("Kylie dùng 3 khăn" → sinh "6 khăn"). Chữ nghĩa truyền qua cache
  tốt, **liên kết số-với-thực-thể thì không** — giới hạn cơ chế mapper.

  **PROBE TRÍCH-XUẤT-SỐ (2026-09-02, chi tiết `STATUS.md`)** — bắt 9B chỉ
  NHẮC LẠI một con số có sẵn trong đề: nhắc số ĐẦU 50,0% đúng / 80,0% có mặt;
  nhắc số CUỐI 15,0% / 27,5%. **Cả hai đều THẤP → thông tin số KHÔNG tới được
  9B nguyên vẹn**; bậc theo độ sâu rõ (đầu đề còn, cuối mất). Đối chiếu
  `needle` 99,2% → vấn đề là MẬT ĐỘ chi tiết số, không phải truy hồi.

- **`joint49cc` (mapper `--gdn-terms` 1→4) — TRAIN + ĐO XONG (2026-09-02),
  chi tiết đầy đủ ở `STATUS.md`**. `suite_swe` 600 mẫu: 49cc **81,0%** vs
  49bb 78,2% → McNemar p≈0,156, **chưa phân biệt được với nhiễu**. `gsm8k`:
  train 13,3% / niêm phong 4,0% (49bb: 8,3/8,0) → **quá khớp nhẹ mới xuất
  hiện** khi mở dung lượng → **dung lượng GDN KHÔNG phải nút thắt gsm8k**.
  ⚠️ Số 81,7/78,2/16,7/5,0 báo cáo lần đầu là SAI do bug kép (eval_big không
  đọc `gdn_terms` từ `_meta` → âm thầm cắt về 1; resume tải lại kết quả sai
  từ HF). Đã có `test_eval_big.py` chống tái phát.
  **Bài học mẫu-nhỏ (lặp lại)**: không kết luận từ val 8-16 mẫu.

- **ORACLE ABLATION (2026-09-02, `oracle_ablation.py`)** — hoán đổi attn/GDN
  mapped bằng cache 9B THẬT, n=30 gsm8k: self **86,7%** (trần) | mapped 0,0%
  | attn-thật+GDN-mapper **26,7%** | attn-mapper+GDN-thật 3,3%. **NGƯỢC giả
  thuyết "GDN là nút thắt duy nhất"**. Đọc tay: hàng cuối sinh RÁC/SUY BIẾN
  hoàn toàn, hàng ba sinh văn mạch lạc chỉ sai số liệu → hai nửa cache cần
  NHẤT QUÁN với nhau; attn mapped cũng đóng góp lỗi.

- **BRIDGE ORACLE (2026-09-02, `bridge_oracle.py`, chi tiết `STATUS.md`) —
  XÁC NHẬN TÍCH CỰC, n=30 gsm8k.** Giữ cache mapped cho toàn ngữ cảnh, CHÈN
  THÊM một đoạn prefill THẬT ngay trước sinh: mapped 0,0% → bridge_full
  **23,3%** (67 token) / bridge_nums **16,7%** (51 token). Đọc tay xác nhận
  đúng cơ chế: bridge sửa CHÍNH loại lỗi đã chẩn đoán (mapped bịa "20% raise"
  khi đề thật 5%); phần còn sai là lỗi suy luận nhiều bước bình thường, không
  còn "bịa số từ hư không". Bản tóm tắt NGẮN gần bằng bản đầy đủ.

- **PIPELINE THẬT bridge tokens (2026-09-02, `real_bridge_4b.py`)** — 4B tự
  sinh bridge (không oracle), n=30 gsm8k: mapped 0,0% → bridge_4b **13,3%**
  (thấp hơn oracle 23,3% do 4B đôi khi tự giải sai). Cơ chế hoạt động thật
  nhưng **user sau đó chuyển ưu tiên sang synthetic-data+GRPO**.

- **Báo cáo toàn cục "Prefill bằng model nhỏ" (quy tắc 6c)**:
  https://claude.ai/code/artifact/8e4cccf6-b447-4439-97c2-14e7ca9ffee1
  (file `prefill-model-nho.html` trong scratchpad). Bao gồm 4 số then chốt,
  trạng thái 6 nhánh, 3 định luật, ràng buộc VRAM 4→9 vs 4→27, và 2 giả
  thuyết H1/H2 chưa phân giải. Báo cáo cũ về serving 9B/L4 vẫn là
  `bao-cao-l4.html` (chủ đề khác, không đè lên).

- **Repo đã refactor xong** (khuôn transformers, user chốt): `models/qwen3_5/`
  (engine/vllm + 19 patches, load/{gguf,quantize,pytorch_tensor}, hardware/l4,
  utils) + `sdk/ loading/ logging/ utils/ bench/ tests/`. **Registry đệ quy**
  (`python register.py --flat`; folder không có register.py = vô hình).
  **`run.sh` = 1 lệnh kiểu vLLM**; notebook A = ĐÚNG 1 CELL, cần gì thêm lệnh.
  Đã kiểm sống trên Colab: fresh 370s / ấm 140s, smoke đúng.
- **Vận hành**: chỉ notebook A (server `colab-mcp`); B/C chờ lệnh đích danh;
  không subagent trừ khi user cho phép.
- **Config production** (chi tiết STATUS.md): 9B mml 65536/mnbt 1088/util
  0.97 (12 phiên, 358,1 tasks/hr warm); 27B mml 8192/mnbt 512/util 0.97
  (decode 15,8 tok/s). Spec decoding ngram OFF mặc định trên L4 (đo −36%/
  −28% tasks/hr). Chiến dịch 9B ĐÓNG SẠCH.

## Nghiên cứu KV-transfer (lệnh user 2026-08-14, đang chạy)

- Paper 2608.03893: mapper tự cài ở `models/qwen3_5/kv_transfer/` (không có code
  chính thức). **E0 phán quyết: context sống ở CẢ GDN state lẫn KV** (needle
  10/10 → 0/10 khi xóa một trong hai) → Phase B GDN-mapping bắt buộc.
- **E1→E8 (2026-08-15) — TÓM TẮT, chi tiết STATUS.md**: copy nguyên cache
  4B→9B giữ 100% needle tới 30K, TTFT 30K ×1,66 (2 GPU)/×1,15-1,2 (đồng trú).
  **Định luật ghép đôi E7**: attention thẳng hàng toàn họ (CCA 0,93-0,98);
  số phận cặp nằm ở GDN — ≥0,9 bê được, ~0,8 học được (4→27B), ~0,23 tường
  ({0.8B,2B} lạc hệ). **E8 đóng**: phương ngữ GDN nhóm nhỏ không sửa được
  bằng adapter nhẹ. Scope copy an toàn: chat/QA/RAG; function-calling hụt.
- **E6 v3.1→v3.5 (2026-08-24→25, mapper 4→27B, chi tiết STATUS.md)**: fix
  CONV_WARM → v3.4 chốt 18/20 BFCL + needle 15/15. v3.5: ifstruct/pbtable là
  nợ của ĐỀ (loại khỏi thang). Học phí: recycle nuốt mapper_v33.
- **PHASE C (KVConnector vLLM thật, 2026-08-25→26, chi tiết STATUS.md)**:
  vá 1 dòng key lmcache → TTFT 30K 11-24s→~1s (×12-16). Nhưng exact-retrieval
  cross 57,1% vs self 100% → **ĐỊNH LUẬT BIÊN MỎNG**: decode đầu trên cache
  ngoại sát mép vực số học, mọi giả thuyết "một con bug" bị bác.
- **User chốt hướng (2026-08-26)**: copy-nguyên 4→9 "hên xui" → train
  mapper functional-loss cho 4→9; dựng `suite_gen.py` (4 họ đề) + `c2suite.sh`.
- **Giai đoạn A + ladder 4→9 XONG (2026-08-27)**: ladder 4096/8192/16384
  KHÔNG OOM → chốt max-ctx 16384.
- **Mở context 27B (2026-08-27, chi tiết STATUS.md)**: gradient checkpointing
  đóng hẳn (OOM do 1 lớp `repeat_kv`); CPU-offload thủ công THẮNG (steady
  12,81GiB, T=8192 OK, T=16384 vẫn OOM) — `load_4bit_cpu_offload_io`.
- **MAPPER 4B→9B TRAIN THẬT XONG (2026-08-27, max-ctx=16384)**: BFCL 23/25 |
  needle 29/29 | score 54, xác nhận E7 (4→9 dễ hơn 4→27). HF `v49/`.
  Baseline 27B thuần đối chứng: BBH 53,8%/GSM8K 80%/MuSR 58,1%, rác 0,0%
  cả 3 bộ (HF `extbench_self/`).
- **Nhánh 4→27B TẠM DỪNG — KHÔNG đóng (user đính chính 2026-09-04)**: nội bộ
  kỷ lục (bfcl 18/20, needle 15/15) nhưng benchmark NGOÀI sụp (BBH 6,0% vs
  self 53,8%) — thông tin CÓ trong cache, lỗi ở khâu DỊCH (chính là vấn đề
  phương pháp entity/relationship + reward phân rã đang nhắm). **Dùng 4→9 để
  TÍCH LUỸ trước, rồi đẩy mạnh vào 4→27** (user chốt): nền VRAM 6,86GiB vs
  16,16GiB, ctx16384+gold256 → lặp nhanh hơn nhiều, công thức tái dùng 100%.
  **27B = CORE TARGET của sản phẩm.**
- **⚠️ Luật đọc CCA (Claude sai 2026-09-04, user bắt)**: CCA/VarExpl là thước
  TUYẾN TÍNH — chỉ kết luận được "bê thẳng/tile/ánh xạ tuyến tính hỏng",
  KHÔNG kết luận được mapper phi tuyến đã train làm được gì. Bằng chứng ngay
  trong dự án: 4→27B CCA-GDN 0,785 + không có tile nguyên, mà mapper train
  vẫn đạt bfcl 18/20 + needle 15/15. Đúng luật error-placement đã kiểm ≥4 lần.
- **NGÀY 28-29/08 — joint 4→9 + 4 bug harness (chi tiết STATUS.md)**:
  pseudo-gold bằng vLLM offline nhanh **49×**; bug nặng nhất **DỪNG SAI TOKEN
  KẾT THÚC** (tokenizer khai 248046, model kết thúc 248044) → 92% tụt 32%, vá
  bằng `e5.stop_ids()`. Đọc tay 20 đầu ra gsm8k: 13/17 ca "văn hoàn hảo, đề
  bài bị bóp méo" — số sống sót, QUAN HỆ bị đảo lộn.
- **MA TRẬN ĐỐI CHỨNG (2026-08-31, 1.650 mẫu)** — bảng đầy đủ ở BÁO CÁO HTML
  + `STATUS.md`; số `suite_swe`/`musr` bản gốc dính lỗi chấm đã vá, đừng trích
  lại. Kết luận còn giá trị: mapper luôn tốt hơn bê thẳng cache; LoRA+mapper
  cộng hưởng chứ không cộng dồn.
- **CÒN TREO**: `4B 45,3% > 9B 30,6%` trên bbh — nghi 9B sai KHUÔN. **vLLM BỎ
  QUA LoRA** dù log báo nạp — đo `+LoRA` phải bằng transformers.
- **TĂNG TỐC (2026-08-31)**: eval gom lô decode 6,7× (`batch_decode.py`);
  KHÔNG flash-attention (attention chỉ 0,03% phép tính); mapper vốn chỉ
  chạy batch 1 — đã sửa (`map_attn`/`map_gdn`), bài kiểm 23/23.
- **`e5.patch_recurrent_rebind()`**: GDN 5.15 cập nhật state bằng `.copy_()`
  IN-PLACE → vỡ autograd (đã dính 4 probe). Bản vá dùng chung ở `e5_train`.
- **Chẩn đoán QUÁ KHỚP (2026-08-30)**: mapper train 86,0%/val 63,3% (chênh
  22,7); suite_swe train 100%/val 28,6%/niêm phong 47,2% — thuộc lòng 190
  mẫu. → phóng to mapper SAI HƯỚNG; giả thuyết "mapper quá nhỏ" bị bác.
- **Train ĐÃ BÃO HOÀ hai lần độc lập**: `joint49v` 67→66→62, `joint49w`
  100→95 — thêm bước/dữ liệu đều không lên.
- **`joint49y` TRAIN XONG (2026-08-31)** — best bước 750/1000, ấm từ `joint49w`
  đúng cấu hình gốc (max-ctx 4096, batch 2 × accum 2 = 4 mẫu/lần cập nhật, chỉ
  chừa hai biến: LoRA-9B + gom lô). **score 103, vượt kỷ lục joint49w (100)**:
  bfcl 15/15 (+1), bbh 48/77 (+1), musr 11/19 (+1), needle/suite_mid/suite_rag
  bằng, **suite_swe = suite_swe của 49w (4/7)** — không tụt bộ nào. Mốc phân
  xử `suite_swe >60%` đo trên val chỉ 7 mẫu (57,1%, sai số ±14 điểm/mẫu) —
  KHÔNG đủ để phân xử; số thật đợi tập niêm phong 123 mẫu.
  Checkpoint đã lên HF `joint49y/`. Hai học phí: (1) đổi 4 biến cùng lúc
  (max-ctx+accum) làm val needle sập không quy được cho ai — ghép lại đúng
  cấu hình 49w mới sửa; (2) `pkill -f e9_joint.py` khớp luôn shell gọi chính
  nó — sửa bằng đọc `/proc/*/cmdline` loại trừ pid của mình.
- **Đợt sửa thang đo (2026-08-31, chi tiết STATUS.md)**: `suite_gen.score`
  ĐÃ VÁ (khớp chuỗi con trên số garble — `test_suite_gen_scoring.py` 15/15).
  Bằng chứng cơ chế mạnh nhất: tỷ lệ suy biến toàn bộ 1.650 mẫu giảm đều
  23,6%→16,8%→16,1%→8,4% qua từng thành phần LoRA-4B/mapper/LoRA-9B. Báo
  cáo HTML: https://claude.ai/code/artifact/b20fe8d6-0e21-44d1-afa8-b1622d62385a

- **`joint49z` (2026-09-01, checkpoint tham chiếu thời điểm đó, nay đã bị
  `joint49bb`→`joint49cc`→`eba_grpo_v2c`→`gsm_grpo_v1c` thay thế nhiều lớp)**:
  pseudo-gold CoT thật từ chính 9B (user đề xuất) — niêm phong 1.650 mẫu:
  `suite_swe` 3,3%→52,8%, `musr`→75,0%, ctx-BỎ sạch cả hai. Chi tiết đầy
  đủ (bug musr ctx-BỎ ban đầu, bbh vượt trần self): `STATUS.md`.

## Hàng đợi (đã duyệt 2026-08-14)
1. ✅ Spec decoding + ✅ util sweep (mặc định 0.97, đỉnh 12 phiên) — đóng bằng số đo.
2. (Hoãn, KV-transfer ưu tiên) profile `serve 4b`/`2b`.
3. Soak test 3-4 giờ chạy nền — đặt cuối ngày.
4. Đóng gói `serve 9b-prefill` (fp8 specialist, số đã đo) + cập nhật HTML report.
5. (Hoãn) P5 ablation nguồn graft; P6 converter toàn-model.

## Nợ dài hạn / quyết định chờ user
- Chưa đo qua mạng thật; chưa soak nhiều giờ. Nộp `upstream/`? Revoke HF token? Mở B/C?
