# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật khi
trạng thái đổi — không hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn `STATUS.md`.

Cập nhật: 2026-09-04.

## Trạng thái hiện tại

- **🎯 EBA + GRPO — RL CÓ CẢI TIẾN THẬT, XÁC NHẬN THỐNG KÊ (2026-09-04)**:
  hướng do user đề xuất — sinh dữ liệu tổng hợp Entity-Binding-Arithmetic
  (thực thể+số+distractor, ground-truth 100% chắc chắn không qua model,
  `eba_gen.py`, 3 lớp điểm A=nhớ giá trị/B=không lẫn distractor/C=đáp số
  cuối đúng) rồi train GRPO 2 pha kiểu Unsloth (SFT-warm-start từ
  `joint49cc` + RL, K=6 nhóm, anchor-CE thay reference model, `eba_grpo.py`)
  — nhắm thẳng lỗi "gán SAI con số vào đúng thực thể" đã chẩn đoán ở gsm8k
  phía dưới. Scale-up 2000 item/1000 bước = `eba_grpo_v2c`.

  **So dứt điểm n=200 held-out (seed=99999≠seed train) + McNemar**:

  | checkpoint | A | B | C |
  |---|---|---|---|
  | baseline `joint49cc` (SFT thuần) | 0,365 | 0,110 | 0,310 |
  | `eba_grpo_v2c/best` (SFT+GRPO) | 0,762 | 0,470 | **0,630** |
  | `eba_grpo_v2c/last` (bước 1000) | 0,797 | 0,500 | **0,650** |

  best/last vs baseline: McNemar **p<0,0001** (χ²=52,2/59,1) — RL cải thiện
  THẬT gấp đôi C, không phải nhiễu. best vs last: p=0,29 — train quá bước
  ~150 không thêm lợi (val nội bộ dao động 0,73-0,80 suốt 850 bước, bão hoà
  sớm). Checkpoint + kết quả lên HF `eba_grpo_v2c/` + `evalbig/eba_*`.

  **NHƯNG đo trên gsm8k THẬT thì `eba_grpo_v2c` KHÔNG cải thiện** (TRAIN
  6,7%/NIÊM PHONG 4,0%, kém hơn `joint49bb` 8,0%) — cải tiến trên proxy EBA
  KHÔNG chuyển giao sang gsm8k thật. → **gộp `eba_grpo.py` thành 1 pipeline
  chung** (cờ `--task {eba,gsm8k}`, cùng engine RL, đổi nguồn dữ liệu+reward)
  rồi RL TRỰC TIẾP trên gsm8k thật + ground-truth CoT do 9B tự sinh có sẵn
  (`pseudo_gold_gsm2.json`, chỉ giữ quỹ đạo 9B làm ĐÚNG — không dạy mapper
  suy luận sai). Warm-start từ `eba_grpo_v2c/best`. **`gsm_grpo_v1c`, 400
  bước, K=3, gen_len=200** (đã vá `sample_rollout_batch` dừng sớm cả vòng
  lặp khi mọi nhánh gặp stop token → nhanh 2,67× — 21,7s→8,1s/bước, AN TOÀN
  tuyệt đối vì phần bỏ qua vốn bị trim sau đó, không đổi output). Học phí:
  K=4 OOM ở bước ~10 (VRAM 22,02/22,03GiB) → hạ K=3 (an toàn hơn, không đụng
  gen_len/gold_cap theo yêu cầu user). VAL nội bộ bão hoà dao động 0,07-0,23
  (đỉnh bước 300).

  **Đo dứt điểm gsm8k THẬT (`run_gsm_traintest.sh`)**:

  | checkpoint | TRAIN (60) | NIÊM PHONG (100) |
  |---|---|---|
  | `joint49bb` (SFT thuần) | 8,3% | 8,0% |
  | `joint49cc` (SFT thuần) | 13,3% | 4,0% |
  | `eba_grpo_v2c` (RL trên EBA proxy) | 6,7% | 4,0% |
  | **`gsm_grpo_v1c` (RL trực tiếp gsm8k)** | **10,0%** | **10,0%** |

  Train≈test (không quá khớp) — cao nhất chiến dịch gsm8k tới nay, vượt cả
  2 baseline SFT và nhánh RL-proxy. n=100 còn nhỏ (10 vs 8 đúng, chênh 2
  mẫu) — CHƯA chạy McNemar để phân xử ý nghĩa thống kê, cần làm ở lượt sau
  trước khi tuyên bố thắng chắc chắn. Checkpoint lên HF `gsm_grpo_v1c/`.
  Học phí: bug `continue` nhảy qua cả val/checkpoint khi reward đồng nhất
  trong nhóm K (đã vá, commit `5a02e1e`); rate-limit HF 60 commit/giờ khi
  save nhiều file riêng lẻ (CHƯA vá — cần `upload_folder` cho lần train sau).

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

  **PROBE TRÍCH-XUẤT-SỐ (2026-09-02)** — bắt 9B chỉ NHẮC LẠI một con số có sẵn
  trong đề, không tính toán (`run_49bb_probe_so.sh`, 40 bài × 2 biến thể):

  | biến thể | số đầu model sinh | có mặt trong đầu ra |
  |---|---|---|
  | nhắc số ĐẦU của đề | 50,0% | 80,0% |
  | nhắc số CUỐI của đề | 15,0% | 27,5% |

  **Cả hai đều THẤP → thông tin số KHÔNG tới được 9B nguyên vẹn** → loại
  `--w-entity`, đòn bẩy tưởng đúng lúc đó là `--gdn-terms`. Bậc theo độ sâu
  rõ (80%→27,5%): đầu đề còn, cuối mất. Đối chiếu `needle` 99,2% → vấn đề là
  MẬT ĐỘ chi tiết số, không phải truy hồi. (Kết luận "gdn-terms là đòn bẩy
  đúng" SAU ĐÓ bị chính oracle ablation bác — xem mục dưới.)

- **`joint49cc` (mapper `--gdn-terms` 1→4, GDN 0,8M→3,2M) — TRAIN + ĐO XONG
  (2026-09-02)**. Một-biến từ `joint49bb`. Val best score 8 ở bước 1000
  (`suite_swe` 7/8, `gsm8k` 1/16); val leo đều 5→6→6→7/8.

  **⚠️ BUG KÉP đã vá — số 81,7%/78,2%/16,7%/5,0% từng báo cáo là SAI**:
  (1) `eval_big.py` dựng mapper chấm điểm KHÔNG đọc `gdn_terms` từ `_meta`
  checkpoint → mặc định về 1 → nạp checkpoint `terms=4` bị **âm thầm cắt cụt**
  về 1 số hạng, không lỗi không cảnh báo. (2) cơ chế "nối lại" (resume) của
  `eval_big.py` tải nhầm kết quả ĐÃ SAI đó từ HF khi chạy lại lần đầu — tưởng
  đã sửa nhưng vẫn đọc cache cũ. Phải xoá cả file HF lẫn local rồi chạy lại
  LẦN 2 mới ra số đúng (xác nhận bằng log `gdn_terms=4` + "HF chưa có kết quả
  dở dang" ở mọi lượt). Đã thêm `test_eval_big.py` chống tái phát lỗi (1).

  **Số ĐÚNG — đo trên bộ `suite_swe` MỚI 600 mẫu** (`run_swe_big.sh`, seed
  90210 khác bộ niêm phong 31337, kiểm rò rỉ 0/600):

  | checkpoint | n | suite_swe |
  |---|---|---|
  | `joint49cc` (terms 4, ĐÚNG) | 600 | **81,0%** |
  | `joint49bb` (terms 1) | 600 | 78,2% |
  | `joint49cc` ctx-BỎ | 600 | **0,0%** (sạch) |

  So cặp: 49cc đúng/49bb sai=72, ngược lại=55 → McNemar χ²=2,02, **p≈0,156**
  — CÀNG không đạt ý nghĩa thống kê so với lần đo sai trước (p=0,076). Kết
  luận giữ nguyên: **chênh lệch chưa phân biệt được với nhiễu**. 21,2% mẫu
  đảo kết quả giữa hai checkpoint.

  **`gsm8k` — số ĐÚNG**: TRAIN 8/60=13,3% | NIÊM PHONG 4/100=4,0% (so
  `joint49bb`: train 8,3%/niêm phong 8,0%, gần bằng nhau). Chênh train>test
  của `joint49cc` (13,3 vs 4,0) LỚN hơn `joint49bb` — dấu hiệu **quá khớp
  nhẹ mới xuất hiện** khi mở dung lượng, dù cả hai vẫn rất thấp so với trần
  89,0%. → **dung lượng GDN KHÔNG phải nút thắt của gsm8k**, và mở thêm còn
  có nguy cơ phản tác dụng (quá khớp) chứ không giúp gì.
  **Bài học mẫu-nhỏ (lặp lại)**: val 8 mẫu báo 7/8 vs 4-5/8 nhưng 600 mẫu +
  McNemar cho thấy chênh không chắc chắn — không kết luận từ val 8-16 mẫu.

- **ORACLE ABLATION (2026-09-02, `oracle_ablation.py`, đề xuất user)** — hoán
  đổi trực tiếp attn/GDN mapped bằng cache 9B THẬT, n=30 gsm8k, không train:
  self **86,7%** (trần) | mapped 0,0% | attn-thật+GDN-mapper **26,7%** |
  attn-mapper+GDN-thật 3,3%. **NGƯỢC giả thuyết "GDN là nút thắt duy nhất"**
  (đúng ra hàng cuối phải gần trần). Đọc tay quyết định: hàng cuối sinh RÁC/
  SUY BIẾN HOÀN TOÀN (không phải sai số thường), hàng ba sinh văn mạch lạc
  chỉ sai số liệu. → cắm GDN thật cạnh attn mapped làm 9B suy biến thay vì
  được cứu — hai nửa cache cần NHẤT QUÁN với nhau; attn mapped (dù CCA E7
  cao) cũng đóng góp lỗi, không "đã tốt sẵn" như giả định cũ. Lên HF
  `evalbig/oracle_ablation.json`.

- **BRIDGE ORACLE (2026-09-02, `bridge_oracle.py`, giai đoạn 3 đề xuất user)
  — XÁC NHẬN TÍCH CỰC, n=30 gsm8k.** Giữ nguyên cache mapped cho toàn ngữ
  cảnh, CHÈN THÊM một đoạn prefill THẬT (không qua mapper) ngay trước sinh:

  | biến thể | tỷ lệ | độ dài bridge |
  |---|---|---|
  | mapped (không bridge) | 0,0% | — |
  | bridge_full (nguyên đề bài) | **23,3%** | 67 token |
  | bridge_nums (chỉ câu có số) | **16,7%** | 51 token |

  Đọc tay xác nhận đúng cơ chế: bridge sửa được CHÍNH LOẠI LỖI đã chẩn đoán —
  mapped bịa "20% raise" (đề thật 5%) → bridge dùng đúng 5%; mapped bịa điểm
  số "78" → bridge dùng đúng "100"; mapped bỏ hệ số "5 liters/pail" → bridge
  tính đúng "5×5=25". Phần còn sai chủ yếu là LỖI SUY LUẬN NHIỀU BƯỚC BÌNH
  THƯỜNG, không còn "bịa số từ hư không". → bản tóm tắt NGẮN (51 token) gần
  bằng bản đầy đủ (67 token). Lên HF `evalbig/bridge_full30.json`.

- **PIPELINE THẬT bridge tokens (2026-09-02, `real_bridge_4b.py`)** — 4B tự
  sinh bridge (không oracle), n=30 gsm8k: mapped 0,0% → bridge_4b **13,3%**
  (thấp hơn oracle 23,3%, do 4B đôi khi tự giải sai/dùng ký hiệu ẩn danh).
  Cơ chế hoạt động thật nhưng cần tinh chỉnh trước khi coi là giải pháp sản
  phẩm — **user sau đó chuyển ưu tiên sang synthetic-data+GRPO** (xem mục
  EBA+GRPO đầu file). Chi tiết đầy đủ + bài học vận hành batch: `STATUS.md`.

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
- **MAPPER 4B→9B TRAIN THẬT XONG (2026-08-27, max-ctx=16384)**: dừng sớm ở
  bước 984/2600 — **BFCL 23/25 | needle 29/29 | score 54**, nhỉnh hơn mapper
  4→27B mà đạt trực tiếp ctx 16384 (27B chỉ tới 4096 trên L4) — xác nhận E7
  (cặp 4→9 dễ hơn 4→27). Checkpoint + data trên HF `v49/`.
- **BENCHMARK NGOÀI — baseline 27B THUẦN (2026-08-27)**: BBH 53,8% | GSM8K
  80% | MuSR 58,1% (756 mẫu: 53,6%). Rác 0,0% cả 3 bộ — mốc đối chứng: khi
  chạy cross, mọi tỷ lệ rác > 0 đều quy được cho mapper. HF `extbench_self/`.
- **Nhánh 4→27B ĐÓNG (2026-08-28)**: test nội bộ kỷ lục (bfcl 18/20) nhưng
  benchmark NGOÀI sụp (BBH 6,0% vs self 53,8%, GSM8K 0% vs 80%). 4B-self
  gsm8k 81,5% cao hơn 27B → thông tin CÓ trong cache, lỗi ở khâu DỊCH.
- **Kiến trúc 2 lớp + chuyển sang cặp 4→9 (2026-08-28, user chốt, chi tiết
  STATUS.md)**: 4→27B nền 2 model 16,16GiB (trống 5,54), gold ≤48 khi ctx≤1024.
  Cặp **4→9 rộng hơn hẳn**: nền 6,86GiB, ctx16384+gold256 chạy được.
- **NGÀY 28-29/08 — joint 4→9 + 4 bug harness (chi tiết STATUS.md)**:
  pseudo-gold bằng vLLM offline nhanh **49×**; 4 bug harness làm mọi số trước
  đó sai, nặng nhất là **DỪNG SAI TOKEN KẾT THÚC** (tokenizer khai 248046
  nhưng model kết thúc bằng 248044) → 92% tụt còn 32%, vá bằng `e5.stop_ids()`.
  Đọc tay 20 đầu ra gsm8k: 13/17 ca "văn hoàn hảo, đề bài bị bóp méo" — số
  sống sót, QUAN HỆ bị đảo lộn. Giả thuyết dung lượng GDN bị bác lần đầu ở đây
  (và bác lại lần hai ở `joint49cc`, xem trên).
- **MA TRẬN ĐỐI CHỨNG ĐẦY ĐỦ (2026-08-31, 1.650 mẫu)** — bảng đầy đủ nằm
  trong BÁO CÁO HTML (link đầu file) + `STATUS.md`; số `suite_swe`/`musr`
  bản gốc dính lỗi chấm điểm đã vá, đừng trích lại. Kết luận còn giá trị:
  mapper luôn tốt hơn bê thẳng cache; LoRA+mapper cộng hưởng chứ không cộng
  dồn; ranh giới truy-hồi/quan-hệ khớp 3 nguồn độc lập. Bỏ chữ "cascade"
  khỏi mọi bảng (user chốt): ghi thẳng thành phần.
- **CÒN TREO**: `4B 45,3% > 9B 30,6%` trên bbh (cả 2 engine); 9B 0/30 trên
  4 task bbh cụ thể → nghi 9B sai KHUÔN, chưa xác minh. **vLLM BỎ QUA LoRA**
  dù log báo có nạp — `+LoRA` phải đo bằng transformers.
- **TĂNG TỐC (2026-08-31)**: eval gom lô decode 6,7× (`batch_decode.py`);
  template-XƯƠNG thay prefill 9B thừa; dùng lại spill 4B giữa các lượt;
  KHÔNG flash-attention (attention chỉ 0,03% phép tính); `probe_train_batch`
  batch 2 = 1,85-1,97× bước train thật; mapper vốn chỉ chạy batch 1 — đã sửa
  (`map_attn`/`map_gdn`), bài kiểm 23/23.
- **BUG CHẶN TRAIN đã sửa**: tham số GDN của Mapper không phải tensor LÁ —
  optimizer ném "can't optimize a non-leaf Tensor". Đã thêm bài kiểm.
- **`e5.patch_recurrent_rebind()`**: GDN 5.15 cập nhật state bằng `.copy_()`
  IN-PLACE → vỡ autograd. e9_joint đã có bản vá kèm ghi chú "học phí 3 probe";
  probe_train_batch thành probe THỨ TƯ dính. Đã đưa ra e5_train dùng chung.
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
- Chưa đo qua mạng thật; chưa soak nhiều giờ. Nộp `upstream/`? Revoke HF
  token sau chiến dịch? Mở B/C khi nào?
