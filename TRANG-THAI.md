# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi — KHÔNG cần hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn
sang `STATUS.md`.

Cập nhật: 2026-08-31.

## Trạng thái hiện tại

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
- **E1→E8 (2026-08-15) — TÓM TẮT, chi tiết STATUS.md**: copy nguyên cache
  4B→9B giữ 100% needle tới 30K, decode parity, TTFT 30K ×1,66 (2 GPU) /
  ×1,15-1,2 (đồng trú E3C: util 0,35 + --kv-cache-memory-bytes + eager; naive
  E3B bất khả thi). **Định luật ghép đôi E7**: attention thẳng hàng toàn họ
  (CCA 0,93-0,98); số phận cặp nằm 100% ở GDN — ≥0,9 bê được, ~0,8 học được
  (4→27B), ~0,23 tường ({0.8B,2B} lạc hệ). **E8 đóng**: phương ngữ GDN nhóm
  nhỏ không sửa được bằng adapter nhẹ (3 đòn LoRA/loss đều 0/5 dù gate thông
  tin sáng 5/5). Scope copy an toàn: chat/QA/RAG; function-calling hụt biên
  mỏng (E6c: bnb gánh nửa vết nứt, nửa còn lại là bottleneck thật).
- **E6 v3.1→v3.5 (2026-08-24→25) — mapper 4→27B, chi tiết STATUS.md**:
  v3.0 chết vì CONV_WARM bỏ token gold đầu khỏi loss; fix (cache cắt T-5 +
  warm 5 token cuối + CE trọn gold) → **v3.4 chốt 18/20 BFCL + needle niêm
  phong 15/15**, ladder trần L4 4096, template-XƯƠNG thay teacher prefill
  (~1,4 s/bước). v3.2 vách đá needle@2K là artifact phân phối train, đã sụp ở
  v3.3 bằng needle curriculum. **v3.5: ifstruct/pbtable là nợ của ĐỀ** —
  27B-self cũng chỉ 1/15 cùng giao thức → loại khỏi thang chính thức.
  **Học phí lần 2**: runtime recycle nuốt mapper_v33 (kỷ lục) + v3.2 →
  từ đó KHÔNG phóng train dài khi chưa có đường upload sống.
- **PHASE C (KVConnector vLLM thật, 2026-08-25→26) — chi tiết STATUS.md**:
  C2a 3/3 tiền đề PASS (block size 9B = 4B = 1056; LMCACHE_EXT_OK 0.5.4).
  **Cơ chế vận chuyển HOÀN CHỈNH: vá 1 dòng key lmcache → TTFT 30K 11-24s
  → ~1s (×12-16)**, kho GDN giao đủ 76/76. Nhưng exact-retrieval **cross
  57,1% (N=240 @8K, CI 51-63) vs self 100%**; scope ngữ nghĩa C2c cũng
  **self 90% | cross 55%**. Mọi giả thuyết "một con bug" đều bị bác (W4A16,
  bf16-consumer, trang-GDN-rơi) → **ĐỊNH LUẬT BIÊN MỎNG**: decode đầu trên
  cache ngoại sát mép vực số học. Hybrid thử-cross-fail-thì-cold đã lời TTFT
  ngay; polisher cần kéo 57→95%+.
- **Hạ tầng vá cùng đợt**: ghim `vllm==0.27.1`; `run.sh serve` chết câm khi
  thiếu `/tmp/vllm_env.sh` → vá `|| true`; HF_TOKEN đọc từ file + assert.
- **User chốt hướng (2026-08-26)**: copy-nguyên 4→9 "hên xui" → train
  mapper functional-loss cho 4→9; dựng `suite_gen.py` (4 họ đề) + `c2suite.sh`.
- **Giai đoạn A + ladder 4→9 XONG (2026-08-27)**: sanity chạy trọn không
  sửa code, ~1,7-1,8 s/bước, peak 8,76GiB; ladder 4096/8192/16384 đều
  KHÔNG OOM (9,1/10,0/11,9GiB) → chốt max-ctx 16384. Vá `hf_up()` ghim
  cứng `"v34/"` → thêm `--hf-prefix` (né đè mapper 4→27B).
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
- **Nhánh 4→27B ĐÓNG (2026-08-28) — chi tiết STATUS.md**: test niêm phong
  trong miền tái lập kỷ lục (bfcl 18/20, needle@2K 15/15) nhưng benchmark
  NGOÀI sụp (BBH 6,0% vs self 53,8%, GSM8K 0% vs 80%). Đối chứng 4B-self
  gsm8k **81,5% — cao hơn 27B** → thông tin CÓ trong cache, lỗi ở khâu DỊCH.
  243 ca self-đúng→cross-sai: 22% sinh rác, 78% lạc đề nhưng mạch lạc.
- **Kiến trúc 2 lớp + chuyển sang cặp 4→9 (2026-08-28, user chốt)** — chi tiết
  STATUS.md: 4→27B nền 2 model 16,16GiB (trống 5,54), gold ≤48 khi ctx ≤1024,
  GC bị loại về nguyên tắc (transformers ép `use_cache=False` mà forward 4B
  tồn tại chính để sinh cache), TBPTT mở được cổng. Cặp **4→9 rộng hơn hẳn**:
  nền 6,86GiB, ctx16384+gold256 chạy được → lấy lại cả ctx dài lẫn gold đầy đủ.
- **NGÀY 28-29/08 — joint 4→9 + 4 bug harness (chi tiết STATUS.md)**:
  pseudo-gold bằng vLLM offline nhanh **49×** (577 vs 11,8 tok/s); 4 bug
  harness làm mọi số trước đó sai, nặng nhất là **DỪNG SAI TOKEN KẾT THÚC**
  (tokenizer khai 248046 nhưng model kết thúc bằng 248044 ở 38/40 ca) → 92%
  tụt còn 32%, vá bằng `e5.stop_ids()`. Hai giả thuyết nghe hợp lý (fp4-vs-nf4,
  thiếu kernel `fla`) đều bị bác bằng đo. Đọc tay 20 đầu ra gsm8k: 13/17 ca là
  "văn hoàn hảo, đề bài bị bóp méo" — số sống sót, QUAN HỆ bị đảo lộn.
  Giả thuyết dung lượng GDN (0,8→25,2M) bị bác ở mốc phân xử đặt trước.
- **MA TRẬN ĐỐI CHỨNG ĐẦY ĐỦ (2026-08-31) — 1.650 mẫu niêm phong, MỌI cột
  cùng engine.** Báo cáo: https://claude.ai/code/artifact/b20fe8d6-0e21-44d1-afa8-b1622d62385a

  | bộ | n | 4B | 4B+LoRA | 9B | 4B→9B | 4B+LoRA→9B | 4B+map→9B | 4B+LoRA+map→9B |
  |---|---|---|---|---|---|---|---|---|
  | bfcl | 200 | 58,5 | 61,0 | 63,5 | — | **2,0** | 85,0 | **88,5** |
  | bbh | 779 | 45,3 | 51,3 | 30,6 | — | 35,4 | 39,3 | **66,0** |
  | needle | 240 | **100** | **100** | 100 | 61,2 | 75,0 | 89,2 | 98,3 |
  | suite_mid | 126 | **100** | **100** | 100 | — | 92,9 | 44,4 | 97,6 |
  | suite_rag | 126 | 98,4 | **100** | 97,6 | — | 42,1 | 23,8 | 90,5 |
  | suite_swe | 123 | **98,4** | 72,4 | 99,2 | 11,4 | 27,6 | 0,0 | 47,2 |
  | musr | 56 | 42,9 | **69,6** | 53,6 | 21,4 | 32,1 | 23,2 | 42,9 |

  **Bỏ chữ "cascade" khỏi mọi bảng (user chốt)** — ghi thẳng thành phần;
  `→ 9B` = 9B sinh câu trả lời từ cache chuyển sang.
- **ĐỌC KẾT QUẢ**:
  (a) **Mapper LUÔN tốt hơn phương án thay thế trực tiếp** (bê thẳng cache):
  bfcl 2,0→88,5 (**+86,5**), suite_rag 42,1→90,5, bbh 35,4→66,0. Mapper
  không "làm hỏng" gì — nó làm rất tốt việc dịch cache.
  (b) **Nhưng so với để 4B TỰ trả lời thì thua ở 5/7 bộ**: musr −26,7,
  suite_swe −25,2, suite_rag −9,5, suite_mid −2,4, needle −1,7. Chỉ bfcl
  (+27,5) và bbh (+14,7) là có lợi. Lý do: 4B vốn đã 98-100% ở các bài truy hồi.
  (c) **LoRA và mapper CỘNG HƯỞNG, không cộng dồn** (ma trận 2×2): musr —
  mapper một mình +1,8, LoRA một mình +10,7, ghép lại +19,7. suite_swe —
  mapper một mình **−11,4**, ghép lại **+19,6**. Chúng train cùng nhau nên
  tách ra là ra ngoài phân phối của nhau.
  (d) **Ranh giới truy-hồi/quan-hệ**, khớp 3 nguồn độc lập: hại nhẹ ở lấy
  nguyên văn (needle −1,7, suite_mid −2,4), hại nặng ở liên hệ thực thể
  (suite_swe −25,2, musr −26,7). Khớp probe kể-lại-đề (số giữ 73%, quan hệ
  giữ 33%) và định luật E7 (attention CCA 0,93-0,98 mang token; GDN
  CCA 0,23-0,9 mang quan hệ).
  (e) **E1 kết luận vượt dữ liệu**: "copy nguyên giữ 100% needle" đo trên
  12 mẫu; trên 240 mẫu với alpha đúng là **61,2%**. Thứ tự thì vẫn đúng
  (needle 61,2 ≫ musr 21,4 ≫ suite_swe 11,4).
- **CÒN TREO — lớn nhất**: `4B 45,3% > 9B 30,6%` trên bbh, tái lập trên **cả
  hai engine** (vLLM 30,3 / transformers 30,6) nên KHÔNG phải lỗi engine.
  Cộng với 9B được **đúng 0/30** trên movie_recommendation, disambiguation_qa,
  geometric_shapes, temporal_sequences → nghi 9B trả lời sai KHUÔN. Nếu đúng,
  +14,7 điểm của mapper ở bbh cũng phải xem lại. Phải đọc tay đầu ra.
- **vLLM BỎ QUA LoRA dù log nói có nạp** (`Using default LoRA kernel configs`):
  suite_swe ra **đúng 120/123 ở cả có lẫn không** adapter, cùng adapter đó
  trên transformers đổi 26 điểm. → mọi cột `+LoRA` phải đo bằng transformers.
  Tôi đã nghi đúng, rồi TỰ BÁC nghi ngờ đúng đó bằng một lập luận nghe hợp lý
  ("LoRA train cho mapper nên không đổi 4B") — bị 2 dòng số bác lại.
- **TĂNG TỐC (2026-08-31)**:
  - **eval gom lô decode: 6,7×** (1.150 mẫu 14 phút vs ~90 phút), cổng kiểm
    24 khớp/0 lệch. AN TOÀN ở decode dù probe đã bác ở prefill: trạng thái GDN
    không có chiều thời gian, RoPE đã áp lúc dựng cache — chỉ cần truyền
    `position_ids` riêng từng hàng. `batch_decode.py`.
  - **template-XƯƠNG thay prefill 9B đầy đủ** trong eval (~40 phút/lượt):
    `build_student_past` thay sạch mọi tensor nên prefill đó là tính toán thừa.
  - **dùng lại spill 4B** giữa các lượt (`spill_base`/`spill_lora`): cache 4B
    không phụ thuộc mapper → so biến thể mapper thì pha A giống hệt.
  - **KHÔNG dùng flash-attention** (user hỏi): đo được decode 1 token trên 9B
    ctx~500 thì nhân trọng số ~18 GFLOP, attention ~6 MFLOP = **0,03%**.
    Nút cổ chai là batch=1 (mỗi token đọc ~5GB trọng số 4-bit từ HBM).
  - **probe_train_batch: batch 2 cho 1,85-1,97× trên BƯỚC TRAIN THẬT**
    (mốc đặt trước 1,3×); batch 4 OOM. 89,5% item gom được thành lô 2 với
    **độ dài trùng khít** (không đệm → đồng nhất toán học). Ước tính
    3,00 → ~1,7 s/bước, kèm vá `gc.collect` mỗi 20 bước → ~1,5.
  - **Mapper vốn CHỈ chạy được batch 1** — `map_attn` ghim `reshape(T, H*dh)`,
    `map_gdn` lấy `S[0]`. Đã sửa giữ chiều batch. Bài kiểm "lô 2 == chạy riêng
    lẻ" bắt được lỗi thật trong chính bản vá: `rms.mean()` trung bình trên cả
    chiều batch (lệch 7,8e-3) → sửa thành theo từng mẫu. 23/23.
- **BUG CHẶN TRAIN đã sửa**: tham số GDN của Mapper không phải tensor LÁ
  (`A_r = base.requires_grad_(True)` rồi `B_r = base.clone()`), optimizer ném
  "can't optimize a non-leaf Tensor" — dính cả với `gdn_terms=1` mặc định.
  Eval không lộ ra vì chỉ chạy forward. Đã thêm bài kiểm dựng optimizer.
- **`e5.patch_recurrent_rebind()`**: GDN 5.15 cập nhật state bằng `.copy_()`
  IN-PLACE → vỡ autograd. e9_joint đã có bản vá kèm ghi chú "học phí 3 probe";
  probe_train_batch thành probe THỨ TƯ dính. Đã đưa ra e5_train dùng chung.
- **Chẩn đoán QUÁ KHỚP (2026-08-30)**: mapper trên train **86,0%** / val
  **63,3%** (chênh 22,7 điểm). suite_swe: train 100% / val 28,6% / niêm phong
  47,2% — thuộc lòng 190 mẫu. → phóng to mapper là SAI HƯỚNG; việc cần là đa
  dạng hoá dữ liệu. Giả thuyết "mapper quá nhỏ" của user bị bác bằng đo.
- **Train ĐÃ BÃO HOÀ — hai lần độc lập**: `joint49v` 67→66→62, `joint49w`
  100→95. Thêm bước/thêm-bớt dữ liệu đều không lên.
- **ĐANG CHẠY — `joint49x`, đợt GỘP (user duyệt 2026-08-31)**: LoRA trên 9B
  (phía ĐỌC) + gom lô theo độ dài, ấm từ `joint49w`. Lý do LoRA 9B: kiến trúc
  BẤT ĐỐI XỨNG — phía mã hoá được thích nghi (LoRA 4B), phía dịch được train
  (mapper), phía ĐỌC đóng băng. Thêm nữa **LoRA 4B chỉ chạm q/k/v/o_proj = chỉ
  lớp attention (8/32 lớp), chưa adapter nào chạm GDN** — mà E7 chỉ ra GDN mới
  là chỗ lệch. Nhắm `q_proj,o_proj,in_proj_qkvz,out_proj` = 5,8M tham số.
  - **Sanity 40 bước PASS**: `verify-meta` KHỚP ở CẢ (T=512,B=1) và (B=2);
    89,5% item vào được lô đầy (khớp đúng con số đo trước); peak 18,30 GiB/23.
  - **Tốc độ — đính chính ước tính của tôi**: 2,82 s/**bước** (không phải ~1,7
    như tôi ước), nhưng mỗi bước nuốt 2 mẫu → **1,41 s/mẫu, nhanh 2,13×**
    (probe đoán 1,85-1,97×). Cùng thời gian, 1.000 bước nay thấy **2.000 mẫu**.
    Đúng đòn bẩy cần: chẩn đoán là QUÁ KHỚP nên xem nhiều dữ liệu hơn/phút mới
    là thứ giúp được. `backward` 47-49% → LoRA 9B đi nhờ đường đã trả tiền.
  - **CHỐT CHẶN**: thêm dung lượng phía SINH làm việc "ăn gian" dễ hơn. Mốc
    phân xử đặt TRƯỚC khi chạy: `suite_swe` **>60% VÀ ctx-BỎ vẫn sập** = thành
    công; ≤60% = không phải phía đọc, nghi phạm quay về **dạng hàm `A·S·B`**;
    điểm lên mà ctx-BỎ **không** sập = ăn gian, bỏ.
  - Vá kèm: `eval_big --lora-t` nạp LoRA phía đọc (KHÔNG merge — 9B đang 4-bit,
    merge là đường đã làm hỏng checkpoint 4B một lần); `run_bigmapped.sh` tự
    phát hiện `lorat_best`. Quên cờ này = đo mapper với người-đọc chưa thích
    nghi, tức đo sai hẳn thứ vừa train.

- **`joint49y` TRAIN XONG (2026-08-31)** — best bước 750/1000, ấm từ `joint49w`
  đúng cấu hình gốc (max-ctx 4096, batch 2 × accum 2 = 4 mẫu/lần cập nhật, chỉ
  chừa hai biến: LoRA-9B + gom lô). **score 103, vượt kỷ lục joint49w (100)**:
  bfcl 15/15 (+1), bbh 48/77 (+1), musr 11/19 (+1), needle/suite_mid/suite_rag
  bằng, **suite_swe = suite_swe của 49w (4/7)** — không tụt bộ nào. Mốc phân
  xử `suite_swe >60%` đo trên val chỉ 7 mẫu (57,1%, sai số ±14 điểm/mẫu) —
  KHÔNG đủ để phân xử; số thật đợi tập niêm phong 123 mẫu.
  Checkpoint (`mapper_best`/`lora_best`/`lorat_best`) đã lên HF `joint49y/`.
  **Hai học phí trong đợt phóng**: (1) lần đầu chạy `max-ctx 16384 + accum 1`
  (khác `joint49w` ở max-ctx VÀ accum cùng lúc, đổi 4 biến chứ không phải 2)
  → val needle sập 15/15→0/15 KHÔNG quy được cho ai; đã dừng, ghép lại đúng
  cấu hình 49w rồi chạy lại tên mới `joint49y`. (2) lệnh dọn tiến trình bằng
  `pkill -f e9_joint.py` khớp luôn dòng lệnh của chính shell gọi nó (bẫy cũ,
  dạng khác) — sửa bằng đọc `/proc/*/cmdline` loại trừ pid của chính mình.
- **ĐANG CHẠY (nền) — đo niêm phong `joint49y`**: `run_joint49y_seal.sh` =
  mapped đầy đủ 7 bộ (1.875 mẫu) + đối chứng ctx-BỎ trên suite_swe+musr (nơi
  4B/9B gặp nhau về QUAN HỆ). Vài giờ, transformers tuần tự. Chốt chặn đã đặt
  TRƯỚC: `suite_swe >60%` VÀ ctx-BỎ vẫn sập = LoRA-9B đọc thật; `suite_swe`
  không vượt = nghi phạm quay về dạng hàm `A·S·B` của mapper; điểm lên mà
  ctx-BỎ KHÔNG sập = ăn gian, bỏ. Vá kèm: `eval_big --lora-t` nạp LoRA phía
  đọc (không merge, 9B đang 4-bit); `run_bigmapped.sh` tự phát hiện `lorat_best`.
- **BƯỚC KẾ (chờ user duyệt)**: (1) ✅ đã làm — gom lô theo độ dài cho train
  (gold có mặt nạ + `verify_meta` cho cặp (T,B)); (2) đọc tay đầu ra 9B trên
  4 tác vụ bị 0/30; (3) thêm NHIỆM VỤ đòi quan hệ vào dữ liệu train — làm ở
  tầng nhiệm vụ chứ KHÔNG so khớp tensor (luật error-placement: nMSE
  10,5→0,96 mà chức năng vẫn 0/5). Bằng chứng ủng hộ: suite_swe 4B một mình
  **98,4%** → thông tin quan hệ CÓ trong cache 4B, mapper làm mất chứ không
  phải nó không tồn tại. Mốc phân xử: suite_swe không vượt 60% và tỷ lệ giữ
  quan hệ không vượt 45% thì nghi phạm chuyển sang DẠNG HÀM `A·S·B`.

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
