# Trạng thái và hàng đợi việc

File này được CLAUDE.md nạp tự động đầu mỗi phiên. Claude TỰ ĐỘNG cập nhật nó khi
trạng thái thay đổi — KHÔNG cần hỏi user. Giới hạn cứng ≤300 dòng; chi tiết dồn
sang `STATUS.md`.

Cập nhật: 2026-08-29.

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
- **USER CHUYỂN HƯỚNG (2026-08-28): "khoan dùng idea mới trên 4-27, dùng
  cặp 4-9 để đảm bảo chất lượng đã"** — đã dừng job 4→27 (chạy được nhưng
  phải hy sinh: gold gsm8k 150-250 token bị cắt còn 48). **Đo lại bao cho
  4→9: rộng hơn hẳn** — nền 2 model chỉ **6,86 GiB** (9B 4-bit ~4,4 vs 27B
  12,7), trống 14,88 GiB. Với tbptt=128 **MỌI cấu hình chạy**:
  ctx4096+gold256 = 15,70GiB/13,1s | ctx8192+gold256 = 17,04/15,0 |
  **ctx16384+gold256 = 19,71/19,1**. (tbptt=0 vẫn OOM ở mọi ctx.)
  → 4→9 lấy lại được CẢ ctx dài LẪN gold đầy đủ; nhánh 27B mất cả hai.
- **TRAIN JOINT 4→9 ĐANG CHẠY (2026-08-28)**: `e9_joint.py` + `gen_data.py`
  (7768 train/330 val: gsm8k 2882 + bbh 2500 + musr 474 + suite 767 + bfcl
  655 + needle 380 + pbtable 110), ctx 8192, tbptt 128, gold 256,
  warm-start mapper v49 (92% BFCL/100% needle), 2000 bước, auto-upload HF
  `joint49/` mỗi mốc val. **Rò rỉ 0/6898** vs 580 mẫu niêm phong (đối chiếu
  chuỗi, chạy trên chính runtime train). Bước 20: ce 1,475 | **4,79s/bước**
  | peak 12,05GiB. Vá 2 bug: `ifstruct` không có `gold` (do B0 sinh, joint
  không có B0), và **cắt TRÁI khi tokenize** (mọi prompt bộ này đặt câu hỏi
  Ở CUỐI — cắt phải mặc định ăn mất câu hỏi mà KHÔNG báo lỗi).
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

- **NGÀY 28-29/08 — CHIẾN DỊCH JOINT 4→9 + 4 BUG HARNESS** (chi tiết STATUS):
  - **Pseudo-gold bằng vLLM offline** (gợi ý user): 577 tok/s vs 11,8 của
    transformers = **nhanh 49×**; 671k token trong 30,7 phút. Thu 3.574/6.098
    mẫu 9B tự làm đúng → đích học đổi từ đáp án người viết sang **quỹ đạo 9B
    tự đi** (user: "cần map gần 9B nhất" → đích trùng luôn thước đo).
  - **BỐN BUG HARNESS** làm mọi số trước đó sai: (1) ngưỡng gold<2 token loại
    16,7% dữ liệu (musr 100%, bbh 25%); (2) `continue` nuốt mốc val im lặng;
    (3) val tính lại cột `self` mỗi mốc dù nó không đổi; (4) **DỪNG SAI TOKEN
    KẾT THÚC** — tokenizer khai 248046 `<|im_end|>` nhưng model kết thúc bằng
    248044 `<|endoftext|>` (38/40 ca), nên vòng sinh KHÔNG DỪNG, sinh tràn rồi
    lan man, bộ chấm gsm8k lấy số cuối → **92% tụt còn 32%**. Ảnh hưởng cả
    `e9_joint` lẫn `ext_bench` → **các baseline 27B-self/4B-self đã báo cáo
    đều đáng nghi, nhiều khả năng bị ép thấp**. Vá bằng `e5.stop_ids()`.
    Hai giả thuyết trước đó (fp4-vs-nf4, thiếu kernel `fla`) đều nghe hợp lý,
    đều có bằng chứng gián tiếp đúng, và đều **bị bác bằng đo**.
  - **Val 150 mẫu, cột `self` tra cứu từ pseudo-gold** (miễn phí + chuẩn hơn).
    Lượt `joint49s` score 48→50→61→64→**67**, chưa bão hoà. Mapper **VƯỢT TRẦN**
    ở bfcl 12/12 (trần 9), musr 12/15 (trần 2), bbh 25/41 (trần 9), suite_mid
    5/5 (trần 0) — tức nó không chỉ dịch cache mà mang thêm kỹ năng vào.
    **Nhưng gsm8k đứng im 1-5/53 trong khi trần 44/53.**
  - **ĐỌC TAY 20 ĐẦU RA gsm8k** (user chỉ đạo "đọc lỗi trước đã"): self 17/20,
    mapped 0/20; lặp trigram 0,137 và 19/20 đúng định dạng → **KHÔNG** phải
    rác/định dạng. 13/17 ca là **"văn hoàn hảo, đề bài bị bóp méo"**: con SỐ
    sống sót, QUAN HỆ và việc gán thuộc tính cho THỰC THỂ bị đảo lộn
    ("Kate 29 tuổi"→"Tully 29 tuổi"; "già hơn nửa tuổi"→"trẻ hơn 20 năm";
    bịa ra con dê không có trong đề).
  - **GIẢ THUYẾT ĐANG KIỂM**: mapper phân bổ tham số **ngược** với nơi thông
    tin nằm — attention 16,8M (CCA 0,93-0,98, đã thẳng hàng sẵn) vs GDN 0,8M
    (CCA 0,23-0,9, chỗ quyết định), và A,B còn dùng chung cho cả 32 head.
    Lượt `joint49u`: GDN 0,8→25,2M (mỗi head một cặp), attention giữ nguyên,
    warm-start chính xác, + `--w-entity 3.0`. **Val 500: score 52 (vs 48)
    nhưng gsm8k 3/53 (vs 2/53) — KHÔNG nhích; bfcl tụt 11→5.** Chờ mốc
    1000-2500. Nếu gsm8k vẫn đứng → bác giả thuyết dung lượng, nghi phạm
    chuyển sang **dạng hàm** (`A·S·B` không biểu diễn nổi quan hệ).
  - Bài học thiết kế: lượt `joint49t` hỏng vì tôi cắt attention xuống hạng 64
    để "giữ tổng tham số" → CE nhảy 0,9→5-12. Giả định "attention thẳng hàng
    nên chỉ cần chỉnh nhẹ" đúng cho KHỞI TẠO, sai cho ma trận ĐÃ TRAIN 5.000
    bước (hạng cao).
  - Công cụ mới: `gen_pseudo_vllm.py`, `inspect_fail.py`, `eval_big.py`
    (~1.900 mẫu niêm phong từ dải chưa đụng), `e5.stop_ids()`,
    `Mapper(attn_rank, gdn_per_head)`.

- **ĐANG CHẠY (2026-08-29, user duyệt "ok" + "test ~2000 samples, training
  ~6000 samples")** — `run_bigeval.sh`, 5 pha nối tiếp trên 1 L4, mỗi pha
  idempotent (recycle chỉ cần chạy lại cell):
  1. `ext_bench gen` — dựng lại tập niêm phong CŨ, chỉ để kiểm rò rỉ.
  2. `eval_big gen` — **2000 mẫu niêm phong MỚI**: bbh 700 (hàng 107-250) +
     bfcl 400 (**dưới** các mốc `build_data` đã ăn) + suite 500 (seed 31337,
     train dùng 777) + needle 240 (seed 500017/510023/520031) + musr 60 +
     gsm8k 100 (giữ nhỏ: là họ đang hỏng và tốn 320 token/mẫu).
     Kiểm rò rỉ đối chiếu CHUỖI prompt với train_items.json **lẫn** tập
     `e6v3.build_data()` sinh lúc chạy — thiếu vế sau thì đúng hai họ
     bfcl/needle lọt lưới.
  3. tải `joint49s` (score 67) từ HF.
  4. `eval_big self` bằng vLLM (~20 phút) — cột trần.
  5. **train `joint49v`**: warm-start joint49s, 2500 bước nữa (49s đi
     48→50→61→64→**67**, CHƯA bão hoà), ctx 4096/tbptt 128/gold 256,
     auto-upload HF `joint49v/` mỗi mốc val.
  Pha 6 `run_bigmapped.sh` (mapped 2000 mẫu, ~4h, resume bằng --slice) chạy
  sau khi chọn checkpoint tốt nhất.
- **Bẫy hạ tầng dính lại lần 3**: `ps aux | grep '[r]un_x.sh'` — mẹo ngoặc
  vuông chỉ tránh grep tự khớp chính nó, dòng `bash -c` CHA vẫn chứa nguyên
  chuỗi mẫu → luôn báo "đang chạy". Từ nay dùng LOCK FILE + `kill -0`.

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
