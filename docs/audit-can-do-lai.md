# Kiểm toán STATUS.md — cái nào còn tin được, cái nào phải đo lại

Ngày kiểm toán: 2026-08-12. Nguồn: `STATUS.md` (đọc toàn bộ, 1124 dòng),
`docs/debate-attack.md`, `docs/debate-defense.md`.

Lý do làm việc này: chiến dịch đã có ít nhất 6 lần một kết luận "champion"
hoặc "trần vật lý" sụp khi đo lại tử tế (cache thổi phồng số prefill;
closed-loop lệch open-loop tới 6×; thiếu warm-up; mốc cũ khác phiên bản;
prefix tổng hợp 1K đo nhầm kịch bản 30K; các mức tải chồng pha). Ngày
2026-08-12 phát hiện thêm: **môi trường đã trôi từ vLLM 0.26 → 0.27.1**,
nên mọi số đo trước ngày này cần dán nhãn "đo trên phiên bản cũ".

Quy ước ký hiệu bảng chính:
- **Phiên bản**: 0.26 / 0.27.1 / không rõ
- **Bằng chứng**: ĐO THẬT (bench script + số liệu ghi lại) / SUY LUẬN (tính
  từ số khác, đọc code, ngoại suy) / SMOKE (2-5 câu dò, không phải cổng ppl)
- **Điều kiện**: 1-req tuần tự / closed-loop nhiều conc / open-loop Poisson;
  có prefix chung đúng cỡ hay không
- **Rủi ro nếu sai**: CAO (ảnh hưởng lựa chọn champion hoặc cấu hình
  production) / TB / THẤP

---

## a. Bảng đầy đủ

| # | Khẳng định | Nguồn (mục TASK) | Phiên bản | Bằng chứng | Điều kiện | Rủi ro | Phán quyết |
|---|---|---|---|---|---|---|---|
| 1 | Champion 2B = AWQ W4A16 tự tạo + Marlin, decode conc32 = 1859 tok/s | TEST 10 | 0.26 | ĐO THẬT | closed-loop conc32 | CAO | ĐÃ LỖI THỜI (0.27.1 chưa đo lại; chỉ dùng cho 2B, dự án đã chuyển trọng tâm sang 9B) |
| 2 | Champion 9B (RedHatAI W4A16) thắng GGUF Q4_K_M cả 2 trục | TEST 12 | 0.26 | ĐO THẬT | closed-loop conc1-32 + long-ctx 0.1 QPS | CAO | CẦN ĐO LẠI trên 0.27.1 (là gốc của mọi champion sau này) |
| 3 | GRAFT (int8 GDN từ GGUF vào RedHatAI) là champion v1, +8-12% & ppl 0,98 | TASK M | 0.26 | ĐO THẬT | closed-loop conc1/32 + ppl 99 đề | CAO | CẦN ĐO LẠI |
| 4 | **CHAMPION HIỆN TẠI v2 = graft int4**, decode conc32 388,7 tok/s, ppl ratio 0,931, TTFT p95 2,75s | TASK N3 | 0.26 | ĐO THẬT (baseline tươi xác nhận 2 lần) | closed-loop conc1-32 (bench_skills, 74 câu thật) | **CAO NHẤT — đây là con số được dùng làm mốc so sánh cho mọi thứ khác** | CẦN ĐO LẠI ƯU TIÊN 1 — R1 (2026-08-12) đã tự phát hiện mốc nền "1.433 tok/s" của cùng champion này sai gấp đôi trên 0.27.1; 388,7 chưa được xác minh lại |
| 5 | MTP: +28,8% conc1, cấm dùng từ conc8 trở lên (TTFT p95 nổ) | TASK N2 | 0.26 | ĐO THẬT | closed-loop 74 câu, acceptance 85% đo qua /metrics | TB | CẦN ĐO LẠI (đổi runtime FlashInfer/CUDAGraph có thể đổi hành vi) |
| 6 | ngram speculative decoding — LOẠI hoàn toàn, sụp ở conc32 (p95 101,3s) | TASK N5a | 0.26 | ĐO THẬT | closed-loop | THẤP (kết luận là "đừng dùng", sai thì chỉ mất cơ hội chứ không phá production) | CÒN TIN ĐƯỢC (biên độ thất bại quá lớn để đảo chiều bởi 1 bản vá minor) |
| 7 | lm_head int8 — bất khả thi (ParallelLMHead không có code path WNA16) | TASK N5b | không rõ (đọc code, không phụ thuộc version thực thi) | SUY LUẬN từ đọc source + 1 lần serve chết | 1-req | THẤP | CÒN TIN ĐƯỢC (giới hạn kiến trúc, không phải benchmark) |
| 8 | async-scheduling mặc định ON, giữ nguyên, +6,6-7,6% throughput | TASK N5c | 0.26 | ĐO THẬT | closed-loop conc16/32 | THẤP | CẦN ĐO LẠI NHẸ (rẻ, nhưng rủi ro thấp vì chỉ là "giữ mặc định") |
| 9 | 128K context: correctness PASS byte-identical; cascade KHÔNG khả dụng (2 khoá độc lập, 1 cái hardcode `return False`) | TASK N6 | 0.26 | ĐO THẬT (correctness) + SUY LUẬN đọc code (cascade) | 1-req cho correctness; closed-loop conc1/4/16 cho decode | TB | Correctness CÒN TIN ĐƯỢC (đọc code không đổi theo phiên bản trong ngắn hạn); số decode CẦN ĐO LẠI |
| 10 | Offline batch mode +34,8% vs server, trần thực ~500-520 tok/s | TASK P1 | 0.26 | ĐO THẬT | batch, không server | TB | CẦN ĐO LẠI (baseline server 387,8 đã lung lay — xem #4) |
| 11 | Kernel GDN chunk=32 thắng chunk=64 +7,6% ở conc1, trung tính conc32 | TASK P2 | 0.26 | ĐO THẬT | closed-loop conc1/32 | TB | **ĐÃ TỰ GHI "CÒN NỢ" trong STATUS — CẦN ĐO LẠI trên 0.27.1** (dự án tự nhận) |
| 12 | chunk=16 trung tính, đóng nhánh chunk-nhỏ | TASK R2 | 0.27.1 (đã đo trên bản mới) | ĐO THẬT | closed-loop conc1/32 | THẤP | CÒN TIN ĐƯỢC (đo đúng trên 0.27.1, đối chứng cùng runtime) |
| 13 | fp8_per_tensor phá trần prefill +34-38% so với champion int4 | TASK R1 | **0.27.1** | ĐO THẬT | 1-req tuần tự (prefill thuần, không phải nhiều request đồng thời) | CAO | CÒN TIN ĐƯỢC nhưng PHẠM VI HẸP — chỉ prefill 1-request; chưa đo dưới tải đồng thời thật |
| 14 | fp8 thua champion ở mọi mức decode conc1-32 (14-47%) và ppl (WARN 1,142) | TASK R1b | 0.27.1 | ĐO THẬT | closed-loop conc1-32 | CAO | CÒN TIN ĐƯỢC (đo mới, cùng runtime với #13) |
| 15 | Mốc nền "trần prefill 1.433 tok/s" — SAI, số thật là 2.789-2.934 (gấp đôi) | DRIFT / TRANH LUẬN VÒNG 1 | 0.26 (số cũ) vs 0.27.1 (số mới) | ĐO THẬT (số mới) | 1-req | CAO | Đã tự phán quyết ĐÃ LỖI THỜI trong chính STATUS — dùng số mới (2.789-2.934), nhưng MỌI con số dung lượng suy ra từ 1.433 (kể cả trong TASK Q2, F, N6) cần tính lại |
| 16 | Toàn bộ mốc số của chiến dịch (champion 388,7; ppl 4,778; SLA 0,3 QPS; P1 522) đo trên 0.26, không tương thích trực tiếp với 0.27.1 | DRIFT MÔI TRƯỜNG | Explicit cảnh báo tự ghi | — | — | **CAO** | Đây là phán quyết KHUNG cho toàn bộ audit — mọi dòng phía trên đề "0.26" nên đọc là CẦN ĐO LẠI trừ khi có ghi chú khác |
| 17 | 2 patch nhà làm nay dư thừa vì upstream đã tự sửa (`patch_vllm_qwen35_hybrid`, `patch_gguf_override_signature`) | DRIFT MÔI TRƯỜNG | 0.27.1 | ĐO THẬT (patch chạy no-op, xác nhận bằng log) | 1-req khởi động | THẤP | CÒN TIN ĐƯỢC |
| 18 | TASK Q2: sessions=1 là điểm duy nhất giữ TTFT p95<3s; sessions=8 là điểm vận hành tối ưu theo tasks/hr | TASK Q2/Q2b | **0.27.1** (2026-08-12, ngày mới nhất) | ĐO THẬT | closed-loop nhiều session đồng thời, open-loop-ish (mỗi session tự schedule lượt) | CAO | CÒN TIN ĐƯỢC — đo mới nhất và methodology đã tự sửa lỗi chồng pha (Q2b) |
| 19 | Prefix 30K chia sẻ thật, khớp mô hình toán học 4/4 điểm (Q2b mục 1) | TASK Q2b | 0.27.1 | ĐO THẬT (đối chiếu KV usage đo được vs dự đoán) | closed-loop, đo riêng từng mức không chồng pha | THẤP | CÒN TIN ĐƯỢC |
| 20 | Scheduler reserve theo worst-case max-model-len → hạ max-model-len tăng ~1,6× dung lượng | TASK Q2b mục 2 | 0.27.1 | SUY LUẬN từ số đo (chưa tự đo trực tiếp — tự ghi "Chưa đo — xem Q2c") | — | TB | **CHƯA BAO GIỜ ĐƯỢC ĐO** — vẫn dùng để định hướng tối ưu tiếp theo |
| 21 | SLA shared-prefix: 0,2 req/s sạch, 0,3 là mép mềm; tune mnbt→1088 nâng lên 0,3 sạch | TASK F2/F2b/F2c | 0.26 | ĐO THẬT | **open-loop Poisson** (đúng phương pháp) | CAO | CẦN ĐO LẠI trên 0.27.1 (là cơ sở của cờ `--max-num-batched-tokens 1088` đang dùng làm mặc định production) |
| 22 | mnbt=1088 là "sàn cứng" vì mamba block_size=1056 chặn assert | TASK F2c | 0.26 | ĐO THẬT (assertion lúc khởi động, không phải hiệu năng) | 1-req khởi động | THẤP | CÒN TIN ĐƯỢC (ràng buộc cấu trúc model, không phụ thuộc benchmark runtime) trừ khi model/mamba block đổi |
| 23 | TASK H/skills-pack: 74/74 câu sạch, hit rate 99,04-99,37%, TTFT p95 sát 3s ở conc32 | TASK H + re-bench GRAFT | 0.26 | ĐO THẬT | closed-loop | CAO | CẦN ĐO LẠI (benchmark production-representative nhất, nhưng trên 0.26) |
| 24 | CPU KV offload: mua context/coverage, KHÔNG mua số phiên (trần mamba block) | TASK C2 | không rõ, code đọc là chính | SUY LUẬN đọc code + 1 lần smoke | 1-req smoke | TB | CHƯA BAO GIỜ ĐƯỢC ĐO thực nghiệm dưới tải — tự ghi "chưa sweep thực nghiệm" |
| 25 | TASK Q1: cache sống dai 60s không bị thu hồi, TTFT nối lại phẳng ~0,33s | TASK Q1 | 0.26 | ĐO THẬT | 1 session tuần tự các lượt (không multi-user) | TB | CẦN ĐO LẠI trên 0.27.1; VÙNG MÙ concurrency (chỉ test 1 phiên tại 1 thời điểm) |
| 26 | Đánh đổi context: max concurrency 9,66×@32K → 5,80×@64K | TASK Q1 | 0.26 | ĐO THẬT (đọc trực tiếp từ log KV pool) | — | TB | CÔNG THỨC TOÁN, không phụ thuộc runtime nhiều — CÒN TIN ĐƯỢC ở dạng tỷ lệ, cần đo lại giá trị tuyệt đối |
| 27 | GPTQ nén GDN (TASK G) nhanh nhất (+36,5% conc32) nhưng ppl WARN 1,15-1,16 — không phong champion | TASK G/G2a | 0.26 | ĐO THẬT | closed-loop | TB (đã bị GRAFT vượt qua nên rủi ro production thấp) | CÒN TIN ĐƯỢC làm bài học phương pháp, nhưng số tuyệt đối lỗi thời |
| 28 | Auto-marlin (TASK K/K2-K4): cơ khí ĐẠT, đúng đắn CHƯA (completion rác) | TASK K | 0.26 | ĐO THẬT (thất bại quan sát trực tiếp) | 1-req smoke | THẤP (đã đóng nhánh) | CÒN TIN ĐƯỢC (là trạng thái treo, không phải kết luận tích cực bị rủi ro) |
| 29 | fastcalib (TASK L) 10-11× nhanh hơn nhưng ppl WARN — chỉ dùng "chế độ nháp" | TASK L | 0.26 | ĐO THẬT | 1 lần | THẤP | CÒN TIN ĐƯỢC |
| 30 | FlashInfer là backend duy nhất hợp lệ với fp8 KV; FLASH_ATTN raise lỗi | A/B attention backend | 0.26 | ĐO THẬT (lỗi khởi động, không phải hiệu năng) | 1-req khởi động | THẤP | CÒN TIN ĐƯỢC (ràng buộc tương thích, ít khả năng đổi giữa minor version) |
| 31 | KV dưới 8-bit bất khả thi trên MỌI GPU Ada — 3 nguồn độc lập đồng quy | mục "Ngõ cụt T4" | không rõ (dựa trên tài liệu bên thứ 3: TRT-LLM, turboquant-vllm, CUTLASS) | SUY LUẬN từ tài liệu ngoài, không tự đo trên stack này | — | THẤP (chỉ đóng một hướng nghiên cứu, không cấu hình production) | CÒN TIN ĐƯỢC (bằng chứng bên ngoài mạnh, không phụ thuộc runtime nội bộ) |
| 32 | Trần điện 72W (Colab T4/L4) là ràng buộc cứng | TRANH LUẬN VÒNG 1 Trần "điện" | — | SUY LUẬN (TDP thiết kế + không có quyền root) | — | THẤP | CÒN TIN ĐƯỢC (vật lý phần cứng, không đổi theo phiên bản phần mềm) |
| 33 | Cascade attention đóng vĩnh viễn — lý do THẬT là lợi ích <1% + rủi ro TREO dưới tải đồng thời (PR #26130 upstream) | TRANH LUẬN VÒNG 1 Trần 3 | không rõ version cụ thể của PR fix | SUY LUẬN (đọc PR upstream + tính % thời gian attention) | — | TB | CÒN TIN ĐƯỢC — nhưng đi kèm cảnh báo methodology quan trọng (xem mục c bên dưới) |
| 34 | Profile kernel: GDN không phải nút thắt (2-12% CUDA time); q6_k_gemm_kernel chiếm 34,3% decode | mục "Profile kernel trên L4" | 0.26 | ĐO THẬT (torch.profiler) | 1-req | TB | CẦN ĐO LẠI (nếu kernel dispatch hoặc torch/CUDA đổi theo 0.27.1) |
| 35 | Bảng định dạng GGUF (Q4_0 1158 tok/s ... BF16 hỏng) | mục "Bảng định dạng GGUF" | 0.26 | ĐO THẬT (đa số) + SMOKE (cột chất lượng) | closed-loop conc32 | TB | CẦN ĐO LẠI; cột chất lượng là smoke 1-2 câu, không phải cổng ppl |
| 36 | Kiểm chứng scale 9B: decode theo byte weights (lệch 9%); prefill scale kém tuyến tính | mục "Kiểm chứng scale 9B" | 0.26 | ĐO THẬT (decode) / SUY LUẬN ngoại suy (prefill, tự ghi "cần bài duration ở mép trước khi chốt") | closed-loop + 1 điểm 0.1 QPS | TB | Prefill: CHƯA BAO GIỜ ĐƯỢC ĐO đầy đủ (tự ghi rõ) |
| 37 | Thang bit thấp 9B: Q4_K_M là SÀN, dưới nữa mất cả tốc độ lẫn chất lượng | TEST 9 | 0.26 | ĐO THẬT (tốc độ) + probe 5-biến-thể có control (chất lượng, đã tự sửa methodology sau vụ "trái cây") | closed-loop conc32 | CAO (quyết định sàn chất lượng production) | CẦN ĐO LẠI (0.27.1) nhưng methodology chất lượng (control-validate) đáng tin, chỉ cần refresh số tốc độ |
| 38 | Hai instance (chat/tài liệu tách biệt) thắng MPS 6-15× ở TTFT/ITL max | mục "Cô lập chat khỏi prefill" | 0.26 | ĐO THẬT | closed-loop, 1 phía chịu tải trong khi phía kia chạy | TB | CẦN ĐO LẠI |
| 39 | TASK F: shared-prefix 32K/65K — correctness byte-identical, hit rate 95,3%, TTFT warm ~1,4s | TASK F | 0.26 | ĐO THẬT | closed-loop conc8/16/32 + 1-req correctness | CAO | CẦN ĐO LẠI (là nền tảng thiết kế "front-load system-prompt" đang dùng production) |
| 40 | Prefix caching KHÔNG thổi phồng số ở điểm champion (TEST R xác nhận lại) | mục "Bẫy đo lường" #5 | 0.26 | ĐO THẬT (đối chứng có/không tag chống-cache) | open-loop | THẤP (đã tự phát hiện + tự sửa bug + tái kiểm) | CÒN TIN ĐƯỢC (methodology mẫu mực, chỉ số tuyệt đối 11.211,6 vẫn là 0.26) |
| 41 | Patch dispatch 3 đường (HYBRID+TRITON_MID) giữ 98% decode fused + gần full prefill dequant | mục "Patch lai kernel-dispatch" | 0.26 | ĐO THẬT | closed-loop + open-loop | TB | CẦN ĐO LẠI nếu còn dùng route GGUF (nhánh phụ, champion hiện tại không phải GGUF thuần) |
| 42 | CUSTOM_VLLM_GGUF_REPACK khoá cứng vĩnh viễn (nguy cơ tính sai âm thầm) | mục "REPACK ngõ cụt" | 0.26 | SUY LUẬN (root-cause bằng số học tĩnh, KHÔNG chạy được trên GPU thật vì máy dev không có GPU) | — | THẤP (đã khoá an toàn) | CÒN TIN ĐƯỢC như quyết định an toàn; nguyên nhân gốc kỹ thuật vẫn CHƯA BAO GIỜ ĐƯỢC XÁC NHẬN RUNTIME |
| 43 | transcode_gguf_to_gptq.py pass dry-run CPU-only | mục "Cập nhật" cuối GGUF | — | SMOKE (dry-run CPU, không GPU) | — | THẤP | CHƯA BAO GIỜ ĐƯỢC ĐO trên GPU thật — tự ghi rõ |

---

## b. Danh sách ĐO LẠI có ưu tiên (rủi ro ÷ chi phí)

Xếp giảm dần theo mức độ "đáng đo trước". Lệnh cụ thể dùng script có sẵn
trong `scripts/`; thời lượng ước tính theo kinh nghiệm các lượt trước ghi
trong STATUS.md.

### P1 — Champion v2 (graft int4) trên 0.27.1, closed-loop cơ bản
- **Đo gì**: decode conc1/8/16/32 + ppl 99 đề, đúng cấu hình production
  (mnbt 1088, fp8 KV, prefix caching, align mode) — tái lập số 388,7 tok/s
  / ppl ratio 0,931 / TTFT p95 2,75s.
- **Lệnh**: `python scripts/bench_skills.py --conc 1 8 16 32 ...` (theo
  cấu hình production đã ghi ở TASK H/N3) rồi `python
  scripts/eval_quality_swebench.py` (hoặc script ppl 99 đề tương ứng đã
  dùng ở TASK N3/R1b).
- **Thời lượng**: ~20-30 phút (tái sử dụng checkpoint graft đã dựng).
- **Xác nhận**: decode conc32 trong biên ±10% của 388,7 VÀ ppl ratio vẫn
  <1,10 (PASS) → giữ nguyên champion. **Bác bỏ**: lệch >15% theo hướng nào
  cũng đủ để dán nhãn lại "ĐO TRÊN 0.27.1: X tok/s" và viết lại phần kết
  luận chiến dịch.

### P2 — Baseline prefill "trần vật lý" đã tự phát hiện sai (R1)
- **Đo gì**: xác nhận số 2.789-2.934 tok/s (không phải 1.433) là số đúng
  và ổn định, không phải artifact của cấu hình đo hôm đó.
- **Lệnh**: `python scripts/bench_serving.py run --rates 0.1 0.3 0.5 --duration 120` với
  cờ tắt prefix-caching, đúng payload prompt dài đã dùng ở TASK F/R1.
- **Thời lượng**: ~10-15 phút.
- **Xác nhận**: số lặp lại trong khoảng 2.700-3.000 tok/s ở ≥2 lượt độc
  lập. **Bác bỏ**: quay về vùng ~1.400 → nghi cấu hình đo khác nhau
  (chunk/mnbt), cần soát lại tham số.

### P3 — SLA shared-prefix (mnbt=1088, F2c) trên 0.27.1
- **Đo gì**: xác nhận 0,3 req/s vẫn là mép SLA sạch (TTFT p95<3s) sau khi
  đổi version — đây là cấu hình production hiện dùng.
- **Lệnh**: `python scripts/bench_sla_prefix.py` (payload prefix-chung +
  suffix-unique, rate sweep 0,2/0,3/0,5, open-loop) — dùng đúng thông số
  F2b/F2c: prefix 30K, suffix 2000-2500 tok, output 400 tok.
- **Thời lượng**: ~20-30 phút (3 mức rate × vài phút mỗi mức + warm-up).
- **Xác nhận**: TTFT p95 ở 0,3 req/s vẫn <3s (vi phạm ≤10%). **Bác bỏ**:
  vi phạm tăng rõ rệt (>20%) → hạ khuyến nghị vận hành xuống 0,2 req/s.

### P4 — TASK Q2b điểm "sessions=8 là đỉnh tasks/hr" trên tải đồng thời thật hơn
- **Đo gì**: tái lập bảng tasks/hr theo sessions {1,4,8,16} — đây là kết
  luận mới nhất (2026-08-12, đã ở 0.27.1) nên rủi ro phiên bản THẤP, nhưng
  rủi ro production CAO (là khuyến nghị "8 phiên" đang dùng) và mẫu đo còn
  ít (1 lượt Q2b).
- **Lệnh**: `python scripts/bench_agent_loop.py --sessions 1 4 8 16
  --synthetic-prefix-tokens 30000 --scenarios light medium heavy` (đo
  từng mức riêng lẻ, KHÔNG chồng pha — đúng bài học Q2b).
- **Thời lượng**: ~40-60 phút (đã có ở lượt Q2b làm mốc).
- **Xác nhận**: đỉnh tasks/hr vẫn rơi vào sessions=8 (±1 mức lân cận).
  **Bác bỏ**: đỉnh dịch chuyển rõ rệt (vd về 4 hoặc 16) → viết lại khuyến
  nghị vận hành.

### P5 — GRAFT/skills-pack production run (TASK H) trên 0.27.1
- **Đo gì**: 74/74 câu sạch, hit rate ~99%, TTFT p95 theo conc — kịch bản
  production gần thực tế nhất trong toàn dự án.
- **Lệnh**: `python scripts/bench_skills.py --conc 1 8 16 32` với đúng
  flags production (mnbt1088, prefix caching, align, fp8 KV, max-model-len
  32768).
- **Thời lượng**: ~15-20 phút.
- **Xác nhận**: 74/74 vẫn sạch VÀ hit rate ≥95%. **Bác bỏ**: bất kỳ lỗi
  request nào xuất hiện, hoặc hit rate rơi xuống <90% (nghi cache-hit
  hành vi đổi theo phiên bản).

### P6 — chunk GDN=32 vs 64, tái kiểm với bench tổng hợp đầy đủ hơn (STATUS tự ghi "còn nợ")
- **Đo gì**: xác nhận +7,6% conc1 của chunk=32 còn đúng trên 0.27.1 +
  dữ liệu thật (không chỉ dữ liệu tổng hợp).
- **Lệnh**: vá `FLA_CHUNK_SIZE` tại
  `vllm/third_party/flash_linear_attention/ops/utils.py` (dò lại vị trí
  trên 0.27.1 trước — đường dẫn cũ đã đổi), rồi `python
  scripts/bench_skills.py --conc 1 32`, revert patch sau đo.
- **Thời lượng**: ~10-15 phút (rẻ, dự án tự ghi "gần như miễn phí").
- **Xác nhận**: decode conc1 tăng >5% so với chunk=64. **Bác bỏ**: trong
  nhiễu (<2%) → đóng hẳn nhánh chunk-tuning, không chỉ "trung tính".

### P7 — MTP dưới tải thấp (conc≤4) trên 0.27.1
- **Đo gì**: xác nhận +28,8% conc1 vẫn đúng và TTFT vẫn <1s ở conc≤4 —
  đây là công tắc vận hành đang khuyến nghị bật có điều kiện.
- **Lệnh**: `vllm serve ... --speculative-config '{"method":
  "qwen3_5_mtp","num_speculative_tokens":1}'` rồi `bench_skills.py --conc 1 4 8`.
- **Thời lượng**: ~15 phút.
- **Xác nhận**: throughput +>20% ở conc1, TTFT p95 vẫn <1s ở conc4.
  **Bác bỏ**: TTFT p95 vượt 1s sớm hơn (ở conc4 thay vì conc8) → thu hẹp
  ngưỡng bật MTP xuống conc≤1-2.

### P8 (thấp ưu tiên, chi phí thấp) — async-scheduling A/B trên 0.27.1
- **Đo gì**: xác nhận ON vẫn thắng OFF +6-7% (hiện là mặc định, không cần
  đổi cấu hình gì nếu đúng).
- **Lệnh**: `bench_skills.py --conc 16 32` với `--async-scheduling` /
  `--no-async-scheduling`.
- **Thời lượng**: ~10 phút.
- **Xác nhận/bác bỏ**: chỉ mang tính xác nhận mặc định, rủi ro thấp dù
  sai (không đổi hành vi production vì ON đã là default).

---

## c. Vùng mù — kết luận chỉ được kiểm ở CHẾ ĐỘ MỘT REQUEST

Tự phát hiện trong TRANH LUẬN VÒNG 1 (Trần 3): cổng byte-identical hiện tại
(TASK F, TASK N6) chỉ gửi **một request tuần tự** để so sánh cache-hit vs
cold, nên **không thể bắt lỗi loại "treo dưới tải nhiều request đồng thời +
prefix cache hit"** — đúng loại lỗi mà PR upstream #26130 (cascade
attention) mô tả nhưng chính tác giả cũng chưa truy ra nguyên nhân gốc.

Các kết luận correctness sau đây nằm trong vùng mù đó — chúng ĐÚNG cho
1 request, nhưng CHƯA được kiểm dưới nhiều request đồng thời tranh chấp
cùng prefix cache:

- TASK F: "output cache-hit byte-identical với output cold ở 32K và 65K"
  (dòng #16 bảng trên) — kiểm bằng 1 request tại một thời điểm.
- TASK N6: "CORRECTNESS GATE PASS ở 128K, byte-identical" — cùng cách kiểm.
- TASK Q1: "cache không bị thu hồi trong 60 giây" — đo bằng 1 session lặp
  lại các lượt, không phải nhiều session đồng thời tranh chấp cache.
- Mọi khẳng định "hybrid GDN + APC align-mode đúng, không lỗi
  state-restore" — chỉ đúng cho luồng 1-request; align-mode tự nhận vẫn
  "gắn nhãn experimental".

**Khuyến nghị**: trước khi coi các cấu hình trên là an toàn tuyệt đối cho
production nhiều người dùng, cần thêm một cổng kiểm correctness chạy N
request đồng thời (N≥8-16) cùng chia sẻ prefix, đối chiếu từng output với
bản 1-request tương ứng — hiện chưa có script nào trong `scripts/` làm
việc này (bench_skills.py và bench_agent_loop.py đo hiệu năng/hit-rate,
không đối chiếu byte-identical dưới đồng thời).

---

## d. Những chỗ CHƯA BAO GIỜ ĐƯỢC ĐO nhưng vẫn dùng để ra quyết định

1. **TASK Q2b mục 2** — "hạ max-model-len xuống đúng nhu cầu (~40K thay vì
   65536) sẽ tăng dung lượng đồng thời ~1,6×": tự ghi rõ "Chưa đo — xem
   Q2c". Không có TASK Q2c nào xuất hiện trong STATUS.md — nhánh bị bỏ dở.
   Đây là một đòn tối ưu được đề xuất nhưng KHÔNG được xác nhận.
2. **TASK C2 (CPU KV offload)** — toàn bộ mục "Trần kiến trúc" là suy luận
   từ đọc code (`build_offloading_config.py`, `single_type_kv_cache_manager.py`),
   chỉ có "smoke sạch trên 9B W4A16" — không có sweep thực nghiệm đo số
   phiên/độ trễ thật dưới offload. Tự ghi "chưa sweep thực nghiệm".
3. **transcode_gguf_to_gptq.py** — chỉ pass dry-run CPU-only, chưa từng
   chạy trên GPU (serve/bench/quality gate) — tự ghi rõ "việc còn lại cần
   GPU L4 thật".
4. **TASK G/G2a giả thuyết "GPTQ-Hessian trên GDN là thủ phạm ~15% ppl"**
   — TASK L tự ghi lại hạ cấp giả thuyết này xuống "chưa cô lập biến" vì
   thiếu một baseline calib chuẩn không-GDN để so sánh 3 chiều; kết luận
   GRAFT (0,98) vẫn đúng độc lập với cách diễn giải, nhưng cơ chế gốc chưa
   xác nhận.
5. **CUSTOM_VLLM_GGUF_REPACK root-cause** — quyết định khoá cứng là đúng
   (an toàn), nhưng nguyên nhân kỹ thuật chính xác (desync tag/data ở đâu)
   được suy ra hoàn toàn từ đọc mã tĩnh, "máy dev này không có GPU nên
   không dò runtime được" — tự ghi rõ trong docstring.
6. **Prefill scale 9B "kém tuyến tính hơn dự đoán FLOPs"** — điểm đo sạch
   duy nhất là 0.1 QPS, tự ghi "có thể chưa chạm trần thật... cần một bài
   duration ở mép 0.15-0.2 QPS trước khi chốt số production" — chưa chạy.

---

## Nếu chỉ được đo lại 5 thứ thì đo gì

1. **Champion v2 (graft int4) decode + ppl trên 0.27.1** (P1). Đây là con
   số neo mọi so sánh khác trong toàn dự án (388,7 tok/s, ppl 0,931); nếu
   nó lệch trên phiên bản mới, mọi bảng "X thắng/thua champion Y%" phía
   sau đều cần viết lại. Rủi ro cao nhất, và cùng lúc rẻ vì checkpoint đã
   dựng sẵn.
2. **Baseline prefill 2.789-2.934 tok/s** (P2). Chính STATUS.md gọi đây là
   "chấn động phụ" — con số nền dùng để lập luận "trần vật lý" suốt chiến
   dịch sai gấp đôi. Nếu số này lại thay đổi lần nữa trên đo lặp, toàn bộ
   lý luận về prefill (bao gồm quyết định thử fp8/int8) phải xét lại.
3. **SLA shared-prefix ở mnbt=1088, rate 0,3 req/s** (P3). Đây là cấu
   hình đang dùng làm khuyến nghị production trực tiếp (không phải một
   con số học thuật) — sai ở đây nghĩa là SLA production bị vi phạm thật
   ngoài đời.
4. **Correctness dưới nhiều request đồng thời cùng prefix cache** (mục c
   — cần viết script mới, không nằm trong danh sách P1-P8 vì đó là công
   cụ chưa tồn tại). Đây là vùng mù duy nhất có thể gây lỗi âm thầm
   (sai dữ liệu, không phải chỉ chậm) trong production nhiều người dùng —
   rủi ro cao nhất về BẢN CHẤT dù chưa ai đo được biên độ.
5. **TASK Q2b "sessions=8 là đỉnh"** (P4). Là khuyến nghị vận hành cụ thể
   nhất (số phiên đồng thời cho phép) và mới chỉ có ĐÚNG MỘT lượt đo sau
   khi sửa lỗi chồng pha — mẫu quá mỏng cho một con số sẽ được dùng để
   cấu hình giới hạn concurrency thật.
