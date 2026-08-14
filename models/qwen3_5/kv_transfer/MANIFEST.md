# kv_transfer — Cross-model KV cache transfer 9B↔27B (nghiên cứu, 2026-08-14)

Nguồn: arXiv:2608.03893 "Cross-Model KV Cache Transfer in LLM Families" (NVIDIA,
08/2026) — ridge mapper đóng, training-free, per-head. Paper tại `out/2608.03893v1.pdf`.

## Hiện trạng mã nguồn thế giới (khảo sát 2026-08-14)

- **Paper KHÔNG kèm code chính thức** (đã kiểm text + web).
- vLLM RFC #44223 "Semantic KV Cache Reuse": còn MỞ, chưa có PR merge, và
  **loại trừ cross-model** (chỉ semantic reuse cùng model, connector interface).
- TensorRT-LLM RFC #14918: cùng họ, cùng trạng thái.
- LMCache: lớp lưu trữ/di chuyển KV (đã clone `out/lmcache`) — hạ tầng vận
  chuyển, KHÔNG có mapper cross-model. SemBlend (`out/semblend`): semantic
  provider, không phải mapper.
- ⇒ Muốn áp là TỰ CÀI theo công thức paper → `ridge_mapper.py` (đã viết, 4/4
  unit test pass trên dữ liệu tổng hợp).

## Điều kiện matched-KV cho Qwen3.5 9B↔27B — KIỂM RỒI

| | 9B | 27B | Khớp? |
|---|---|---|---|
| Lớp attention (interval 4) | 8/32 | 16/64 | mapper xử lý lệch số lớp (top-k) ✓ |
| num_key_value_heads | 4 | 4 | ✅ |
| head_dim | 256 | 256 | ✅ |
| GDN v-heads × d | 32×128 | 48×128 | ❌ LỆCH — ngoài phạm vi paper |

Lớp attention khớp Y HỆT điều kiện cặp tốt nhất của paper (Qwen3 14B→32B,
retention 97,6%). Nhưng **3/4 số lớp là GDN mang state đệ quy** — paper tuyên bố
đây là future work. Đây vừa là rủi ro vừa là cơ hội nghiên cứu độc quyền.

## Câu hỏi sống còn (quyết định cả hướng)

**Bao nhiêu phần "trí nhớ ngữ cảnh" nằm trong GDN state so với attention KV?**
Nếu GDN state trống mà model vẫn dùng được context (nhờ 25% lớp attention) →
transfer chỉ-attention đã có giá trị. Nếu không → phải giải bài GDN mapping
(32→48 v-heads, mỗi state 128×128/head, chỉ 1 mẫu/sequence — bài toán fit
khác hẳn, cần dạng bilinear S_t ≈ A·S_s·B hoặc MLP).

→ **E0 ĐÃ CHẠY (2026-08-14, Qwen3.5-4B bf16, 10 trial needle-in-context 1,5K tok):**

    điều kiện            needle    NLL/token
    nguyên vẹn           10/10     0,069
    xóa GDN (giữ KV)     0/10      5,6
    xóa KV (giữ GDN)     0/10      11,8

**PHÁN QUYẾT: context sống ở CẢ HAI — transfer chỉ-attention KHÔNG đủ cho họ
hybrid này; Phase B (GDN mapping) là BẮT BUỘC.** (Hai lần chạy đầu vô hiệu vì
zero không chạm tensor — guard hard-fail đã thêm; cache thật: DynamicCache với
layers[i] là DynamicLayer{keys,values} | LinearAttentionLayer{conv_states,
recurrent_states}.)

Cặp ưu tiên đổi từ 9B↔27B sang **4B↔9B**: GDN shapes trùng hệt (16/32×128),
lớp 1:1 (cùng 32 lớp interval 4), cả hai vừa CÙNG một L4 → use-case cascade
1-GPU thật. Phase B ba nấc: (a) copy nguyên state (shapes trùng, chi phí 0),
(b) ridge per-head theo CỘT state (200 seq × 128 cột = 25,6K mẫu 128-dim — đủ
xác định), (c) bilinear/MLP.

## Kế hoạch phase (GPU L4, notebook A, mọi bước nohup nền)

- **E0** — xoá-GDN-state probe trên 9B (transformers, ~30 phút GPU). Quyết định đi tiếp.
- **A** — thu calibration 2 model (`collect_calib.py`, tuần tự từng model,
  200×1024 stride 4 ≈ 51K token — App. C paper nói N=200 đủ) → fit ridge CPU
  (chỉ 16 lớp đích × 4 head × dim ~4K·k — vài phút, không như 47-87 phút/H100
  của paper vì ta chỉ có lớp attention) → đo R² + ppl với KV ghép.
- **B** (nếu E0 nói GDN quan trọng) — nghiên cứu GDN-state mapping: bilinear/
  MLP 32→48 head; ĐÂY LÀ ĐẤT CHƯA AI CÀY (paper future work).
- **C** — tích hợp serving: bơm mapped-KV vào vLLM. Đường vào: theo dõi RFC
  #44223 connector interface; tạm thời PoC bằng transformers generate với
  past_key_values ghép.

## Nói thật về giá trị trên 1×L4

L4 đơn không giữ nổi 2 model cùng lúc (18,6+9,1GB) — use-case swap-giữa-hội-thoại
trên 1 GPU phải reload model (~140s), lấn át tiền tiết kiệm prefill (~10-30s cho
30K token). **Giá trị thật nằm ở multi-GPU routing** (9B trên máy này, 27B máy
khác — trục Multi-Node của sản phẩm) và ở tri thức bán hàng. Làm trên 1×L4 vẫn
đúng: fit + đánh giá chất lượng không cần 2 model đồng thời.
