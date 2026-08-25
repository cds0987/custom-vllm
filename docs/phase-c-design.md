# Phase C — thiết kế đưa cross-model KV transfer vào vLLM serving thật

Ngày 2026-08-25 (user chốt hướng "C"). Nền: E1-E3 (4B→9B copy nguyên CHẠY,
TTFT 30K ×1,66 hai-GPU / ×1,15-1,2 đồng trú), E3B/E3C (đồng trú 2 server),
`upstream/research-persistent-kv-cache.md` (mổ KVConnector/LMCache nội bộ),
E6 v3.1-v3.4 (mapper 4B→27B + template-xương). Source vLLM local:
`D:\Training\AI_Module\vllm\vllm\vllm`.

## Mục tiêu C2 (mốc sản phẩm đầu tiên)

**4B prefill-helper → 9B decode, trên vLLM W4A16 Marlin thật**, giữ nguyên
scope an toàn đã đo (chat/QA/RAG; function-calling để sau vì cần polisher).
Con số bán hàng: TTFT cold-miss prefix dài giảm ×1,5+ mà decode 9B không đổi.

## Kiến trúc chọn: 2 instance vLLM + LMCache MP làm kho trung chuyển

```
[4B vllm serve] --kv_role=kv_producer--> [lmcache server (L1/L2 POSIX)]
                                              |
[9B vllm serve] <--kv_role=kv_consumer-- (key theo TOKEN-HASH, bỏ model-id)
```

- Lý do KHÔNG tự viết connector từ đầu: bài lưu/nạp GDN state (conv+SSM,
  opaque page) LMCache đã giải và vLLM upstream có sẵn máy móc
  (`ssm_conv_transfer_utils.py` — GDN conv=[Q,K,V], temporal=(num_v_heads,
  v_dim,k_dim), dùng thật bởi NixlConnector). Tự chế = làm lại đúng lớp
  phức tạp đó (mục C research cũ đã khuyến cáo bỏ).
- **Điểm vá duy nhất: KEY NAMESPACE.** LMCache đặt key cache theo
  (model, token-prefix-hash, ...). Vá derivation để 4B-producer và
  9B-consumer chia sẻ cùng key (token-hash thuần) → 9B "tưởng" cache của
  mình mà thực ra 4B viết. Hợp lệ VÌ cặp 4B/9B cache-compatible THÔ
  (E1: needle 12/12, decode parity — không cần transform).
- Vá nằm ở package `lmcache` (pip, ngoài repo vLLM) — bản
  `LMCacheMPConnector` external; fallback builtin trong vLLM CRASH trên
  model lai (assert HMA, `lmcache_mp_connector.py:78-81`) → pin
  `lmcache>=0.5.2` và coi import-fail là gate an toàn.

## Tiền đề phải KIỂM trên GPU trước khi code sâu (C2a, ~30 phút)

1. Unified block size N của server 4B và 9B PHẢI bằng nhau (đọc log boot
   từng server; mamba block 1056 đã biết cho 9B config cũ — đọc lại).
2. Shape per-page attn KV + GDN state của 4B == 9B ở mức vLLM (E1 chứng
   minh ở mức transformers; vLLM page layout có thể chèn padding TP/head).
3. `lmcache.integration.vllm.lmcache_mp_connector` import được; serve 9B
   một mình + LMCache chạy sạch (recipe Qwen3.5 chính thức) trước khi
   thêm 4B.
4. Đồng trú 1 L4: dùng combo E3C (util 0,35 + --kv-cache-memory-bytes +
   eager); 2-GPU (2 notebook) là kịch bản giá trị chính — cần lệnh user
   đích danh mới mở notebook B.

## Cổng đúng đắn C2b (thứ tự cứng, luật error-placement)

1. Cùng prompt, temp=0: 9B đọc-cache-4B vs 9B tự prefill — so logprob
   từng vị trí (docs LMCache: GDN kernel không batch-invariant, đừng đòi
   byte-identical dưới tải; MỘT request tuần tự thì phải gần tuyệt đối).
2. Needle 1,5K/8K/30K functional (khuôn E2) qua đường serving thật.
3. Chỉ khi 1+2 pass mới đo TTFT (cold vs cross-warm) và throughput.

## C3 (sau C2): mapper 4B→27B vào cùng khung

Khác biệt bản chất: cache không còn passthrough — cần TRANSFORM khi load.
Điểm cắm: đường đọc L2 của LMCache (hook trước khi trang được ghi vào
VRAM) hoặc connector riêng kế thừa `OffloadingConnector` (đã SupportsHMA).
Template-xương của v3.4 tái dùng nguyên: consumer không cần chạy prefill
27B để có khuôn — dựng từ meta + trang 4B đã map. Mapper chạy fp32 ~35M
tham số, ~0,08s/transplant (E1) — nằm lọt trong ngân sách TTFT.

## Rủi ro & vùng chưa biết (trung thực)

- Patch key-namespace đụng package ngoài — bản quyền/độ sâu API chưa đọc
  (chỉ đọc được sau `pip install lmcache` trên Colab).
- Chunk alignment 4B vs 9B nếu block size khác nhau → phải chọn chunk-size
  chung hoặc re-chunk khi ghi.
- Trang GDN là byte-opaque với LMCache: shape mismatch sẽ KHÔNG được phát
  hiện bởi LMCache — cổng C2b số 1 là lưới duy nhất.
- Mọi khẳng định "LMCache hỗ trợ GDN đúng" vẫn từ docs của chính họ —
  chưa có xác nhận độc lập; C2b chính là xác nhận độc lập đầu tiên.

## Trình tự đề xuất (mỗi bước GPU cần duyệt riêng theo quy tắc)

- C2a: dựng env + serve 9B + LMCache đơn lẻ + đọc log block/shape (GPU ~1h).
- C2b: thêm 4B producer + patch key + 3 cổng đúng đắn (GPU ~2h).
- C2c: bench TTFT/throughput + báo cáo số bán hàng (GPU ~1h).
- C3: thiết kế chi tiết sau khi C2 chốt số.
