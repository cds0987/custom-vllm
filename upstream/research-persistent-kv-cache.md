# Nghiên cứu khả thi: KV cache bền/chia sẻ (LMCache) cho kịch bản shared-prefix khổng lồ trên Qwen3.5-9B hybrid GDN

Ngày: 2026-08-11. Chỉ nghiên cứu (đọc code/docs), KHÔNG cài đặt/chạy GPU.

## Bối cảnh (số đo đã có, mốc so sánh)

- Cold prefill 30K token: **10,52 s** (TASK H, `STATUS.md:602`).
- Cold prefill 120K token: **62,9 s** (~1.908 tok/s hiệu dụng) (TASK N6, `STATUS.md:663-664`).
- Warm (cache hit trong VRAM): TTFT 0,2–1,0 s, hit rate 99,0–99,4% (TASK H/N6).
- Đã biết: `OffloadingConnector` (CPU offload trong-tiến-trình) chỉ quản KV attention;
  trạng thái mamba/GDN nằm pool riêng, không offload được — bản thân server
  restart vẫn mất sạch cả hai (TASK C2, `STATUS.md:568-588`).

Câu hỏi trung tâm: có cơ chế nào giữ prefix KV **qua restart** hoặc **chia sẻ giữa
nhiều tenant/prefix** mà vẫn đúng với kiến trúc lai (75% GDN + 25% full attention)?

## 1. LMCache — có hỗ trợ model lai không? (bằng chứng, không suy đoán)

**Có, và có recipe riêng cho đúng model Qwen3.5/Qwen3.6.**

- Trang chính thức: [LMCache Hybrid Attention Models](https://docs.lmcache.ai/mp/hybrid_models.html)
  — nguyên văn: *"Models that interleave Mamba / Gated-DeltaNet (GDN) linear-attention
  layers with full attention...are supported."*
- Trang recipe riêng cho đúng family này: [Qwen3.5 / Qwen3.6 series — LMCache](https://docs.lmcache.ai/recipes/qwen3_5.html),
  có lệnh serve cụ thể cho Qwen3.6-27B và Qwen3.5-0.8B.
- Cách LMCache xử lý phần mamba: *"their linear-attention layers keep a recurrent
  state cache (a convolution + SSM state). LMCache reinterprets that state as an
  opaque page at registration time"* → **LMCache lưu/khôi phục CẢ conv-state lẫn
  SSM-state của GDN, không chỉ KV attention** — khác hẳn giới hạn đã thấy ở
  `OffloadingConnector` (TASK C2). Đây là câu trả lời cho câu hỏi sống còn của
  nhiệm vụ: LMCache không mắc lỗi "nạp KV attention mà thiếu state mamba" — họ
  coi state mamba là một loại trang cache riêng (byte-opaque), lưu và nạp cùng cơ chế
  với KV attention, không phải "tính lại phần mamba mỗi lần" — nhưng đọc kỹ thì đây
  là khẳng định của TÀI LIỆU CHÍNH THỨC, chưa có xác nhận thực nghiệm độc lập.

### Cấu hình bắt buộc (từ recipe Qwen3.5/3.6, trích trực tiếp)

```
lmcache server --chunk-size <N> --separate-object-groups \
    --l1-size-gb 100 --eviction-policy LRU

vllm serve Qwen/Qwen3.5-... \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --max-num-batched-tokens <2N-1> \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'
```

- `--chunk-size` (LMCache) PHẢI khớp block size hợp nhất của vLLM (N) — model-specific,
  đọc từ log khởi động vLLM, KHÔNG tự tính (giống hệt bài học cũ của repo này về
  `mamba block_size = 1056` bắt buộc `--max-num-batched-tokens >= 1056`, TASK F2c,
  `STATUS.md:553-556` — cùng một ràng buộc kiến trúc, LMCache chỉ đặt tên khác).
- `--mamba-cache-mode align` bắt buộc — GDN không hỗ trợ mode `all`. Đây CHÍNH LÀ
  mode mà TASK F đã kiểm chứng byte-identical trên checkpoint này (`STATUS.md:414-417`)
  — một điểm cộng cho độ tin cậy.
- `--separate-object-groups` bắt buộc cho hybrid — tách state GDN thành cache-object
  riêng khỏi KV attention.
- **Đây là mode multi-process** (`LMCacheMPConnector`), đòi hỏi chạy một tiến trình
  `lmcache server` RIÊNG cạnh `vllm serve` — không phải chế độ đơn giản
  `LMCacheConnectorV1` in-process.

### Cảnh báo đúng đắn (đọc kỹ, không phải "SAI hoàn toàn" nhưng có giới hạn thật)

Từ cùng trang recipe:
- *"Generation is not bit-exact between a cached and a fresh run [...] under concurrent
  load"* — vì backend kernel GDN **không hỗ trợ batch-invariant mode**, kết quả có thể
  đổi theo thành phần batch. Đây LÀ MỘT GIỚI HẠN THẬT nhưng khác bản chất so với câu hỏi
  gốc: nó là nhiễu số học do kernel (batch-dependent), không phải lỗi "thiếu state".
  Khuyến nghị chính thức: kiểm bằng so sánh điểm số/likelihood, không so token-level
  diff dưới tải đồng thời.
- *"Cached pages for the KDA/GDN and MLA groups are byte-opaque views, so
  content-aware processing (CacheGen, CacheBlend) does not apply"* — không dùng được
  các tối ưu nén/blend cache nâng cao của LMCache trên các layer GDN.
- Tài liệu KHÔNG nói rõ hành vi khi state mamba bị thiếu/hỏng lúc load (evicted một
  phần, lỗi mạng L2...) — không tìm thấy đoạn xử lý fallback tường minh trong docs đã
  đọc. Đây là khoảng trống bằng chứng, không phải xác nhận an toàn.

## 2. Đối chiếu với source vLLM local (`D:\Training\AI_Module\vllm\vllm\vllm`)

Repo vLLM local ĐÃ có sẵn cả hai connector LMCache, đăng ký trong
`vllm/distributed/kv_transfer/kv_connector/factory.py:164-174`:
- `LMCacheConnectorV1` → module `kv_connector.v1.lmcache_connector` (in-process, đơn giản).
- `LMCacheMPConnector` → module `kv_connector.v1.lmcache_mp_connector` (multi-process,
  connector mà recipe Qwen3.5 chính thức dùng).

Phát hiện quan trọng qua đọc code (chưa có trong docs LMCache):

### a) `LMCacheConnectorV1` (mode đơn giản) KHÔNG khai báo hỗ trợ HMA

`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:72` —
`class LMCacheConnectorV1(KVConnectorBase_V1):` — không kế thừa `SupportsHMA`
(so với `class OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` ở
`offloading_connector.py:46`).

`SupportsHMA` là interface đánh dấu connector tương thích với Hybrid Memory
Allocator — cơ chế quản lý riêng KV attention và mamba-state thành nhóm khác nhau
(`vllm/distributed/kv_transfer/kv_connector/v1/base.py:85-90`). Factory check tại
`factory.py:54-60`: nếu HMA bật mà connector không hỗ trợ → lỗi (hoặc tự tắt HMA,
xem dưới).

Ở `vllm/config/vllm.py:1579-1602`: nếu người dùng KHÔNG set tường minh
`--disable-hybrid-kv-cache-manager`, vLLM **tự động tắt HMA** khi connector không
hỗ trợ, kèm log cảnh báo *"Turning off hybrid kv cache manager because
`--kv-transfer-config` selects a KV connector that does not support HMA"*. Nếu
người dùng set tường minh `--disable-hybrid-kv-cache-manager=False` (ép bật) mà
connector không hỗ trợ → raise `ValueError` cứng (`vllm.py:1603-1612`).

Hệ quả với `LMCacheConnectorV1` trên model lai GDN: **HMA bị tắt tự động**. Còn
`unify_hybrid_kv_cache_specs()` (`vllm/v1/core/kv_cache_utils.py:1430-1450`) — hàm
chạy khi HMA tắt để "gộp" các loại KV cache spec khác nhau — chỉ xử lý
`SlidingWindowSpec`/`ChunkedLocalAttentionSpec` → `FullAttentionSpec`
(`kv_cache_utils.py:1479-1511`); **không có nhánh nào xử lý `MambaSpec`**. Nói cách
khác: đọc code không tìm thấy bằng chứng rằng tắt HMA "gộp an toàn" được
Mamba+FullAttention — đường này chưa được xác nhận đúng đắn qua source, RỦI RO,
không nên dùng `LMCacheConnectorV1` (mode đơn giản) trên model lai mà không kiểm
thực nghiệm kỹ.

### b) `LMCacheMPConnector` — nhánh dùng thật khác nhánh có sẵn trong vLLM local

`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py:1200-1223`:
tại import-time, nếu gói `lmcache` (pip) có module
`lmcache.integration.vllm.lmcache_mp_connector`, vLLM **ưu tiên dùng bản đó**
(class `_ExternalLMCacheMPConnector`), chỉ fallback về class builtin
`LMCacheMPConnectorUpstream` định nghĩa ngay trong file này nếu import lỗi.

Class builtin fallback này có một assert rõ ràng:
`lmcache_mp_connector.py:78-81` — hàm `reformat_block_ids()`:
```
if len(block_ids) > 1:
    raise RuntimeError(
        "LMCacheMPConnector only works without hybrid kv cache manager. "
        "Please pass --disable-hybrid-kv-cache-manager when starting vllm"
    )
```
Model lai GDN+FullAttention tạo ≥2 kv_cache_groups khi HMA bật → `block_ids` có
`len>1` → bản fallback builtin **crash thẳng** trên model lai nếu HMA còn bật.
Recipe chính thức của LMCache (mục 1) KHÔNG nhắc `--disable-hybrid-kv-cache-manager`
→ suy ra recipe đó CHỈ chạy đúng khi package `lmcache` (pip, ngoài vLLM) được cài
và phiên bản đủ mới để cung cấp `LMCacheMPConnector` ngoài — bản này (không có
trong repo local, đóng gói riêng trong package `lmcache`) chắc hẳn có khai báo
`SupportsHMA` khác đi, KHÔNG kiểm chứng được bằng đọc source local. Đây là khoảng
trống thật: **báo cáo này không thể xác nhận từ code local rằng LMCacheMPConnector
(external) hỗ trợ HMA đúng đắn — chỉ có tài liệu công bố xác nhận, không có source
để đọc**. Cần cài `lmcache>=0.5.2` (yêu cầu từ thông báo release chính thức, xem
dưới) mới xem được source thật.

### c) Version tương thích vLLM 0.26

Từ thông báo release LMCache v0.5.2 (X/Twitter chính thức của LMCache Lab, tìm qua
tìm kiếm web, không fetch được trực tiếp bài gốc do là social post — trích lại nội
dung index hoá): *"Upgrade to v0.5.2 to run against vLLM ≥ 0.26.0"*. Tức là bản
LMCache cũ hơn v0.5.2 KHÔNG bảo đảm chạy đúng với vLLM 0.26 (đúng bản đang dùng
trong repo này). Cần pin `lmcache>=0.5.2` khi thử nghiệm.

### d) Các connector khác trong factory — không cái nào rẻ hơn cho single-node

`factory.py:148-243` liệt kê toàn bộ: `NixlConnector`/`NixlPull/Push` (RDMA
disaggregated P/D, không phải persist-to-disk), `MultiConnector` (bọc nhiều
connector), `MooncakeConnector`/`MooncakeStoreConnector` (cần cụm Mooncake riêng),
`FlexKVConnectorV1`, `SimpleCPUOffloadConnector` (CPU RAM, cùng hạn chế không-bền
như `OffloadingConnector`), `HF3FSKVConnector` (cần hệ thống file phân tán 3FS của
DeepSeek — hạ tầng multi-node, không hợp lý cho một máy L4 đơn). Trong toàn bộ danh
sách, **LMCache (MP mode) là connector duy nhất có tài liệu xác nhận tường minh hỗ
trợ đúng kiến trúc GDN của model này VÀ có backend đĩa cục bộ đơn giản (POSIX/local
file)**.

`ssm_conv_transfer_utils.py` (toàn file, `vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py`)
xác nhận thêm: vLLM upstream đã có sẵn logic tách sub-projection conv-state riêng
cho `GDN_ATTN`/`Mamba1`/`Mamba2` để truyền qua NIXL (dùng bởi `NixlConnector` cho
disaggregated serving) — bằng chứng độc lập rằng **vLLM upstream coi việc
transfer/persist trạng thái GDN là bài toán đã giải cho ít nhất một connector
(NIXL)**, không phải khoảng trống kiến trúc tuyệt đối. Đây củng cố thêm độ tin cậy
cho khẳng định của LMCache rằng GDN state transfer khả thi.

## 3. Precompute-once-then-reload — có đáng làm không?

Ba lựa chọn, xếp theo rủi ro tăng dần:

### (A) Warm-up script tại boot — RẺ, CHẮC CHẮN, khuyến nghị làm ngay bất kể có LMCache hay không

Gửi 1 request chứa prefix chuẩn ngay sau khi server sẵn sàng (health check pass),
trước khi mở traffic thật. Không cần thay đổi gì trong vLLM/LMCache. Chi phí = đúng
1 lần cold prefill (10,5–62,9 s tuỳ độ dài) mỗi lần restart, xảy ra ngoài đường
critical-path phục vụ user thật. Đã CÓ TIỀN LỆ chính trong repo: TASK H tự mô tả
"Warm-up cold một lần: 10,52s" như một bước vận hành bình thường (`STATUS.md:602`),
tức quy trình này vốn đã ngầm định trong thiết kế hiện tại — chỉ cần đóng gói
thành script chạy tự động lúc khởi động (systemd/entrypoint) thay vì làm tay.
**Không giải quyết được multi-tenant (nhiều prefix khác nhau đá nhau khỏi VRAM)**,
chỉ giải quyết cold-start sau restart cho MỘT prefix chính.

### (B) LMCache MP mode + L2 disk backend — GIẢI QUYẾT ĐÚNG BÀI TOÁN, chi phí hạ tầng + rủi ro đúng đắn chưa kiểm chứng độc lập

Theo [L2 Storage docs](https://docs.lmcache.ai/mp/l2_storage/index.html): backend
`POSIX`/filesystem cục bộ tồn tại (`{"backend": "POSIX", "backend_params":
{"file_path": "/data/lmcache/l2"}}`), dữ liệu "keep them across instance restarts
(the bytes outlive any single reporter)" — tức SỐNG SÓT qua restart, đúng thứ cần.
Nhưng: (a) đòi thêm tiến trình `lmcache server` chạy song song — thêm một điểm
lỗi/vận hành; (b) trang L2 storage KHÔNG nhắc gì đến hybrid/`--separate-object-groups`
khi mô tả L2 — chưa rõ object-group tách biệt mamba-state có tự động persist đúng
xuống L2 hay chỉ path CPU-tier (L1) mới tách; (c) khẳng định "hỗ trợ GDN" nằm ở
tài liệu chính thức, KHÔNG có bằng chứng test độc lập (báo cáo GitHub issue, log
thực nghiệm) được tìm thấy trong phạm vi nghiên cứu này — mọi trích dẫn ở mục 1 đều
từ docs.lmcache.ai do chính dự án LMCache viết.

### (C) Tự dump/load tensor KV thô (hack tự chế) — KHÔNG khuyến nghị

Có thể tự viết code đọc trực tiếp `kv_caches` tensor (giống cách `register_kv_caches`
nhận `dict[str, torch.Tensor]` ở mọi connector) và ghi ra đĩa bằng `torch.save`,
tự nạp lại lúc boot bằng cách ghi thẳng vào block pool trước khi accept request.
Rủi ro rất cao: phải tự đảm bảo block-id mapping, block-hash prefix-cache khớp,
và ĐẶC BIỆT phải tự xử lý đồng bộ conv-state + SSM-state của GDN (2 tensor riêng,
theo `MambaSpec.shapes[0]`/`shapes[1]`, xem `ssm_conv_transfer_utils.py:26-44`) —
đúng lớp phức tạp mà LMCache/NIXL đã viết riêng hẳn một module để xử lý cho đúng.
Tự làm lại việc này ngoài phạm vi "nghiên cứu khả thi" — không đáng, vì (A) đã rẻ
và (B) đã có người làm sẵn nếu chấp nhận thêm hạ tầng.

## 4. Kết luận

**Có-với-điều-kiện.**

- LMCache (đường MP mode, `LMCacheMPConnector` + package `lmcache>=0.5.2`) là
  lựa chọn DUY NHẤT tìm được có tài liệu chính thức xác nhận đúng kiến trúc GDN
  của Qwen3.5, có recipe validate sẵn cho đúng model family, và có backend đĩa cục
  bộ sống sót qua restart.
- Rủi ro cụ thể với model lai: (1) không bit-exact dưới tải đồng thời (do GDN thiếu
  batch-invariant kernel — vốn đã biết là giới hạn kiến trúc sm89, xem
  `STATUS.md:89-93` — không phải bug LMCache mới); (2) mode đơn giản
  `LMCacheConnectorV1` (in-process) RỦI RO trên model lai vì không khai báo
  `SupportsHMA` và code `unify_hybrid_kv_cache_specs` không có nhánh xử lý Mamba —
  PHẢI dùng MP mode, không dùng mode đơn giản; (3) class `LMCacheMPConnector` mà
  vLLM local ưu tiên dùng phụ thuộc hoàn toàn vào package `lmcache` cài thêm — bản
  fallback builtin trong vLLM source KHÔNG chạy được trên model lai
  (raise RuntimeError khi HMA bật, `lmcache_mp_connector.py:78-81`); (4) mọi khẳng
  định "GDN state persist đúng" đến từ tài liệu của chính LMCache, chưa có xác nhận
  độc lập trong nghiên cứu này — CẦN correctness gate riêng trước khi tin.
- Trước khi đầu tư LMCache, **triển khai (A) ngay** — rẻ, không phụ thuộc gì, và đã
  gần như là quy trình ngầm định hiện tại của repo.

### Runbook thử nghiệm gọn (nếu quyết định thử B)

1. Cài `pip install "lmcache>=0.5.2"` (khớp yêu cầu vLLM 0.26) trên môi trường đã
   dựng theo `scripts/setup_env.sh`; xác nhận `lmcache.integration.vllm.lmcache_mp_connector`
   import được (nếu không, vLLM tự fallback về builtin — SẼ crash trên model lai,
   xem mục 2b — coi đây là gate an toàn: nếu package không cài đúng, server nên
   fail-fast chứ không âm thầm chạy nhánh không hỗ trợ).
2. Đọc log khởi động `vllm serve` để lấy block size hợp nhất N thật của checkpoint
   graft/champion hiện tại (đã biết mamba block_size=1056 với config production cũ
   ở TASK F2c — nhưng PHẢI đọc lại cho checkpoint N3/graft-int4 hiện hành, có thể
   khác do config GDN thay đổi).
3. Chạy `lmcache server --chunk-size N --separate-object-groups --l1-size-gb <X>
   --eviction-policy LRU`, rồi `vllm serve ... --enable-prefix-caching
   --mamba-cache-mode align --max-num-batched-tokens <2N-1> --kv-transfer-config
   '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'`.
4. **Cổng đúng đắn trước tiên** (theo đúng luật "không bypass chất lượng" đã dùng
   xuyên suốt STATUS.md): cùng prompt (prefix 30K + suffix cố định), temp=0, so
   sánh output cache-hit (sau khi lưu xuống L2 rồi RESTART CẢ vllm serve LẪN lmcache
   server) với output cold — đúng phương pháp đã dùng ở TASK F/N6. Nếu không
   byte-identical, kiểm bằng so sánh logprob/score (theo đúng khuyến nghị docs
   LMCache) trước khi kết luận PASS/FAIL.
5. Đo TTFT sau restart với cache đã có trên đĩa, ở 30K và 120K — so trực tiếp với
   mốc 10,52 s / 62,9 s. Thắng rõ nếu TTFT-sau-restart tiệm cận mức "warm trong-GPU"
   hiện có (0,2–1,0 s) thay vì phải trả lại phí cold prefill.
6. Đo thêm chi phí phụ: RAM/đĩa dùng bởi L1/L2, thời gian load-from-disk (có thể
   không miễn phí — vẫn phải copy conv-state + SSM-state + KV vào VRAM), và tác
   động lên throughput decode khi `lmcache server` cùng chạy trên máy (cạnh tranh
   CPU/PCIe với engine chính).
7. Nếu (4) hoặc (5) fail, dừng — quay lại phương án (A) đã đủ rẻ và chắc cho use
   case một-prefix-chính; multi-tenant nhiều prefix thì cân nhắc đơn giản hơn:
   nhiều instance vLLM (mỗi instance giữ một prefix nóng trong VRAM) thay vì một
   instance dùng LMCache.

## Nguồn

- [LMCache — Hybrid Attention Models](https://docs.lmcache.ai/mp/hybrid_models.html)
- [LMCache — Qwen3.5 / Qwen3.6 series recipe](https://docs.lmcache.ai/recipes/qwen3_5.html)
- [LMCache — Kimi-Linear recipe](https://docs.lmcache.ai/recipes/kimi_linear.html)
- [LMCache — Local storage backend](https://docs.lmcache.ai/kv_cache/local_storage.html)
- [LMCache — L2 (persistent) storage](https://docs.lmcache.ai/mp/l2_storage/index.html)
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\factory.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\v1\lmcache_connector.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\v1\lmcache_mp_connector.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\v1\base.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\v1\offloading_connector.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\distributed\kv_transfer\kv_connector\v1\ssm_conv_transfer_utils.py`
- `D:\Training\AI_Module\vllm\vllm\vllm\config\vllm.py` (dòng 1547-1616)
- `D:\Training\AI_Module\vllm\vllm\vllm\v1\core\kv_cache_utils.py` (dòng 1430-1511)
- `d:\Training\AI_Module\custom_vllm\STATUS.md` (TASK F, C2, H, N6, và mục "Ngõ cụt trên T4/sm89")
