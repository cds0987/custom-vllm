# Cẩm nang phục vụ vòng lặp agent (nhiều user đồng thời)

Phạm vi: workload đọc input → gọi tool → đọc kết quả tool → gọi tool → ... → trả lời,
phục vụ nhiều user đồng thời, trên hệ đã tối ưu của dự án (Qwen3.5-9B lai GDN,
vLLM 0.26, 1×L4 23GB, champion v2 = graft int4). Mọi con số trích từ `STATUS.md`
(đường dẫn/mục nêu rõ từng chỗ) hoặc đọc trực tiếp source vLLM local
(`D:\Training\AI_Module\vllm\vllm\vllm`, dẫn `file:dòng`). Kịch bản chưa có số đo
ghi rõ **CHƯA ĐO** kèm tên thí nghiệm cần chạy — không suy diễn số liệu.

Điểm mấu chốt cần hiểu trước khi đọc 20 kịch bản dưới: **một "phiên agent" không
tồn tại trong vLLM như một khái niệm sống**. Mỗi lượt (gọi model → nhận tool_calls
→ chạy tool ngoài → gọi lại model) là một HTTP request **độc lập**; cái duy nhất nối
chúng lại là prefix-cache KV còn nằm trong VRAM giữa hai request đó. vLLM không biết
gì về "agent", "tool", hay "phiên" — nó chỉ thấy chuỗi request rời rạc và một pool
block KV/mamba dùng chung LRU. Toàn bộ 20 kịch bản dưới đây là hệ quả của sự thật
đơn giản này.

## Nhóm A — Bộ nhớ & cache

### (1) Phiên nằm im chờ tool bị evict, nối lại phải tính lại toàn bộ lịch sử

**Mặc định.** Khi request của lượt N kết thúc, các block KV của nó có `ref_cnt` về 0
và bị đẩy vào **hàng đợi LRU toàn cục** dùng chung cho MỌI request/tenant —
`BlockPool.free_blocks()` (`vllm/v1/core/block_pool.py`): block có hash (tức đã
cache được) vào **đuôi** hàng đợi (`free_block_queue.append_n`), block không hash
vào đầu (evict trước). `get_new_blocks()` → `_maybe_evict_cached_block()` chỉ evict
**khi có allocation mới cần chỗ** — không có timer, không có TTL. `touch()` chỉ bảo
vệ block trong lúc nó đang được một request tham chiếu (`ref_cnt>0`); ngay khi lượt
kết thúc, bảo vệ đó biến mất. Không có cơ chế pin/priority theo phiên hay tenant nào
trong `block_pool.py`. Vì vậy: khoảng chờ tool 5-30s **tự nó không nguy hiểm** —
nguy hiểm là các phiên/tenant KHÁC tiêu thụ đủ pool trong lúc đó để chạm tới đúng
block ở đuôi LRU của phiên đang chờ.

**Cách phát hiện.** Không có tín hiệu per-session built-in. Theo dõi gián tiếp:
TTFT của lượt kế tiếp trong cùng phiên bất ngờ nhảy từ mức "warm" (0,2-1,0s @30K,
TASK H/N6) về gần mức "cold" (10,5s @30K, 62,9s @120K) dù prompt giống hệt lượt
trước + suffix mới — dấu hiệu prefix của chính phiên đó (không phải skills-pack
chung) đã bị đá khỏi VRAM. Ứng dụng nên tự log token count + độ trễ chờ tool mỗi
lượt để phân biệt "chờ tool lâu" với "TTFT model tăng".

**Cách xử lý (rẻ → đắt).** (a) Rút ngắn round-trip tool khi có thể (không tự vLLM
sửa được). (b) Tăng `--gpu-memory-utilization` để pool lớn hơn, chịu được nhiều
working-set song song hơn trước khi LRU chạm tới phiên đang chờ. (c) Nếu tool
thường xuyên chậm > vài chục giây và traffic đông, cân nhắc gửi 1 request "giữ nóng"
(no-op, max_tokens nhỏ) lặp lại prompt hiện có của phiên trong lúc chờ — đắt về băng
thông, chỉ đáng khi TTFT-lượt-sau quan trọng hơn phí tổn. (d) `LMCache` MP mode
persist-to-disk (Nhóm E-20) không giải bài này — nó cứu qua *restart*, không cứu
tranh chấp LRU giữa các phiên đang sống.

**Thí nghiệm chứng minh.** CHƯA ĐO. Cần `scripts/bench_agent_loop.py` (chưa viết,
xem mục cuối) với cờ đề xuất `--tool-think-time <dist>` (phân phối thời gian chờ
tool mỗi lượt) và `--competing-noise-sessions N` (traffic nền cạnh tranh pool),
đo TTFT lượt-kế-tiếp theo hàm của (thời gian chờ × tải nền).

### (2) KV pool cạn vì nhiều phiên × lịch sử dài → preemption dây chuyền

**Mặc định.** Khi `allocate_slots()` không cấp đủ block cho bước schedule hiện tại,
scheduler preempt. Chính sách mặc định FCFS chọn nạn nhân là **request chạy gần
nhất được admit** (`self.running.pop()`, `vllm/v1/core/sched/scheduler.py:608`);
với `--scheduling-policy priority` là request có `(priority, arrival_time)` thấp ưu
tiên nhất bị đá (`scheduler.py:583-587`). `_preempt_request()`
(`scheduler.py:1190`) đặt `status=PREEMPTED`, **`num_computed_tokens=0`**, giải
phóng toàn bộ block, đẩy về **đầu** hàng đợi waiting — đây là **tính lại từ đầu**,
không phải swap ra CPU (vLLM v1 không có swap-space path ở đây); lượt kế tiếp có
thể hit lại một phần prefix cache nếu block đó chưa bị evict song song, còn không
thì trả giá full recompute.

**Cách phát hiện.** Log định kỳ có dòng `"Preemptions: %d"` khi >0
(`vllm/v1/metrics/loggers.py:246-248`); metric Prometheus `vllm:num_preemptions`
(counter, `loggers.py:624-630`, tăng mỗi vòng lặp `:1154`). Kèm theo:
`kv_cache_usage_perc` áp sát 100% ngay trước đó.

**Cách xử lý.** (a) Tăng `--gpu-memory-utilization` (rẻ nhất, thuần config). (b) Giới
hạn độ dài lịch sử mỗi phiên (liên hệ kịch bản 13 — trim/tóm tắt trước khi context
phình). (c) `--scheduling-policy priority` + gắn `priority` thấp cho lượt agent
ngắn để chúng không bị preempt bởi phiên đuôi dài — CHƯA ĐO trực tiếp cho workload
này nhưng cơ chế đã xác nhận trong source (`scheduler.py:583-587`). (d) **Không**
dùng `--max-num-seqs` để chặn admission phía server — đã đóng hướng này ở TASK N1
(mục 17 dưới). Về cấu trúc: nhiều phiên agent với lịch sử ngày càng dài đúng là
tệ hơn traffic 1-lượt ngắn một cách **cấu trúc**, vì working-set (tổng số block cần
giữ sống cho các request đang chạy + cache gần đây) tăng không giới hạn trong khi
pool cố định — không phải hiện tượng ngẫu nhiên.

**Thí nghiệm chứng minh.** CHƯA ĐO cho hình dạng agent cụ thể (nhiều phiên, lịch sử
tăng dần theo lượt). TASK N1 (`STATUS.md` mục "admission cap phía server") đã chứng
minh phần "không cap phía server", nhưng đó là traffic đồng nhất conc32 một cấu
hình, không phải nhiều phiên lịch sử tăng dần. Cần
`scripts/bench_agent_loop.py --num-sessions N --turns T --history-mode accumulate`,
theo dõi `vllm:num_preemptions` theo N×T.

### (3) Slot mamba cạn (pool riêng, đặc thù model lai)

**Mặc định.** `MambaManager` (`vllm/v1/core/single_type_kv_cache_manager.py:1216`)
quản mamba-state **tách biệt hoàn toàn** khỏi pool KV attention (đúng như TASK C2
đã ghi: `STATUS.md:568-588` — offload CPU chỉ chạm tới KV attention, mamba-state
"nằm pool riêng"). Trần số phiên đồng thời với model lai là **số block mamba**, đọc
trực tiếp từ log khởi động (TEST 12: 9B ở fp8 footprint có 299 block @0.90 util;
9B FP8-dynamic chỉ 172 block → `max_num_seqs` phải ≤ 128, `STATUS.md:323-326`) —
đây là trần **cứng hơn** trần KV attention trong nhiều cấu hình vì mamba-state không
scale theo context length (O(1)/request) trong khi KV attention scale theo token.

Phiên IDLE (chờ tool, request đã kết thúc) **không** giữ slot mamba đặc quyền gì:
`pop_blocks_for_free()` (`single_type_kv_cache_manager.py:1591`) trả block về đúng
`BlockPool` dùng chung — bị LRU y hệt cơ chế kịch bản (1). Điểm khác biệt của model
lai nằm ở **cách tái tạo**, không phải cách bảo vệ:

- `--mamba-cache-mode align` cho phép `find_longest_cache_hit()`
  (`single_type_kv_cache_manager.py:1237`) tra hash mamba-state y hệt block KV
  attention — nếu còn cache, state SSM tại đúng ranh giới token đó được dùng lại
  thẳng, không cần "chạy lại SSM scan". Nếu bị evict → cache miss → rơi về
  `allocate_new_blocks()` (`:1485`), engine chạy lại SSM scan từ đầu cho phần chưa
  cache — đúng cơ chế "evict rồi tái tạo qua prefix cache" mà TASK F đã kiểm chứng
  **byte-identical** ở cả 32K lẫn 65K (`STATUS.md:414-417`).
- `reachable_block_mask()` (`:1315`) quyết định ranh giới hợp lệ để replay/cache
  qua tham số `retention_interval` (None=dense, 0=chỉ ranh giới mới nhất, >0=thưa
  theo đoạn) — đòn bẩy mật độ, không phải pin.
- `remove_skipped_blocks()` (`:1369`) giải phóng block state 2 bước trước
  (`last_state_block_idx`) ngay khi state running đã copy-forward — cơ chế
  copy-forward giữ đúng 1 block "nóng" mỗi request, phần cũ hơn thành ảnh chụp có
  thể cache/evict.
- Guard đúng đắn: `get_num_blocks_to_allocate()` (`:1416`) trả về giá trị "bất khả
  thi, thử lại bước sau" nếu một block hash cần dùng vừa được request KHÁC tạo
  trong CÙNG bước lịch — vì mamba-state không share được giữa các request trong
  cùng 1 bước (khác KV attention, vốn share được ngay).

**Cách phát hiện.** Không có metric Prometheus riêng cho occupancy mamba (đọc code
không thấy). Phải tự đọc số block ceiling từ **log khởi động** (dòng vLLM in số
block khi build KV cache spec) rồi tự so với `num_requests_running`; khi chạm trần,
biểu hiện ra ngoài là request mới không được admit / lỗi cấp phát — không có cảnh
báo sớm built-in.

**Cách xử lý.** (a) LUÔN đọc số block ceiling ở log khởi động trước khi đặt
`--max-num-seqs` — bài học TEST 12 áp dụng y nguyên cho agent loop
(`max_num_seqs ≤ số block log báo`). (b) Bật `--mamba-cache-mode align` +
`--enable-prefix-caching` để phiên idle được TÁI TẠO RẺ thay vì lỗi cấp phát khi
quay lại — đã PASS correctness ở TASK F, đây là điều kiện tiên quyết cho toàn bộ
workload agent nhiều-phiên. (c) Không có cách "mua thêm" slot mamba bằng offload —
TASK C2 đã đóng hướng này ("Mamba-state offload thật sự thì chưa ai ship").

**Thí nghiệm chứng minh.** TASK C2 (đọc code, chưa sweep thực nghiệm số phiên) +
TASK F (correctness align mode, đã PASS). CHƯA ĐO: số phiên agent đồng thời tăng
dần tới khi chạm trần block mamba và quan sát hành vi lỗi/degrade — cần
`scripts/bench_agent_loop.py --num-sessions` quét qua trần đã đọc từ log.

### (4) Prefix skills 30K bị lịch sử các phiên đá ra khỏi cache → thảm họa toàn hệ

**Mặc định.** Xác nhận trực tiếp từ source: `block_pool.py` không có khái niệm
tenant/session ID, không trọng số ưu tiên, không API "pin prefix này". Toàn bộ pool
là MỘT `free_block_queue` LRU dùng chung — prefix skills-pack 30K và lịch sử riêng
của từng phiên agent **cạnh tranh y hệt nhau** để giữ chỗ trong cùng một hàng đợi.
Đây là kịch bản nghiêm trọng nhất nhóm A vì nó phá vỡ chính lợi ích trung tâm mà
toàn dự án đã dựa vào (hit rate 99,4% TASK H, TTFT warm 0,2-1,0s) — nếu traffic
agent (nhiều lịch sử riêng, dài) đủ lớn để liên tục đẩy block skills-pack ra đuôi
LRU rồi bị evict, MỌI request tiếp theo (kể cả của tenant khác) trả về mức cold
(10,5s/62,9s) thay vì warm.

**Cách phát hiện.** `/metrics`: `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits`
(tên thật sau khi vLLM 0.26 đổi hậu tố `_total`, xem bug đã vá trong
`scripts/bench_skills.py` — trước đó `bench_skills.py` báo `cache_hit_rate=None`
oan vì tên metric đổi, `STATUS.md:607-608`). Tính `hits_delta/queries_delta`; cảnh
báo khi tụt rõ khỏi baseline đã lập (99,04-99,37%, TASK H/re-bench champion graft).
Đi kèm: TTFT p50 trôi dần từ 0,2-1,0s về gần 10,5s dù request KHÔNG liên quan đến
lịch sử dài (dấu hiệu phân biệt với kịch bản (1), vốn chỉ ảnh hưởng một phiên).

**Cách xử lý (rẻ → đắt).**
- (a) **Cô lập vật lý** theo tiền lệ đã đo: tách traffic "nặng lịch sử/tool result
  dài" khỏi traffic "câu hỏi ngắn ăn theo skills-pack" ra 2 instance
  (`--gpu-memory-utilization 0.40` mỗi cái) — đúng kiến trúc đã chứng minh ở mục
  "Cô lập chat khỏi prefill tài liệu" (TTFT max 1,985s→0,327s = 6×, ITL max
  1,257s→0,083s = 15×, `STATUS.md`). Rẻ vì weights nhỏ so với VRAM (2B: 1,8GiB×2);
  với 9B cần kiểm lại ngân sách VRAM (xem mục 19).
- (b) Giữ "nóng" prefix skills-pack bằng request no-op định kỳ (đẩy lại lên đầu
  hàng đợi recency) — chi phí thấp, không cần đổi kiến trúc, nhưng chỉ là giảm
  xác suất chứ không loại trừ (vẫn cùng một LRU dùng chung).
- (c) Tăng `--gpu-memory-utilization`/kích thước pool để tổng working-set (skills
  30K + mọi lịch sử phiên đang sống) vừa, tránh chạm ngưỡng evict.
- (d) `LMCache` MP mode (Nhóm E-20) — theo đúng chữ trong
  `upstream/research-persistent-kv-cache.md` (mục 3A): **không giải quyết được
  multi-tenant** ("Không giải quyết được multi-tenant (nhiều prefix khác nhau đá
  nhau khỏi VRAM)") — chỉ cứu cold-start sau restart, không cứu tranh chấp sống.
  Đừng kỳ vọng LMCache là lời giải cho đúng kịch bản này.

**Thí nghiệm chứng minh.** CHƯA ĐO. Cần `scripts/bench_agent_loop.py` chạy traffic
skills-pack (như `bench_skills.py` đã có) SONG SONG với `--competing-noise-sessions`
(lịch sử dài, không share prefix) tăng dần, đo đường cong tụt hit-rate + TTFT trôi —
tái dùng cơ chế scrape `/metrics` đã có sẵn trong `bench_skills.py`
(`scrape_prefix_cache_metrics`/`diff_prefix_cache_metrics`).

### (5) Đa khách hàng, mỗi khách một prefix khác nhau tranh chỗ

**Mặc định.** Tổng quát hoá của (4): N prefix riêng biệt (mỗi khách hàng một
skills-pack/system-prompt khác nhau) cùng cạnh tranh MỘT LRU toàn cục, không phân
biệt tenant. Nếu tổng kích thước N prefix vượt ngân sách pool, prefix ít được
"chạm" gần đây nhất (LRU tail) bị evict trước — kể cả khi đó là khách hàng đang trả
tiền nhiều nhất. Không có cơ chế reservation/quota theo tenant trong vLLM.

**Cách phát hiện.** vLLM không expose hit-rate theo từng prefix/tenant — chỉ có
tổng hợp toàn instance (`vllm:prefix_cache_queries/hits`). Muốn biết tenant nào bị
ảnh hưởng phải tự đo ở tầng ứng dụng: gắn tenant-id vào log, tự tính TTFT phân vị
theo từng nhóm request, so với baseline warm riêng của từng prefix.

**Cách xử lý.** (a) Capacity planning tường minh: tổng (Σ kích thước prefix mong
muốn giữ nóng) phải nhỏ hơn ngân sách KV pool thực đo (KV pool ~470K token @128K-config,
theo TASK N6) trừ hao hụt cho traffic biến động — nếu vượt, một số prefix chắc chắn
bị đá dù không có lỗi cấu hình nào. (b) Sharding theo prefix: route mỗi khách hàng
lớn (hoặc nhóm khách hàng dùng chung 1 prefix) sang instance/route riêng — bản chất
là mở rộng kiểu (4a) ra N nhóm thay vì 2. (c) Không có đòn config rẻ nào khác vì
đây là giới hạn kiến trúc (không có API pin) — bất kỳ giải pháp "công bằng multi-tenant"
nào đều phải làm ở tầng ứng dụng (rate limit theo tenant, ưu tiên theo hợp đồng SLA).

**Thí nghiệm chứng minh.** CHƯA ĐO — chưa có bench nào đo multi-prefix contention.
Cần `scripts/bench_agent_loop.py --prefix-set <N files>` gửi round-robin nhiều
prefix riêng, đo hit-rate tổng hợp và (nếu tự thêm log tenant-id) hit-rate từng
prefix theo tỉ lệ N × kích thước / ngân sách pool.

## Nhóm B — Lịch trình & độ trễ

Lưu ý khung cho cả nhóm này: trong pattern tool-calling chuẩn (OpenAI-compatible),
model **kết thúc lượt** với `tool_calls`, ứng dụng chạy tool NGOÀI vLLM, rồi gửi
**request HTTP mới** cho lượt kế. Nghĩa là vLLM **không hề bị block** chờ tool —
GPU rảnh hoàn toàn trong lúc chờ. Rủi ro thật của "tool chậm" nằm ở cache (Nhóm A),
không nằm ở việc chiếm giữ compute.

### (6) Tool chậm bất thường (30s+)

**Mặc định.** Như trên: không có request nào "treo" trên GPU chờ tool — GPU không
bị chiếm. Rủi ro duy nhất là cache (kịch bản A-1): thời gian chờ càng dài, xác suất
traffic khác đẩy block của phiên ra khỏi LRU càng cao.

**Cách phát hiện.** Tách biệt hai độ trễ ở tầng ứng dụng: `tool_latency` (đo ngoài
vLLM, giữa lúc nhận `tool_calls` và lúc gửi lại request có `tool_result`) và
`llm_latency` (TTFT+decode của chính request). Nếu dashboard gộp chung "độ trễ mỗi
lượt" sẽ không phân biệt được lỗi tool và lỗi phục vụ model.

**Cách xử lý.** Không có đòn cấu hình vLLM nào áp dụng trực tiếp (vì GPU đã rảnh).
Chỉ cần đảm bảo anti-pattern KHÔNG xảy ra: đừng giữ streaming connection mở/chờ vô
ích trong lúc tool chạy (nếu framework nào làm vậy, đó là lỗi client giữ tài
nguyên, xem kịch bản 16). Nếu tool chậm thường xuyên và lịch sử dài, cân nhắc timeout
cứng + fallback trả lời "đang xử lý" để tránh phiên treo vô thời hạn ở tầng ứng
dụng (không phải vấn đề vLLM).

**Thí nghiệm chứng minh.** Trùng với A-1 (`--tool-think-time`).

### (7) Tool lỗi/timeout → thử lại

**Mặc định.** Retry ở tầng ứng dụng tạo thêm request tới vLLM — với cùng lịch sử
(nếu retry gọi lại chính lượt vừa lỗi) hoặc lịch sử mở rộng (nếu model được yêu cầu
thử tool khác). Nếu retry đồng loạt trên nhiều phiên cùng lúc (ví dụ tool ngoài bị
down rồi phục hồi, mọi phiên retry cùng lúc), đây chính là kịch bản (8) — dồn cục.

**Cách phát hiện.** Spike đột ngột ở `num_requests_running` và TTFT p95 không tương
ứng với tăng traffic tổ chức thật — dấu hiệu retry storm.

**Cách xử lý.** Backoff có jitter ở tầng ứng dụng (rẻ, không đụng vLLM), circuit
breaker khi tool lỗi liên tục (dừng retry sau N lần, trả lỗi rõ ràng thay vì loop).
Kiểm soát tải bằng client-side/gateway (bài học N1, mục 17), KHÔNG dùng cap phía
server để "chặn" retry storm — đã chứng minh cap server làm đuôi trễ tệ hơn.

**Thí nghiệm chứng minh.** CHƯA ĐO.

### (8) Dồn cục đồng bộ (mọi phiên cùng vào lượt mới sau một API chung)

**Mặc định.** Đây chính là hình dạng traffic mà TASK N1 thử nghiệm ngầm (client đổ
nguyên batch conc32 cùng lúc, closed-loop): khi server bị cap (`--max-num-seqs`),
8 request tràn cap phải xếp hàng và "chi phối đuôi" — p95 nổ từ 3,03s (không cap)
lên 64,6s (mns=24), 57,4s (mns=16), 122,7s (mns=8) (`STATUS.md` TASK N1). Nguyên
nhân cấu trúc: burst đồng bộ nghĩa là admission đến dồn một cục thay vì rải theo
Poisson — bất kỳ giới hạn cứng nào ở admission phía server đều biến "chậm đều" thành
"một số request chết đói ở cuối hàng đợi".

**Cách phát hiện.** `num_requests_waiting` nhảy vọt đồng thời với TTFT p95 tăng vọt
ngay sau một sự kiện đồng bộ hoá bên ngoài (ví dụ cron trigger nhiều agent cùng lúc,
hoặc một API chung giải phóng N phiên cùng lúc).

**Cách xử lý.** (a) KHÔNG cap `--max-num-seqs` phía server (N1, đã đóng hướng).
(b) Rải burst ở tầng ứng dụng: thêm jitter ngẫu nhiên trước khi mỗi phiên gửi lượt
đầu sau sự kiện kích hoạt chung. (c) Giới hạn concurrency ở gateway theo đúng sức
chứa đã đo (SLA sạch 0,2-0,3 req/s cho kịch bản shared-prefix 32K theo TASK F2/F2c;
throughput server đạt 387,8 tok/s @conc32 theo TASK P1) — kiểm soát tại client, để
server mns rộng rãi.

**Thí nghiệm chứng minh.** TASK N1 (đã có, chứng minh phần "cap server thua").
CHƯA ĐO hình dạng cụ thể "burst đồng bộ sau 1 sự kiện chung của agent" — cần
`scripts/bench_agent_loop.py --burst-mode --burst-size N` (tất cả N phiên gửi lượt
đầu trong cùng 1 khoảnh khắc) so với cùng N request rải Poisson.

### (9) Phiên đuôi dài 20 lượt chiếm tài nguyên

**Mặc định.** Với prefix caching + align mode bật, mỗi lượt mới của một phiên dài
chỉ cần prefill phần SUFFIX mới (nếu lịch sử trước đó vẫn còn cache) — không phải
tính lại toàn bộ. Nhưng KV attention của phiên đó **tích luỹ** theo số lượt (mỗi
lượt thêm block mới được hash và giữ), chiếm ngày càng nhiều pool — đúng cơ chế
LRU-cạnh-tranh của Nhóm A áp dụng cho một phiên đơn lẻ ngày càng "nặng". Về tốc độ
per-user, TASK N6 đã đo trực tiếp hệ quả context dài: decode per-user giảm khi
nhiều luồng cùng quét KV dài (conc1 28,9 → conc4 21,7 tok/s ở 120K, giảm 25%) — một
phiên 20 lượt tích luỹ context lớn tự nó làm chậm cả chính nó lẫn (gián tiếp, qua
tranh chấp compute — xem kịch bản 14) các phiên khác cùng batch.

**Cách phát hiện.** Theo dõi token-count tích luỹ mỗi phiên (tầng ứng dụng, vLLM
không tự phân biệt "phiên"), `kv_cache_usage_perc` leo dần, decode tok/s của chính
phiên đó giảm dần theo lượt (đo phía client).

**Cách xử lý.** (a) Giới hạn số lượt/độ dài hội thoại tối đa mỗi phiên, kết hợp
trim/tóm tắt trước khi context vượt ngân sách (liên hệ trực tiếp kịch bản 13 — đọc
kỹ cạm bẫy prefix-cache ở đó trước khi trim). (b) `--scheduling-policy priority`:
gán priority thấp hơn cho phiên đã biết là đuôi dài để không chiếm ưu tiên preemption
so với các lượt agent ngắn khác — CHƯA ĐO, chỉ suy ra từ cơ chế `scheduler.py:583-587`.
(c) Không có đòn nào chặn riêng "phiên dài" ở mức KV attention mà không đụng tới độ
dài hội thoại thật — đây là chi phí cấu trúc của việc giữ ngữ cảnh dài.

**Thí nghiệm chứng minh.** Số decode-per-user-theo-context đã có (TASK N6). CHƯA ĐO
tích luỹ 20 lượt thật (không phải 1 lượt dài): cần
`scripts/bench_agent_loop.py --turns 20 --history-mode accumulate`, đo TTFT/decode
mỗi lượt theo số thứ tự lượt, và ảnh hưởng lan sang phiên khác cùng batch.

### (10) Trộn traffic agent với chat tương tác trên cùng GPU

**Mặc định.** MPS SM-pinning **không hoạt động** trên bản vLLM hiện tại:
`CUDA_MPS_*` tới được `APIServer` nhưng **không truyền vào `EngineCore`** (do
spawn multiprocessing) — bug đã ghi trong `upstream/10-vllm-cuda-mps-env-not-propagated-to-enginecore.md`
và `STATUS.md` (mục "Cô lập chat khỏi prefill tài liệu"). Nếu trộn traffic agent
(prefill nặng — tool result dài, lịch sử dài) với chat tương tác (latency-sensitive)
trên MỘT instance, phía chat hứng đuôi trễ khi phía agent tải nặng — đúng cơ chế
tranh chấp compute-trong-batch đã đo ở TASK F2b (mục 14 dưới).

**Cách phát hiện.** TTFT/ITL của traffic chat tăng đột biến đồng thời với traffic
agent tăng, dù `num_requests_waiting` có thể vẫn 0 (tranh chấp compute, không phải
admission — cùng chẩn đoán F2b).

**Cách xử lý.** Đã đo trực tiếp: **2 instance thường** (mỗi cái
`--gpu-memory-utilization 0.40`, dựa vào driver time-slicing, KHÔNG cần MPS) thắng
1 instance trộn: TTFT max 1,985s → **0,327s (6×)**, ITL max 1,257s → **0,083s
(15×)** (`STATUS.md`). Khuyến nghị production: tách agent/chat ra 2 instance trên
cùng card — rẻ vì weights nhỏ so với VRAM còn dư (xem cảnh báo ngân sách VRAM ở
mục 19 cho 9B). Nếu không đủ VRAM cho 2 instance, fallback duy nhất còn lại là
`--scheduling-policy priority` trên 1 instance (ưu tiên lượt chat ngắn) — CHƯA ĐO
số liệu cụ thể cho fallback này.

**Thí nghiệm chứng minh.** Số 2-instance đã có (`STATUS.md`, mục "Cô lập chat khỏi
prefill tài liệu"). CHƯA ĐO: đúng cặp "agent loop nhiều lượt" vs "chat tương tác"
(số hiện có là "chat" vs "tài liệu dài", tương tự nhưng chưa phải agent thật) — cần
`scripts/bench_agent_loop.py` chạy song song với traffic chat ngắn, đo cả 1-instance
và 2-instance.

## Nhóm C — Nội dung & đúng đắn

### (11) JSON gọi tool sai cú pháp → lượt thừa

**Mặc định.** Không có ràng buộc cú pháp nào tự áp dụng trừ khi request yêu cầu.
Nếu model sinh JSON hỏng, ứng dụng phải phát hiện (parse lỗi) rồi thường retry toàn
bộ lượt — chi phí là **1 round-trip đầy đủ thêm** (phần lớn prefill có thể hit cache
nếu lịch sử không đổi, nhưng decode phải sinh lại từ đầu) mỗi lần lỗi; lặp 2-3 lần
thì nhân trực tiếp số round-trip cần cho 1 tác vụ, ăn thẳng vào ngân sách SLA đã đo
(0,2-0,3 req/s sạch cho kịch bản shared-prefix, TASK F2/F2c) — 1 lượt lỗi giữa
chừng gần như tương đương gấp đôi tải hiệu dụng của phiên đó trong khoảng thời gian
retry.

**Đường sửa đúng — structured outputs (tra cứu trực tiếp `protocol.py`/`sampling_params.py`
trong `D:\Training\AI_Module\vllm\vllm\vllm`, KHÔNG phải cú pháp `guided_json` cũ
đã bị loại khỏi bản này):**
- Field OpenAI-compatible chuẩn: `response_format` (kiểu `AnyResponseFormat`) —
  hỗ trợ `{"type": "json_object"}`, `{"type": "json_schema", "json_schema": {...}}`,
  `{"type": "structural_tag", ...}` (`entrypoints/openai/chat_completion/protocol.py`).
- Field vLLM-native mạnh hơn: `structured_outputs` (kiểu `StructuredOutputsParams`,
  `sampling_params.py`), gồm `json`/`regex`/`choice`/`grammar`/`json_object`/
  `structural_tag` (đúng 1 trong số này phải được set) cộng `disable_any_whitespace`,
  `disable_additional_properties`, `whitespace_pattern`.
- Backend chọn ở **cấp server**, không phải per-request:
  `--structured-outputs-config.backend {auto,xgrammar,guidance,outlines,lm-format-enforcer}`
  (`vllm/config/structured_outputs.py`). `disable_any_whitespace` chỉ hợp lệ với
  `xgrammar`/`guidance`; `disable_additional_properties` chỉ hợp lệ với `guidance`.

**Cách phát hiện.** Đếm tỉ lệ lượt phải retry do JSON parse lỗi ở tầng ứng dụng
(vLLM không tự biết "đây là tool call hỏng"); nếu bật structured outputs, theo dõi
`finish_reason` và độ trễ decode có tăng nhẹ do grammar masking mỗi bước hay không
(đánh đổi nhỏ, gần như luôn đáng với workload gọi tool nhiều).

**Cách xử lý.** Bật `response_format={"type":"json_schema",...}` (di động, tương
thích OpenAI SDK) hoặc `structured_outputs={"json": schema}` (native, kiểm soát chi
tiết hơn) cho MỌI lượt có khả năng gọi tool; chọn backend `xgrammar` ở cấp server
qua `--structured-outputs-config.backend xgrammar`. Đây gần như luôn rẻ hơn chấp
nhận retry, vì retry tốn nguyên 1 round-trip còn grammar chỉ tốn vài % decode mỗi
bước.

**Thí nghiệm chứng minh.** CHƯA ĐO. Cần `scripts/bench_agent_loop.py
--json-mode {none,response_format,structured_outputs}`, đo (a) decode tok/s chênh
lệch do overhead grammar, (b) tỉ lệ lượt-thừa do JSON hỏng ở chế độ `none` so với có
ràng buộc.

### (12) Model lặp vô hạn cùng một lời gọi tool

**Mặc định — liên hệ trực tiếp thang bit thấp 9B (TASK 9, `STATUS.md`).** Chữ ký
lỗi đã quan sát ở model reasoning khi giảm bit KHÔNG phải sai kiến thức mà là
**NON-TERMINATION**: "kiến thức + số học sống tới tận 2-bit; thứ chết đầu tiên là
khả năng CHỐT đáp án sau chuỗi suy luận (lặp tự-kiểm-chứng, tự-sửa-đổi bất tận)".
Với vòng lặp agent, cùng bệnh này biểu hiện thành lặp lại y hệt một lời gọi tool —
model không "chốt" được rằng tool đã trả lời đủ. Q4_K_M là **sàn chất lượng đã xác
nhận PASS sạch** cho 9B (274,3 tok/s decode); mọi bậc thấp hơn (UD-Q3_K_XL,
Q3_K_M, UD-Q2_K_XL, UD-IQ2_M) đều FAIL vì đúng triệu chứng non-termination
(`STATUS.md` mục "Thang bit thấp 9B"). Quy tắc control-validate cũng áp dụng: một
câu dò "sập vòng lặp" phải được so với control full-precision trước khi kết luận là
lỗi do quant — có câu (ví dụ "kể 3 trái cây") sập lặp NGAY Ở BF16 vì đó là điểm hút
thoái hoá của chính base model, không liên quan bit-width.

**Cách phát hiện.** `finish_reason="length"` (chạm `max_tokens` mà không có
`tool_calls` hợp lệ) là tín hiệu trực tiếp nhất. Ở tầng ứng dụng: theo dõi chữ ký
tool-call trùng lặp (cùng tên hàm + cùng tham số, hoặc tham số gần giống hệt) xuất
hiện ≥2 lần liên tiếp trong một phiên.

**Cách xử lý.** (a) `max_tokens` cứng mỗi lượt (chặn chi phí vô hạn, không sửa được
nguyên nhân). (b) Circuit breaker tầng ứng dụng: cùng chữ ký tool-call lặp N lần →
dừng vòng lặp, trả lỗi/hỏi lại người dùng thay vì tiếp tục gọi model. (c) Structured
outputs (kịch bản 11) giúp MỘT PHẦN — ép đúng schema JSON không ép được việc model
"muốn" gọi lại đúng tool đó, chỉ đảm bảo cú pháp hợp lệ mỗi lần. (d) **Không hạ
xuống dưới Q4_K_M cho 9B khi cần agent loop tin cậy** — đúng workload agent càng
cần khả năng CHỐT sạch hơn workload chat tự do, nên độ nhạy với bậc bit thấp ở đây
còn quan trọng hơn ở chat.

**Thí nghiệm chứng minh.** TASK 9 (đã có, xác nhận hiện tượng ở cấp model, không
riêng agent loop). CHƯA ĐO: bộ dò "tool-call lặp cùng chữ ký" cụ thể cho workload
agent — cần `scripts/bench_agent_loop.py` với probe tool cố ý mơ hồ (kết quả tool
không đủ để model tự tin chốt), đếm tỉ lệ lặp theo từng bậc bit.

### (13) Lịch sử vượt max-model-len — chính sách cắt/tóm tắt và HỆ QUẢ LÊN PREFIX CACHE

**Đây là cái bẫy lớn nhất nhóm C — đọc kỹ trước khi implement bất kỳ chính sách trim
nào.** Prefix caching khớp theo **hash các block TIỀN TỐ** — bất kỳ chỉnh sửa nào ở
vị trí token `k` làm mọi block SAU vị trí đó (dù nội dung giống hệt cũ) có hash
khác, tức **mất cache từ điểm sửa trở về sau**, bất kể phần TRƯỚC điểm sửa còn
nguyên hay không.

Hệ quả theo từng chính sách:
- **Sliding-window giữ K lượt gần nhất + system-prompt/skills-pack cố định ở đầu,
  KHÔNG đổi 1 byte nào của phần đầu đó**: chỉ phần giữa (các lượt bị bỏ) đổi, và
  toàn bộ phần SAU điểm cắt (các lượt còn giữ + lượt mới) trở thành chuỗi token
  mới — cache RIÊNG của phiên đó (không phải skills-pack chung) mất từ điểm cắt trở
  đi, phải tính lại 1 lần (giống 1 lần "cold" con, không phải cold toàn bộ). Prefix
  skills-pack CHUNG (đầu tuyệt đối) vẫn còn nguyên NẾU không bị đụng — hit rate toàn
  hệ (kịch bản A-4) không bị ảnh hưởng.
- **Tóm tắt phần giữa thành 1 message ngắn rồi chèn vào**: cùng hệ quả — mọi thứ
  sau điểm chèn đổi hash, phải tính lại; đây là chi phí MỘT LẦN mỗi lần tóm tắt (quy
  mô nhỏ hơn cold-start toàn phần vì thường chỉ vài nghìn token, không phải 30-120K).
- **"Cắt đầu" theo nghĩa đen (bỏ luôn N token đầu tiên, kể cả system-prompt/skills-pack)**:
  đây là **thảm hoạ đúng như đề bài mô tả** — nếu implementation tự động sinh lại
  system message với nội dung động (ví dụ nhúng timestamp, số lượt, id phiên vào
  đầu prompt) thì TOKEN 0 đổi mỗi lượt → **không bao giờ hit cache**, kể cả phần
  skills-pack lẽ ra dùng chung được — biến một workload cache-bound (390-520 tok/s,
  TASK N3/P1) thành workload prefill-bound-mỗi-lượt (mức GGUF fused ~130-274 tok/s
  decode, chưa kể phí prefill lặp lại liên tục). Đây chính là cơ chế đã ghi nhận ở
  "Bẫy đo lường #5" của `STATUS.md` (prefix caching khớp theo TIỀN TỐ, không phải
  vị trí bất kỳ) áp dụng ngược lại: những gì từng "thổi phồng số đo" khi prompt lặp
  y hệt, giờ là thứ MẤT ĐI khi prompt không còn lặp y hệt ở phần đầu.

**Khuyến nghị chính sách.** Không bao giờ sửa/regenerate byte của system-prompt/
skills-pack cố định. Trim/tóm tắt CHỈ áp dụng cho phần lịch sử SAU nó, càng gần
đầu của phần "riêng phiên" càng tốt (giảm thiểu phần phải giữ ổn định), và chấp
nhận 1 lần recompute-cục-bộ mỗi lần trim (không phải cold toàn bộ, vì skills-pack
vẫn hit). Không trim đuôi (mất ngữ cảnh gần nhất — sai hướng cho agent, luôn cần
tool-result gần nhất).

**Cách phát hiện.** Theo dõi TTFT của phiên NGAY SAU một sự kiện trim — nếu tăng
tương ứng với "1 lần cold cục bộ" (vài trăm ms tới vài giây tuỳ kích thước phần bị
đổi) là bình thường; nếu tăng đột biến về mức cold-toàn-phần (giây tới hàng chục
giây) là dấu hiệu chính sách trim đang vô tình đụng vào phần đầu cố định. Theo dõi
song song `vllm:prefix_cache_hits/queries` tổng hệ thống — nếu hit rate tụt đồng
thời với các sự kiện trim của nhiều phiên, nghi ngờ (c) đang xảy ra.

**Thí nghiệm chứng minh.** Baseline correctness/cost cho context dài đã có: TASK N6
(128K, prefix 120.036 token, cache-hit byte-identical, cold 62,9s vs warm cùng
prompt 3,06s). CHƯA ĐO trực tiếp 3 chính sách trim/tóm tắt trên phiên vượt
max-model-len — cần `scripts/bench_agent_loop.py
--context-policy {slide,summarize,truncate-head-naive}`, đo TTFT/hit-rate trước và
sau mỗi sự kiện trim cho từng chính sách.

### (14) Kết quả tool khổng lồ (20K+ token) chen giữa hội thoại

**Mặc định — cùng cơ chế đã chẩn đoán ở TASK F2b.** Một tool-result khổng lồ được
chèn vào giữa hội thoại tạo ra một prefill lớn phải chunk qua nhiều bước; các chunk
này **chen vào giữa các bước decode của những request khác đang chạy cùng batch**.
TASK F2b đo chính xác cơ chế này với suffix chỉ ~2,25K token/req: ở rate 0,3 req/s,
`num_requests_waiting = 0` suốt 180s (KHÔNG phải hết KV, KHÔNG phải nghẽn admission)
nhưng TTFT/decode của TẤT CẢ request chậm đi khi `running` leo lên 15-20 — đúng là
**tranh chấp compute trong batch**, không phải thiếu tài nguyên. Một tool-result
20K+ token lớn gấp ~9× suffix đã đo, nên mức độ chen lấn/tranh chấp compute cho mỗi
lần chèn sẽ nặng hơn đáng kể theo cùng cơ chế.

**Cách phát hiện.** Đúng công thức chẩn đoán F2b: `/metrics` mỗi vài giây, thấy
`num_requests_waiting = 0` NHƯNG `kv_cache_usage_perc` và (quan trọng hơn)
TTFT/ITL của các request khác cùng tăng đúng lúc `num_requests_running` leo cao —
đây là dấu hiệu compute contention, phân biệt với thiếu KV (kịch bản A-2, có
`num_preemptions`>0) và thiếu admission (waiting>0 kéo dài).

**Cách xử lý (rẻ → đắt).**
- (a) **Rẻ nhất, sửa tận gốc**: đừng chèn nguyên văn 20K+ token tool-result vào
  prompt — tóm tắt/trích đoạn liên quan trước khi chèn lại (retrieval/rerank phía
  ứng dụng). Loại bỏ vấn đề tại nguồn thay vì hấp thụ nó ở tầng scheduler.
- (b) Nếu buộc phải chèn nguyên văn: hạ `--max-num-batched-tokens` về gần sàn cứng
  của model lai (**1088**, sàn được xác định bởi `mamba block_size = 1056` — dưới
  đó server không khởi động được, `AssertionError`, TASK F2c) để giới hạn mỗi bước
  schedule nhường bao nhiêu cho prefill lớn trước khi quay lại decode. Đã đo trực
  tiếp: `mnbt=1088` cho `p95=2,51s` tại rate 0,3 (so với `mnbt=8192` mặc định cho
  `p95=2,99s`) trên đúng kịch bản shared-prefix — cùng đòn bẩy áp dụng cho tool-result
  khổng lồ, dù chưa đo riêng ở kích thước 20K.
- (c) Không giúp được ở rate cao: F2b/F2c đã ghi nhận đòn `mnbt` KHÔNG mở được SLA
  ở rate 0,5 (vấn đề khi đó là decode-residency/tồn đọng concurrency, không phải
  prefill chen) — nếu tool-result khổng lồ xảy ra thường xuyên ở tải cao, (a) là
  đường duy nhất còn hiệu quả.

**Thí nghiệm chứng minh.** TASK F2b/F2c (đã có, cùng cơ chế, kích thước suffix nhỏ
hơn). CHƯA ĐO đúng kích thước 20K+ chèn giữa hội thoại — cần
`scripts/bench_agent_loop.py --tool-result-size 20000 --tool-result-position mid`,
quét `--max-num-batched-tokens` quanh sàn 1088 và so với mặc định 8192.

## Nhóm D — Vận hành

### (15) Restart server → mọi phiên mất cache cùng lúc (đàn voi giẫm)

**Mặc định.** Prefix-cache sống trong VRAM, không trên đĩa — `vllm serve` mới khởi
động với pool trống hoàn toàn. Nếu request thật đầu tiên sau restart là request trả
tiền tokens phải trả giá cold-prefill (10,5s @30K / 62,9s @120K), họ nhìn thấy đó
là latency/timeout; tệ hơn, health-check "server đã lên" **không đồng nghĩa** cache
đã ấm — health check pass chỉ nghĩa là process trả lời được, không phải lượt tiếp
theo sẽ nhanh.

**Cách phát hiện.** TTFT p50 spike ngay lập tức sau mọi lần deploy/restart; hit-rate
`/metrics` reset về 0 lúc khởi động (kỳ vọng, không phải bug — nhưng nếu traffic
thật đổ vào ngay lập tức, nhiều request đầu cùng ăn cold-prefill = "đàn voi giẫm").

**Cách xử lý (rẻ → đắt).**
- (a) **`scripts/warmup_prefix.py`** — đã có sẵn trong repo, đúng mục đích: gửi 1
  request chứa đủ prefix chuẩn (max_tokens nhỏ) NGAY sau khi server health-check
  pass nhưng TRƯỚC khi mở traffic thật, trả giá cold-prefill 1 lần ngoài
  critical-path người dùng thật. Script kiên nhẫn poll health/`/v1/models` với
  backoff (vì HTTP server có thể lên trước khi engine nạp/compile xong).
- (b) **Blue-green**: giữ instance CŨ phục vụ traffic thật cho tới khi instance MỚI
  chạy xong `warmup_prefix.py`, rồi mới cắt traffic sang — tránh việc "đàn voi" của
  MỌI phiên đồng thời cùng đập vào một cache trống ngay thời điểm cutover.
- (c) `LMCache` MP mode + L2 disk backend (Nhóm E-20) — persist cache **qua restart
  thật sự**, không cần trả cold-prefill lại; nặng hơn (thêm tiến trình `lmcache
  server`, hạ tầng phụ trợ) và chưa có xác nhận độc lập ngoài tài liệu chính thức
  của LMCache cho model lai này — chỉ đáng đầu tư sau khi (a)/(b) không đủ (ví dụ
  restart quá thường xuyên để warmup theo kịp, hoặc prefix quá lớn khiến 1 lần
  warmup tốn quá lâu).

**Thí nghiệm chứng minh.** `scripts/warmup_prefix.py` (đã có, kèm `scripts/test_warmup_prefix.py`).
Số cold-prefill đã đo (TASK H/N6: 10,52s @30K, 62,9s @120K). CHƯA ĐO: kịch bản
blue-green cutover thật với tải đồng thời đổ vào ngay lúc chuyển traffic.

### (16) Client ngắt giữa chừng — GPU có tiếp tục sinh không, cách hủy

**Mặc định — đọc trực tiếp source, KHÔNG phải suy đoán.** vLLM **thực sự hủy**
generation khi client ngắt kết nối, không chạy tới hết:
- `with_cancellation()` (`vllm/entrypoints/serve/utils/api_utils.py:52`) chạy đua
  `handler_task` với `listen_for_disconnect()` (`await request.receive()` chờ
  `http.disconnect`) bằng `asyncio.wait(..., FIRST_COMPLETED)`; task thua bị
  `.cancel()`. Docstring nói rõ tránh dùng `request.is_disconnected` vì "không hoạt
  động đúng với middleware".
- `AsyncLLM` bắt `asyncio.CancelledError` và gọi `await self.abort(request_id,
  internal=True)`, log `"Request %s aborted."` (`vllm/v1/engine/async_llm.py`).
- `abort()` → `output_processor.abort_requests()` +
  `engine_core.abort_requests_async()` → scheduler
  `finish_requests(ids, RequestStatus.FINISHED_ABORTED)`
  (`vllm/v1/core/sched/scheduler.py:2089`, docstring: *"the API server can abort a
  request when the client disconnects"*) → giải phóng block KV/mamba **ngay lập
  tức** (trừ trường hợp đang chờ KV-connector transfer, `delay_free_blocks`).

**Cách phát hiện.** Không cần cơ chế phát hiện riêng — hành vi đúng đắn diễn ra tự
động, MIỄN LÀ client thực sự đóng kết nối. Anti-pattern cần cảnh giác: một số
SSE/streaming client "ngừng đọc" response nhưng KHÔNG đóng connection TCP — trường
hợp đó `request.receive()` không bao giờ nhận `http.disconnect`, request tiếp tục
chạy trên GPU vô ích dù không ai đọc kết quả. Đây là nguồn lãng phí GPU âm thầm, khó
phát hiện bằng metric (không có tín hiệu riêng biệt) — chỉ phát hiện gián tiếp qua
`num_requests_running` cao bất thường so với traffic đang active thật ở tầng ứng
dụng.

**Cách xử lý.** Đảm bảo agent framework/orchestrator: (a) đặt timeout hợp lý mỗi
lượt (theo ngân sách TTFT+decode kỳ vọng) và THỰC SỰ hủy HTTP request khi timeout
(đóng connection, không chỉ dừng đọc); (b) khi người dùng huỷ tác vụ ở tầng ứng
dụng, propagate huỷ xuống tận HTTP client library đang gọi vLLM. Timeout tích cực
là AN TOÀN và có lợi — giải phóng GPU ngay, không chỉ tiết kiệm băng thông.

**Thí nghiệm chứng minh.** CHƯA ĐO — chưa có bench nào trong repo đo độ trễ từ lúc
hủy client tới lúc block KV thật sự được giải phóng (`/metrics` `kv_cache_usage_perc`
giảm). Cần script client tự ngắt kết nối có kiểm soát, đối chiếu thời điểm hủy với
metric.

### (17) Rate-limit/admission phải đặt ở gateway chứ không phải server (bài học N1)

**Mặc định.** Đã đóng dứt điểm ở TASK N1: sweep `--max-num-seqs {8,16,24}` trên
champion graft với client vẫn đổ nguyên conc32 — baseline không cap: p95=3,03s,
370,6 tok/s; `mns=24`: p95=**64,6s**; `mns=16`: p95=**57,4s**; `mns=8`: **thảm hoạ**,
122,7s/193 tok/s. Kết luận nguyên văn: "cap server chỉ chuyển chi phí từ 'cùng
chậm' sang 'đuôi chết đói', p95 luôn tệ hơn".

**Cách phát hiện.** Nếu thấy p50 đẹp nhưng p95 cực xấu khi có cap `max-num-seqs`
đang bật cùng traffic burst — đúng chữ ký của lỗi cấu hình này (một số request tràn
cap chờ hàng đợi chi phối đuôi).

**Cách xử lý.** Kiểm soát concurrency/QPS đúng chỗ — ở **API gateway/load balancer**,
sized theo sức chứa đã đo thực nghiệm: SLA sạch 0,2-0,3 req/s cho kịch bản
shared-prefix 32K (TASK F2/F2c), throughput server ~387,8 tok/s @conc32 đạt được
(TASK N5c/P1), trần compute thật ~500-520 tok/s ở chế độ offline batch (TASK P1).
Server nên để `--max-num-seqs` rộng rãi (TASK N6b: mns 16/32 không ảnh hưởng KV
pool đáng kể — trần bộ nhớ thật do `--gpu-memory-utilization` quyết định, không phụ
thuộc mns) và để mọi kiểm soát tải nằm hoàn toàn phía client.

**Thí nghiệm chứng minh.** TASK N1 (đã có, đầy đủ).

### (18) Giám sát: chỉ số nào phải cảnh báo

| Metric/log | Ngưỡng cảnh báo | Kịch bản liên quan |
|---|---|---|
| `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (tỉ lệ, tự tính — tên đổi hậu tố `_total` ở vLLM 0.26, xem bug đã vá `STATUS.md:607-608`) | Tụt rõ khỏi baseline đã lập (99,04-99,37% theo TASK H/re-bench champion) | A-1, A-4, A-5 |
| `vllm:num_preemptions` / log `"Preemptions: %d"` | Bất kỳ giá trị >0 kéo dài (khác đột biến nhất thời) | A-2 |
| `num_requests_waiting` | >0 kéo dài (phân biệt với compute-contention có waiting=0, xem cách đọc ở dưới) | B-8, B-9 |
| `kv_cache_usage_perc` | Áp sát 100% sustained (cảnh báo TRƯỚC khi chạm preemption, không phải sau) | A-2, A-3 |
| TTFT p95 (đo phía client, so baseline theo workload) | Trôi khỏi baseline warm đã lập (0,2-1,0s @30K theo TASK H/N6; SLA vận hành TTFT p95<3s theo TASK F2/F2c) | A-1, A-2, A-4, A-5, B-8, C-14 |
| Decode tok/s per-user (đo phía client theo concurrency) | Giảm không tương ứng với tăng concurrency dự kiến (so bảng N6/F2b) | B-9, C-14 |
| Số block mamba đang dùng vs ceiling (tự đọc log khởi động + tự đếm `num_requests_running`, KHÔNG có metric riêng) | Áp sát ceiling đã đọc từ log lúc start | A-3 |

**Cách đọc `waiting=0` nhưng vẫn chậm (đúng chẩn đoán TASK F2b):** nếu
`num_requests_waiting=0` suốt một khoảng dài NHƯNG TTFT/ITL vẫn xấu đi khi
`num_requests_running` leo cao, đó là **tranh chấp compute trong batch**
(chunked-prefill của người mới chen decode người cũ), không phải thiếu KV hay
nghẽn admission — đòn xử lý là `--max-num-batched-tokens` (kịch bản C-14), không
phải tăng pool hay giảm cap.

## Nhóm E — Mở rộng

### (19) Tách instance theo loại workload (agent vs chat) và MPS

Đã trình bày cơ chế và số liệu ở kịch bản B-10. Bổ sung tính kinh tế theo cỡ model:
- **2B**: 2 instance × `--gpu-memory-utilization 0.40` rẻ vì weights chỉ 1,8GiB —
  còn dư nhiều VRAM cho cả hai KV pool trên 23GB L4.
- **9B**: weights riêng đã chiếm 5,29GiB đĩa/8,5GiB VRAM (Q4_K_M) hoặc tương đương
  cho AWQ/W4A16 champion; 2 instance × weights 9B + 2 KV pool trên cùng 23GB L4 là
  **eo hẹp hơn nhiều** so với 2B — CHƯA ĐO liệu 2 instance 9B có fit ở context hữu
  ích (ví dụ 32K mỗi bên) trên 1 L4 hay không; TASK N6 cho thấy 1 instance 9B ở
  128K context đã dùng 17,7/23,0GiB một mình. Khuyến nghị: với 9B, ưu tiên **2 GPU
  vật lý riêng** cho agent/chat nếu có, hoặc chấp nhận 1 instance + fallback
  `--scheduling-policy priority` (chưa đo) nếu chỉ có 1 L4.
- MPS SM-pinning: đóng hướng, bug xác nhận (`upstream/10-...md`) — không phải hướng
  khả thi cho tới khi patch worker-bootstrap của vLLM (ngoài phạm vi cấu hình).

**Thí nghiệm chứng minh.** Số 2B đã có. CHƯA ĐO 9B: cần thử fit 2 instance 9B trên
1 L4 ở context vừa phải (16-32K mỗi bên) và đo VRAM thực tế trước khi khuyến nghị.

### (20) Offload KV sang CPU / KV bền (LMCache) cho phiên ngủ đông

**CPU KV offload (`OffloadingConnector`, TASK C2) — đã đóng, chỉ mua context/độ
phủ prefix, KHÔNG mua số phiên.** Chỉ quản tier KV attention; mamba-state nằm pool
riêng không offload được (`single_type_kv_cache_manager.py`) — request đang chạy
vẫn cần 1 slot mamba sống, nên offload không nâng được số phiên đồng thời quá trần
block mamba. `align` chỉ đồng bộ evict mamba-state với evict KV-block dưới prefix
caching (phiên idle/finished resume qua prefix cache), không phải cách "chạy thêm"
phiên live.

**LMCache MP mode — tóm tắt từ `upstream/research-persistent-kv-cache.md`, ĐỦ ĐIỀU
KIỆN, không phải giải pháp không rủi ro.**
- **Có hỗ trợ hybrid GDN, có tài liệu chính thức xác nhận** ("Models that
  interleave Mamba/GDN linear-attention layers with full attention...are
  supported", + recipe riêng cho family Qwen3.5/3.6). LMCache lưu/khôi phục CẢ
  conv-state lẫn SSM-state của GDN (coi là "trang cache byte-opaque"), khác hẳn
  giới hạn của `OffloadingConnector` (chỉ KV attention).
- **Điều kiện bắt buộc (trích recipe chính thức)**: chạy tiến trình `lmcache
  server` RIÊNG (`--chunk-size <N> --separate-object-groups --l1-size-gb X
  --eviction-policy LRU`, N phải khớp block size hợp nhất thật đọc từ log vLLM,
  cùng bài học `mamba block_size=1056` đã biết, chỉ đổi tên); vLLM serve cần
  `--enable-prefix-caching --mamba-cache-mode align
  --max-num-batched-tokens <2N-1>` và `--kv-transfer-config
  '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'`; **BẮT BUỘC dùng
  package `lmcache` pip ngoài, phiên bản ≥0.5.2** (yêu cầu tương thích vLLM 0.26) —
  bản `LMCacheMPConnector` builtin fallback trong vLLM source **crash thẳng** trên
  model lai nếu Hybrid Memory Allocator (HMA) còn bật
  (`RuntimeError` ở `reformat_block_ids()`, "LMCacheMPConnector only works without
  hybrid kv cache manager"). Mode đơn giản `LMCacheConnectorV1` (in-process) **rủi
  ro trên model lai** — không khai báo `SupportsHMA`, và
  `unify_hybrid_kv_cache_specs()` (`kv_cache_utils.py`) không có nhánh xử lý
  `MambaSpec` — chưa xác nhận đúng đắn qua source, đừng dùng.
- **Cảnh báo đúng đắn từ chính docs LMCache**: generation KHÔNG bit-exact giữa
  chạy cached và chạy fresh dưới tải đồng thời — vì kernel GDN không hỗ trợ
  batch-invariant mode (nhiễu số học phụ thuộc thành phần batch, khác bản chất với
  "thiếu state"). Khuyến nghị chính thức: so sánh điểm số/likelihood, không so
  token-level diff dưới tải đồng thời. Cached pages của KDA/GDN là byte-opaque —
  không dùng được tối ưu nén/blend nâng cao (CacheGen/CacheBlend) trên các layer
  GDN.
- **Không giải quyết multi-tenant** (A-4/A-5) — chỉ cứu cold-start SAU restart
  (D-15), không cứu tranh chấp LRU giữa các phiên đang SỐNG.
- **Thứ tự đầu tư khuyến nghị**: triển khai warmup script (D-15a, đã có sẵn
  `scripts/warmup_prefix.py`) TRƯỚC — rẻ, chắc chắn, không phụ thuộc hạ tầng ngoài.
  Chỉ cân nhắc LMCache MP mode khi (a) restart xảy ra đủ thường xuyên để warmup
  không theo kịp, HOẶC (b) cần chia sẻ cache prefix giữa NHIỀU instance/replica
  (bài toán khác — không nằm trong phạm vi 20 kịch bản này, single-node). Trước khi
  tin dùng: chạy đúng cổng correctness đã dùng xuyên suốt dự án (TASK F/N6 style —
  so sánh output cache-hit qua restart cả `vllm serve` lẫn `lmcache server` với
  output cold, temp=0, byte-identical hoặc so logprob nếu không).

**Thí nghiệm chứng minh.** Kiến trúc CPU offload đã đọc code (TASK C2, chưa sweep
thực nghiệm số phiên). LMCache: nghiên cứu khả thi đầy đủ đã có
(`upstream/research-persistent-kv-cache.md`), **CHƯA CÀI ĐẶT/CHẠY THẬT trên GPU**
(tài liệu tự ghi rõ "Chỉ nghiên cứu, KHÔNG cài đặt/chạy GPU"). Runbook thử nghiệm
gọn đã có sẵn trong chính tài liệu đó (mục "Runbook thử nghiệm gọn") — bước kế tiếp
nếu quyết định thử là chạy đúng runbook này trên GPU thật.

## (a) Bảng kịch bản → mức rủi ro → giải pháp rẻ nhất

Xếp theo ưu tiên triển khai (rủi ro cao + giải pháp rẻ trước):

| # | Kịch bản | Rủi ro | Giải pháp rẻ nhất |
|---|---|---|---|
| 17 | Rate-limit ở server thay vì gateway | **Cao** — đã đo p95 tệ gấp 20× (N1) | Chuyển toàn bộ cap sang gateway/client ngay, để `max-num-seqs` rộng |
| 4 | Skills-pack 30K bị đá khỏi cache bởi traffic agent | **Cao** — phá vỡ lợi ích trung tâm dự án (99%→cold) | Request keepalive định kỳ + tăng `gpu-memory-utilization`; cô lập instance nếu vẫn không đủ |
| 13 | Trim/tóm tắt lịch sử đụng vào system-prompt cố định | **Cao** nhưng dễ tránh bằng kỷ luật code | Quy tắc cứng: không bao giờ regenerate byte đầu prompt; trim chỉ ở phần riêng-phiên |
| 15 | Restart mất cache đồng loạt | Trung bình — đã có công cụ sẵn | Chạy `scripts/warmup_prefix.py` trước khi mở traffic (đã có, chỉ cần đóng gói vào quy trình deploy) |
| 11 | JSON tool-call sai cú pháp → lượt thừa | Trung bình — cộng dồn theo % lỗi | Bật `response_format`/`structured_outputs` + backend `xgrammar` |
| 14 | Tool-result khổng lồ chen prefill | Trung bình, tuỳ tần suất | Tóm tắt/trích trước khi chèn (a); nếu không được thì hạ `max-num-batched-tokens` về 1088 |
| 2 | KV pool cạn → preemption dây chuyền | Trung bình, tăng theo tải | Tăng `gpu-memory-utilization`; giới hạn độ dài lịch sử/phiên |
| 8 | Burst đồng bộ sau sự kiện chung | Trung bình, tuỳ pattern trigger | Jitter rải burst ở tầng ứng dụng; không cap server |
| 12 | Model lặp vô hạn cùng tool call | Trung bình, tuỳ bậc bit | Giữ ≥Q4_K_M cho 9B; circuit breaker theo chữ ký tool-call lặp |
| 10 | Trộn agent + chat trên 1 GPU | Trung bình, tuỳ đồng thời | 2 instance (đã đo 6-15× tail) nếu đủ VRAM |
| 3 | Slot mamba cạn | Thấp-trung bình (đã hiểu cơ chế, ceiling đọc được) | Luôn đọc block ceiling ở log khởi động trước khi đặt `max-num-seqs` |
| 16 | Client disconnect không hủy đúng | Thấp nhưng âm thầm (lãng phí GPU) | Đảm bảo orchestrator đóng connection thật khi timeout/hủy |
| 1/9 | Idle/đuôi dài chiếm cache | Thấp-trung bình, tích luỹ | Giới hạn số lượt/độ dài phiên; theo dõi TTFT trôi |
| 5 | Đa khách hàng nhiều prefix tranh chỗ | Thấp trừ khi nhiều tenant lớn | Capacity planning Σprefix < ngân sách pool; sharding nếu vượt |
| 7 | Tool lỗi → retry storm | Thấp, phòng ngừa rẻ | Backoff+jitter, circuit breaker |
| 6 | Tool chậm 30s+ | Thấp (GPU không bị chiếm) | Tách log tool_latency/llm_latency; không cần đổi cấu hình vLLM |
| 20 | Không có KV bền qua restart | Thấp trừ khi restart rất thường xuyên | Chỉ đầu tư LMCache MP mode sau khi (15) không đủ |

## (b) Cấu hình khuyến nghị cho workload agent — khác gì so với chat

| Cờ | Chat (đã khuyến nghị, `STATUS.md`) | Agent loop | Vì sao khác |
|---|---|---|---|
| `--enable-prefix-caching` | BẮT BUỘC bật tường minh (không mặc định ở build này) | Giống hệt, càng quan trọng hơn (mỗi lượt agent là 1 request rời rạc dựa hoàn toàn vào cache để nối phiên) | Cơ chế chung |
| `--mamba-cache-mode align` | BẮT BUỘC cùng với prefix caching (model lai) | Giống hệt — còn là điều kiện tiên quyết để phiên idle-chờ-tool tái tạo state rẻ (A-3) | Cơ chế chung |
| `--kv-cache-dtype fp8_e4m3` | Luôn bật trên sm89+, miễn phí tốc độ | Giống hệt | Không phụ thuộc workload |
| `--max-num-batched-tokens` | 8192 (TEST 11a, cân bằng throughput/TTFT cho tài liệu dài 16K) | **1088** (sàn mamba block_size=1056) cho hình dạng "prefix chung + suffix ngắn mỗi lượt" (TASK F2c) — SLA crossing 0,2→0,3 req/s | Agent loop = nhiều request nhỏ chen nhau, cần ngân sách chunk nhỏ để giảm tranh chấp compute; nếu workload có tool-result RẤT lớn thường xuyên (C-14), cân nhắc route riêng với mnbt cao hơn — đánh đổi CHƯA giải quyết trọn vẹn, xem C-14 |
| `--scheduling-policy` | `fcfs` (mặc định, chưa có lý do đổi cho chat) | Cân nhắc `priority` — ưu tiên lượt agent ngắn không bị preempt bởi phiên đuôi dài (A-2, B-9) | CHƯA ĐO số liệu cụ thể, chỉ có cơ sở cơ chế từ source |
| `structured_outputs`/`response_format` | Không cần (tự do sinh văn bản) | **Bật cho mọi lượt có khả năng gọi tool** (`xgrammar` backend) | Loại trừ lượt-thừa do JSON hỏng (C-11), rẻ hơn retry |
| Speculative decoding (MTP) | Bật khi conc≤4 (TASK D/N2, +28,8% decode conc1, TTFT vẫn <1s) | **Thường TẮT** cho instance phục vụ nhiều phiên agent đồng thời — tổng hợp traffic nhiều lượt/nhiều phiên hành xử như conc≥8 hầu hết thời gian, mà TTFT p95 nổ từ conc8 trở lên với MTP (4,64s tại conc8, TASK N2) | Chỉ bật nếu chắc chắn 1 instance dành riêng cho traffic thấp (≤4 đồng thời) |
| ngram speculative | Loại ở mọi mức (TASK N5a) | Giống hệt — càng tệ hơn với prefix chung dài (tắt async scheduling + overhead matching trên prefix 30K mỗi bước, sụp ở conc32: p95 101,3s) | Cơ chế chung, agent-loop cũng dùng shared-prefix nên chịu y hệt |
| `--gpu-memory-utilization` | 0,90 (mặc định các test) | Cân nhắc cao hơn nếu 1-instance (không tách agent/chat) để pool đủ chịu nhiều phiên history dài đồng thời (A-2, A-4) | Agent loop tích luỹ working-set nhiều phiên hơn traffic chat ngắn |
| Tách instance theo workload | — | **Khuyến nghị nếu VRAM cho phép** (đã đo 6-15× tail cho cặp chat/tài-liệu, B-10/E-19) | Tranh chấp compute giữa agent (prefill nặng, tool-result lớn) và chat (latency-sensitive) là thật |

## (c) Danh sách thí nghiệm cần chạy

**Đã có bench, chỉ cần chạy đúng kịch bản này:**
- Cold/warm prefix, warmup workflow → `scripts/warmup_prefix.py` (+ `test_warmup_prefix.py`)
- Hit-rate/TTFT theo concurrency trên shared-prefix skills-pack → `scripts/bench_skills.py`
- SLA open-loop Poisson trên kịch bản shared-prefix (rate sweep, `max-num-batched-tokens` sweep) → `scripts/bench_sla_prefix.py`
- Cap server-side thất bại dưới burst → đã đo trong TASK N1 (chưa đóng gói thành script argparse riêng — nằm rải trong lịch sử `STATUS.md`)
- Long-context correctness (128K, cascade attention) → phương pháp TASK N6 (chưa đóng gói script riêng)
- Offline batch mode cho automation theo lô → phương pháp TASK P1 (LLM.chat() trong-tiến-trình, chưa đóng gói script riêng)

**CHƯA CÓ bench — cần viết `scripts/bench_agent_loop.py` (chưa tồn tại trong repo),
đề xuất bộ cờ tối thiểu để phủ các kịch bản CHƯA ĐO ở trên:**

```
--tool-think-time <dist>          # A-1, B-6: phân phối thời gian chờ tool mỗi lượt
--competing-noise-sessions N      # A-1, A-4: traffic nền cạnh tranh pool/LRU
--num-sessions N --turns T
  --history-mode {accumulate,slide,summarize}
                                   # A-2, A-3, B-9, C-13: tích luỹ lịch sử nhiều lượt/phiên
--context-policy {slide,summarize,truncate-head-naive}
                                   # C-13: đo hệ quả từng chính sách trim lên hit-rate/TTFT
--json-mode {none,response_format,structured_outputs}
                                   # C-11: chi phí grammar vs chi phí lượt-thừa
--tool-result-size N --tool-result-position {start,mid,end}
                                   # C-14: tool-result khổng lồ chen giữa hội thoại
--burst-mode --burst-size N       # B-8: dồn cục đồng bộ vs Poisson rải đều
--retry-on-error --retry-jitter   # B-7: retry storm
--prefix-set <files...>           # A-5: nhiều prefix riêng biệt tranh chỗ
```

Ưu tiên viết trước (theo đúng thứ tự bảng (a)): `--json-mode` (C-11, rẻ và độc lập,
tận dụng lại phần scrape `/metrics` đã có trong `bench_skills.py`), rồi
`--competing-noise-sessions` (A-4, kịch bản rủi ro cao nhất chưa có số đo), rồi
`--tool-result-size`/`--max-num-batched-tokens` sweep (C-14, tái dùng gần như
nguyên vẹn hạ tầng `bench_sla_prefix.py` đã có).
