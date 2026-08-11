# Nghiên cứu định tuyến/lập lịch: Dynamo + vLLM → đề xuất cho hệ của ta

Nguồn đọc: `D:\Training\AI_Module\dynamo` (NVIDIA Dynamo, clone 2026-08) và
`D:\Training\AI_Module\vllm\vllm\vllm` (vLLM, checkout dev gần 0.26, layout
`v1/core/sched/`). Mọi khẳng định dưới đây trỏ file:dòng thật, đọc trực tiếp
mã nguồn (không suy đoán từ tài liệu marketing). Bối cảnh áp dụng: 1×L4, một
instance vLLM, workload agent-loop nhiều lượt/nhiều user, xem `STATUS.md`
(đặc biệt TASK F/F2/F2b/F2c, N1, N6, H, P1).

---

## PHẦN 1 — Khảo sát

### 1a. Dynamo — KV-aware router, disagg routing, SLA planner, session affinity

**Router có hai bản song song**: crate mới `lib/kv-router/` (đang phát triển
tích cực) và bản cũ `lib/llm/src/kv_router/`. Công thức trích từ bản mới.

**Hàm chi phí chọn worker** — `lib/kv-router/src/scheduling/selector/default.rs`,
hàm `worker_logit` (dòng 216-328). Đây là một hàm **cost** (thấp hơn = tốt hơn),
worker được chọn là worker có logit nhỏ nhất (hoặc lấy mẫu softmax theo
temperature):

```rust
// dòng 284-286
let adjusted_prefill_blocks = (load.raw_prefill_blocks - overlap_credit_blocks).max(0.0);
let prefill_cost_blocks = weights.prefill_load_scale * adjusted_prefill_blocks;
let logit = prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks;
```

`overlap_credit_blocks` (dòng 227-253) — điểm thưởng cho worker theo số block
prefix trùng, tính trên CẢ BA tầng cache (device/host/disk):

```rust
let overlap_credit_blocks = effective_overlap_score_credit * cache.device_overlap_blocks
    + kv_router_config.host_cache_hit_weight * cache.host_overlap_blocks
    + kv_router_config.disk_cache_hit_weight * cache.disk_overlap_blocks
    + shared_overlap_blocks;
let active_request_cost_blocks = kv_router_config.decode_active_request_weight * load.active_requests as f64;
```

Trọng số mặc định (`lib/kv-router/src/scheduling/config.rs`, dòng 818-859,
61-90): `overlap_score_credit=1.0`, `overlap_score_credit_decay=0.0`,
`prefill_load_scale=1.0`, `decode_active_request_weight=0.0` (TẮT mặc định),
`host_cache_hit_weight=0.75`, `disk_cache_hit_weight=0.25`,
`router_temperature=0.0` (argmin thuần, hoà thì random tie-break).

`overlap_credit_decay` (dòng 233-245) làm mềm điểm thưởng khi backlog prefill
của worker đã cao hơn worker ít tải nhất — tránh dồn hết traffic prefix-trùng
vào một worker đang nghẽn:

```rust
let excess_active_prefill_blocks = load.active_prefill_tokens.saturating_sub(context.min_active_prefill_tokens) as f64 / context.block_size as f64;
let normalized_prefill_load = excess_active_prefill_blocks / context.request_blocks as f64;
1.0 / (1.0 + weights.overlap_score_credit_decay * normalized_prefill_load)
```

**Theo dõi block ở mỗi worker: radix tree**, `lib/kv-router/src/indexer/`
(`mod.rs` dòng 4-33: "implements a KV store using a Radix Tree structure...
`RouterEvent` ... applied to the Radix Tree to update its state"). Có cả biến
thể nén/concurrent (`concurrent_radix_tree_compressed/`) và một chỉ mục thay
thế dựa hash (`indexer/cuckoo/`). Cập nhật là **push, event-based**: worker
publish `RouterEvent{KvCacheEventData::Stored|Removed}` qua ZMQ
(`lib/llm/src/kv_router/publisher/zmq_listener.rs:136`), điều khiển bởi
`use_kv_events` (mặc định `true`). Khi tắt, router dùng TTL tường
(`router_ttl_secs=120.0`). Có thêm cơ chế "predict-on-route": router tự ghi
nhận block ngay lúc ĐỊNH TUYẾN (trước khi có event xác nhận từ worker) để
tránh route trùng vào cùng worker trong khoảng trễ event
(`config.rs:759-766`).

**Định tuyến prefill/decode phân tách (disaggregated)**: không có ngưỡng độ
dài prompt đơn giản; thay vào đó có "conditional disagg" — quyết định có
BYPASS về prefill cục bộ (aggregated) thay vì gửi sang worker prefill riêng.
`lib/kv-router/src/conditional_disagg.rs`, hằng số mặc định (dòng 14-20):
`DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_THRESHOLD=2048` token,
`DEFAULT_CONDITIONAL_DISAGG_EFF_ISL_RATIO_THRESHOLD=0.7`. Logic
(`should_bypass_remote_prefill`, dòng 155-166):

```rust
let eff_isl = input.net_new_tokens(); // prompt_tokens - decode_chosen_cached_tokens
if eff_isl >= self.eff_isl_threshold { return false; }
let ratio = eff_isl as f64 / input.prompt_tokens.max(1) as f64;
ratio < self.eff_isl_ratio_threshold
```

Tức: bypass về decode-worker-tự-prefill chỉ khi số token PHẢI prefill thật
(net-new, sau khi trừ phần cache-hit) vừa nhỏ tuyệt đối (<2048) vừa nhỏ
tương đối (<70% prompt) — ý tưởng cốt lõi: **request gần như toàn bộ đã có
trong cache thì không đáng trả giá điều phối sang worker khác.**

**SLA planner / autoscaling**: `components/src/dynamo/planner/core/load_scaling.py`.
Có 2 chế độ: "sla" (mô hình hồi quy dự đoán TTFT/ITL) và "load" (ngưỡng tĩnh).
Quyết định scale (dòng 986-1036):

```python
if all(t > sla for t in estimates):
    return num_workers + 1
if num_workers > 1 and can_scale_down:
    return max(num_workers - 1, self._config.min_endpoint)
return None
```

Mô hình hồi quy TTFT — `perf_model/prefill.py`, `estimate_next_ttft`
(dòng 66-97):

```python
scale = 1.0 - _clamp_kv_hit_rate(kv_hit_rate)
total_tokens = (queued_prefill_tokens + self._avg_isl.value) * scale
num_iterations = math.ceil(total_tokens / max_num_batched_tokens)
# tổng wall_time dự đoán cho từng chunk kích thước max_num_batched_tokens
```

Đây CHÍNH LÀ mô hình chi phí ước lượng TTFT theo prefix-hit-rate mà đề bài
hỏi — quan trọng cho Phần 1b/Phần 2 dưới. Bên decode có mô hình 2 biến
tương tự (`perf_model/decode.py`, `estimate_next_itl` dòng 72-82) theo
`(num_decode_requests, sum_decode_kv_tokens)`.

**Session/sticky affinity**: không pin cứng theo `session_id` trong hàm cost
(trường `session_id` tồn tại ở `scheduling/types.rs:251,280` nhưng chỉ
passthrough). Cơ chế thật là **router hint**: `lib/kv-router/src/router_hint.rs`
— request mang theo `block_hashes` + `source_control_endpoint` của worker đã
phục vụ lượt trước, để worker đích có thể kéo/khớp KV prefix nóng
(`RouterHintRootCandidates::best_source`, dòng 32-64: chọn worker sở hữu
nhiều block prefix nhất). Có cả cơ chế PIN cứng
(`eligibility.pinned_worker()`, `scheduling/filter.rs:79`, được tôn trọng
TRƯỚC khi chấm điểm cost, `selector/default.rs:405-411`).

### 1b. `aisimulate/` và `benchmarks/`

`aisimulate/` **không phải** simulator per-request dự đoán TTFT theo
hit-rate — nó là công cụ dò hình dạng triển khai (parallelism sweep). Trong
`aisimulate/src/aisimulate/sweeper/kv_estimate.py` (dòng 60-143) chỉ tính khả
thi bộ nhớ (`total_kv_size_tokens > max_seq_len`), và `score.py` (dòng 52-89,
109-152) chấm điểm/Pareto-front theo `throughput_per_gpu` từ log replay thật,
không có công thức TTFT-theo-hit-rate. **Công thức TTFT-theo-hit-rate thật sự
nằm ở planner** (`perf_model/prefill.py`, trích ở trên) — ta có thể MƯỢN mô
hình này (không phải mượn code, mượn Ý TƯỞNG: TTFT ≈ f(token thật phải
prefill sau khi trừ hit, chunking theo max_num_batched_tokens, backlog hiện
tại)) để tự calibrate trên chính dữ liệu đo được của ta (xem Phần 2, P1).

`benchmarks/` là wrapper mỏng quanh công cụ ngoài AIPerf, chạy tải thật lên
server đang sống rồi đo (`benchmarks/README.md`) — không có cost model nội
tại; giống vai trò `bench_serving.py`/`bench_load.py` của ta nhưng không có
gì để mượn thêm ngoài cấu trúc harness.

### 1c. vLLM — scheduler, preemption, prefix cache, routing đa-instance

**Chính sách lập lịch**: `vllm/v1/core/sched/request_queue.py` dòng 13-18,
`SchedulingPolicy` enum có `FCFS` và `PRIORITY`. `PriorityRequestQueue`
(dòng 131) dùng heapq theo `(priority, arrival_time)` — **priority thấp hơn
= chạy trước**, tie-break theo thời gian đến. Mặc định `"fcfs"`
(`vllm/config/scheduler.py:109`). Quan trọng: API OpenAI-compatible ĐÃ CÓ
sẵn field `priority: int` trên request
(`vllm/entrypoints/openai/chat_completion/protocol.py:351-358`: "lower
means earlier handling... raises error if server not using priority
scheduling") — nghĩa là gateway của ta có thể gán priority per-request MÀ
KHÔNG CẦN PATCH vLLM, chỉ cần bật `policy: priority` ở scheduler config lúc
serve.

Vòng lập lịch chính `schedule()` trong
`vllm/v1/core/sched/scheduler.py` (dòng 432-1183): Bước 1 lập lịch các
request đang RUNNING trước (dòng 476-661, kể cả request đang giữa chừng
prefill — V1 không phân biệt "pha prefill"/"pha decode", dòng 434-444), trừ
dần vào NGÂN SÁCH TOKEN DUY NHẤT `token_budget = max_num_scheduled_tokens`
(≈`max_num_batched_tokens`). Bước 2 chỉ xét WAITING SAU KHI bước 1 tiêu hết
phần của nó (dòng 673+). Đây CHÍNH LÀ cơ chế đứng sau phát hiện thực nghiệm
TASK F2b/F2c của ta ("chunked-prefill người mới chen vào decode của người cũ"):
một request mới được nhận vào dù chỉ 1 token cũng lập tức nằm trong
`self.running` và bước 1 (chạy TRƯỚC bước 2) sẽ ưu tiên phục vụ toàn bộ phần
prefill còn lại của nó ở MỌI bước sau, cạnh tranh trực tiếp ngân sách với các
decode đang chạy — khớp chính xác quan sát "num_requests_waiting=0 suốt
180s nhưng vẫn chậm" trong STATUS.md TASK F2b.

**Preemption**: chỉ có MỘT chế độ — RECOMPUTE. Không tìm thấy
`PreemptionMode`/`SWAP` ở đâu trong cây V1 (đã grep toàn repo). Điều kiện
kích hoạt: `allocate_slots()` trả `None` (hết block KV) trong vòng lặp
running (dòng 568-618); dưới FCFS request bị đuổi là
`self.running.pop()` (đuôi danh sách = request nhận gần nhất); dưới PRIORITY
là `max(self.running, key=lambda r: (r.priority, r.arrival_time))`.
`_preempt_request` (dòng 1190-1212) giải phóng TOÀN BỘ block KV và reset
`num_computed_tokens=0`, đưa lại đầu hàng đợi waiting — request bị đuổi phải
prefill lại từ đầu (dù có thể hit lại chính block vừa giải phóng nếu chưa bị
tái sử dụng, nhờ cơ chế LRU dưới đây).

**Prefix cache**: `vllm/v1/core/block_pool.py`, class `BlockPool` (dòng 143).
Không phải cây trie node-con-trỏ cổ điển mà là **hash map phẳng theo hash
chuỗi** (`cached_block_hash_to_block`, dòng 183-185) — mỗi block hash được
tính CHAIN qua toàn bộ block trước đó nên tra cứu tuần tự cho ngữ nghĩa
radix-tree mà không cần cấu trúc cây tường minh
(`KVCacheManager.get_computed_blocks` → `find_longest_cache_hit`,
`kv_cache_manager.py:207,233-237`). **Eviction là LRU thật**, cài bằng hàng
đợi liên kết đôi `FreeKVCacheBlockQueue`
(`vllm/v1/core/kv_cache_utils.py`, dòng 184, docstring 193-198: "least
recent used ở đầu; cùng thời điểm thì block nhiều hash-token hơn (đuôi
chuỗi) ở đầu"). Evict thật sự diễn ra lúc CẤP PHÁT
(`block_pool.py`, `get_new_blocks`→`_maybe_evict_cached_block`, dòng
647-700) — pop LRU khỏi hàng đợi rảnh, nếu nó còn mang hash thì xoá khỏi
prefix cache rồi tái sử dụng.

**Routing đa-instance**: có sẵn CHỈ trong phạm vi một tiến trình API-server
quản lý nhiều DP rank cùng máy — `DPLBAsyncMPClient`
(`vllm/v1/engine/core_client.py`, dòng 1380+), công thức chọn engine
(dòng 1413-1447): `score = waiting*4 + running`, quét vòng round-robin bắt
đầu xoay để tránh thiên vị (TODO trong code: "use P2C alg for larger DP
sizes" — hiện tại là min-score tuyến tính, chưa power-of-two-choices).
**Giữa các TIẾN TRÌNH SERVER RIÊNG BIỆT (nhiều instance/replica) thì KHÔNG
có router nội bộ** — `DPSupervisor`
(`vllm/entrypoints/openai/dp_supervisor.py:266`) chỉ spawn nhiều process,
docstring nói thẳng "Assumes external load-balancing by default" — trùng
khớp phát hiện của ta ở mục "Cô lập chat khỏi prefill tài liệu" trong
STATUS.md (hai instance thường, driver time-slicing, không có router, ta tự
route bằng tay/hai cổng riêng).

**Fairness prefill dài/ngắn**: `vllm/config/scheduler.py` dòng 70-82 —
`long_prefill_token_threshold` (mặc định 0=tắt, tự set = 4% max_model_len
nếu `max_num_partial_prefills>1`) chặn TRẦN số token một request dài được
cấp mỗi bước bất kể ngân sách còn dư, và `max_long_partial_prefills` giới
hạn số request DÀI được prefill đồng thời — cho phép request NGẮN
"chen ngang" hàng đợi trước request dài (docstring dòng 76-78). Đây chính là
cờ ta đã dùng trong TASK F2c (`mnbt=1088`, gần khớp block_size mamba 1056)
— khảo sát này xác nhận cơ chế đứng sau con số đã đo, không phải may rủi.

---

## PHẦN 2 — Đề xuất cải tiến (ưu tiên: rẻ, đo được, hợp workload agent)

Bối cảnh: một instance, không có gì để "route giữa nhiều worker" — mọi cải
tiến phải diễn ra ở TẦNG GATEWAY/CLIENT (đã được N1 xác nhận: cap phía
server thua) hoặc ở việc CHỌN CỜ SCHEDULER sẵn có của vLLM (đã có API,
không cần patch). Priority-queue của vLLM và cơ chế "net-new tokens" của
Dynamo conditional-disagg là hai viên gạch có thể ghép trực tiếp.

### P1 (LÀM TRƯỚC) — Cổng nhận request bằng mô hình chi phí TTFT, thay cho cap mù

**Ý tưởng**: N1 đã chứng minh cap CỨNG số lượng request đồng thời ở server
(`--max-num-seqs`) thua rõ rệt (p95 64,6s ở mns=24 vs 3,03s không cap) vì cap
đếm SỐ REQUEST chứ không đếm TẢI THẬT — hai request "đồng thời" có thể một
cái chỉ cần 50 token mới (turn ngắn, hit cache 99%) còn cái kia cần prefill
2000 token suffix mới toanh (F2b). Cap theo số lượng đối xử chúng như nhau.
Thay bằng gateway ước lượng TTFT dự kiến của MỖI request TRƯỚC khi gửi, và
quyết định nhận ngay / xếp hàng / từ chối theo dự đoán đó.

**Cơ sở mượn từ đâu**: công thức của Dynamo
`perf_model/prefill.py:66-97` — `TTFT ≈ g(token phải prefill thật sau khi
trừ hit, backlog hàng đợi hiện tại, chunk size)`. Ta không mượn code (Python
khác stack, khác model) mà mượn HÌNH DẠNG mô hình và tự calibrate bằng chính
dữ liệu đã đo (TASK F2c, N6, H đều có cặp (net_new_tokens, tải hiện tại) →
TTFT đo được, đủ để hồi quy tuyến tính).

**Thiết kế**:
```
# Gateway giữ 2 số liệu cập nhật mỗi request hoàn tất (hoặc poll /metrics mỗi ~1s):
#   backlog_tokens = tổng token-phải-prefill của các request đang RUNNING/WAITING trên server
#   (đọc gần đúng qua vllm:num_requests_running + vllm:num_requests_waiting
#    hoặc tính tay từ nhật ký request đã gửi - đã trả lời)

def net_new_tokens(session, new_prompt):
    matched = longest_prefix_match(session.last_sent_prompt, new_prompt)  # ta tự biết,
                                                                            # vì ta xây prompt agent-loop
    return len(new_prompt) - matched  # xấp xỉ tốt, không cần đọc block hash thật của vLLM

def predict_ttft(net_new, backlog_tokens):
    # hồi quy tuyến tính calibrate offline từ dữ liệu STATUS.md + log production
    return alpha * net_new + beta * backlog_tokens + gamma

def admit(request):
    p = predict_ttft(net_new_tokens(...), current_backlog)
    if p <= SLA_TARGET:            # 3s, để biên an toàn dùng 2.4s (0.8x)
        send_now(request)
    elif p <= SLA_TARGET * 2:
        enqueue_local(request)     # giữ ở gateway, thử lại mỗi 200ms
    else:
        reject_429(request, retry_after=estimate_wait())
```

**Chi phí cài đặt**: thấp — gateway Python thuần (~150-200 dòng), không đụng
vLLM. Calibrate alpha/beta/gamma bằng hồi quy tuyến tính trên dữ liệu ĐÃ CÓ
(không cần GPU thêm để có bản v0; refine online bằng cách log dự đoán vs
thực tế mỗi request thật).

**Cách đo thắng/thua**: chạy lại đúng kịch bản F2b/F2c (Poisson, shared-prefix
32K, rate sweep 0.1→1.0) hai lần — (a) baseline: gateway chỉ cap concurrency
cứng (như hiện tại), (b) gateway dùng TTFT predictor. So sánh ở CÙNG rate
chào: % request admit ngay, TTFT p95 thực đo so với SLA 3s, và sai số RMSE
của mô hình dự đoán so với TTFT thực đo (mục tiêu: RMSE < 20% để mô hình còn
dùng được). Thắng nếu ở cùng SLA target, (b) admit được NHIỀU request hơn
(a) tại cùng % vi phạm SLA.

### P2 (LÀM SAU P1, chi phí gần bằng 0) — Hàng đợi hai mức: phiên nóng ưu tiên phiên nguội

**Ý tưởng**: trong agent-loop, một phiên đang dở (đã có KV cache nóng, lượt
tiếp theo chỉ cần prefill vài trăm token mới) và một phiên hoàn toàn mới
(phải prefill system-prompt/skills từ đầu, hàng nghìn-hàng vạn token — TASK H
đo prefix skills ~28.760 token) đang bị đối xử NGANG NHAU bởi FCFS. Điều này
lãng phí: giữ phiên nóng chờ xử lý phiên nguội tốn nhiều lần thời gian hơn để
"làm ấm lại" tưởng tượng chỉ vì tới sau vài mili-giây.

**Cơ sở mượn từ đâu**: HAI thứ có sẵn, ghép lại không cần code mới ở vLLM:
(1) vLLM đã CÓ `SchedulingPolicy.PRIORITY`
(`vllm/v1/core/sched/request_queue.py:13-18`, heapq theo
`(priority, arrival_time)`) và API đã CÓ field `priority` per-request
(`chat_completion/protocol.py:351-358`) — chỉ cần bật
`--policy priority` lúc serve, gateway tự gán priority mỗi request.
(2) Công thức GÁN priority mượn ý tưởng WSPT (Smith's rule) của Dynamo
(`lib/kv-router/src/scheduling/policy.rs:96-113`:
`(strict_priority, weight/new_tokens)`) VÀ ngưỡng "net-new nhỏ" của
`conditional_disagg.rs:155-166` (eff_isl<2048 và eff_isl_ratio<0.7) — ý
tưởng chung: **ưu tiên request có tỷ lệ (token-phải-làm-mới / lợi-ích) thấp
nhất**, tức phiên nóng đi trước vì nó "rẻ" để phục vụ ngay.

**Thiết kế**:
```
def assign_priority(request):
    net_new = net_new_tokens(request.session, request.prompt)   # như P1
    # priority thấp = chạy trước (đúng ngữ nghĩa PriorityRequestQueue)
    if net_new < WARM_THRESHOLD:        # vd 1024 token — phiên đang dở
        return 0
    else:                                # phiên mới/cold, hoặc turn nhảy chủ đề
        return 10
    # (mở rộng sau: priority = round(net_new / 512) để có nhiều mức thay vì 2 mức)
```
Server serve với `--policy priority`. Không cần patch vLLM.

**Rủi ro cần đo, không giả định**: priority cứng có thể làm phiên cold ĐÓI
vô hạn nếu phiên warm dồn dập liên tục — vLLM PriorityRequestQueue tie-break
theo `arrival_time` trong CÙNG priority, không có aging/boost theo thời gian
chờ giữa các mức priority khác nhau (đọc code không thấy cơ chế starvation
-prevention nào ở tầng gateway hay scheduler). Vì vậy nhất thiết phải thêm
timeout aging phía gateway: nếu một request cold chờ quá T giây, nâng
priority về 0.

**Chi phí cài đặt**: rất thấp — 1 cờ serve (`--policy priority`) + ~30 dòng
gateway gán priority + aging timer. Không cần patch vLLM hay retrain gì.

**Cách đo thắng/thua**: kịch bản mô phỏng 2 luồng đồng thời — luồng A: nhiều
phiên "hot" turn ngắn liên tục (giả lập agent loop bận), luồng B: thỉnh
thoảng 1 phiên "cold" mới join. Đo TTFT p50/p95 của luồng B dưới FCFS (nay)
vs PRIORITY (P2) ở CÙNG tải luồng A. Thắng nếu TTFT trung vị của A giảm rõ
rệt mà TTFT p95 của B không vượt quá X% so với FCFS (X ngưỡng chấp nhận được,
vd 2×, vẫn dưới SLA 3s nếu có thể).

### P3 (CÂN NHẮC, ưu tiên thấp) — Cửa sổ gom request cùng prefix

**Ý tưởng nêu trong đề bài**: trì hoãn một khoảng ngắn (vd 50-200ms) để gom
các request có cùng prefix vào chung một bước lập lịch, giảm interleave giữa
các phiên khác prefix.

**Đánh giá thẳng — CHƯA ĐÁNG làm trước, vì**: đọc `block_pool.py` (Phần 1c)
cho thấy eviction chỉ xảy ra LÚC CẤP PHÁT khi hàng đợi block rảnh cạn — nghĩa
là xen kẽ request khác-prefix KHÔNG tự động "thrash" cache trừ khi tổng
KV-footprint của các phiên đang sống vượt quá pool. TASK H đã đo hit rate
99,04-99,37% ở conc32 chính vì các phiên CÙNG chia sẻ một prefix skills lớn
(~28-29K token) — tức hiện trạng đã không hề bị thrash dù request tới xen kẽ
tự nhiên. Batching-window chỉ có giá trị THẬT khi: (a) nhiều prefix KHÁC
NHAU cạnh tranh cùng lúc VÀ (b) tổng footprint gần chạm trần KV pool (N6 đã
đo: conc16-32 ở 128K context bắt đầu bị nghẽn — không phải do thrash cache mà
do cạnh tranh compute quét KV dài, một cơ chế KHÁC). Vì vậy P3 nên xếp sau
P1/P2, và chỉ đáng làm SAU KHI hệ mở rộng ra nhiều loại prefix khác nhau
đồng thời (nhiều "loại agent"/nhiều bộ skill khác nhau chạy chung 1 GPU).

**Nếu làm**: thiết kế tối thiểu — gateway giữ hàng đợi cục bộ theo
`hash(session.system_prompt_prefix)`, trì hoãn tối đa `W` ms để nhóm cùng
prefix trước khi thả vào server theo lô. **Đo trước khi cài**: bật
`--enable-prefix-caching` (đã bật) và log `vllm:prefix_cache_hits/queries`
qua `/metrics` khi cố tình cho 2+ workload có prefix khác nhau chạy chung
(vd chat 2B skills-pack A + skills-pack B xen kẽ) — nếu hit rate rớt rõ rệt
so với chạy tuần tự thì P3 mới có bằng chứng đáng làm; nếu không rớt (nhiều
khả năng, theo phân tích trên) thì bỏ qua P3 hoàn toàn, tiết kiệm công.

### P4 (THIẾT KẾ SẴN, KHÔNG CÀI NGAY) — Session affinity + KV-aware routing khi lên 2-4 L4

**Ý tưởng**: khi thêm GPU, vLLM tự nó KHÔNG có router liên-instance
(`DPSupervisor` — "assumes external load-balancing", Phần 1c) nên bắt buộc
phải tự xây tầng gateway. Mượn trực tiếp công thức của Dynamo, đơn giản hoá
cho quy mô 2-4 instance (không cần Rust/radix-tree phân tán, dùng Python +
dict là đủ):

```
# Gateway giữ, mỗi instance i: active_requests[i], và với mỗi session:
#   last_worker[session_id] -> worker vừa phục vụ lượt trước (router hint kiểu Dynamo)

def score(worker_i, request):
    overlap_blocks = cached_prefix_len(session, worker_i) // BLOCK_SIZE
    # cached_prefix_len: gateway tự theo dõi — mỗi lần gửi request tới worker_i,
    # ghi nhận (session_id -> worker_i, độ dài prompt đã gửi) làm proxy cho
    # "worker_i chắc chắn có prefix này trong cache", KHÔNG cần đọc radix tree
    # thật của vLLM (chưa lộ API); đủ chính xác vì ta biết prompt mình gửi.
    prefill_cost = max(0, request.net_new_tokens - overlap_blocks * BLOCK_SIZE)
    load_cost = active_requests[worker_i]        # xấp xỉ decode_cost của Dynamo
    return prefill_cost + LOAD_WEIGHT * load_cost  # công thức rút gọn của
                                                     # worker_logit (default.rs:284-286)

def route(request):
    if session_id in last_worker and overlap_blocks(last_worker[session]) > MIN_STICKY:
        return last_worker[session_id]            # sticky nếu vẫn đáng
    return min(workers, key=lambda w: score(w, request))
```

Đây là bản rút gọn của `worker_logit` (Dynamo, Phần 1a) bỏ phần host/disk-tier
cache (ta chưa có CPU/disk KV offload đa-tầng thật — TASK C2 cho thấy
OffloadingConnector là 1 tier CPU đơn giản, không multi-tier như Dynamo) và
bỏ `overlap_score_credit_decay` (không cần ở quy mô 2-4 worker, phức tạp
thừa). Priority pin kiểu `pinned_worker()` (`filter.rs:79`) tương ứng với
`last_worker[session_id]` ở trên — sticky routing đơn giản.

**Chi phí**: thiết kế xong, KHÔNG cài — chỉ đáng cài khi thật sự có GPU thứ
2. Cài đặt ước tính 1-2 ngày (gateway Python theo dõi 2 dict, không cần
service phát hiện phân tán).

**Cách đo khi cài**: lặp lại thí nghiệm "Cô lập chat khỏi prefill tài liệu"
đã có trong STATUS.md (TTFT max, ITL max, 1 instance trộn vs 2 instance) —
thêm biến thứ 3 "2 instance + gateway KV-aware routing" và so P95 TTFT/ITL
với "2 instance + round-robin ngây thơ" để đo lợi ích THẬT của phần
overlap-score (không chỉ lợi ích của việc tách 2 instance, cái đó đã biết).

### P5 — Ý tưởng từ Dynamo chưa từng nghĩ tới: mô hình hồi quy TTFT tự-calibrate của planner

Ngoài 4 đề xuất trên (đã lồng ý tưởng planner vào P1), điểm ĐÁNG CHÚ Ý riêng
là cách Dynamo planner tách biệt: (a) một MÔ HÌNH HỒI QUY được fit từ dữ liệu
đo thật của CHÍNH deployment đang chạy (không phải công thức lý thuyết cố
định) — `_BaseRegressionModel` trong `perf_model/prefill.py`/`decode.py`; và
(b) một bộ GIẢI NGƯỢC (`find_best_engine_prefill_rps`,
`find_best_engine_decode_rps`) suy ra "QPS tối đa giữ được SLA" từ mô hình đó
bằng binary search, thay vì đo bằng tay qua rate-sweep thủ công như ta đang
làm (TASK F2/F2c, N6 — mỗi lần đổi cấu hình phải chạy lại sweep vài chục
phút GPU). **Tính khả thi trên 1 GPU: cao và rẻ** — không cần hạ tầng
autoscaler (không có gì để scale với 1 GPU), nhưng phần "fit regression từ
log thật + suy ngược QPS-tối-đa" có thể tách riêng thành 1 script offline
(`scripts/fit_ttft_model.py`) chạy trên các file kết quả `bench_serving.py`/
`bench_skills.py` đã có sẵn trong repo, dùng để: (i) cấp tham số
alpha/beta/gamma cho P1 mà không cần đoán tay, và (ii) trả lời nhanh "đổi
`max_num_batched_tokens` hay lên context dài hơn thì QPS an toàn còn bao
nhiêu" mà KHÔNG cần chạy sweep GPU mới mỗi lần — tiết kiệm đúng thứ TASK
N6/F2c đang tốn (nhiều lượt sweep 180-300s mỗi điểm). Đây là phần rẻ nhất
trong toàn bộ 5 đề xuất vì thuần offline, dùng lại dữ liệu đã có.

### Thứ tự làm trước — tóm tắt

| # | Đề xuất | Vì sao trước/sau |
|---|---|---|
| **P1** | Cổng TTFT-predictor thay cap mù | RẺ NHẤT + trực tiếp sửa nhược điểm đã CHỨNG MINH của N1; đo được ngay bằng kịch bản F2b có sẵn |
| **P5** | Fit mô hình hồi quy offline từ log sẵn có | Gần như miễn phí (không cần GPU), là tiền đề số liệu cho P1 — làm cùng lúc/trước P1 |
| **P2** | Hàng đợi 2 mức warm/cold | Rẻ (1 cờ serve có sẵn + gán priority), hợp đúng workload agent-loop (giữ ấm phiên dở) nhưng cần đo rủi ro đói (starvation) trước khi bật production |
| **P3** | Batching window theo prefix | Ưu tiên thấp — bằng chứng hiện có (TASK H hit rate 99%+) cho thấy CHƯA có vấn đề để giải; chỉ đo-thử trước khi cài |
| **P4** | KV-aware routing đa-instance | Thiết kế sẵn, KHÔNG cài — chờ tới khi có GPU thứ 2 |

---

## PHẦN 3 — Dynamo có đáng dùng trực tiếp không?

**Chạy 1 GPU**: khả thi về mặt kỹ thuật. Quickstart
(`docs/fern/pages/cli/getting-started/quickstart.mdx:59`) minh hoạ chạy
frontend + 1 worker trên cùng máy với `--discovery-backend file` để KHÔNG
cần etcd. Không tìm thấy recipe nào gắn nhãn "single-GPU L4" cụ thể trong
`recipes/` (toàn bộ 9 thư mục recipe đọc được đều là model lớn multi-GPU:
deepseek, glm, gpt-oss-120b, kimi, llama-3-70b, nemotron-3-*, qwen3-*) —
`recipes/qwen3-0.6b` là recipe nhỏ nhất, gần nhất với quy mô của ta nhưng
chưa xác nhận là single-GPU.

**Hạ tầng vận hành**: `dev/docker-compose.yml` (dòng 1-36) dựng CẢ hai
`nats-server` (`nats:2.11.4`) và `etcd-server` (`bitnamilegacy/etcd:3.6.1`)
làm "bare minimum infrastructure". etcd có đường tránh
(`--discovery-backend file`), nhưng **không tìm thấy cách tránh NATS** trong
các file đã đọc — NATS là message/event plane cho cả router events lẫn
request plane (`docs/fern/.../event-plane.md`, `request-plane.md`). Tức
ngay cả ở cấu hình tối giản nhất vẫn cần vận hành thêm ít nhất 1 service
ngoài (NATS), cộng thêm việc học một hệ Rust+Python mới, hai đội quân
lib/components riêng.

**Hỗ trợ backend vLLM**: XÁC NHẬN CÓ, đầy đủ — `components/src/dynamo/vllm/`
(`main.py`, `worker_factory.py`, `handlers.py`, `publisher.py` — publisher sự
kiện KV cho router, `router_hints.py` — phía consume router-hint, và
`kv_connector_protocols.py`). Tích hợp sâu, không phải wrapper hời hợt.

**Hỗ trợ model lai (Mamba/GDN)**: XÁC NHẬN CÓ nhưng ở tầng RECIPE/deploy,
KHÔNG phải logic Dynamo tự viết. `recipes/nemotron-3-super/README.md:8`:
"NVIDIA-Nemotron-3-Super-120B-A12B... a ~120B **hybrid Mamba/Attention/MoE**
model". Đây là bằng chứng Dynamo BIẾT ĐIỀU PHỐI một model hybrid tồn tại
(qua config deploy cho vLLM/SGLang/TRT-LLM), nhưng KHÔNG có mã Rust/Python
nào trong Dynamo tự xử lý đặc thù Mamba-state (KV-router radix tree giả định
block KV attention đồng nhất kích thước — không thấy nhánh nào xử lý state
mamba kích thước cố định riêng). Việc chạy được model lai vẫn hoàn toàn phụ
thuộc backend (vLLM) đã hỗ trợ nó — đúng những gì repo `custom_vllm` này đã
tự vá (16 bug GGUF/GDN) chứ Dynamo không thêm giá trị ở lớp đó.

### Kết luận: **Chưa cần**

Lý do, xếp theo trọng số:
1. Toàn bộ giá trị cốt lõi của Dynamo (KV-aware router, disagg
   prefill/decode, SLA autoscaler) giải quyết bài toán **NHIỀU worker/nhiều
   node**. Hệ hiện tại có ĐÚNG MỘT GPU, MỘT instance — không có gì để
   "router" giữa các worker. Router/planner Dynamo sẽ chạy nhưng vô dụng
   (định tuyến 1-trên-1, autoscaler không có gì để scale).
2. Chi phí vận hành thêm KHÔNG rẻ: tối thiểu 1 service ngoài (NATS, không
   tránh được theo bằng chứng đọc được), một hệ Rust để build/debug, một mô
   hình triển khai Kubernetes-first (phần lớn tài liệu/recipe hướng cluster)
   — lệch hẳn quy mô "1 Colab L4, script Python" hiện tại.
3. Những Ý TƯỞNG hữu ích nhất (cost-function overlap-score, WSPT/net-new
   priority, mô hình hồi quy TTFT tự-calibrate) đều **mượn được dưới dạng
   công thức/thiết kế** (P1-P5 ở trên) mà KHÔNG cần vận hành cả hệ Dynamo —
   đúng tinh thần "học công thức, không học hệ thống".
4. Mốc để XÉT LẠI: khi hệ có ≥3-4 GPU/instance thật (không phải time-slicing
   1 card) VÀ cần disaggregate prefill/decode thật (không phải mô phỏng bằng
   2 instance trộn như hiện tại) — lúc đó phần KV-router + conditional-disagg
   của Dynamo mới bắt đầu trả giá trị tương xứng chi phí vận hành. Cho tới
   đó, P4 (thiết kế gateway tự viết, rút gọn từ công thức Dynamo) là đủ.
