---
name: colab-mcp
description: Cách dùng colab-mcp để điều khiển notebook Colab - mapping server/notebook, quy tắc 1 kết nối, chạy cell blocking, bố cục notebook 1 cell, chuẩn phóng task nền (Popen+PID+flag), bootstrap runtime mới. Nạp trước khi thao tác bất kỳ thứ gì trên Colab.
---

# Cách dùng colab-mcp

## Server ↔ notebook

colab-mcp bản patch local (`D:\Training\AI_Module\colab-mcp`, nhận env
`COLAB_MCP_NOTEBOOK`), cấu hình trong `.mcp.json`:

| Server MCP | Notebook | Vai trò |
|---|---|---|
| `colab-mcp` | `/drive/134iXTgK_uXdwx4-yZerjkfhyZJTW4dgs` | **A — GPU chính, mặc định duy nhất** |
| `colab-mcp-2` | `/drive/1qzdKjAo44KB_Q-nrPurGbo-IOY8ZsqpW` | B — chờ lệnh user |
| `colab-mcp-3` | `/drive/1dhgbZejNZ9_t0ddBIzcGVlvMjFYV6HGV` | C — chờ lệnh user |

## Quy tắc kết nối

- Colab giữ **1 kết nối live mỗi notebook** — mở cùng notebook ở nơi thứ hai là nơi
  thứ nhất disconnect, mất toàn bộ. Mỗi server chỉ đụng đúng notebook của nó.
- Kết nối: `open_colab_browser_connection` → user bấm link/authorize trên trình duyệt.
- Runtime bị recycle THƯỜNG XUYÊN (đã dính ≥3 lần): `/content` sạch trơn, GPU 0 MiB.
  Kiểm tra đầu phiên: `ls /content` + `nvidia-smi` — thấy trống thì bootstrap lại từ
  đầu, đừng giả định file/tiến trình cũ còn sống.

## Chạy cell

- `run_code_cell` là **BLOCKING, timeout 30 phút**: gom lệnh vào 1 cell rồi chờ
  trong lượt. Không sleep/poll, không kết thúc lượt để "đợi".
- Task nhỏ (<1-2 phút) chạy thẳng trong cell. Task LỚN (serve, bench dài, train,
  download, soak): **BẮT BUỘC chạy nền** theo chuẩn dưới — cell trả về ngay.

## Chuẩn phóng task nền (đúc kết sau nhiều lần mất job)

`!nohup ... &` từng phóng IM LẶNG THẤT BẠI (không PID, không log). Chuẩn hiện tại:

```python
import os, subprocess, time
os.makedirs("/content/logs", exist_ok=True)
FLAG = "/content/logs/<job>_launch.flag"          # idempotent: chạy lại cell = no-op
if not os.path.exists(FLAG):
    p = subprocess.Popen(["python","-u","<script>", "<args>"],
        stdout=open("/content/logs/<job>.log","w"), stderr=subprocess.STDOUT,
        start_new_session=True)                    # tách session, sống qua lượt cell
    open("/content/logs/<job>.pid","w").write(str(p.pid))
    open(FLAG,"w").write("1")
    time.sleep(10); print("alive:", p.poll() is None)   # bắt chết-ngay-lúc-phóng
```

Cell status kiểm sau: `kill -0 $(cat pid)` + `tail log` + `nvidia-smi` + `df -h`.

Bẫy đã dính — tránh:
- **killpg giết cả server con**: job launcher đã chết nhưng process group còn server
  sống — killpg cả chuỗi cũ sẽ giết nhầm. Server dài hạn phải phóng tách session.
- **2 writer 1 log = null bytes**; mỗi job một file log riêng.
- **pkill/pgrep tự khớp chính nó** khi pattern chứa tên lệnh đang chạy.
- **Disk full cắt log giữa dòng** (spill cache 100+GB đã dính): task spill nhiều
  phải kèm `df -h` vào cell status; file spill đặt tên theo tham số để khỏi dùng
  nhầm file cũ (stale).
- Cần lệnh chuỗi (pip build dài + train): `bash -c "buoc1; buoc2; exec python ..."`
  trong Popen — build `causal-conv1d` mất ~13 phút, đừng chờ trong cell.
- **pkill -f 'vllm serve' TỰ SÁT**: bash -c đang chạy có pattern trong cmdline
  → pkill giết chính shell, output rỗng không dấu vết. Dùng `'vllm serv[e]'`.
- **Kill cha không chết con giữ cổng** (lmcache, server spawn): tìm chủ cổng
  bằng `ss -tlnp` → đọc `/proc/PID/cmdline` → `kill -9` đích danh; xác minh
  `ss -tln` sạch RỒI MỚI phóng cái mới (ZMQ bind fail là chết câm).
- **2 tiến trình chia CUDA-IPC phải CÙNG torch**: tiến trình phụ (lmcache
  server...) phải khởi động SAU khi env vLLM đã setup (source /tmp/vllm_env.sh)
  — lệch bản torch = "sharable handle from a future version of torch".
- Cổng 8080 bị dịch vụ Colab chiếm — service phụ dời 8081+.
- Python con của chuỗi cũ sống sót kill cha vẫn giữ fd log → chuỗi mới cùng
  file log = null bytes; mỗi lần relaunch dùng TÊN LOG MỚI + pkill cả tên script.

## Bố cục notebook: ĐÚNG 1 CELL DUY NHẤT

Serve chuẩn (kiểu vLLM):

```
!cd /content && (test -d custom_vllm || git clone -q https://github.com/cds0987/custom-vllm.git custom_vllm) \
  && cd custom_vllm && git pull -q && bash run.sh serve 9b && bash run.sh status
```

Pipeline nghiên cứu (kv_transfer/E-series) dùng clone tại `/content/custom-vllm`
(bootstrap trong cell Python: clone nếu thiếu → `git pull` → pip deps → Popen).
Cần gì THÊM LỆNH/SỬA cell đó — cấm thêm cell; mọi lệnh idempotent. `bash run.sh help`
liệt kê: setup / serve 9b|27b|4b / status / logs / bench <tên> / eval / registry
--flat / stop. Đã đo: runtime mới → server 9B sẵn sàng ~6-8 phút; cache ấm ~2,5 phút.

- Đường cũ `loading/colab_bootstrap.sh` vẫn còn (fallback graft từ nguồn); run.sh là
  giao diện chính.
- Script vLLM offline: `LLM()` phải nằm trong `if __name__ == "__main__":` (vLLM spawn).
- Kernel GDN nhanh cho transformers (train/nghiên cứu): `pip install
  flash-linear-attention` + `pip install causal-conv1d --no-build-isolation`
  (build isolation không thấy torch là lý do fail chuẩn).

## Server MCP không lên sau reload (học phí 2026-08-26)

Triệu chứng: tool `mcp__colab-mcp__*` không xuất hiện; log
`%TEMP%\colab-mcp-logs-*\colab-mcp.*.log` cho thấy sau "Starting WebSocket
server" kẹt ~10s ở `GET https://pypi.org/pypi/fastmcp/json` (fastmcp
update-check), rồi "Starting worker" → NGAY "server closing" (client đã hết
kiên nhẫn, đóng stdin). Lúc reload 3 server + uvx cùng gọi pypi → nghẽn.
Fix: `.mcp.json` đặt `FASTMCP_CHECK_FOR_UPDATES=off` cho cả 3 server (đã
làm). Chẩn đoán nhanh: `ls -t %TEMP% | grep colab-mcp-logs` → cat log mới
nhất; đo tay: pipe JSON `initialize` vào `uvx --from D:\...\colab-mcp
colab-mcp` — bình thường trả lời sau ~3s.
