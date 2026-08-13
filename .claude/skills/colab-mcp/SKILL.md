---
name: colab-mcp
description: Cách dùng colab-mcp để điều khiển notebook Colab - mapping server/notebook, quy tắc 1 kết nối, chạy cell blocking, bố cục notebook 5 cell, bootstrap runtime mới. Nạp trước khi thao tác bất kỳ thứ gì trên Colab.
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
- Ô đầu notebook là WHOAMI (SESSION_A/B/C + hostname + GPU). Chạy WHOAMI trước mọi
  thao tác để chắc đúng máy — Colab có thể cấp backend mới (kể cả CPU-only) sau recycle.

## Chạy cell

- `run_code_cell` là **BLOCKING, timeout 30 phút**: gom lệnh dài vào 1 cell rồi chờ
  trong lượt. Không sleep/poll, không kết thúc lượt để "đợi".
- Việc >30 phút: chạy nền (`nohup ... > /content/logs/x.log &`) rồi xem lại bằng
  `bash run.sh status` / `logs` ở lần chạy cell sau.

## Bố cục notebook: ĐÚNG 1 CELL DUY NHẤT

```
!cd /content && (test -d custom_vllm || git clone -q https://github.com/cds0987/custom-vllm.git custom_vllm) \
  && cd custom_vllm && git pull -q && bash run.sh serve 9b && bash run.sh status
```

Cần gì THÊM LỆNH vào cell đó — cấm thêm cell. Mọi lệnh idempotent, chạy lại luôn an
toàn. `bash run.sh help`: setup / serve 9b|27b / status / logs / bench <tên> / eval /
registry --flat / stop. Đã đo: runtime mới → server 9B sẵn sàng ~6-8 phút; chạy lại
(cache ấm) ~2,5 phút.

- Đường cũ `loading/colab_bootstrap.sh` vẫn còn (fallback graft từ nguồn); run.sh là
  giao diện chính.
- Script vLLM offline: `LLM()` phải nằm trong `if __name__ == "__main__":` (vLLM spawn).
