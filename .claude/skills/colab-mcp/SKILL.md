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
- Việc >30 phút: chạy nền trong cell (`nohup ... > /content/task.log 2>&1 &`) rồi tail
  log bằng cell LOG ở các lần gọi sau.
- Bố cục chuẩn đúng 5 cell — **cấm thêm cell mới**, việc mới ghi đè cell TASK:
  1. `WHOAMI` 2. `BOOTSTRAP` 3. `SERVE` 4. `TASK` 5. `LOG`

## Bootstrap runtime mới (~15 phút)

```bash
git clone https://github.com/cds0987/custom-vllm.git /content/custom_vllm
cd /content/custom_vllm && bash loading/colab_bootstrap.sh
```

- `colab_bootstrap.sh`: env (uv, test-first sdist, skip llmcompressor trừ khi
  `CUSTOM_VLLM_TOOLS=1`) → pull champion prebuilt `gunnybd01/qwen35-9b-champion`
  (fallback: tải frame+GGUF song song rồi graft) → patch loader → in lệnh serve.
- Rebuild env nhanh hơn nữa: `python utils/env_snapshot.py restore --repo <hf-repo>`
  (có guard manifest python/CUDA; exit 2 = không có/không hợp → build thường).
- Script vLLM offline: `LLM()` phải nằm trong `if __name__ == "__main__":` (vLLM spawn).
