---
name: huggingface
description: Cách dùng HuggingFace trong dự án - login đúng cách (account free, user gunnybd01), tải nhanh, các repo trung chuyển checkpoint/env giữa các phiên Colab, quy tắc an toàn token. Nạp khi cần tải/upload model, checkpoint, hoặc snapshot môi trường.
---

# Cách dùng HuggingFace

## Account & token (quy trình mới, user chốt 2026-08-24 — KHÔNG paste token nữa)

- Account free, username `gunnybd01`.
- **Local (Windows)**: token nằm ở env var `HF_TOKEN` (User scope) — Python/CLI
  local tự đọc; MCP server `huggingface` trong `.mcp.json` cũng đọc qua
  `${HF_TOKEN}` (remote https://huggingface.co/mcp — search/browse hub).
- **Colab**: dùng **Colab Secrets** (icon chìa khóa, panel trái) — user thêm key
  `HF_TOKEN` MỘT LẦN cho notebook A + bật "Notebook access". Cell login:

  ```python
  from google.colab import userdata
  from huggingface_hub import login
  login(token=userdata.get("HF_TOKEN"))   # token KHÔNG xuất hiện trong cell
  ```

- Tuyệt đối **không ghi token vào file/commit/log/cell text**. Token nào từng
  xuất hiện trong chat = coi như lộ → nhắc user rotate (tạo mới, revoke cũ).
- Account free đủ dùng: repo model public/private không giới hạn dung lượng thực tế
  cho cỡ dự án này (champion 9.1GB upload 1.1 phút từ Colab).

## Login — bẫy quan trọng

`huggingface-cli login` ĐÃ HỎNG (CLI đổi tên thành `hf`; gọi tên cũ in help rồi thoát 0,
tức là im lặng KHÔNG login). Luôn dùng Python API:

```python
from huggingface_hub import login
login(token="<token do user cấp>")   # chạy trong cell, token không lưu vào repo
```

## Tải nhanh

- Backend Xet của huggingface_hub hiện đã rất nhanh (13.8GB ~48s trên Colab);
  `HF_HUB_ENABLE_HF_TRANSFER` đã deprecated nhưng vô hại nếu set.
- Hai file độc lập thì tải song song (2 process/thread) — xem `colab_bootstrap.sh`.

## Các repo trung chuyển (chia sẻ giữa các phiên / sống sót qua recycle)

| Repo | Nội dung | Dùng |
|---|---|---|
| `gunnybd01/qwen35-9b-champion` | Champion v2 dựng sẵn (9.1GB) | `snapshot_download(...)` — thay cho tải 13.8GB + graft 5 phút |
| `gunnybd01/qwen35-9b-env` (nếu đã save) | Tarball dist-packages | `python utils/env_snapshot.py restore --repo ...` (guard manifest; exit 2 = build thường) |

Upload checkpoint mới:

```python
from huggingface_hub import HfApi
api = HfApi(); api.create_repo("gunnybd01/<ten-repo>", exist_ok=True)
api.upload_folder(folder_path="/content/champion", repo_id="gunnybd01/<ten-repo>")
```

## Nguyên tắc

- **(Quy tắc 6d, user chốt 2026-08-24) MẶC ĐỊNH save MỌI THỨ làm ra lên HF trong
  CÙNG PHIÊN** — checkpoint, mapper/LoRA (kể cả kết quả ÂM: vẫn cần để kiểm chứng
  lại/falsification), pseudo-gold, data dựng công phu. Học phí thật: mapper E6v3
  `.last` không upload → recycle qua đêm → retrain ~2h chỉ để chạy một bài kiểm.
  Thiếu token write lúc chốt → hỏi user NGAY trong phiên.
- Kết quả/checkpoint chỉ nằm trên runtime Colab = sẽ mất khi recycle. Chốt xong là
  upload HF (checkpoint) hoặc commit git (số liệu, code) ngay.
- Google Drive mount KHÔNG dùng được (cần click auth thủ công, treo cell) — HF là
  đường trung chuyển duy nhất.
