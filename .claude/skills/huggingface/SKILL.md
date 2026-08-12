---
name: huggingface
description: Cách dùng HuggingFace trong dự án - login đúng cách (account free, user gunnybd01), tải nhanh, các repo trung chuyển checkpoint/env giữa các phiên Colab, quy tắc an toàn token. Nạp khi cần tải/upload model, checkpoint, hoặc snapshot môi trường.
---

# Cách dùng HuggingFace

## Account

- Account free, username `gunnybd01`. Token do user cấp trong chat (không nằm trong
  repo) — **không ghi token vào file/commit/log**; nhắc user revoke sau chiến dịch.
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
| `gunnybd01/qwen35-9b-env` (nếu đã save) | Tarball dist-packages | `python scripts/env_snapshot.py restore --repo ...` (guard manifest; exit 2 = build thường) |

Upload checkpoint mới:

```python
from huggingface_hub import HfApi
api = HfApi(); api.create_repo("gunnybd01/<ten-repo>", exist_ok=True)
api.upload_folder(folder_path="/content/champion", repo_id="gunnybd01/<ten-repo>")
```

## Nguyên tắc

- Kết quả/checkpoint chỉ nằm trên runtime Colab = sẽ mất khi recycle. Chốt xong là
  upload HF (checkpoint) hoặc commit git (số liệu, code) ngay.
- Google Drive mount KHÔNG dùng được (cần click auth thủ công, treo cell) — HF là
  đường trung chuyển duy nhất.
