# <tên-kiến-trúc> — khuôn adapter cho kiến trúc mới

Copy folder này thành `models/<tên>/` khi bắt đầu thuần hóa một kiến trúc.
Registry (`models/auto/registry.py`) TỰ QUÉT — không phải đăng ký ở đâu cả.

## Cấu trúc bắt buộc

```
models/<tên>/
├── MANIFEST.md          # file này: ma trận hỗ trợ, số ĐÃ ĐO, cảnh báo
├── engine/
│   └── <engine>/        # engine nhiều file → folder
│       ├── adapter.py   # ADAPTER = {...} + SERVE_CONFIGS đã đo
│       └── patches/     # patch CỦA RIÊNG engine đó (nằm cạnh thứ nó sửa)
├── load/
│   └── <đường>.py       # mỗi đường load 1 file, có ADAPTER = {...}
├── hardware/
│   └── <gpu>.py         # bộ số đo được cho GPU đó, có ADAPTER = {...}
└── utils/               # đồ chung CỦA MODEL, trung lập engine
```

## Luật (tests/test_structure.py ép tự động)

1. **File adapter phải có `ADAPTER = {...}`** literal ở mức module
   (khóa: axis, variant, requires, input/output/tradeoff tùy trục).
   Registry đọc bằng ast — KHÔNG import — nên adapter được phép import nặng.
2. **Cấm import chéo** giữa `models/<a>` và `models/<b>`. Trùng lặp thì copy
   và ghi `# Copied from models/<a>/...` (khuôn transformers).
3. **Patch nằm cạnh thứ nó sửa**: sửa engine nào → `engine/<đó>/patches/`;
   sửa thư viện trung lập (transformers...) → `utils/` của model.
4. Phụ thuộc chéo trục không giấu được thì khai báo: `"requires": {"engine": "vllm"}`.
5. File = mặc định; folder chỉ khi một biến thể cần >2 file thật.
6. Con số nào vào MANIFEST/adapter đều phải ĐO ĐƯỢC, kèm nguồn (STATUS.md).
7. Thứ ≥2 model cùng cần → thăng cấp lên `utils/` gốc (chiều import một chiều:
   models → utils gốc OK; utils gốc → models CẤM).
