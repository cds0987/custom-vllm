"""Bai kiem suite_gen.score sau khi vay loi khop CHUOI CON tren so garble.

Hoc phi (2026-08-31): decode tren cache ngoai hay garble/lap so o cuoi dinh
danh (vd dap an dung "flush_buffers_10" nhung model sinh
"flush_buffers_1012_1012..."). Khop chuoi con tho tinh "flush_buffers_10" la
DUNG vi no la tien to cua chuoi garble do -> tinh sai thanh dung. Doc tay
39 mau "ca hai deu dung" cua joint49w/joint49y: 0/39 sinh ten ham SACH.

Fix: ky tu NGAY SAU vi tri khop khong duoc la chu so.

Chay: python -u test_suite_gen_scoring.py
"""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "sg", str(Path(__file__).parent / "suite_gen.py"))
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

TESTS = [
    # --- ca garble THAT doc duoc trong log joint49w/joint49y (phai la 0) ---
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "flush_buffers_1012_1012_1012_1012_1", 0),
    ({"family": "swe", "expect": ["resolve_shards_30"]},
     "resolve_shards_30393930936630303036", 0),
    ({"family": "swe", "expect": ["dispatch_tokens_57"]},
     "dispatch_tokens_577577dispatch_tokens_577757_57777", 0),
    ({"family": "swe", "expect": ["hydrate_records_18"]},
     "hydrate_records_185885hydrate_records_1858858_1858", 0),

    # --- dap an SACH (token tron ven) van phai la 1 — KHONG duoc vay qua tay ---
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "The function is flush_buffers_10", 1),
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "flush_buffers_10\n\nQuestion:", 1),
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "flush_buffers_10.", 1),
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "flush_buffers_10, then returns", 1),

    # --- neu KHONG co so garble (dap an sai han) van phai la 0 ---
    ({"family": "swe", "expect": ["flush_buffers_10"]},
     "collect_records_26", 0),

    # --- dap an KHONG ket thuc bang so (rag/mid): hanh vi cu khong doi ---
    ({"family": "rag", "expect": ["Thornfield"]},
     "The manor is called Thornfield, built in 1820.", 1),
    ({"family": "rag", "expect": ["Thornfield"]},
     "no relevant name found", 0),
    ({"family": "mid", "expect": ["482913"]},
     "the code is 482913 hidden in the text", 1),
    # neu ma so bi garble thanh so DAI HON, van phai bi tu choi (dung logic moi)
    ({"family": "mid", "expect": ["482913"]},
     "the code is 4829130000000000", 0),

    # --- math: khong doi (van tach chu so, khong qua nhanh moi) ---
    ({"family": "math", "expect": ["42"]}, "the answer is 42 dollars", 1),
    ({"family": "math", "expect": ["42"]}, "the answer is 420 dollars", 0),
]

ok = fail = 0
for item, text, want in TESTS:
    got = sg.score(item, text)
    if got == want:
        ok += 1
    else:
        fail += 1
        print(f"FAIL family={item['family']} expect={item['expect']} "
              f"text={text[:50]!r} want={want} got={got}")

print(f"\n{ok} dat / {fail} hong")
sys.exit(1 if fail else 0)
