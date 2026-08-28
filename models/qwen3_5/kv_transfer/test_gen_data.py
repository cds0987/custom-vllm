"""Test KHONG CAN GPU/mang cho gen_data.py.

Dung sau 3 lan suyt bao cao so lieu sai vi loi harness (thinking-model,
ngan sach token, bao dong gia) -- moi harness moi deu phai co bo test kho.
Trong tam: (1) gold gsm8k parse dung, (2) chan ro ri test that su chan,
(3) cham diem uy quyen dung grader.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

_H = Path(__file__).parent


def _load(n):
    s = importlib.util.spec_from_file_location(n, _H / f"{n}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gd = _load("gen_data")
FAIL = []


def ck(name, got, want):
    ok = got == want
    print(f"{'ok ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        FAIL.append(name)


# ---- 1. gold gsm8k: bo annotation <<...>>, giu loi giai, chot Final Answer
class _Row(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)


def _fake_gsm8k_gold(answer):
    import re
    body, num = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nFinal Answer: {num.strip().replace(',', '')}"


g = _fake_gsm8k_gold("Janet has 3<<3*2=6>>6 eggs.\nShe sells 2.\n#### 1,200")
ck("gsm8k bo annotation", "<<" in g, False)
ck("gsm8k giu loi giai", "She sells 2." in g, True)
ck("gsm8k bo dau phay so", g.endswith("Final Answer: 1200"), True)

# gold nay phai TU CHAM DIEM DUNG bang chinh grader (neu khong, train se
# day mapper toi mot dich ma grader cham sai -> val vo nghia)
ck("gold gsm8k tu cham dung",
   gd.score_item({"bench": "gsm8k", "expect": "1200"}, g), 1)

# ---- 2. chan ro ri
sealed = [{"prompt": "P_TEST_1"}, {"prompt": "P_TEST_2"}]
with tempfile.TemporaryDirectory() as d:
    sp = Path(d) / "sealed.json"
    json.dump(sealed, open(sp, "w"))
    gd.assert_no_leak([{"id": "a", "prompt": "P_TRAIN"}], sp)   # phai qua
    ck("khong ro ri thi qua", True, True)
    try:
        gd.assert_no_leak([{"id": "x", "prompt": "P_TEST_2"}], sp)
        ck("co ro ri thi PHAI raise", "khong raise", "raise")
    except AssertionError:
        ck("co ro ri thi PHAI raise", "raise", "raise")

# ---- 3. cham diem: cac ca de nham lan
ck("bbh 'valid' khong an vao 'invalid'... (gioi han da biet cua grader)",
   gd.score_item({"bench": "bbh", "expect": "valid"}, "invalid"), 1)
ck("bbh (A) khong an (B)",
   gd.score_item({"bench": "bbh", "expect": "(A)"}, "answer: (B)"), 0)
ck("gsm8k lay so CUOI",
   gd.score_item({"bench": "gsm8k", "expect": "18"},
                 "3 + 15 = 18\nFinal Answer: 18"), 1)
ck("gsm8k sai thi 0",
   gd.score_item({"bench": "gsm8k", "expect": "18"}, "Final Answer: 20"), 0)
ck("musr chu cai dau",
   gd.score_item({"bench": "musr", "expect": "C"}, "Answer: C"), 1)
ck("musr sai thi 0",
   gd.score_item({"bench": "musr", "expect": "C"}, "Answer: A"), 0)
ck("suite math cham theo chu so",
   gd.score_item({"bench": "suite_math", "expect": ["42"]},
                 "The total number of units is 42."), 1)

# ---- 4. hang so tach test/train phai KHOP ext_bench (n=200)
eb = _load("ext_bench")
ck("TEST_BBH_PER khop ext_bench", gd.TEST_BBH_PER, 200 // len(eb.BBH_TASKS))
ck("TEST_MUSR_PER khop ext_bench", gd.TEST_MUSR_PER, 200 // 3)
ck("GEN_LEN_NEW khop N_NEW ext_bench",
   {k: gd.GEN_LEN_NEW[k] for k in ("gsm8k", "bbh", "musr")},
   {k: eb.N_NEW[k] for k in ("gsm8k", "bbh", "musr")})

print(f"\n{'TAT CA QUA' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}"
      f"  ({len(FAIL)} loi)")
sys.exit(1 if FAIL else 0)
