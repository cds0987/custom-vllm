import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "eb", str(Path(__file__).parent / "ext_bench.py"))
eb = importlib.util.module_from_spec(spec)
sys.modules.setdefault("torch", types.ModuleType("torch"))
spec.loader.exec_module(eb)

TESTS = [
    ({"bench": "bbh", "expect": "(A)"}, " (A)", 1),
    ({"bench": "bbh", "expect": "(A)"}, " (B)", 0),
    ({"bench": "bbh", "expect": "valid"}, " The argument is valid.", 1),
    ({"bench": "bbh", "expect": "Yes"}, " no", 0),
    ({"bench": "gsm8k", "expect": "18"}, "steps...\nFinal Answer: 18", 1),
    ({"bench": "gsm8k", "expect": "18"}, "steps...\nFinal Answer: 20", 0),
    ({"bench": "gsm8k", "expect": "1200"}, r"so total is \boxed{1,200}", 1),
    ({"bench": "gsm8k", "expect": "72"}, "he earns 72 dollars total", 1),
    ({"bench": "math500", "expect": r"\frac{1}{2}"}, r"thus \boxed{\frac{1}{2}}", 1),
    ({"bench": "math500", "expect": r"\frac{1}{2}"}, r"thus \boxed{\frac{1}{3}}", 0),
    ({"bench": "math500", "expect": "7"}, "answer 7.0", 1),
    # quy uoc AIME: dap an 3 chu so co so 0 dan -> \boxed{033} == 33 = HIT
    ({"bench": "aime", "expect": "33"}, r"long think... \boxed{033}", 1),
    ({"bench": "aime", "expect": "33"}, r"long think... \boxed{33}", 1),
    ({"bench": "musr", "expect": "B"}, " B\n\n**Reasoning:** ...", 1),
]

ok = 0
for it, txt, want in TESTS:
    got = eb.score_text(it, txt)
    ok += got == want
    print(f"{'OK ' if got == want else 'FAIL'} {it['bench']:8} "
          f"expect={it['expect']!r:18} txt={txt[:38]!r:42} got={got} want={want}")
print(f"\nscoring self-test: {ok}/{len(TESTS)}")
