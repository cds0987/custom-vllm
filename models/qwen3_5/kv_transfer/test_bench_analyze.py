import importlib.util

spec = importlib.util.spec_from_file_location(
    "ba", str(__import__("pathlib").Path(__file__).parent / "bench_analyze.py"))
ba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ba)

fails = []


def check(name, got, want):
    ok = got == want
    if not ok:
        fails.append(name)
    print(f"{'OK ' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")


# --- bao dong gia phai HET ---
check("hallu tieu de markdown",
      ba.hallu_entity(" B\n\n**Reasoning:**\n1.  **Initial State:** The story begins",
                      "the story text here"), [])
check("hallu ten bia THAT",
      ba.hallu_entity("the killer was clearly Zorblax indeed",
                      "Alice and Bob at the party"), ["Zorblax"])

a = ba.analyse_row({"id": "x", "bench": "musr", "hit": 1, "text": " B"})
check("musr ' B' khong phai rac", a["garbage"], False)
check("musr ' B' khong bao cut-hai", a["truncated"], False)

a2 = ba.analyse_row({"id": "y", "bench": "musr", "hit": 1,
                     "text": " A\n\n**Reasoning:**\n1.  **Motive**: Both suspects had"})
check("musr cat SAU khi tra loi -> vo hai", a2["truncated"], False)

a3 = ba.analyse_row({"id": "z", "bench": "gsm8k", "hit": 0,
                     "text": "he then computed the next step and " * 60})
check("gsm8k cat TRUOC dap an -> co hai", a3["truncated"], True)
check("gsm8k khong co dap an", a3["no_answer"], True)

a4 = ba.analyse_row({"id": "w", "bench": "gsm8k", "hit": 1,
                     "text": "12 * 3 = 36 then 36 + 4 = 41.\nFinal Answer: 40."})
check("bat phep tinh sai", a4["n_arith_slip"], 1)

a5 = ba.analyse_row({"id": "v", "bench": "musr", "hit": 0,
                     "text": "1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. "})
check("degen that -> la rac", a5["garbage"], True)

print(f"\nanalyze self-test: {'ALL OK' if not fails else 'FAIL ' + str(fails)}")
