"""Kiem KHONG CAN GPU cho eval_big: cham diem cac ho moi + rang buoc noi bo.

Dung sau khi 3 lan suyt bao cao so lieu sai vi loi harness (thinking-model,
ngan sach token, bao dong gia) — moi lan them ho de la them mot cho de sai
cham diem, nen dong test truoc khi dot 4 gio GPU.
"""
import importlib.util
import sys
from pathlib import Path

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_shell_scripts():
    """Bat bay '\n' viet thanh CHUOI thay vi xuong dong trong file .sh.

    Da tai dien 3 lan trong mot buoi: moi lan va bang heredoc qua tool, mot lop
    backslash bi an mat nen '\\' den Python chi con '\'. Trieu chung o
    Colab la argparse bao 'unrecognized arguments: n n' SAU khi da nap xong
    model — tuc mat vai phut moi biet.
    """
    bad = []
    for f in sorted(_H.glob("*.sh")):
        for i, ln in enumerate(f.read_text(encoding="utf-8").split(chr(10)), 1):
            if chr(92) + "n" in ln and not ln.lstrip().startswith("#"):
                bad.append(f"{f.name}:{i}: {ln.strip()[:70]}")
    return bad


def main():
    eb = _load("eval_big")
    ok = fail = 0

    def chk(name, got, want):
        nonlocal ok, fail
        if got == want:
            ok += 1
        else:
            fail += 1
            print(f"FAIL {name}: duoc {got!r}, muon {want!r}")

    # --- bfcl: gold la TEN HAM, cham "co xuat hien" (y het e6v3) ---
    it = {"bench": "bfcl", "expect": "calc_area"}
    chk("bfcl dung", eb.score(it, "calc_area(w=3, h=4)"), 1)
    chk("bfcl sai", eb.score(it, "get_area(w=3)"), 0)
    chk("bfcl rong", eb.score(it, ""), 0)

    # --- needle: ma 6 so ---
    it = {"bench": "needle", "expect": "481920"}
    chk("needle dung", eb.score(it, "The code is 481920."), 1)
    chk("needle sai", eb.score(it, "The code is 481921."), 0)

    # --- suite_*: uy quyen suite_gen.score, expect la DANH SACH ---
    it = {"bench": "suite_rag", "expect": ["Thornfield"]}
    chk("suite dung", eb.score(it, "It is Thornfield, clearly."), 1)
    chk("suite sai", eb.score(it, "It is Ashcombe."), 0)

    # --- moi bench co N_NEW: thieu -> KeyError giua chung run 4 gio ---
    benches = ["bbh", "gsm8k", "musr", "bfcl", "needle",
               "suite_rag", "suite_mid", "suite_math", "suite_swe"]
    for b in benches:
        chk(f"N_NEW[{b}]", b in eb.N_NEW, True)

    # --- score() luon tra int 0/1 (agg cong don) ---
    v = eb.score({"bench": "bfcl", "expect": "f"}, "f(")
    chk("score tra int", isinstance(v, int), True)

    sh_bad = check_shell_scripts()
    for b in sh_bad:
        fail += 1
        print(f"FAIL .sh co backslash-n van ban -> {b}")
    if not sh_bad:
        ok += 1

    print(f"\n{ok} dat / {fail} hong")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
