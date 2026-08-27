"""bench_analyze -- soi CHAT LUONG SINH, khong chi ty le dung/sai.

User 2026-08-27: "ko chi xem ket qua, kha hep, phai xem cach model suy luan,
co tao du lieu rac ko, co hallu ko".

Dac biet quan trong cho duong CROSS (mapper): ca chien dich Phase C truoc da
cho thay chu ky hong dac trung cua cache ngoai la "ra dung vai ky tu dau roi
DEGENERATE" (lap vo han, tut sang noi dung khac, so lung tung). Ty le hit
KHONG bat duoc su khac biet giua "sai vi suy luan kem" va "sai vi sinh rac".

Cac chi so (deu tinh duoc offline, khong can GPU):
  degen_rep     -- ty le trigram TRUNG LAP (1 - distinct_3). >0,6 = lap nhieu
  max_repeat    -- so lan lap lien tiep cua doan 12 ky tu dai nhat
  truncated     -- sinh het ngan sach token ma chua ket thuc cau
  no_answer     -- khong parse duoc dap an theo dinh dang yeu cau
  empty         -- output rong/gan rong
  hallu_entity  -- (musr) ten rieng trong cau tra loi KHONG co trong de bai
  fmt_break     -- (musr/bbh) tra loi khong dung dinh dang (vd tra so thay
                   vi chu cai) -- dau hieu lac dinh dang, khong phai sai noi dung
  arith_slip    -- (gsm8k/math) buoc tinh trong loi giai co phep tinh sai
                   (chi kiem cac phep "a op b = c" don gian)

Chay:
  python bench_analyze.py --glob '/content/logs/extbench_self*.json'
  python bench_analyze.py --glob '...' --dump 8       # in vi du that
  python bench_analyze.py --glob-a '<self>' --glob-b '<cross>'   # so sanh
"""

import argparse
import glob as globmod
import json
import re
from collections import Counter


# ------------------------------------------------------------- chi so ---

def _trigrams(text):
    toks = re.findall(r"\w+", text.lower())
    return [tuple(toks[i:i + 3]) for i in range(max(0, len(toks) - 2))]


def degen_rep(text):
    """1 - (trigram doc nhat / tong trigram). Cao = lap lai nhieu."""
    tg = _trigrams(text)
    if len(tg) < 5:
        return 0.0
    return round(1 - len(set(tg)) / len(tg), 3)


def max_repeat(text, w=12):
    """So lan doan w ky tu xuat hien lai nhieu nhat (bat '1. 1. 1. ...')."""
    if len(text) < w * 2:
        return 1
    c = Counter(text[i:i + w] for i in range(len(text) - w))
    return c.most_common(1)[0][1] if c else 1


def looks_truncated(text, n_new, bench):
    """Sinh cham ngan sach ma khong ket thuc bang dau cau/tag ket thuc."""
    approx_tokens = len(re.findall(r"\w+|[^\w\s]", text))
    near_budget = approx_tokens >= 0.9 * n_new
    ends_clean = bool(re.search(r"[.!?}\]\)]\s*$", text.strip()))
    return bool(near_budget and not ends_clean)


CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
STOP = {"The", "This", "That", "There", "Answer", "Reasoning", "Analysis",
        "Based", "Here", "While", "Both", "First", "Second", "Third", "Step",
        "Final", "Solution", "Problem", "Question", "However", "Therefore",
        "Since", "Given", "Note", "Let", "Thus", "Motive", "Means",
        "Opportunity", "Initial", "Location", "Yes", "No", "None", "All"}


def hallu_entity(text, prompt):
    """Ten rieng xuat hien trong cau tra loi nhung KHONG co trong de bai.
    Xap xi tho cho 'bia ten' -- chi dung cho musr (de bai co nhan vat)."""
    # BAO DONG GIA da gap (756 mau MuSR): tieu de markdown
    # '**Initial State:**' lam 'State' bi tinh la ten bia. Loai truoc:
    # (a) tu trong '**..**', (b) tu ngay truoc ':', (c) tu dau cau.
    clean = re.sub(r"\*\*[^*]{0,40}\*\*", " ", text)
    clean = re.sub(r"\b[A-Z][a-z]+\s*:", " ", clean)
    clean = re.sub(r"(?:^|[.!?\n])\s*[A-Z][a-z]+", " ", clean)
    names = {m for m in CAP_RE.findall(clean) if m not in STOP}
    if not names:
        return []
    return sorted(n for n in names if n not in prompt)


ARITH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*x×/])\s*(\d+(?:\.\d+)?)\s*=\s*"
                      r"(\d+(?:\.\d+)?)")


def arith_slip(text, limit=40):
    """Bat phep tinh SAI ngay trong loi giai (dau hieu 'suy luan troi')."""
    bad = []
    for m in ARITH_RE.finditer(text):
        if len(bad) >= limit:
            break
        a, op, b, c = m.groups()
        try:
            a, b, c = float(a), float(b), float(c)
        except ValueError:
            continue
        want = {"+": a + b, "-": a - b, "*": a * b, "x": a * b,
                "×": a * b, "/": (a / b if b else None)}[op]
        if want is None:
            continue
        if abs(want - c) > 1e-6 * max(1.0, abs(want)):
            bad.append(m.group(0))
    return bad


N_NEW = {"musr": 24, "aime": 2560, "compute": 900,
         "bbh": 48, "gsm8k": 320, "math500": 640}


def _has_answer(bench, text):
    """Da dua ra dap an dung dinh dang chua?"""
    if bench == "musr":
        return bool(re.search(r"\b[A-F]\b",
                              re.sub(r"<think>.*?</think>", " ", text,
                                     flags=re.S)))
    if bench in ("gsm8k", "math500", "aime"):
        return bool(re.search(r"\\boxed\{", text)
                    or re.search(r"Final Answer:", text))
    return len(text.strip()) >= 1


def analyse_row(r, item=None):
    text = r.get("text", "") or ""
    bench = r.get("bench", "?")
    # ho tra loi NGAN (dap an 1 chu cai / 1 tu): NGAN KHONG PHAI RAC
    short_ok = bench in ("musr", "bbh")
    has_ans = _has_answer(bench, text)
    a = {
        "id": r.get("id"), "bench": bench, "hit": r.get("hit", 0),
        "n_chars": len(text.strip()),
        "degen_rep": degen_rep(text),
        "max_repeat": max_repeat(text),
        # cat giua chung CHI CO HAI khi chua kip dua ra dap an
        "truncated": bool(looks_truncated(text, N_NEW.get(bench, 512), bench)
                          and not has_ans),
        "truncated_any": looks_truncated(text, N_NEW.get(bench, 512), bench),
        "empty": len(text.strip()) < (1 if short_ok else 3),
    }
    a["garbage"] = bool(a["degen_rep"] > 0.6 or a["max_repeat"] >= 8
                        or a["empty"])
    if bench == "musr":
        a["no_answer"] = not re.search(r"\b[A-F]\b", re.sub(r"<think>.*?</think>", " ", text, flags=re.S))
        a["fmt_break"] = bool(re.match(r"\s*\d", text))   # tra so thay vi chu cai
        if item:
            h = hallu_entity(text, item.get("prompt", ""))
            a["hallu_entity"] = h[:5]
            a["n_hallu"] = len(h)
    elif bench in ("gsm8k", "math500", "aime"):
        slips = arith_slip(text)
        a["arith_slip"] = slips[:3]
        a["n_arith_slip"] = len(slips)
        a["no_answer"] = not (re.search(r"\\boxed\{", text)
                              or re.search(r"Final Answer:", text))
    elif bench == "bbh":
        a["no_answer"] = len(text.strip()) < 2
    return a


# ------------------------------------------------------------ bao cao ---

def summarise(rows, label):
    by = {}
    for a in rows:
        by.setdefault(a["bench"], []).append(a)
    print(f"\n===== {label} =====")
    hdr = (f"{'bench':<9}{'n':>5}{'hit%':>7}{'rac%':>7}{'lap>0.6':>9}"
           f"{'cut-hai%':>10}{'ko-dap-an%':>12}{'hallu':>7}{'sai-tinh':>10}"
           f"{'ky-tu-TB':>10}")
    print(hdr)
    for bench, rs in sorted(by.items()):
        n = len(rs)
        f = lambda k: sum(1 for a in rs if a.get(k)) / n
        print(f"{bench:<9}{n:>5}{sum(a['hit'] for a in rs) / n:>7.1%}"
              f"{f('garbage'):>7.1%}"
              f"{sum(1 for a in rs if a['degen_rep'] > 0.6) / n:>9.1%}"
              f"{f('truncated'):>10.1%}{f('no_answer'):>12.1%}"
              f"{sum(a.get('n_hallu', 0) for a in rs) / n:>7.2f}"
              f"{sum(a.get('n_arith_slip', 0) for a in rs) / n:>10.2f}"
              f"{sum(a['n_chars'] for a in rs) / n:>10.0f}")
    return by


def dump_examples(rows, results, items, k):
    """In VI DU THAT -- doc bang mat, khong chi con so."""
    by_id = {r["id"]: r for r in results}
    for kind, pred in (("RAC/DEGEN", lambda a: a["garbage"]),
                       ("SAI-TINH", lambda a: a.get("n_arith_slip", 0) > 0),
                       ("HALLU-TEN", lambda a: a.get("n_hallu", 0) > 0),
                       ("SAI nhung SACH", lambda a: not a["hit"] and not a["garbage"])):
        sel = [a for a in rows if pred(a)][:k]
        if not sel:
            continue
        print(f"\n---------- VI DU: {kind} ({sum(1 for a in rows if pred(a))} ca) ----------")
        for a in sel:
            txt = (by_id.get(a["id"], {}).get("text", "") or "")[:400]
            extra = ""
            if a.get("n_arith_slip"):
                extra = f" | phep sai: {a['arith_slip']}"
            if a.get("n_hallu"):
                extra = f" | ten bia: {a['hallu_entity']}"
            print(f"[{a['bench']}/{a['id']}] hit={a['hit']} lap={a['degen_rep']} "
                  f"rep={a['max_repeat']}{extra}\n   {txt!r}\n")


def load(pattern):
    rows = []
    for f in sorted(globmod.glob(pattern)):
        d = json.load(open(f))
        rows += d["runs"] if isinstance(d, dict) else d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/content/logs/extbench_self*.json")
    ap.add_argument("--glob-b", default="", help="duong thu 2 (vd cross) de so")
    ap.add_argument("--items", default="/content/ext_bench_items.json")
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--out", default="/content/logs/bench_quality.json")
    args = ap.parse_args()

    try:
        items = {it["id"]: it for it in json.load(open(args.items))}
    except OSError:
        items = {}

    res_a = load(args.glob)
    rows_a = [analyse_row(r, items.get(r["id"])) for r in res_a]
    by_a = summarise(rows_a, f"A: {args.glob} ({len(rows_a)} mau)")
    if args.dump:
        dump_examples(rows_a, res_a, items, args.dump)

    out = {"a": rows_a}
    if args.glob_b:
        res_b = load(args.glob_b)
        rows_b = [analyse_row(r, items.get(r["id"])) for r in res_b]
        summarise(rows_b, f"B: {args.glob_b} ({len(rows_b)} mau)")
        if args.dump:
            dump_examples(rows_b, res_b, items, args.dump)
        out["b"] = rows_b
        # so sanh tren CUNG item
        ida = {a["id"]: a for a in rows_a}
        idb = {b["id"]: b for b in rows_b}
        both = sorted(set(ida) & set(idb))
        if both:
            print(f"\n===== SO SANH TREN CUNG {len(both)} ITEM =====")
            print(f"{'chi so':<16}{'A':>10}{'B':>10}{'chenh':>10}")
            for k, lbl in (("hit", "dung"), ("garbage", "sinh rac"),
                           ("truncated", "bi cat"), ("no_answer", "ko dap an")):
                va = sum(bool(ida[i].get(k)) for i in both) / len(both)
                vb = sum(bool(idb[i].get(k)) for i in both) / len(both)
                print(f"{lbl:<16}{va:>10.1%}{vb:>10.1%}{vb - va:>+10.1%}")
            deg_a = sum(ida[i]["degen_rep"] for i in both) / len(both)
            deg_b = sum(idb[i]["degen_rep"] for i in both) / len(both)
            print(f"{'lap trigram TB':<16}{deg_a:>10.3f}{deg_b:>10.3f}{deg_b - deg_a:>+10.3f}")
            flip = [i for i in both if ida[i]["hit"] and not idb[i]["hit"]]
            flip_g = [i for i in flip if idb[i]["garbage"]]
            print(f"\nA dung -> B sai: {len(flip)} ca, trong do B SINH RAC: "
                  f"{len(flip_g)} ({len(flip_g) / max(len(flip), 1):.0%}) "
                  f"-- ty le nay cho biet mapper hong vi SINH RAC hay vi SUY LUAN KEM")
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nchi tiet -> {args.out}")
    print("BENCH_ANALYZE_DONE")


if __name__ == "__main__":
    main()
