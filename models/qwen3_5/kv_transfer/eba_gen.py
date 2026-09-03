"""eba_gen -- "Entity Binding Arithmetic": bo de synthetic RIENG cho van de
gsm8k dang bi mapper lam hong (user 2026-09-03: "vi sao adapter be tac" ->
"tim cach sinh synthetic data ... RL ep mapper dem dung relationship entities").

Khac voi suite_gen.py (4 ho de do NANG LUC cua cache ngoai o muc tho: co/khong
mang duoc mot su that), eba_gen do dung THU MA gsm8k that dang hong: MOT con so
gan dung THUC THE dung, dung QUAN HE (chon dung thuc the cho dung vai tro,
khong lay nham thuc the gay nhieu -- "distractor"), roi moi den dap so cuoi.
Ba lop nay TACH RIENG (khong gop 1 diem) vi dung 1 diem se khong biet loi nam
o dau -- dung bai hoc "error-placement" (xem TRANG-THAI.md).

GROUND TRUTH KHONG DUA VAO 9B TU SINH (user 2026-09-03: "neu 9B self sinh sai
thi phai chinh ground truth chinh xac cho no de mapper ko hoc thuoc tu 9B"):
moi item sinh tu generator co san JSON {entities, used_in_question,
distractors, operation, final_answer} -- CHINH XAC 100% vi minh tu dat so.
`gold_template()` dung JSON nay de dung CAU COT THO (chac chan dung, van
phong cung) lam gold MAC DINH -- KHONG can 9B de co gold dung. 9B chi duoc
dung sau nay (buoc rieng, ngoai file nay) de PARAPHRASE cho tu nhien hon, va
moi ban paraphrase PHAI cham lai bang chinh score_eba() o day so voi JSON goc
truoc khi duoc chap nhan thay gold_template -- neu am diem o lop A/B thi loai,
giu nguyen ban template. Xem MANIFEST.md muc eba khi viet buoc paraphrase.

    python -u eba_gen.py --out /content/eba_items.json --n 400
"""

import argparse
import json
import random
import re

# ten thuc the: gioi han tap co dinh de de doc tay, tach han khoi ten dung
# trong suite_gen.py (PLACES/ITEMS/PEOPLE) de khong dam lan neu chung du lieu.
NAMES = ["Mara", "Tobin", "Priya", "Declan", "Yuki", "Soren", "Nadia", "Cole",
          "Imogen", "Rafi", "Sena", "Ottoline", "Kip", "Anouk", "Bram"]
ATTRS = [("towels", "towel"), ("marbles", "marble"), ("stickers", "sticker"),
         ("pencils", "pencil"), ("crates", "crate"), ("badges", "badge"),
         ("tokens", "token"), ("bricks", "brick")]

OPS = ("sum", "diff", "product")


def _entity_names(rng, k):
    return rng.sample(NAMES, k)


def gen_item(rng, difficulty):
    """difficulty 0..3: dieu chinh K (so thuc the), co/khong distractor, va
    operation don/chuoi. Escalate dan giong probe trich-so da lam truoc do."""
    k_used = 2 if difficulty < 3 else 3
    n_distractor = 0 if difficulty == 0 else rng.randint(1, 2)
    k_total = k_used + n_distractor
    names = _entity_names(rng, k_total)
    used, distractors = names[:k_used], names[k_used:]
    attr_pl, attr_sg = rng.choice(ATTRS)

    lo, hi = (2, 19) if difficulty == 0 else (2, 99)
    values = {}
    while len(values) < k_total:
        v = rng.randint(lo, hi)
        # gia tri PHAI phan biet doi mot -- trung nhau lam layer B khong the
        # phan biet "dung thuc thi" voi "trung so ngau nhien"
        if v not in values.values():
            values[names[len(values)]] = v

    op = "sum" if k_used > 2 else rng.choice(OPS)
    vs = [values[n] for n in used]
    if op == "sum":
        ans = sum(vs)
        expr = " + ".join(str(v) for v in vs)
    elif op == "diff":
        a, b = max(vs), min(vs)
        ans = a - b
        # giu dung thu tu THUC THE (khong phai gia tri) trong bieu thuc de
        # layer B kiem duoc dung entity nao dong vai "bi tru"
        big_name = used[0] if vs[0] == a else used[1]
        small_name = used[1] if big_name == used[0] else used[0]
        used = [big_name, small_name]
        expr = f"{a} - {b}"
    else:  # product
        ans = vs[0] * vs[1]
        expr = " x ".join(str(v) for v in vs)

    fact_sents = [f"{n} has {values[n]} {attr_pl}." for n in names]
    rng.shuffle(fact_sents)
    fact = " ".join(fact_sents)

    if op == "sum":
        who = (", ".join(used[:-1]) + f" and {used[-1]}") if len(used) > 2 \
              else f"{used[0]} and {used[1]}"
        q = f"How many {attr_pl} do {who} have in total?"
    elif op == "diff":
        q = f"How many more {attr_pl} does {used[0]} have than {used[1]}?"
    else:
        q = (f"If you multiply the number of {attr_pl} {used[0]} has by the "
             f"number {used[1]} has, what do you get?")

    prompt = (f"Solve step by step, then give the final numeric answer after "
              f"'Final Answer: '.\n\nProblem: {fact} {q}\n\n"
              f"<think>\n\n</think>\n\nSolution: ")
    meta = {"entities": values, "used_in_question": used,
            "distractors": distractors, "operation": op,
            "expr": expr, "attr_sg": attr_sg, "attr_pl": attr_pl,
            "final_answer": ans}
    return {"kind": "eba", "bench": "eba", "prompt": prompt,
            "gold": gold_template(meta), "expect": str(ans), "meta": meta}


def gold_template(meta):
    """CoT tho tu JSON -- LUON DUNG vi lay thang tu meta, khong qua model
    nao. Van phong cung nhung khong bao gio sai -- dung khi chua co (hoac
    chua duyet) ban paraphrase cua 9B."""
    used, vals = meta["used_in_question"], meta["entities"]
    lines = [f"{n} has {vals[n]} {meta['attr_pl']}." for n in used]
    lines.append(f"{meta['expr']} = {meta['final_answer']}.")
    lines.append(f"Final Answer: {meta['final_answer']}")
    return "\n".join(lines)


_NUM = re.compile(r"\d[\d,]*")
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _all_nums(text):
    """finditer tren _NUM luon khop CHUOI SO LON NHAT tai moi vi tri (khong
    the khop tien to lung chung nhu "100" trong "1000") -- an toan truoc bai
    hoc suite_gen.score (khop chuoi con tho lam garble tinh thanh dung)."""
    return [int(nm.group().replace(",", "")) for nm in _NUM.finditer(text)]


def _nums_near(text, name):
    """Tra ve danh sach so nguyen trong CUNG CAU voi ten -- ban dau dung
    +-N ky tu nhung mot cau ngan ke ben ("Tobin has 7 towels" dung sau "Mara
    has some towels") lot vao trong window ky tu, lam Mara "an gian" duoc so
    cua Tobin (bat boi test_entity_omitted). Cau la ranh gioi ngu nghia dung
    hon ky tu."""
    out = []
    for sent in _SENT_SPLIT.split(text):
        if name not in sent:
            continue
        out.extend(_all_nums(sent))
    return out


def score_eba(item, text):
    """3 lop TACH RIENG (khong gop 1 diem) -- xem docstring dau file.

    A: entity-value recall.  moi thuc thi trong used_in_question:
       +1 dung gia tri gan ten (khop chinh xac trong cua so)
        0 ten xuat hien nhung KHONG co so nao trong cua so (roi thong tin)
       -1 ten xuat hien kem SO SAI (bia so -- dung loi da doc tay bridge_oracle)
       tra ve TRUNG BINH tren so thuc thi trong used_in_question (0 neu ten
       khong xuat hien lan nao trong text).
    B: relational correctness.  +1 neu KHONG co gia tri cua distractor nao
       xuat hien gan MOT thuc thi dung vai tro (nham distractor); -1 neu co;
       0 neu khong co distractor de kiem (difficulty 0).
    C: final answer -- so sanh CHINH XAC voi meta['final_answer'], tach token
       so (khong dung substring tho -- bai hoc suite_gen.score/musr/probe-so).
    """
    m = item["meta"]
    used, vals = m["used_in_question"], m["entities"]

    a_scores = []
    for n in used:
        near = _nums_near(text, n)
        if not near:
            a_scores.append(0)
        elif vals[n] in near:
            a_scores.append(1)
        else:
            a_scores.append(-1)
    A = sum(a_scores) / len(a_scores) if a_scores else 0.0

    distractors = m["distractors"]
    if not distractors:
        B = 0.0
    else:
        leaked = 0
        for dn in distractors:
            dv = vals[dn]
            for n in used:
                if dv in _nums_near(text, n):
                    leaked += 1
                    break
        B = -1.0 if leaked else 1.0

    # KHONG dung re.sub(r"[^\d]"," ",...) roi split -- day thay dau phay
    # bang khoang trang, be "10,000" thanh "10"+"000" (bat boi
    # test_comma_formatted_number). _all_nums giu nguyen chuoi so co phay.
    C = int(m["final_answer"] in _all_nums(text))

    return {"A": round(A, 3), "B": B, "C": C,
            "combined": round(0.3 * A + 0.2 * B + 0.5 * C, 3)}


def build(n, out_path, seed=0):
    rng = random.Random(seed)
    items = []
    for j in range(n):
        difficulty = j % 4
        it = gen_item(random.Random(seed * 1_000_003 + j), difficulty)
        it["id"] = f"eba/{j}"
        it["difficulty"] = difficulty
        items.append(it)
    with open(out_path, "w") as fh:
        json.dump(items, fh, indent=1)
    print(f"eba_gen: {len(items)} item -> {out_path}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/eba_items.json")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
