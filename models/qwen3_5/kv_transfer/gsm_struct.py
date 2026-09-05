"""gsm_struct -- dinh dang co CAU TRUC cho gsm8k + 4 ham reward phan ra.

DINH DANG (user chot 2026-09-05). Phan MODEL SINH RA (prompt goc giu nguyen,
ket thuc o "Solution: "):

    <think>
    (phan tich: doc de, nhan dien thuc the va quan he)
    </think>
    ENTITIES:
    Kylie = 20
    Robert = Kylie + 5
    STEPS:
    Robert = 20 + 5 = 25
    total = 20 + 25 = 45
    Final Answer: 45

Khoi <think> nam TRONG phan sinh ra -> khong phai sua template luc finetune
(yeu cau user: hanh vi phai nam trong TRONG SO, khong nho prompt nhac).

PHAN LOAI DONG trong khoi ENTITIES (quyet dinh cach cham):
  - "ten = SO"        -> RANG BUOC THUC THE (su kien lay thang tu de) -> R_ent
  - "ten = bieu thuc" -> QUAN HE giua cac thuc the                    -> R_rel

VI SAO CHAM THEO NOI DUNG DA PARSE, KHONG THEO CHUOI: de mapper khong hoc vet
van phong cua 9B ma phai di qua DUNG cac dai luong 9B da di qua (user 2026-09-04:
"de mapper biet duoc cach 9b thuc su suy nghi thay vi bat hoc vet").

PHAT NANG SAI ENTITY (user chot 2026-09-05): gan SAI so cho mot thuc the la loi
chi mang -- dung loi da chan doan o gsm8k ("Kylie dung 3 khan" -> sinh "6 khan").
Nen R_ent khong chi ve 0 ma AM MANH (-2,0), theo dung kieu bat doi xung cua
`no_cheating` (-20) trong notebook GRPO cua Unsloth.
"""
import re

# so: cho phep dau phay ngan nghin, dau am, thap phan, $ va % di kem
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
_THINK = re.compile(r"<think>.*?</think>", re.S)
# PHAT NANG khi gan SAI so cho thuc the (user chot 2026-09-05). Chon -8,0 chu
# khong phai -2,0: voi trong so mac dinh w_ent=0,25 thi -2,0 chi thanh -0,5,
# bi ba thanh phan con lai (toi da +1,45) bu lai -> mau gan sai so VAN duoc
# diem DUONG (+0,95). Voi -8,0: 0,25 x (-8) = -2,0 -> tong = -0,55, AM han,
# tuc la te hon ca khong tra loi gi. Dung tinh than bat doi xung cua
# `no_cheating` (-20) trong notebook GRPO Unsloth.
PENALTY_WRONG_ENTITY = -8.0


def _num(s):
    """Chuan hoa mot chuoi so -> float. None neu khong phai so."""
    if s is None:
        return None
    t = str(s).strip().strip(".").replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(t)
    except ValueError:
        return None


def _name(s):
    """Chuan hoa ten thuc the: thuong hoa, bo dau cau/khoang trang thua."""
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _last_rhs(line):
    """'total = 20 + 25 = 45' -> '45' (ve phai CUOI CUNG)."""
    parts = line.split("=")
    return parts[-1].strip() if len(parts) >= 2 else None


def _operands(expr):
    """Tach toan hang + toan tu cua bieu thuc mot phep tinh.
    Tra (op, [toan hang da chuan hoa]) hoac (None, []) neu khong nhan dang duoc."""
    for op in ("+", "-", "*", "x", "/"):
        # tach tren toan tu DAU TIEN tim thay; ' x ' phai co khoang trang de
        # khong cat nham ten co chu x (vd 'box')
        pat = f" {op} "
        if pat in expr:
            a, b = expr.split(pat, 1)
            o = "*" if op == "x" else op
            return o, [a.strip(), b.strip()]
    return None, []


def parse(text):
    """Phan tich dau ra thanh cau truc may doc duoc.

    Tra dict: ok / entities / relations / steps / answer.
      entities  : set (ten, gia_tri)          -- dong 'ten = SO'
      relations : set (ten, op, khoa_toan_hang) -- dong 'ten = bieu thuc'
      steps     : list gia tri KET QUA theo DUNG thu tu cac dong STEPS
      answer    : float dap so cuoi, None neu thieu
    ok = True chi khi CO DU ca ENTITIES, STEPS va Final Answer.
    """
    out = {"ok": False, "entities": set(), "relations": set(),
           "steps": [], "answer": None, "has_think": False}
    if not text:
        return out
    out["has_think"] = bool(_THINK.search(text))
    body = _THINK.sub(" ", text)          # bo khoi think khoi phan cham diem

    up = body
    i_ent = up.find("ENTITIES:")
    i_stp = up.find("STEPS:")
    m_ans = list(re.finditer(r"Final Answer:\s*([^\n]+)", body))
    if i_ent < 0 or i_stp < 0 or i_stp < i_ent or not m_ans:
        return out
    out["answer"] = _num(_NUM.search(m_ans[-1].group(1)).group(0)) \
        if _NUM.search(m_ans[-1].group(1)) else None

    # ---- khoi ENTITIES -> thuc the + quan he -------------------------------
    for line in body[i_ent + len("ENTITIES:"):i_stp].splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        nm, rhs = _name(lhs), rhs.strip()
        if not nm:
            continue
        v = _num(rhs)
        if v is not None:                      # 'ten = SO' -> thuc the
            out["entities"].add((nm, v))
            continue
        op, ops = _operands(rhs)               # 'ten = bieu thuc' -> quan he
        if op and len(ops) == 2:
            keys = []
            for o in ops:
                ov = _num(o)
                keys.append(ov if ov is not None else _name(o))
            # cong/nhan giao hoan -> dung frozenset; tru/chia thi giu THU TU
            key = frozenset(map(str, keys)) if op in "+*" else tuple(map(str, keys))
            out["relations"].add((nm, op, key))

    # ---- khoi STEPS -> day gia tri trung gian THEO THU TU -------------------
    seg = body[i_stp + len("STEPS:"):]
    stop = seg.find("Final Answer:")
    for line in (seg if stop < 0 else seg[:stop]).splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or "=" not in line:
            continue
        v = _num(_last_rhs(line))
        if v is None:
            m = _NUM.findall(line)
            v = _num(m[-1]) if m else None
        if v is not None:
            out["steps"].append(v)

    out["ok"] = bool(out["entities"] or out["relations"]) and \
        bool(out["steps"]) and out["answer"] is not None
    return out


def _f1(pred, gold):
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def r_ent(pred, gold):
    """F1 tren tap (ten, gia tri). PHAT NANG (-2,0) neu co bat ky thuc the nao
    duoc gan SAI so -- khong phai 'khong duoc diem' ma la 'bi tru' (user chot).
    Thieu mot thuc the (khong nhac toi) chi lam tut F1, khong bi phat nang:
    bia SAI nguy hiem hon la bo sot."""
    gmap = dict(gold["entities"])
    wrong = sum(1 for nm, v in pred["entities"] if nm in gmap and gmap[nm] != v)
    if wrong:
        return PENALTY_WRONG_ENTITY
    return _f1(pred["entities"], gold["entities"])


def r_rel(pred, gold):
    return _f1(pred["relations"], gold["relations"])


def _lcs(a, b):
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x == y else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def r_step(pred, gold):
    """Ty le gia tri trung gian cua 9B xuat hien trong dau ra mapped DUNG THU
    TU (LCS). Khong so chuoi chu -> dien dat khac van duoc diem, mien la di qua
    dung cac dai luong."""
    if not gold["steps"]:
        return 0.0
    return _lcs(pred["steps"], gold["steps"]) / len(gold["steps"])


def r_ans(pred, gold_answer):
    """Chia BAC theo sai so tuong doi (kieu correctness_check cua Unsloth) thay
    vi nhi phan 0/1: co gradient giua 'truot phep cong cuoi' va 'sai hoan toan'.
    Hong format (khong parse duoc) bi PHAT -1,0, khong phai chi 0."""
    if not pred["ok"] or pred["answer"] is None:
        return -1.0
    if gold_answer is None:
        return 0.0
    if pred["answer"] == gold_answer:
        return 1.0
    denom = abs(gold_answer) if gold_answer else 1.0
    err = abs(pred["answer"] - gold_answer) / denom
    if err < 0.01:
        return 0.5
    if err < 0.10:
        return 0.2
    return 0.0


def score(text, gold_struct, gold_answer=None, w=None):
    """Cham mot dau ra. gold_struct = parse() cua quy dao 9B tu sinh (da loc
    dung). Tra (tong_diem, chi_tiet)."""
    w = w or {"ent": 0.25, "rel": 0.25, "step": 0.2, "ans": 1.0}
    p = parse(text)
    if gold_answer is None:
        gold_answer = gold_struct.get("answer")
    d = {"ent": r_ent(p, gold_struct), "rel": r_rel(p, gold_struct),
         "step": r_step(p, gold_struct), "ans": r_ans(p, gold_answer),
         "ok": p["ok"], "has_think": p["has_think"]}
    total = sum(w[k] * d[k] for k in ("ent", "rel", "step", "ans"))
    return total, d
