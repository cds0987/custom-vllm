"""suite_gen — bo de DA HO cho chien dich cross-model o quy mo lon
(user chot 2026-08-26: "test luon vai ngan prompts tren du loai swe, rag-qa,
tim mid information, reasoning-math ... target 80-90 nhu normal decode").

4 ho de, moi ho do MOT NANG LUC KHAC NHAU cua cache ngoai:
  rag    — hieu + tra loi paraphrase tu van ban THAT (wikitext) + 1 fact cay
           vao. Dap an la DANH TU RIENG doc nhat (Thornfield/Marrowgate...)
           -> khong dinh nhieu dong nghia (bai hoc C2c: 'barn' vs 'stables'
           lam ban ket qua).
  mid    — needle ma 6 so o DO SAU 10/50/90% context: do do nhay VI TRI
           (gia thuyet: cache ngoai hong dan theo do sau).
  math   — bai toan so hoc nhieu buoc cay giua context: dap an KHONG nam
           san trong van ban, phai TINH -> tach "truy hoi" khoi "suy luan".
  swe    — module python tong hop nhieu ham, hoi ham nao co hanh vi X.
           Dap an = ten dinh danh -> do truy hoi tren VAN BAN CO CAU TRUC.

Moi item: {"family","ctx","prompt","expect":[...],"meta":{...}} — cham diem
"chua bat ky chuoi nao trong expect" (math: so sanh sau khi loc chu so).

Align T ≡ ~5 (mod BLOCK) bang binary search tren filler (dieu kien lmcache
hit da chot o C2b-4/C2b-N). Filler cua rag/mid/math/swe deu la van ban that
hoac ma nguon that -> khong dung filler rac (bai hoc lab-check: filler tong
hop lam bien MONG hon thuc te).
"""

import json
import random
import re

BLOCK = 1056

PLACES = ["Thornfield", "Marrowgate", "Ashcombe", "Quillhaven", "Draymoor",
          "Fenwick", "Larkspire", "Hollowmere", "Bramblewick", "Stonevale",
          "Wraymouth", "Cindermoor", "Galewick", "Netherby", "Ospreyhead"]
ITEMS = ["astrolabe", "manuscript", "sextant", "chalice", "tapestry",
         "orrery", "reliquary", "codex", "theodolite", "chronometer"]
PEOPLE = ["Doctor Vance", "Professor Ito", "Captain Reyes", "Curator Lam",
          "Archivist Pena", "Colonel Draye", "Sister Maud", "Warden Kib"]


def _corpus(cache={}):
    """wikitext-2 raw train, tach tu — filler VAN BAN THAT."""
    if "w" not in cache:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="train")
        cache["w"] = " ".join(t.strip() for t in ds["text"] if t.strip()).split()
    return cache["w"]


def _fit(mk, bank, nt, tstar):
    """binary search so tu filler sao cho T ~ tstar, roi tinh chinh bang ' a'."""
    lo, hi = 0, len(bank)
    while lo < hi:
        mid = (lo + hi) // 2
        if nt(mk(bank[:mid], [])) < tstar:
            lo = mid + 1
        else:
            hi = mid
    k = max(2, lo - 1)
    pad = []
    txt = mk(bank[:k], pad)
    T = nt(txt)
    for _ in range(16):
        if T >= tstar - 3:
            break
        pad.append("a")
        txt = mk(bank[:k], pad)
        T = nt(txt)
    return txt, T


# ---------------------------------------------------------------- ho de ---

def _rag(rng):
    who, item = rng.choice(PEOPLE), rng.choice(ITEMS)
    place = rng.choice(PLACES)
    fact = (f"For many years {who} kept the {item} locked inside the "
            f"{place} vault, and almost nobody outside the family knew it.")
    q = f"Where did {who} keep the {item}?"
    return fact, f"Question: {q}\nAnswer: It was kept inside the ", [place], {}


def _mid(rng, depth):
    code = "".join(rng.choice("0123456789") for _ in range(6))
    name = f"PRJ{rng.randint(100, 999)}"
    fact = f"IMPORTANT: The access code for project {name} is {code}."
    q = f"What is the access code for project {name}?"
    return (fact, f"Question: {q}\nAnswer: The access code is ", [code],
            {"depth": depth})


def _math(rng):
    crates = rng.randint(23, 89)
    per = rng.randint(7, 19)
    dmg = rng.randint(2, 9)
    extra = rng.randint(11, 47)
    ans = (crates - dmg) * per + extra
    fact = (f"Depot ledger: warehouse {rng.choice(PLACES)} received {crates} "
            f"crates; each crate holds {per} units. Inspectors condemned "
            f"{dmg} crates as damaged and removed them before dispatch. A "
            f"later truck added {extra} loose units to the same shipment.")
    q = ("Using the depot ledger above, how many units were dispatched in "
         "total?")
    return (fact, f"Question: {q}\nAnswer: The total number of units is ",
            [str(ans)], {"answer": ans})


_SWE_BODIES = [
    ("def {n}(rows):\n    if not rows:\n        raise ValueError('empty rows')\n"
     "    return sum(r['qty'] for r in rows)", "raises ValueError on empty input"),
    ("def {n}(path):\n    with open(path) as fh:\n        return "
     "[l.rstrip() for l in fh]", "reads a file into a list of lines"),
    ("def {n}(d, k, default=0):\n    try:\n        return int(d[k])\n"
     "    except (KeyError, TypeError):\n        return default",
     "returns a default when the key is missing"),
    ("def {n}(xs):\n    seen = set()\n    return [x for x in xs if not "
     "(x in seen or seen.add(x))]", "removes duplicates while keeping order"),
    ("def {n}(a, b):\n    if b == 0:\n        return float('inf')\n"
     "    return round(a / b, 4)", "guards against division by zero"),
]


def _swe(rng):
    verbs = ["collect", "flush", "resolve", "merge", "sanitize", "hydrate",
             "prune", "rebalance", "tally", "dispatch"]
    nouns = ["records", "buffers", "tokens", "shards", "batches", "ledgers"]
    picks = rng.sample(range(len(_SWE_BODIES)), 3)
    names = [f"{rng.choice(verbs)}_{rng.choice(nouns)}_{rng.randint(10,99)}"
             for _ in picks]
    mod = f"utils_{rng.randint(1000, 9999)}.py"
    src = "\n\n".join(_SWE_BODIES[p][0].format(n=n) for p, n in zip(picks, names))
    tgt = rng.randrange(3)
    fact = f"# ---- {mod} ----\n{src}\n# ---- end {mod} ----"
    q = (f"In module {mod} above, which function {_SWE_BODIES[picks[tgt]][1]}?")
    return (fact, f"Question: {q}\nAnswer: The function is ",
            [names[tgt]], {"module": mod, "fns": names})


# --------------------------------------------------------------- builder ---

def build_suite(n, ctxs, families, out_path, tok, block=BLOCK, seed=0):
    """Sinh n item chia deu cho families x ctxs, align T ~ m*block+5."""
    corpus = _corpus()
    items = []
    combos = [(f, c) for f in families for c in ctxs]
    for j in range(n):
        fam, ctx = combos[j % len(combos)]
        rng = random.Random(seed * 1_000_003 + j)
        depth = (0.1, 0.5, 0.9)[j % 3]
        if fam == "rag":
            fact, tail, expect, meta = _rag(rng)
        elif fam == "mid":
            fact, tail, expect, meta = _mid(rng, depth)
        elif fam == "math":
            fact, tail, expect, meta = _math(rng)
        elif fam == "swe":
            fact, tail, expect, meta = _swe(rng)
        else:
            raise ValueError(fam)
        d = depth if fam == "mid" else 0.5
        start = rng.randrange(max(1, len(corpus) - ctx - 800))
        bank = corpus[start:start + ctx + 400]

        def mk(ws, pad, _f=fact, _t=tail, _d=d):
            cut = int(len(ws) * _d)
            return (" ".join(ws[:cut]) + "\n" + _f + "\n" + " ".join(ws[cut:])
                    + (" " + " ".join(pad) if pad else "") + "\n" + _t)

        def nt(txt):
            return len(tok(txt)["input_ids"])
        m = max(1, round((ctx - 5) / block))
        txt, T = _fit(mk, bank, nt, m * block + 5)
        items.append({"family": fam, "ctx": ctx, "prompt": txt,
                      "expect": expect, "meta": meta, "T": T,
                      "rem": T % block})
        if j % 50 == 0:
            print(f"suite {j}/{n} {fam} ctx{ctx} T={T} rem={T % block}",
                  flush=True)
    # SAP XEP theo ctx tang dan: moi wave dong nhat do dai -> suc chua kho
    # L1 doan duoc (bai hoc C2b-N: L1 tran thi evict truoc khi doc = miss gia)
    items.sort(key=lambda it: (it["ctx"], it["family"]))
    with open(out_path, "w") as fh:
        json.dump(items, fh)
    # ke hoach wave: so item/wave theo do dai (kho L1 ~32GB)
    per = {2000: 96, 4000: 48, 8000: 24, 16000: 12}
    waves, i = [], 0
    while i < len(items):
        w = per.get(items[i]["ctx"], 24)
        j = i
        while j < len(items) and items[j]["ctx"] == items[i]["ctx"] and j - i < w:
            j += 1
        waves.append([i, j])
        i = j
    with open(out_path.replace(".json", "_waves.json"), "w") as fh:
        json.dump(waves, fh)
    print(f"suite: {len(items)} items -> {out_path} | {len(waves)} wave")
    return items


def score(item, text):
    """1 neu dap an xuat hien trong text sinh ra."""
    t = text.lower()
    if item["family"] == "math":
        digits = re.sub(r"[^\d]", " ", text).split()
        return int(any(e in digits for e in item["expect"]))
    return int(any(e.lower() in t for e in item["expect"]))
