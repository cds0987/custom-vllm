"""mcnemar -- so cap 2 checkpoint tren CUNG tap mau, tu cac file ket qua
per-item cua eval_big.

Vi sao phai McNemar chu khong so hai ty le: hai checkpoint cham tren CUNG bo
de -> mau ghep cap, khong doc lap. So 47% vs 52% bang kiem dinh ty le doc lap
la SAI mo hinh. McNemar chi nhin hai o LECH NHAU (b = A dung/B sai,
c = A sai/B dung) -- cac mau ca hai cung dung hoac cung sai khong mang thong
tin phan biet.

Da dinh 2 lan trong du an: ket luan "cai tien" tu chenh lech ty le tren n nho
(p=0,149 va p=0,114 o n=100 -- xu huong thang 3-4:1 nhung KHONG du bang chung).

    python mcnemar.py a.json b.json [--nhan-a ... --nhan-b ...]

File vao: dict {id: {"hit": 0/1, ...}} (dung dinh dang eval_big ghi ra).
"""
import argparse
import json
import pathlib


def doc(path):
    d = json.loads(pathlib.Path(path).read_text())
    if isinstance(d, dict) and "items" in d:
        d = d["items"]
    return {k: int(v["hit"] if isinstance(v, dict) else v) for k, v in d.items()}


def mcnemar(a, b):
    """Tra dict: n (mau chung), b_thang (a dung/b sai), c_thang (b dung/a sai),
    chi2 (co hieu chinh lien tuc Yates), p (xap xi 2 phia).
    Dung kiem dinh CHINH XAC (nhi thuc) khi b+c nho -- xap xi chi-binh-phuong
    khong dang tin o duoi ~25 ca lech, va chinh o day du an hay ket luan voi
    n=100."""
    chung = sorted(set(a) & set(b))
    n01 = sum(1 for k in chung if a[k] == 1 and b[k] == 0)
    n10 = sum(1 for k in chung if a[k] == 0 and b[k] == 1)
    m = n01 + n10
    out = {"n": len(chung), "a_dung": sum(a[k] for k in chung),
           "b_dung": sum(b[k] for k in chung),
           "a_thang": n01, "b_thang": n10, "n_lech": m}
    if m == 0:
        out.update(chi2=0.0, p=1.0, cach="khong co mau lech")
        return out
    chi2 = (abs(n01 - n10) - 1) ** 2 / m
    out["chi2"] = round(chi2, 3)
    if m < 25:
        # kiem dinh dau nhi thuc chinh xac, 2 phia (p=0,5)
        from math import comb
        k = min(n01, n10)
        p = sum(comb(m, i) for i in range(0, k + 1)) / 2 ** m * 2
        out["p"] = round(min(1.0, p), 4)
        out["cach"] = f"nhi thuc chinh xac (n_lech={m} < 25)"
    else:
        from math import erfc, sqrt
        out["p"] = round(erfc(sqrt(chi2 / 2)), 4)
        out["cach"] = f"chi-binh-phuong Yates (n_lech={m})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--nhan-a", default="A")
    ap.add_argument("--nhan-b", default="B")
    args = ap.parse_args()
    r = mcnemar(doc(args.a), doc(args.b))
    print(f"{args.nhan_a}: {r['a_dung']}/{r['n']} = "
          f"{100*r['a_dung']/max(r['n'],1):.1f}%")
    print(f"{args.nhan_b}: {r['b_dung']}/{r['n']} = "
          f"{100*r['b_dung']/max(r['n'],1):.1f}%")
    print(f"lech: {args.nhan_a} thang {r['a_thang']} / "
          f"{args.nhan_b} thang {r['b_thang']}")
    print(f"chi2={r['chi2']} p={r['p']} ({r['cach']})")
    print("KET LUAN: " + ("co y nghia thong ke (p<0,05)" if r["p"] < 0.05
                          else "CHUA du bang chung (p>=0,05) -- khong duoc "
                               "goi la cai tien"))


if __name__ == "__main__":
    main()
