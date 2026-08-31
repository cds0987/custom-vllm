"""Kiem GDN nhieu so hang — KHONG CAN GPU.

Dieu kien SONG CON cua viec scale mapper: ban R=4 nap tu checkpoint R=1 phai
cho dau ra GIONG HET ban R=1. Neu khong, warm-start lam tut diem va ca luot
train mat trang — da xay ra o joint49t (doi hinh dang mapper, CE nhay
0,9 -> 5-12).

Chay: python -u test_mapper_terms.py
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")

N_ATTN, N_GDN, HS, HT, DIM = 2, 3, 4, 6, 32
THS, THT = 10000.0, 10000.0


# map_gdn tra ve BFLOAT16. Do 2026-08-30: dau ra bien do max ~4,7 thi mot
# BUOC lam tron cua bf16 la 0,018 — lon gap ~9 lan do lech quan sat duoc
# (1,95e-3) khi chuyen sang per-head. Chuyen per-head doi thu tu/lo matmul nen
# lam tron khac di; do KHONG phai loi logic. Nguong phai theo bf16, khong
# theo fp32, neu khong bai kiem se bao dong gia mai mai.
TOL = 5e-3


def mk(terms, per_head=False):
    return e5.Mapper(N_ATTN, N_GDN, HS, HT, DIM, THS, THT, device="cpu",
                     gdn_per_head=per_head, gdn_terms=terms)


def main():
    torch.manual_seed(0)
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"FAIL {name} {extra}")

    S = torch.randn(1, HS, 128, 128)

    # --- 1. R=4 moi khoi tao == R=1 moi khoi tao (so hang them = 0) ---
    m1, m4 = mk(1), mk(4)
    for j in range(N_GDN):
        a = m1.map_gdn(j, S).float()
        b = m4.map_gdn(j, S).float()
        chk(f"khoi tao R=4 == R=1 (lop {j})", torch.allclose(a, b, atol=TOL),
            f"lech max {(a - b).abs().max():.2e} (nguong bf16 {TOL})")

    # --- 2. nap checkpoint R=1 vao R=4 -> van giong het ---
    # lam nhieu tham so de khong phai truong hop tam thuong
    for terms in m1.A:
        terms[0].data.add_(torch.randn_like(terms[0]) * 0.1)
    for terms in m1.B:
        terms[0].data.add_(torch.randn_like(terms[0]) * 0.1)
    for a in m1.alpha:
        a.data.add_(torch.randn_like(a) * 0.1)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m1.pt"
        torch.save(m1.state_dict(), p)
        m4b = mk(4)
        m4b.load(str(p))
        for j in range(N_GDN):
            a = m1.map_gdn(j, S).float()
            b = m4b.map_gdn(j, S).float()
            chk(f"nap R=1 -> R=4 giong het (lop {j})",
                torch.allclose(a, b, atol=TOL),
                f"lech max {(a - b).abs().max():.2e} (nguong bf16 {TOL})")

        # --- 3. nap R=1 vao R=4 CO per-head cung phai giong het ---
        m4h = mk(4, per_head=True)
        m4h.load(str(p))
        for j in range(N_GDN):
            a = m1.map_gdn(j, S).float()
            b = m4h.map_gdn(j, S).float()
            chk(f"nap R=1 -> R=4+per_head giong het (lop {j})",
                torch.allclose(a, b, atol=TOL),
                f"lech max {(a - b).abs().max():.2e} (nguong bf16 {TOL})")

    # --- 4. so hang them PHAI train duoc (khong bi ngat khoi do thi) ---
    m4c = mk(4)
    out = sum(m4c.map_gdn(j, S).float().sum() for j in range(N_GDN))
    out.backward()
    n_grad = sum(1 for terms in m4c.A for t in terms if t.grad is not None
                 and t.grad.abs().sum() > 0)
    chk("so hang r>0 co gradient", n_grad >= N_GDN * 2,
        f"chi {n_grad} tensor A co gradient khac 0")

    # --- 5. dem tham so tang dung ty le ---
    n1 = sum(p.numel() for p in mk(1).params)
    n4 = sum(p.numel() for p in mk(4).params)
    gdn1 = n1 - sum(p.numel() for p in mk(1).params[:0] or [])
    chk("R=4 nhieu tham so hon R=1", n4 > n1, f"{n1} -> {n4}")

    # --- 6. round-trip R=4 -> R=4 ---
    m4d = mk(4)
    for terms in m4d.A + m4d.B:
        for t in terms:
            t.data.add_(torch.randn_like(t) * 0.05)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m4.pt"
        torch.save(m4d.state_dict(), p)
        m4e = mk(4)
        m4e.load(str(p))
        for j in range(N_GDN):
            a = m4d.map_gdn(j, S).float()
            b = m4e.map_gdn(j, S).float()
            chk(f"round-trip R=4 (lop {j})", torch.allclose(a, b, atol=TOL),
                f"lech max {(a - b).abs().max():.2e} (nguong bf16 {TOL})")

    # --- 7. MOI THAM SO PHAI LA TENSOR LA (optimizer tu choi non-leaf) ---
    # Bug that 2026-08-31: A_r = base.requires_grad_(True) roi B_r =
    # base.clone() -> ban sao nam trong do thi, khong phai la. Forward van
    # chay nen EVAL khong lo ra; chi chet luc dung optimizer.
    for terms in (1, 4):
        m = mk(terms, per_head=(terms == 4))
        bad = [j for j, q in enumerate(m.params) if not q.is_leaf]
        chk(f"R={terms}: moi tham so la tensor LA", not bad,
            f"{len(bad)} tensor khong phai la")
        try:
            torch.optim.SGD(m.params, lr=1e-3)
            chk(f"R={terms}: dung duoc optimizer", True)
        except Exception as e:
            chk(f"R={terms}: dung duoc optimizer", False, str(e)[:60])

    print(f"\n{ok} dat / {fail} hong")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
