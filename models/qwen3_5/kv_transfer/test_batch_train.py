"""Kiem phan gom lo cho train — KHONG CAN GPU.

Hai cho de sai AM THAM nhat:

  1. meta_for_len phai doi chieu DAU (batch) o MOI tensor, ke ca GDN. Lech
     chieu batch = cache vo nghia ma khong bao loi nao.
  2. CE co mat na: gold trong lo co do dai khac nhau nen phai dem bang -100,
     roi ZERO trong so o cho dem. Quen buoc zero thi CE tinh ca token dem va
     moi con so deu sai — khong exception, khong canh bao.

Chay: python -u test_batch_train.py
"""

import copy
import importlib.util
import sys
from pathlib import Path

import torch

_H = Path(__file__).parent


def load_meta_fn():
    """Lay meta_for_len tu e9_joint ma KHONG chay main (no can GPU)."""
    src = (_H / "e9_joint.py").read_text(encoding="utf-8")
    i = src.index("def meta_for_len(")
    j = src.index("def verify_meta(")
    ns = {"copy": copy}
    exec(src[i:j], ns)
    return ns["meta_for_len"]


def main():
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"FAIL {name} {extra}")

    # ---------- 1. meta_for_len doi CA chieu T lan chieu BATCH ----------
    meta_for_len = load_meta_fn()
    T_BASE = 512
    base = {"cache_ints": {"seen": T_BASE, "khac": 7},
            "layers": [
                {"kind": "a", "ints": {"n": T_BASE},
                 "k": ((1, 8, T_BASE, 64), "torch.bfloat16"),
                 "v": ((1, 8, T_BASE, 64), "torch.bfloat16")},
                {"kind": "g", "ints": {"m": 3},
                 "rec": ((1, 32, 128, 128), "torch.bfloat16"),
                 "conv": ((1, 32, 4), "torch.bfloat16")}]}

    m = meta_for_len(base, T_BASE, 1024, 2)
    chk("cache_ints doi theo T", m["cache_ints"]["seen"] == 1024)
    chk("truong int khac giu nguyen", m["cache_ints"]["khac"] == 7)
    chk("attn k: T doi, batch doi",
        m["layers"][0]["k"][0] == (2, 8, 1024, 64), m["layers"][0]["k"][0])
    chk("attn v: T doi, batch doi",
        m["layers"][0]["v"][0] == (2, 8, 1024, 64), m["layers"][0]["v"][0])
    chk("GDN rec: batch doi, KHONG doi theo T",
        m["layers"][1]["rec"][0] == (2, 32, 128, 128), m["layers"][1]["rec"][0])
    chk("GDN conv: batch doi",
        m["layers"][1]["conv"][0] == (2, 32, 4), m["layers"][1]["conv"][0])

    m1 = meta_for_len(base, T_BASE, T_BASE, 1)
    chk("B=1, T=T_BASE tra ve DUNG meta goc", m1 == base,
        "khong idempotent -> ban B=1 co the da bi doi hanh vi")

    # ---------- 2. CE co mat na ----------
    # mo phong: lo 2 mau, gold dai 3 va 1 -> dem -100
    gold = torch.tensor([[5, 6, 7], [9, -100, -100]])
    valid = gold >= 0
    g_safe = gold.clamp(min=0)
    logp = torch.log_softmax(torch.randn(2, 3, 20), -1)
    nll = -logp.gather(2, g_safe.unsqueeze(-1)).squeeze(-1)
    wts = valid.float()
    ce_mask = (nll * wts).sum() / wts.sum()

    # doi chieu: tinh rieng tung mau roi gop, phai RA CUNG mot so
    n0 = -logp[0, :3].gather(1, gold[0, :3, None]).squeeze(1)
    n1 = -logp[1, :1].gather(1, gold[1, :1, None]).squeeze(1)
    ce_ref = torch.cat([n0, n1]).sum() / 4
    chk("CE co mat na == tinh rieng tung mau",
        torch.allclose(ce_mask, ce_ref, atol=1e-6),
        f"{ce_mask.item():.6f} vs {ce_ref.item():.6f}")

    # neu QUEN zero trong so o cho dem thi phai RA KHAC (bai kiem nay bao ve
    # dung cai bug do — neu no khong khac thi bai kiem vo dung)
    ce_bug = nll.mean()
    chk("quen mat na THI RA KHAC (bai kiem co hieu luc)",
        not torch.allclose(ce_bug, ce_ref, atol=1e-6))

    # ---------- 3. FIRST_W chi ap cho token dau HOP LE ----------
    FIRST_W = 3.0
    w = valid.float()
    w[:, 0] = FIRST_W * valid[:, 0].float()
    chk("FIRST_W ap cho ca hai hang (token dau deu hop le)",
        bool((w[:, 0] == FIRST_W).all()))
    gold2 = torch.tensor([[-100, 6], [9, 10]])
    v2 = gold2 >= 0
    w2 = v2.float()
    w2[:, 0] = FIRST_W * v2[:, 0].float()
    chk("hang co token dau la DEM thi trong so = 0", float(w2[0, 0]) == 0.0)

    print(f"\n{ok} dat / {fail} hong")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
