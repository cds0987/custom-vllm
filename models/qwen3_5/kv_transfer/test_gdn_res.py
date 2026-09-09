"""test_gdn_res -- duong residual cho nua GDN: S' = f(S) + gamma*S.

Y user (2026-09-09): "dang y = x + f(x) tu cac component de thong tin truyen
qua nhau". Nua attention DA co dang do (nhanh attn_rank); nua GDN thi KHONG:
S bi chia cho rms roi tron head bang alpha, ma alpha khoi tao UNIFORM 1/Hs ->
moi head dich = trung binh CA Hs head nguon.

Do tren checkpoint sft_struct_v4 (2026-09-09): sau ca chien dich, duong cheo
alpha chi gap 3,4-4,7 lan ngoai duong cheo va chiem ~13% khoi luong moi hang
-> ~87% noi dung moi head dich VAN la hon hop.

DIEU KIEN SONG CON cua phep so mot-bien: gamma khoi tao 0 -> mapper co duong
residual phai chay GIONG HET ban khong co no, va nap checkpoint cu phai ra
dung ket qua cu. Neu khong thi moi chenh lech sau nay khong quy duoc cho ai.

    python -m pytest test_gdn_res.py -q
"""
import importlib.util
import pathlib

import torch

_H = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("e5_train", _H / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

KW = dict(n_attn_tgt=2, n_gdn_tgt=3, gdn_heads_s=4, gdn_heads_t=4,
          attn_dim=16, theta_s=1e4, theta_t=1e4, device="cpu")


def _S(b=2, h=4):
    """A,B trong Mapper ghim cung 128x128 -> trang thai GDN PHAI la (B,H,128,128)."""
    torch.manual_seed(0)
    return torch.randn(b, h, 128, 128)


def test_1_gamma_khoi_tao_bang_0():
    m = e5.Mapper(**KW, gdn_res=True)
    assert m.gdn_res is not None and len(m.gdn_res) == 3
    for g in m.gdn_res:
        assert torch.count_nonzero(g) == 0, "gamma phai khoi tao 0 (no-op)"
        assert g.shape == (4, 1, 1), f"phai theo tung head, duoc {tuple(g.shape)}"


def test_2_luc_khoi_tao_ra_KET_QUA_Y_HET_ban_khong_co_residual():
    """Dieu kien song con: bat co ma khong doi gi -> so sanh mot-bien sach."""
    a = e5.Mapper(**KW, gdn_res=False)
    b = e5.Mapper(**KW, gdn_res=True)
    S = _S()
    for j in range(3):
        assert torch.equal(a.map_gdn(j, S), b.map_gdn(j, S)), f"lech o lop {j}"


def test_3_gamma_khac_0_thi_cong_dung_S_GOC():
    """Cong S CHUA chuan hoa (giu thang do), khong phai S da chia rms."""
    m = e5.Mapper(**KW, gdn_res=True)
    S = _S()
    goc = m.map_gdn(0, S).float()
    with torch.no_grad():
        m.gdn_res[0].fill_(0.5)
    moi = m.map_gdn(0, S).float()
    cho_doi = (goc + 0.5 * S).to(torch.bfloat16).float()
    assert torch.allclose(moi, cho_doi, atol=2e-2), (moi - cho_doi).abs().max()


def test_4_gamma_theo_TUNG_HEAD_khong_phai_vo_huong():
    m = e5.Mapper(**KW, gdn_res=True)
    S = _S()
    goc = m.map_gdn(0, S).float()
    with torch.no_grad():
        m.gdn_res[0][0].fill_(1.0)      # chi head 0
    moi = m.map_gdn(0, S).float()
    assert not torch.allclose(moi[:, 0], goc[:, 0]), "head 0 phai doi"
    assert torch.allclose(moi[:, 1:], goc[:, 1:], atol=1e-3), "head khac phai y nguyen"


def test_5_gamma_co_gradient_va_nam_trong_params():
    m = e5.Mapper(**KW, gdn_res=True)
    assert all(any(g is p for p in m.params) for g in m.gdn_res)
    S = _S().requires_grad_(False)
    m.map_gdn(0, S).float().sum().backward()
    assert m.gdn_res[0].grad is not None
    assert torch.count_nonzero(m.gdn_res[0].grad) > 0, "gamma phai nhan gradient"


def test_6_nap_checkpoint_CU_thi_giu_no_op(tmp_path=None):
    """Checkpoint cu khong co khoa 'gdn_res' -> gamma o nguyen 0 -> ket qua
    y het ban cu. Day la dieu kien de warm-start tu sft_struct_v4."""
    cu = e5.Mapper(**KW, gdn_res=False)
    f = _H / "_tmp_test_gdn_res.pt"
    try:
        torch.save(cu.state_dict(), f)
        moi = e5.Mapper(**KW, gdn_res=True)
        moi.load(str(f))
        assert "gdn_res" not in torch.load(f, map_location="cpu")
        for g in moi.gdn_res:
            assert torch.count_nonzero(g) == 0
        S = _S()
        for j in range(3):
            assert torch.equal(cu.map_gdn(j, S), moi.map_gdn(j, S))
    finally:
        f.unlink(missing_ok=True)


def test_7_luu_roi_nap_lai_giu_dung_gamma():
    m = e5.Mapper(**KW, gdn_res=True)
    with torch.no_grad():
        for j, g in enumerate(m.gdn_res):
            g.copy_(torch.full_like(g, 0.1 * (j + 1)))
    f = _H / "_tmp_test_gdn_res2.pt"
    try:
        torch.save(m.state_dict(), f)
        m2 = e5.Mapper(**KW, gdn_res=True)
        m2.load(str(f))
        for g1, g2 in zip(m.gdn_res, m2.gdn_res):
            assert torch.allclose(g1, g2)
        assert torch.load(f, map_location="cpu")["_meta"]["gdn_res"] is True
    finally:
        f.unlink(missing_ok=True)


def test_8_so_head_lech_thi_TAT_chu_khong_no():
    """4B->27B co the khac so head GDN; khong cong thang S duoc -> phai tat
    an toan, khong duoc nem loi giua luc train."""
    kw = dict(KW); kw["gdn_heads_s"], kw["gdn_heads_t"] = 4, 6
    m = e5.Mapper(**kw, gdn_res=True)
    assert m.gdn_res is None
    m.map_gdn(0, _S(h=4))     # van chay duoc


def test_9_MOI_script_dung_Mapper_deu_doc_gdn_res_tu_meta():
    """CHONG TAI PHAT bug joint49cc: eval_big dung Mapper nhung KHONG doc
    `gdn_terms` tu `_meta` -> am tham ve mac dinh -> checkpoint bi CAT CUT khi
    cham diem, khong loi khong canh bao, ra so SAI da bao cao ra ngoai.
    Moi co cau hinh moi (gdn_res) deu phai duoc doc o TAT CA cho dung Mapper."""
    import re
    thieu = []
    for f in ("eval_big.py", "eba_grpo.py", "oracle_ablation.py",
              "sft_struct.py"):
        s = (_H / f).read_text(encoding="utf-8")
        if "e5.Mapper(" not in s:
            continue
        for khoa in ("attn_rank", "gdn_per_head", "gdn_terms", "gdn_res"):
            if f'_meta.get("{khoa}"' not in s:
                thieu.append(f"{f}: khong doc {khoa} tu _meta")
    assert not thieu, "\n".join(thieu)
