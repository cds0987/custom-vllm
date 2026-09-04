"""test_tok_channel -- bai kiem cho KENH DANH TINH TOKEN (y tuong EAGLE) them
vao Mapper ngay 2026-09-04.

Dieu QUAN TRONG NHAT phai kiem: kenh moi khoi tao ZERO nen mapper co kenh phai
cho ra ket qua Y HET mapper khong kenh. Neu sai dieu nay thi moi so sanh
"mot-bien" sau do deu vo nghia (diem xuat phat da khac nhau).

Chay tren CPU, khong can GPU:
    python -m pytest test_tok_channel.py -q
"""
import importlib.util
import pathlib

import torch

_H = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")

# so chieu nho cho nhanh; ty le giong that (attn_dim = H*dh)
H, DH, T, B = 4, 8, 6, 1
ATTN_DIM = H * DH
D_MODEL = 16
NA, NG, HS, HT = 2, 2, 4, 4


def _mk(tok_rank):
    torch.manual_seed(0)
    return e5.Mapper(NA, NG, HS, HT, ATTN_DIM, 1e6, 1e6, device="cpu",
                     attn_rank=0, gdn_per_head=False, gdn_terms=1,
                     tok_rank=tok_rank, d_model=D_MODEL)


def _inputs():
    torch.manual_seed(1)
    k = torch.randn(B, H, T, DH)
    v = torch.randn(B, H, T, DH)
    emb = torch.randn(B, T, D_MODEL)
    return k, v, emb


def test_kenh_luc_khoi_tao_la_no_op():
    """ZERO-init => co kenh va khong kenh phai ra KET QUA GIONG HET."""
    k, v, emb = _inputs()
    m0, m1 = _mk(0), _mk(64)
    k0, v0 = m0.map_attn(0, k, v)              # khong kenh
    k1, v1 = m1.map_attn(0, k, v, emb)         # co kenh, TV=0
    assert torch.equal(k0, k1), "kenh zero-init da lam doi ket qua K!"
    assert torch.equal(v0, v1), "kenh zero-init da lam doi ket qua V!"


def test_bo_qua_emb_khi_khong_truyen():
    """Co kenh nhung khong truyen emb -> van chay, ket qua nhu khong kenh."""
    k, v, _ = _inputs()
    m0, m1 = _mk(0), _mk(64)
    assert torch.equal(m0.map_attn(0, k, v)[0], m1.map_attn(0, k, v)[0])


def test_kenh_co_tac_dung_sau_khi_TV_khac_zero():
    """Sau khi TV != 0 thi ket qua PHAI doi -- chung to kenh that su noi vao
    duong tinh, khong phai nhanh chet."""
    k, v, emb = _inputs()
    m = _mk(64)
    base = m.map_attn(0, k, v, emb)[0].clone()
    with torch.no_grad():
        m.TV_K.add_(0.1)
        m.TV_V.add_(0.1)
    after = m.map_attn(0, k, v, emb)[0]
    assert not torch.equal(base, after), "kenh khong anh huong dau ra -> hong"


def test_gradient_chay_ve_tham_so_kenh():
    """Backward phai toi duoc TU/TV, neu khong thi kenh khong hoc duoc gi."""
    k, v, emb = _inputs()
    m = _mk(64)
    with torch.no_grad():           # thoat khoi diem zero de gradient khac 0
        m.TV_K.add_(0.05)
        m.TV_V.add_(0.05)
    mk, mv = m.map_attn(0, k, v, emb)
    (mk.float().sum() + mv.float().sum()).backward()
    for name in ("TU_K", "TV_K", "TU_V", "TV_V"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"{name} khong nhan gradient"


def test_do_dai_lech_thi_bo_qua_kenh():
    """emb dai khac T -> phai BO QUA an toan, khong nem loi (bao ve luc eval
    dung cache do dai khac)."""
    k, v, _ = _inputs()
    m = _mk(64)
    emb_sai = torch.randn(B, T + 3, D_MODEL)
    out = m.map_attn(0, k, v, emb_sai)[0]
    assert torch.equal(out, m.map_attn(0, k, v)[0])


def test_luu_va_nap_giu_nguyen_kenh():
    """state_dict/load phai bao toan TU/TV; va checkpoint CU (khong co kenh)
    nap vao mapper co kenh phai giu zero-init -> chay giong ban cu."""
    import tempfile
    k, v, emb = _inputs()
    m = _mk(64)
    with torch.no_grad():
        m.TV_K.add_(0.3)
        m.TV_V.add_(0.2)
    want = m.map_attn(0, k, v, emb)[0].clone()
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/m.pt"
        torch.save(m.state_dict(), p)
        m2 = _mk(64)
        m2.load(p)
        assert torch.equal(m2.map_attn(0, k, v, emb)[0], want), \
            "nap lai khong khoi phuc dung kenh"

        # checkpoint CU: xoa cac khoa kenh di
        sd = m.state_dict()
        for key in ("TU_K", "TV_K", "TU_V", "TV_V"):
            sd.pop(key)
        sd["_meta"].pop("tok_rank", None)
        p2 = f"{d}/old.pt"
        torch.save(sd, p2)
        m3 = _mk(64)
        m3.load(p2)
        assert torch.equal(m3.map_attn(0, k, v, emb)[0],
                           m3.map_attn(0, k, v)[0]), \
            "nap checkpoint CU ma kenh khong con la no-op"


def test_meta_ghi_dung_cau_hinh():
    """_meta phai mang tok_rank/d_model de eval dung lai KHONG bi cat cut am
    tham -- dung bai hoc gdn_terms bi mac dinh ve 1 lam sai ca dot do."""
    m = _mk(64)
    meta = m.state_dict()["_meta"]
    assert meta["tok_rank"] == 64 and meta["d_model"] == D_MODEL
    assert _mk(0).state_dict()["_meta"]["tok_rank"] == 0
