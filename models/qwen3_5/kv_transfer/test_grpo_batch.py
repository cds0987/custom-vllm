"""test_grpo_batch -- bai kiem cho duong GOP LO B mau/buoc (2026-09-05).

Hai cho de sai AM THAM khi gop lo, khong bao loi ma chi ra so xau:
  1. chia lo: neu lo lan sang mau KHAC DO DAI prompt -> phai dem -> pha
     attention/GDN. Bai kiem bat buoc moi lo dong nhat do dai.
  2. advantage: reward cua CAC MAU KHAC NHAU khong duoc tron chung khi chuan
     hoa Z-score (de khac nhau, thang diem khac nhau). Phai tinh RIENG trong
     nhom K cua tung mau, theo dung anh xa hang r <-> mau r % B, nhanh r // B.

    python -m pytest test_grpo_batch.py -q
"""
import importlib.util
import pathlib

_H = pathlib.Path(__file__).parent


def _load_isolated():
    """Nap eba_grpo KHONG chay main() va khong keo model -- chi lay ham thuan."""
    spec = importlib.util.spec_from_file_location("eba_grpo", _H / "eba_grpo.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


eg = _load_isolated()


class FakeTok:
    """Tokenizer gia: so token = so tu. Du de kiem logic chia lo."""

    def __call__(self, text, truncation=False, max_length=None):
        return {"input_ids": text.split()}


TOK = FakeTok()


def _items(specs):
    """specs: list do dai (so tu) -> item co prompt dung do dai do."""
    return [{"id": f"i{n}", "prompt": " ".join(["w"] * L)}
            for n, L in enumerate(specs)]


def test_1_moi_lo_dong_nhat_do_dai():
    """Rang buoc CUNG: khong bao gio duoc tron hai do dai vao mot lo."""
    its = _items([5, 5, 5, 7, 7, 9])
    for bk in eg.make_buckets(its, 4, TOK):
        assert len({len(x["prompt"].split()) for x in bk}) == 1


def test_2_khong_bo_sot_va_khong_lap_mau_nao():
    """Lo le phai duoc chay o lo nho hon, khong bi vut."""
    its = _items([5, 5, 5, 7, 7, 9])
    got = [x["id"] for bk in eg.make_buckets(its, 4, TOK) for x in bk]
    assert sorted(got) == sorted(x["id"] for x in its)


def test_3_kich_thuoc_lo_khong_vuot_bsz():
    its = _items([5] * 10 + [7] * 3)
    assert all(len(bk) <= 4 for bk in eg.make_buckets(its, 4, TOK))


def test_4_lo_le_van_duoc_giu():
    """9 mau cung do dai, B=4 -> 4+4+1, KHONG phai 4+4 (bo 1)."""
    bks = eg.make_buckets(_items([5] * 9), 4, TOK)
    assert sorted(len(b) for b in bks) == [1, 4, 4]


def test_5_bsz_1_giu_nguyen_hanh_vi_cu():
    its = _items([5, 7, 9])
    assert all(len(b) == 1 for b in eg.make_buckets(its, 1, TOK))


def test_6_advantage_khong_tron_giua_cac_mau():
    """Cot loi. B=2, K=3, hang r <-> mau r%B nhanh r//B.
    Mau 0 co reward [0,0,0] (dong nhat -> adv 0); mau 1 co [1,2,3].
    Neu LO TRON ca 6 reward vao mot lan chuan hoa thi mau 0 se nhan adv KHAC 0
    -- day dung la loi bai kiem nay chan."""
    B, K = 2, 3
    rewards = [0.0, 1.0,   0.0, 2.0,   0.0, 3.0]   # xen ke theo r%B
    adv = [0.0] * len(rewards)
    n_active = 0
    for i in range(B):
        g = eg.group_advantage([rewards[i + j * B] for j in range(K)])
        if g is None:
            continue
        n_active += 1
        for j in range(K):
            adv[i + j * B] = g[j]
    assert n_active == 1                       # chi mau 1 co tin hieu
    assert [adv[i * B] for i in range(K)] == [0.0, 0.0, 0.0]   # mau 0 im lang
    m1 = [adv[1 + j * B] for j in range(K)]
    assert m1[0] < m1[1] < m1[2]               # mau 1 giu dung thu tu
    assert abs(sum(m1)) < 1e-5                 # Z-score: trung binh 0


def test_7_anh_xa_hang_ve_mau_khop_voi_repeat():
    """clone_cache_repeat dung .repeat(K,...) -> LAP CA KHOI B. Kiem tra anh
    xa r -> r%B trung khop voi cach torch lap khoi."""
    import torch
    B, K = 3, 2
    x = torch.arange(B).reshape(B, 1)
    rep = x.repeat(K, 1)[:, 0].tolist()
    assert rep == [r % B for r in range(B * K)]
