"""test_gsm_struct -- bai kiem cho parser + 4 reward (rule 15: thang do MOI
phai doc tay >=8 mau kem diem TRUOC khi tin con so tong).

Ba lan da dinh truoc day: suite_gen.score khop chuoi con tren so garble; musr
bat chu A-F dau tien trung dau ra suy bien; probe trich-so dung scorer lay SO
CUOI trong khi dap an nam o SO DAU. Nen moi ca duoi day deu la mot loai loi
CU THE da tung gap hoac co the gap.

    python -m pytest test_gsm_struct.py -q
"""
import importlib.util
import pathlib

_H = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("gsm_struct", _H / "gsm_struct.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

GOLD_TXT = """<think>
Kylie co 20 vo so. Robert nhieu hon Kylie 5 cai. Hoi tong.
</think>
ENTITIES:
Kylie = 20
Robert = Kylie + 5
STEPS:
Robert = 20 + 5 = 25
total = 20 + 25 = 45
Final Answer: 45"""
GOLD = gs.parse(GOLD_TXT)


def test_1_parse_gold_dung_het():
    """Ca goc: phai tach dung thuc the / quan he / day buoc / dap so."""
    assert GOLD["ok"] and GOLD["has_think"]
    assert GOLD["entities"] == {("kylie", 20.0)}
    assert len(GOLD["relations"]) == 1
    assert GOLD["steps"] == [25.0, 45.0]
    assert GOLD["answer"] == 45.0


def test_2_dung_hoan_toan_duoc_diem_toi_da():
    tot, d = gs.score(GOLD_TXT, GOLD)
    assert (d["ent"], d["rel"], d["step"], d["ans"]) == (1.0, 1.0, 1.0, 1.0)
    assert abs(tot - 1.70) < 1e-9


def test_3_GAN_SAI_SO_CHO_THUC_THE_bi_phat_nang():
    """Loi da chan doan that o gsm8k ('Kylie dung 3 khan' -> sinh '6 khan'):
    ten dung, SO SAI -> phai AM, khong duoc chi la 0."""
    txt = GOLD_TXT.replace("Kylie = 20", "Kylie = 60")
    _, d = gs.score(txt, GOLD)
    assert d["ent"] == gs.PENALTY_WRONG_ENTITY == -8.0


def test_3b_sai_so_thuc_the_lam_TONG_diem_AM():
    """Phat phai DU NANG de thang ca ba thanh phan con lai: du rel/step/ans
    deu hoan hao, mau gan SAI so van phai bi diem AM (te hon khong tra loi)."""
    txt = GOLD_TXT.replace("Kylie = 20", "Kylie = 60")
    tot, _ = gs.score(txt, GOLD)
    assert tot < 0, f"tong {tot} khong am -> phat chua du nang"


def test_4_bo_sot_thuc_the_chi_tut_diem_khong_bi_phat():
    """Bo sot nhe hon bia sai -> chi F1 giam, khong am."""
    txt = GOLD_TXT.replace("Kylie = 20\n", "")
    _, d = gs.score(txt, GOLD)
    assert d["ent"] == 0.0          # khong con thuc the nao -> F1 = 0
    assert d["ent"] > gs.PENALTY_WRONG_ENTITY


def test_5_sai_quan_he_mat_diem_rel_nhung_ent_van_nguyen():
    txt = GOLD_TXT.replace("Robert = Kylie + 5", "Robert = Kylie + 10")
    _, d = gs.score(txt, GOLD)
    assert d["ent"] == 1.0 and d["rel"] == 0.0


def test_6_quan_he_giao_hoan_van_khop():
    """'Kylie + 5' va '5 + Kylie' phai duoc coi la MOT."""
    txt = GOLD_TXT.replace("Robert = Kylie + 5", "Robert = 5 + Kylie")
    _, d = gs.score(txt, GOLD)
    assert d["rel"] == 1.0


def test_7_truot_phep_tinh_cuoi_duoc_diem_TUNG_PHAN():
    """Day chinh la ly do bo nhi phan 0/1: gan dung phai hon han sai hoan toan."""
    txt = GOLD_TXT.replace("total = 20 + 25 = 45", "total = 20 + 25 = 44") \
                  .replace("Final Answer: 45", "Final Answer: 44")
    tot, d = gs.score(txt, GOLD)
    assert d["step"] == 0.5          # con [25], mat [45]
    assert d["ans"] == 0.2           # sai 2,2% -> bac <10%
    assert 0 < tot < 1.70


def test_8_dau_ra_RAC_bi_phat_khong_phai_chi_0():
    tot, d = gs.score("blah blah khong co gi ca", GOLD)
    assert d["ok"] is False and d["ans"] == -1.0 and tot < 0


def test_9_thieu_khoi_STEPS_coi_la_hong_format():
    txt = "<think>x</think>\nENTITIES:\nKylie = 20\nFinal Answer: 45"
    _, d = gs.score(txt, GOLD)
    assert d["ok"] is False and d["ans"] == -1.0


def test_10_dien_dat_khac_chu_van_full_diem_step():
    """Chong hoc vet: chu khac, nhung DI QUA dung cac dai luong 25 roi 45."""
    txt = """<think>khac han</think>
ENTITIES:
Kylie = 20
Robert = Kylie + 5
STEPS:
so vo so cua Robert = 20 + 5 = 25
tong cong hai ban = 20 + 25 = 45
Final Answer: 45"""
    _, d = gs.score(txt, GOLD)
    assert d["step"] == 1.0 and d["ans"] == 1.0


def test_11_sai_THU_TU_buoc_bi_tru_diem():
    """LCS: dung gia tri nhung nguoc thu tu -> khong duoc full diem."""
    txt = GOLD_TXT.replace("Robert = 20 + 5 = 25\ntotal = 20 + 25 = 45",
                           "total = 20 + 25 = 45\nRobert = 20 + 5 = 25")
    _, d = gs.score(txt, GOLD)
    assert d["step"] == 0.5


def test_12_so_co_dau_phay_va_don_vi_van_doc_dung():
    """'1,200' va '$1,200' phai bang 1200 -- loi dau phay tung lam sai thang do."""
    g = gs.parse("<think>x</think>\nENTITIES:\nA = 1,200\nSTEPS:\nt = 1,200 + 0 = $1,200\nFinal Answer: 1,200")
    assert g["entities"] == {("a", 1200.0)} and g["answer"] == 1200.0
    assert g["steps"] == [1200.0]


def test_13_thieu_think_van_parse_duoc_nhung_co_co_bao():
    txt = GOLD_TXT.split("</think>\n", 1)[1]
    p = gs.parse(txt)
    assert p["ok"] is True and p["has_think"] is False


def test_14_khong_lay_so_trong_khoi_think():
    """Khoi think co the chua so nhap; KHONG duoc tinh vao thuc the/buoc."""
    txt = GOLD_TXT.replace("Hoi tong.", "Thu 999 xem sao. 777 nua.")
    p = gs.parse(txt)
    assert p["entities"] == {("kylie", 20.0)}
    assert 999.0 not in p["steps"] and 777.0 not in p["steps"]
