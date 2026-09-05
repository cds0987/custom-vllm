"""test_score_how -- giu score_how DONG BO voi score_text, va tai lap dung
duong duong-tinh-gia da phat hien khi doc tay (quy tac 15, 2026-09-05).

Rang buoc cung:  (score_how != "")  <=>  (score_text == 1)
Neu ai sua score_text ma quen score_how, test nay do.

    python -m pytest test_score_how.py -q
"""
import importlib.util
import pathlib

_H = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("ext_bench", _H / "ext_bench.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

IT = {"bench": "gsm8k", "expect": "90"}

DAY_DU = """<think>
Total amount: 30 + 60 = 90 liters.
</think>
ENTITIES:
Initial = 120
STEPS:
Remaining = 120 - 90 = 30
Collected = 2 * 30 = 60
Total = 30 + 60 = 90
Final Answer: 90"""

CAT_GIUA_STEPS = """<think>
Total amount: 30 + 60 = 90 liters.
</think>
ENTITIES:
Initial = 120
STEPS:
Remaining = 120 - 90 = 30
Collected = 2 * 30 = 60
Total = 30 + 60 = 90"""

CAT_SOM = """<think>
Total amount: 30 + 60 = 90 liters.
</think>
ENTITIES:
Initial = 120
STEPS:
Remaining = 120 - 90 = 30"""


def test_1_co_final_answer_thi_nhanh_la_final():
    assert eb.score_how(IT, DAY_DU) == "final"
    assert eb.score_text(IT, DAY_DU) == 1


def test_2_CAT_GIUA_STEPS_van_duoc_diem_nhung_qua_nhanh_du_phong():
    """DUONG DUONG-TINH-GIA: chua bao gio viet 'Final Answer:' ma van 1 diem,
    chi vi so cuoi cua khoi STEPS tinh co dung. Phai NHIN THAY duoc."""
    assert eb.score_text(IT, CAT_GIUA_STEPS) == 1
    assert eb.score_how(IT, CAT_GIUA_STEPS) == "so_cuoi"


def test_3_cat_som_thi_truot_han():
    assert eb.score_text(IT, CAT_SOM) == 0
    assert eb.score_how(IT, CAT_SOM) == ""


def test_4_so_trong_think_KHONG_duoc_tinh_o_nhanh_du_phong():
    """<think> da tinh ra 90 nhung phan sinh ra ngoai think khong co 90 ->
    phai TRUOT. Neu khong, moi dau ra cut deu duoc diem."""
    txt = "<think>\nTong la 90.\n</think>\nENTITIES:\nA = 5\nSTEPS:\nx = 1 + 1 = 2"
    assert eb.score_text(IT, txt) == 0
    assert eb.score_how(IT, txt) == ""


def test_5_dong_bo_tren_nhieu_ca():
    cases = [DAY_DU, CAT_GIUA_STEPS, CAT_SOM, "", "linh tinh",
             "Final Answer: 91", "Final Answer: 90", "\\boxed{90}",
             "<think>90</think>", "STEPS:\nt = 89 + 1 = 90"]
    for txt in cases:
        assert (eb.score_how(IT, txt) != "") == (eb.score_text(IT, txt) == 1), \
            f"lech o {txt!r}"


def test_6_final_answer_thang_so_cuoi_khi_ca_hai_deu_co():
    """Uu tien phai giong score_text: Final Answer duoc xet TRUOC so cuoi."""
    txt = "Final Answer: 90\nghi chu them 77"
    assert eb.score_how(IT, txt) == "final"
