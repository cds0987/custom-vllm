"""test_mcnemar -- bai kiem cho phep so cap (rule 15: thang do MOI phai kiem
truoc khi tin). Cac ca duoi day deu la loi CU THE de mac:
 - so hai ty le nhu mau doc lap (bo qua ghep cap)
 - dung xap xi chi-binh-phuong khi so ca lech qua it
 - khong giao dung tap id giua hai file

    python -m pytest test_mcnemar.py -q
"""
import importlib.util
import pathlib

_H = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("mcnemar", _H / "mcnemar.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


def _bo(bits):
    return {f"gsm8k/{i}": v for i, v in enumerate(bits)}


def test_1_giong_het_nhau_thi_p_bang_1():
    a = _bo([1, 0, 1, 0, 1])
    r = mc.mcnemar(a, dict(a))
    assert r["n_lech"] == 0 and r["p"] == 1.0


def test_2_dem_dung_hai_o_lech():
    a = _bo([1, 1, 0, 0])
    b = _bo([1, 0, 1, 0])
    r = mc.mcnemar(a, b)
    assert (r["a_thang"], r["b_thang"]) == (1, 1)


def test_3_chi_lay_id_CHUNG_cua_hai_file():
    """Neu mot ben thieu mau, khong duoc tinh no vao -- ghep cap moi hop le."""
    a = _bo([1, 1, 1])
    b = {"gsm8k/0": 0, "gsm8k/1": 1, "gsm8k/9": 1}
    r = mc.mcnemar(a, b)
    assert r["n"] == 2 and r["a_thang"] == 1


def test_4_it_ca_lech_thi_dung_nhi_thuc_chinh_xac():
    """3 thang / 0 thua: chi2 Yates cho p~0,08 nhung nhi thuc chinh xac cho
    p=0,25 -- dung xap xi o day la bao dong gia."""
    a = _bo([1, 1, 1] + [1] * 20)
    b = _bo([0, 0, 0] + [1] * 20)
    r = mc.mcnemar(a, b)
    assert "nhi thuc" in r["cach"]
    assert abs(r["p"] - 0.25) < 1e-9


def test_5_lech_nhieu_va_lech_han_thi_co_y_nghia():
    a = _bo([1] * 30 + [0] * 30 + [1] * 40)
    b = _bo([0] * 30 + [0] * 30 + [1] * 40)
    r = mc.mcnemar(a, b)
    assert r["a_thang"] == 30 and r["b_thang"] == 0
    assert r["p"] < 0.05


def test_6_ty_le_bang_nhau_van_co_the_khac_nhau_that():
    """Cot loi cua ghep cap: hai ben CUNG 50% nhung lech nhau o MOI mau ->
    so ty le se bao 'giong het', McNemar bao 'khong ket luan duoc' chu khong
    bao 'giong nhau'."""
    a = _bo([1, 0] * 10)
    b = _bo([0, 1] * 10)
    r = mc.mcnemar(a, b)
    assert r["a_dung"] == r["b_dung"] == 10
    assert r["n_lech"] == 20 and r["p"] > 0.05


def test_7_xu_huong_thang_nhung_n_nho_thi_KHONG_ket_luan():
    """Tai lap dung tinh huong da gap: 9 thang / 3 thua -> p~0,15."""
    a = _bo([1] * 9 + [0] * 3 + [1] * 88)
    b = _bo([0] * 9 + [1] * 3 + [1] * 88)
    r = mc.mcnemar(a, b)
    assert (r["a_thang"], r["b_thang"]) == (9, 3)
    assert r["p"] > 0.05


def test_8_doc_duoc_dinh_dang_eval_big():
    """eval_big ghi {id: {"hit": 0/1, "txt": ...}} -- doc() phai rut dung
    truong hit, khong duoc coi ca dict la gia tri."""
    import json
    f = _H / "_tmp_test_mcnemar.json"
    try:
        f.write_text(json.dumps({"gsm8k/0": {"hit": 1, "txt": "x"},
                                 "gsm8k/1": {"hit": 0, "txt": "y"}}))
        assert mc.doc(str(f)) == {"gsm8k/0": 1, "gsm8k/1": 0}
    finally:
        f.unlink(missing_ok=True)
