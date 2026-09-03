"""8+ mau DOC TAY cho score_eba (rule 15 -- quy-tac.md: PHAI doc tay >=8 mau
truoc khi tin diem tong cua mot phep do MOI). Moi case duoi day duoc chon de
mo phong DUNG cac kieu loi da doc tay that trong phien nay (bridge_oracle,
oracle_ablation, probe trich-so): bia so gan dung ten, nham distractor, roi
thong tin, van ban suy bien/rac."""
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "eba_gen", _HERE.parent / "models" / "qwen3_5" / "kv_transfer" / "eba_gen.py")
eba = importlib.util.module_from_spec(spec)
sys.modules["eba_gen"] = eba
spec.loader.exec_module(eba)


def _item():
    return {
        "meta": {
            "entities": {"Mara": 3, "Tobin": 7, "Priya": 12},
            "used_in_question": ["Mara", "Tobin"],
            "distractors": ["Priya"],
            "operation": "sum", "expr": "3 + 7",
            "attr_sg": "towel", "attr_pl": "towels",
            "final_answer": 10,
        }
    }


# 1. hoan toan dung -- A=1, B=1, C=1
def test_all_correct():
    it = _item()
    txt = "Mara has 3 towels. Tobin has 7 towels. 3 + 7 = 10.\nFinal Answer: 10"
    s = eba.score_eba(it, txt)
    assert s["A"] == 1.0 and s["B"] == 1.0 and s["C"] == 1


# 2. bia so gan DUNG ten (loi that da doc tay: "Kylie dung 3 khan" -> "6 khan")
def test_wrong_number_substitution():
    it = _item()
    txt = "Mara has 6 towels. Tobin has 7 towels. 6 + 7 = 13.\nFinal Answer: 13"
    s = eba.score_eba(it, txt)
    assert s["A"] < 1.0          # Mara sai -> keo trung binh xuong
    assert s["A"] == 0.0         # (1 dung [Tobin] + 1 sai [-1 Mara]) / 2 = 0
    assert s["C"] == 0           # dap so cuoi cung sai theo


# 3. mot thuc thi bi ROI (ten xuat hien, khong co so nao ke ben)
def test_entity_omitted():
    it = _item()
    txt = "Mara has some towels. Tobin has 7 towels. Final Answer: 7"
    s = eba.score_eba(it, txt)
    assert s["A"] == 0.5         # Mara=0 (roi), Tobin=1 (dung) -> tb 0.5
    assert s["C"] == 0


# 4. nham DISTRACTOR (dung dung ca hai thuc thi nhung LAY NHAM so cua Priya)
def test_distractor_leak():
    it = _item()
    txt = "Mara has 3 towels. Tobin has 12 towels. 3 + 12 = 15.\nFinal Answer: 15"
    s = eba.score_eba(it, txt)
    assert s["B"] == -1.0        # 12 la gia tri THAT cua Priya (distractor)
    assert s["C"] == 0


# 5. van ban suy bien/rac (nhu da thay o oracle_ablation D-variant) -- KHONG
#    duoc crash, phai tra ve 0 o moi lop
def test_degenerate_garbage():
    it = _item()
    txt = "p p p p p p p p p p p p p p p p"
    s = eba.score_eba(it, txt)
    assert s["A"] == 0.0 and s["C"] == 0
    assert isinstance(s["combined"], float)


# 6. so co dau phay (6,000) -- phai parse duoc, khong bi gay boi dau phay
def test_comma_formatted_number():
    it2 = _item()
    it2["meta"]["entities"]["Mara"] = 3000
    it2["meta"]["final_answer"] = 10000
    txt = "Mara has 3,000 towels. Tobin has 7 towels. Final Answer: 10,000"
    s = eba.score_eba(it2, txt)
    assert s["A"] == 1.0
    assert s["C"] == 1


# 7. dap so cuoi dung nhung sai o giua (mo phong "arithmetic tot, binding
#    lech" -- CAN tach duoc lop A/B khoi lop C, dung muc dich thiet ke)
def test_right_final_wrong_binding_midway():
    it = _item()
    # 9 khong phai gia tri that cua ai (Mara=3,Tobin=7,Priya=12) nhung dap so
    # cuoi (10) van dung -- mo phong 9B "doan dung' dap so du binding hong
    txt = "Mara has 9 towels. Tobin has 1 towels. 9 + 1 = 10.\nFinal Answer: 10"
    s = eba.score_eba(it, txt)
    assert s["A"] == -1.0        # ca hai deu bi bia so sai
    assert s["C"] == 1           # nhung C van cham dung -- CHINH LA ly do
                                  # phai tach lop, khong duoc chi nhin C


# 8. gold_template() tu meta LUON dung -- khong qua model nao nen phai
#    tu-cham diem toi da (A=1,B=1,C=1) tren chinh no
def test_gold_template_self_consistent():
    it = _item()
    it["gold"] = eba.gold_template(it["meta"])
    s = eba.score_eba(it, it["gold"])
    assert s == {"A": 1.0, "B": 1.0, "C": 1, "combined": 1.0}


# 9. gen_item() sinh ra item hop le o ca 4 muc difficulty, va gold tu-cham
#    toi da (kiem tra ca bo sinh, khong chi ham cham diem)
def test_gen_item_all_difficulties_self_consistent():
    import random
    for d in range(4):
        it = eba.gen_item(random.Random(d), d)
        s = eba.score_eba(it, it["gold"])
        assert s["A"] == 1.0 and s["C"] == 1, (d, it, s)
