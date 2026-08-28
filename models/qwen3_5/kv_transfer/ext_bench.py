"""ext_bench -- benchmark THAT ngoai (khong tu tong hop) cho mapper 4B->27B,
user chi dinh 2026-08-27: "training-test 4-27 tren 1 bo dai vai ngin
samples voi du the loai cau hoi ... thu thuan 27B, sau do apply training
va re-test, de thay tong quat kha nang cua mapper".

3 bo (da tra loi AA-LCR bo, SWE-bench doi sang AIME_2024 vi ly do ha tang):
  musr    -- TAUR-Lab/MuSR (756 mau, 3 the loai suy luan nhieu buoc,
             trắc nghiệm). Cham: dung index dap an.
  aime    -- Maxwell-Jia/AIME_2024 (30 bai toan thi HSG, dap so nguyen
             0-999). Cham: so cuoi cung sinh ra == Answer.
  compute -- nvidia/compute-eval (config "default", CUDA/C++). Cham
             BANG BIEN DICH + CHAY THAT (nvcc build_command roi
             test_command, exit code 0 = pass) -- KHONG phai so khop
             van ban. Can nvcc tren runtime (kiem tra truoc khi chay).

Hai duong danh gia:
  self  -- 27B tu prefill toan bo prompt, KHONG mapper (co so "thuan").
  cross -- 4B prefill -> mapper (checkpoint --mapper) -> 27B decode tiep
           (giong cascade_427.py, dung chung ham voi e5_train.py).

Chay tung buoc (Colab):
  python ext_bench.py gen --bench musr,aime,compute --n-compute 60
  python ext_bench.py self --bench musr,aime,compute
  python ext_bench.py cross --mapper /content/mapper_v427_8k.pt --bench musr,aime,compute
  python ext_bench.py agg
"""

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

WARM_P = 5
# ngan sach token sinh: musr/compute da chan thinking san trong
# prompt nen khong can nhieu; aime CAN suy luan that (30 item).
N_NEW = {"musr": 24, "aime": 2560, "compute": 900,
         "bbh": 48, "gsm8k": 320, "math500": 640}
LOG_DIR = Path("/content/logs")
PROMPTS_F = "/content/ext_bench_items.json"


# --------------------------------------------------------------- loaders ---

def _musr_items(n=None):
    """n = tong so mau muon lay -- chia DEU 3 the loai (giu da dang). Cac id
    sinh ra on dinh theo thu tu goc nen ket qua self cu (756 mau) van dung
    lai duoc cho tap con."""
    from datasets import load_dataset
    items = []
    per = None if n is None else max(1, n // 3)
    for split in ("murder_mysteries", "object_placements", "team_allocation"):
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        for i, ex in enumerate(ds):
            if per is not None and i >= per:
                break
            # "choices" la python-repr (nhay don), khong phai JSON chuan
            # (bug thuc te: json.loads FAIL "Expecting value: line 1 column
            # 2" tren du lieu that) -> ast.literal_eval an toan hon eval.
            import ast
            try:
                choices = json.loads(ex["choices"])
            except json.JSONDecodeError:
                choices = ast.literal_eval(ex["choices"])
            letters = [chr(65 + j) for j in range(len(choices))]
            opts = "\n".join(f"{l}) {c}" for l, c in zip(letters, choices))
            # Qwen3.5 la model "thinking" -- mac dinh mo <think> roi suy luan
            # dai. BUG THAT da gap: max_new=8 -> 198/198 hit=0 vi model chua
            # kip thoat khoi think. Dong san khoi think de tra loi NGAY
            # (thu thuat nay thay trong chinh output that: "<think>\n\n</think>").
            prompt = (f"{ex['narrative']}\n\n{ex['question']}\n{opts}\n\n"
                      f"Answer with the letter of the correct choice.\n"
                      f"<think>\n\n</think>\n\nAnswer: ")
            items.append({"bench": "musr", "sub": split, "id": f"{split}{i}",
                          "prompt": prompt, "expect": letters[ex["answer_index"]]})
    return items


def _aime_items():
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    items = []
    for ex in ds:
        prompt = (f"Solve this competition math problem. Give your final "
                  f"answer as a single integer on the last line, prefixed "
                  f"with 'Final Answer: '.\n\nProblem: {ex['Problem']}\n\n"
                  f"Solution: ")
        items.append({"bench": "aime", "sub": "2024", "id": ex["ID"],
                      "prompt": prompt, "expect": str(ex["Answer"])})
    return items


def _parquet(repo, path, rev="refs/convert/parquet"):
    """Doc thang parquet -- ne loi validate Features cua `datasets` (da dinh
    voi compute-eval: feature type 'Json' khong nhan dien duoc)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset",
                                           revision=rev))


BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]


def _bbh_items(n):
    """BIG-Bench Hard: lay DEU tren 26 tac vu de do dien rong suy luan."""
    per = max(1, n // len(BBH_TASKS))
    items = []
    for task in BBH_TASKS:
        try:
            df = _parquet("lukaemon/bbh", f"{task}/test/0000.parquet")
        except Exception as e:
            print(f"bbh skip {task}: {type(e).__name__}")
            continue
        for i, row in df.head(per).iterrows():
            items.append({
                "bench": "bbh", "sub": task, "id": f"{task}/{i}",
                "prompt": (f"{row['input']}\n\nAnswer concisely.\n"
                           f"<think>\n\n</think>\n\nAnswer: "),
                "expect": str(row["target"]).strip()})
        if len(items) >= n:
            break
    return items[:n]


def _gsm8k_items(n):
    df = _parquet("openai/gsm8k", "main/test/0000.parquet")
    items = []
    for i, row in df.head(n).iterrows():
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        items.append({
            "bench": "gsm8k", "sub": "main", "id": f"gsm8k/{i}",
            "prompt": (f"Solve step by step, then give the final numeric "
                       f"answer after 'Final Answer: '.\n\n"
                       f"Problem: {row['question']}\n\n"
                       f"<think>\n\n</think>\n\nSolution: "),
            "expect": gold})
    return items


def _math500_items(n):
    df = _parquet("HuggingFaceH4/MATH-500", "default/test/0000.parquet")
    items = []
    for i, row in df.head(n).iterrows():
        items.append({
            "bench": "math500", "sub": str(row["subject"]), "id": f"math500/{i}",
            "prompt": (f"Solve this problem. Put your final answer in "
                       f"\\boxed{{}}.\n\nProblem: {row['problem']}\n\n"
                       f"Solution: "),
            "expect": str(row["answer"]).strip()})
    return items


def _compute_items(n):
    # datasets.load_dataset() FAIL that: schema compute-eval dung feature
    # type "Json" (cot timing_mode) ma ban `datasets` tren runtime khong
    # nhan dien ("Feature type 'Json' not found"). Ne bang cach doc THANG
    # file parquet qua pandas/huggingface_hub, bo qua lop validate Features
    # cua thu vien datasets hoan toan.
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("nvidia/compute-eval", "default/eval/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    df = pd.read_parquet(path)
    items = []
    for i, row in df.iterrows():
        if i >= n:
            break
        ex = row.to_dict()
        # pyarrow struct-list co the ve dang numpy array cua dict-like
        # (khong luon la dict thuan) -> ep ve list[dict] tuong minh de
        # score_compute() sau nay doc bang f["path"]/f["content"] on dinh.
        cfiles = [dict(f) for f in ex["context_files"]]
        tfiles = [dict(f) for f in ex["test_files"]]
        ctx = "\n\n".join(f"// {f['path']}\n{f['content']}" for f in cfiles)
        prompt = (f"{ex['prompt']}\n\nContext files (do not repeat these, "
                  f"they are already provided):\n{ctx}\n\n"
                  f"Write the complete contents of solution.cu implementing "
                  f"the function. Output ONLY the code, no explanation, "
                  f"wrapped in a single ```cuda code block.\n"
                  f"<think>\n\n</think>\n\n```cuda\n")
        items.append({"bench": "compute", "sub": ex["group"], "id": ex["task_id"],
                      "prompt": prompt,
                      "context_files": cfiles,
                      "test_files": tfiles,
                      "build_command": ex["build_command"],
                      "test_command": ex["test_command"],
                      "timeout": float(ex["timeout_seconds"]) if ex["timeout_seconds"] else 60})
    return items


def gen(bench_list, n_compute, n_each=500):
    items = []
    if "musr" in bench_list:
        items += _musr_items(n_each)
    if "aime" in bench_list:
        items += _aime_items()
    if "bbh" in bench_list:
        items += _bbh_items(n_each)
    if "gsm8k" in bench_list:
        items += _gsm8k_items(n_each)
    if "math500" in bench_list:
        items += _math500_items(n_each)
    if "compute" in bench_list:
        items += _compute_items(n_compute)
    with open(PROMPTS_F, "w") as fh:
        json.dump(items, fh)
    by = {}
    for it in items:
        by[it["bench"]] = by.get(it["bench"], 0) + 1
    print(f"gen: {len(items)} items {by} -> {PROMPTS_F}")


# --------------------------------------------------------------- scoring ---

def _extract_code(text):
    m = re.search(r"```(?:cuda|cpp|c\+\+)?\n(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def _strip_think(text):
    """Bo khoi <think>...</think> neu model van tu mo (du prompt da dong)."""
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.S)


def _all_boxed(text):
    r"""Trich MOI \boxed{...} voi ngoac CAN BANG. Regex [^}]* that bai voi
    LaTeX long nhau: \boxed{\frac{1}{2}} bi cat thanh '\frac{1' (bug that,
    bat duoc bang self-test truoc khi chay GPU)."""
    out = []
    for m in re.finditer(r"\\boxed\{", text):
        i = m.end()
        depth, start = 1, i
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start:i - 1])
    return out


def _norm_math(x):
    """Chuan hoa dap an toan: bo dau phay, $, khoang trang, .0 thua."""
    x = str(x).strip().replace(",", "").replace("$", "").replace(" ", "")
    x = x.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", x):
        x = x.split(".")[0]
    return x


def score_text(it, text):
    t = _strip_think(text)
    if it["bench"] == "bbh":
        # dap an BBH thuong la (A)/(B), Yes/No, hoac chuoi ngan -> so khop
        # sau khi chuan hoa hoa/thuong + bo ngoac.
        exp = it["expect"].strip()
        low_t, low_e = t.lower(), exp.lower()
        if re.fullmatch(r"\([A-Z]\)", exp):
            m = re.search(r"\(([A-Za-z])\)", t)
            return int(bool(m) and f"({m.group(1).upper()})" == exp)
        return int(low_e in low_t)
    if it["bench"] in ("gsm8k", "math500"):
        exp = _norm_math(it["expect"])
        boxed = _all_boxed(text)
        if boxed and _norm_math(boxed[-1]) == exp:
            return 1
        fin = re.findall(r"Final Answer:\s*\$?([^\n$]+)", text)
        if fin and _norm_math(fin[-1]) == exp:
            return 1
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        return int(bool(nums) and _norm_math(nums[-1]) == exp)
    if it["bench"] == "musr":
        m = re.search(r"\b([A-F])\b", t)
        return int(bool(m) and m.group(1) == it["expect"])
    if it["bench"] == "aime":
        # \boxed{} CUOI CUNG tren van ban DAY DU (ke ca trong <think>):
        # do la ket luan cua model theo chuan eval AIME. Khong dung
        # text da _strip_think vi dap an thuong nam trong do.
        # AIME ghi dap an 3 chu so co so 0 dan (\boxed{033} = 33) -- so khop
        # chuoi tho se cham HUT diem model. So sanh theo GIA TRI SO.
        def _eq(x):
            try:
                return int(x) == int(it["expect"])
            except ValueError:
                return False
        boxed = re.findall(r"\\boxed\{\s*(-?\d+)", text)
        if boxed:
            return int(_eq(boxed[-1]))
        fin = re.findall(r"Final Answer:\s*\$?\\?b?o?x?e?d?\{?\s*(-?\d+)", text)
        if fin:
            return int(_eq(fin[-1]))
        nums = re.findall(r"-?\d+", t)
        return int(bool(nums) and _eq(nums[-1]))
    raise ValueError(it["bench"])


def score_compute(it, text, workdir):
    """Bien dich THAT + chay test THAT. Tra ve (hit, log[:2000])."""
    code = _extract_code(text)
    d = Path(workdir)
    (d / "include").mkdir(parents=True, exist_ok=True)
    (d / "test").mkdir(parents=True, exist_ok=True)
    for f in it["context_files"]:
        p = d / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"])
    for f in it["test_files"]:
        p = d / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"])
    (d / "solution.cu").write_text(code)
    try:
        b = subprocess.run(it["build_command"], shell=True, cwd=d,
                           capture_output=True, text=True, timeout=120)
        if b.returncode != 0:
            return 0, f"BUILD_FAIL rc={b.returncode}\n{b.stderr[-1500:]}"
        t = subprocess.run(it["test_command"], shell=True, cwd=d,
                           capture_output=True, text=True,
                           timeout=it.get("timeout", 60))
        return int(t.returncode == 0), f"TEST rc={t.returncode}\n{t.stdout[-800:]}{t.stderr[-800:]}"
    except subprocess.TimeoutExpired:
        return 0, "TIMEOUT"
    except Exception as e:
        return 0, f"HARNESS_ERROR {type(e).__name__}: {e}"


def check_nvcc():
    r = subprocess.run(["bash", "-c", "nvcc --version"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout


# ------------------------------------------------------------- self run ---

def run_self(bench_list, tgt_model, max_len, sl, tgt_quant="bnb"):
    import torch
    # self PHAI nap cung dang luong tu voi cross, neu khong thi so sanh
    # retention (mapped/self) la so sanh hai model khac nhau
    tok, m = _load_target(tgt_model, tgt_quant)
    items = json.load(open(PROMPTS_F))
    items = [it for it in items if it["bench"] in bench_list]
    a, b = (int(x) for x in sl.split(":")) if sl else (0, len(items))
    items = items[a:b]
    out = []
    for i, it in enumerate(items):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=max_len).to("cuda")
        n_new = N_NEW[it["bench"]]
        t0 = time.time()
        with torch.no_grad():
            gen_ids = m.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        row = {"id": it["id"], "bench": it["bench"], "text": txt, "lat": round(time.time() - t0, 2)}
        if it["bench"] == "compute":
            with tempfile.TemporaryDirectory() as td:
                hit, log = score_compute(it, txt, td)
            row["hit"], row["log"] = hit, log
        else:
            row["hit"] = score_text(it, txt)
        out.append(row)
        print(f"self {it['bench']} {it['id']} hit={row['hit']} lat={row['lat']}s "
              f"{txt[:50]!r}")
        torch.cuda.empty_cache()
    LOG_DIR.mkdir(exist_ok=True)
    suffix = f"_{sl.replace(':', '_')}" if sl else ""
    path = LOG_DIR / f"extbench_self{suffix}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"self saved -> {path}")


# ------------------------------------------------------------ cross run ---

def _load_target(name, quant):
    """Nap model DICH theo dung dang se SERVE.

    Khe ho train/serve (user 2026-08-28): mapper hoc tren stock 9B + bnb NF4,
    nhung production chay champion (khung W4A16 + trong so GDN ghep tu GGUF
    Q4_K_M) -- khac CA luong tu LAN trong so. Bang chung cu mau thuan: E6c do
    bnb ganh NUA vet nut (bf16 9/20 vs bnb 4-6/20), con C2b-8 do bf16 va W4A16
    GIONG HET. Chua lan nao do tren mot mapper DA HUAN LUYEN -> phai do.

    quant='bnb'  : bnb NF4 (giong luc train)
    quant='auto' : de checkpoint tu khai (champion = compressed-tensors W4A16)
    quant='bf16' : khong luong tu — moc doi chung sach
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if quant == "bnb":
        return e5.load_4bit(name)
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(device_map="cuda")
    if quant == "bf16":
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"nap dich {name} (quant={quant})", flush=True)
    return tok, model


def run_cross(bench_list, src_model, tgt_model, mapper_path, max_len, sl,
              lora_path="", tgt_quant="bnb"):
    """HAI PHA (bat buoc tren L4): 4B va 27B KHONG BAO GIO cung tren GPU.

    Bug that da gap: nap ca hai cung luc = 3,5GB + 18GB = 21,5GB tren card
    22GB -> OOM tu item thu 9. Docstring e5_train canh bao tu dau. Cach
    giai giong cascade_427.py: pha 1 spill cache 4B ra dia, pha 2 doc lai.
    """
    import gc
    import torch
    from transformers import AutoConfig

    items = json.load(open(PROMPTS_F))
    items = [it for it in items if it["bench"] in bench_list]
    a, b = (int(x) for x in sl.split(":")) if sl else (0, len(items))
    items = items[a:b]
    spill = Path("/content/cross_spill")
    spill.mkdir(parents=True, exist_ok=True)

    # ---------------- PHA 1: 4B mot minh -> spill cache ra dia -------------
    tok_s, model_s = e5.load_4bit(src_model)
    if lora_path:
        # kien truc 2 lop (e9_joint): 4B da duoc LoRA ep "doc ho" cho model
        # dich. Khong nap LoRA o day = do NHAM mapper cu tren 4B goc.
        from peft import PeftModel
        model_s = PeftModel.from_pretrained(model_s, lora_path)
        model_s = model_s.merge_and_unload()
        model_s.eval()
        print(f"da nap+merge LoRA 4B: {lora_path}", flush=True)
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    # transformers 5.15 boc recurrent_states/keys trong dict {0: tensor}
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s
    for i, it in enumerate(items):
        pth = spill / f"x{i}.pt"
        if pth.exists():
            continue
        ids = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=max_len)["input_ids"].to("cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_s, ids[:, :-WARM_P])
        e5.spill_cache(past, pth)
        del past
        torch.cuda.empty_cache()
        if i % 50 == 0:
            print(f"cross-A {it['bench']} {i}/{len(items)}", flush=True)
    del model_s
    gc.collect()
    torch.cuda.empty_cache()
    print("CROSS_PHASE_A_DONE", flush=True)

    # ---------------- PHA 2: 27B mot minh -> map + decode ------------------
    tok_t, model_t = _load_target(tgt_model, tgt_quant)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.load(mapper_path)
    print(f"mapper nap: {mapper_path}", flush=True)

    out = []
    for i, it in enumerate(items):
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src_past = e5.load_cache(spill / f"x{i}.pt")
        n_new = N_NEW[it["bench"]]
        t0 = time.time()
        with torch.no_grad():
            tpl = e5.prefill_chunked(model_t, cut)
            student_past = e5.build_student_past(tpl, src_past, mapper)
            del tpl
            cur, gen_ids, inp = student_past, [], warm
            o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen_ids.append(int(inp))
            for _ in range(n_new - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen_ids.append(int(inp))
                if inp.item() == tok_t.eos_token_id:
                    break
        txt = tok_t.decode(gen_ids)
        row = {"id": it["id"], "bench": it["bench"], "text": txt,
               "lat": round(time.time() - t0, 2)}
        row["hit"] = score_text(it, txt)
        out.append(row)
        print(f"cross {it['bench']} {it['id']} hit={row['hit']} "
              f"lat={row['lat']}s {txt[:50]!r}", flush=True)
        del src_past, student_past, cur, o
        torch.cuda.empty_cache()
    LOG_DIR.mkdir(exist_ok=True)
    suffix = f"_{sl.replace(':', '_')}" if sl else ""
    path = LOG_DIR / f"extbench_cross{suffix}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"cross saved -> {path}")
    # don cache spill de khong day dia cho bo tiep theo
    shutil.rmtree(spill, ignore_errors=True)


# ------------------------------------------------------------------ agg ---

def agg():
    import glob
    from collections import defaultdict
    cells = defaultdict(lambda: [0, 0])
    for kind in ("self", "cross"):
        for f in sorted(glob.glob(str(LOG_DIR / f"extbench_{kind}*.json"))):
            for r in json.load(open(f)):
                key = (r["bench"], kind)
                cells[key][0] += 1
                cells[key][1] += r["hit"]
    print(f"{'bench':<10}{'kind':<8}{'n':>6}{'hit':>6}{'rate':>8}")
    for (bench, kind), (n, hit) in sorted(cells.items()):
        print(f"{bench:<10}{kind:<8}{n:>6}{hit:>6}{hit / max(n,1):>8.1%}")
    print("AGG_EXTBENCH_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "self", "cross", "agg", "check-nvcc"])
    ap.add_argument("--bench", default="musr,bbh,gsm8k,math500,aime")
    ap.add_argument("--n-compute", type=int, default=60)
    ap.add_argument("--n-each", type=int, default=500,
                    help="so mau moi bo bbh/gsm8k/math500")
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--mapper", default="/content/mapper_v427_8k.pt")
    ap.add_argument("--tgt-quant", default="bnb",
                    choices=["bnb", "auto", "bf16"],
                    help="dang luong tu cua model DICH: bnb = giong luc train; "
                         "auto = de checkpoint tu khai (champion W4A16); "
                         "bf16 = moc doi chung sach. Do khe ho train/serve.")
    ap.add_argument("--lora", default="",
                    help="thu muc LoRA cua 4B (kien truc 2 lop e9_joint). "
                         "Bo trong = 4B goc, chi co mapper.")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--slice", default="")
    args = ap.parse_args()
    bl = args.bench.split(",")
    if args.mode == "gen":
        gen(bl, args.n_compute, args.n_each)
    elif args.mode == "check-nvcc":
        ok, out = check_nvcc()
        print("NVCC_OK" if ok else "NVCC_MISSING", out[:200])
    elif args.mode == "self":
        run_self(bl, args.tgt_model, args.max_len, args.slice,
                 args.tgt_quant)
    elif args.mode == "cross":
        run_cross(bl, args.src_model, args.tgt_model, args.mapper,
                  args.max_len, args.slice, args.lora, args.tgt_quant)
    else:
        agg()
