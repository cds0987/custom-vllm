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
import subprocess
import tempfile
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

WARM_P = 5
LOG_DIR = Path("/content/logs")
PROMPTS_F = "/content/ext_bench_items.json"


# --------------------------------------------------------------- loaders ---

def _musr_items():
    from datasets import load_dataset
    items = []
    for split in ("murder_mysteries", "object_placements", "team_allocation"):
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        for i, ex in enumerate(ds):
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
            prompt = (f"{ex['narrative']}\n\n{ex['question']}\n{opts}\n\n"
                      f"Answer with the letter of the correct choice.\nAnswer: ")
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
                  f"wrapped in a single ```cuda code block.\n\n```cuda\n")
        items.append({"bench": "compute", "sub": ex["group"], "id": ex["task_id"],
                      "prompt": prompt,
                      "context_files": cfiles,
                      "test_files": tfiles,
                      "build_command": ex["build_command"],
                      "test_command": ex["test_command"],
                      "timeout": float(ex["timeout_seconds"]) if ex["timeout_seconds"] else 60})
    return items


def gen(bench_list, n_compute):
    items = []
    if "musr" in bench_list:
        items += _musr_items()
    if "aime" in bench_list:
        items += _aime_items()
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


def score_text(it, text):
    if it["bench"] == "musr":
        m = re.search(r"\b([A-F])\b", text)
        return int(bool(m) and m.group(1) == it["expect"])
    if it["bench"] == "aime":
        nums = re.findall(r"-?\d+", text)
        return int(bool(nums) and nums[-1] == it["expect"])
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

def run_self(bench_list, tgt_model, max_len, sl):
    import torch
    tok, m = e5.load_4bit(tgt_model)
    items = json.load(open(PROMPTS_F))
    items = [it for it in items if it["bench"] in bench_list]
    a, b = (int(x) for x in sl.split(":")) if sl else (0, len(items))
    items = items[a:b]
    out = []
    for i, it in enumerate(items):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=max_len).to("cuda")
        n_new = 800 if it["bench"] == "compute" else (600 if it["bench"] == "aime" else 8)
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

def run_cross(bench_list, src_model, tgt_model, mapper_path, max_len, sl):
    import torch
    from transformers import AutoConfig
    tok_s, model_s = e5.load_4bit(src_model)
    tok_t, model_t = e5.load_4bit(tgt_model)
    assert tok_s.vocab_size == tok_t.vocab_size or len(tok_s) == len(tok_t), \
        "tokenizer 4B/27B lech vocab"
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(src_model).get_text_config())
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    k0 = next(iter(a_s.values())).keys
    attn_dim = k0.shape[1] * k0.shape[3]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.load(mapper_path)
    print(f"mapper nap: {mapper_path}")

    items = json.load(open(PROMPTS_F))
    items = [it for it in items if it["bench"] in bench_list]
    a, b = (int(x) for x in sl.split(":")) if sl else (0, len(items))
    items = items[a:b]
    out = []
    for i, it in enumerate(items):
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        with torch.no_grad():
            src_past = e5.prefill_chunked(model_s, cut)
            tpl = e5.prefill_chunked(model_t, cut)
            student_past = e5.build_student_past(tpl, src_past, mapper)
            n_new = 800 if it["bench"] == "compute" else (600 if it["bench"] == "aime" else 8)
            t0 = time.time()
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
        if it["bench"] == "compute":
            with tempfile.TemporaryDirectory() as td:
                hit, log = score_compute(it, txt, td)
            row["hit"], row["log"] = hit, log
        else:
            row["hit"] = score_text(it, txt)
        out.append(row)
        print(f"cross {it['bench']} {it['id']} hit={row['hit']} lat={row['lat']}s "
              f"{txt[:50]!r}")
        del src_past, tpl, student_past
        torch.cuda.empty_cache()
    LOG_DIR.mkdir(exist_ok=True)
    suffix = f"_{sl.replace(':', '_')}" if sl else ""
    path = LOG_DIR / f"extbench_cross{suffix}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"cross saved -> {path}")


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
    ap.add_argument("--bench", default="musr,aime,compute")
    ap.add_argument("--n-compute", type=int, default=60)
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--mapper", default="/content/mapper_v427_8k.pt")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--slice", default="")
    args = ap.parse_args()
    bl = args.bench.split(",")
    if args.mode == "gen":
        gen(bl, args.n_compute)
    elif args.mode == "check-nvcc":
        ok, out = check_nvcc()
        print("NVCC_OK" if ok else "NVCC_MISSING", out[:200])
    elif args.mode == "self":
        run_self(bl, args.tgt_model, args.max_len, args.slice)
    elif args.mode == "cross":
        run_cross(bl, args.src_model, args.tgt_model, args.mapper, args.max_len, args.slice)
    else:
        agg()
