"""gen_struct_gold -- BUOC 0: sinh ground truth CO CAU TRUC bang 9B tu giai.

Khac gen_pseudo_vllm.py (sinh CoT tu do): o day ep 9B xuat ra dung khuon
<think> ... </think> + ENTITIES + STEPS + Final Answer, de mapper co cai ma
HOC theo trong buoc SFT, va co cai ma CHAM theo trong buoc RL.

PROMPT: khau NAY (tao du lieu) duoc dung prompt CO CHI DAN format -- 9B doc de
GOC THAT, khong qua mapper. Nhung phan luu lai lam `gold` se duoc dung voi
PROMPT GOC luc train/serve (ket thuc o "Solution: "), nen model phai hoc cach
tu sinh khuon do tu trong TRONG SO, khong nho prompt nhac (user chot 2026-09-04).

LOC HAI TANG -- chi giu mau vua parse duoc DU 3 khoi, vua ra dap so DUNG:
9B lam sai thi LOAI, khong bao gio day mapper theo loi giai sai.

    python -u gen_struct_gold.py --n 7473 --out /content/struct_gold_gsm.json
"""
import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gs = _load("gsm_struct")

# Chi dan format CHI dung o khau sinh du lieu nay. Vi du mau (one-shot) de 9B
# bam dung khuon -- khong co vi du thi model hay bo khoi ENTITIES.
FORMAT_HINT = """Answer in EXACTLY this format, nothing else:

<think>
Brief analysis: identify each entity, its value, and how entities relate.
</think>
ENTITIES:
<name> = <number>            (one line per number stated in the problem)
<name> = <name> <op> <number>  (one line per relation between entities)
STEPS:
<label> = <expression> = <result>   (one line per computation, in order)
Final Answer: <number>

Example:
<think>
Kylie has 20 shells. Robert has 5 more than Kylie. Need the total.
</think>
ENTITIES:
Kylie = 20
Robert = Kylie + 5
STEPS:
Robert = 20 + 5 = 25
total = 20 + 25 = 45
Final Answer: 45

Now solve this problem.

"""


def build_prompt(problem_text):
    """problem_text = phan 'Problem: ...' trich tu prompt goc."""
    return (FORMAT_HINT + problem_text +
            "\n\n<think>\n")   # ep model bat dau ngay bang phan tich


_FA = re.compile(r"Final Answer:[^\n]*")


def cut_after_answer(text):
    """CAT bo moi thu sau dong 'Final Answer: X'.

    Doc tay 200 mau dau (rule 15) bat duoc: 9B hay noi them sau khi da tra loi
    xong -- 'Now solve this problem.' / 'Wait, I need to check the format
    strictly. The example shows: ENTITIES: ...'. Neu de nguyen lam gold thi
    dang DAY model thoi lam nham sau dap an, va lam hong luon phep cham (khoi
    ENTITIES gia o duoi se bi parse nham)."""
    m = _FA.search(text)
    return text[:m.end()].rstrip() if m else text.strip()


def extract_problem(prompt):
    """Lay lai de bai tu prompt goc (giua 'Problem:' va '<think>')."""
    i = prompt.find("Problem:")
    j = prompt.find("<think>")
    if i < 0:
        return prompt.strip()
    return prompt[i:(j if j > i else len(prompt))].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--out", default="/content/struct_gold_gsm.json")
    ap.add_argument("--model", default="gunnybd01/qwen35-9b-champion",
                    help="MAC DINH champion W4A16 (compressed-tensors) -- vLLM "
                         "0.28 DA BO ho tro 'bitsandbytes' (danh sach quant moi "
                         "khong con no), nen duong bnb cu cua gen_pseudo_vllm.py "
                         "khong chay duoc nua. Champion la ban 4-bit cua CHINH "
                         "9B, ppl 4,7637 (bf16: 5,13) -- da do trong du an.")
    ap.add_argument("--quant", default="auto", choices=["auto", "bnb", "bf16"],
                    help="auto = de vLLM tu doc tu config (compressed-tensors)")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--n", type=int, default=0, help="0 = tat ca")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="384 lam cat cut mau co khoi <think> dai (doc tay lan "
                         "thu 200 thay day la mot phan cua 32% parse hong)")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="struct_gold")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    data = json.loads(Path(args.data_file).read_text())
    pool = [it for sp in ("train", "val") for it in data.get(sp, [])
            if it.get("kind") == "gsm8k"]
    if args.n:
        pool = pool[:args.n]
    print(f"se sinh cho {len(pool)} mau gsm8k", flush=True)

    kw = dict(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.util, enforce_eager=False)
    if args.quant == "bnb":
        kw["quantization"] = "bitsandbytes"
    # quant == "auto"/"bf16": khong truyen co -> vLLM doc tu config cua model
    t0 = time.time()
    llm = LLM(**kw)
    print(f"vLLM nap xong {time.time()-t0:.0f}s (quant={args.quant})", flush=True)

    prompts = [build_prompt(extract_problem(it["prompt"])) for it in pool]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        stop=["\nProblem:", "\nExample:"])
    t1 = time.time()
    res = llm.generate(prompts, sp)
    print(f"sinh xong {time.time()-t1:.0f}s", flush=True)

    out, n_parse, n_ans = {}, 0, 0
    for it, r in zip(pool, res):
        txt = "<think>\n" + r.outputs[0].text        # bu lai phan bi cat o prompt
        txt = cut_after_answer(txt)                  # bo rac duoi (xem docstring)
        p = gs.parse(txt)
        if not p["ok"]:
            continue
        n_parse += 1
        want = gs._num(it.get("expect"))
        if want is None or p["answer"] != want:      # TANG 2: 9B sai -> LOAI
            continue
        n_ans += 1
        out[it["id"]] = {
            "gold": txt.strip(),
            "answer": p["answer"],
            "entities": sorted([list(x) for x in p["entities"]]),
            "relations": len(p["relations"]),
            "n_steps": len(p["steps"]),
        }

    n = len(pool)
    print(f"\n=== KET QUA BUOC 0 ===")
    print(f"tong mau           : {n}")
    print(f"parse duoc du 3 khoi: {n_parse} ({100*n_parse/max(n,1):.1f}%)")
    print(f"VA dap so DUNG      : {n_ans} ({100*n_ans/max(n,1):.1f}%)  <- giu lai")
    print(f"tong thoi gian      : {(time.time()-t0)/60:.1f} phut")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"da ghi {args.out}")

    if n_ans < 0.5 * n:
        print("\n*** CANH BAO: giu lai <50% -- theo cong kiem da dat ra, PHAI "
              "chinh prompt truoc khi di tiep, KHONG chay SFT voi du lieu nay.")

    # doc tay: in 3 mau dau de kiem bang mat (rule 15)
    for i, (k, v) in enumerate(list(out.items())[:3]):
        print(f"\n--- MAU {i+1} ({k}) ---\n{v['gold'][:600]}")

    if args.hf_repo and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=args.out,
                path_in_repo=f"{args.hf_prefix}/{Path(args.out).name}",
                repo_id=args.hf_repo)
            print(f"\nHF-UP {Path(args.out).name}")
        except Exception as ex:
            print(f"HF-UP FAIL: {type(ex).__name__}: {ex}")
    print("GEN_STRUCT_GOLD_EXIT", flush=True)


if __name__ == "__main__":   # BAT BUOC: LLM() o module level chet vi spawn
    main()
