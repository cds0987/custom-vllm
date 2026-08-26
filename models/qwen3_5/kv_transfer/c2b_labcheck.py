"""c2b_labcheck — user chi dao 2026-08-26: kiem tra 4B->9B copy NGUYEN o
TRANSFORMERS tren DUNG bo prompt C2b dang fail, TRUOC khi mo vLLM ra dieu tra.

Neu lab PASS (copy = self) -> loi nam trong duong van chuyen vLLM/LMCache
(cung co gia thuyet trang GDN roi). Neu lab cung FAIL -> E1 co dieu kien
bien ma bo de moi cham vao — xet lai truoc khi trach xe.

Protocol E1 nguyen ban: prefill ids[:, :-1], copy TRON cache (attn KV +
GDN recurrent + conv — khong zero gi het, khong mapper), feed token cuoi,
greedy 16. Model bf16 GOC hai dau (dung dieu kien E1 da chung minh 12/12).
"""

import argparse
import gc
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

PROMPTS_F = "/content/c2b_prompts.json"
SPILL = Path("/content/labchk")


def load_bf16(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return tok, m


def greedy(model, past, inp, n=16):
    import torch
    cur, out = past, []
    with torch.no_grad():
        for _ in range(n):
            o = model(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(int(inp))
    return out


def raw_copy(tpl, src):
    """Ghi de TRON moi tensor cua template 9B bang vo 4B (shape trung)."""
    a_s, g_s = e5.split_layers(src)
    a_t, g_t = e5.split_layers(tpl)
    for ks, kt in zip(sorted(a_s), sorted(a_t)):
        a_t[kt].keys = a_s[ks].keys
        a_t[kt].values = a_s[ks].values
    for gs, gt in zip(sorted(g_s), sorted(g_t)):
        e5._set_like(g_t[gt], "recurrent_states",
                     e5._get(g_s[gs].recurrent_states))
        e5._set_like(g_t[gt], "conv_states", e5._get(g_s[gs].conv_states))
    return tpl


def main():
    import re
    import torch
    prompts = json.load(open(PROMPTS_F))
    SPILL.mkdir(exist_ok=True)

    # PHA 1: 4B bf16 mot minh — spill tron vo
    if not (SPILL / "DONE4B").exists():
        tok4, m4 = load_bf16("Qwen/Qwen3.5-4B")
        with torch.no_grad():
            for i, p in enumerate(prompts):
                enc = tok4(p["prompt"], return_tensors="pt").to("cuda")
                past = e5.prefill_chunked(m4, enc["input_ids"][:, :-1])
                e5.spill_cache(past, SPILL / f"lab{i}.pt")
                del past
                torch.cuda.empty_cache()
                print(f"4B spill {i} T={enc['input_ids'].shape[1]}")
        del m4
        gc.collect(); torch.cuda.empty_cache()
        (SPILL / "DONE4B").touch()

    # PHA 2: 9B bf16 — self vs copy-nguyen tren cung prompt
    tok9, m9 = load_bf16("Qwen/Qwen3.5-9B")
    res = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            enc = tok9(p["prompt"], return_tensors="pt").to("cuda")
            pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
            row = {"i": i, "ctx": p["ctx"], "code": p["code"]}
            for cond in ("self", "copy"):
                tpl = e5.prefill_chunked(m9, pre)
                if cond == "copy":
                    src = e5.load_cache(SPILL / f"lab{i}.pt")
                    tpl = raw_copy(tpl, src)
                txt = tok9.decode(greedy(m9, tpl, last))
                hit = int(p["code"] in re.sub(r"\D", "", txt))
                row[cond] = hit
                row[cond + "_txt"] = txt[:60]
                if cond == "copy":
                    del src
                del tpl
                torch.cuda.empty_cache()
            res.append(row)
            print(f"LAB {i} ctx{p['ctx']}: self={row['self']} "
                  f"copy={row['copy']} copy_txt={row['copy_txt']!r}")
    ns = sum(r["self"] for r in res)
    nc = sum(r["copy"] for r in res)
    print(f"===== LABCHECK: self {ns}/{len(res)} | copy-nguyen {nc}/{len(res)} =====")
    with open("/content/logs/labcheck.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("LABCHECK_DONE")


if __name__ == "__main__":
    main()
