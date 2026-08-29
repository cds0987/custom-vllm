"""eval_big -- eval NIEM PHONG quy mo vai nghin mau (user 2026-08-29:
"phai tren vai ngin sample eval chu", "co offline vllm roi ma").

BA DAI DU LIEU TACH BIET (khong mau nao dung lai):
    bo      test cu (da bao cao)   train (gen_data)      NIEM PHONG LON (o day)
    bbh     hang 0-7 moi tac vu    hang 7-107            hang 107-250  (~3700)
    gsm8k   test[0:200]            split `train`         test[200:1319] (~1100)
    musr    0-66 moi split         66-232                232-het        (~60)
`--check-overlap` doi chieu CHUOI PROMPT voi ca hai tap kia truoc khi chay.

HAI CHE DO, hai engine, vi ly do vat ly:
  self   -- vLLM offline. Nhanh (~580 tok/s do duoc), va la con so DUNG:
            vLLM dung dung token ket thuc, con vong greedy tay tung dung sai
            (xem e5.stop_ids: 92% vs 32% tren cung 40 mau).
  mapped -- BUOC PHAI o transformers: vLLM khong cho tiem cache do mapper
            dung. Tuan tu, ~12 tok/s. gsm8k (320 token/mau) la phan dat nhat.

Chay:
  python -u eval_big.py gen    --n-bbh 1300 --n-gsm8k 500 --n-musr 60
  python -u eval_big.py self                       # vLLM
  python -u eval_big.py mapped --mapper ... --lora ...
  python -u eval_big.py agg
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

_H = Path(__file__).parent
ITEMS_F = "/content/eval_big_items.json"
OUT_DIR = Path("/content/logs")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gd = _load("gen_data")
eb = _load("ext_bench")

N_NEW = {"gsm8k": 320, "bbh": 48, "musr": 24}
WARM_P = 5

# ranh gioi dai da dung o noi khac (PHAI khop ext_bench/gen_data)
BBH_SKIP = gd.TEST_BBH_PER + 100        # 7 (test cu) + 100 (train)
MUSR_SKIP = gd.TEST_MUSR_PER + 166      # 66 + 166
GSM8K_SKIP = 200                        # test cu lay 200 dau


# ------------------------------------------------------------------ gen ----

def build_items(n_bbh, n_gsm8k, n_musr):
    items = []
    per = max(1, n_bbh // len(eb.BBH_TASKS))
    for task in eb.BBH_TASKS:
        try:
            df = eb._parquet("lukaemon/bbh", f"{task}/test/0000.parquet")
        except Exception as e:
            print(f"bbh bo qua {task}: {type(e).__name__}")
            continue
        for i, row in df.iloc[BBH_SKIP:BBH_SKIP + per].iterrows():
            items.append({"bench": "bbh", "kind": "bbh", "sub": task,
                          "id": f"big/{task}/{i}",
                          "prompt": (f"{row['input']}\n\nAnswer concisely.\n"
                                     f"<think>\n\n</think>\n\nAnswer: "),
                          "expect": str(row["target"]).strip()})
    df = eb._parquet("openai/gsm8k", "main/test/0000.parquet")
    for i, row in df.iloc[GSM8K_SKIP:GSM8K_SKIP + n_gsm8k].iterrows():
        gold = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        items.append({"bench": "gsm8k", "kind": "gsm8k", "sub": "main",
                      "id": f"big/gsm8k/{i}",
                      "prompt": (f"Solve step by step, then give the final "
                                 f"numeric answer after 'Final Answer: '.\n\n"
                                 f"Problem: {row['question']}\n\n"
                                 f"<think>\n\n</think>\n\nSolution: "),
                      "expect": gold})
    import ast
    from datasets import load_dataset
    per_m = max(1, n_musr // 3)
    for split in ("murder_mysteries", "object_placements", "team_allocation"):
        ds = load_dataset("TAUR-Lab/MuSR", split=split)
        for i, ex in enumerate(ds):
            if i < MUSR_SKIP or i >= MUSR_SKIP + per_m:
                continue
            try:
                ch = json.loads(ex["choices"])
            except json.JSONDecodeError:
                ch = ast.literal_eval(ex["choices"])
            L = [chr(65 + j) for j in range(len(ch))]
            opts = "\n".join(f"{l}) {c}" for l, c in zip(L, ch))
            items.append({"bench": "musr", "kind": "musr", "sub": split,
                          "id": f"big/{split}/{i}",
                          "prompt": (f"{ex['narrative']}\n\n{ex['question']}\n"
                                     f"{opts}\n\nAnswer with the letter of the "
                                     f"correct choice.\n<think>\n\n</think>\n\n"
                                     f"Answer: "),
                          "expect": L[ex["answer_index"]]})
    return items


def check_overlap(items, others):
    """Doi chieu CHUOI PROMPT — khong tin suy luan chi so."""
    seen = set()
    for f in others:
        if not Path(f).exists():
            print(f"  CANH BAO: khong thay {f}")
            continue
        d = json.loads(Path(f).read_text())
        pool = (d["train"] + d["val"]) if isinstance(d, dict) else d
        seen |= {x["prompt"] for x in pool}
    bad = [it["id"] for it in items if it["prompt"] in seen]
    print(f"kiem ro ri: {len(bad)}/{len(items)} trung {len(seen)} mau da dung")
    assert not bad, f"RO RI: {bad[:5]}"


# ----------------------------------------------------------------- self ----

def run_self(args):
    from vllm import LLM, SamplingParams
    items = json.loads(Path(ITEMS_F).read_text())
    llm = LLM(model=args.tgt_model, quantization="bitsandbytes",
              max_model_len=args.max_len, gpu_memory_utilization=0.90)
    by = defaultdict(list)
    for it in items:
        by[it["bench"]].append(it)
    out, t0 = {}, time.time()
    for bench, grp in by.items():
        sp = SamplingParams(temperature=0.0, max_tokens=N_NEW[bench])
        res = llm.generate([g["prompt"] for g in grp], sp)
        hit = 0
        for g, r in zip(grp, res):
            h = gd.score_item(g, r.outputs[0].text)
            out[g["id"]] = h
            hit += h
        print(f"self {bench}: {hit}/{len(grp)} = {100*hit/len(grp):.1f}%",
              flush=True)
    json.dump(out, open(OUT_DIR / "evalbig_self.json", "w"))
    print(f"self xong {(time.time()-t0)/60:.1f} phut -> evalbig_self.json")


# --------------------------------------------------------------- mapped ----

def run_mapped(args):
    import torch
    from transformers import AutoConfig
    from peft import PeftModel
    e5 = _load("e5_train")
    items = json.loads(Path(ITEMS_F).read_text())
    a, b = ((int(x) for x in args.slice.split(":")) if args.slice
            else (0, len(items)))
    items = items[a:b]
    spill = Path("/content/big_spill")
    spill.mkdir(parents=True, exist_ok=True)

    # PHA 1: 4B mot minh -> cache ra dia (bo cuc hai pha, xem e5_train)
    tok_s, model_s = e5.load_4bit(args.src_model)
    if args.lora:
        model_s = PeftModel.from_pretrained(model_s, args.lora)
        model_s = model_s.merge_and_unload()
        model_s.eval()
        print(f"da nap+merge LoRA 4B: {args.lora}", flush=True)
    tok_s.truncation_side = "left"
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    with torch.no_grad():
        pr = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                     use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(pr)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del pr
    for i, it in enumerate(items):
        pth = spill / f"x{i}.pt"
        if pth.exists():
            continue
        ids = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_s, ids[:, :-WARM_P])
        e5.spill_cache(past, pth)
        del past
        torch.cuda.empty_cache()
        if i % 100 == 0:
            print(f"pha A {i}/{len(items)}", flush=True)
    del model_s
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("PHA A XONG", flush=True)

    # PHA 2: 9B mot minh -> map + decode
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"mapper {args.mapper} | token dung {sorted(STOPS)}", flush=True)

    out, t0 = {}, time.time()
    for i, it in enumerate(items):
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src = e5.load_cache(spill / f"x{i}.pt")
        with torch.no_grad():
            tpl = e5.prefill_chunked(model_t, cut)
            past = e5.build_student_past(tpl, src, mapper)
            del tpl, src
            o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen = [int(inp)]
            for _ in range(N_NEW[it["bench"]] - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen.append(int(inp))
                if int(inp) in STOPS:
                    break
        txt = tok_t.decode(gen, skip_special_tokens=True)
        out[it["id"]] = {"hit": gd.score_item(it, txt), "txt": txt[:400],
                         "n_tok": len(gen)}
        del past, cur, o
        torch.cuda.empty_cache()
        if i % 25 == 24:
            h = sum(v["hit"] for v in out.values())
            el = (time.time() - t0) / 60
            print(f"mapped {i+1}/{len(items)} dung {h} ({100*h/len(out):.1f}%) "
                  f"| {el:.0f} phut | con ~{el/(i+1)*(len(items)-i-1):.0f} phut",
                  flush=True)
            json.dump(out, open(OUT_DIR / "evalbig_mapped.json", "w"))
    json.dump(out, open(OUT_DIR / "evalbig_mapped.json", "w"))
    print("MAPPED XONG", flush=True)


# ------------------------------------------------------------------ agg ----

def run_agg():
    items = {it["id"]: it for it in json.loads(Path(ITEMS_F).read_text())}
    slf = json.loads((OUT_DIR / "evalbig_self.json").read_text())
    mpd = json.loads((OUT_DIR / "evalbig_mapped.json").read_text())
    st = defaultdict(lambda: {"n": 0, "self": 0, "mapped": 0})
    for i, m in mpd.items():
        b = items[i]["bench"]
        st[b]["n"] += 1
        st[b]["self"] += slf.get(i, 0)
        st[b]["mapped"] += m["hit"]
    print(f"\n{'bo':8} {'n':>6} {'self':>8} {'mapped':>8} {'giu duoc':>10}")
    tn = ts = tm = 0
    for b, v in sorted(st.items()):
        r = 100 * v["mapped"] / max(v["self"], 1)
        print(f"{b:8} {v['n']:6} {100*v['self']/v['n']:7.1f}% "
              f"{100*v['mapped']/v['n']:7.1f}% {r:9.1f}%")
        tn += v["n"]; ts += v["self"]; tm += v["mapped"]
    print(f"{'TONG':8} {tn:6} {100*ts/tn:7.1f}% {100*tm/tn:7.1f}% "
          f"{100*tm/max(ts,1):9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "self", "mapped", "agg"])
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", default="")
    ap.add_argument("--lora", default="")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--slice", default="")
    ap.add_argument("--n-bbh", type=int, default=1300)
    ap.add_argument("--n-gsm8k", type=int, default=500)
    ap.add_argument("--n-musr", type=int, default=60)
    args = ap.parse_args()

    if args.mode == "gen":
        items = build_items(args.n_bbh, args.n_gsm8k, args.n_musr)
        check_overlap(items, ["/content/train_items.json",
                              "/content/ext_bench_items.json"])
        json.dump(items, open(ITEMS_F, "w"))
        print(f"da ghi {len(items)} item -> {ITEMS_F}")
        print(" ", dict(Counter(i["bench"] for i in items)))
    elif args.mode == "self":
        run_self(args)
    elif args.mode == "mapped":
        run_mapped(args)
    else:
        run_agg()
    print("EVALBIG_EXIT", flush=True)


if __name__ == "__main__":   # BAT BUOC: LLM() o module level chet vi spawn
    main()
