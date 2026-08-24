"""E6 v3 baselines — do de bao dam cham dung (user 2026-08-24).

Bang diem test niem phong con thieu 3 cot de ket luan sach:
  1. 4B SELF tren cung 30 mau test (neu 4B mot minh lam duoc thi cascade
     phai thang no moi co y nghia kinh doanh)
  2. 27B doc mapper CHUA-TRAIN (identity-init) — diem xuat phat cua phep dich
     (test hom truoc dung best=step149 gan-non, chua phai init that)
  3. needle2k cham dung: enc khong duoc cat 1024 (TRAIN_MAX lam mat cau hoi
     -> self cung 0 — artifact da ghi nhan); do lai voi max_len 4096 cho ca
     self / mapped(best) / mapped(init) / no_ctx.

Dung lai data.json + cache test bfcl (1024, khop diem mapped 4/20 da do);
needle2k spill cache moi full-length. Ket qua -> e6v3_baselines.json.
"""

import argparse
import gc
import importlib.util
import json
import re
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e6v3_ce", Path(__file__).parent / "e6v3_ce.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
e5 = v3.e5

MAXLEN_NK = 4096
GEN = 24


def grade(it, txt):
    if it["kind"] == "bfcl":
        return int(it["fn"] in txt)
    return int(it["code"] in re.sub(r"\D", "", txt))


def greedy(model, past, inp, tok):
    import torch
    cur, gen = past, []
    with torch.no_grad():
        for _ in range(GEN):
            o = model(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
    del cur
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--mapper-best", default="/content/mapper_v3.pt")
    ap.add_argument("--cache-dir", default="/content/v3_src")
    ap.add_argument("--results", default="/content/logs/e6v3_baselines.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig

    cdir = Path(args.cache_dir)
    data = json.loads((cdir / "data.json").read_text())
    test = data["test"]
    res = {}

    def maxlen(it):
        return MAXLEN_NK if it["kind"] == "needle" else 1024

    # ---- PHA 1: 4B — self tren 30 mau + spill needle2k full-length ----
    tok, model_s = e5.load_4bit(args.src_model)
    hits = {"bfcl": [], "needle": []}
    with torch.no_grad():
        for i, it in enumerate(test):
            enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=maxlen(it)).to("cuda")
            pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
            past = model_s(input_ids=pre, use_cache=True,
                           logits_to_keep=1).past_key_values
            if it["kind"] == "needle":
                e5.spill_cache(past, cdir / f"nk4096_{i}.pt")
            txt = greedy(model_s, past, last, tok)
            hits["bfcl" if it["kind"] == "bfcl" else "needle"].append(
                grade(it, txt))
            del past
            torch.cuda.empty_cache()
            print(f"4B-self {i+1}/{len(test)}")
    res["4B_self"] = {k: f"{sum(v)}/{len(v)}" for k, v in hits.items() if v}
    print("4B_SELF:", res["4B_self"])
    del model_s
    gc.collect(); torch.cuda.empty_cache()

    # ---- PHA 2: 27B — mapped(init) bfcl + needle2k 4 dieu kien ----
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    src0 = e5.load_cache(cdir / "train0.pt")
    a_s, g_s = e5.split_layers(src0)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    attn_dim = (next(iter(a_s.values())).keys.shape[1]
                * next(iter(a_s.values())).keys.shape[3])
    del src0
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper_init = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim,
                            theta_s, theta_t)
    mapper_best = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim,
                            theta_s, theta_t)
    mapper_best.load(args.mapper_best)

    def eval_cond(items_idx, cache_name, mapper, tag):
        hits = {"bfcl": [], "needle": []}
        with torch.no_grad():
            for i in items_idx:
                it = test[i]
                enc = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                            max_length=maxlen(it)).to("cuda")
                pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
                if mapper is None:
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                else:
                    src = e5.load_cache(cdir / cache_name(i))
                    tpl = model_t(input_ids=pre, use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = e5.build_student_past(tpl, src, mapper)
                    del src, tpl
                txt = greedy(model_t, past, last, tok_t)
                hits["bfcl" if it["kind"] == "bfcl" else "needle"].append(
                    grade(it, txt))
                del past
                torch.cuda.empty_cache()
        out = {k: f"{sum(v)}/{len(v)}" for k, v in hits.items() if v}
        print(tag, out)
        return out

    bfcl_idx = [i for i, it in enumerate(test) if it["kind"] == "bfcl"]
    nk_idx = [i for i, it in enumerate(test) if it["kind"] == "needle"]

    res["27B_mapped_INIT_bfcl"] = eval_cond(
        bfcl_idx, lambda i: f"test{i}.pt", mapper_init, "MAPPED-INIT bfcl")
    res["27B_self_needle2k"] = eval_cond(nk_idx, None, None, "27B-SELF nk")
    res["27B_mapped_BEST_needle2k"] = eval_cond(
        nk_idx, lambda i: f"nk4096_{i}.pt", mapper_best, "MAPPED-BEST nk")
    res["27B_mapped_INIT_needle2k"] = eval_cond(
        nk_idx, lambda i: f"nk4096_{i}.pt", mapper_init, "MAPPED-INIT nk")

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    print("E6V3_BASELINES_DONE")


if __name__ == "__main__":
    main()
