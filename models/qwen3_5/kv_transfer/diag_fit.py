"""diag_fit -- mapper HOC DUOC tap train den dau? (quá khớp hay thiếu dung lượng)

User 2026-08-30: "them data va steps ko len co the do mapper qua nho".

Gia thuyet do PHAI duoc tach khoi mot gia thuyet khac truoc khi dot vai gio
GPU de phong to mapper:

  train ~95% / val ~60%  -> QUA KHOP. Mapper du suc hoc; phong to lam TE HON.
                            Viec can lam la da dang hoa du lieu.
  train ~60% / val ~60%  -> THIEU DUNG LUONG hoac SAI DANG HAM. Phong to dung.
  train ~60% nhung CE~0  -> SAI DANG HAM: khop duoc token ma khong tai tao
                            duoc chuc nang. (Luat error-placement cua du an:
                            R2/NLL/nMSE khong du doan chuc nang — da kiem 4 lan.)

Dau hieu hien co chua ket luan duoc: CE train LUONG CUC — nhieu buoc 0,0002
(khop gan hoan hao) xen ke buoc 1,2-2,0 (hong han). Thieu dung luong thuan thi
CE phai ket cao DEU; qua khop thuan thi phai thap DEU.

Cham diem BANG DUNG bo grader cua val (e9_joint._grade / gen_data.score_item)
va DUNG giao thuc WARM_P=5 — neu lech giao thuc thi hai cot khong so duoc.

Chay:
  python -u diag_fit.py --mapper /content/joint49w/mapper_best.pt \\
      --lora /content/joint49w/lora_best --n 150
"""

import argparse
import importlib.util
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
gd = _load("gen_data")

WARM_P = 5
NEW_KINDS = {"gsm8k", "bbh", "musr", "suite_rag", "suite_mid",
             "suite_math", "suite_swe"}
GEN_LEN = {"bfcl": 24, "needle": 16, "ifstruct": 160, "pbtable": 120,
           "gsm8k": 320, "bbh": 48, "musr": 24, "suite_rag": 24,
           "suite_mid": 24, "suite_math": 24, "suite_swe": 24}


def grade(it, txt):
    """Y HET e9_joint._grade — hai cot chi so duoc khi cung grader."""
    if it["kind"] in NEW_KINDS:
        return int(gd.score_item(it, txt))
    if it["kind"] == "bfcl":
        return int(it["fn"] in txt)
    if it["kind"] == "needle":
        return int(it["code"] in re.sub(r"\D", "", txt))
    return int(re.sub(r"\s+", " ", it["gold"])[:30]
               in re.sub(r"\s+", " ", txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--drop-kinds", default="gsm8k,suite_math")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out", default="/content/logs/diag_fit.json")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="diag_fit")
    args = ap.parse_args()

    import os
    from transformers import AutoConfig

    # ---- du lieu: DUNG pool ma train da dung (e6v3 + gen_data) ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    tok_s.truncation_side = "left"      # cau hoi nam O CUOI prompt
    e6 = _load("e6v3_ce")
    data = e6.build_data(tok_s, max_ctx=min(args.max_ctx, 2000))
    extra = json.loads(Path(args.data_file).read_text())
    data["train"] += extra["train"]
    data["val"] += extra["val"]
    for k in ("train", "val"):
        data[k] = [it for it in data[k] if it.get("gold")]
    if args.drop_kinds:
        drop = set(args.drop_kinds.split(","))
        for k in ("train", "val"):
            data[k] = [it for it in data[k] if it["kind"] not in drop]
    random.Random(7).shuffle(data["train"])
    random.Random(7).shuffle(data["val"])
    tr = data["train"][:args.n]
    va = data["val"][:args.n]
    print(f"train {len(tr)} mau | val {len(va)} mau", flush=True)

    if args.lora:
        from peft import PeftModel
        model_s = PeftModel.from_pretrained(model_s, args.lora)
        model_s = model_s.merge_and_unload()
        model_s.eval()
        print(f"da merge LoRA: {args.lora}", flush=True)

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

    tok_t, model_t = e5.load_4bit(args.tgt_model)
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    meta = torch.load(args.mapper, map_location="cpu").get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=meta.get("attn_rank", 0),
                       gdn_per_head=meta.get("gdn_per_head", False),
                       gdn_terms=meta.get("gdn_terms", 1))
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)

    @torch.no_grad()
    def run(it):
        ml = min(args.max_ctx, 4096)
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=ml)["input_ids"].to("cuda")
        if ids.shape[1] <= WARM_P:
            return None
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        ids_s = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                      max_length=ml)["input_ids"].to("cuda")
        src = e5.prefill_chunked(model_s, ids_s[:, :-WARM_P])
        tpl = e5.prefill_chunked(model_t, cut)
        past = e5.build_student_past(tpl, src, mapper)
        del tpl, src
        o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
        cur = o.past_key_values
        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [int(inp)]
        for _ in range(GEN_LEN.get(it["kind"], 24) - 1):
            o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
            if int(inp) in STOPS:
                break
        del past, cur, o
        torch.cuda.empty_cache()
        return tok_t.decode(gen, skip_special_tokens=True)

    res = {}
    for split, items in (("train", tr), ("val", va)):
        per = defaultdict(lambda: [0, 0])
        for i, it in enumerate(items):
            txt = run(it)
            if txt is None:
                continue
            h = grade(it, txt)
            per[it["kind"]][0] += 1
            per[it["kind"]][1] += h
            if i % 25 == 24:
                n = sum(v[0] for v in per.values())
                hh = sum(v[1] for v in per.values())
                print(f"  {split} {i+1}/{len(items)}: {hh}/{n}", flush=True)
        res[split] = {k: v for k, v in per.items()}
        json.dump(res, open(args.out, "w"))

    print("\n===== MAPPER TREN TRAIN vs VAL =====")
    print(f"{'ho':12}{'train':>14}{'val':>14}")
    tot = {"train": [0, 0], "val": [0, 0]}
    for k in sorted(set(res["train"]) | set(res["val"])):
        row = ""
        for sp in ("train", "val"):
            n, h = res[sp].get(k, [0, 0])
            tot[sp][0] += n
            tot[sp][1] += h
            row += f"{(f'{h}/{n}' if n else '-'):>14}"
        print(f"{k:12}{row}")
    a = 100 * tot["train"][1] / max(tot["train"][0], 1)
    b = 100 * tot["val"][1] / max(tot["val"][0], 1)
    s_tr = f"{tot['train'][1]}/{tot['train'][0]}"
    s_va = f"{tot['val'][1]}/{tot['val'][0]}"
    print(f"{'TONG':12}{s_tr:>14}{s_va:>14}")
    print(f"{'':12}{a:13.1f}%{b:13.1f}%   chenh {a-b:+.1f} diem")
    print("\nDOC KET QUA:")
    print("  chenh LON  (train >> val) -> QUA KHOP: phong to mapper lam te hon,")
    print("                                viec can lam la da dang hoa du lieu.")
    print("  chenh NHO  (train ~ val)  -> THIEU DUNG LUONG hoac SAI DANG HAM:")
    print("                                phong to mapper la huong dung.")
    if args.hf_repo and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=args.out, repo_id=args.hf_repo,
                path_in_repo=f"{args.hf_prefix}/{Path(args.out).name}")
            print("HF-UP", args.out)
        except Exception as ex:
            print("HF-UP FAIL", type(ex).__name__, str(ex)[:80])
    print("DIAG_FIT_EXIT", flush=True)


if __name__ == "__main__":
    main()
