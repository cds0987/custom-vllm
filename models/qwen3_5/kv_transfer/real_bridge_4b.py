"""PIPELINE THAT — bridge do CHINH 4B tu sinh (khong trich tu de goc nhu
bridge_oracle.py). Tai dung cache 4B da prefill (khong prefill lai): noi them
mot cau lenh yeu cau liet ke con so, de 4B GREEDY sinh ban tom tat ngan; ban
do duoc PREFILL THAT (khong qua mapper) truoc khi 9B sinh cau tra loi.

2 bien the moi mau:
  B. mapped       — duong ong hien tai (doi chung)
  G. bridge_4b    — mapped + bridge do 4B TU SINH (khong phai oracle)

So voi bridge_oracle.py (n=30): mapped 0%, bridge_full 23,3%, bridge_nums
16,7%. Neu bridge_4b gan bridge_nums -> 4B tu tom tat DU TOT, dung duoc that.
Neu bridge_4b gan mapped -> 4B khong tom tat noi, huong nay can sua cach
gen bridge (vd few-shot, hoac train rieng 4B cho viec nay).

    python real_bridge_4b.py --n 30
"""
import argparse
import json
import pathlib
import sys

import torch

INSTR = ("\n\nList only the key numbers mentioned above and what each one "
         "refers to, one short line each like 'X = 3'. Do not solve the "
         "problem or compute anything.\n\nKey numbers:\n")


def extract_problem(prompt):
    if "Problem:" not in prompt:
        return None
    return prompt.split("Problem:", 1)[1].split("<think>")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", default="/content/joint49bb/mapper_best.pt")
    ap.add_argument("--lora", default="/content/joint49bb/lora_best")
    ap.add_argument("--lora-t", default="/content/joint49bb/lorat_best")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bridge-gen-len", type=int, default=70)
    ap.add_argument("--gen-len", type=int, default=320)
    ap.add_argument("--out", default="/content/logs/real_bridge_4b.json")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import e5_train as e5
    from transformers import AutoConfig
    from peft import PeftModel

    WARM_P = 5

    items = json.loads(pathlib.Path("/content/eval_big_items.json").read_text())
    gsm = [it for it in items if it["bench"] == "gsm8k"][:args.n]
    gsm = [it for it in gsm if extract_problem(it["prompt"])]
    print(f"dung {len(gsm)} mau gsm8k", flush=True)

    # ---- 4B: prefill THAT, roi TIEP TUC sinh ban tom tat (tai dung cache) ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    if args.lora:
        model_s = PeftModel.from_pretrained(model_s, args.lora)
        model_s = model_s.merge_and_unload()
        model_s.eval()
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
    STOPS_S = e5.stop_ids(tok_s, model_s)

    bridge_texts = {}
    for it in gsm:
        ids = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_s, ids[:, :-WARM_P])
        pth = pathlib.Path(f"/content/_bridge_4b_{it['id'].replace('/', '_')}.pt")
        e5.spill_cache(past, pth)  # dung lai duoc cho cac lan chay khac

        # tiep tuc sinh: dung WARM_P cuoi cua prompt goc + cau lenh, roi
        # GREEDY bang chinh 4B (khong phai 9B) — day la buoc MOI so voi
        # bridge_oracle.py (o do bridge lay thang tu de bai, khong sinh).
        with torch.no_grad():
            warm_s = ids[:, -WARM_P:]
            o = model_s(input_ids=warm_s, past_key_values=past, use_cache=True)
            cur = o.past_key_values
            instr_ids = tok_s(INSTR, return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to("cuda")
            o = model_s(input_ids=instr_ids, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen = [int(inp)]
            for _ in range(args.bridge_gen_len - 1):
                o = model_s(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen.append(int(inp))
                if int(inp) in STOPS_S:
                    break
            bridge_texts[it["id"]] = tok_s.decode(gen, skip_special_tokens=True)
        del past, cur, o
        torch.cuda.empty_cache()
    print("4B prefill + tu sinh bridge xong", flush=True)
    del model_s
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 9B ----
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    if args.lora_t:
        model_t = PeftModel.from_pretrained(model_t, args.lora_t)
        model_t.eval()
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    _meta = torch.load(args.mapper, map_location="cpu").get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    mapper.load(args.mapper)
    STOPS_T = e5.stop_ids(tok_t, model_t)
    print(f"mapper nap xong (gdn_terms={_meta.get('gdn_terms', 1)})", flush=True)

    import copy as _copy
    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _p = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_p)
    del _p, _ids
    torch.cuda.empty_cache()

    def meta_for_len(t):
        m = _copy.deepcopy(base_meta)
        m["cache_ints"] = {k: (t if v == T_BASE else v)
                           for k, v in m["cache_ints"].items()}
        for lay in m["layers"]:
            lay["ints"] = {k: (t if v == T_BASE else v)
                           for k, v in lay["ints"].items()}
            if lay["kind"] == "a":
                for key in ("k", "v"):
                    sh, dt = lay[key]
                    lay[key] = (tuple(t if d == T_BASE else d for d in sh), dt)
        return m

    def build_mapped(cut_len, src_4b):
        tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut_len))
        return e5.build_student_past(tpl, src_4b, mapper)

    @torch.no_grad()
    def greedy(past, warm):
        o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
        cur = o.past_key_values
        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [int(inp)]
        for _ in range(args.gen_len - 1):
            o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
            if int(inp) in STOPS_T:
                break
        del cur, o
        return tok_t.decode(gen, skip_special_tokens=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_big", pathlib.Path(__file__).parent / "eval_big.py")
    ebmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ebmod)

    results = {"mapped": {}, "bridge_4b": {}}
    texts = {"mapped": {}, "bridge_4b": {}}
    bridge_lens = {}

    for i, it in enumerate(gsm):
      with torch.no_grad():
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src_4b = e5.load_cache(
            pathlib.Path(f"/content/_bridge_4b_{it['id'].replace('/', '_')}.pt"))

        # B: mapped
        p = build_mapped(cut.shape[1], src_4b)
        txt = greedy(p, warm.clone())
        results["mapped"][it["id"]] = ebmod.score(it, txt); texts["mapped"][it["id"]] = txt[:200]
        del p

        # G: mapped + bridge do CHINH 4B tu sinh, PREFILL THAT boi 9B
        bt = "\n\nKey facts: " + bridge_texts[it["id"]]
        bridge_ids = tok_t(bt, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to("cuda")
        p = build_mapped(cut.shape[1], src_4b)
        o = model_t(input_ids=bridge_ids, past_key_values=p, use_cache=True)
        txt = greedy(o.past_key_values, warm.clone())
        results["bridge_4b"][it["id"]] = ebmod.score(it, txt); texts["bridge_4b"][it["id"]] = txt[:200]
        bridge_lens[it["id"]] = int(bridge_ids.shape[1])
        del p, o
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(gsm)} xong", flush=True)

    print(f"\n{'bien the':12} {'dung':>6} {'n':>4} {'ty le':>8}")
    for k in ("mapped", "bridge_4b"):
        h = sum(results[k].values())
        n = len(results[k])
        print(f"{k:12} {h:6} {n:4} {100*h/n:7.1f}%")
    avg = sum(bridge_lens.values()) / len(bridge_lens)
    print(f"do dai bridge 4B tu sinh trung binh: {avg:.0f} token")

    print("\nDOC (doi chieu bridge_oracle.py n=30: mapped 0%, bridge_full")
    print("23,3%, bridge_nums 16,7%):")
    print("  bridge_4b gan bridge_nums -> 4B tu tom tat DU TOT, dung duoc that")
    print("  bridge_4b gan mapped -> 4B chua tom tat noi, can sua cach gen")

    out = {"results": results, "texts": texts, "bridge_texts": bridge_texts,
          "bridge_lens": bridge_lens, "n": len(gsm)}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nda ghi {args.out}")

    import os
    from huggingface_hub import HfApi
    try:
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=args.out, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
            path_in_repo="evalbig/" + pathlib.Path(args.out).name)
        print("HF-UP", pathlib.Path(args.out).name)
    except Exception as e:
        print("HF-UP FAIL", type(e).__name__, str(e)[:100])


if __name__ == "__main__":
    main()
