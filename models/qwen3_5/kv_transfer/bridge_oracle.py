"""BRIDGE ORACLE — neu cham gsm8k qua cache mapped (ket qua ~0-8%), sau do cho
9B PREFILL THAT (khong qua mapper) mot doan ngan nhac lai chinh de bai, thi
diem co nhay len khong?

Neu co -> huong bridge tokens (4B tom tat cuc ngan, 9B tu prefill lai doan
do) dang thu that su — khong can sua mapper. Neu khong -> cache mapped da
"dau doc" qua trinh sinh theo cach ma mot doan prefill ngan khong sua duoc.

3 bien the moi mau (batch=1, khong train):
  B. mapped        — duong ong hien tai (nhu oracle_ablation.py)
  E. bridge_full    — B + prefill THAT lai TOAN BO doan "Problem: ..." truoc
                      khi sinh (oracle ran nhat: bridge = chinh de bai)
  F. bridge_nums    — B + prefill THAT chi cau chua cac con so (bridge NGAN
                      hon — gan voi "4B tom tat" thuc te se lam)

    python bridge_oracle.py --n 30
"""
import argparse
import copy
import json
import pathlib
import re
import sys

import torch

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_problem(prompt):
    """Tach doan 'Problem: ...' (truoc <think>) — chinh la phan mang thong
    tin, phan con lai chi la khuon cau lenh."""
    if "Problem:" not in prompt:
        return None
    body = prompt.split("Problem:", 1)[1].split("<think>")[0]
    return body.strip()


def extract_num_sentences(problem):
    """Chi giu cau CO SO — ngan hon nguyen de bai, gan voi 'ban tom tat cuc
    ngan do 4B xuat ra' ma huong bridge tokens thuc te se lam."""
    sents = re.split(r"(?<=[.?!])\s+", problem)
    keep = [s for s in sents if NUM.search(s)]
    return " ".join(keep) if keep else problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", default="/content/joint49bb/mapper_best.pt")
    ap.add_argument("--lora", default="/content/joint49bb/lora_best")
    ap.add_argument("--lora-t", default="/content/joint49bb/lorat_best")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gen-len", type=int, default=320)
    ap.add_argument("--out", default="/content/logs/bridge_oracle.json")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import e5_train as e5
    from transformers import AutoConfig
    from peft import PeftModel

    WARM_P = 5

    items = json.loads(pathlib.Path("/content/eval_big_items.json").read_text())
    gsm = [it for it in items if it["bench"] == "gsm8k"][:args.n]
    gsm = [it for it in gsm if extract_problem(it["prompt"])]
    print(f"dung {len(gsm)} mau gsm8k co tach duoc 'Problem:'", flush=True)

    # ---- 4B: prefill toan bo, spill cache ----
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

    for it in gsm:
        ids = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_s, ids[:, :-WARM_P])
        pth = pathlib.Path(f"/content/_bridge_4b_{it['id'].replace('/', '_')}.pt")
        e5.spill_cache(past, pth)
        del past
        torch.cuda.empty_cache()
    print("4B prefill xong", flush=True)
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
    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"mapper nap xong (gdn_terms={_meta.get('gdn_terms', 1)})", flush=True)

    # ---- template-XUONG (nhu eval_big.py: KHONG can 9B prefill that de lay
    # hinh dang — dung dung chi phi cua duong ong san xuat that) ----
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
            if int(inp) in STOPS:
                break
        del cur, o
        return tok_t.decode(gen, skip_special_tokens=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_big", pathlib.Path(__file__).parent / "eval_big.py")
    ebmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ebmod)

    results = {"mapped": {}, "bridge_full": {}, "bridge_nums": {}}
    texts = {"mapped": {}, "bridge_full": {}, "bridge_nums": {}}
    bridge_lens = {"bridge_full": {}, "bridge_nums": {}}

    for i, it in enumerate(gsm):
      with torch.no_grad():
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src_4b = e5.load_cache(
            pathlib.Path(f"/content/_bridge_4b_{it['id'].replace('/', '_')}.pt"))
        problem = extract_problem(it["prompt"])

        # B: mapped (khong bridge)
        p = build_mapped(cut.shape[1], src_4b)
        txt = greedy(p, warm.clone())
        results["mapped"][it["id"]] = ebmod.score(it, txt); texts["mapped"][it["id"]] = txt[:200]
        del p

        # E: mapped + bridge = TOAN BO de bai, PREFILL THAT (khong qua mapper)
        bridge_text_full = "\n\nRecap the problem: " + problem
        bridge_ids_full = tok_t(bridge_text_full, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to("cuda")
        p = build_mapped(cut.shape[1], src_4b)
        o = model_t(input_ids=bridge_ids_full, past_key_values=p, use_cache=True)
        txt = greedy(o.past_key_values, warm.clone())
        results["bridge_full"][it["id"]] = ebmod.score(it, txt); texts["bridge_full"][it["id"]] = txt[:200]
        bridge_lens["bridge_full"][it["id"]] = int(bridge_ids_full.shape[1])
        del p, o

        # F: mapped + bridge = CHI cau co so, PREFILL THAT (ngan hon E)
        nums_only = extract_num_sentences(problem)
        bridge_text_nums = "\n\nKey facts: " + nums_only
        bridge_ids_nums = tok_t(bridge_text_nums, return_tensors="pt",
                                add_special_tokens=False)["input_ids"].to("cuda")
        p = build_mapped(cut.shape[1], src_4b)
        o = model_t(input_ids=bridge_ids_nums, past_key_values=p, use_cache=True)
        txt = greedy(o.past_key_values, warm.clone())
        results["bridge_nums"][it["id"]] = ebmod.score(it, txt); texts["bridge_nums"][it["id"]] = txt[:200]
        bridge_lens["bridge_nums"][it["id"]] = int(bridge_ids_nums.shape[1])
        del p, o
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(gsm)} xong", flush=True)

    print(f"\n{'bien the':14} {'dung':>6} {'n':>4} {'ty le':>8} {'do dai bridge tb':>18}")
    for k in ("mapped", "bridge_full", "bridge_nums"):
        h = sum(results[k].values())
        n = len(results[k])
        bl = bridge_lens.get(k)
        avg = f"{sum(bl.values())/len(bl):.0f} token" if bl else "0 (khong co)"
        print(f"{k:14} {h:6} {n:4} {100*h/n:7.1f}% {avg:>18}")

    print("\nDOC:")
    print("  bridge_full/bridge_nums >> mapped -> huong bridge tokens dung,")
    print("    theo duoi (4B tu tom tat, khong can prefill lai NGUYEN de bai)")
    print("  bridge_full ~ mapped -> cache mapped da 'dau doc' theo cach ma")
    print("    mot doan prefill ngan them vao KHONG sua duoc — can huong khac")
    print("  bridge_nums ~ bridge_full -> khong can bridge dai, ban tom tat")
    print("    NGAN (gan voi thuc te 4B se sinh) van du")

    out = {"results": results, "texts": texts, "bridge_lens": bridge_lens,
          "n": len(gsm)}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nda ghi {args.out}")


if __name__ == "__main__":
    main()
