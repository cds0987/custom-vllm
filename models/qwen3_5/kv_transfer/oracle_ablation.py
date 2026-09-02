"""ORACLE ABLATION — cau hoi: GDN co THAT SU la nut that cua gsm8k khong?

Thay vi suy luan gian tiep (E7, probe trich-xuat-so, gdn-terms), o day HOAN
DOI TRUC TIEP tung nua cache bang cache 9B THAT (tu chinh 9B tu prefill cung
prompt), giu nua con lai la ban do tu mapper (duong ong joint49bb hien tai).

4 bien the moi mau (batch=1, khong train):
  A. self       — 9B tu prefill toan bo (tran nang luc that, doi chieu 89,0%)
  B. mapped     — duong ong hien tai: ca attn lan GDN deu qua mapper (~8%)
  C. attn_that  — attn LAY THAT tu 9B, GDN van qua mapper
  D. gdn_that   — attn qua mapper, GDN LAY THAT tu 9B

Doc: neu C ~ B (thay attn that khong cuu duoc) va D ~ A (thay GDN that CUU
DUOC gan het) -> xac nhan GDN la nut that that. Neu ca hai deu khong cuu
duoc -> ca attn lan GDN mapped deu co van de, hoac loi nam o cho khac (vi du
chinh 9B/LoRA-9B khong doc duoc chuoi nhieu-thuc-the du cache dung).

    python oracle_ablation.py --n 30
"""
import argparse
import json
import pathlib
import sys

import torch


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
    ap.add_argument("--out", default="/content/logs/oracle_ablation.json")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import e5_train as e5
    from transformers import AutoConfig
    from peft import PeftModel

    WARM_P = 5

    items = json.loads(pathlib.Path("/content/eval_big_items.json").read_text())
    gsm = [it for it in items if it["bench"] == "gsm8k"][:args.n]
    print(f"dung {len(gsm)} mau gsm8k niem phong", flush=True)

    # ---- 4B: prefill toan bo, giu cache tren RAM (n nho, khong can spill) ----
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
        pth = pathlib.Path(f"/content/_oracle_4b_{it['id'].replace('/', '_')}.pt")
        e5.spill_cache(past, pth)
        del past
        torch.cuda.empty_cache()
    print("4B prefill xong", flush=True)
    del model_s
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 9B: nap + LoRA-9B (khong merge, giong duong ong that) ----
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    if args.lora_t:
        model_t = PeftModel.from_pretrained(model_t, args.lora_t)
        model_t.eval()
    tok_t.truncation_side = "left"
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    _meta = torch.load(args.mapper, map_location="cpu").get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s,
                       e5.e1.get_rope_theta(model_t.config.get_text_config()),
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"mapper nap xong (gdn_terms={_meta.get('gdn_terms', 1)}) | "
          f"stop tokens {sorted(STOPS)}", flush=True)

    def deep_clone_cache(past):
        """clone_cache_struct CHI tao container moi, tensor BEN TRONG van
        dung chung storage voi ban goc. GDN cap nhat bang .copy_() TAI CHO khi
        suy luan (khong grad -> patch_recurrent_rebind khong kich hoat, xem
        e5_train.patch_recurrent_rebind) -> decode MOT ban se lam hong CA
        NHUNG ban con lai dung chung tensor. Phai clone() tung tensor that."""
        new = e5.clone_cache_struct(past)
        attn_n, gdn_n = e5.split_layers(new)   # cung cach xac dinh loai lop
        for l in attn_n.values():              # nhu moi noi khac trong repo
            l.keys = l.keys.clone()
            l.values = l.values.clone()
        for l in gdn_n.values():
            r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
            e5._set_like(l, "recurrent_states", r.clone())
            e5._set_like(l, "conv_states", c.clone())
        return new

    def build_hybrid(real_past, src_4b, real_attn, real_gdn):
        """Nhu build_student_past nhung chon TUNG NUA lay tu 9B THAT hay mapper.
        deep_clone_cache dam bao khong con tensor nao dung chung storage voi
        real_past hay cac bien the khac cung mau."""
        past = deep_clone_cache(real_past)
        attn_s, gdn_s = e5.split_layers(src_4b)
        attn_t, gdn_t = e5.split_layers(past)
        ks, kt = sorted(attn_s), sorted(attn_t)
        amap = e5.depth_map(len(ks), len(kt))
        for j, it in enumerate(kt):
            if real_attn:
                continue  # da la tensor rieng (deep_clone_cache), giu nguyen
            src = attn_s[ks[amap[j]]]
            mk, mv = mapper.map_attn(j, src.keys, src.values)
            attn_t[it].keys = mk
            attn_t[it].values = mv
        gs, gt = sorted(gdn_s), sorted(gdn_t)
        gmap = e5.depth_map(len(gs), len(gt))
        for j, it in enumerate(gt):
            if real_gdn:
                continue  # da la tensor rieng, giu nguyen trang thai that
            src = gdn_s[gs[gmap[j]]]
            e5._set_like(gdn_t[it], "recurrent_states",
                         mapper.map_gdn(j, e5._get(src.recurrent_states)))
            e5._set_like(gdn_t[it], "conv_states",
                         torch.zeros_like(e5._get(gdn_t[it].conv_states)))
        return past

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
    spec = importlib.util.spec_from_file_location("eval_big",
                                                    pathlib.Path(__file__).parent / "eval_big.py")
    ebmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ebmod)

    results = {"self": {}, "mapped": {}, "attn_that": {}, "gdn_that": {}}
    texts = {"self": {}, "mapped": {}, "attn_that": {}, "gdn_that": {}}

    for i, it in enumerate(gsm):
      with torch.no_grad():
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        real_past = e5.prefill_chunked(model_t, cut)
        src_4b = e5.load_cache(
            pathlib.Path(f"/content/_oracle_4b_{it['id'].replace('/', '_')}.pt"))

        # A: self (9B that toan bo). PHAI deep_clone_cache — decode mutate
        # tensor TAI CHO (.copy_() cua GDN), clone_cache_struct thuong CHI tao
        # container moi con tensor van chung storage voi real_past -> se lam
        # hong du lieu real_past truoc khi B/C/D con dung.
        txt = greedy(deep_clone_cache(real_past), warm.clone())
        results["self"][it["id"]] = ebmod.score(it, txt); texts["self"][it["id"]] = txt[:200]

        # B: mapped (duong ong hien tai)
        p = build_hybrid(real_past, src_4b, real_attn=False, real_gdn=False)
        txt = greedy(p, warm.clone())
        results["mapped"][it["id"]] = ebmod.score(it, txt); texts["mapped"][it["id"]] = txt[:200]
        del p

        # C: attn THAT, GDN qua mapper
        p = build_hybrid(real_past, src_4b, real_attn=True, real_gdn=False)
        txt = greedy(p, warm.clone())
        results["attn_that"][it["id"]] = ebmod.score(it, txt); texts["attn_that"][it["id"]] = txt[:200]
        del p

        # D: attn qua mapper, GDN THAT
        p = build_hybrid(real_past, src_4b, real_attn=False, real_gdn=True)
        txt = greedy(p, warm.clone())
        results["gdn_that"][it["id"]] = ebmod.score(it, txt); texts["gdn_that"][it["id"]] = txt[:200]
        del p, real_past
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(gsm)} xong", flush=True)

    print(f"\n{'bien the':14} {'dung':>6} {'n':>4} {'ty le':>8}")
    for k in ("self", "mapped", "attn_that", "gdn_that"):
        h = sum(results[k].values())
        n = len(results[k])
        print(f"{k:14} {h:6} {n:4} {100*h/n:7.1f}%")

    print("\nDOC:")
    print("  C(attn_that) ~ B(mapped) VA D(gdn_that) ~ A(self)")
    print("    -> XAC NHAN: GDN mapped la nut that that, attn mapped du tot")
    print("  Ca C lan D deu khong len gan A -> GDN khong phai nut that DUY NHAT")
    print("  C len ro rang ma D khong -> attn mapped moi la van de (bat ngo)")

    out = {"results": results, "texts": texts, "n": len(gsm)}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"\nda ghi {args.out}")


if __name__ == "__main__":
    main()
