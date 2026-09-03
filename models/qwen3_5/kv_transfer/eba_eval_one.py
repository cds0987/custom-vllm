"""eba_eval_one -- cham MOT checkpoint eba_grpo tren bo EBA HELD-OUT (seed
KHAC voi luc train/val trong eba_grpo.py mac dinh seed=0 -- dam bao khong
dam mau). Chay 2 lan (vd --tag best / --tag last) roi gop + kiem dinh
McNemar o run_eba_compare.sh, dung dung pattern run_gsm_traintest.sh.

    python eba_eval_one.py --ckpt-dir /content/eba_grpo_v1 --tag best --n 200
"""
import argparse
import importlib.util
import json
import pathlib

import torch

_H = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
eba = _load("eba_gen")
e5.patch_recurrent_rebind()
WARM_P = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--ckpt-dir", default="/content/eba_grpo_v1")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--difficulty-max", type=int, default=1)
    ap.add_argument("--seed", type=int, default=99999,
                    help="KHAC seed=0 dung luc train/val trong eba_grpo.py -- "
                         "dam bao held-out that")
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--out", default="/content/logs/eba_eval_one.json")
    args = ap.parse_args()

    from peft import PeftModel

    items = eba.build(args.n * 3, "/tmp/_eba_eval_big.json", seed=args.seed)
    items = [it for it in items if it["difficulty"] <= args.difficulty_max][:args.n]
    print(f"held-out: {len(items)} item (seed={args.seed})", flush=True)

    tok_s, model_s = e5.load_4bit(args.src_model)
    model_s = PeftModel.from_pretrained(model_s, f"{args.ckpt_dir}/lora_{args.tag}")
    model_s = model_s.merge_and_unload()
    model_s.eval()
    tok_s.truncation_side = "left"
    from transformers import AutoConfig
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    tok_t, model_t = e5.load_4bit(args.tgt_model)
    model_t = PeftModel.from_pretrained(model_t, f"{args.ckpt_dir}/lorat_{args.tag}")
    model_t.eval()
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    STOPS = e5.stop_ids(tok_t, model_t)

    mp = f"{args.ckpt_dir}/mapper_{args.tag}.pt"
    _meta = torch.load(mp, map_location="cpu").get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    mapper.load(mp)
    print(f"nap xong checkpoint {args.tag} (gdn_terms={_meta.get('gdn_terms', 1)})",
          flush=True)

    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _p = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_p)
    del _p, _ids
    torch.cuda.empty_cache()
    import copy as _copy

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

    results, texts = {}, {}
    for i, it in enumerate(items):
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=2048)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        with torch.no_grad():
            src = e5.prefill_chunked(model_s, cut)
            tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
            st = e5.build_student_past(tpl, src, mapper)
            o = model_t(input_ids=warm, past_key_values=st, use_cache=True)
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
        txt = tok_t.decode(gen, skip_special_tokens=True)
        results[it["id"]] = eba.score_eba(it, txt)
        texts[it["id"]] = txt[:200]
        del st, tpl, src, cur, o
        torch.cuda.empty_cache()
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(items)} xong", flush=True)

    n = len(results)
    print(f"\n{args.tag:10} A={sum(v['A'] for v in results.values()) / n:.3f} "
          f"B={sum(v['B'] for v in results.values()) / n:.3f} "
          f"C={sum(v['C'] for v in results.values()) / n:.3f} (n={n})")

    out = {"tag": args.tag, "n": n, "seed": args.seed,
          "results": results, "texts": texts}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"da ghi {args.out}")


if __name__ == "__main__":
    main()
