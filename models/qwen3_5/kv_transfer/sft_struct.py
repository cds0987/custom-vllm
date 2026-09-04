"""sft_struct -- BUOC 2: SFT day mapper+LoRA sinh dung CAU TRUC cho RL.

MUC TIEU DUY NHAT la CAU TRUC, khong phai do chinh xac (user chot 2026-09-05:
"toi chi muon training cho no sinh duoc dung format"). Nen:
  - chi so chinh = TY LE PARSE DUOC (co du <think>/ENTITIES/STEPS/Final Answer)
  - cong ra Buoc 3 = parse >= 90%
  - LUU MAU DAU RA de doc tay (thieu sot da mac o vong truoc: eval chi dem
    dung/sai, khong giu van ban -> khong doc tay duoc, vi pham rule 15)

PROMPT GIU NGUYEN template goc (ket thuc o "Solution: "). Gold bat dau bang
<think> -> model phai HOC cach tu sinh khuon do tu TRONG SO, khong nho prompt
nhac (user chot: "instruction cua model thuong khong co cu the nhu vay, mapper
+ lora buoc model hieu phai lam nhu vay").

GOM LO: theo DUNG quy uoc e9_joint.make_batches -- chi ghep cac mau CUNG DO
DAI PROMPT CHINH XAC, khong dem mot token nao o prompt (probe_batch da do:
dem trai lam attention keys sai 96%, GDN xau hon 3,7 lan). Gold dai khac nhau
thi dem -100 va cham CE co mat na (dem o gold vo hai vi khong qua prefill).

    python -u sft_struct.py --epochs 1 --batch 4 --accum 4
"""
import argparse
import gc
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
gs = _load("gsm_struct")
e5.patch_recurrent_rebind()
WARM_P = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--init-dir", default="/content/gsm_grpo_v1")
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--struct-gold", default="/content/struct_gold_gsm.json")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=0, help="0 = tinh tu epochs")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--mapper-ckpt", type=int, default=1,
                    help="1 = bat co ckpt san co cua Mapper: tinh lai map_attn "
                         "trong backward thay vi giu activation fp32 (~0,5GB "
                         "@1K ctx moi lop). Doi CHUT toc do lay CHO de tang "
                         "batch -- B=2 OOM khi tat co nay.")
    ap.add_argument("--max-ctx", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--gold-cap", type=int, default=320)
    ap.add_argument("--gen-len", type=int, default=320)
    ap.add_argument("--tbptt", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/content/sft_struct_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="sft_struct_v1")
    ap.add_argument("--sanity", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    _api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    # ---- du lieu: CHI item co gold CO CAU TRUC (Buoc 0 da loc 2 tang) -------
    data = json.loads(Path(args.data_file).read_text())
    sg = json.loads(Path(args.struct_gold).read_text())
    items = []
    for sp in ("train", "val"):
        for it in data.get(sp, []):
            g = sg.get(it.get("id", ""))
            if it.get("kind") == "gsm8k" and g and g.get("gold"):
                it = dict(it)
                it["gold"] = g["gold"]
                it["gold_struct"] = g
                items.append(it)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    ev_items, tr_items = items[:args.n_eval], items[args.n_eval:]
    print(f"struct-gold: {len(items)} mau -> {len(tr_items)} train / "
          f"{len(ev_items)} eval", flush=True)

    # ---- 9B + LoRA-9B -------------------------------------------------------
    t0 = time.time()
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    model_t = get_peft_model(model_t, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "o_proj", "in_proj_qkvz", "out_proj"],
        task_type="CAUSAL_LM"))
    model_t.train()
    p_lt = Path(args.init_dir) / "lorat_best"
    if p_lt.exists():
        set_peft_model_state_dict(model_t, load_file(str(p_lt / "adapter_model.safetensors")))
        print(f"warm-start LoRA-9B tu {p_lt}", flush=True)
    lora_t_params = [p for p in model_t.parameters() if p.requires_grad]
    tok_t.truncation_side = "left"
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    STOPS = e5.stop_ids(tok_t, model_t)

    # ---- 4B + LoRA-4B -------------------------------------------------------
    tok_s, model_s = e5.load_4bit(args.src_model)
    for p in model_s.parameters():
        p.requires_grad_(False)
    model_s = get_peft_model(model_s, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    p_l = Path(args.init_dir) / "lora_best"
    if p_l.exists():
        set_peft_model_state_dict(model_s, load_file(str(p_l / "adapter_model.safetensors")))
        print(f"warm-start LoRA-4B tu {p_l}", flush=True)
    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    tok_s.truncation_side = "left"
    theta_s = e5.e1.get_rope_theta(
        __import__("transformers").AutoConfig.from_pretrained(
            args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    mp = Path(args.init_dir) / "mapper_best.pt"
    _meta = torch.load(mp, map_location="cpu").get("_meta", {}) if mp.exists() else {}
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    if mp.exists():
        mapper.load(str(mp))
        print(f"warm-start mapper tu {mp}", flush=True)
    mapper.ckpt = bool(args.mapper_ckpt)
    print(f"nap xong {time.time()-t0:.0f}s | mapper.ckpt={mapper.ckpt}", flush=True)

    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _p = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_p)
    del _p, _ids
    torch.cuda.empty_cache()
    import copy as _copy

    def meta_for(t, b):
        """t = do dai ngu canh, b = KICH THUOC LO.

        BUG DA SUA (2026-09-05, bat khi do batch B=2): ban dau chi va chieu
        batch cho lop ATTENTION (k/v) ma bo sot lop GDN (rec/conv) -> template
        giu conv_states o batch 1, trong khi dau vao batch 2 ->
        'Sizes of tensors must match except in dimension 2. Expected size 1 but
        got size 2'. build_student_past thay recurrent_states bang dau ra
        mapper (dung batch) nhung conv_states lay zeros_like TU TEMPLATE, nen
        template sai batch la hong."""
        m = _copy.deepcopy(base_meta)
        m["cache_ints"] = {k: (t if v == T_BASE else v)
                           for k, v in m["cache_ints"].items()}
        for lay in m["layers"]:
            lay["ints"] = {k: (t if v == T_BASE else v) for k, v in lay["ints"].items()}
            keys = ("k", "v") if lay["kind"] == "a" else ("rec", "conv")
            for key in keys:
                sh, dt = lay[key]
                sh = tuple(t if d == T_BASE else d for d in sh)
                lay[key] = ((b,) + sh[1:], dt)   # chieu 0 = batch, ca a lan g
        return m

    def prefill_tbptt(ids, wnd):
        past, cutp = None, max(0, ids.shape[1] - wnd)
        if cutp:
            with torch.no_grad():
                for s in range(0, cutp, 1024):
                    o = model_s(input_ids=ids[:, s:min(s + 1024, cutp)],
                                past_key_values=past, use_cache=True, logits_to_keep=1)
                    past = o.past_key_values
        for s in range(cutp, ids.shape[1], 1024):
            o = model_s(input_ids=ids[:, s:s + 1024], past_key_values=past,
                        use_cache=True, logits_to_keep=1)
            past = o.past_key_values
        return past

    def enc(it):
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.max_ctx)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        gold = tok_t(it["gold"], add_special_tokens=False,
                     return_tensors="pt")["input_ids"][:, :args.gold_cap].to("cuda")
        feed = torch.cat([warm, gold[:, :-1]], 1)
        return cut, warm, gold, feed

    def enc_batch(group):
        """Y HET quy uoc e9_joint.enc_batch: prompt PHAI cung do dai (khong dem
        token nao); gold dem -100 va cham CE co mat na."""
        encs = [enc(it) for it in group]
        cut = torch.cat([e[0] for e in encs], 0)
        warm = torch.cat([e[1] for e in encs], 0)
        gmax = max(e[2].shape[1] for e in encs)
        gold = torch.full((len(encs), gmax), -100, dtype=torch.long, device="cuda")
        feed = torch.cat([torch.nn.functional.pad(
            e[3], (0, gmax - e[2].shape[1]), value=0) for e in encs], 0)
        for i, e in enumerate(encs):
            gold[i, :e[2].shape[1]] = e[2][0]
        return cut, warm, gold, feed

    def make_batches(xs, B):
        """Gom theo DO DAI PROMPT CHINH XAC (e9_joint). Mau bi cat ve max_ctx
        chay DON LE de khong ghep cap 2 x max_ctx (hoc phi OOM joint49aa)."""
        if B <= 1:
            return [[x] for x in xs]
        by, over = {}, []
        for x in xs:
            n = len(tok_t(x["prompt"], add_special_tokens=False, truncation=True,
                          max_length=args.max_ctx)["input_ids"])
            (over if n >= args.max_ctx else by.setdefault(n, [])).append(x)
        out_, left = [], list(over)
        for n, g in by.items():
            for k in range(0, len(g) - len(g) % B, B):
                out_.append(g[k:k + B])
            left += g[len(g) - len(g) % B:]
        out_ += [[x] for x in left]
        random.Random(args.seed).shuffle(out_)
        n_full = sum(len(g) for g in out_ if len(g) == B)
        print(f"gom lo B={B}: {len(out_)} lo | {n_full}/{len(xs)} item "
              f"({100*n_full/max(len(xs),1):.1f}%) vao duoc lo day", flush=True)
        return out_

    def student_past(cut):
        src = prefill_tbptt(cut, args.tbptt)
        tpl = e5.build_template_from_meta(probe_t, meta_for(cut.shape[1], cut.shape[0]))
        st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit([{"params": mapper.params, "lr": args.lr},
                              {"params": lora_params, "lr": args.lora_lr},
                              {"params": lora_t_params, "lr": args.lora_lr}])

    @torch.no_grad()
    def evaluate(save_to=None):
        """CHI SO CHINH = ty le PARSE DUOC (muc tieu cua buoc SFT nay).
        Luu ca van ban de DOC TAY (rule 15)."""
        n_ok = n_think = n_ans = 0
        texts = []
        for it in ev_items:
            cut, warm, _, _ = enc(it)
            src = e5.prefill_chunked(model_s, cut)
            tpl = e5.build_template_from_meta(probe_t, meta_for(cut.shape[1], 1))
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
            p = gs.parse(txt)
            n_ok += int(p["ok"])
            n_think += int(p["has_think"])
            n_ans += int(p["answer"] is not None)
            texts.append({"id": it["id"], "ok": p["ok"], "text": txt[:900]})
            del st, tpl, src, cur, o
            torch.cuda.empty_cache()
        n = max(len(ev_items), 1)
        if save_to:
            Path(save_to).write_text(json.dumps(texts, ensure_ascii=False, indent=1))
        return {"parse": n_ok / n, "think": n_think / n, "has_ans": n_ans / n}

    results = {"args": vars(args), "eval": [], "train": []}

    def save_all(tag):
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        model_t.save_pretrained(str(out / f"lorat_{tag}"))
        (out / "results.json").write_text(json.dumps(results, indent=1))
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(f"{args.hf_prefix}/mapper_{tag}.pt",
                                  str(out / f"mapper_{tag}.pt")),
               CommitOperationAdd(f"{args.hf_prefix}/results.json",
                                  str(out / "results.json"))]
        sp = out / f"samples_{tag}.json"
        if sp.exists():
            ops.append(CommitOperationAdd(f"{args.hf_prefix}/samples_{tag}.json", str(sp)))
        for sub in (f"lora_{tag}", f"lorat_{tag}"):
            for f in sorted((out / sub).glob("*")):
                if f.is_file():
                    ops.append(CommitOperationAdd(f"{args.hf_prefix}/{sub}/{f.name}", str(f)))
        try:
            _api.create_commit(repo_id=args.hf_repo, operations=ops,
                               commit_message=f"{args.hf_prefix} {tag}")
            print(f"HF-UP {tag} ({len(ops)} file, 1 commit)", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL {tag}: {type(ex).__name__}: {ex}", flush=True)

    batches = make_batches(tr_items, args.batch)
    steps = args.steps or int(args.epochs * len(batches))

    e0 = evaluate(out / "samples_step0.json")
    print(f"TRUOC train: parse={e0['parse']*100:.1f}% think={e0['think']*100:.1f}% "
          f"co_dap_so={e0['has_ans']*100:.1f}%", flush=True)
    results["eval"].append({"step": 0, **e0})

    best = -1.0
    t_start = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        grp = batches[(step - 1) % len(batches)]
        cut, warm, gold, feed = (enc(grp[0]) if len(grp) == 1 else enc_batch(grp))
        if gold.shape[1] < 1:
            continue
        st = student_past(cut)
        o = model_t(input_ids=feed, past_key_values=st, use_cache=True)
        lp = torch.log_softmax(o.logits[:, WARM_P - 1:WARM_P - 1 + gold.shape[1]].float(), -1)
        valid = (gold >= 0).float()
        nll = -lp.gather(2, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid
        loss = nll.sum() / valid.sum().clamp(min=1) / args.accum
        loss.backward()
        lv = float(loss.detach()) * args.accum
        if step % args.accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        del st, o, lp, nll, loss
        gc.collect()
        torch.cuda.empty_cache()

        if step % 25 == 0:
            results["train"].append([step, round(lv, 4)])
            print(f"buoc {step}/{steps} CE={lv:.4f} B={len(grp)} "
                  f"{(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB "
                  f"con ~{(steps-step)*(time.time()-t_start)/step/60:.0f} phut",
                  flush=True)
        if args.sanity and step >= args.sanity:
            print(f"SANITY xong {step} buoc B={args.batch} accum={args.accum}, "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB, "
                  f"{(time.time()-t_start)/step:.2f}s/buoc", flush=True)
            print("SFT_STRUCT_SANITY_EXIT", flush=True)
            return
        if step % args.eval_every == 0 or step == steps:
            ev = evaluate(out / "samples_last.json")
            results["eval"].append({"step": step, **ev})
            print(f"=== EVAL buoc {step}: parse={ev['parse']*100:.1f}% "
                  f"think={ev['think']*100:.1f}% co_dap_so={ev['has_ans']*100:.1f}% "
                  f"(dau: parse={e0['parse']*100:.1f}%)", flush=True)
            save_all("last")
            if ev["parse"] > best:
                best = ev["parse"]
                save_all("best")
                print(f"    ky luc parse={best*100:.1f}%", flush=True)
            if ev["parse"] >= 0.90:
                print("    *** DAT CONG >=90% parse -> du dieu kien sang Buoc 3 (RL)",
                      flush=True)

    print(f"\nXONG {steps} buoc. parse tot nhat = {best*100:.1f}%", flush=True)
    print("SFT_STRUCT_EXIT", flush=True)


if __name__ == "__main__":
    main()
