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
import pathlib
import sys
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

def score(it, txt):
    """gd.score_item chi biet bbh/gsm8k/musr/suite_*. bfcl/needle cham y het
    e6v3_ce (bfcl: ten ham xuat hien; needle: ma 6 so xuat hien)."""
    b = it["bench"]
    if b == "bfcl":
        return int(it["expect"] in txt)
    if b == "needle":
        return int(it["expect"] in txt)
    return int(gd.score_item(it, txt))


N_NEW = {"gsm8k": 320, "bbh": 48, "musr": 24, "bfcl": 24, "needle": 16,
         "suite_rag": 24, "suite_mid": 24, "suite_math": 24, "suite_swe": 24}
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


def build_beam(n_bfcl, n_needle, n_suite, tok):
    """Cac ho DANG VUOT TRAN (bfcl/needle/suite) o quy mo lon, dai NIEM PHONG.

    bfcl   -- build_data() cua e6v3 an: exec_simple[:100], simple[:400],
              parallel[:100], multiple[:100]. O day lay DUOI cac moc do.
    needle -- sinh moi voi seed CHUA TUNG dung (build_data dung 30000-45000
              va 100000-110000; o day 500000+).
    suite  -- suite_gen seed 777 da dung cho train -> doi seed 31337.
    """
    e6 = _load("e6v3_ce")
    items = []
    # ---- bfcl: lay phan duoi cac moc build_data da an ----
    for fname, skip in (("BFCL_v3_exec_simple.json", 100),
                        ("BFCL_v3_simple.json", 400),
                        ("BFCL_v3_parallel.json", 100),
                        ("BFCL_v3_multiple.json", 100)):
        try:
            xs = e6.bfcl_load(fname, skip + n_bfcl // 4)[skip:]
        except Exception as e:
            print(f"bfcl bo qua {fname}: {type(e).__name__}: {e}")
            continue
        for j, it in enumerate(xs):
            items.append({"bench": "bfcl", "kind": "bfcl", "sub": fname,
                          "id": f"big/{fname}/{skip+j}", "prompt": it["prompt"],
                          "expect": it["fn"]})
    # ---- needle: seed moi hoan toan, 3 do dai ----
    # seed KHONG chia het cho 400: e6.needle_items lay stream seed = seed%400,
    # nen seed tron 400 se tai dung DUNG lat filler cua train (30000%400 == 0).
    for ctx, n, seed in ((700, n_needle // 2, 500017),
                         (2000, n_needle // 3, 510023),
                         (4000, n_needle - n_needle // 2 - n_needle // 3,
                          520031)):
        if n <= 0:
            continue
        for j, it in enumerate(e6.needle_items(tok, n, seed, ctx_tok=ctx)):
            items.append({"bench": "needle", "kind": "needle",
                          "sub": f"ctx{ctx}", "id": f"big/needle{ctx}/{j}",
                          "prompt": it["prompt"], "expect": it["code"]})
    # ---- suite: seed khac han train ----
    if n_suite:
        sg = _load("suite_gen")
        out = "/tmp/_suite_big.json"
        sg.build_suite(n_suite, [1024, 2048, 4096],
                       ["rag", "mid", "math", "swe"], out, tok, seed=31337)
        for j, it in enumerate(json.load(open(out))):
            fam = it["family"]
            items.append({"bench": f"suite_{fam}", "kind": f"suite_{fam}",
                          "sub": str(it.get("ctx")), "id": f"big/{fam}/{j}",
                          "prompt": it["prompt"], "expect": it["expect"]})
    return items


def e6_prompts(tok):
    """Prompt bfcl/needle/ifstruct/pbtable ma e6v3.build_data() sinh LUC CHAY.

    Chung KHONG nam trong train_items.json (file do chi co gsm8k/bbh/musr/
    suite), nen neu chi doi chieu file thi kiem ro ri se BO SOT dung hai ho
    dang duoc do o day. Dung lai chinh build_data voi max_ctx lon nhat da
    dung khi train (4096) de phu het bucket needle."""
    e6 = _load("e6v3_ce")
    d = e6.build_data(tok, max_ctx=4096)
    return {x["prompt"] for x in d["train"] + d["val"] + d["test"]}


def check_overlap(items, others, extra=None, max_drop=0.05):
    """Doi chieu CHUOI PROMPT — khong tin suy luan chi so.

    LOAI BO thay vi assert, nhung CO NGUONG. Ly do: hai kieu "trung" khac han
    nhau ve ban chat, va gop chung lai thi mat thong tin.

      - trung LE TE  = ban than dataset co dong lap (bbh/causal_judgement co
        cau hoi lap nguyen van o nhieu hang). Bo di la dung, khong co gi de sua.
      - trung HANG LOAT = dai chi so tinh sai (vd tap niem phong duoc dung lai
        voi n khac luc bao cao). Bo di la GIAU LOI — phai dung, va sua dai.

    Nguong 5%: duoi thi bo + ghi ro, tren thi nem. In ro tung bo de biet duong
    nao lech, thay vi mot dong "RO RI" khong noi duoc gi.
    """
    seen = set(extra or ())
    for f in others:
        if not Path(f).exists():
            print(f"  CANH BAO: khong thay {f}")
            continue
        d = json.loads(Path(f).read_text())
        pool = (d["train"] + d["val"]) if isinstance(d, dict) else d
        seen |= {x["prompt"] for x in pool}
    bad = [it for it in items if it["prompt"] in seen]
    r = len(bad) / max(len(items), 1)
    print(f"kiem ro ri: {len(bad)}/{len(items)} ({100*r:.1f}%) trung "
          f"{len(seen)} mau da dung")
    if bad:
        print("  theo bo:", dict(Counter(it["bench"] for it in bad)))
        print("  vi du :", [it["id"] for it in bad[:5]])
    if r > max_drop:
        raise AssertionError(
            f"RO RI {100*r:.1f}% > nguong {100*max_drop:.0f}% — day la lech "
            f"DAI CHI SO, khong phai dong lap trong dataset. Sua dai truoc khi "
            f"chay tiep. Vi du: {[it['id'] for it in bad[:5]]}")
    keep = [it for it in items if it["prompt"] not in seen]
    if bad:
        print(f"  -> bo {len(bad)}, con {len(keep)} mau")
    return keep


# ----------------------------------------------------------------- self ----

def self_tag(args):
    """Ten file theo MODEL (+lora): chay 4B ma ghi de len cot tran cua 9B thi
    mat luon 12,3 phut da ton, va tro thanh so bao cao sai."""
    t = args.tgt_model.split("/")[-1].replace(".", "").lower()
    return t + ("_lora" if args.lora else "")


def run_self(args):
    from vllm import LLM, SamplingParams
    items = json.loads(Path(ITEMS_F).read_text())
    if args.benches:
        keep = set(args.benches.split(","))
        items = [it for it in items if it["bench"] in keep]
        print(f"loc bo: con {len(items)} mau", flush=True)

    model_path = args.tgt_model
    if args.lora:
        # vLLM khong nap duoc adapter tren duong bnb-4bit mot cach chac chan
        # -> merge vao ban BF16 roi luu ra dia. 4B bf16 ~8GB, vua.
        import shutil
        import torch as _t
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        dst = "/content/_merged_" + self_tag(args)
        if not Path(dst, "config.json").exists():
            print(f"merge LoRA vao ban bf16 -> {dst}", flush=True)
            m = AutoModelForCausalLM.from_pretrained(
                args.tgt_model, dtype=_t.bfloat16, device_map="cpu")
            m = PeftModel.from_pretrained(m, args.lora).merge_and_unload()
            m.save_pretrained(dst)
            AutoTokenizer.from_pretrained(args.tgt_model).save_pretrained(dst)
            del m
            import gc
            gc.collect()
        model_path = dst

    kw = dict(model=model_path, max_model_len=args.max_len,
              gpu_memory_utilization=0.90)
    if not args.lora:
        kw["quantization"] = "bitsandbytes"
    llm = LLM(**kw)
    by = defaultdict(list)
    for it in items:
        by[it["bench"]].append(it)
    out, t0 = {}, time.time()
    for bench, grp in by.items():
        sp = SamplingParams(temperature=0.0, max_tokens=N_NEW[bench])
        res = llm.generate([g["prompt"] for g in grp], sp)
        hit = 0
        for g, r in zip(grp, res):
            h = score(g, r.outputs[0].text)
            out[g["id"]] = h
            hit += h
        print(f"self {bench}: {hit}/{len(grp)} = {100*hit/len(grp):.1f}%",
              flush=True)
    fn = OUT_DIR / f"evalbig_self_{self_tag(args)}.json"
    json.dump(out, open(fn, "w"))
    print(f"self xong {(time.time()-t0)/60:.1f} phut -> {fn.name}")
    if args.hf_prefix and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(fn), repo_id=args.hf_repo,
                path_in_repo=f"evalbig/{fn.name}")
            print("HF-UP", fn.name)
        except Exception as e:
            print("HF-UP FAIL", type(e).__name__, str(e)[:80])


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
    if args.benches:
        keep = set(args.benches.split(","))
        n0 = len(items)
        items = [it for it in items if it["bench"] in keep]
        print(f"loc bo: {n0} -> {len(items)} mau ({sorted(keep)})", flush=True)
    spill = Path("/content/big_spill")
    spill.mkdir(parents=True, exist_ok=True)

    # Doc ket qua da cham TRUOC pha A. Neu khong, sau moi lan recycle pha A se
    # prefill lai ca 1875 mau (~50 phut) du phan lon da co diem.
    done = set()
    mp = OUT_DIR / f"{args.hf_prefix or 'evalbig'}_mapped.json"
    if not mp.exists() and args.hf_prefix:
        try:
            from huggingface_hub import hf_hub_download
            p_ = hf_hub_download(args.hf_repo,
                                 f"{args.hf_prefix}/{mp.name}",
                                 token=os.environ.get("HF_TOKEN"))
            mp.write_bytes(pathlib.Path(p_).read_bytes())
            print(f"keo ket qua do dang tu HF", flush=True)
        except Exception as e:
            print(f"HF chua co ket qua do dang ({type(e).__name__})", flush=True)
    if mp.exists():
        done = set(json.loads(mp.read_text()))
        print(f"NOI LAI: da cham {len(done)} mau", flush=True)
    items = [it for it in items if it["id"] not in done]
    print(f"con {len(items)} mau phai chay", flush=True)
    if not items:
        print("MAPPED XONG (khong con gi)", flush=True)
        return

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
        pth = spill / (it["id"].replace("/", "_") + ".pt")
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
    probe_t = probe
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    import torch as _t
    _meta = _t.load(args.mapper, map_location="cpu").get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False))
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"mapper {args.mapper} | token dung {sorted(STOPS)}", flush=True)

    # ---- TEMPLATE-XUONG thay cho prefill 9B day du ----
    # build_student_past thay sach moi tensor cua template, chi dung hinh dang
    # va cac truong int -> prefill 9B chi de lay hinh dang la tinh toan thua.
    import copy as _copy
    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _past = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_past)
    del _past, _ids
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

    # DOI CHIEU voi prefill THAT: template lech vi tri = cache vo nghia ma
    # KHONG bao loi. Khong co buoc nay thi tiet kiem thoi gian bang cach lam
    # sai ket qua.
    for _t in (256, 1024):
        with torch.no_grad():
            _ids = torch.randint(1000, 5000, (1, _t), device="cuda")
            _p = e5.prefill_chunked(model_t, _ids)
        if e5.cache_meta(_p) != meta_for_len(_t):
            raise SystemExit(f"VERIFY-META LECH o T={_t} — dung lai, KHONG "
                             "duoc dung template suy ra")
        del _p, _ids
        torch.cuda.empty_cache()
    print(f"verify-meta KHOP o T=256 va T=1024 (base {T_BASE})", flush=True)

    # NOI LAI: nap ket qua da co (local, hoac keo tu HF neu recycle xoa sach)
    out = json.loads(mp.read_text()) if mp.exists() else {}

    def _flush(tag=""):
        json.dump(out, open(mp, "w"))
        if args.hf_prefix and os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                    path_or_fileobj=str(mp), repo_id=args.hf_repo,
                    path_in_repo=f"{args.hf_prefix}/{mp.name}")
            except Exception as e:
                print(f"  HF-UP loi: {type(e).__name__}", flush=True)

    bd = _load("batch_decode")

    def build_one(it):
        """cache da map + warm token cua MOT mau."""
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src = e5.load_cache(spill / (it["id"].replace("/", "_") + ".pt"))
        tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
        past = e5.build_student_past(tpl, src, mapper)
        del tpl, src
        return past, warm

    @torch.no_grad()
    def run_one(it):
        past, warm = build_one(it)
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
        del past, cur, o
        torch.cuda.empty_cache()
        return tok_t.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def run_group(grp):
        """Ca lo CUNG bench (nen cung n_new)."""
        if len(grp) == 1:
            return [run_one(grp[0])]
        pasts, warms = [], []
        for it in grp:
            p_, w_ = build_one(it)
            pasts.append(p_); warms.append(w_)
        past, lens, T_max = bd.stack_students(pasts)
        del pasts
        gen = bd.greedy_batch(model_t, past, lens, T_max, warms,
                              N_NEW[grp[0]["bench"]], STOPS)
        del past
        torch.cuda.empty_cache()
        return [tok_t.decode(g, skip_special_tokens=True) for g in gen]

    # ---- CONG KIEM: batch 1 vs batch B tren cung mau ----
    if args.verify_batch and args.decode_batch > 1:
        vs = [it for it in items][:args.verify_batch]
        by = defaultdict(list)
        for it in vs:
            by[it["bench"]].append(it)
        n_ok = n_bad = 0
        for b, grp in by.items():
            got_b = run_group(grp)
            for it, tb in zip(grp, got_b):
                t1 = run_one(it)
                if score(it, t1) == score(it, tb):
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f"  LECH {it['id']}\n    b1={t1[:100]!r}\n"
                          f"    bB={tb[:100]!r}", flush=True)
        print(f"cong kiem batch: {n_ok} khop / {n_bad} lech", flush=True)
        if n_bad:
            raise SystemExit("GOM LO SAI KET QUA — dung lai, khong dung so nay")

    t0, n_new = time.time(), 0
    todo = [it for it in items if it["id"] not in out]
    groups, cur_g = [], []
    for it in todo:
        if cur_g and (it["bench"] != cur_g[0]["bench"]
                      or len(cur_g) >= args.decode_batch):
            groups.append(cur_g); cur_g = []
        cur_g.append(it)
    if cur_g:
        groups.append(cur_g)
    i = 0
    for grp in groups:
        txts = run_group(grp)
        for it, txt in zip(grp, txts):
            out[it["id"]] = {"hit": score(it, txt), "txt": txt[:400],
                             "n_tok": len(txt.split())}
        i += len(grp)
        n_new += len(grp)
        if n_new % 25 < len(grp):
            h = sum(v["hit"] for v in out.values())
            el = (time.time() - t0) / 60
            con = len(todo) - i
            print(f"mapped {i}/{len(todo)} dung {h} ({100*h/len(out):.1f}%) "
                  f"| {el:.0f} phut | con ~{el/max(n_new,1)*con:.0f} phut",
                  flush=True)
            _flush()          # ghi + len HF moi 25 mau: recycle chi mat <25 mau
    _flush()
    print("MAPPED XONG", flush=True)


# ------------------------------------------------------------------ agg ----

def run_agg():
    items = {it["id"]: it for it in json.loads(Path(ITEMS_F).read_text())}
    slf = json.loads((OUT_DIR / "evalbig_self.json").read_text())
    mpd = json.loads((OUT_DIR / (os.environ.get("EVALBIG_PREFIX",
        "evalbig") + "_mapped.json")).read_text())
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
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--slice", default="")
    ap.add_argument("--decode-batch", type=int, default=1,
                    help="Gom lo o buoc decode. 1 = nhu cu. Nut co chai la batch=1 (moi token doc ~5GB trong so 4-bit tu HBM), khong phai attention (chiem 0,03% phep tinh).")
    ap.add_argument("--verify-batch", type=int, default=0,
                    help="Chay N mau dau O CA batch 1 lan batch B roi doi chieu. Bat buoc truoc khi tin so: mot loi mask/vi tri se lam sai TOAN BO ket qua ma khong nem loi nao.")
    ap.add_argument("--benches", default="",
                    help="Chi chay cac bo nay (phay ngan cach). Bo 1875 mau "
                         "GIU NGUYEN tren dia de con tai lap — loc chi ap luc "
                         "chay. Giai doan 1 chi do cac ho DA GHI NHAN chay: "
                         "gsm8k ton 320 token/mau = phan dat nhat ca luot ma "
                         "mapper moi dat 9%, suite_math tran 1,6% nen khong "
                         "phan biet duoc mapper tot hay xau.")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="evalbig",
                    help="rong = tat noi-lai va upload")
    ap.add_argument("--n-bbh", type=int, default=1300)
    ap.add_argument("--n-gsm8k", type=int, default=500)
    ap.add_argument("--n-musr", type=int, default=60)
    ap.add_argument("--n-bfcl", type=int, default=400)
    ap.add_argument("--n-needle", type=int, default=240)
    ap.add_argument("--n-suite", type=int, default=500)
    args = ap.parse_args()

    if args.mode == "gen":
        items = build_items(args.n_bbh, args.n_gsm8k, args.n_musr)
        if args.n_bfcl or args.n_needle or args.n_suite:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.src_model)
            items += build_beam(args.n_bfcl, args.n_needle, args.n_suite, tok)
            extra = e6_prompts(tok)
            print(f"tap e6v3 (bfcl/needle/...) dung khi train: {len(extra)} "
                  "prompt dua vao kiem ro ri", flush=True)
        else:
            extra = None
        print("truoc khi kiem:", dict(Counter(i["bench"] for i in items)),
              flush=True)
        items = check_overlap(items, ["/content/train_items.json",
                                      "/content/ext_bench_items.json"], extra)
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
    # os._exit THAY VI return: `datasets`/`huggingface_hub` de lai thread nen
    # KHONG-daemon, nen interpreter ket o luc join va tien trinh treo VO HAN
    # sau khi da in xong ket qua. Da dinh hai lan trong mot buoi — mot lan tren
    # duong loi (traceback roi treo), mot lan tren duong THANH CONG (in
    # EVALBIG_EXIT roi treo, chan ca chuoi 5 pha phia sau). Moi thu can ghi da
    # ghi va flush truoc dong nay.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
