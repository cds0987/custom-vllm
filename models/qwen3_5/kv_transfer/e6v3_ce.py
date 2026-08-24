"""E6 v3 — mapper 4B->27B train bang CE-GOLD tren mien that (user duyet 2026-08-23).

Thay doi cot loi so v2 (user chi ra KL-only chua du):
  L = CE(gold)  +  BETA*KL(teacher||student)  +  lam(t)*auxMSE  +  GAMMA*dense
  - CE tinh TREN TOKEN DAP AN THAT (response-only — dung bai
    train_on_responses_only cua Unsloth: context nam trong cache nen moi vi
    tri duoc cham diem deu la gold token; bo CONV_WARM vi tri dau).
  - Chon checkpoint bang VALIDATION CHUC NANG (hit/valid/needle), khong bang
    loss (luat error-placement). Test NIEM PHONG mo dung 1 lan cuoi.
  - Chap nhan narrow-domain (user chot): train dung mien se dung.

Data (khao sat schema that 2026-08-23):
  - BFCL exec_simple: 20 item dau = TEST NIEM PHONG (y het E6); phan con lai
    + BFCL_v3_simple.json -> train/val. Gold = "fn_name(".
  - ifstruct (LiquidAI/ifstruct-v1.0, split test): KHONG co gold — dung
    PSEUDO-GOLD = greedy output cua chinh 27B tu doc (muc tieu user: bao dam
    dau ra cua 27B). Val metric = validator chuong trinh (parse YAML/JSON,
    check code-block/top_level_key theo spec cua tung item).
  - ParseBench (config parse-bench, split table): input goc la PDF -> khong
    dung thang; dung expected_markdown lam CONTEXT: "bang trong cache ->
    trich dung hang thu k", gold cat regex tu chinh bang — suite
    structured-retrieval.
  - Needle tong hop (khuon e6.make_train_ids): giu truc truy xuat dai.

Modes:
  --dry-data       : local, khong GPU — dung + in data mix de user duyet
  (mac dinh)       : Colab — phase A (4B spill) -> phase B (27B train+val)
                     -> test niem phong -> luu best-by-val
"""

import argparse
import gc
import importlib.util
import json
import random
import re
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e6_suite", Path(__file__).parent / "e6_suite.py")
e6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e6)
e5, e2 = e6.e5, e6.e2

TRAIN_MAX = 1024     # phong bi da kiem chung; 1536/2048 OOM (5.15 luu GDN
                     # state fp32 — cache 27B nang gap doi runtime cu)
NK_MAXLEN = 4096     # needle khong bao gio duoc cat (bug needle2k cu)
GOLD_MAX = 64        # fallback
# v3.2: gold/gen rieng tung loai — ifstruct/pbtable 0 diem vi gold 64 cat cut
GMAX = {"bfcl": 16, "needle": 12, "ifstruct": 128, "pbtable": 96}
GEN_LEN = {"bfcl": 24, "needle": 16, "ifstruct": 160, "pbtable": 120}
CONV_WARM = e5.CONV_WARM   # (v3.0 — giu cho tham chieu)
# v3.1 (user duyet 2026-08-24): CONV_WARM skip 4 vi tri dau cua GOLD = khong
# bao gio day token quyet dinh (fn name dau). Giao thuc moi: cache cat tai
# T-WARM_P, warm conv bang WARM_P token CUOI PROMPT (token that, khong phai
# dap an) -> CE cham TRON 100% gold, token dau trong so FIRST_W.
WARM_P = 5
FIRST_W = 3.0
BETA = 0.3           # trong so KL phu
CE_FLOOR = 0.2       # Unsloth: train loss <0,2 = overfit -> dung (user chot)
GAMMA = 0.05         # dense supervision
N_NEEDLE_TRAIN = 200
VAL_EVERY = 250   # v3.2: val 55 mau ~10 phut/lan, 2000 buoc = 8 moc
SEED = 7


# ------------------------------ data ----------------------------------------

def bfcl_load(fname, n):
    # KHONG dung load_dataset: datasets ban moi (Colab py3.13) parse JSON
    # kieu khac -> "Trailing data". Tai file tho + tu parse (JSON hoac JSONL).
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                           fname, repo_type="dataset")
    txt = Path(path).read_text(encoding="utf-8")
    try:
        ds = json.loads(txt)
    except ValueError:
        ds = [json.loads(l) for l in txt.splitlines() if l.strip()]
    items = []
    for ex in ds:
        q = ex.get("question")
        if isinstance(q, list):
            try:
                q = q[0][0]["content"]
            except Exception:
                q = str(q)
        fns = ex.get("function") or []
        if isinstance(fns, dict):
            fns = [fns]
        fname_ = fns[0].get("name") if fns else None
        if not q or not fname_:
            continue
        prompt = ("You can call these functions:\n" + json.dumps(fns)[:3000]
                  + f"\nUser request: {q}\nRespond with one function call.\nCall: ")
        items.append({"kind": "bfcl", "prompt": prompt,
                      "gold": fname_ + "(", "fn": fname_})
        if len(items) >= n:
            break
    return items


def ifstruct_load(n):
    from datasets import load_dataset
    ds = load_dataset("LiquidAI/ifstruct-v1.0", split="test")
    items = []
    for ex in ds:
        p = ex.get("prompt")
        if not (isinstance(p, str) and p.strip()):
            continue
        items.append({"kind": "ifstruct", "prompt": p[:4000] + "\nOutput: ",
                      "gold": None,          # pseudo-gold: 27B tu sinh o phase B0
                      "spec": {"fmt": ex.get("output_format"),
                               "key": ex.get("top_level_key"),
                               "code_block": ex.get("require_code_block")}})
        if len(items) >= n:
            break
    return items


def pbtable_load(n):
    from datasets import load_dataset
    ds = load_dataset("llamaindex/ParseBench", "parse-bench", split="table")
    rng = random.Random(SEED)
    items = []
    for ex in ds:
        md = ex.get("expected_markdown") or ""
        rows = re.findall(r"<tr>.*?</tr>", md, flags=re.S)
        if len(rows) < 3 or len(md) < 400:
            continue
        k = rng.randint(1, min(len(rows) - 1, 6))
        gold = re.sub(r"\s+", " ", rows[k]).strip()[:400]
        if len(gold) < 20:
            continue
        prompt = (md[:5000]
                  + f"\n\nRepeat EXACTLY row number {k+1} (the {k+1}-th <tr>...</tr>) "
                  + "of the table above, as one line.\nRow: ")
        items.append({"kind": "pbtable", "prompt": prompt, "gold": gold})
        if len(items) >= n:
            break
    return items


def needle_items(tok, n, seed0, ctx_tok=700):
    # MOT stream duy nhat roi cat lat (khuon E6). token_stream skip `seed`
    # document truoc khi lay token — seed lon = tai-va-vut hang van doc (bug
    # treo 20 phut da dinh 2026-08-23). Seed giu nho, xao tron bang rng.
    stream = e2.token_stream(tok, n * ctx_tok + ctx_tok, seed=seed0 % 400)
    items = []
    for i in range(n):
        rng = random.Random(seed0 + i)
        name = rng.choice(e2.NAMES) + str(rng.randint(0, 99))
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = stream[i * ctx_tok:(i + 1) * ctx_tok]
        half = ctx_tok // 2
        prompt = (tok.decode(ids[:half])
                  + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                  + tok.decode(ids[half:]) + "\n" + e2.build_q(name) + " ")
        items.append({"kind": "needle", "prompt": prompt, "gold": code + ".",
                      "code": code})
    return items


def build_data(tok=None):
    """v3.2 SCALE-UP (user duyet 2026-08-24). tok=None -> bo needle (dry)."""
    rng = random.Random(SEED)
    exec_simple = bfcl_load("BFCL_v3_exec_simple.json", 100)
    test = exec_simple[:20]                      # NIEM PHONG — y het E6
    train = exec_simple[20:]
    simple = bfcl_load("BFCL_v3_simple.json", 400)
    test_prompts = {it["prompt"] for it in test}
    simple = [it for it in simple if it["prompt"] not in test_prompts]
    rng.shuffle(simple)
    val = simple[:15]
    train += simple[15:]
    ifs = ifstruct_load(150)
    val += ifs[:15]
    train += ifs[15:]
    pbt = pbtable_load(120)
    val += pbt[:10]
    train += pbt[10:]
    if tok is not None:
        # train trong phong bi 1024 (grad); MIEN DAI do o EVAL (no-grad):
        # val needle 1500 + test 2000 — baselines chung minh eval @2000 OK
        train += needle_items(tok, 250, 30000)                  # ngan 700
        train += needle_items(tok, 100, 40000, ctx_tok=950)
        val += needle_items(tok, 10, 31000)
        val += needle_items(tok, 5, 41000, ctx_tok=1500)
        test += needle_items(tok, 10, 32000, ctx_tok=2000)   # needle 2K nhu E6
    rng.shuffle(train)
    return {"train": train, "val": val, "test": test}


def ifstruct_valid(text, spec):
    """Validator chuong trinh cho ifstruct (khong can gold)."""
    ok = True
    body = text
    if spec.get("code_block"):
        ok &= "```" in text
        m = re.search(r"```[a-z]*\n(.*?)(```|$)", text, flags=re.S)
        body = m.group(1) if m else text
    fmt = (spec.get("fmt") or "").lower()
    try:
        if fmt == "json":
            json.loads(body)
        elif fmt == "yaml":
            import yaml
            yaml.safe_load(body)
    except Exception:
        ok = False
    key = spec.get("key")
    if key:
        ok &= key in text
    return int(ok)


# ------------------------------ main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/content/mapper_v3.pt")
    ap.add_argument("--cache-dir", default="/content/v3_src")
    ap.add_argument("--results", default="/content/logs/e6v3_results.json")
    ap.add_argument("--dry-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-check", action="store_true",
                    help="falsification: nap mapper .last (thuoc long), "
                         "greedy tren 30 mau TRAIN — phai ra dap an neu "
                         "memorization la that")
    args = ap.parse_args()

    if args.dry_data:
        data = build_data(tok=None)
        for split, items in data.items():
            from collections import Counter
            cnt = Counter(it["kind"] for it in items)
            print(f"{split}: {len(items)} items  {dict(cnt)}")
        for kind in ("bfcl", "ifstruct", "pbtable"):
            it = next(x for x in data["train"] if x["kind"] == kind)
            print(f"\n===== SAMPLE {kind} =====")
            print("PROMPT:", it["prompt"][:400].replace("\n", " | "))
            print("GOLD:", (it["gold"] or "(pseudo-gold tu 27B)")[:200])
        print(f"\n(+{N_NEEDLE_TRAIN} needle train / 10 val / 10 test@2K "
              "them vao khi chay that)")
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig

    # transformers 5.15: update_recurrent_state lam .copy_() IN-PLACE len
    # tensor state ("static address for cudagraphs") — pha vo autograd khi
    # initial_state la tensor mapper mang grad (fla backward: "modified by an
    # inplace operation"). Rebind khi co grad; giu copy_ cho moi ca khac.
    try:
        from transformers.cache_utils import LinearAttentionLayer
        _orig_urs = LinearAttentionLayer.update_recurrent_state

        def _urs(self, recurrent_states, state_idx=0, **kw):
            cur = (self.recurrent_states.get(state_idx)
                   if isinstance(self.recurrent_states, dict) else None)
            if cur is not None and (cur.requires_grad
                                    or recurrent_states.requires_grad):
                self.recurrent_states[state_idx] = recurrent_states
                return recurrent_states
            return _orig_urs(self, recurrent_states, state_idx, **kw)

        LinearAttentionLayer.update_recurrent_state = _urs
        print("patched update_recurrent_state (rebind khi co grad)")
    except ImportError:
        pass

    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    results = {"val_curve": [], "config": {k: v for k, v in vars(args).items()}}

    def save_results():
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w") as fh:
            json.dump(results, fh, indent=1)

    # ---------------- PHASE A: 4B mot minh — spill cache moi item ------------
    marker = cdir / "PHASE_A_DONE"
    tok = None
    if not marker.exists():
        tok_s, model_s = e5.load_4bit(args.src_model)
        tok = tok_s
        data = build_data(tok_s)
        with open(cdir / "data.json", "w") as fh:
            json.dump(data, fh)
        with torch.no_grad():
            for split in ("train", "val", "test"):
                for i, it in enumerate(data[split]):
                    pth = cdir / f"{split}{i}.pt"
                    if pth.exists():
                        continue
                    ml = NK_MAXLEN if it["kind"] == "needle" else TRAIN_MAX
                    enc = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                                max_length=ml).to("cuda")
                    # v3.1: cache cat tai T-WARM_P (GDN khong tua nguoc duoc)
                    past = model_s(input_ids=enc["input_ids"][:, :-WARM_P],
                                   use_cache=True, logits_to_keep=1).past_key_values
                    e5.spill_cache(past, pth)
                    del past
                    torch.cuda.empty_cache()
                    if i % 50 == 0:
                        print(f"A {split} {i}")
        del model_s
        gc.collect(); torch.cuda.empty_cache()
        marker.touch()
        print("PHASE_A_DONE")

    # ---------------- PHASE B: 27B mot minh ---------------------------------
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    if tok is None:
        tok = tok_t
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    data = json.loads((cdir / "data.json").read_text())

    def _maxlen(it):
        return NK_MAXLEN if it["kind"] == "needle" else TRAIN_MAX

    def enc_item(it):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=_maxlen(it)).to("cuda")
        return enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]

    def enc_cut(it):
        """v3.1: (cache_ids cat tai T-WARM_P, warm = WARM_P token cuoi prompt)."""
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=_maxlen(it)).to("cuda")
        ids = enc["input_ids"]
        return ids[:, :-WARM_P], ids[:, -WARM_P:]

    # --- B0: pseudo-gold cho ifstruct (27B tu doc, greedy GOLD_MAX) ---
    pg_path = cdir / "pseudo_gold.json"
    if not pg_path.exists():
        pg = {}
        with torch.no_grad():
            for split in ("train", "val"):
                for i, it in enumerate(data[split]):
                    if it["kind"] != "ifstruct":
                        continue
                    pre, last = enc_item(it)
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    cur, gen, inp = past, [], last
                    for _ in range(GMAX["ifstruct"]):
                        o = model_t(input_ids=inp, past_key_values=cur,
                                    use_cache=True)
                        cur = o.past_key_values
                        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                        gen.append(int(inp))
                    pg[f"{split}{i}"] = tok.decode(gen)
                    del past, cur
                    torch.cuda.empty_cache()
        pg_path.write_text(json.dumps(pg))
        print(f"B0: {len(pg)} pseudo-gold")
    pg = json.loads(pg_path.read_text())
    for split in ("train", "val"):
        for i, it in enumerate(data[split]):
            if it["kind"] == "ifstruct":
                it["gold"] = pg.get(f"{split}{i}", "")

    # --- mapper + dense hooks (nguyen khuon e6) ---
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
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    if args.skip_train:
        mapper.load(args.out)
    elif Path(args.out + ".last").exists():
        mapper.load(args.out + ".last")
        print("RESUME tu checkpoint .last")

    captured = []
    cap_on = {"v": False}   # hook chi ghi khi bat — val/teacher-ctx-prefill
    hooks = []              # tung nhet 16x160MB/luot vao captured -> OOM @VAL
    def _hook(mod, inp, out):
        if cap_on["v"]:
            captured.append(out[0] if isinstance(out, tuple) else out)
    for name, mod in model_t.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls and "Linear" not in cls and hasattr(mod, "o_proj"):
            hooks.append(mod.register_forward_hook(_hook))

    def student_forward(it, sid, feed):
        src = e5.load_cache(cdir / f"{sid}.pt")
        pre, _ = enc_item(it)
        with torch.no_grad():
            tpl = model_t(input_ids=pre, use_cache=True,
                          logits_to_keep=1).past_key_values
        past = e5.build_student_past(tpl, src, mapper)
        del src
        return model_t(input_ids=feed, past_key_values=past, use_cache=True), tpl

    def run_val(tag, split="val", limit=None):
        """Validation CHUC NANG qua duong mapper (split='train' = kiem
        thuoc-long: greedy tren chinh mau da train phai ra dap an)."""
        import statistics
        sc = {"bfcl": [], "needle": [], "ifstruct": [], "pbtable": []}
        with torch.no_grad():
            for i, it in enumerate(data[split][:limit]):
                sid = f"{split}{i}"
                cut, warm = enc_cut(it)
                src = e5.load_cache(cdir / f"{sid}.pt")
                tpl = model_t(input_ids=cut, use_cache=True,
                              logits_to_keep=1).past_key_values
                past = e5.build_student_past(tpl, src, mapper)
                del src, tpl
                o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen = [int(inp)]
                for _ in range(GEN_LEN.get(it["kind"], GOLD_MAX) - 1):
                    o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                txt = tok.decode(gen)
                if it["kind"] == "bfcl":
                    sc["bfcl"].append(int(it["fn"] in txt))
                elif it["kind"] == "needle":
                    sc["needle"].append(int(it["code"] in re.sub(r"\D", "", txt)))
                elif it["kind"] == "ifstruct":
                    sc["ifstruct"].append(ifstruct_valid(txt, it["spec"]))
                else:
                    gold_head = re.sub(r"\s+", " ", it["gold"])[:60]
                    sc["pbtable"].append(int(gold_head[:30] in
                                             re.sub(r"\s+", " ", txt)))
                del past, cur
                torch.cuda.empty_cache()
        out = {k: f"{sum(v)}/{len(v)}" for k, v in sc.items() if v}
        score = sum(sum(v) for v in sc.values())
        print(f"VAL[{tag}]: {out} -> score {score}")
        return score, out

    if args.train_check:
        mapper.load(args.out + ".last")
        print("TRAIN-CHECK: mapper .last (CE 0.008)")
        score, detail = run_val("train-check", split="train", limit=30)
        results["train_check"] = detail
        save_results()
        print("E6V3_TRAINCHECK_DONE")
        return

    if not args.skip_train:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(mapper.params, lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        n_train = len(data["train"])
        best, stale = -1, 0
        ce_hist = []
        t0 = time.time()
        for step in range(args.steps):
            gc.collect(); torch.cuda.empty_cache()
            # VAL o DAU vong lap: khong bi item-skip (continue) nuot moc nua
            if step % VAL_EVERY == VAL_EVERY - 1:
                score, detail = run_val(f"step{step}")
                results["val_curve"].append({"step": step, "score": score,
                                             **detail})
                save_results()
                if score > best:
                    best, stale = score, 0
                    torch.save(mapper.state_dict(), args.out)
                    print(f"  best-by-val {best} -> saved")
                else:
                    stale += 1
                    if stale >= 3:
                        print("VAL dung im 3 moc — DUNG SOM (ky luat E8)")
                        break
            if step % 50 == 49:   # day an toan: checkpoint bat ke val
                torch.save(mapper.state_dict(), args.out + ".last")
            it = data["train"][step % n_train]
            sid = f"train{step % n_train}"
            if not it.get("gold"):
                continue
            cut, warm = enc_cut(it)
            gm = GMAX.get(it["kind"], GOLD_MAX)
            gold_ids = tok(it["gold"], add_special_tokens=False,
                           return_tensors="pt")["input_ids"][:, :gm].to("cuda")
            if gold_ids.shape[1] < 2:
                continue
            feed = torch.cat([warm, gold_ids[:, :-1]], 1)
            with torch.no_grad():
                import copy as _c
                tch_past = model_t(input_ids=cut, use_cache=True,
                                   logits_to_keep=1).past_key_values
                captured.clear()
                cap_on["v"] = True
                tch_ext = _c.deepcopy(tch_past)
                tch_logp = torch.log_softmax(
                    model_t(input_ids=feed, past_key_values=tch_ext,
                            use_cache=True).logits[:, WARM_P - 1:].float(), -1)
                cap_on["v"] = False
                tch_caps = [c.detach() for c in captured]
                del tch_ext
                torch.cuda.empty_cache()
            src = e5.load_cache(cdir / f"{sid}.pt")
            student_past = e5.build_student_past(tch_past, src, mapper)
            lam = max(0.0, 1.0 - step / (0.2 * args.steps))
            aux = e5.aux_mse(student_past, tch_past)
            captured.clear()
            cap_on["v"] = True
            out = model_t(input_ids=feed, past_key_values=student_past,
                          use_cache=True)
            cap_on["v"] = False
            stu_caps = list(captured)
            logp = torch.log_softmax(out.logits[:, WARM_P - 1:].float(), -1)
            nll = -logp.gather(2, gold_ids.unsqueeze(-1)).squeeze(-1)
            wts = torch.ones_like(nll)
            wts[:, 0] = FIRST_W          # token quyet dinh — trong so x3
            ce = (nll * wts).sum() / wts.sum()
            kl = F.kl_div(logp, tch_logp, log_target=True, reduction="batchmean")
            dense = sum((s.float() - t.float()).pow(2).mean()
                        / (t.float().pow(2).mean() + 1e-6)
                        for s, t in zip(stu_caps, tch_caps)) / max(len(tch_caps), 1)
            loss = ce + BETA * kl + lam * aux + GAMMA * dense
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            cev, klv = float(ce), float(kl)
            del (student_past, out, logp, tch_logp, src, tch_past, ce, kl,
                 dense, loss, stu_caps, tch_caps)
            captured.clear()
            torch.cuda.empty_cache()
            ce_hist.append(cev)
            if step % 25 == 0:
                print(f"step {step}/{args.steps} CE {cev:.3f} KL {klv:.3f} "
                      f"lam {lam:.2f} ({time.time()-t0:.0f}s)")
            if step > 150 and len(ce_hist) >= 50 \
                    and sum(ce_hist[-50:]) / 50 < CE_FLOOR:
                print(f"CE trung binh 50 buoc < {CE_FLOOR} — nguy co overfit "
                      "(quy tac Unsloth, user chot) -> chay VAL roi DUNG")
                score, detail = run_val(f"cefloor-step{step}")
                results["val_curve"].append({"step": step, "score": score,
                                             "stop": "ce_floor", **detail})
                save_results()
                if score > best:
                    best = score
                    torch.save(mapper.state_dict(), args.out)
                break
        print("TRAIN_DONE")
        mapper.load(args.out)

    # ---------------- TEST NIEM PHONG (mo dung 1 lan) ------------------------
    for h in hooks:
        h.remove()
    test_res = {"bfcl": {"self": 0, "mapped": 0, "no_ctx": 0, "n": 0},
                "needle2k": {"self": 0, "mapped": 0, "no_ctx": 0, "n": 0}}
    with torch.no_grad():
        for i, it in enumerate(data["test"]):
            key = "bfcl" if it["kind"] == "bfcl" else "needle2k"
            test_res[key]["n"] += 1
            pre, last = enc_item(it)
            for cond in ("self", "mapped", "no_ctx"):
                if cond == "self":
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    cur, gen, inp = past, [], last
                elif cond == "no_ctx":
                    past = model_t(input_ids=last, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    cur, gen, inp = past, [], last
                else:   # v3.1: warm conv bang WARM_P token cuoi prompt
                    cut, warm = enc_cut(it)
                    src = e5.load_cache(cdir / f"test{i}.pt")
                    tpl = model_t(input_ids=cut, use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = e5.build_student_past(tpl, src, mapper)
                    del src, tpl
                    o = model_t(input_ids=warm, past_key_values=past,
                                use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen = [int(inp)]
                for _ in range(24 - len(gen)):
                    o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                txt = tok.decode(gen)
                hit = (int(it["fn"] in txt) if it["kind"] == "bfcl"
                       else int(it["code"] in re.sub(r"\D", "", txt)))
                test_res[key][cond] += hit
                del past, cur
                torch.cuda.empty_cache()
            print(f"TEST {i+1}/{len(data['test'])}: {test_res}")
    results["test"] = test_res
    save_results()
    print("===== E6V3 KET QUA NIEM PHONG =====")
    print(json.dumps(test_res, indent=1))
    print("E6V3_DONE")


if __name__ == "__main__":
    main()
