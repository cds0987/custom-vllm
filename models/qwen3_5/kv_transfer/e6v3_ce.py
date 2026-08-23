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

TRAIN_MAX = 1024     # ctx token toi da khi train (BFCL ~800-900 la vua)
GOLD_MAX = 64
CONV_WARM = e5.CONV_WARM
BETA = 0.3           # trong so KL phu
CE_FLOOR = 0.2       # Unsloth: train loss <0,2 = overfit -> dung (user chot)
GAMMA = 0.05         # dense supervision
N_NEEDLE_TRAIN = 200
VAL_EVERY = 150   # val 50 mau ~7 phut/lan — 100 la qua day
SEED = 7


# ------------------------------ data ----------------------------------------

def bfcl_load(fname, n):
    from datasets import load_dataset
    ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                      data_files=fname, split="train")
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
    items = []
    for i in range(n):
        rng = random.Random(seed0 + i)
        name = rng.choice(e2.NAMES) + str(rng.randint(0, 99))
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = e2.token_stream(tok, ctx_tok, seed=seed0 + i)
        half = ctx_tok // 2
        prompt = (tok.decode(ids[:half])
                  + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                  + tok.decode(ids[half:]) + "\n" + e2.build_q(name) + " ")
        items.append({"kind": "needle", "prompt": prompt, "gold": code + ".",
                      "code": code})
    return items


def build_data(tok=None):
    """Return dict(train, val, test). tok=None -> bo needle (dry-data)."""
    rng = random.Random(SEED)
    exec_simple = bfcl_load("BFCL_v3_exec_simple.json", 100)
    test = exec_simple[:20]                      # NIEM PHONG — y het E6
    train = exec_simple[20:]
    simple = bfcl_load("BFCL_v3_simple.json", 220)
    test_prompts = {it["prompt"] for it in test}
    simple = [it for it in simple if it["prompt"] not in test_prompts]
    rng.shuffle(simple)
    val = simple[:15]
    train += simple[15:]
    ifs = ifstruct_load(75)
    val += ifs[:15]
    train += ifs[15:]
    pbt = pbtable_load(50)
    val += pbt[:10]
    train += pbt[10:]
    if tok is not None:
        train += needle_items(tok, N_NEEDLE_TRAIN, 30000)
        val += needle_items(tok, 10, 31000)
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
                    enc = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                                max_length=TRAIN_MAX).to("cuda")
                    past = model_s(input_ids=enc["input_ids"][:, :-1],
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

    def enc_item(it):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=TRAIN_MAX).to("cuda")
        return enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]

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
                    for _ in range(GOLD_MAX):
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
    Ht = next(iter(g_t.values())).recurrent_states.shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    if args.skip_train:
        mapper.load(args.out)

    captured = []
    hooks = []
    def _hook(mod, inp, out):
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

    def run_val(tag):
        """Validation CHUC NANG qua duong mapper."""
        import statistics
        sc = {"bfcl": [], "needle": [], "ifstruct": [], "pbtable": []}
        with torch.no_grad():
            for i, it in enumerate(data["val"]):
                sid = f"val{i}"
                pre, last = enc_item(it)
                src = e5.load_cache(cdir / f"{sid}.pt")
                tpl = model_t(input_ids=pre, use_cache=True,
                              logits_to_keep=1).past_key_values
                past = e5.build_student_past(tpl, src, mapper)
                del src, tpl
                cur, gen, inp = past, [], last
                for _ in range(GOLD_MAX):
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
            it = data["train"][step % n_train]
            sid = f"train{step % n_train}"
            if not it.get("gold"):
                continue
            pre, last = enc_item(it)
            gold_ids = tok(it["gold"], add_special_tokens=False,
                           return_tensors="pt")["input_ids"][:, :GOLD_MAX].to("cuda")
            if gold_ids.shape[1] <= CONV_WARM + 1:
                continue
            feed = torch.cat([last, gold_ids[:, :-1]], 1)
            with torch.no_grad():
                import copy as _c
                tch_past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                captured.clear()
                tch_ext = _c.deepcopy(tch_past)
                tch_logp = torch.log_softmax(
                    model_t(input_ids=feed, past_key_values=tch_ext,
                            use_cache=True).logits[:, CONV_WARM:].float(), -1)
                tch_caps = [c.detach() for c in captured]
                del tch_ext
                torch.cuda.empty_cache()
            src = e5.load_cache(cdir / f"{sid}.pt")
            student_past = e5.build_student_past(tch_past, src, mapper)
            lam = max(0.0, 1.0 - step / (0.2 * args.steps))
            aux = e5.aux_mse(student_past, tch_past)
            captured.clear()
            out = model_t(input_ids=feed, past_key_values=student_past,
                          use_cache=True)
            stu_caps = list(captured)
            logp = torch.log_softmax(out.logits[:, CONV_WARM:].float(), -1)
            ce = -logp.gather(2, gold_ids[:, CONV_WARM:].unsqueeze(-1)).mean()
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
                elif cond == "no_ctx":
                    past = model_t(input_ids=last, use_cache=True,
                                   logits_to_keep=1).past_key_values
                else:
                    src = e5.load_cache(cdir / f"test{i}.pt")
                    tpl = model_t(input_ids=pre, use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = e5.build_student_past(tpl, src, mapper)
                    del src, tpl
                cur, gen, inp = past, [], last
                for _ in range(24):
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
