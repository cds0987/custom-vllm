"""E6 — train mapper 4B->27B to convergence + score on REAL benchmarks.

User order 2026-08-15: measure true performance at training convergence on
real suites — BFCL (function calling), LiquidAI/ifstruct-v1.0, llamaindex/
ParseBench. Success = the prefill bottleneck is solvable with quality assured.

Upgrades over e5 v1 (each targets a recorded suspect, STATUS mục E5):
  1. steps 2000 + cosine LR         (suspect 1: undertrained, KL ~1/token)
  2. needle-aware training mix      (suspect 2: plain prose never exercises
     40% of samples embed a fact     the retrieval directions)
     and the suffix asks for it
  3. dense per-layer supervision    (suspect 3: end-logits signal too sparse;
     MSE on the 16 full-attention    hooks capture teacher/student attention
     block outputs, weight 0.05)     outputs over the suffix)
  4. (two-hop 4B->9B->27B left for v3 if this fails)

Benchmark scoring (schema-agnostic, no per-suite harness): per item,
teacher-forced NLL/token of the GOLD output given the prompt-cache, under
  self (27B prefills) / mapped (4B cache + trained mapper) / no_ctx (floor)
plus greedy exact-prefix hit for BFCL (does the call name the right function).
BONUS stage: the same benchmark items through the PROVEN 4B->9B raw-copy path
(9B bnb) — the "quality when infra upgrades" number for the shipping pair.

Phases (models never co-resident; reuses e5's spill/load/Mapper):
  A: 4B alone — spill train caches + benchmark prompt caches
  B: 27B alone — train to convergence, then benchmark eval
  C: 9B alone — benchmark eval of raw copy
"""

import argparse
import gc
import importlib.util
import json
import random
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)
e2 = e5.e2

L_CTX, T2, CONV_WARM = e5.L_CTX, e5.T2, e5.CONV_WARM
NEEDLE_FRAC = 0.4
DENSE_W = 0.05
BENCH_N = 20          # items per suite
GOLD_MAX = 96         # gold tokens scored


# ------------------------------ training data -------------------------------

UNIQUE = 600   # unique cached samples; steps cycle over them (disk: 2000
               # spills = 112GB filled the disk — postmortem 2026-08-15)


def make_train_ids(tok, stream, idx):
    """L_CTX token ids for sample idx (deterministic per idx); 40% embed a
    fact whose answer sits in the suffix."""
    rng = random.Random(1000 + idx)
    base = stream[idx * L_CTX:(idx + 1) * L_CTX]
    if rng.random() >= NEEDLE_FRAC:
        return base
    name = rng.choice(e2.NAMES) + str(rng.randint(0, 99))
    code = "".join(rng.choice("0123456789") for _ in range(6))
    qa = tok(f"\n{e2.build_q(name)} {code}.", add_special_tokens=False)["input_ids"]
    fact = tok(f"\nIMPORTANT: The secret code for project {name} is {code}.\n",
               add_special_tokens=False)["input_ids"]
    room = L_CTX - len(qa) - len(fact)
    cut = int(room * rng.uniform(0.2, 0.8))
    return (base[:cut] + fact + base[cut:room] + qa)[:L_CTX]


# ------------------------------ benchmarks ----------------------------------

def _pick(ex, keys):
    for k in keys:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def load_benches():
    """Return {suite: [(prompt, gold, fn_name|None), ...]}. Robust to schema."""
    from datasets import load_dataset
    out = {}
    try:
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                          data_files="BFCL_v3_exec_simple.json", split="train")
        items = []
        for ex in ds:
            q = ex.get("question")
            if isinstance(q, list):    # BFCL nests [[{role,content}]]
                try:
                    q = q[0][0]["content"]
                except Exception:
                    q = str(q)
            fns = ex.get("function") or []
            if isinstance(fns, dict):
                fns = [fns]
            fname = fns[0].get("name") if fns else None
            if not q or not fname:
                continue
            prompt = ("You can call these functions:\n"
                      + json.dumps(fns)[:3000]
                      + f"\nUser request: {q}\nRespond with one function call.\nCall: ")
            items.append((prompt, fname + "(", fname))
            if len(items) >= BENCH_N:
                break
        out["bfcl"] = items
    except Exception as e:
        print(f"[bench] bfcl SKIPPED: {type(e).__name__}: {e}")
    for suite, repo in (("ifstruct", "LiquidAI/ifstruct-v1.0"),
                        ("parsebench", "llamaindex/ParseBench")):
        try:
            ds = load_dataset(repo, split={"ifstruct": "test",
                                       "parsebench": "table"}[suite])
            items = []
            for ex in ds:
                p = _pick(ex, ["prompt", "question", "instruction", "input",
                               "query", "text", "markdown"])
                g = _pick(ex, ["output", "answer", "response", "completion",
                               "target", "ground_truth", "expected_output",
                               "json", "structured_output"])
                if p and g:
                    items.append((p[:6000], g, None))
                if len(items) >= BENCH_N:
                    break
            if items:
                out[suite] = items
            else:
                print(f"[bench] {suite}: no usable (prompt, gold) fields")
        except Exception as e:
            print(f"[bench] {suite} SKIPPED: {type(e).__name__}: {e}")
    return out


def bench_metrics(model, tok, prompt, gold, fn_name, past, last):
    """NLL/token of gold under the given cache + optional greedy fn-name hit."""
    import torch
    gold_ids = tok(gold, add_special_tokens=False,
                   return_tensors="pt")["input_ids"][:, :GOLD_MAX].to("cuda")
    import copy as _c
    feed = torch.cat([last, gold_ids[:, :-1]], 1)
    out = model(input_ids=feed, past_key_values=_c.deepcopy(past), use_cache=True)
    lp = torch.log_softmax(out.logits.float(), -1)
    nll = float(-lp.gather(2, gold_ids.unsqueeze(-1)).mean())
    hit = None
    if fn_name:
        cur, gen, inp = past, [], last
        for _ in range(24):
            o = model(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
        hit = int(fn_name in tok.decode(gen))
    return nll, hit


def eval_suites(model, tok, benches, cdir, tag, mapper=None, copy_mode=False):
    """Run all suites under self / transplant / no_ctx on the CURRENT model."""
    import torch
    results = {}
    with torch.no_grad():
        for suite, items in benches.items():
            rows = []
            for bi, (prompt, gold, fn_name) in enumerate(items):
                enc = tok(prompt, return_tensors="pt",
                          truncation=True, max_length=2048).to("cuda")
                pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
                row = {}
                for cond in ("self", "xfer", "no_ctx"):
                    if cond == "self":
                        past = model(input_ids=pre, use_cache=True,
                                     logits_to_keep=1).past_key_values
                        l = last
                    elif cond == "no_ctx":
                        past = model(input_ids=last, use_cache=True,
                                     logits_to_keep=1).past_key_values
                        # cache holds just the final prompt token = floor
                        l = last
                    else:
                        src = e5.load_cache(cdir / f"bn_{suite}_{bi}.pt")
                        tpl = model(input_ids=pre, use_cache=True,
                                    logits_to_keep=1).past_key_values
                        if copy_mode:      # 4B->9B raw copy, identical shapes
                            for lt, ls in zip(tpl.layers, src.layers):
                                if hasattr(ls, "keys"):
                                    lt.keys[0] = ls.keys[0]
                                    lt.values[0] = ls.values[0]
                                else:
                                    lt.recurrent_states[0] = ls.recurrent_states[0]
                                    lt.conv_states[0] = ls.conv_states[0]
                            past = tpl
                        else:
                            past = e5.build_student_past(tpl, src, mapper)
                        del src
                        l = last
                    nll, hit = bench_metrics(model, tok, prompt, gold,
                                             fn_name, past, l)
                    row[f"{cond}_nll"] = nll
                    if hit is not None:
                        row[f"{cond}_hit"] = hit
                    del past
                    torch.cuda.empty_cache()
                rows.append(row)
                if (bi + 1) % 5 == 0:
                    print(f"  [{tag}] {suite} {bi+1}/{len(items)}")
            results[suite] = rows
            import statistics as st
            line = f"[{tag}] {suite}: "
            for cond in ("self", "xfer", "no_ctx"):
                line += f"{cond} NLL {st.mean(r[f'{cond}_nll'] for r in rows):.3f} "
                if f"{cond}_hit" in rows[0]:
                    line += f"hit {sum(r[f'{cond}_hit'] for r in rows)}/{len(rows)} "
            print(line)
    return results


# ------------------------------ main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--mid-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/content/mapper_e6.pt")
    ap.add_argument("--cache-dir", default="/content/e6_src")
    ap.add_argument("--results", default="/content/logs/e6_results.json")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig

    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(4)
    benches = load_benches()
    print("suites:", {k: len(v) for k, v in benches.items()})
    marker = cdir / f"DONE_u{UNIQUE}_L{L_CTX}_T{T2}"

    # ---- PHASE A: 4B alone ----
    if not marker.exists():
        tok_s, model_s = e5.load_4bit(args.src_model)
        stream = e2.token_stream(tok_s, UNIQUE * L_CTX + L_CTX, seed=11)
        with torch.no_grad():
            for step in range(UNIQUE):
                ids = make_train_ids(tok_s, stream, step)
                pth = cdir / f"tr{step}.pt"
                if not pth.exists():
                    t = torch.tensor([ids[:-T2]], device="cuda")
                    past = model_s(input_ids=t, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    e5.spill_cache(past, pth)
                    del past
                    torch.cuda.empty_cache()
                if step % 200 == 0:
                    print(f"A {step}/{UNIQUE}")
            for suite, items in benches.items():
                for bi, (prompt, _, _) in enumerate(items):
                    enc = tok_s(prompt, return_tensors="pt",
                                truncation=True, max_length=2048).to("cuda")
                    past = model_s(input_ids=enc["input_ids"][:, :-1],
                                   use_cache=True, logits_to_keep=1).past_key_values
                    e5.spill_cache(past, cdir / f"bn_{suite}_{bi}.pt")
                    del past
                    torch.cuda.empty_cache()
        del model_s
        gc.collect(); torch.cuda.empty_cache()
        marker.touch()
        print("PHASE_A_DONE")

    # ---- PHASE B: 27B — train to convergence + bench eval ----
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    src0 = e5.load_cache(cdir / "tr0.pt")
    a_s, g_s = e5.split_layers(src0)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    attn_dim = (next(iter(a_s.values())).keys.shape[1]
                * next(iter(a_s.values())).keys.shape[3])
    del src0
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = next(iter(g_t.values())).recurrent_states.shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)

    # dense supervision hooks: capture full-attention block outputs
    captured = []
    def _hook(mod, inp, out):
        captured.append(out[0] if isinstance(out, tuple) else out)
    hooks = []
    for name, mod in model_t.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls and "Linear" not in cls and hasattr(mod, "o_proj"):
            hooks.append(mod.register_forward_hook(_hook))
    print(f"dense hooks on {len(hooks)} attention blocks")

    if not args.skip_train:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(mapper.params, lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        stream = e2.token_stream(tok_t, UNIQUE * L_CTX + L_CTX, seed=11)
        t0 = time.time()
        for step in range(args.steps):
            gc.collect(); torch.cuda.empty_cache()
            idx = step % UNIQUE
            ids = make_train_ids(tok_t, stream, idx)
            full = torch.tensor([ids], device="cuda")
            ctx, suffix = full[:, :-T2], full[:, -T2:]
            with torch.no_grad():
                import copy as _c
                tch_past = model_t(input_ids=ctx, use_cache=True,
                                   logits_to_keep=1).past_key_values
                captured.clear()
                tch_ext = _c.deepcopy(tch_past)
                tch_logits = model_t(input_ids=suffix, past_key_values=tch_ext,
                                     use_cache=True).logits
                tch_logp = torch.log_softmax(
                    tch_logits[:, CONV_WARM:].float(), -1)
                tch_caps = [c.detach() for c in captured]
                del tch_ext
                torch.cuda.empty_cache()
            src_past = e5.load_cache(cdir / f"tr{idx}.pt")
            student_past = e5.build_student_past(tch_past, src_past, mapper)
            lam = max(0.0, 1.0 - step / (0.2 * args.steps))
            aux = e5.aux_mse(student_past, tch_past)
            captured.clear()
            out = model_t(input_ids=suffix, past_key_values=student_past,
                          use_cache=True)
            stu_caps = list(captured)
            stu_logp = torch.log_softmax(out.logits[:, CONV_WARM:].float(), -1)
            kl = F.kl_div(stu_logp, tch_logp, log_target=True,
                          reduction="batchmean")
            dense = sum(
                (s.float() - t.float()).pow(2).mean()
                / (t.float().pow(2).mean() + 1e-6)
                for s, t in zip(stu_caps, tch_caps)) / max(len(tch_caps), 1)
            loss = kl + lam * aux + DENSE_W * dense
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            klv, dv = float(kl), float(dense)
            del (student_past, out, stu_logp, tch_logp, src_past, tch_past,
                 kl, dense, loss, stu_caps, tch_caps)
            captured.clear()
            torch.cuda.empty_cache()
            if step % 25 == 0:
                print(f"step {step}/{args.steps} KL {klv:.3f} dense {dv:.3f} "
                      f"lam {lam:.2f} lr {sched.get_last_lr()[0]:.5f} "
                      f"({time.time()-t0:.0f}s)")
            if step % 100 == 99 or step == args.steps - 1:
                torch.save(mapper.state_dict(), args.out)
        print("TRAIN_DONE")
    else:
        mapper.load(args.out)
    for h in hooks:
        h.remove()

    res27 = eval_suites(model_t, tok_t, benches, cdir, "27B-mapped",
                        mapper=mapper)
    del model_t
    gc.collect(); torch.cuda.empty_cache()
    print("PHASE_B_DONE")

    # ---- PHASE C: 9B — raw-copy path on the same suites ----
    tok_m, model_m = e5.load_4bit(args.mid_model)
    res9 = eval_suites(model_m, tok_m, benches, cdir, "9B-copy",
                       copy_mode=True)
    del model_m
    gc.collect(); torch.cuda.empty_cache()

    with open(args.results, "w") as fh:
        json.dump({"27B_mapped": res27, "9B_copy": res9}, fh, indent=1)
    print("saved", args.results)
    print("E6_ALL_DONE")


if __name__ == "__main__":
    main()
