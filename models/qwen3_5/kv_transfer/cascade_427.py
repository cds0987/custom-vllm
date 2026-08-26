"""cascade_427 — dong goi CASCADE 4B -> mapper v3.4 -> 27B thanh cong cu 1 lenh
(buoc 2 user duyet 2026-08-26). Muc dich: tai san da chung minh (BFCL 18/20,
needle <=4K tuyet doi) tro thanh thu SO duoc + bang so kinh te TTFT.

Duong chay cho MOI prompt (cong thuc v3.1/v3.4 da tai lap):
  ids = tok(prompt); cut = ids[:-5]; warm = ids[-5:]
  src  = 4B.prefill(cut)                       (chunked 1024)
  tpl  = template-XUONG 27B cho do dai T=len(cut)  (khong prefill 27B!)
  past = build_student_past(tpl, src, mapper)  (attn: WK/WV+RoPE swap;
                                                GDN: alpha/A/B; conv zero)
  27B.forward(warm, past) -> greedy N token     (5 token warm dung lai conv)
So voi baseline 27B tu prefill(cut) -> cung warm -> greedy.

Template-xuong: v3.4 tpl-check 8/8 dung tung bit khi meta lay tu prefill
that cung do dai. O day meta duoc TONG HOP tu probe 3 token (thay moi int/
shape == 3 bang T) — mode --check doi chieu logits voi template prefill that
tren vai prompt TRUOC khi tin (luat do-hon-suy-luan).

VRAM L4 (e5 docstring): bnb-4bit khong luong tu embed/lm_head — 27B ~18GB,
4B ~3.5GB. Mode:
  two-phase   (mac dinh, an toan): 4B mot minh ghi cache ra dia -> xa ->
              27B mot minh doc. TTFT cascade = t_4B + t_map + t_warm (do
              tung khau, cong lai — dung cho bang kinh te, khong lai).
  co-resident (thu nghiem): ca hai cung GPU — neu vua thi cascade song
              that trong 1 process; OOM thi bao, khong doan.

Chay: python cascade_427.py --prompts prompts.json [--mode two-phase]
                            [--check 2] [--max-new 32] [--hf-up]
prompts.json: [{"prompt": "...", "expect": "chuoi/keyword (tuy chon)"}]
"""

import argparse
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

WARM_P = 5
HF_REPO = "gunnybd01/qwen35-kv-mapper-4b-27b"
MAPPER_IN_REPO = "v34/mapper_v34.pt"


def load_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    for p in (Path(__file__).resolve().parents[3] / ".env",
              Path("/content/custom-vllm/.env")):
        try:
            for l in p.read_text().splitlines():
                if l.strip().startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = l.split("=", 1)[1].strip()
                    return os.environ["HF_TOKEN"]
        except OSError:
            continue
    return None


def fetch_mapper(local):
    if Path(local).exists():
        return local
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, MAPPER_IN_REPO, token=load_token())
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(p, local)
    return local


def gen_prompts(n, path, ctxs=(1500, 2000, 4000)):
    """Bo prompt tu sinh (khong can dataset): needle ma 6 so + fact ngu
    nghia paraphrase, filler tong hop, expect = ma/keyword."""
    import random
    out = []
    for j in range(n):
        rng = random.Random(4270 + j)
        ctx = ctxs[j % len(ctxs)]
        nw = int(ctx / 2.5)
        words = [f"w{rng.randint(0, 9999)}" for _ in range(nw)]
        half = len(words) // 2
        if j % 2 == 0:
            code = "".join(rng.choice("0123456789") for _ in range(6))
            name = f"PRJ{rng.randint(100, 999)}"
            mid = chr(10) + f"IMPORTANT: The secret code for project {name} is {code}." + chr(10)
            tail = (chr(10) + f"Question: What is the secret code for project {name}?"
                    + chr(10) + "Answer: The secret code is ")
            exp = code
        else:
            who = rng.choice(["Doctor Vance", "Captain Reyes", "Curator Lam"])
            obj = rng.choice(["telescope", "manuscript", "compass", "violin"])
            pl = rng.choice(["attic", "basement", "chapel", "lighthouse"])
            mid = (chr(10) + f"For many years {who} kept the {obj} hidden away in the "
                   f"{pl}, and almost nobody ever knew about it." + chr(10))
            tail = (chr(10) + f"Question: Where did {who} hide the {obj}?"
                    + chr(10) + "Answer: It was hidden in the ")
            exp = pl
        out.append({"prompt": " ".join(words[:half]) + mid + " ".join(words[half:]) + tail,
                    "expect": exp, "ctx": ctx})
    json.dump(out, open(path, "w"))
    print(f"gen: {n} prompts -> {path}")


def synth_meta(probe, T):
    """Meta template cho do dai T tu probe (3 token): moi int == 3 va moi
    chieu shape == 3 cua attn K/V -> T. GDN shape khong phu thuoc T."""
    meta = e5.cache_meta(probe)
    P = 3
    meta["cache_ints"] = {k: (T if v == P else v)
                          for k, v in meta["cache_ints"].items()}
    for m in meta["layers"]:
        m["ints"] = {k: (T if v == P else v) for k, v in m["ints"].items()}
        if m["kind"] == "a":
            for key in ("k", "v"):
                sh, dt = m[key]
                m[key] = (tuple(T if d == P else d for d in sh), dt)
    return meta


def load_src(name, quant):
    import torch
    if quant == "4bit":
        return e5.load_4bit(name)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return tok, m


def greedy(model, past, inp, n):
    import torch
    cur, out = past, []
    with torch.no_grad():
        o = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(int(nxt))
        for _ in range(n - 1):
            o = model(input_ids=nxt, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
            out.append(int(nxt))
    return out, o.logits[:, -1, :]


def sync_time():
    import torch
    torch.cuda.synchronize()
    return time.time()


def main():
    import torch
    from transformers import AutoConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gen", type=int, default=0,
                    help="tu sinh N prompt (needle+fact, ctx 1500/2000/4000) "
                         "ra --prompts roi chay")
    ap.add_argument("--mode", choices=["two-phase", "co-resident"],
                    default="two-phase")
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--src-quant", choices=["bf16", "4bit"], default="bf16",
                    help="two-phase: bf16 (dung dieu kien train v3.4); "
                         "co-resident tu ep 4bit")
    ap.add_argument("--mapper", default="/content/mapper_v34.pt")
    ap.add_argument("--spill", default="/content/cascade_src")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--check", type=int, default=0,
                    help="doi chieu template-xuong tong hop vs prefill that "
                         "tren N prompt dau (logits token dau)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--results", default="/content/logs/cascade427.json")
    ap.add_argument("--hf-up", action="store_true")
    args = ap.parse_args()
    if args.mode == "co-resident":
        args.src_quant = "4bit"

    if args.gen:
        gen_prompts(args.gen, args.prompts)
    prompts = json.load(open(args.prompts))
    spill = Path(args.spill)
    spill.mkdir(parents=True, exist_ok=True)
    res = {"mode": args.mode, "src_quant": args.src_quant, "runs": []}

    # ---------- PHA 1: 4B prefill (ghi dia o two-phase) ----------
    tok_s, model_s = load_src(args.src_model, args.src_quant)
    t_src = []
    for i, p in enumerate(prompts):
        ids = tok_s(p["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut = ids[:, :-WARM_P]
        t0 = sync_time()
        past = e5.prefill_chunked(model_s, cut)
        dt = sync_time() - t0
        t_src.append(dt)
        if args.mode == "two-phase":
            e5.spill_cache(past, spill / f"src{i}.pt")
            del past
            torch.cuda.empty_cache()
        else:
            p["_src"] = past
        print(f"4B prefill {i} T={cut.shape[1]} {dt:.2f}s")
    if args.mode == "two-phase":
        del model_s
        gc.collect(); torch.cuda.empty_cache()
    src_vram = torch.cuda.max_memory_allocated() / 2**30

    # ---------- PHA 2: 27B + mapper ----------
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    assert len(tok_t) == len(tok_s), "tokenizer 4B/27B lech vocab"
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    src0 = (e5.load_cache(spill / "src0.pt") if args.mode == "two-phase"
            else prompts[0]["_src"])
    a_s, g_s = e5.split_layers(src0)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    if args.mode == "two-phase":
        del src0
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.load(fetch_mapper(args.mapper))
    print(f"mapper v34 nap: attn {len(a_t)} lop, gdn {len(g_t)} lop, "
          f"Hs={Hs} Ht={Ht} attn_dim={attn_dim}")

    def cascade(i, p, check=False):
        ids = tok_t(p["prompt"], return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        T = cut.shape[1]
        src = (e5.load_cache(spill / f"src{i}.pt") if args.mode == "two-phase"
               else p["_src"])
        with torch.no_grad():
            t0 = sync_time()
            tpl = e5.build_template_from_meta(probe, synth_meta(probe, T))
            past = e5.build_student_past(tpl, src, mapper)
            t_map = sync_time() - t0
            t0 = sync_time()
            gen, logit = greedy(model_t, past, warm, args.max_new)
            t_gen = sync_time() - t0
            # TTFT cascade = warm forward (token dau) — tach rieng
            t0 = sync_time()
            o = model_t(input_ids=warm,
                        past_key_values=e5.build_student_past(
                            e5.build_template_from_meta(probe, synth_meta(probe, T)),
                            src, mapper), use_cache=True)
            t_warm = sync_time() - t0
            row = {"i": i, "T": T, "t_4b": round(t_src[i], 3),
                   "t_map": round(t_map, 3), "t_warm": round(t_warm, 3),
                   "ttft_cascade": round(t_src[i] + t_map + t_warm, 3),
                   "t_gen": round(t_gen, 3), "text": tok_t.decode(gen)}
            if check:
                tpl_real = e5.prefill_chunked(model_t, cut)
                past_r = e5.build_student_past(tpl_real, src, mapper)
                o_r = model_t(input_ids=warm, past_key_values=past_r, use_cache=True)
                d = (o.logits[:, -1, :].float() - o_r.logits[:, -1, :].float()).abs().max()
                row["check_max_dlogit"] = float(d)
                row["check_same_argmax"] = bool(
                    o.logits[:, -1, :].argmax() == o_r.logits[:, -1, :].argmax())
                del tpl_real, past_r, o_r
            del tpl, past, o
        if args.mode == "two-phase":
            del src
        torch.cuda.empty_cache()
        return row, cut, warm

    for i, p in enumerate(prompts):
        row, cut, warm = cascade(i, p, check=(i < args.check))
        if not args.no_baseline:
            with torch.no_grad():
                t0 = sync_time()
                past_b = e5.prefill_chunked(model_t, cut)
                t_pre = sync_time() - t0
                t0 = sync_time()
                gen_b, _ = greedy(model_t, past_b, warm, args.max_new)
                t_gb = sync_time() - t0
                del past_b
                torch.cuda.empty_cache()
            row.update({"t_27b_prefill": round(t_pre, 3),
                        "ttft_self": round(t_pre + t_gb / max(args.max_new, 1), 3),
                        "text_self": tok_t.decode(gen_b),
                        "speedup_ttft": round((t_pre) / max(row["ttft_cascade"], 1e-6), 2)})
        if p.get("expect"):
            row["hit"] = int(p["expect"].lower() in row["text"].lower())
            if "text_self" in row:
                row["hit_self"] = int(p["expect"].lower() in row["text_self"].lower())
        res["runs"].append(row)
        print(f"CASCADE {i} T={row['T']} ttft {row['ttft_cascade']}s"
              + (f" | self prefill {row['t_27b_prefill']}s x{row['speedup_ttft']}"
                 if "t_27b_prefill" in row else "")
              + (f" | hit {row.get('hit')}/{row.get('hit_self')}" if "hit" in row else "")
              + (f" | check dlogit {row['check_max_dlogit']:.4f} same={row['check_same_argmax']}"
                 if "check_max_dlogit" in row else "")
              + f" | {row['text'][:50]!r}")

    res["vram_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["vram_src_phase_gib"] = round(src_vram, 2)
    runs = res["runs"]
    if runs:
        agg = {"n": len(runs),
               "ttft_cascade_mean": round(sum(r["ttft_cascade"] for r in runs) / len(runs), 3)}
        if "t_27b_prefill" in runs[0]:
            agg["t_27b_prefill_mean"] = round(sum(r["t_27b_prefill"] for r in runs) / len(runs), 3)
            agg["speedup_ttft_mean"] = round(agg["t_27b_prefill_mean"] / agg["ttft_cascade_mean"], 2)
        if any("hit" in r for r in runs):
            agg["hit"] = f"{sum(r.get('hit', 0) for r in runs)}/{sum('hit' in r for r in runs)}"
            agg["hit_self"] = f"{sum(r.get('hit_self', 0) for r in runs)}/{sum('hit_self' in r for r in runs)}"
        if any("check_same_argmax" in r for r in runs):
            agg["check_same"] = f"{sum(r.get('check_same_argmax', 0) for r in runs)}/{sum('check_same_argmax' in r for r in runs)}"
            agg["check_max_dlogit"] = max(r.get("check_max_dlogit", 0) for r in runs)
        res["agg"] = agg
        print("===== CASCADE427 AGG =====")
        print(json.dumps(agg, indent=1))
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.results, "w"), indent=1)
    if args.hf_up and load_token():
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=args.results,
                            path_in_repo="cascade427/" + Path(args.results).name,
                            repo_id=HF_REPO)
        print("HF-UP cascade427")
    print("CASCADE427_DONE")


if __name__ == "__main__":
    main()
