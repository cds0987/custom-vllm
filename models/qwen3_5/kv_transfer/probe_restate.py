"""probe_restate -- BAT CACHE KE LAI DE BAI.

Phan xu giua hai nghi pham con lai sau khi gia thuyet DUNG LUONG GDN bi bac
(GDN 0,8M -> 25,2M, gap 30 lan, gsm8k van 1-3/53 o ca 3 moc):

  (a) SAI DANG HAM  -- thong tin CO trong cache nhung `A.S.B` (song tuyen,
      chi xoay/co gian hai truc doc lap) khong bieu dien noi phep dich
  (b) TUONG          -- ban tom tat GDN cua 4B khong chua cau truc quan he
      o dang 9B doc lai duoc

Cach phan biet: KHONG hoi dap an, ma bat model THUAT LAI DE BAI tu cache,
roi dem xem giu duoc bao nhieu:
    - CON SO     (attention giu token -> ky vong con)
    - TEN RIENG  (thuc the)
    - QUAN HE    (tu khoa nhu "hon/kem", "gap", "moi", "con lai"...)

  giu SO nhung mat QUAN HE  -> (a) thong tin co, mapper khong dich noi
  mat CA SO lan QUAN HE     -> (b) tuong

So truc tiep voi `self` thuat lai CUNG de -> tach duoc "mapper lam mat" khoi
"chinh model thuat lai kem".

Chay:
  python -u probe_restate.py --mapper ... --lora ... --n 20
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
WARM_P = 5

# tu khoa QUAN HE — thu bi pha theo quan sat doc tay 20 dau ra gsm8k
REL_WORDS = [
    "more", "less", "fewer", "than", "twice", "half", "each", "every",
    "total", "remaining", "left", "per", "times", "as many", "older",
    "younger", "before", "after", "then", "first", "second", "third",
    "sold", "bought", "gave", "spent", "cost", "shared", "divided",
]


def facts(txt):
    """Rut ba loai su kien de doi chieu."""
    nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", txt))
    # ten rieng: viet hoa GIUA cau (bo dau cau de khoi dinh chu dau dong)
    body = re.sub(r"(?m)^[^\w]*", " ", txt)
    names = set(re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", body))
    low = txt.lower()
    rels = {w for w in REL_WORDS if w in low}
    return nums, names, rels


def cover(ref, got):
    return len(ref & got) / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", default="/content/train_items.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="/content/logs/probe_restate.json")
    args = ap.parse_args()

    from transformers import AutoConfig
    d = json.loads(Path(args.data).read_text())
    val = list(d["val"])
    random.Random(7).shuffle(val)
    items = [it for it in val if it["kind"] == "gsm8k"][:args.n]
    print(f"bat ke lai de: {len(items)} mau gsm8k", flush=True)

    # ---- PHA A: 4B (+LoRA) -> cache ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    if args.lora:
        from peft import PeftModel
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
    spill = Path("/content/restate_spill")
    spill.mkdir(parents=True, exist_ok=True)

    # PROMPT KE LAI: thay "Solution:" bang yeu cau thuat lai de bai. Phan
    # NGU CANH (de bai) van y het -> cache van la cache cua de bai do.
    def restate_prompt(it):
        p = it["prompt"]
        cut = p.rfind("<think>")
        head = p[:cut] if cut > 0 else p
        return (head + "Restate the problem in your own words, listing every "
                "number and who it belongs to.\n<think>\n\n</think>\n\n"
                "The problem says: ")

    for i, it in enumerate(items):
        ids = tok_s(restate_prompt(it), return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_s, ids[:, :-WARM_P])
        e5.spill_cache(past, spill / f"x{i}.pt")
        del past
        torch.cuda.empty_cache()
    del model_s
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("pha A xong", flush=True)

    # ---- PHA B: 9B -> map + ke lai, va self de doi chieu ----
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    sd = torch.load(args.mapper, map_location="cuda")
    meta = sd.get("_meta", {})
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=meta.get("attn_rank", 0),
                       gdn_per_head=meta.get("gdn_per_head", False))
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)

    @torch.no_grad()
    def greedy(past, warm, n_new=160):
        o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
        cur = o.past_key_values
        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
        g = [int(inp)]
        for _ in range(n_new - 1):
            o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            g.append(int(inp))
            if int(inp) in STOPS:
                break
        del cur, o
        return tok_t.decode(g, skip_special_tokens=True)

    rows, agg = [], {"map": [0, 0, 0], "self": [0, 0, 0]}
    for i, it in enumerate(items):
        rp = restate_prompt(it)
        ids = tok_t(rp, return_tensors="pt", truncation=True,
                    max_length=args.max_len)["input_ids"].to("cuda")
        cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
        src = e5.load_cache(spill / f"x{i}.pt")
        with torch.no_grad():
            tpl = e5.prefill_chunked(model_t, cut)
            st = e5.build_student_past(tpl, src, mapper)
            del tpl, src
        t_map = greedy(st, warm)
        del st
        torch.cuda.empty_cache()
        with torch.no_grad():
            slf = e5.prefill_chunked(model_t, cut)
        t_slf = greedy(slf, warm)
        del slf
        torch.cuda.empty_cache()

        ref = facts(it["prompt"])
        gm, gs = facts(t_map), facts(t_slf)
        cm = [cover(ref[k], gm[k]) for k in range(3)]
        cs = [cover(ref[k], gs[k]) for k in range(3)]
        for k in range(3):
            agg["map"][k] += cm[k]
            agg["self"][k] += cs[k]
        rows.append({"id": it["id"], "restate_mapped": t_map,
                     "restate_self": t_slf,
                     "cover_mapped": {"so": cm[0], "ten": cm[1], "quanhe": cm[2]},
                     "cover_self": {"so": cs[0], "ten": cs[1], "quanhe": cs[2]}})
        print(f"\n== {i+1}/{len(items)} {it['id']}", flush=True)
        print(f"  SELF  : {t_slf[:180]!r}", flush=True)
        print(f"  MAPPED: {t_map[:180]!r}", flush=True)
        print(f"  giu duoc | self so={cs[0]:.2f} ten={cs[1]:.2f} quanhe={cs[2]:.2f}"
              f" | mapped so={cm[0]:.2f} ten={cm[1]:.2f} quanhe={cm[2]:.2f}",
              flush=True)
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)

    n = len(rows)
    print("\n===== TONG HOP: TY LE GIU DUOC KHI KE LAI DE =====")
    print(f"{'':10} {'CON SO':>9} {'TEN RIENG':>11} {'QUAN HE':>9}")
    for k, lbl in (("self", "9B tu doc"), ("map", "qua mapper")):
        a = agg[k]
        print(f"{lbl:10} {100*a[0]/n:8.1f}% {100*a[1]/n:10.1f}% "
              f"{100*a[2]/n:8.1f}%")
    ds = [100 * (agg["map"][k] - agg["self"][k]) / n for k in range(3)]
    print(f"{'chenh':10} {ds[0]:8.1f}  {ds[1]:10.1f}  {ds[2]:8.1f}")
    print("\nDOC KET QUA: giu SO nhung mat QUAN HE -> thong tin CO, mapper "
          "khong dich noi (doi DANG HAM). Mat CA HAI -> tuong.")
    print("RESTATE_EXIT", flush=True)


if __name__ == "__main__":
    main()
