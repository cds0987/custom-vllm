"""inspect_fail -- DOC TAY dau ra hong, thay vi dem them mau.

User 2026-08-29 duyet. Boi canh: sau mot ngay gan nhu chi sua BO MAY DO (3 bug
harness, 2 gia thuyet sai), cau hoi nghien cuu that van chua duoc dung toi --
vi sao gsm8k hong? Val n=150 da du sac de biet HUONG (bfcl/bbh/musr vuot tran;
gsm8k 1/35 trong khi tran 29/35). Them mau chi lam hep khoang tin cay, KHONG
doi ket luan. Chi CO CHE moi quyet dinh phai chua gi.

Bon kieu hong -- bon cach chua khac han nhau:
  1. lap/rac ngay tu token dau        -> dinh luat bien mong, can lop gia co
  2. suy luan mach lac nhung tinh sai -> cache du, mapper meo nhe -> train them
  3. suy luan dung, khong ra "Final Answer:" -> loi dinh dang, va gan mien phi
  4. lac sang de khac                 -> mapper thieu dung luong -> doi lop ham

In CA hai dieu kien tren CUNG mau de doi chieu truc tiep, kem chi so tu dong.

Chay:
  python -u inspect_fail.py --mapper ... --lora ... --n 20
"""

import argparse
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e5 = _load("e5_train")
gd = _load("gen_data")
WARM_P = 5


def trigram_rep(txt):
    w = txt.split()
    if len(w) < 6:
        return 0.0
    t = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(t)) / len(t)


def diag(txt, expect):
    nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
    return {
        "n_tu": len(txt.split()),
        "lap_trigram": round(trigram_rep(txt), 3),
        "co_final_answer": bool(re.search(r"Final Answer", txt)),
        "so_cuoi": nums[-1] if nums else None,
        "dap_an_dung": expect,
        "dap_an_co_xuat_hien": expect in nums,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", default="/content/train_items.json")
    ap.add_argument("--kind", default="gsm8k")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default="/content/logs/inspect_fail.json")
    args = ap.parse_args()

    from transformers import AutoConfig
    d = json.loads(Path(args.data).read_text())
    val = list(d["val"])
    random.Random(7).shuffle(val)
    items = [it for it in val if it["kind"] == args.kind][:args.n]
    print(f"soi {len(items)} mau {args.kind}", flush=True)

    # ---- PHA A: 4B (da merge LoRA) -> cache ra dia ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    if args.lora:
        from peft import PeftModel
        model_s = PeftModel.from_pretrained(model_s, args.lora)
        model_s = model_s.merge_and_unload()
        model_s.eval()
        print(f"LoRA da merge: {args.lora}", flush=True)
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
    spill = Path("/content/insp_spill")
    spill.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        ids = tok_s(it["prompt"], return_tensors="pt", truncation=True,
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

    # ---- PHA B: 9B -> map + decode, va self de doi chieu ----
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    tok_t.truncation_side = "left"
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.load(args.mapper)
    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"mapper {args.mapper} | token dung {sorted(STOPS)}", flush=True)

    @torch.no_grad()
    def greedy(past, warm, n_new=320):
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

    rows = []
    for i, it in enumerate(items):
        ids = tok_t(it["prompt"], return_tensors="pt", truncation=True,
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
        r = {"id": it["id"], "expect": it["expect"],
             "mapped": t_map, "self": t_slf,
             "hit_mapped": gd.score_item(it, t_map),
             "hit_self": gd.score_item(it, t_slf),
             "chan_doan_mapped": diag(t_map, it["expect"]),
             "chan_doan_self": diag(t_slf, it["expect"])}
        rows.append(r)
        print(f"\n===== {i+1}/{len(items)}  id={it['id']}  "
              f"dap an dung = {it['expect']}", flush=True)
        print(f"  [SELF   hit={r['hit_self']}] {t_slf[:260]!r}", flush=True)
        print(f"  [MAPPED hit={r['hit_mapped']}] {t_map[:260]!r}", flush=True)
        print(f"  chan doan mapped: {r['chan_doan_mapped']}", flush=True)
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)

    print("\n===== TONG HOP =====")
    print(f"self dung   : {sum(r['hit_self'] for r in rows)}/{len(rows)}")
    print(f"mapped dung : {sum(r['hit_mapped'] for r in rows)}/{len(rows)}")
    bad = [r for r in rows if r["hit_self"] and not r["hit_mapped"]]
    print(f"self dung -> mapped sai: {len(bad)} ca")
    c = Counter()
    for r in bad:
        m = r["chan_doan_mapped"]
        if m["lap_trigram"] > 0.25:
            c["1_lap_rac"] += 1
        elif not m["co_final_answer"]:
            c["3_khong_ra_dinh_dang"] += 1
        elif m["dap_an_co_xuat_hien"]:
            c["3b_co_dap_an_nhung_cham_truot"] += 1
        else:
            c["2_hoac_4_mach_lac_nhung_sai"] += 1
    print("phan loai:", dict(c))
    print("INSPECT_EXIT", flush=True)


if __name__ == "__main__":
    main()
