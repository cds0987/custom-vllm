"""probe_batch -- CO GOM LO DUOC KHONG, va gom thi duoc gi.

User 2026-08-29: "batch = 1 la ko duoc phai tang batch + gradient accumulation
de tang do general pattern capture va speedup".

Dung. Nhung batch that su doi hoi DEM prompt cho bang do dai, ma Qwen3.5 co
~75% lop GDN = lop HOI QUY: trang thai duoc quet tuan tu qua tung token. Neu
vong quet khong ton trong attention_mask thi token DEM bi nuot vao trang thai
nho, cache sinh ra SAI -- va khong co loi nao duoc nem ra. Do dung la thu
mapper hoc de dich. Khong duoc gia dinh, phai do.

BA CAU HOI, mot lan nap model:

  A. AN TOAN -- prefill mot minh vs prefill trong lo co DEM: recurrent_states
     va KV co trung khong? Trung -> dem vo hai. Lech -> PHAI gom lo theo do
     dai (cung do dai = khong dem = dong nhat ve mat toan hoc).
  B. VRAM   -- batch 1/2/4/8 x ctx 1024/2048/4096 co vua 22,5GB khong.
  C. TOC DO -- giay/mau thuc te. O T=4096 GPU co the DA bao hoa, luc do gom lo
     gan nhu khong nhanh len; o T ngan (bfcl ~500, needle 700) thi moi lai.
     Day la cau hoi DO duoc, khong phai cau hoi suy luan duoc.

Chay:
  python -u probe_batch.py --src-model Qwen/Qwen3.5-4B
"""

import argparse
import importlib.util
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


def gib():
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--ctxs", default="1024,2048,4096")
    ap.add_argument("--batches", default="1,2,4,8")
    args = ap.parse_args()

    tok, model = e5.load_4bit(args.src_model)
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"        # cau hoi nam O CUOI prompt

    # ---------------------------------------------------------------- A ----
    print("\n===== A. DEM TRAI CO LAM HONG TRANG THAI GDN KHONG =====",
          flush=True)
    torch.manual_seed(0)
    long_ids = torch.randint(1000, 50000, (1, 600), device="cuda")
    short_ids = torch.randint(1000, 50000, (1, 400), device="cuda")

    def solo(ids):
        with torch.no_grad():
            return model(input_ids=ids, use_cache=True,
                         logits_to_keep=1).past_key_values

    p_long, p_short = solo(long_ids), solo(short_ids)

    pad = tok.pad_token_id
    batch = torch.cat([long_ids,
                       torch.cat([torch.full((1, 200), pad, device="cuda"),
                                  short_ids], 1)], 0)
    mask = torch.ones_like(batch)
    mask[1, :200] = 0
    with torch.no_grad():
        p_batch = model(input_ids=batch, attention_mask=mask, use_cache=True,
                        logits_to_keep=1).past_key_values

    a_b, g_b = e5.split_layers(p_batch)
    a_l, g_l = e5.split_layers(p_long)
    a_s, g_s = e5.split_layers(p_short)

    def cmp(name, bt, solo_t, row, tail=None):
        x = e5._get(bt)[row:row + 1]
        y = e5._get(solo_t)
        if tail is not None:                 # KV: chi so phan KHONG phai dem
            x = x[:, :, -tail:]
            y = y[:, :, -tail:]
        d = (x.float() - y.float()).abs()
        rel = d.max().item() / max(y.float().abs().max().item(), 1e-9)
        print(f"  {name:26} lech tuyet doi {d.max().item():.3e} | "
              f"tuong doi {rel:.3e}", flush=True)
        return rel

    k0 = sorted(g_b)[0]
    ka = sorted(a_b)[0]
    print(" hang 0 (KHONG dem -- moc doi chung, phai ~0):", flush=True)
    r0g = cmp("GDN recurrent_states", g_b[k0].recurrent_states,
              g_l[k0].recurrent_states, 0)
    r0k = cmp("attention keys", a_b[ka].keys, a_l[ka].keys, 0, tail=400)
    print(" hang 1 (CO 200 token dem trai):", flush=True)
    r1g = cmp("GDN recurrent_states", g_b[k0].recurrent_states,
              g_s[k0].recurrent_states, 1)
    r1k = cmp("attention keys", a_b[ka].keys, a_s[ka].keys, 1, tail=400)

    TOL = 1e-3
    print(f"\n  PHAN XU (nguong tuong doi {TOL}):", flush=True)
    print(f"   - hang khong dem : {'DAT' if max(r0g, r0k) < TOL else 'HONG'}"
          "  (hong o day = loi phep do, khong phai loi dem)", flush=True)
    ok = max(r1g, r1k) < TOL
    print(f"   - hang co dem    : {'DAT' if ok else 'HONG'}", flush=True)
    print("  => " + ("DEM VO HAI: gom lo thang duoc." if ok else
                     "DEM LAM HONG: BAT BUOC gom lo THEO DO DAI (cung do dai "
                     "= khong dem = dong nhat)."), flush=True)
    if not ok and r1k < TOL <= r1g:
        print("     (attention sach, GDN ban -> dung nhu nghi ngo: vong quet "
              "hoi quy khong ton trong mask)", flush=True)

    del p_long, p_short, p_batch
    torch.cuda.empty_cache()

    # -------------------------------------------------------------- B+C ----
    print("\n===== B+C. VRAM VA TOC DO THEO BATCH x CTX =====", flush=True)
    print(f"{'ctx':>6} {'batch':>6} {'peak GiB':>9} {'s/lo':>7} {'s/mau':>7} "
          f"{'nhanh hon b=1':>14}", flush=True)
    base = {}
    for ctx in [int(x) for x in args.ctxs.split(",")]:
        for b in [int(x) for x in args.batches.split(",")]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(1000, 50000, (b, ctx), device="cuda")
            try:
                with torch.no_grad():           # nong may
                    model(input_ids=ids, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    past = model(input_ids=ids, use_cache=True,
                                 logits_to_keep=1).past_key_values
                torch.cuda.synchronize()
                dt = time.time() - t0
                per = dt / b
                if b == 1:
                    base[ctx] = per
                sp = base.get(ctx, per) / per
                print(f"{ctx:6} {b:6} {gib():9.2f} {dt:7.2f} {per:7.3f} "
                      f"{sp:13.2f}x", flush=True)
                del past
            except torch.cuda.OutOfMemoryError:
                print(f"{ctx:6} {b:6} {'OOM':>9}", flush=True)
                torch.cuda.empty_cache()
                break
            del ids
    print("\nLUU Y: day moi la prefill 4B mot minh (khong grad). Buoc train "
          "that con co 9B forward + backward; ty le nhanh len se KHAC, nhung "
          "tran VRAM va xu huong bao hoa thi doc duoc tu day.", flush=True)
    print("PROBE_BATCH_EXIT", flush=True)


if __name__ == "__main__":
    main()
