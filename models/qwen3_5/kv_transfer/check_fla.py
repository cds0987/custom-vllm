"""check_fla -- xac nhan fla (flash-linear-attention) duoc transformers TU
DONG dung (khong can sua code -- xem out/transformers/src/transformers/
integrations/hub_kernels.py: use_kernel_func_from_hub_with_fallback thu
`import fla` luc MODULE nap, uu tien no hon ban torch-thuan neu co) VA cho
ket qua nhat quan truoc khi tin dung cho train/eval (rule 15).

Quyet dinh dung fla hay khong xay ra LUC IMPORT module Qwen3_5GatedDeltaNet
-- khong the bat/tat giua chung trong CUNG mot tien trinh python, nen phai
chay 2 LAN, 2 TIEN TRINH RIENG (truoc/sau khi cai):

    python check_fla.py --tag truoc            # chay khi CHUA cai fla
    pip install flash-linear-attention[cuda]
    python check_fla.py --tag sau               # chay SAU khi cai
    python check_fla.py --diff truoc sau         # so token-exact-match
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import torch

_H = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_check(args):
    try:
        import fla
        fla_status = f"CO ({getattr(fla, '__version__', '?')})"
    except ImportError as e:
        fla_status = f"KHONG CO ({e})"
    print(f"fla import: {fla_status}", flush=True)

    e5 = _load("e5_train")
    eba = _load("eba_gen")
    e5.patch_recurrent_rebind()

    tok, model = e5.load_4bit(args.tgt_model)
    model.eval()

    items = eba.build(args.n, "/tmp/_fla_check.json", seed=12345)
    out = {"fla": fla_status, "gens": {}, "texts": {}}
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=1024)["input_ids"].to("cuda")
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=40, do_sample=False,
                               temperature=None, top_p=None, top_k=None)
        gen_ids = o[0, ids.shape[1]:].tolist()
        out["gens"][it["id"]] = gen_ids
        out["texts"][it["id"]] = tok.decode(gen_ids, skip_special_tokens=True)
        print(it["id"], gen_ids[:12], "...", flush=True)

    p = pathlib.Path(f"/content/logs/fla_check_{args.tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False))
    print(f"da ghi {p}")


def run_diff(tag_a, tag_b):
    a = json.loads(pathlib.Path(f"/content/logs/fla_check_{tag_a}.json").read_text())
    b = json.loads(pathlib.Path(f"/content/logs/fla_check_{tag_b}.json").read_text())
    print(f"{tag_a}: fla={a['fla']}")
    print(f"{tag_b}: fla={b['fla']}")
    ids = sorted(set(a["gens"]) & set(b["gens"]))
    n_exact, n_tot = 0, 0
    for i in ids:
        ga, gb = a["gens"][i], b["gens"][i]
        n_tot += 1
        if ga == gb:
            n_exact += 1
        else:
            print(f"  LECH {i}: {a['texts'][i][:80]!r} != {b['texts'][i][:80]!r}")
    print(f"\ntoken-exact-match: {n_exact}/{n_tot}")
    print("Khop 100% -> an toan dung ngay. Lech vai ca -> doc tay (thuong la "
          "khac biet dau phay dong luon tai buoc gan-hoa, khong phai loi that "
          "-- nhung PHAI doc tay truoc khi tin, dung suy doan).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--diff", nargs=2, default=None, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()

    if args.diff:
        run_diff(*args.diff)
    else:
        if not args.tag:
            print("can --tag hoac --diff A B"); sys.exit(1)
        run_check(args)


if __name__ == "__main__":
    main()
