"""probe_decode_speed -- do TOC DO DECODE cua 9B trong vong sampling RL.

BOI CANH (user 2026-09-05, "tim cach lay duoc toc do cua vllm luon"): vong
GRPO hien tai 21s/buoc, trong do sampling chiem 87-90%. vLLM khong cam thang
vao duoc (rollout phai bat dau tu CACHE DO MAPPER SINH, va LoRA-9B doi moi
buoc) -- nen phep kiem nay tach 3 NGUON toc do cua vLLM va do rieng tung cai
xem lay duoc bao nhieu NGAY TRONG PROCESS:

  A. KERNEL   : Marlin W4A16 (compressed-tensors) thay bnb-NF4 dequant
  B. BATCH    : decode la weight-bound o batch nho -> k=2 lang phi GPU
  C. OVERHEAD : vong lap hien tai goi int(inp[i,0]) -> DONG BO GPU->CPU k lan
                MOI TOKEN. Gom thanh mot .tolist() la mien phi.

KHONG do gi lien quan mapper o day: chi phi decode chi phu thuoc HINH DANG
cache, khong phu thuoc cache do ai sinh -> prefill mot prompt dai tuong duong
gsm8k (~256 token) roi decode, dung bang chi phi thuc.

CANH BAO khi doc so: champion la checkpoint DA SUA (frame W4A16 + GDN graft
tu GGUF), KHONG cung trong so voi Qwen/Qwen3.5-9B stock ma mapper da hoc tren
do. Phep kiem nay tra loi "kernel co nhanh hon khong", KHONG tra loi "dung
duoc ngay khong" -- cau sau can do lai chat luong.

    python3 -u probe_decode_speed.py --variant bnb
    python3 -u probe_decode_speed.py --variant w4a16
"""
import argparse
import importlib.util
import json
import pathlib
import time

import torch

_H = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


STOCK = "Qwen/Qwen3.5-9B"
CHAMPION = "gunnybd01/qwen35-9b-champion"


def load_w4a16(name):
    """Nap checkpoint compressed-tensors (kernel Marlin). Khong truyen
    quantization_config -- cau hinh nam san trong config.json cua checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="cuda", dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def decode_loop(model, past, inp, n_new, sync_each_token):
    """Vong decode toi gian. sync_each_token=True mo phong DUNG vong hien tai
    trong eba_grpo.sample_rollout_batch (int(inp[i,0]) moi token, moi hang);
    False = gom ve mot .tolist() sau khi xong -> khong dong bo giua chung."""
    k = inp.shape[0]
    cur = past
    acc = []
    for _ in range(n_new):
        o = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        probs = torch.softmax(o.logits[:, -1, :].float(), -1)
        inp = torch.multinomial(probs, 1)
        if sync_each_token:
            for i in range(k):
                _ = int(inp[i, 0])          # <- dong bo GPU->CPU
        else:
            acc.append(inp)
    if not sync_each_token:
        _ = torch.cat(acc, 1).tolist()      # mot lan dong bo duy nhat
    del cur, o
    return


def bench(model, tok, k, n_new, prompt_len, sync_each_token, warmup=1):
    text = ("Question: " + "Natalia sold clips to 48 of her friends in April. "
            * 40 + "\nSolution: ")
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_len].cuda()
    ids = ids.repeat(k, 1)
    for _ in range(warmup):
        o = model(input_ids=ids, use_cache=True)
        decode_loop(model, o.past_key_values, o.logits[:, -1:, :].argmax(-1), 4, True)
        del o
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    o = model(input_ids=ids, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.time() - t0
    first = o.logits[:, -1:, :].argmax(-1)
    t0 = time.time()
    decode_loop(model, o.past_key_values, first, n_new, sync_each_token)
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    del o
    torch.cuda.empty_cache()
    return {"k": k, "n_new": n_new, "sync": sync_each_token,
            "prefill_s": round(t_prefill, 3),
            "decode_s": round(t_dec, 3),
            "tok_s_tong": round(k * n_new / t_dec, 1),
            "ms_moi_buoc": round(1000 * t_dec / n_new, 2),
            "peak_GiB": round(peak, 2)}


def test_backward_lora(model, tok):
    """Cau hoi sinh tu: co gan LoRA + lan nguoc duoc tren nen W4A16 khong?
    bnb co autograd.Function rieng (nen QLoRA chay); kernel Marlin thi khong
    chac -- neu khong co backward theo DAU VAO thi gradient khong chay nguoc
    ve LoRA o cac lop duoi -> KHONG train duoc."""
    from peft import LoraConfig, get_peft_model
    m = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    m.train()
    ids = tok("Question: Natalia sold clips.\nSolution: ",
              return_tensors="pt").input_ids.cuda()
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    co_grad = [n for n, p in named if p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0]
    # lop THAP NHAT co grad moi la bang chung gradient chay HET chieu sau
    sau_nhat = max((int(n.split("layers.")[1].split(".")[0])
                    for n in co_grad if "layers." in n), default=-1)
    thap_nhat = min((int(n.split("layers.")[1].split(".")[0])
                     for n in co_grad if "layers." in n), default=-1)
    r = {"tham_so_lora": len(named), "co_grad": len(co_grad),
         "loss": float(out.loss), "lop_thap_nhat_co_grad": thap_nhat,
         "lop_cao_nhat_co_grad": sau_nhat}
    del out, m
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bnb", choices=["bnb", "w4a16"])
    ap.add_argument("--model", default="")
    ap.add_argument("--ks", default="2,4,8,16")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-backward", type=int, default=0)
    args = ap.parse_args()

    e5 = _load("e5_train")
    name = args.model or (STOCK if args.variant == "bnb" else CHAMPION)
    print(f"[{args.variant}] nap {name}", flush=True)
    t0 = time.time()
    tok, model = e5.load_4bit(name) if args.variant == "bnb" else load_w4a16(name)
    print(f"[{args.variant}] nap xong {time.time()-t0:.0f}s "
          f"static={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    rows = []
    for k in [int(x) for x in args.ks.split(",") if x]:
        for sync in (True, False):
            try:
                r = bench(model, tok, k, args.n_new, args.prompt_len, sync)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  k={k} sync={sync} OOM", flush=True)
                continue
            r["variant"] = args.variant
            rows.append(r)
            print(f"  k={r['k']:2d} sync={int(sync)} "
                  f"decode {r['decode_s']:6.2f}s = {r['tok_s_tong']:7.1f} tok/s tong | "
                  f"{r['ms_moi_buoc']:6.2f} ms/buoc | prefill {r['prefill_s']:.2f}s | "
                  f"peak {r['peak_GiB']:.2f}GiB", flush=True)

    bw = None
    if not args.skip_backward:
        try:
            bw = test_backward_lora(model, tok)
            print(f"[backward+LoRA] {bw}", flush=True)
        except Exception as e:
            bw = {"loi": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[backward+LoRA] HONG -> {bw['loi']}", flush=True)

    out = args.out or f"/content/probe_decode_{args.variant}.json"
    pathlib.Path(out).write_text(json.dumps(
        {"variant": args.variant, "model": name, "rows": rows, "backward": bw},
        ensure_ascii=False, indent=1))
    print("GHI", out, flush=True)
    print("PROBE_DECODE_EXIT", flush=True)


if __name__ == "__main__":
    main()
