"""probe_train_batch -- gom LO cho BUOC TRAIN co dang xay khong.

User 2026-08-31: "xay dung cac cach cai tien toc do eval-finetune".

VI SAO PHAI DO TRUOC KHI XAY. probe_batch cu chi do PREFILL 4B, ma prefill chi
chiem 7-18% buoc train (do duoc: backward 52-56%, 9B-forward 17%, 4B-grad 17%,
4B-nograd 7-17%, mapper 1%). Neu gom lo khong lam nhanh phan BACKWARD thi ca
cong xay bo gom-lo-theo-do-dai chi doi lay ~6%.

Va gom lo cho TRAIN kho hon cho eval: prefill 4B PHAI khong dem (dem trai lam
attention keys sai 96%, GDN xau hon nen 3,7 lan — probe_batch da do). Nen
duong duy nhat la GOM THEO DO DAI CHINH XAC. Truoc khi viet bo gom do, phai
biet hai so:

  1. Gom lo co lam nhanh buoc train that khong, va bao nhieu?
  2. Trong pool train, co bao nhieu mau CUNG DO DAI de gom duoc?

Cau 2 do bang tokenizer (khong can GPU) — xem --dist.

Chay:
  python -u probe_train_batch.py --batches 1,2,4 --ctxs 512,1024
  python -u probe_train_batch.py --dist          # chi dem do dai, khong GPU
"""

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dist_only(args):
    """Bao nhieu mau CUNG DO DAI CHINH XAC -> gom duoc lo bao lon."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.src_model)
    data = json.loads(Path(args.data_file).read_text())
    lens = Counter()
    for it in data["train"]:
        if it["kind"] in set(args.drop_kinds.split(",")):
            continue
        lens[len(tk(it["prompt"], add_special_tokens=False)["input_ids"])] += 1
    n = sum(lens.values())
    print(f"{n} item, {len(lens)} do dai khac nhau")
    for B in (2, 4, 8):
        # so item gom duoc thanh lo day B khi CHI ghep item cung do dai
        usable = sum((c // B) * B for c in lens.values())
        print(f"  lo {B}: gom duoc {usable}/{n} = {100*usable/n:.1f}% so item")
    # noi long: cho phep CAT TRAI ve boi so cua `q` de tang co hoi trung
    for q in (32, 64, 128):
        b2 = Counter()
        for L, c in lens.items():
            b2[(L // q) * q] += c
        for B in (4, 8):
            usable = sum((c // B) * B for c in b2.values())
            print(f"  lam tron xuong boi {q}, lo {B}: {100*usable/n:.1f}% "
                  f"(mat toi da {q-1} token dau moi mau)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--drop-kinds", default="gsm8k,suite_math")
    ap.add_argument("--batches", default="1,2,4")
    ap.add_argument("--ctxs", default="512,1024")
    ap.add_argument("--gold", type=int, default=48)
    ap.add_argument("--tbptt", type=int, default=128)
    ap.add_argument("--dist", action="store_true")
    args = ap.parse_args()

    if args.dist:
        return dist_only(args)

    import torch
    from transformers import AutoConfig
    e5 = _load("e5_train")

    tok_s, model_s = e5.load_4bit(args.src_model)
    from peft import LoraConfig, get_peft_model
    model_s = get_peft_model(model_s, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    lora_params = [p for p in model_s.parameters() if p.requires_grad]

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

    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit([{"params": mapper.params, "lr": 1e-3},
                              {"params": lora_params, "lr": 1e-4}])

    def one_step(B, T):
        """Mot buoc train THAT: prefill 4B (tbptt) -> mapper -> 9B fwd -> bwd."""
        ids = torch.randint(1000, 50000, (B, T), device="cuda")
        gold = torch.randint(1000, 50000, (B, args.gold), device="cuda")
        cut = max(0, T - args.tbptt)
        past = None
        with torch.no_grad():
            for s_ in range(0, cut, 1024):
                past = model_s(input_ids=ids[:, s_:min(s_ + 1024, cut)],
                               past_key_values=past, use_cache=True,
                               logits_to_keep=1).past_key_values
        for s_ in range(cut, T, 1024):
            past = model_s(input_ids=ids[:, s_:s_ + 1024],
                           past_key_values=past, use_cache=True,
                           logits_to_keep=1).past_key_values
        tpl = e5.build_template_from_meta(probe, e5.cache_meta(
            _prefill_shape(model_t, B, T)))
        st = e5.build_student_past(tpl, past, mapper)
        o = model_t(input_ids=gold, past_key_values=st, use_cache=True)
        ce = torch.nn.functional.cross_entropy(
            o.logits.reshape(-1, o.logits.shape[-1]).float(), gold.reshape(-1))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        opt.step()
        del ids, gold, past, tpl, st, o, ce

    _shape_cache = {}

    def _prefill_shape(m, B, T):
        if (B, T) not in _shape_cache:
            with torch.no_grad():
                _shape_cache[(B, T)] = m(
                    input_ids=torch.randint(1000, 5000, (B, T), device="cuda"),
                    use_cache=True, logits_to_keep=1).past_key_values
        return _shape_cache[(B, T)]

    print(f"\n{'ctx':>6}{'batch':>7}{'peak GiB':>10}{'s/lo':>8}{'s/mau':>8}"
          f"{'nhanh hon b=1':>15}")
    base = {}
    for T in [int(x) for x in args.ctxs.split(",")]:
        for B in [int(x) for x in args.batches.split(",")]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                one_step(B, T)                       # nong may
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(3):
                    one_step(B, T)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 3
                per = dt / B
                if B == 1:
                    base[T] = per
                sp = base.get(T, per) / per
                print(f"{T:6}{B:7}{torch.cuda.max_memory_allocated()/2**30:9.2f}"
                      f"{dt:8.2f}{per:8.3f}{sp:14.2f}x", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{T:6}{B:7}{'OOM':>10}", flush=True)
                torch.cuda.empty_cache()
                _shape_cache.clear()
                break
    print("\nDOC: neu b=2 khong dat >=1,3x thi gom lo cho TRAIN khong dang xay "
          "— backward chiem 52-56% va no khong duoc loi tu gom lo.")
    print("PROBE_TRAIN_BATCH_EXIT", flush=True)


if __name__ == "__main__":
    main()
