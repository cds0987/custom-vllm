"""probe_joint_lora -- CONG DI/KHONG cho kien truc 2 LOP (user chot
2026-08-28: "1 cai tien 2 lop mapper, 1 gan vao 4b de ep 4b cho viec doc
cho 27B sau do merge cai do vao 4b, sau do huan luyen mapper dich 4-27").

CAU HOI DUY NHAT: gradient phai chay tu output 27B, QUA mapper, VAO LoRA
cua 4B -> ca hai model phai cung tren GPU KEM DO THI AUTOGRAD. Tren L4
22GB: 27B bnb-4bit ~18GB + 4B ~3,5GB (4bit) hoac ~8GB (bf16) = 21,5-26GB
truoc khi tinh activation. Da OOM that o run_cross voi dung cau hinh nay
(ma khong he co autograd!) -- nen day KHONG phai cau hoi ly thuyet.

Duong tranh duy nhat (loss khop-trang-thai, chi can 4B + cache 27B doc tu
dia) DA CHET 4 LAN trong du an (E8 v3 nMSE 10,5->1,06 ket, needle 0/5;
luat error-placement). Khong di lai. => phai do.

Do THUC: forward 4B CO GRAD -> mapper -> forward 27B -> CE -> backward,
kiem grad LoRA khac None. Bao peak VRAM + t/buoc cho tung (src_dtype, ctx).

Chay:  python -u probe_joint_lora.py [--ctxs 1024,2048,4096]
"""

import argparse
import gc
import importlib.util
import time
import traceback
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "e5_train", Path(__file__).parent / "e5_train.py")
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)

WARM_P = 5

# hoc phi 2026-08-27 (2 probe truoc, cung bug): transformers 5.15
# update_recurrent_state .copy_() IN-PLACE len state -> vo autograd khi
# initial_state mang grad. PHAI patch TRUOC khi nap model.
try:
    from transformers.cache_utils import LinearAttentionLayer
    _orig_urs = LinearAttentionLayer.update_recurrent_state

    def _urs(self, recurrent_states, state_idx=0, **kw):
        cur = (self.recurrent_states.get(state_idx)
               if isinstance(self.recurrent_states, dict) else None)
        if cur is not None and (cur.requires_grad or recurrent_states.requires_grad):
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states
        return _orig_urs(self, recurrent_states, state_idx, **kw)

    LinearAttentionLayer.update_recurrent_state = _urs
    print("patched update_recurrent_state (rebind khi co grad)", flush=True)
except ImportError as e:
    print("PATCH_IMPORT_FAIL", e, flush=True)


def gib():
    return torch.cuda.max_memory_allocated() / 2**30


def now():
    return torch.cuda.memory_allocated() / 2**30


_STAGE = []


def stage(name):
    """Luot 1 chi bao 'OOM peak 21,76' cho ca 3 ctx -> khong biet chang nao
    an bo nho. Ghi moc tung chang de bien phong doan thanh so do."""
    free, tot = torch.cuda.mem_get_info()
    _STAGE.append((name, round(now(), 2), round(gib(), 2), round(free / 2**30, 2)))
    print(f"    . {name:22} dang={now():6.2f} peak={gib():6.2f} "
          f"trong={free/2**30:5.2f} GiB", flush=True)


def load_src_bf16(name):
    """4B bf16 -- E6c da do: student luong tu hoa mat NUA bien; docs du an
    'KHONG QLoRA tren Qwen3.5'. Nhung bf16 = ~8GB thay vi 3,5GB, nen phai
    do CA HAI roi de SO DO quyet dinh, khong suy luan."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda")
    return tok, model


def attach_lora(model, r=16):
    """LoRA that (peft) neu co; neu khong, mo grad tren dung cac ma tran
    tuong ung de con so bo nho van trung thuc (activation moi la phan lon)."""
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    try:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
            target_modules=targets, task_type="CAUSAL_LM"))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"peft LoRA r={r}: {n/1e6:.1f}M param train", flush=True)
        return model, "peft"
    except Exception as e:
        print(f"peft KHONG dung duoc ({type(e).__name__}: {e}) -> fallback "
              "mo grad truc tiep (chi de do bo nho)", flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        n = 0
        for nm, p in model.named_parameters():
            if any(t in nm for t in targets):
                p.requires_grad_(True)
                n += p.numel()
        print(f"fallback: {n/1e6:.1f}M param train", flush=True)
        return model, "fallback"


def prefill_tbptt(model, ids, w, chunk=1024):
    """Cat ngan lan truyen nguoc theo thoi gian (TBPTT) -- duong thay the cho
    gradient checkpointing sau khi luot 3 chung minh GC va use_cache=True loai
    tru nhau (ma o day forward 4B TON TAI de sinh cache).

    T-w token dau chay no_grad (chi lay GIA TRI state), w token cuoi chay co
    grad. Bo nho activation khi do phu thuoc w chu khong phu thuoc T.

    XAP XI, khong phai chinh xac: LoRA chi nhan gradient tu w vi tri cuoi.
    Voi GDN (hoi quy) day dung la TBPTT kinh dien; voi attention thi K/V cua
    token cu thanh hang so -- tuc LoRA khong hoc duoc tu chung. Phai bao cao
    ro nhu mot danh doi, khong duoc lang."""
    past = None
    cut = max(0, ids.shape[1] - w)
    if cut:
        with torch.no_grad():
            for s_ in range(0, cut, chunk):
                o = model(input_ids=ids[:, s_:min(s_ + chunk, cut)],
                          past_key_values=past, use_cache=True, logits_to_keep=1)
                past = o.past_key_values
    for s_ in range(cut, ids.shape[1], chunk):
        o = model(input_ids=ids[:, s_:s_ + chunk], past_key_values=past,
                  use_cache=True, logits_to_keep=1)
        past = o.past_key_values
    return past


def prefill_grad(model, ids, chunk=1024):
    """prefill_chunked ban CO GRAD -- day la diem khac cot loi: activation
    cua 4B phai duoc giu lai cho backward, nen peak TANG theo T (khong con
    'transient peak doc lap T' nhu ban no_grad)."""
    past = None
    for s in range(0, ids.shape[1], chunk):
        o = model(input_ids=ids[:, s:s + chunk], past_key_values=past,
                  use_cache=True, logits_to_keep=1)
        past = o.past_key_values
    return past


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--ctxs", default="1024,2048,4096")
    ap.add_argument("--src-dtypes", default="4bit,bf16")
    ap.add_argument("--gold", type=int, default=16)
    ap.add_argument("--golds", default="",
                    help="quet nhieu do dai gold, vd '16,64,128,256'. gold la "
                         "SO VI TRI feed vao 27B nen no an bo nho THAT SU: "
                         "gsm8k gold la loi giai day du ~256 token, gap 16 lan "
                         "probe dau (16) -- phai do, khong duoc suy dien.")
    ap.add_argument("--tbptts", default="",
                    help="quet nhieu cua so TBPTT, vd '128,64,32'")
    ap.add_argument("--tbptt", type=int, default=0,
                    help="cat lan truyen nguoc: chi w vi tri CUOI co grad "
                         "(0 = tat). Duong thay the cho GC vi GC ep "
                         "use_cache=False -- xem prefill_tbptt.")
    ap.add_argument("--gc-src", action="store_true",
                    help="gradient checkpointing tren 4B (chi phi MOI cua "
                         "duong joint la activation 4B khi prefill co grad)")
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctxs.split(",")]
    golds = ([int(x) for x in args.golds.split(",")] if args.golds
             else [args.gold])
    tbptts = ([int(x) for x in args.tbptts.split(",")] if args.tbptts
              else [args.tbptt])
    combos = [(T, g, w) for T in ctxs for g in golds for w in tbptts]
    print(f"quet {len(combos)} cau hinh (ctx x gold x tbptt)", flush=True)

    from transformers import AutoConfig

    # ---- 27B nap MOT LAN, CPU-offload embed/lm_head (do 2026-08-27:
    #      tiet kiem that 4,85GiB, 2 lan goi lien tiep deu OK) ----------
    t0 = time.time()
    tok_t, model_t = e5.load_4bit_cpu_offload_io(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    print(f"27B nap xong {time.time()-t0:.0f}s, tinh {gib():.2f}GiB", flush=True)
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]

    # bo xuong template cho tung ctx (teacher prefill 1 lan, no_grad)
    metas = {}
    for T in ctxs:
        ids = torch.randint(1000, 5000, (1, T - WARM_P), device="cuda")
        with torch.no_grad():
            p = e5.prefill_chunked(model_t, ids)
        metas[T] = e5.cache_meta(p)
        del p, ids
        gc.collect()
        torch.cuda.empty_cache()
    print(f"template meta xong cho {ctxs}", flush=True)

    rows = []
    for sd in args.src_dtypes.split(","):
        print(f"\n===== 4B dtype={sd} =====", flush=True)
        try:
            if sd == "4bit":
                tok_s, model_s = e5.load_4bit(args.src_model)
            else:
                tok_s, model_s = load_src_bf16(args.src_model)
            for p in model_s.parameters():
                p.requires_grad_(False)
            model_s, lora_kind = attach_lora(model_s)
            if args.gc_src:
                try:
                    model_s.gradient_checkpointing_enable()
                    model_s.enable_input_require_grads()
                    print("gradient checkpointing BAT tren 4B", flush=True)
                except Exception as e:
                    print(f"gc_src that bai: {type(e).__name__}: {e}", flush=True)
            model_s.train()
            theta_s = e5.e1.get_rope_theta(
                AutoConfig.from_pretrained(args.src_model).get_text_config())
            with torch.no_grad():
                probe_s = model_s(
                    input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                    use_cache=True, logits_to_keep=1).past_key_values
            a_s, g_s = e5.split_layers(probe_s)
            Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
            k0 = e5._get(next(iter(a_s.values())).keys)
            attn_dim = k0.shape[1] * k0.shape[3]
            del probe_s
            mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim,
                               theta_s, theta_t)
            mapper.ckpt = True     # v3.3: recompute map_attn trong backward
            gc.collect()
            torch.cuda.empty_cache()
            print("ca hai model tren GPU: "
                  f"{torch.cuda.memory_allocated()/2**30:.2f}GiB tinh",
                  flush=True)
        except Exception as e:
            print(f"NAP THAT BAI dtype={sd}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

        for (T, GOLD, W) in combos:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() / 2**30
            t0 = time.time()
            src_past = tpl = student = out = None
            try:
                ids = torch.randint(1000, 5000, (1, T), device="cuda")
                cut, warm = ids[:, :-WARM_P], ids[:, -WARM_P:]
                gold = torch.randint(1000, 5000, (1, GOLD), device="cuda")
                stage("nen")
                src_past = (prefill_tbptt(model_s, cut, W) if W else
                            prefill_grad(model_s, cut))       # (1) 4B CO GRAD
                stage("sau prefill 4B")
                tpl = e5.build_template_from_meta(probe_t, metas[T])
                stage("sau template 27B")
                student = e5.build_student_past(tpl, src_past, mapper)  # (2)
                stage("sau mapper")
                feed = torch.cat([warm, gold[:, :-1]], 1)
                out = model_t(input_ids=feed, past_key_values=student,
                              use_cache=True)                  # (3) 27B
                stage("sau forward 27B")
                lg = out.logits[:, -GOLD + 1:, :].float()
                loss = torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]), gold[:, 1:].reshape(-1))
                loss.backward()
                stage("sau backward")
                gsum = sum(int(p.grad is not None) for p in model_s.parameters()
                           if p.requires_grad)
                gmap = sum(int(p.grad is not None) for p in mapper.params)
                dt = time.time() - t0
                rows.append((f"{sd} g{GOLD} w{W}", T, "OK", round(gib(), 2),
                             round(dt, 1), gsum, gmap))
                print(f"[{sd} T={T} gold={GOLD} tbptt={W}] OK "
                      f"peak={gib():.2f}GiB (nen {base:.2f}) "
                      f"t={dt:.1f}s loss={loss.item():.3f} "
                      f"grad_lora={gsum} grad_mapper={gmap}/{len(mapper.params)}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append((f"{sd} g{GOLD} w{W}", T, "OOM", round(gib(), 2),
                             round(time.time() - t0, 1), 0, 0))
                print(f"[{sd} T={T} gold={GOLD} tbptt={W}] OOM "
                      f"peak={gib():.2f}GiB", flush=True)
            except Exception as e:
                rows.append((f"{sd} g{GOLD} w{W}", T, type(e).__name__,
                             round(gib(), 2), round(time.time() - t0, 1), 0, 0))
                print(f"[{sd} T={T} gold={GOLD} tbptt={W}] LOI {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            finally:
                del src_past, tpl, student, out
                model_s.zero_grad(set_to_none=True)
                for p in mapper.params:
                    p.grad = None
                gc.collect()
                torch.cuda.empty_cache()

        del model_s, mapper
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== KET QUA JOINT PROBE ===", flush=True)
    print(f"{'cau hinh':16} {'ctx':>6} {'trang thai':>12} {'peak GiB':>9} "
          f"{'t/buoc':>7} {'gLoRA':>6} {'gMap':>5}")
    for r in rows:
        print(f"{r[0]:16} {r[1]:6d} {r[2]:>12} {r[3]:9.2f} {r[4]:7.1f}s "
              f"{r[5]:6d} {r[6]:5d}")
    ok = [r for r in rows if r[2] == "OK" and r[5] > 0 and r[6] > 0]
    if ok:
        best = max(ok, key=lambda r: r[1])
        print(f"\nPHAN QUYET: DI DUOC — cau hinh lon nhat chay tron: "
              f"{best[0]} @ctx {best[1]} ({best[3]}GiB, {best[4]}s/buoc)",
              flush=True)
    else:
        print("\nPHAN QUYET: KHONG cau hinh nao chay tron "
              "=> kien truc 2 lop can duong khac", flush=True)
    print("PROBE_JOINT_EXIT", flush=True)


if __name__ == "__main__":
    main()
