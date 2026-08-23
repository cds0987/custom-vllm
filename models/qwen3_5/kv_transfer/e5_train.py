"""E5 — trainable cache mapper 4B->27B with FUNCTIONAL loss (user-approved).

E4 measured verdict (STATUS.md): attention x->27B CCA 0.93-0.97 (GO), mid-GDN
CCA 0.27 (linear wall), deep-GDN heavy tails (sv_max ~110) => per-layer
lightweight mapper trained by matching the TARGET MODEL'S OUTPUTS (KL on
logits), with a decaying MSE-to-teacher-cache auxiliary as ridge-like warm
start.

Two OOM postmortems shaped this file (L4 = 22GiB):
  - bnb-4bit does NOT quantize embed/lm_head: 27B alone ~18GB, 4B ~3.5GB —
    both resident + autograd does not fit. => TWO-PHASE layout: phase A runs
    the 4B ALONE and spills every source cache to disk (fp16); phase B runs
    the 27B ALONE for teacher prefill + training. Nothing shares the GPU with
    the backward pass.
  - The functional loss only needs gradients through a SHORT suffix forward
    (T2 tokens); the mapped cache enters as input tensors, so autograd
    reaches the mapper without backprop through the prefill.

Mapper (~35M params): attention per tgt layer WK,WV 1024x1024 on RoPE-stripped
K (identity-init); GDN per tgt layer head-mix alpha(48x32) + A,B(128x128) on
RMS-normalized states. Source layer chosen by relative depth. conv_states are
zeroed; the first CONV_WARM suffix tokens rebuild them and the loss skips
those positions.

Run:  python e5_train.py --steps 400 --out /content/mapper_e5.pt
Eval: python e5_train.py --eval-only --out /content/mapper_e5.pt
"""

import argparse
import gc
import importlib.util
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e2_suite", Path(__file__).parent / "e2_suite.py")
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)
spec1 = importlib.util.spec_from_file_location(
    "e1_pipeline", Path(__file__).parent / "e1_pipeline.py")
e1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(e1)

L_CTX = 512   # 768: step-0 passed, Adam-state alloc tipped step-1 OOM
T2 = 32   # 96 OOMed: backward graph through 48 GDN torch-fallback layers ~3.7GB
CONV_WARM = 4


def load_4bit(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def rope_cs(T, dim, theta, device):
    import torch
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    ang = torch.arange(T, device=device).float()[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)   # (T, dim/2)


def rope_apply(x, cos, sin, sign):
    """x: (..., T, dim). sign=+1 apply, -1 strip (inverse rotation)."""
    import torch
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    s = sign * sin
    return torch.cat([x1 * cos - x2 * s, x2 * cos + x1 * s], -1)


class Mapper:
    def __init__(self, n_attn_tgt, n_gdn_tgt, gdn_heads_s, gdn_heads_t,
                 attn_dim, theta_s, theta_t, device="cuda"):
        import torch
        self.theta_s, self.theta_t = theta_s, theta_t
        self.params = []
        self.WK, self.bK, self.WV, self.bV = [], [], [], []
        for _ in range(n_attn_tgt):
            for lst in (self.WK, self.WV):
                w = torch.eye(attn_dim, device=device).float().requires_grad_(True)
                lst.append(w); self.params.append(w)
            for lst in (self.bK, self.bV):
                b = torch.zeros(attn_dim, device=device, requires_grad=True)
                lst.append(b); self.params.append(b)
        self.alpha, self.A, self.B = [], [], []
        for _ in range(n_gdn_tgt):
            a = torch.full((gdn_heads_t, gdn_heads_s), 1.0 / gdn_heads_s,
                           device=device).requires_grad_(True)
            A = torch.eye(128, device=device).float().requires_grad_(True)
            B = torch.eye(128, device=device).float().requires_grad_(True)
            self.alpha.append(a); self.A.append(A); self.B.append(B)
            self.params += [a, A, B]

    def map_attn(self, j, k, v):
        """k,v: (1, H, T, dh) source tensors on cuda. Returns mapped bf16."""
        import torch
        _, H, T, dh = k.shape
        cos_s, sin_s = rope_cs(T, dh, self.theta_s, k.device)
        cos_t, sin_t = rope_cs(T, dh, self.theta_t, k.device)
        k_st = rope_apply(k.float(), cos_s, sin_s, -1)         # strip src rope
        flat_k = k_st.permute(0, 2, 1, 3).reshape(T, H * dh)   # (T, 1024)
        flat_v = v.float().permute(0, 2, 1, 3).reshape(T, H * dh)
        mk = (flat_k @ self.WK[j] + self.bK[j]).reshape(1, T, H, dh).permute(0, 2, 1, 3)
        mv = (flat_v @ self.WV[j] + self.bV[j]).reshape(1, T, H, dh).permute(0, 2, 1, 3)
        mk = rope_apply(mk, cos_t, sin_t, +1)                  # apply tgt rope
        return mk.to(torch.bfloat16), mv.to(torch.bfloat16)

    def map_gdn(self, j, S):
        """S: (1, Hs, dk, dv) -> (1, Ht, dk, dv)."""
        import torch
        S = S[0].float()
        rms = S.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-6
        Sn = S / rms                                            # tame sv_max~110
        mapped = torch.einsum("ts,sij->tij", self.alpha[j],
                              self.A[j] @ Sn @ self.B[j])
        return (mapped * rms.mean())[None].to(torch.bfloat16)

    def state_dict(self):
        return {"WK": self.WK, "bK": self.bK, "WV": self.WV, "bV": self.bV,
                "alpha": self.alpha, "A": self.A, "B": self.B}

    def load(self, path):
        import torch
        sd = torch.load(path, map_location="cuda")
        for name in ("WK", "bK", "WV", "bV", "alpha", "A", "B"):
            for dst, src in zip(getattr(self, name), sd[name]):
                dst.data.copy_(src.data)


# ---------------- disk-spilled source caches (phase A -> phase B) -----------

def _get(x):
    """transformers >=5.15 boc recurrent/conv states trong dict {0: tensor}."""
    return x[0] if isinstance(x, dict) else x


def _set_like(layer, attr, value):
    """Gan state vao layer song, giu nguyen kieu boc (dict hay tensor)."""
    cur = getattr(layer, attr)
    if isinstance(cur, dict):
        setattr(layer, attr, {0: value.to(cur[0].dtype)})
    else:
        setattr(layer, attr, value.to(cur.dtype))


class _FA:
    def __init__(self, k, v):
        self.keys, self.values = k, v


class _FG:
    def __init__(self, r, c):
        self.recurrent_states, self.conv_states = r, c


class FakeCache:
    def __init__(self, layers):
        self.layers = layers


def spill_cache(past, path):
    import torch
    d = []
    for l in past.layers:
        if "LinearAttention" in type(l).__name__:
            d.append(("g", _get(l.recurrent_states).to(torch.float16).cpu(),
                      _get(l.conv_states).to(torch.float16).cpu()))
        else:
            d.append(("a", l.keys.to(torch.float16).cpu(),
                      l.values.to(torch.float16).cpu()))
    torch.save(d, path)


def load_cache(path):
    import torch
    layers = []
    for t, x, y in torch.load(path, map_location="cpu"):
        x = x.cuda().to(torch.bfloat16)
        y = y.cuda().to(torch.bfloat16)
        layers.append(_FA(x, y) if t == "a" else _FG(x, y))
    return FakeCache(layers)


def split_layers(past):
    attn, gdn = {}, {}
    for i, l in enumerate(past.layers):
        is_gdn = isinstance(l, _FG) or "LinearAttention" in type(l).__name__
        (gdn if is_gdn else attn)[i] = l
    return attn, gdn


def depth_map(n_src, n_tgt):
    return [round(j * (n_src - 1) / max(n_tgt - 1, 1)) for j in range(n_tgt)]


def build_student_past(tpl_past, src_past, mapper):
    """Deepcopy 27B template (positions/shapes right), swap in mapped
    grad-tracking tensors by ATTRIBUTE replacement (not in-place: autograd)."""
    import copy
    import torch
    past = copy.deepcopy(tpl_past)
    attn_s, gdn_s = split_layers(src_past)
    attn_t, gdn_t = split_layers(past)
    ks, kt = sorted(attn_s), sorted(attn_t)
    amap = depth_map(len(ks), len(kt))
    for j, it in enumerate(kt):
        src = attn_s[ks[amap[j]]]
        mk, mv = mapper.map_attn(j, src.keys, src.values)
        attn_t[it].keys = mk
        attn_t[it].values = mv
    gs, gt = sorted(gdn_s), sorted(gdn_t)
    gmap = depth_map(len(gs), len(gt))
    for j, it in enumerate(gt):
        src = gdn_s[gs[gmap[j]]]
        _set_like(gdn_t[it], "recurrent_states",
                  mapper.map_gdn(j, _get(src.recurrent_states)))
        _set_like(gdn_t[it], "conv_states",
                  torch.zeros_like(_get(gdn_t[it].conv_states)))
    return past


def aux_mse(student_past, teacher_past):
    loss, n = 0.0, 0
    for ls, lt in zip(student_past.layers, teacher_past.layers):
        if "LinearAttention" in type(lt).__name__:
            loss = loss + (_get(ls.recurrent_states).float()
                           - _get(lt.recurrent_states).float()).pow(2).mean()
        else:
            loss = loss + (ls.keys.float() - lt.keys.float()).pow(2).mean() \
                        + (ls.values.float() - lt.values.float()).pow(2).mean()
        n += 1
    return loss / n


def make_eval_trials(tok, n_trials):
    import random
    rng = random.Random(0)
    trials = []
    for ti in range(n_trials):
        name = rng.choice(e2.NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = e2.token_stream(tok, 1400, seed=900 + ti)
        ctx = (tok.decode(ids[:700])
               + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
               + tok.decode(ids[700:]) + "\n" + e2.build_q(name))
        trials.append((name, code, ctx))
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/content/mapper_e5.pt")
    ap.add_argument("--src-cache-dir", default="/content/e5_src")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-trials", type=int, default=10)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig

    cdir = Path(args.src_cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    done_marker = cdir / f"DONE_{args.steps}_{args.eval_trials}_L{L_CTX}_T{T2}"

    # ---------------- PHASE A: 4B alone -> spill source caches --------------
    if not done_marker.exists():
        tok_s, model_s = load_4bit(args.src_model)
        stream = e2.token_stream(tok_s, args.steps * L_CTX + L_CTX, seed=11)
        with torch.no_grad():
            for step in range(args.steps):
                pth = cdir / f"tr{step}_L{L_CTX}_T{T2}.pt"
                if pth.exists():
                    continue
                ids = torch.tensor([stream[step * L_CTX:(step + 1) * L_CTX]],
                                   device="cuda")
                past = model_s(input_ids=ids[:, :-T2], use_cache=True,
                               logits_to_keep=1).past_key_values
                spill_cache(past, pth)
                del past
                torch.cuda.empty_cache()
                if step % 50 == 0:
                    print(f"A train-cache {step}/{args.steps}")
            for ti, (name, code, ctx) in enumerate(
                    make_eval_trials(tok_s, args.eval_trials)):
                enc = tok_s(ctx, return_tensors="pt").to("cuda")
                past = model_s(input_ids=enc["input_ids"][:, :-1],
                               use_cache=True, logits_to_keep=1).past_key_values
                spill_cache(past, cdir / f"ev{ti}_L{L_CTX}_T{T2}.pt")
                del past
                torch.cuda.empty_cache()
            print("A eval-caches done")
        del model_s
        gc.collect()
        torch.cuda.empty_cache()
        done_marker.touch()
        print("PHASE_A_DONE")

    theta_s = e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    tok_t, model_t = load_4bit(args.tgt_model)
    theta_t = e1.get_rope_theta(model_t.config.get_text_config())

    # probe structures: source from a spilled cache, target via dummy forward
    src0 = load_cache(cdir / f"tr0_L{L_CTX}_T{T2}.pt")
    a_s, g_s = split_layers(src0)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    attn_dim = (next(iter(a_s.values())).keys.shape[1]
                * next(iter(a_s.values())).keys.shape[3])
    n_as, n_gs = len(a_s), len(g_s)
    del src0
    torch.cuda.empty_cache()
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = split_layers(probe_t)
    Ht = next(iter(g_t.values())).recurrent_states.shape[1]
    print(f"attn {n_as}->{len(a_t)} (dim {attn_dim}), "
          f"gdn {n_gs}->{len(g_t)} heads {Hs}->{Ht}, "
          f"theta {theta_s}/{theta_t}")
    mapper = Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    if args.eval_only:
        mapper.load(args.out)

    # ---------------- PHASE B: 27B alone -> train ---------------------------
    if not args.eval_only:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.Adam8bit(mapper.params, lr=args.lr)
            print("optimizer: Adam8bit")
        except Exception:
            opt = torch.optim.Adam(mapper.params, lr=args.lr, foreach=False)
            print("optimizer: Adam fp32 (bnb khong co)")
        stream = e2.token_stream(tok_t, args.steps * L_CTX + L_CTX, seed=11)
        t0 = time.time()
        for step in range(args.steps):
            gc.collect()
            torch.cuda.empty_cache()
            ids = torch.tensor([stream[step * L_CTX:(step + 1) * L_CTX]],
                               device="cuda")
            ctx, suffix = ids[:, :-T2], ids[:, -T2:]
            with torch.no_grad():
                import copy as _c
                tch_past = model_t(input_ids=ctx, use_cache=True,
                                   logits_to_keep=1).past_key_values
                tch_ext = _c.deepcopy(tch_past)
                tch_logp = torch.log_softmax(
                    model_t(input_ids=suffix, past_key_values=tch_ext,
                            use_cache=True).logits[:, CONV_WARM:].float(), -1)
                del tch_ext
                torch.cuda.empty_cache()
            src_past = load_cache(cdir / f"tr{step}_L{L_CTX}_T{T2}.pt")
            student_past = build_student_past(tch_past, src_past, mapper)
            lam = max(0.0, 1.0 - step / (0.3 * args.steps))
            aux = aux_mse(student_past, tch_past)   # BEFORE forward extends cache
            out = model_t(input_ids=suffix, past_key_values=student_past,
                          use_cache=True)
            stu_logp = torch.log_softmax(out.logits[:, CONV_WARM:].float(), -1)
            kl = F.kl_div(stu_logp, tch_logp, log_target=True,
                          reduction="batchmean")
            loss = kl + lam * aux
            opt.zero_grad(); loss.backward(); opt.step()
            klv = float(kl)
            del student_past, out, stu_logp, tch_logp, src_past, tch_past, kl, loss
            torch.cuda.empty_cache()
            if step % 10 == 0:
                print(f"step {step}/{args.steps} KL {klv:.4f} "
                      f"lam {lam:.2f} ({time.time()-t0:.0f}s)")
            if step % 50 == 49 or step == args.steps - 1:
                torch.save(mapper.state_dict(), args.out)
        print("TRAIN_DONE")

    # ---------------- eval: needle, mapped vs self vs no_ctx ----------------
    import re
    res = {c: 0 for c in ("self", "mapped", "no_ctx")}
    trials = make_eval_trials(tok_t, args.eval_trials)
    with torch.no_grad():
        for ti, (name, code, ctx) in enumerate(trials):
            enc = tok_t(ctx, return_tensors="pt").to("cuda")
            pre, last0 = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
            for cond in ("self", "mapped", "no_ctx"):
                last = last0
                if cond == "self":
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                elif cond == "no_ctx":
                    q = tok_t(e2.build_q(name), return_tensors="pt").to("cuda")
                    past = model_t(input_ids=q["input_ids"][:, :-1],
                                   use_cache=True,
                                   logits_to_keep=1).past_key_values
                    last = q["input_ids"][:, -1:]
                else:
                    src_past = load_cache(cdir / f"ev{ti}_L{L_CTX}_T{T2}.pt")
                    tpl = model_t(input_ids=pre, use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = build_student_past(tpl, src_past, mapper)
                    del src_past, tpl
                cur, gen, inp = past, [], last
                for _ in range(10):
                    o = model_t(input_ids=inp, past_key_values=cur,
                                use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                res[cond] += int(code in re.sub(r"\D", "", tok_t.decode(gen)))
                del past, cur
                torch.cuda.empty_cache()
            print(f"eval {ti+1}/{len(trials)}: " +
                  " ".join(f"{c}:{res[c]}" for c in res))
    print("===== E5 KET QUA =====")
    for c, h in res.items():
        print(f"{c:8s} needle {h}/{args.eval_trials}")
    print("E5_DONE")


if __name__ == "__main__":
    main()
