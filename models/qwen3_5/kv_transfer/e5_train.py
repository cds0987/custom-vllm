"""E5 — trainable cache mapper 4B->27B with FUNCTIONAL loss (user-approved).

E4 measured verdict (STATUS.md):
  - attention x->27B: CCA 0.93-0.97 (linear structure GO) but identity dead
  - mid-GDN: CCA 0.27 — linear closed forms cannot carry it
  - deep GDN heavy-tailed (A-hat sv_max ~110) — raw MSE explodes
  => per-layer lightweight mapper, trained by matching the TARGET MODEL'S
     OUTPUTS (KL on logits), not cache values. MSE-to-teacher-cache is used
     only as a decaying auxiliary to warm-start (plays the role of ridge init
     without a separate calib phase).

Key engineering trick that makes this trainable on one L4: the functional
loss only needs gradients through a SHORT suffix forward (T2=128 tokens).
The mapped cache enters that forward as input tensors, so autograd reaches
the mapper without ever backpropping through the full prefill.

Per training step (both models bnb-4bit, frozen, co-resident ~18GB):
  1. text (L=1024 tok); split ctx = [0..L-T2), suffix = [L-T2..L)
  2. no_grad: 4B prefill(ctx) -> src cache; 27B prefill(ctx) -> teacher cache
     + teacher logits over suffix
  3. grad: mapped = Mapper(src cache); 27B forward(suffix, past=mapped)
     loss = KL(teacher || student) + lam_aux * MSE(mapped, teacher cache)
  4. conv_states: zeroed + first 4 suffix tokens rebuild them (kernel=4);
     loss skips those warmup positions.

Mapper (~35M params):
  attention (16 tgt layers): per layer WK,WV: 1024->1024 on RoPE-stripped K
    (strip src theta, apply tgt theta), identity-init.
  GDN (48 tgt layers): S_t[h] = sum_s alpha[h,s] * A S_s[s] B  — head-mix
    (48x32) + shared per-layer A,B (128x128, identity-init); src layer chosen
    by relative depth. RMS-normalize state before map, restore scale after
    (E4: sv_max 110 => raw scales unusable).

Run:  python e5_train.py --steps 400 --out /content/mapper_e5.pt
Eval: python e5_train.py --eval-only --out /content/mapper_e5.pt
"""

import argparse
import importlib.util
import math
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

L_CTX = 1024
T2 = 128
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
            for lst, init in ((self.WK, torch.eye(attn_dim)),
                              (self.WV, torch.eye(attn_dim))):
                w = init.to(device).float().requires_grad_(True)
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

    def map_attn(self, j, k, v, positions):
        """k,v: (1, H, T, dh) from source. Returns mapped bf16 tensors."""
        import torch
        _, H, T, dh = k.shape
        cos_s, sin_s = rope_cs(T, dh, self.theta_s, k.device)
        cos_t, sin_t = rope_cs(T, dh, self.theta_t, k.device)
        k_st = rope_apply(k.float(), cos_s, sin_s, -1)         # strip src rope
        flat_k = k_st.permute(0, 2, 1, 3).reshape(T, H * dh)   # (T, 1024)
        flat_v = v.float().permute(0, 2, 1, 3).reshape(T, H * dh)
        mk = flat_k @ self.WK[j] + self.bK[j]
        mv = flat_v @ self.WV[j] + self.bV[j]
        mk = mk.reshape(1, T, H, dh).permute(0, 2, 1, 3)
        mv = mv.reshape(1, T, H, dh).permute(0, 2, 1, 3)
        mk = rope_apply(mk, cos_t, sin_t, +1)                  # apply tgt rope
        return mk.to(torch.bfloat16), mv.to(torch.bfloat16)

    def map_gdn(self, j, S):
        """S: (1, Hs, dk, dv) source recurrent state -> (1, Ht, dk, dv)."""
        import torch
        S = S[0].float()                                        # (Hs,128,128)
        rms = S.pow(2).mean((-2, -1), keepdim=True).sqrt() + 1e-6
        Sn = S / rms                                            # tame sv_max~110
        mapped = torch.einsum("ts,sij->tij", self.alpha[j],
                              self.A[j] @ Sn @ self.B[j])
        scale = rms.mean()                                      # restore scale
        return (mapped * scale)[None].to(torch.bfloat16)

    def state_dict(self):
        return {"WK": self.WK, "bK": self.bK, "WV": self.WV, "bV": self.bV,
                "alpha": self.alpha, "A": self.A, "B": self.B}

    def load(self, path):
        import torch
        sd = torch.load(path, map_location="cuda")
        for name in ("WK", "bK", "WV", "bV", "alpha", "A", "B"):
            for dst, src in zip(getattr(self, name), sd[name]):
                dst.data.copy_(src.data)


def split_layers(past):
    attn, gdn = {}, {}
    for i, l in enumerate(past.layers):
        (gdn if "LinearAttention" in type(l).__name__ else attn)[i] = l
    return attn, gdn


def depth_map(n_src, n_tgt):
    return [round(j * (n_src - 1) / max(n_tgt - 1, 1)) for j in range(n_tgt)]


def build_student_past(tpl_past, src_past, mapper):
    """Deepcopy 27B template (shapes/positions right), swap in mapped
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
        T = src.keys.shape[2]
        mk, mv = mapper.map_attn(j, src.keys, src.values, T)
        attn_t[it].keys = mk
        attn_t[it].values = mv
    gs, gt = sorted(gdn_s), sorted(gdn_t)
    gmap = depth_map(len(gs), len(gt))
    for j, it in enumerate(gt):
        src = gdn_s[gs[gmap[j]]]
        gdn_t[it].recurrent_states = mapper.map_gdn(j, src.recurrent_states)
        gdn_t[it].conv_states = torch.zeros_like(gdn_t[it].conv_states)
    return past


def aux_mse(student_past, teacher_past):
    import torch
    loss, n = 0.0, 0
    for ls, lt in zip(student_past.layers, teacher_past.layers):
        if "LinearAttention" in type(ls).__name__:
            loss = loss + (ls.recurrent_states.float()
                           - lt.recurrent_states.float()).pow(2).mean()
        else:
            loss = loss + (ls.keys.float() - lt.keys.float()).pow(2).mean() \
                        + (ls.values.float() - lt.values.float()).pow(2).mean()
        n += 1
    return loss / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/content/mapper_e5.pt")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-trials", type=int, default=10)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    tok_s, model_s = load_4bit(args.src_model)
    tok_t, model_t = load_4bit(args.tgt_model)
    cfg_s = model_s.config.get_text_config()
    cfg_t = model_t.config.get_text_config()
    theta_s, theta_t = e1.get_rope_theta(cfg_s), e1.get_rope_theta(cfg_t)

    # probe layer structure with a dummy forward
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = split_layers(probe_s)
    a_t, g_t = split_layers(probe_t)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    Ht = next(iter(g_t.values())).recurrent_states.shape[1]
    attn_dim = (next(iter(a_s.values())).keys.shape[1]
                * next(iter(a_s.values())).keys.shape[3])
    print(f"attn {len(a_s)}->{len(a_t)} (dim {attn_dim}), "
          f"gdn {len(g_s)}->{len(g_t)} heads {Hs}->{Ht}, "
          f"theta {theta_s}/{theta_t}")

    mapper = Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    if args.eval_only:
        mapper.load(args.out)

    if not args.eval_only:
        opt = torch.optim.Adam(mapper.params, lr=args.lr)
        stream = e2.token_stream(tok_s, args.steps * L_CTX + L_CTX, seed=11)
        t0 = time.time()
        for step in range(args.steps):
            ids = torch.tensor([stream[step * L_CTX:(step + 1) * L_CTX]],
                               device="cuda")
            ctx, suffix = ids[:, :-T2], ids[:, -T2:]
            with torch.no_grad():
                src_past = model_s(input_ids=ctx, use_cache=True,
                                   logits_to_keep=1).past_key_values
                tch = model_t(input_ids=ctx, use_cache=True,
                              logits_to_keep=1)
                tch_out = model_t(input_ids=suffix,
                                  past_key_values=tch.past_key_values,
                                  use_cache=True)
                tch_logp = torch.log_softmax(
                    tch_out.logits[:, CONV_WARM:].float(), -1)
                # teacher cache re-prefill for aux target (ctx-only cache)
                tch_past = model_t(input_ids=ctx, use_cache=True,
                                   logits_to_keep=1).past_key_values
            student_past = build_student_past(tch_past, src_past, mapper)
            out = model_t(input_ids=suffix, past_key_values=student_past,
                          use_cache=True)
            stu_logp = torch.log_softmax(out.logits[:, CONV_WARM:].float(), -1)
            kl = F.kl_div(stu_logp, tch_logp, log_target=True,
                          reduction="batchmean")
            lam = max(0.0, 1.0 - step / (0.3 * args.steps))
            loss = kl + lam * aux_mse(student_past, tch_past)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 10 == 0:
                print(f"step {step}/{args.steps} KL {float(kl):.4f} "
                      f"lam {lam:.2f} ({time.time()-t0:.0f}s)")
            if step % 50 == 49 or step == args.steps - 1:
                torch.save(mapper.state_dict(), args.out)
        print("TRAIN_DONE")

    # ---- eval: needle @~1.5K, mapper vs no_ctx vs 27B self ----
    import random, re
    rng = random.Random(0)
    res = {c: 0 for c in ("self", "mapped", "no_ctx")}
    with torch.no_grad():
        pass
    for ti in range(args.eval_trials):
        name = rng.choice(e2.NAMES)
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = e2.token_stream(tok_t, 1400, seed=900 + ti)
        ctx_txt = (tok_t.decode(ids[:700])
                   + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                   + tok_t.decode(ids[700:]) + "\n" + e2.build_q(name))
        enc = tok_t(ctx_txt, return_tensors="pt").to("cuda")
        pre, last = enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]
        for cond in ("self", "mapped", "no_ctx"):
            with torch.no_grad():
                if cond == "self":
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                elif cond == "no_ctx":
                    q = tok_t(e2.build_q(name), return_tensors="pt").to("cuda")
                    past = model_t(input_ids=q["input_ids"][:, :-1],
                                   use_cache=True, logits_to_keep=1).past_key_values
                    last = q["input_ids"][:, -1:]
                else:
                    src_past = model_s(input_ids=pre, use_cache=True,
                                       logits_to_keep=1).past_key_values
                    tpl = model_t(input_ids=pre, use_cache=True,
                                  logits_to_keep=1).past_key_values
                    past = build_student_past(tpl, src_past, mapper)
                cur, gen = past, []
                logp = None
                inp = last
                for _ in range(10):
                    o = model_t(input_ids=inp, past_key_values=cur,
                                use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                ok = code in re.sub(r"\D", "", tok_t.decode(gen))
                res[cond] += int(ok)
            last = enc["input_ids"][:, -1:]
        print(f"eval {ti+1}/{args.eval_trials}: " +
              " ".join(f"{c}:{res[c]}" for c in res))
    print("===== E5 KET QUA =====")
    for c, h in res.items():
        print(f"{c:8s} needle {h}/{args.eval_trials}")
    print("E5_DONE")


if __name__ == "__main__":
    main()
