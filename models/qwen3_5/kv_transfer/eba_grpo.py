"""eba_grpo -- GRPO (buoc 2, sau SFT) tren du lieu Entity Binding Arithmetic
(eba_gen.py). User 2026-09-03: "sinh synthesis data + RL ep mapper dem dung
relationship entities ... tham khao cach unsloth xay dung grpo qua 2 buoc
sft + rlhf".

VI SAO GRPO (khao sat Unsloth docs, xem TRANG-THAI.md muc "Unsloth GRPO"):
  - reward = TONG nhieu ham reward rieng biet, moi ham tra list[float] --
    o day la 3 lop A/B/C cua eba_gen.score_eba (tach vi mot diem gop se
    khong biet loi nam o dau -- bai hoc "error-placement").
  - advantage = CHUAN HOA Z-SCORE trong CHINH NHOM K completion cua MOT
    prompt -- khong can them mot value-network/critic rieng. Quan trong voi
    kien truc nay: moi tang them (critic, model tham chieu rieng) la mot
    noi co the tai pham loi tensor-alias GDN da bat 2 lan trong phien nay.
  - Unsloth canh bao: "neu xac suat luon la 0 (hay reward luon giong nhau
    trong nhom) RL se khong hoc duoc gi" -> BAT BUOC warm-start tu checkpoint
    SFT (joint49bb/joint49cc) de nhom sample co it nhat vai completion dung.
    Buoc SFT (e9_joint.py) coi nhu DA XONG, khong lam lai o day.

THIET KE HAI-PHA MOI PROMPT (tranh phai giu do thi gradient song qua toan
bo K nhanh sinh -- rat dat, xem TRANG-THAI.md phan uoc luong chi phi):
  1. student_past() MOT LAN CO GRAD (mapper + 4B TBPTT prefill nhu e9_joint).
  2. K nhanh SAMPLING: deep_clone_cache RỒI .detach() tung tensor -> decode
     KHONG grad (re, ngan) de lay van ban + tinh reward (Unsloth cung lam
     "generate roi moi tinh logp o pass rieng", khong phai sang tao rieng).
  3. K nhanh TEACHER-FORCE CO GRAD: deep_clone_cache LAI tu chinh st0_grad
     (van con gan graph vi .clone() la op kha vi) -> feed dung chuoi token
     DA SAMPLE o buoc 2 -> logp co grad -> loss = -advantage * sum(logp).
  4. Cong THEM mot nhanh CE(gold_template) trong so nho (--anchor-w) tu
     CUNG st0_grad -- thay cho mot model tham chieu dong lanh rieng (se ton
     gap doi VRAM): giu vai tro "ptx" cua InstructGPT (tron SFT-loss vao RL-
     loss de chong troi), re hon nhieu tren 1 GPU L4.

CHUA co PPO-clip (ty le xac suat cu/moi) vi rollout va update xay ra CUNG
mot buoc (on-policy tuyet doi, ty le = 1) -- neu sau nay tach rollout khoi
update (nhieu buoc update tren cung 1 lo rollout) thi PHAI them clip, hien
tai chua can.

    python -u eba_grpo.py --steps 300 --k 6 --sanity 5
"""

import argparse
import gc
import importlib.util
import json
import os
import random
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
eba = _load("eba_gen")
e5.patch_recurrent_rebind()   # BAT BUOC truoc khi nap model (xem docstring)

WARM_P = 5


def deep_clone_cache(past):
    """Nhu oracle_ablation.py: clone_cache_struct CHI tao container moi, tensor
    BEN TRONG van dung chung storage. GDN cap nhat .copy_() TAI CHO khi khong
    grad -> phai clone() tung tensor that de nhieu nhanh doc lap khong pha
    nhau. .clone() VAN la mot phep toan kha vi -- goi tren tensor requires_grad
    khong cat dut graph, chi khi .detach() sau do moi cat."""
    new = e5.clone_cache_struct(past)
    attn_n, gdn_n = e5.split_layers(new)
    for l in attn_n.values():
        l.keys = l.keys.clone()
        l.values = l.values.clone()
    for l in gdn_n.values():
        r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
        e5._set_like(l, "recurrent_states", r.clone())
        e5._set_like(l, "conv_states", c.clone())
    return new


def detach_cache(past):
    """Cat graph cho nhanh SAMPLING (khong can grad, chi can gia tri dung)."""
    attn, gdn = e5.split_layers(past)
    for l in attn.values():
        l.keys = l.keys.detach()
        l.values = l.values.detach()
    for l in gdn.values():
        r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
        e5._set_like(l, "recurrent_states", r.detach())
        e5._set_like(l, "conv_states", c.detach())
    return past


@torch.no_grad()
def sample_rollout(model, tok, past, warm, n_new, temperature, stops):
    """Sinh CO NHIET DO (khong greedy -- can da dang trong nhom de co advantage
    khac 0). Tra ve (list[int] token id, text). CHI con dung cho K=1 / debug --
    duong chinh la sample_rollout_batch (xem docstring ham do: goc cham 40s/
    buoc la vong nay lap K=6 LAN RIENG LE, tuong duong 6xgen_len forward don-
    token TUAN TU, khop dung so do bnb-4bit 11,8 tok/s trong gen_pseudo_vllm.py)."""
    o = model(input_ids=warm, past_key_values=past, use_cache=True)
    cur = o.past_key_values
    probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
    inp = torch.multinomial(probs, 1)
    gen = [int(inp)]
    for _ in range(n_new - 1):
        o = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
        inp = torch.multinomial(probs, 1)
        gen.append(int(inp))
        if int(inp) in stops:
            break
    del cur, o
    return gen, tok.decode(gen, skip_special_tokens=True)


def clone_cache_repeat(past, k):
    """Nhu deep_clone_cache nhung MO RONG batch-dim tu 1 thanh k (repeat doc
    theo chieu 0) -- gop K nhanh sampling THANH MOT cache batch=K, thay vi K
    cache batch=1 rieng le. Day la fix goc cho bottleneck 11,8 tok/s: chuyen
    K*n_new lan forward DON-TOKEN tuan tu thanh n_new lan forward BATCH=K --
    dung ky thuat da do trong batch_decode.py (batch 2 = 1,85-1,97x thong
    luong, khong phai 2x thoi gian). .repeat() luon CAT DUT graph (khong
    goi tren tensor con requires_grad o day) -- dung cho nhanh SAMPLING vi no
    da la @torch.no_grad(), khac han deep_clone_cache (giu graph cho nhanh
    teacher-force)."""
    new = e5.clone_cache_struct(past)
    attn_n, gdn_n = e5.split_layers(new)
    for l in attn_n.values():
        l.keys = l.keys.detach().repeat(k, *([1] * (l.keys.dim() - 1)))
        l.values = l.values.detach().repeat(k, *([1] * (l.values.dim() - 1)))
    for l in gdn_n.values():
        r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
        e5._set_like(l, "recurrent_states",
                     r.detach().repeat(k, *([1] * (r.dim() - 1))))
        e5._set_like(l, "conv_states",
                     c.detach().repeat(k, *([1] * (c.dim() - 1))))
    return new


@torch.no_grad()
def sample_rollout_batch(model, tok, past_k, warm, n_new, temperature, stops, k):
    """past_k: cache DA O BATCH=K (clone_cache_repeat). Decode CA K nhanh
    CUNG LUC moi buoc thay vi K vong rieng -- day la duong chinh, thay cho
    K lan goi sample_rollout(). KHONG dung som theo tung hang rieng (phuc
    tap hoa vong lap) -- decode DU n_new buoc cho ca K roi CAT SAU tai vi
    tri stop dau tien cua tung hang (bai hoc: don gian hoa dung muc, gen_len
    da ngan (~30-50) nen phan tinh du thua khong dang ke so voi loi ich gop
    lo)."""
    warm_b = warm.repeat(k, 1)
    o = model(input_ids=warm_b, past_key_values=past_k, use_cache=True)
    cur = o.past_key_values
    probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
    inp = torch.multinomial(probs, 1)
    gens = [[int(inp[i, 0])] for i in range(k)]
    for _ in range(n_new - 1):
        o = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
        inp = torch.multinomial(probs, 1)
        for i in range(k):
            gens[i].append(int(inp[i, 0]))
    del cur, o
    trimmed, texts = [], []
    for g in gens:
        cut = len(g)
        for j, t in enumerate(g):
            if t in stops:
                cut = j + 1
                break
        g2 = g[:cut]
        trimmed.append(g2)
        texts.append(tok.decode(g2, skip_special_tokens=True))
    return trimmed, texts


def teacher_force_logp(model, past, warm, gen_ids, device):
    """Forward CO GRAD, ep dung theo chuoi gen_ids (da sample o pha 1), tra ve
    tong log-prob (scalar co grad) -- dung CHINH quy uoc chi so cua e9_joint
    (logits[:, WARM_P-1:] du bao dung gen_ids[0..]). CHI con dung cho K=1/debug
    -- duong chinh la teacher_force_logp_batch (do 2026-09-03: pha nay van
    lap K=6 forward rieng, chiem 18-19%/buoc, gop lo cung theo huong da
    lam voi pha 1)."""
    gen_t = torch.tensor([gen_ids], device=device)
    feed = (torch.cat([warm, gen_t[:, :-1]], 1) if gen_t.shape[1] > 1
            else warm)
    o = model(input_ids=feed, past_key_values=past, use_cache=True)
    logp = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
    lp = logp.gather(2, gen_t.unsqueeze(-1)).squeeze(-1)
    return lp.sum()


def clone_cache_repeat_grad(past, k):
    """Nhu clone_cache_repeat NHUNG GIU GRAPH (khong .detach()) -- dung cho
    pha 2 (teacher-force CO grad): .repeat() la phep toan kha vi, nen K nhanh
    van noi nguoc ve DUNG MOT lan build st0 (mapper+4B) -- gradient tu ca K
    nhanh CONG DON dung vao cung tham so, khong phai tinh rieng tung nhanh."""
    new = e5.clone_cache_struct(past)
    attn_n, gdn_n = e5.split_layers(new)
    for l in attn_n.values():
        l.keys = l.keys.repeat(k, *([1] * (l.keys.dim() - 1)))
        l.values = l.values.repeat(k, *([1] * (l.values.dim() - 1)))
    for l in gdn_n.values():
        r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
        e5._set_like(l, "recurrent_states",
                     r.repeat(k, *([1] * (r.dim() - 1))))
        e5._set_like(l, "conv_states",
                     c.repeat(k, *([1] * (c.dim() - 1))))
    return new


def teacher_force_logp_batch(model, past_k, warm, gens, device):
    """Gop K nhanh teacher-force THANH MOT forward co grad (thay vi K lan
    rieng). Feed = warm + gen_ids[:-1], PAD PHAI ve do dai lon nhat trong
    nhom, dem bang -100 khi tinh logp -- dung Y HET quy uoc enc_batch cua
    e9_joint.py: causal attention/GDN CHI phu thuoc vi tri TRUOC, nen dem SAU
    (token gia o cuoi hang ngan) khong lam sai logit o cac vi tri THAT truoc
    do -- da duoc dung that trong e9_joint (khong phai gia dinh moi)."""
    k = len(gens)
    gmax = max(len(g) for g in gens)
    gold = torch.full((k, gmax), -100, dtype=torch.long, device=device)
    for i, g in enumerate(gens):
        gold[i, :len(g)] = torch.tensor(g, device=device)
    warm_b = warm.repeat(k, 1)
    if gmax > 1:
        feed_pad = torch.zeros(k, gmax - 1, dtype=torch.long, device=device)
        for i, g in enumerate(gens):
            if len(g) > 1:
                feed_pad[i, :len(g) - 1] = torch.tensor(g[:-1], device=device)
        feed = torch.cat([warm_b, feed_pad], 1)
    else:
        feed = warm_b
    o = model(input_ids=feed, past_key_values=past_k, use_cache=True)
    logp = torch.log_softmax(o.logits[:, WARM_P - 1:WARM_P - 1 + gmax].float(), -1)
    valid = (gold >= 0).float()
    lp = logp.gather(2, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid
    return lp.sum(dim=1)   # (k,) -- tong logp moi nhanh, co grad


def group_advantage(rewards, eps=1e-4):
    """Chuan hoa Z-score TRONG NHOM (GRPO, khong critic). None neu std~0 --
    dung canh bao Unsloth: reward giong nhau ca nhom = khong co gradient,
    BO QUA item nay thay vi chia cho so gan 0."""
    r = torch.tensor(rewards, dtype=torch.float32)
    std = r.std(unbiased=False).item()
    if std < eps:
        return None
    return ((r - r.mean()) / (std + eps)).tolist()


def reward_of(item, text, w):
    s = eba.score_eba(item, text)
    return w["A"] * s["A"] + w["B"] * s["B"] + w["C"] * s["C"], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--init-mapper", default="/content/joint49cc/mapper_best.pt",
                    help="BAT BUOC warm-start SFT -- xem canh bao Unsloth o "
                         "docstring dau file")
    ap.add_argument("--init-lora", default="/content/joint49cc/lora_best")
    ap.add_argument("--init-lora-t", default="/content/joint49cc/lorat_best")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-t-r", type=int, default=16)
    ap.add_argument("--lora-t-modules",
                    default="q_proj,o_proj,in_proj_qkvz,out_proj")
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--difficulty-max", type=int, default=1,
                    help="chi dung item co difficulty <= gia tri nay (0..3) "
                         "-- bat dau tu de nhat, escalate dan giong probe "
                         "trich-so da lam truoc do")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="so completion/prompt")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--tbptt", type=int, default=64)
    ap.add_argument("--w-a", type=float, default=0.3)
    ap.add_argument("--w-b", type=float, default=0.2)
    ap.add_argument("--w-c", type=float, default=0.5)
    ap.add_argument("--anchor-w", type=float, default=0.2,
                    help="trong so CE(gold_template) tron vao loss RL -- thay "
                         "cho mot model tham chieu dong lanh rieng (ton gap "
                         "doi VRAM tren 1 GPU L4); dung vai tro 'ptx' cua "
                         "InstructGPT chong troi khoi SFT")
    ap.add_argument("--lr", type=float, default=2e-4, help="lr mapper")
    ap.add_argument("--lora-lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=30)
    ap.add_argument("--out", default="/content/eba_grpo_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="eba_grpo_v1")
    ap.add_argument("--sanity", type=int, default=0,
                    help="chay N buoc roi dung + in VRAM/toc do, khong val "
                         "khong luu ckpt -- dung TRUOC khi cam ket --steps day")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w = {"A": args.w_a, "B": args.w_b, "C": args.w_c}

    from huggingface_hub import HfApi
    _api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    def hf_up(local, dest):
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        try:
            _api.upload_file(path_or_fileobj=str(local),
                             path_in_repo=f"{args.hf_prefix}/{dest}",
                             repo_id=args.hf_repo)
            print(f"HF-UP {dest}", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL {dest}: {type(ex).__name__}: {ex}", flush=True)

    # ---- data: sinh tai cho, KHONG doc/ghi dia -- item la thuan JSON+meta ----
    items = eba.build(args.n_items, "/tmp/_eba_items.json", seed=args.seed)
    items = [it for it in items if it["difficulty"] <= args.difficulty_max]
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(8, int(len(items) * args.val_frac))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"eba: {len(train_items)} train, {len(val_items)} val "
          f"(difficulty<={args.difficulty_max})", flush=True)

    # ---- nap 9B + LoRA-9B (warm-start SFT) ----
    t0 = time.time()
    tok_t, model_t = e5.load_4bit(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    mods_t = [x for x in args.lora_t_modules.split(",") if x]
    model_t = get_peft_model(model_t, LoraConfig(
        r=args.lora_t_r, lora_alpha=2 * args.lora_t_r, lora_dropout=0.0,
        bias="none", target_modules=mods_t, task_type="CAUSAL_LM"))
    model_t.train()
    if Path(args.init_lora_t).exists():
        set_peft_model_state_dict(model_t, load_file(
            str(Path(args.init_lora_t) / "adapter_model.safetensors")))
        print(f"warm-start LoRA-9B tu {args.init_lora_t}", flush=True)
    else:
        print(f"CANH BAO: khong thay {args.init_lora_t} -- LoRA-9B train tu "
              f"0, VI PHAM dieu kien 'SFT truoc RL' cua Unsloth", flush=True)
    lora_t_params = [p for p in model_t.parameters() if p.requires_grad]
    tok_t.truncation_side = "left"
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]

    # ---- nap 4B + LoRA-4B (warm-start SFT) ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    for p in model_s.parameters():
        p.requires_grad_(False)
    model_s = get_peft_model(model_s, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    if Path(args.init_lora).exists():
        set_peft_model_state_dict(model_s, load_file(
            str(Path(args.init_lora) / "adapter_model.safetensors")))
        print(f"warm-start LoRA-4B tu {args.init_lora}", flush=True)
    else:
        print(f"CANH BAO: khong thay {args.init_lora}", flush=True)
    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    tok_s.truncation_side = "left"
    theta_s = e5.e1.get_rope_theta(
        __import__("transformers").AutoConfig.from_pretrained(
            args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    # ---- mapper (warm-start SFT, BAT BUOC) ----
    _meta = torch.load(args.init_mapper, map_location="cpu").get("_meta", {}) \
        if Path(args.init_mapper).exists() else {}
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=_meta.get("attn_rank", 0),
                       gdn_per_head=_meta.get("gdn_per_head", False),
                       gdn_terms=_meta.get("gdn_terms", 1))
    if Path(args.init_mapper).exists():
        mapper.load(args.init_mapper)
        print(f"warm-start mapper tu {args.init_mapper} "
              f"(gdn_terms={_meta.get('gdn_terms', 1)})", flush=True)
    else:
        print(f"CANH BAO: khong thay {args.init_mapper} -- mapper train tu "
              f"0, VI PHAM dieu kien 'SFT truoc RL'", flush=True)
    print(f"nap xong {time.time()-t0:.0f}s", flush=True)

    STOPS = e5.stop_ids(tok_t, model_t)

    def prefill_tbptt(ids, wnd):
        """Y het e9_joint.prefill_tbptt: T-w token dau no_grad, w cuoi co grad
        (GC bi loai vi forward can use_cache=True de sinh cache)."""
        past, cutp = None, max(0, ids.shape[1] - wnd)
        if cutp:
            with torch.no_grad():
                for s in range(0, cutp, 1024):
                    o = model_s(input_ids=ids[:, s:min(s + 1024, cutp)],
                               past_key_values=past, use_cache=True,
                               logits_to_keep=1)
                    past = o.past_key_values
        for s in range(cutp, ids.shape[1], 1024):
            o = model_s(input_ids=ids[:, s:s + 1024], past_key_values=past,
                       use_cache=True, logits_to_keep=1)
            past = o.past_key_values
        return past

    T_BASE = 512
    with torch.no_grad():
        _ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
        _p = e5.prefill_chunked(model_t, _ids)
    base_meta = e5.cache_meta(_p)
    del _p, _ids
    torch.cuda.empty_cache()

    import copy as _copy

    def meta_for_len(t):
        m = _copy.deepcopy(base_meta)
        m["cache_ints"] = {k: (t if v == T_BASE else v)
                           for k, v in m["cache_ints"].items()}
        for lay in m["layers"]:
            lay["ints"] = {k: (t if v == T_BASE else v)
                           for k, v in lay["ints"].items()}
            if lay["kind"] == "a":
                for key in ("k", "v"):
                    sh, dt = lay[key]
                    lay[key] = (tuple(t if d == T_BASE else d for d in sh), dt)
        return m

    def student_past_grad(cut):
        """CO GRAD -- goi DUNG MOT LAN moi buoc, moi nhanh sau do chi
        deep_clone_cache() tu day (van gan graph, xem docstring dau file)."""
        src = prefill_tbptt(cut, args.tbptt)
        tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
        st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    def enc(it):
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=2048)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        gold_ids = tok_t(it["gold"], add_special_tokens=False,
                         return_tensors="pt")["input_ids"][:, :64].to("cuda")
        return cut, warm, gold_ids

    import bitsandbytes as bnb
    groups = [{"params": mapper.params, "lr": args.lr},
              {"params": lora_params, "lr": args.lora_lr},
              {"params": lora_t_params, "lr": args.lora_lr}]
    opt = bnb.optim.Adam8bit(groups)

    def gib():
        return torch.cuda.max_memory_allocated() / 2**30

    @torch.no_grad()
    def run_val(n):
        """Bao cao reward TRUNG BINH (A/B/C rieng) tren val -- KHONG phai
        vong val gsm8k day du cua e9_joint, chi de theo doi GRPO co tien
        khong. Dung greedy (nhiet do 0) cho on dinh giua cac moc."""
        agg = {"A": [], "B": [], "C": []}
        for it in val_items[:n]:
            cut, warm, _ = enc(it)
            src = e5.prefill_chunked(model_s, cut)
            tpl = e5.build_template_from_meta(probe_t, meta_for_len(cut.shape[1]))
            st = e5.build_student_past(tpl, src, mapper)
            o = model_t(input_ids=warm, past_key_values=st, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen = [int(inp)]
            for _ in range(args.gen_len - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen.append(int(inp))
                if int(inp) in STOPS:
                    break
            txt = tok_t.decode(gen, skip_special_tokens=True)
            s = eba.score_eba(it, txt)
            for kk in ("A", "B", "C"):
                agg[kk].append(s[kk])
            del st, tpl, src, cur, o
            torch.cuda.empty_cache()
        return {kk: round(sum(v) / max(len(v), 1), 3) for kk, v in agg.items()}

    results = {"args": vars(args), "val": [], "train": []}

    def save_results():
        (out / "results.json").write_text(json.dumps(results, indent=1))

    def save_ckpt(tag):
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        model_t.save_pretrained(str(out / f"lorat_{tag}"))
        hf_up(out / f"mapper_{tag}.pt", f"mapper_{tag}.pt")
        for sub in (f"lora_{tag}", f"lorat_{tag}"):
            for f in sorted((out / sub).glob("*")):
                if f.is_file():
                    hf_up(f, f"{sub}/{f.name}")

    # Dong ho tung chang (y het e9_joint.py: PHAI synchronize truoc khi doc
    # dong ho, CUDA chay bat dong bo). Them SAU khi do duoc gop lo giam
    # 38-40s/buoc -> 15,2s/buoc: ty trong chi phi da DOI, khong con chac
    # chan sampling van la phan lon nhat -- do tiep truoc khi doan tiep
    # (bai hoc flash-attn: da tung toi uu SAI cho, attention chi 0,03%).
    T_ACC = {}

    class clock:
        def __init__(self, k):
            self.k = k

        def __enter__(self):
            torch.cuda.synchronize()
            self.t = time.time()
            return self

        def __exit__(self, *a):
            torch.cuda.synchronize()
            T_ACC[self.k] = T_ACC.get(self.k, 0.0) + time.time() - self.t
            return False

    best = -1e9
    t_start = time.time()
    for step in range(1, args.steps + 1):
        it = train_items[(step - 1) % len(train_items)]
        cut, warm, gold_ids = enc(it)

        with clock("student_past_grad"):
            st0 = student_past_grad(cut)

        # pha 1: K nhanh sampling GOP LO (1 vong decode batch=K thay vi K
        # vong rieng le -- fix goc bottleneck 11,8 tok/s, xem docstring
        # sample_rollout_batch)
        with clock("sampling(pha1)"):
            branch_k = clone_cache_repeat(st0, args.k)
            gens, texts = sample_rollout_batch(
                model_t, tok_t, branch_k, warm, args.gen_len,
                args.temperature, STOPS, args.k)
            del branch_k
        with clock("reward"):
            rewards, sub = [], []
            for txt in texts:
                r, s = reward_of(it, txt, w)
                rewards.append(r)
                sub.append(s)

        adv = group_advantage(rewards)
        if adv is None:
            del st0
            torch.cuda.empty_cache()
            if step % 20 == 0:
                print(f"buoc {step}: reward dong nhat ({rewards[0]:.2f}) -> "
                      f"bo qua (khong co gradient)", flush=True)
            continue

        # pha 2: K nhanh teacher-force CO grad GOP LO (1 forward batch=K,
        # thay vi K forward rieng -- cung ky thuat da dung o pha 1)
        with clock("teacher_force(pha2)"):
            branch_k = clone_cache_repeat_grad(st0, args.k)
            lp_k = teacher_force_logp_batch(model_t, branch_k, warm, gens, "cuda")
            adv_t = torch.tensor(adv, device="cuda")
            pg_loss = -(adv_t * lp_k).mean()
            del branch_k

        anchor_ce = torch.tensor(0.0, device="cuda")
        with clock("anchor_ce"):
            if args.anchor_w > 0 and gold_ids.shape[1] >= 1:
                branch = deep_clone_cache(st0)
                gold_list = gold_ids[0].tolist()
                feed = torch.cat([warm, gold_ids[:, :-1]], 1) \
                    if gold_ids.shape[1] > 1 else warm
                o = model_t(input_ids=feed, past_key_values=branch, use_cache=True)
                logp_g = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
                nll = -logp_g.gather(
                    2, gold_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                anchor_ce = nll.mean()
                del branch, o, logp_g

        with clock("backward"):
            loss = pg_loss + args.anchor_w * anchor_ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        del st0
        gc.collect()
        torch.cuda.empty_cache()

        mean_r = sum(rewards) / len(rewards)
        mean_c = sum(x["C"] for x in sub) / len(sub)
        if step % 10 == 0:
            results["train"].append([step, round(mean_r, 3), round(mean_c, 3)])
            tot = sum(T_ACC.values()) or 1.0
            share = " ".join(f"{k} {100*v/tot:.0f}%"
                             for k, v in sorted(T_ACC.items(),
                                                key=lambda x: -x[1]))
            print(f"buoc {step}/{args.steps} pg={pg_loss.item():.4f} "
                  f"anchor_ce={anchor_ce.item():.4f} reward_tb={mean_r:.3f} "
                  f"C_tb={mean_c:.3f} {(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={gib():.2f}GiB", flush=True)
            print(f"    thoi gian: {tot/10:.2f}s/buoc do duoc | {share}",
                  flush=True)
            T_ACC.clear()

        if args.sanity and step >= args.sanity:
            print(f"SANITY xong {step} buoc, peak={gib():.2f}GiB, "
                  f"{(time.time()-t_start)/step:.2f}s/buoc", flush=True)
            print("EBA_GRPO_SANITY_EXIT", flush=True)
            return

        if step % args.val_every == 0:
            vs = run_val(args.val_n)
            score = vs["C"]
            results["val"].append([step, vs])
            print(f"=== VAL buoc {step}: {vs}", flush=True)
            save_results()
            hf_up(out / "results.json", "results.json")
            save_ckpt("last")
            if score > best:
                best = score
                save_ckpt("best")
                print(f"    ky luc moi C_tb={best:.3f}", flush=True)

    if args.steps % args.val_every != 0:
        vs = run_val(args.val_n)
        results["val"].append([args.steps, vs])
        print(f"=== VAL cuoi: {vs}", flush=True)
        save_results()
        hf_up(out / "results.json", "results.json")
        save_ckpt("last")
        if vs["C"] > best:
            save_ckpt("best")
    print("EBA_GRPO_EXIT", flush=True)


if __name__ == "__main__":
    main()
