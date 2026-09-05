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
gd = _load("gen_data")
gs = _load("gsm_struct")
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
def sample_rollout_batch(model, tok, past_k, warm_rows, n_new, temperature,
                         stops, check_every=16):
    """past_k: cache DA O BATCH = so HANG (clone_cache_repeat). warm_rows: da
    mo rong san ve dung so hang do. Decode CA cac hang CUNG LUC moi buoc thay
    vi tung vong rieng -- day la duong chinh, thay cho nhieu lan goi
    sample_rollout().

    So hang = B (so mau trong lo) x K (so nhanh GRPO). Thu tu hang do
    clone_cache_repeat quyet dinh: .repeat(K,...) LAP CA KHOI B, nen
    hang r <-> mau r % B, nhanh r // B.

    VI SAO GOP CANG NHIEU HANG CANG LOI (do 2026-09-05,
    probe_decode_speed.py, 9B bnb-4bit, decode 64 token):
        k= 2 -> 95,7 ms/buoc |  20,9 tok/s
        k= 4 -> 94,4 ms/buoc |  42,4 tok/s
        k= 8 -> 95,2 ms/buoc |  84,1 tok/s
        k=16 -> 100,1 ms/buoc | 159,8 tok/s
    Thoi gian moi buoc decode GAN NHU KHONG DOI tu 2 den 16 hang (decode o
    batch nho bi chan boi bang thong doc TRONG SO, khong phai phep tinh) ->
    thong luong x7,7 gan nhu mien phi. Day dung la co che toc do cua vLLM
    (continuous batching), lay duoc NGAY TRONG PROCESS ma khong can vLLM --
    vLLM khong cam thang vao duoc vi rollout phai bat dau tu cache DO MAPPER
    SINH va LoRA-9B doi moi buoc.

    DUNG SOM CA VONG LAP khi MOI hang deu da gap stop (phan con lai chac chan
    bi cat o buoc trim ben duoi): gsm8k gen_len=224 nhieu loi giai xong o
    ~80-120 token, chay het la lang phi 40-60% vong lap.

    KHONG dong bo GPU->CPU moi token nua (do duoc: int(inp[i,0]) tung hang
    tung token lam k=2 tu 86,1 -> 95,7 ms/buoc, mat 9-10%). Thay bang kiem
    tra stop MOI check_every token tren GPU -> 1 lan dong bo / 16 token."""
    n = warm_rows.shape[0]
    dev = warm_rows.device
    stop_t = torch.tensor(sorted(stops), device=dev)
    o = model(input_ids=warm_rows, past_key_values=past_k, use_cache=True)
    cur = o.past_key_values
    probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
    inp = torch.multinomial(probs, 1)
    toks = [inp]
    done = torch.isin(inp[:, 0], stop_t)
    for t in range(n_new - 1):
        o = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = o.past_key_values
        probs = torch.softmax(o.logits[:, -1, :].float() / temperature, -1)
        inp = torch.multinomial(probs, 1)
        toks.append(inp)
        if (t + 2) % check_every == 0:
            done |= torch.isin(torch.cat(toks[-check_every:], 1), stop_t).any(1)
            if bool(done.all()):      # <- lan dong bo DUY NHAT moi 16 token
                break
    del cur, o
    gens = torch.cat(toks, 1).tolist()      # mot lan chuyen ve CPU
    trimmed, texts = [], []
    for g in gens:
        cut = len(g)
        for j, tk in enumerate(g):
            if tk in stops:
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


def clone_cache_index(past, idx):
    """Lay MOT TAP CON cac hang cua cache theo chi so idx, GIU GRAPH.

    Dung cho pha 2 khi chia nho teacher-force: index_select la phep kha vi
    (khac .repeat o cho chon duoc dung nhung hang can), nen gradient tu tung
    mieng van cong don nguoc ve cung mot lan build st0 (mapper + 4B).

    VI SAO PHAI CHIA NHO (do 2026-09-05): OOM khong nam o sampling ma o
    torch_chunk_gated_delta_rule (lop GDN) TRONG pha teacher-force CO GRAD --
    GDN luu hoat hoa cho ca 32 lop, ty le thuan voi so hang. Sampling
    (@no_grad) thi khong luu gi nen gop rong bao nhieu cung duoc. => gop RONG
    o pha 1 (cho toc do), chia NHO o pha 2 (cho bo nho)."""
    new = e5.clone_cache_struct(past)
    attn_n, gdn_n = e5.split_layers(new)
    for l in attn_n.values():
        l.keys = l.keys.index_select(0, idx)
        l.values = l.values.index_select(0, idx)
    for l in gdn_n.values():
        r, c = e5._get(l.recurrent_states), e5._get(l.conv_states)
        e5._set_like(l, "recurrent_states", r.index_select(0, idx))
        e5._set_like(l, "conv_states", c.index_select(0, idx))
    return new


def teacher_force_logp_batch(model, past_k, warm_rows, gens, device):
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
    warm_b = warm_rows
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


def make_buckets(items, bsz, tok, shuffle=True, seed=0):
    """Chia items thanh cac LO CUNG DO DAI PROMPT CHINH XAC.

    Vi sao phai chinh xac chu khong dem: dem prompt pha attention (do duoc sai
    96%) va pha GDN nang hon (3,7 lan) -- GDN la trang thai hoi quy chay doc
    ca chuoi, mot token dem la doi trang thai. Nen thay vi dem, gom cac mau
    tinh co cung do dai.

    Do duoc tren dung pool gsm8k co cau truc (2073 mau, do dai 41-201 token,
    113 do dai khac nhau): B=2 phu 97,1% mau vao lo day, B=4 phu 91,8%,
    B=8 phu 84,5%. Phan le KHONG bi bo -- chay o lo nho hon."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_len = defaultdict(list)
    for it in items:
        by_len[len(tok(it["prompt"], truncation=True,
                       max_length=2048)["input_ids"])].append(it)
    buckets = []
    for L in sorted(by_len):
        lst = by_len[L]
        if shuffle:
            rng.shuffle(lst)
        for i in range(0, len(lst), bsz):
            buckets.append(lst[i:i + bsz])
    if shuffle:
        rng.shuffle(buckets)
    return buckets


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


def reward_of_gsm8k(item, text, w):
    """Reward THAT tren gsm8k (khong phai proxy EBA): dung CHINH grader
    ext_bench.score_text da dung trong eval_big.py/run_gsm_traintest.sh --
    khong bia mot ham cham diem moi (rule 15: thang do moi phai doc tay
    truoc khi tin, ham nay DA duoc doc tay/dung o nhieu noi trong du an)."""
    hit = gd.score_item(item, text)
    return float(hit), {"C": float(hit)}


def reward_of_gsm_struct(item, text, w):
    """BUOC 3 -- reward PHAN RA tren dinh dang co cau truc (user chot 2026-09-05).

    4 thanh phan doc lap cong lai (kieu Unsloth: nhieu ham reward rieng, khong
    mot diem gop), so theo NOI DUNG DA PARSE chu khong theo chuoi chu:
      R_ent  F1 tap (thuc the, gia tri) -- PHAT NANG -2,0 khi gan SAI so
      R_rel  F1 tap quan he da chuan hoa
      R_step ty le gia tri trung gian dung THU TU (LCS)
      R_ans  chia bac theo sai so tuong doi; hong format -1,0
    Nho vay mapper bi cham diem theo "co di qua dung cac dai luong 9B da di
    qua khong", khong phai "co chep dung chu khong" -> chong hoc vet."""
    total, d = gs.score(text, item["gold_struct_parsed"],
                        item["gold_struct_parsed"].get("answer"), w)
    return total, {"C": float(d["ans"] > 0), **d}


def load_gsm_struct_pool(data_path, struct_gold_path, limit=0):
    """Chi lay item CO gold cau truc (Buoc 0 da loc 2 tang: parse duoc VA dap
    so dung). Parse san gold mot lan de vong RL khong parse lai moi buoc."""
    data = json.loads(Path(data_path).read_text())
    sg = json.loads(Path(struct_gold_path).read_text())
    items = []
    for sp in ("train", "val"):
        for it in data.get(sp, []):
            g = sg.get(it.get("id", ""))
            if it.get("kind") == "gsm8k" and g and g.get("gold"):
                it = dict(it)
                it["gold"] = g["gold"]
                it["gold_struct_parsed"] = gs.parse(g["gold"])
                items.append(it)
    if limit:
        items = items[:limit]
    print(f"gsm8k CO CAU TRUC: {len(items)} mau", flush=True)
    return items


def load_gsm8k_pool(data_path, pseudo_gold_path, limit):
    """Doc dung format /content/train_items_gsm.json (data['train'], kind==
    'gsm8k') -- CHINH tap run_gsm_traintest.sh dung cho 'TRAIN', tach hoan
    toan khoi 100 mau NIEM PHONG (gsm8k_train() trong gen_data.py: split rieng
    voi test da do). Ap dung pseudo-gold 9B tu sinh (da loc CHI item 9B lam
    DUNG, gen_pseudo_vllm.py) giong HET co che e9_joint.py dong 213-227: item
    9B lam sai GIU gold goc cua bo du lieu, khong bao gio day mapper hoc theo
    quy dao sai cua 9B."""
    data = json.loads(Path(data_path).read_text())
    items = [it for it in data["train"] if it.get("kind") == "gsm8k"]
    if limit:
        items = items[:limit]
    n_rep = 0
    if pseudo_gold_path and Path(pseudo_gold_path).exists():
        pg = json.loads(Path(pseudo_gold_path).read_text())
        for it in items:
            g = pg.get(it.get("id", ""))
            if g and g.get("gold"):
                it["gold"] = g["gold"]
                it["pseudo"] = True
                n_rep += 1
        print(f"pseudo-gold gsm8k: thay {n_rep}/{len(items)} item bang quy "
              f"dao 9B tu sinh (con lai giu gold goc cua bo du lieu)",
              flush=True)
    else:
        print(f"CANH BAO: khong thay pseudo-gold {pseudo_gold_path} -- "
              f"tat ca {len(items)} item dung gold goc (khong phai CoT 9B)",
              flush=True)
    return items


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
    ap.add_argument("--struct-gold", default="/content/struct_gold_gsm.json")
    ap.add_argument("--w-ent", type=float, default=0.25)
    ap.add_argument("--w-rel", type=float, default=0.25)
    ap.add_argument("--w-step", type=float, default=0.2)
    ap.add_argument("--w-ans", type=float, default=1.0)
    ap.add_argument("--task", default="eba",
                    choices=["eba", "gsm8k", "gsm8k_struct"],
                    help="eba = du lieu tong hop Entity-Binding (proxy); "
                         "gsm8k = gsm8k THAT + pseudo-gold 9B tu sinh -- "
                         "dung engine GRPO nhu nhau, chi doi nguon du lieu "
                         "+ ham reward (user 2026-09-04: gop thanh 1 pipeline)")
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--difficulty-max", type=int, default=1,
                    help="[chi --task eba] chi dung item co difficulty <= "
                         "gia tri nay (0..3)")
    ap.add_argument("--gsm-data", default="/content/train_items_gsm.json")
    ap.add_argument("--pseudo-gold", default="/content/pseudo_gold_gsm2.json")
    ap.add_argument("--gsm-limit", type=int, default=1200,
                    help="[chi --task gsm8k] gioi han so item TRAIN doc tu "
                         "gsm-data (0 = tat ca)")
    ap.add_argument("--gold-cap", type=int, default=0,
                    help="tran token gold cho anchor-CE. 0 = tu dong: 64 cho "
                         "eba, 256 cho gsm8k (khop GOLD_CAP cua gen_pseudo_vllm.py)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=6, help="so completion/prompt")
    ap.add_argument("--bsz", type=int, default=1,
                    help="so MAU (prompt) gop chung mot lo. Lo chi gom cac mau "
                         "CO DO DAI PROMPT CHINH XAC BANG NHAU (khong dem -- "
                         "dem pha attention 96%% va GDN nang hon). So hang "
                         "decode = bsz*k; do 2026-09-05 cho thay thoi gian moi "
                         "buoc decode gan nhu khong doi tu 2 den 16 hang, nen "
                         "tang bsz gan nhu mien phi. Rang buoc that la VRAM.")
    ap.add_argument("--tf-chunk", type=int, default=2,
                    help="so hang moi mieng teacher-force (pha 2). OOM do duoc "
                         "2026-09-05 nam o lop GDN cua pha nay, khong phai o "
                         "sampling -> pha 1 gop rong bsz*k hang, pha 2 chia "
                         "mieng nay va cong don gradient. Tong loss khong doi.")
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
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="luu them checkpoint step{N} (khong ghi de best/last) "
                         "moi N buoc -- 0=tat. Can de soi CA DUONG CONG sau "
                         "nay (McNemar tren nhieu diem), khong chi best/last "
                         "-- lan truoc 2 diem cho thay khong don dieu, phai "
                         "nhin ca duong moi ket luan duoc.")
    ap.add_argument("--out", default="/content/eba_grpo_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="eba_grpo_v1")
    ap.add_argument("--start-step", type=int, default=0,
                    help="NOI LAI sau khi Colab recycle: bat dau tu buoc N thay "
                         "vi 1. Vong lap lay item theo (step-1) %% len(train) "
                         "nen dat dung N se di TIEP tu cho da dung, khong lap "
                         "lai phan da hoc. Dung kem --init-mapper tro vao "
                         "checkpoint cuoi. Can vi 1 epoch RL ~13 gio ma phien "
                         "Colab khong song lau vay (user chon huong 1: chay du "
                         "1 epoch, noi qua nhieu phien).")
    ap.add_argument("--mapper-ckpt", type=int, default=1,
                    help="1 = tinh lai map_attn trong backward thay vi giu "
                         "activation fp32 (~0,5GB/1K ctx). Can cho task "
                         "gsm8k_struct: gen_len 320 (dai hon 200 cua vong "
                         "truoc) lam K=3 OOM neu khong bat.")
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

    # ---- data ----
    if args.task == "gsm8k_struct":
        items = load_gsm_struct_pool(args.gsm_data, args.struct_gold, args.gsm_limit)
        gold_cap = args.gold_cap or 320
        do_reward = reward_of_gsm_struct
        w = {"ent": args.w_ent, "rel": args.w_rel,
             "step": args.w_step, "ans": args.w_ans}
        if args.anchor_w:
            print(f"CANH BAO: --task gsm8k_struct nen chay anchor_w=0 (user chot "
                  f"bo hann anchor-CE o buoc RL -- day la cho gay hoc vet); "
                  f"dang la {args.anchor_w}", flush=True)
    elif args.task == "gsm8k":
        items = load_gsm8k_pool(args.gsm_data, args.pseudo_gold, args.gsm_limit)
        gold_cap = args.gold_cap or 256
        do_reward = reward_of_gsm8k
    else:
        items = eba.build(args.n_items, "/tmp/_eba_items.json", seed=args.seed)
        items = [it for it in items if it["difficulty"] <= args.difficulty_max]
        gold_cap = args.gold_cap or 64
        do_reward = reward_of
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(8, int(len(items) * args.val_frac))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"{args.task}: {len(train_items)} train, {len(val_items)} val, "
          f"gold_cap={gold_cap}", flush=True)

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
    mapper.ckpt = bool(args.mapper_ckpt)
    print(f"nap xong {time.time()-t0:.0f}s | mapper.ckpt={mapper.ckpt}", flush=True)

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

    def meta_for(t, b=1):
        """t = do dai ngu canh, b = KICH THUOC LO.

        BUG DA SUA (bat khi do batch B=2 o sft_struct.py, 2026-09-05): ban dau
        chi va chieu batch cho lop ATTENTION (k/v) ma bo sot lop GDN
        (rec/conv) -> template giu conv_states o batch 1 trong khi dau vao
        batch B -> 'Expected size 1 but got size 2'. build_student_past thay
        recurrent_states bang dau ra mapper (dung batch) nhung conv_states lay
        zeros_like TU TEMPLATE, nen template sai batch la hong."""
        m = _copy.deepcopy(base_meta)
        m["cache_ints"] = {k: (t if v == T_BASE else v)
                           for k, v in m["cache_ints"].items()}
        for lay in m["layers"]:
            lay["ints"] = {k: (t if v == T_BASE else v)
                           for k, v in lay["ints"].items()}
            keys = ("k", "v") if lay["kind"] == "a" else ("rec", "conv")
            for key in keys:
                sh, dt = lay[key]
                sh = tuple(t if d == T_BASE else d for d in sh)
                lay[key] = ((b,) + sh[1:], dt)   # chieu 0 = batch, ca a lan g
        return m

    def student_past_grad(cut):
        """CO GRAD -- goi DUNG MOT LAN moi buoc cho CA LO, moi nhanh sau do
        chi clone_cache_repeat_grad() tu day (van gan graph)."""
        src = prefill_tbptt(cut, args.tbptt)
        tpl = e5.build_template_from_meta(
            probe_t, meta_for(cut.shape[1], cut.shape[0]))
        st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    def enc(it):
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=2048)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        gold_ids = tok_t(it["gold"], add_special_tokens=False,
                         return_tensors="pt")["input_ids"][:, :gold_cap].to("cuda")
        return cut, warm, gold_ids

    def enc_bucket(items):
        """Gop B mau CUNG DO DAI PROMPT thanh mot lo. KHONG DEM: dem prompt se
        pha attention (da do: sai 96%) va GDN (te hon 3,7 lan) -- nen lo duoc
        chia theo do dai token CHINH XAC (make_buckets). Lo le chay B nho hon,
        khong bo mau nao."""
        es = [tok_t(it["prompt"], return_tensors="pt", truncation=True,
                    max_length=2048)["input_ids"] for it in items]
        Ls = {e.shape[1] for e in es}
        assert len(Ls) == 1, f"lo lech do dai prompt: {sorted(Ls)}"
        e = torch.cat(es, 0).to("cuda")
        return e[:, :-WARM_P], e[:, -WARM_P:]

    import bitsandbytes as bnb
    groups = [{"params": mapper.params, "lr": args.lr},
              {"params": lora_params, "lr": args.lora_lr},
              {"params": lora_t_params, "lr": args.lora_lr}]
    opt = bnb.optim.Adam8bit(groups)

    def gib():
        return torch.cuda.max_memory_allocated() / 2**30

    @torch.no_grad()
    def run_val(n):
        """Bao cao reward TRUNG BINH tren val -- KHONG phai vong val gsm8k
        day du cua e9_joint/run_gsm_traintest.sh (khong niem phong, chi tap
        con TRAIN de theo doi GRPO co tien khong). Dung greedy (nhiet do 0)
        cho on dinh giua cac moc."""
        keys = (("A", "B", "C") if args.task == "eba" else
                ("ent", "rel", "step", "ans", "C") if args.task == "gsm8k_struct"
                else ("C",))
        agg = {kk: [] for kk in keys}
        stop_t = torch.tensor(sorted(STOPS), device="cuda")
        # val cung gop lo theo do dai chinh xac: 32 mau x 224 token x 95ms
        # = ~11 phut neu chay tung mau; gop lo dua ve ~3 phut (cung co che
        # weight-bound da do o probe_decode_speed.py).
        for bk in make_buckets(val_items[:n], args.bsz, tok_t, shuffle=False):
            cut, warm = enc_bucket(bk)
            src = e5.prefill_chunked(model_s, cut)
            tpl = e5.build_template_from_meta(
                probe_t, meta_for(cut.shape[1], cut.shape[0]))
            st = e5.build_student_past(tpl, src, mapper)
            o = model_t(input_ids=warm, past_key_values=st, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            toks = [inp]
            done = torch.isin(inp[:, 0], stop_t)
            for t in range(args.gen_len - 1):
                o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                toks.append(inp)
                if (t + 2) % 16 == 0:
                    done |= torch.isin(torch.cat(toks[-16:], 1), stop_t).any(1)
                    if bool(done.all()):
                        break
            for it, g in zip(bk, torch.cat(toks, 1).tolist()):
                cutg = len(g)
                for j, tkn in enumerate(g):
                    if tkn in STOPS:
                        cutg = j + 1
                        break
                txt = tok_t.decode(g[:cutg], skip_special_tokens=True)
                if args.task == "eba":
                    s = eba.score_eba(it, txt)
                elif args.task == "gsm8k_struct":
                    _, s = do_reward(it, txt, w)
                else:
                    s = {"C": float(gd.score_item(it, txt))}
                for kk in keys:
                    agg[kk].append(s[kk])
            del st, tpl, src, cur, o, toks
            torch.cuda.empty_cache()
        return {kk: round(sum(v) / max(len(v), 1), 3) for kk, v in agg.items()}

    results = {"args": vars(args), "val": [], "train": []}
    # NOI LAI: giu lai lich su val/train cua cac phien truoc, neu khong moi lan
    # recycle se xoa sach duong cong da do duoc.
    _rp = out / "results.json"
    if args.start_step and _rp.exists():
        try:
            old = json.loads(_rp.read_text())
            results["val"] = old.get("val", [])
            results["train"] = old.get("train", [])
            print(f"noi lai lich su: {len(results['val'])} moc val, "
                  f"{len(results['train'])} moc train", flush=True)
        except Exception as ex:
            print(f"khong doc duoc results.json cu: {type(ex).__name__}", flush=True)

    def save_results():
        (out / "results.json").write_text(json.dumps(results, indent=1))

    def save_ckpt(tag):
        """Luu local + upload HF trong DUNG 1 COMMIT (CommitOperationAdd +
        create_commit) thay vi ~8-10 upload_file rieng le -- fix rate-limit
        60 commit/gio da dinh that trong eba_grpo_v2c (log 'HF-UP FAIL...429
        Too Many Requests'), khong phai loi dang nhap/xac thuc (cung token
        da thanh cong hang chuc lan truoc do trong dung phien)."""
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        model_t.save_pretrained(str(out / f"lorat_{tag}"))
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(
            path_in_repo=f"{args.hf_prefix}/mapper_{tag}.pt",
            path_or_fileobj=str(out / f"mapper_{tag}.pt"))]
        for sub in (f"lora_{tag}", f"lorat_{tag}"):
            for f in sorted((out / sub).glob("*")):
                if f.is_file():
                    ops.append(CommitOperationAdd(
                        path_in_repo=f"{args.hf_prefix}/{sub}/{f.name}",
                        path_or_fileobj=str(f)))
        try:
            _api.create_commit(repo_id=args.hf_repo, operations=ops,
                               commit_message=f"{args.hf_prefix} ckpt {tag}")
            print(f"HF-UP {tag} ({len(ops)} file, 1 commit)", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL ckpt {tag}: {type(ex).__name__}: {ex}",
                  flush=True)

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

    best = max([v[1].get("C", -1e9) for v in results["val"]], default=-1e9)
    t_start = time.time()
    if args.start_step:
        print(f"NOI LAI tu buoc {args.start_step + 1}/{args.steps} "
              f"(best cu = {best:.3f})", flush=True)
    buckets = make_buckets(train_items, args.bsz, tok_t, seed=args.seed)
    print(f"lo train: {len(buckets)} lo tu {len(train_items)} mau "
          f"(bsz={args.bsz}, k={args.k} -> {args.bsz * args.k} hang/decode); "
          f"1 epoch = {len(buckets)} buoc", flush=True)
    for step in range(args.start_step + 1, args.steps + 1):
        bk = buckets[(step - 1) % len(buckets)]
        B = len(bk)
        cut, warm = enc_bucket(bk)
        gold_ids = enc(bk[0])[2] if args.anchor_w > 0 else None

        with clock("student_past_grad"):
            st0 = student_past_grad(cut)

        # pha 1: B mau x K nhanh sampling trong MOT vong decode (B*K hang).
        # Chi phi moi buoc decode gan nhu khong doi tu 2 den 16 hang (do
        # 2026-09-05) -> gop them mau vao lo gan nhu mien phi. Thu tu hang:
        # hang r <-> mau r % B, nhanh r // B (do .repeat lap ca khoi B).
        with clock("sampling(pha1)"):
            branch_k = clone_cache_repeat(st0, args.k)
            warm_rows = warm.repeat(args.k, 1)
            gens, texts = sample_rollout_batch(
                model_t, tok_t, branch_k, warm_rows, args.gen_len,
                args.temperature, STOPS)
            del branch_k
        with clock("reward"):
            rewards, sub = [], []
            for r_i, txt in enumerate(texts):
                r, s = do_reward(bk[r_i % B], txt, w)
                rewards.append(r)
                sub.append(s)

        # BUG DA SUA (2026-09-03): `continue` o day truoc kia nhay qua LUON
        # ca khoi val/checkpoint ben duoi -- neu mot buoc == boi so val_every
        # (vd 50,100,150...) VUA VAN roi dung vao nhom reward dong nhat (rat
        # hay xay ra khi policy da tot len, K mau deu giong diem), thi CA
        # MOC VAL DO BIEN MAT, khong ghi checkpoint. Da bat qua thuc te (v2b
        # resume: tu buoc 200 chay ~500 buoc lien tuc khong lan nao dung moc
        # val/checkpoint nao ca, vi toan trung vao buoc bi bo qua). Sua bang
        # cach: KHONG continue, chi bo qua backward, van roi xuong duoi.
        # Advantage tinh RIENG TRONG NHOM K CUA TUNG MAU -- KHONG tron reward
        # giua cac mau khac nhau trong lo (do khac de, thang diem khac nhau ->
        # tron vao la hong dung tinh than GRPO). Mau co reward dong nhat ca
        # nhom nhan adv = 0 (khong dong gop gradient) thay vi bo ca lo.
        adv = [0.0] * len(rewards)
        n_active = 0
        for i in range(B):
            g = group_advantage([rewards[i + j * B] for j in range(args.k)])
            if g is None:
                continue
            n_active += 1
            for j in range(args.k):
                adv[i + j * B] = g[j]
        skipped = n_active == 0
        pg_loss = torch.tensor(0.0)
        anchor_ce = torch.tensor(0.0)
        if skipped:
            if step % 20 == 0:
                print(f"buoc {step}: ca {B} mau deu co reward dong nhat trong "
                      f"nhom -> bo qua gradient (VAN cham val/checkpoint)",
                      flush=True)
        else:
            # pha 2: teacher-force CO grad, CHIA MIENG tf_chunk hang mot.
            # OOM do duoc nam o lop GDN cua pha nay (no luu hoat hoa 32 lop,
            # ty le thuan so hang), KHONG nam o sampling -- nen pha 1 gop rong
            # con pha 2 chia nho, lan nguoc ngay tung mieng va CONG DON
            # gradient. Tong loss dong nhat voi lam mot lan:
            #   sum_mieng[-(adv*lp).sum()/K] == -(adv*lp).sum()/K
            # Chia cho K = trung binh trong nhom, cong don qua cac mau -> dung
            # ngu nghia accum=B: do lon gradient MOI MAU giu nguyen nhu khi
            # chay B=1, chi gop lai thanh 1 lan opt.step(). (Hoc phi
            # sft_struct: accum=16 -> 132 lan cap nhat/epoch, chi hoc duoc
            # <think>; accum=4 -> 527 lan, hoc du cau truc.)
            n_rows = len(gens)
            need_anchor = bool(args.anchor_w > 0 and B == 1
                               and gold_ids is not None and gold_ids.shape[1] >= 1)
            adv_all = torch.tensor(adv, device="cuda")
            row_item = torch.tensor([r % B for r in range(n_rows)],
                                    device="cuda")
            opt.zero_grad(set_to_none=True)
            pg_tot = 0.0
            chunks = [(s, min(s + args.tf_chunk, n_rows))
                      for s in range(0, n_rows, args.tf_chunk)]
            with clock("teacher_force(pha2)+backward"):
                for ci, (s, e_) in enumerate(chunks):
                    idx = row_item[s:e_]
                    branch = clone_cache_index(st0, idx)
                    lp = teacher_force_logp_batch(
                        model_t, branch, warm.index_select(0, idx),
                        gens[s:e_], "cuda")
                    loss_c = -(adv_all[s:e_] * lp).sum() / args.k
                    # retain_graph cho moi mieng TRU mieng cuoi (graph cua st0
                    # = mapper + 4B dung chung cho ca cac mieng). Neu con
                    # anchor-CE phia sau thi mieng cuoi cung phai giu graph.
                    loss_c.backward(retain_graph=(ci < len(chunks) - 1
                                                  or need_anchor))
                    pg_tot += float(loss_c.detach())
                    del branch, lp, loss_c
            pg_loss = torch.tensor(pg_tot)

            anchor_ce = torch.tensor(0.0, device="cuda")
            with clock("anchor_ce"):
                if need_anchor:
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
                # gradient cua pg da duoc cong don tung mieng o tren; o day
                # chi con anchor-CE (neu bat) roi cap nhat MOT lan.
                if need_anchor:
                    (args.anchor_w * anchor_ce).backward()
                opt.step()

        del st0
        gc.collect()
        torch.cuda.empty_cache()

        mean_r = sum(rewards) / len(rewards)
        mean_c = sum(x["C"] for x in sub) / len(sub)
        if step % 10 == 0 and not skipped:
            results["train"].append([step, round(mean_r, 3), round(mean_c, 3)])
            tot = sum(T_ACC.values()) or 1.0
            share = " ".join(f"{k} {100*v/tot:.0f}%"
                             for k, v in sorted(T_ACC.items(),
                                                key=lambda x: -x[1]))
            print(f"buoc {step}/{args.steps} pg={pg_loss.item():.4f} "
                  f"anchor_ce={anchor_ce.item():.4f} reward_tb={mean_r:.3f} "
                  f"C_tb={mean_c:.3f} {(time.time()-t_start)/max(step-args.start_step,1):.2f}s/buoc "
                  f"peak={gib():.2f}GiB", flush=True)
            print(f"    thoi gian: {tot/10:.2f}s/buoc do duoc | {share}",
                  flush=True)
            T_ACC.clear()

        if args.sanity and step >= args.sanity:
            print(f"SANITY xong {step} buoc, peak={gib():.2f}GiB, "
                  f"{(time.time()-t_start)/max(step-args.start_step,1):.2f}s/buoc", flush=True)
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
            if args.snapshot_every and step % args.snapshot_every == 0:
                save_ckpt(f"step{step}")

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
