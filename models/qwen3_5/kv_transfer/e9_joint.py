"""e9_joint -- KIEN TRUC 2 LOP (user chot 2026-08-28).

    4B + LoRA  --(cache)-->  mapper  -->  27B  --> CE(gold)
       ^_______________ gradient chay nguoc het duong _______|

Khac han moi vong train truoc (e6v3_ce.py): o do 4B chi la NGUON tinh mot
lan roi spill cache ra dia, 27B train mapper doc lai tu dia -- 4B khong bao
gio hoc gi. O day 4B CUNG HOC: LoRA cua no bi ep "doc ho cho 27B". Sau khi
train, merge LoRA vao 4B -> luc serve KHONG co module thua tren duong nong.

DIEU KIEN VAT LY (do that, probe_joint_lora.py, 6 luot):
  - hai model cung tren GPU = 16,16 GiB tinh, con trong 5,54 GiB
  - thu pham: prefill 4B CO GRAD (+3,34 GiB ngay o T=256)
  - gradient checkpointing BI LOAI VE NGUYEN TAC: transformers ep
    use_cache=False khi bat GC, ma forward 4B o day ton tai CHINH DE sinh
    cache -> hai thu loai tru nhau
  - TBPTT mo duoc cong: T-w token dau no_grad, w token cuoi co grad

HAI CAI BAY DA XU LY (cai thu 2 la bug that neu bo qua):
 1. Template 27B: build_template_from_meta can "bo xuong" shape cho DUNG do
    dai T. Chay teacher prefill moi buoc thi qua dat. Giai: suy meta tu mot
    meta goc (chi doi chieu T) + TU KIEM bang meta THAT o vai do dai
    (--verify-meta). Sai thi dung ngay, khong doan.
 2. CAT TRAI khi tokenize. Moi prompt trong bo nay dat CAU HOI O CUOI
    (musr: narrative 1000-1500 token roi moi hoi). truncation mac dinh cua
    tokenizer cat PHAI = cat mat cau hoi -> item thanh vo nghia ma van chay
    tron, khong bao loi. Bat buoc truncation_side='left'.

Loss: CE(gold) co trong so, KHONG co KL/aux/dense -- ba thanh phan do can
teacher logits tien tinh (B1), ma o che do joint thi cache thay doi moi buoc
nen khong tien tinh duoc. v3.1 da do: CE-gold la thanh phan quyet dinh.
"""

import argparse
import copy
import gc
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path

import torch

_H = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _H / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# PHAI patch TRUOC khi nap model (hoc phi 3 probe: 5.15 update_recurrent_state
# lam .copy_() IN-PLACE len state -> vo autograd khi state mang grad)
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

e5 = _load("e5_train")
gd = _load("gen_data")

WARM_P = 5
FIRST_W = 3.0
GOLD_MAX = 64
GMAX = {"bfcl": 16, "needle": 12, "ifstruct": 96, "pbtable": 64,
        "gsm8k": 256, "bbh": 24, "musr": 8,
        "suite_rag": 16, "suite_mid": 16, "suite_math": 16, "suite_swe": 16}
GEN_LEN = {"bfcl": 24, "needle": 16, "ifstruct": 160, "pbtable": 120,
           "gsm8k": 320, "bbh": 48, "musr": 24,
           "suite_rag": 24, "suite_mid": 24, "suite_math": 24, "suite_swe": 24}
NEW_KINDS = {"gsm8k", "bbh", "musr", "suite_rag", "suite_mid",
             "suite_math", "suite_swe"}
SEED = 7


class _SkipItem(Exception):
    """Bo qua mot item ma VAN di tiep het than vong lap (de khoi val o cuoi
    khong bi nhay qua) — thay cho `continue`."""


def gib():
    return torch.cuda.max_memory_allocated() / 2**30


# BAO DO THAT (probe_joint_lora, 42 cau hinh, 2026-08-28). gold = SO VI TRI
# feed vao 27B; moi vi tri giu them state GDN cho backward nen no dat hon
# ctx nhieu. Tuong CUNG o gold=64 voi MOI T (256..2048) — khong phai gioi han
# tron, la nguong cap phat.
#
#     T      gold toi da    peak GiB
#     256        48           20,75
#     512        48           20,92
#    1024        48           21,24
#    1536        16           20,97
#    2048        16           21,30
# Bao do PHU THUOC CAP MODEL (27B chiem 12,7GiB nen chat; 9B nhe hon nhieu)
# -> khong duoc ghim cung. Doi qua --gold-envelope.
GOLD_ENVELOPE = [(1024, 48), (2048, 16)]


def parse_envelope(spec):
    """'1024:48,2048:16' -> [(1024,48),(2048,16)] (sap theo T tang dan)."""
    if not spec:
        return GOLD_ENVELOPE
    out = []
    for part in spec.split(","):
        t, g = part.split(":")
        out.append((int(t), int(g)))
    return sorted(out)


def gold_cap_for(t, hard_cap, envelope=None):
    env = envelope or GOLD_ENVELOPE
    for t_max, g in env:
        if t <= t_max:
            return min(g, hard_cap)
    return min(env[-1][1], hard_cap)


# ---------------------------------------------------------------- template --

def meta_for_len(base, t_base, t):
    """Suy 'bo xuong' template cho do dai t tu meta goc do o t_base.

    Chi co shape cua keys/values (chieu T) va vai truong int bang t_base la
    phu thuoc do dai; state GDN thi khong. Ham nay PHAI qua verify_meta()
    truoc khi tin -- neu co truong int nao trung t_base mot cach tinh co thi
    kiem tra se bat duoc."""
    m = copy.deepcopy(base)
    m["cache_ints"] = {k: (t if v == t_base else v)
                       for k, v in m["cache_ints"].items()}
    for lay in m["layers"]:
        lay["ints"] = {k: (t if v == t_base else v)
                       for k, v in lay["ints"].items()}
        if lay["kind"] == "a":
            for key in ("k", "v"):
                sh, dt = lay[key]
                lay[key] = (tuple(t if d == t_base else d for d in sh), dt)
    return m


def verify_meta(model_t, base, t_base, lens):
    """Doi chieu meta SUY RA voi meta THAT (chay prefill that o tung do dai).
    Sai mot li = template lech vi tri = cache vo nghia ma khong bao loi."""
    ok = True
    for t in lens:
        ids = torch.randint(1000, 5000, (1, t), device="cuda")
        with torch.no_grad():
            past = e5.prefill_chunked(model_t, ids)
        real = e5.cache_meta(past)
        derived = meta_for_len(base, t_base, t)
        if real != derived:
            ok = False
            print(f"  VERIFY-META LECH o T={t}", flush=True)
            for k in real["cache_ints"]:
                if real["cache_ints"][k] != derived["cache_ints"].get(k):
                    print(f"    cache_ints[{k}]: that={real['cache_ints'][k]} "
                          f"suy={derived['cache_ints'].get(k)}", flush=True)
            for j, (r, d) in enumerate(zip(real["layers"], derived["layers"])):
                if r != d:
                    print(f"    layer {j}: that={r} suy={d}", flush=True)
                    break
        else:
            print(f"  verify-meta T={t}: KHOP", flush=True)
        del past, ids
        gc.collect()
        torch.cuda.empty_cache()
    return ok


# ------------------------------------------------------------------- data ---

def load_data(args, tok_s):
    e6 = _load("e6v3_ce")
    data = e6.build_data(tok_s, max_ctx=min(args.max_ctx, 2000))
    if args.data_file:
        extra = json.loads(Path(args.data_file).read_text())
        data["train"] += extra["train"]
        data["val"] += extra["val"]
    # ifstruct KHONG co truong "gold": trong e6v3_ce gold cua no do 27B TU
    # SINH o buoc B0 (pseudo-gold), ma che do joint khong co B0 -> tok(None)
    # nem "You need to specify either `text` or `text_target`". Loai han, va
    # ghi ro so luong: phan quyet v3.5 da bo ifstruct/pbtable khoi thang do
    # chinh (no cua DE, khong phai cua mapper) nen mat mat nay khong dang tiec.
    for k in ("train", "val"):
        before = len(data[k])
        data[k] = [it for it in data[k] if it.get("gold")]
        if before != len(data[k]):
            print(f"loai {before - len(data[k])} item khong co gold khoi {k}",
                  flush=True)
    # PSEUDO-GOLD (user 2026-08-28: "cho no hoc ca buoc reasoning cua model
    # large", "can map gan 9b nhat"): thay gold tham chieu bang chinh quy dao
    # 9B tu di — nhung CHI voi item 9B lam DUNG (gen_pseudo.py da loc). Item
    # 9B lam sai giu gold cu: khong mat mau, khong day mapper suy luan sai.
    if args.pseudo_gold and Path(args.pseudo_gold).exists():
        pg = json.loads(Path(args.pseudo_gold).read_text())
        n_rep = 0
        for split in ("train", "val"):
            for it in data[split]:
                g = pg.get(it.get("id", ""))
                if g and g.get("gold"):
                    it["gold"] = g["gold"]
                    it["pseudo"] = True
                    n_rep += 1
        print(f"pseudo-gold: thay {n_rep} item bang quy dao 9B tu sinh",
              flush=True)
    elif args.pseudo_gold:
        print(f"CANH BAO: khong thay {args.pseudo_gold} — chay voi gold cu",
              flush=True)
    if args.drop_kinds:
        drop = set(args.drop_kinds.split(","))
        for k in ("train", "val", "test"):
            n0 = len(data[k])
            data[k] = [it for it in data[k] if it["kind"] not in drop]
            print(f"loai {sorted(drop)} khoi {k}: {n0} -> {len(data[k])}",
                  flush=True)
    random.Random(SEED).shuffle(data["train"])
    random.Random(SEED).shuffle(data["val"])
    from collections import Counter
    for k in ("train", "val", "test"):
        print(f"{k}: {len(data[k])} — "
              f"{dict(Counter(x['kind'] for x in data[k]))}", flush=True)
    return data


# ------------------------------------------------------------------ train ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--data-file", default="/content/train_items.json")
    ap.add_argument("--pseudo-gold", default="",
                    help="file do gen_pseudo.py sinh: thay gold tham chieu "
                         "bang quy dao model dich TU DI (chi item no lam "
                         "dung). Muc tieu la bam sat self, khong phai vuot.")
    ap.add_argument("--max-ctx", type=int, default=2048,
                    help="user 2026-08-28 'tang tran 1024-2048'; da do: 2048 "
                         "chay tron voi tbptt=64 (gold tu dong ha ve 16 o do)")
    ap.add_argument("--tbptt", type=int, default=64,
                    help="do duoc: w=128 OOM o ctx>=1536; w=64 chay tron toi "
                         "2048 (probe_joint_lora luot 6)")
    ap.add_argument("--gold-cap", type=int, default=48,
                    help="tran token gold. gold = SO VI TRI feed vao 27B nen "
                         "an bo nho that (do trong probe: gold 256 khac han "
                         "gold 16). Cat theo cau hinh da do duoc.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="lr cua mapper")
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-n", type=int, default=40)
    ap.add_argument("--ce-floor", type=float, default=0.2)
    ap.add_argument("--drop-kinds", default="",
                    help="Loai han cac ho nay khoi train VA val (phay ngan "
                         "cach). User chot 2026-08-30 giai doan 1: "
                         "gsm8k,suite_math — tap trung cac ho DA GHI NHAN "
                         "mapper chay. Gia phai tra: mat 2583 quy dao "
                         "pseudo-gold cua gsm8k (nguon giam sat sach nhat).")
    ap.add_argument("--accum", type=int, default=1,
                    help="Gop gradient qua N mau roi moi buoc optimizer. "
                         "KHONG nhanh hon (van tung ay luot forward), nhung "
                         "huong gradient bot nhieu: CE cua MOT item dao dong "
                         "0,02-2,46 tuy item (do that), nen batch hieu dung 1 "
                         "la di theo tung mau mot. Khac han batch THAT: gop "
                         "gradient khong can DEM prompt nen khong dung toi lop "
                         "GDN hoi quy — an toan vo dieu kien.")
    ap.add_argument("--patience", type=int, default=0,
                    help="Dung khi val score KHONG lap ky luc trong N moc lien "
                         "tiep (0 = tat). Ly do them: lich su cho thay 3/4 luot "
                         "train bi cat ngang vi runtime bi thu hoi hoac vi het "
                         "so buoc — KHONG luot nao dung vi hoi tu. Do mot mapper "
                         "chua hoi tu thi con so chi la CAN DUOI. Voi patience, "
                         "--steps chi con la tran an toan.")
    ap.add_argument("--init-mapper", default="",
                    help="warm-start tu mapper da co (v427_4k / v49)")
    ap.add_argument("--init-lora", default="",
                    help="nap lai LoRA da luu (thu muc lora_last/ hoac "
                         "lora_best/) — de tiep tuc mot run bi dung giua "
                         "chung ma khong mat cong da train")
    ap.add_argument("--out", default="/content/joint_v1")
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b")
    ap.add_argument("--hf-prefix", default="joint_v1")
    ap.add_argument("--verify-meta", default="256,512,1024")
    ap.add_argument("--no-offload", action="store_true",
                    help="tat CPU-offload embed/lm_head cua model dich. BAT "
                         "co nay cho 9B: offload chi can cho 27B, con voi 9B "
                         "no lam moi buoc greedy phai qua lm_head tren CPU "
                         "(val 40 mau: ~14 phut thay vi vai phut).")
    ap.add_argument("--w-entity", type=float, default=1.0,
                    help="trong so CE cho token la CHU SO hoac TEN RIENG "
                         "(viet hoa). 1.0 = tat. Nham dung thu quan sat duoc "
                         "la bi pha khi doc tay dau ra gsm8k hong.")
    ap.add_argument("--attn-rank", type=int, default=0,
                    help="hang thap cho ma tran attention cua mapper (0 = "
                         "day du 1024x1024). Attention da thang hang san "
                         "(CCA 0,98) nen khong can day du.")
    ap.add_argument("--gdn-per-head", action="store_true",
                    help="moi head GDN mot cap A,B rieng thay vi dung chung. "
                         "Cung voi --attn-rank = DOI NGAN SACH tham so tu "
                         "attention sang GDN, giu tong xap xi khong doi.")
    ap.add_argument("--gold-envelope", default="",
                    help="bao do 'T:gold,...' do bang probe_joint_lora cho "
                         "CAP MODEL dang dung (27B khac 9B). Rong = dung bao "
                         "do cua 4->27B.")
    ap.add_argument("--sanity", type=int, default=0,
                    help="chay N buoc roi dung + in VRAM/toc do")
    args = ap.parse_args()

    envelope = parse_envelope(args.gold_envelope)
    print(f"bao do gold: {envelope} (hard cap {args.gold_cap})", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi
    _api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    def hf_up(local, dest):
        """quy tac 11b: khong phong train dai khi duong upload chua song."""
        if not args.hf_repo or not os.environ.get("HF_TOKEN"):
            return
        try:
            _api.upload_file(path_or_fileobj=str(local),
                             path_in_repo=f"{args.hf_prefix}/{dest}",
                             repo_id=args.hf_repo)
            print(f"HF-UP {dest}", flush=True)
        except Exception as ex:
            print(f"HF-UP FAIL {dest}: {type(ex).__name__}: {ex}", flush=True)

    # ---- nap 27B (CPU-offload embed/lm_head: tiet kiem 4,85GiB da do) ----
    t0 = time.time()
    # CPU-offload embed/lm_head sinh ra DE cuu 27B (tiet kiem 4,85GiB tren
    # card 22GB). Voi 9B thi THUA: nen 2 model chi 6,86GiB, con trong 14,88.
    # Va no dat: moi token greedy phai chay lm_head (vocab ~152k) TREN CPU ->
    # val 40 mau mat ~14 phut thay vi vai phut. Mac dinh TAT cho 9B.
    load_tgt = e5.load_4bit if args.no_offload else e5.load_4bit_cpu_offload_io
    print(f"nap 27B/9B bang {load_tgt.__name__}", flush=True)
    tok_t, model_t = load_tgt(args.tgt_model)
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    print(f"27B nap xong {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        probe_t = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe_t)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]

    # meta goc + TU KIEM (bay 1)
    T_BASE = 512
    ids = torch.randint(1000, 5000, (1, T_BASE), device="cuda")
    with torch.no_grad():
        p0 = e5.prefill_chunked(model_t, ids)
    base_meta = e5.cache_meta(p0)
    del p0, ids
    gc.collect()
    torch.cuda.empty_cache()
    lens = [int(x) for x in args.verify_meta.split(",") if x]
    if not verify_meta(model_t, base_meta, T_BASE, lens):
        print("META SUY RA KHONG KHOP META THAT -> DUNG (khong doan mo)",
              flush=True)
        return
    print("verify-meta: TAT CA KHOP", flush=True)

    # ---- nap 4B + LoRA ----
    tok_s, model_s = e5.load_4bit(args.src_model)
    for p in model_s.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    model_s = get_peft_model(model_s, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model_s.train()
    if args.init_lora:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(str(Path(args.init_lora) / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model_s, sd)
        n_miss = len(getattr(res, "unexpected_keys", []) or [])
        print(f"nap lai LoRA tu {args.init_lora} ({len(sd)} tensor, "
              f"{n_miss} khoa la)", flush=True)
        assert n_miss == 0, "khoa LoRA khong khop — DUNG, khong train mu"
    lora_params = [p for p in model_s.parameters() if p.requires_grad]
    print(f"LoRA r={args.lora_r}: "
          f"{sum(p.numel() for p in lora_params)/1e6:.1f}M param", flush=True)

    # BAY 2: cat TRAI — moi prompt bo nay dat cau hoi O CUOI
    tok_s.truncation_side = "left"
    tok_t.truncation_side = "left"

    from transformers import AutoConfig
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    with torch.no_grad():
        probe_s = model_s(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                          use_cache=True, logits_to_keep=1).past_key_values
    a_s, g_s = e5.split_layers(probe_s)
    Hs = e5._get(next(iter(g_s.values())).recurrent_states).shape[1]
    k0 = e5._get(next(iter(a_s.values())).keys)
    attn_dim = k0.shape[1] * k0.shape[3]
    del probe_s

    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t,
                       attn_rank=args.attn_rank,
                       gdn_per_head=args.gdn_per_head)
    mapper.ckpt = True
    if args.init_mapper and Path(args.init_mapper).exists():
        mapper.load(args.init_mapper)
        print(f"warm-start mapper: {args.init_mapper}", flush=True)

    data = load_data(args, tok_s)

    def enc(it):
        """(cut, warm, gold_ids, feed). Cat TRAI (bay 2)."""
        e = tok_t(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=args.max_ctx)["input_ids"].to("cuda")
        cut, warm = e[:, :-WARM_P], e[:, -WARM_P:]
        # tran gold theo BAO DO THAT, phu thuoc do dai prompt (xem
        # GOLD_ENVELOPE): gold dai an bo nho hon ca ctx dai
        gm = min(GMAX.get(it["kind"], GOLD_MAX),
                 gold_cap_for(cut.shape[1], args.gold_cap, envelope))
        gold_ids = tok_t(it["gold"], add_special_tokens=False,
                         return_tensors="pt")["input_ids"][:, :gm].to("cuda")
        feed = torch.cat([warm, gold_ids[:, :-1]], 1)
        return cut, warm, gold_ids, feed

    # Dong ho tung chang. BAT BUOC synchronize truoc khi doc dong ho: CUDA
    # chay bat dong bo, khong dong bo thi moi thoi gian se don het vao chang
    # cuoi cung co doc ket qua ve CPU -> ket luan nghen SAI.
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

    def prefill_tbptt(ids, w):
        """T-w token dau no_grad, w token cuoi co grad (xem docstring dau
        file: GC bi loai ve nguyen tac nen day la duong duy nhat chan duoc
        activation cua 4B)."""
        past, cutp = None, max(0, ids.shape[1] - w)
        if cutp:
            with clock("4B-nograd"), torch.no_grad():
                for s in range(0, cutp, 1024):
                    o = model_s(input_ids=ids[:, s:min(s + 1024, cutp)],
                                past_key_values=past, use_cache=True,
                                logits_to_keep=1)
                    past = o.past_key_values
        with clock("4B-grad"):
            for s in range(cutp, ids.shape[1], 1024):
                o = model_s(input_ids=ids[:, s:s + 1024], past_key_values=past,
                            use_cache=True, logits_to_keep=1)
                past = o.past_key_values
        return past

    def student_past(cut):
        src = prefill_tbptt(cut, args.tbptt)
        with clock("template"):
            tpl = e5.build_template_from_meta(
                probe_t, meta_for_len(base_meta, T_BASE, cut.shape[1]))
        with clock("mapper"):
            st = e5.build_student_past(tpl, src, mapper)
        del tpl
        return st

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit(
        [{"params": mapper.params, "lr": args.lr},
         {"params": lora_params, "lr": args.lora_lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    results = {"args": vars(args), "val": [], "train_loss": []}

    def save_results():
        (out / "results.json").write_text(json.dumps(results, indent=1))

    def save_ckpt(tag):
        torch.save(mapper.state_dict(), out / f"mapper_{tag}.pt")
        model_s.save_pretrained(str(out / f"lora_{tag}"))
        hf_up(out / f"mapper_{tag}.pt", f"mapper_{tag}.pt")
        for f in sorted((out / f"lora_{tag}").glob("*")):
            if f.is_file():
                hf_up(f, f"lora_{tag}/{f.name}")

    _self_cache = {}
    # tra cuu self tu pseudo_gold.json: {id: 1} voi moi mau 9B TU LAM DUNG.
    # Mau khong co trong file = 9B lam SAI (gen_pseudo chi ghi ca dung).
    _self_lookup = {}
    if args.pseudo_gold and Path(args.pseudo_gold).exists():
        pgd = json.loads(Path(args.pseudo_gold).read_text())
        for it_ in data["val"]:
            i_ = it_.get("id")
            if i_ and it_["kind"] in NEW_KINDS:
                _self_lookup[i_] = 1 if i_ in pgd else 0
        print(f"tra cuu self tu pseudo-gold: {len(_self_lookup)} mau "
              f"(khong phai tinh lai)", flush=True)

    def _grade(it, txt):
        if it["kind"] in NEW_KINDS:
            return gd.score_item(it, txt)
        if it["kind"] == "bfcl":
            return int(it["fn"] in txt)
        if it["kind"] == "needle":
            return int(it["code"] in re.sub(r"\D", "", txt))
        return int(re.sub(r"\s+", " ", it["gold"])[:30]
                   in re.sub(r"\s+", " ", txt))

    STOPS = e5.stop_ids(tok_t, model_t)
    print(f"token dung sinh: {sorted(STOPS)} "
          f"({[tok_t.decode([i]) for i in sorted(STOPS)]})", flush=True)

    @torch.no_grad()
    def _greedy(past, warm, n_new):
        o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
        cur = o.past_key_values
        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
        gen = [int(inp)]
        for _ in range(n_new - 1):
            o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
            cur = o.past_key_values
            inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
            gen.append(int(inp))
            if int(inp) in STOPS:   # xem e5.stop_ids: checkpoint khai eos
                break               # bat nhat quan, dung sai = sinh tran
        del cur, o
        return tok_t.decode(gen)

    @torch.no_grad()
    def run_val(limit):
        """(user 2026-08-28) Do CA HAI dieu kien tren CUNG item:
          self   = 9B tu doc prompt  -> TRAN THAT cua bai do
          mapped = 9B doc cache 4B qua mapper

        Khong co so self thi "gsm8k 0/13" KHONG DIEN GIAI DUOC: khong phan
        biet duoc mapper sinh rac hay chinh 9B cung khong lam duoc bai do.
        Dung phep doi chung da cuu ket luan lan truoc (4B-self gsm8k 81,5%
        chung minh loi nam o khau DICH, khong phai model nguon yeu).

        Bo qua no_ctx: o bo de nay "context" CHINH LA cau hoi nen no_ctx vo
        nghia (khac han needle/bfcl cu, noi context la filler quanh dap an)."""
        from collections import defaultdict
        sc = defaultdict(lambda: {"self": [], "mapped": []})
        for i, it in enumerate(data["val"][:limit]):
            cut, warm, gold_ids, _ = enc(it)
            if gold_ids.shape[1] < 1:
                continue
            n_new = GEN_LEN.get(it["kind"], 24)
            st = student_past(cut)
            sc[it["kind"]]["mapped"].append(_grade(it, _greedy(st, warm, n_new)))
            del st
            torch.cuda.empty_cache()
            # self KHONG can tinh lai TI NAO nua: pseudo_gold.json (do bang
            # vLLM tren 3000/2600/498 mau) da ghi CHINH XAC tung mau nao 9B tu
            # lam dung. Tra cuu la ra — mien phi, va chuan hon han vi do bang
            # engine dung eos dung, tren quy mo lon. Bo duoc MOT NUA chi phi
            # val -> dồn het cho cot mapped.
            if it.get("id") in _self_lookup:
                sc[it["kind"]]["self"].append(_self_lookup[it["id"]])
            elif i not in _self_cache:      # ho khong co trong pseudo-gold
                slf = e5.prefill_chunked(model_t, cut)
                _self_cache[i] = _grade(it, _greedy(slf, warm, n_new))
                del slf
                torch.cuda.empty_cache()
                sc[it["kind"]]["self"].append(_self_cache[i])
            else:
                sc[it["kind"]]["self"].append(_self_cache[i])
        return {k: (f"{sum(v['mapped'])}/{len(v['mapped'])}"
                    f" [self {sum(v['self'])}/{len(v['self'])}]")
                for k, v in sc.items()}, \
               sum(sum(v["mapped"]) for v in sc.values())

    best, n_skip, n_flat, n_acc = -1, 0, 0, 0
    opt.zero_grad(set_to_none=True)
    t_start = time.time()
    for step in range(1, args.steps + 1):
        it = data["train"][(step - 1) % len(data["train"])]
        cev, skipped = None, False
        try:
            cut, warm, gold_ids, feed = enc(it)
            # BUG DA SUA (2026-08-28, dem duoc bang tokenizer): nguong cu la
            # `< 2` -> nem BO moi item co gold DUNG 1 token. Do that tren
            # 6623 item du lieu moi: musr 474/474 = 100% bi bo, bbh 634/2500
            # = 25,4%, tong 1108 = 16,7%. Dap an trac nghiem ("A"/"B") va dap
            # an ngan cua bbh ("valid"/"True"/so) DEU la 1 token. Gold 1 token
            # HOAN TOAN hop le: feed = warm (5 token), logits[:, WARM_P-1:]
            # cho dung 1 vi tri, CE cham dung token do.
            if gold_ids.shape[1] < 1:
                skipped = True
                raise _SkipItem
            st = student_past(cut)
            with clock("9B-forward"):
                o = model_t(input_ids=feed, past_key_values=st, use_cache=True)
            logp = torch.log_softmax(o.logits[:, WARM_P - 1:].float(), -1)
            nll = -logp.gather(2, gold_ids.unsqueeze(-1)).squeeze(-1)
            wts = torch.ones_like(nll)
            wts[:, 0] = FIRST_W          # token quyet dinh
            if args.w_entity > 1.0:
                # (user duyet 2026-08-29) Doc tay 20 dau ra gsm8k hong cho
                # thay thu bi pha la CON SO va TEN THUC THE: "Kate 29 tuoi"
                # thanh "Tully 29 tuoi", "gia hon nua tuoi" thanh "tre hon 20
                # nam", de khong co de ma mapper bia ra de. Trong so FIRST_W
                # hien chi danh vao token DAU — dung cho bfcl (token dau la
                # ten ham) nhung vo nghia cho gsm8k. Danh thang vao chu so va
                # ten rieng = nham dung trieu chung da quan sat.
                pieces = tok_t.convert_ids_to_tokens(gold_ids[0].tolist())
                for pi, pc in enumerate(pieces):
                    if not pc:
                        continue
                    txt = pc.lstrip("Ġ▁ ")
                    if txt and (txt[0].isdigit() or txt[0].isupper()):
                        wts[0, pi] = max(float(wts[0, pi]), args.w_entity)
            ce = (nll * wts).sum() / wts.sum()
            cev = ce.detach().item()
            # chia cho accum de tong gradient = TRUNG BINH, khong phai TONG
            # (khong chia thi lr hieu dung nhan len accum lan)
            with clock("backward"):
                (ce / args.accum).backward()
            n_acc += 1
            if n_acc >= args.accum:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_acc = 0
            del st, o, logp, nll, ce
        except _SkipItem:
            n_skip += 1
        except torch.cuda.OutOfMemoryError:
            # BUG DA SUA: truoc day dung `continue` -> nhay LUON qua khoi val
            # o cuoi vong. Buoc val roi trung mot item bi bo = MAT HAN moc val
            # do, im lang (do that: moc 1500 bien mat khoi log). Gio chi danh
            # dau skipped roi di tiep, val van chay.
            print(f"  buoc {step} OOM (kind={it['kind']}, "
                  f"len={len(it['prompt'])}) -> bo qua", flush=True)
            # OOM giua backward de lai gradient CONG DO DANG cho item nay. Voi
            # --accum no lam ban ca cua so gop, khong chi mot buoc -> vut ca
            # cua so. Mat mot cua so re hon la buoc mot huong gradient hong.
            opt.zero_grad(set_to_none=True)
            n_acc = 0
            skipped, n_skip = True, n_skip + 1
        gc.collect()
        torch.cuda.empty_cache()

        if step % 20 == 0 and not skipped:
            results["train_loss"].append([step, round(cev, 4)])
            tot = sum(T_ACC.values()) or 1.0
            share = " ".join(f"{k} {100*v/tot:.0f}%"
                             for k, v in sorted(T_ACC.items(),
                                                key=lambda x: -x[1]))
            print(f"buoc {step}/{args.steps} ce={cev:.4f} "
                  f"{(time.time()-t_start)/step:.2f}s/buoc "
                  f"peak={gib():.2f}GiB", flush=True)
            print(f"    thoi gian: {tot/20:.2f}s/buoc do duoc | {share}",
                  flush=True)
            T_ACC.clear()
        if args.sanity and step >= args.sanity:
            print(f"SANITY xong {step} buoc, peak={gib():.2f}GiB, "
                  f"{(time.time()-t_start)/step:.2f}s/buoc", flush=True)
            print("E9_SANITY_EXIT", flush=True)
            return
        if step % args.val_every == 0:
            vs, score = run_val(args.val_n)
            results["val"].append([step, score, vs])
            print(f"=== VAL buoc {step}: score={score} {vs}", flush=True)
            save_results()
            hf_up(out / "results.json", "results.json")
            save_ckpt("last")
            if score > best:
                best, n_flat = score, 0
                save_ckpt("best")
                print(f"    ky luc moi: {score}", flush=True)
            else:
                n_flat += 1
                print(f"    khong lap ky luc ({score} <= {best}), "
                      f"{n_flat}/{args.patience or '-'} moc phang", flush=True)
            if args.patience and n_flat >= args.patience:
                print(f"PATIENCE: {n_flat} moc lien tiep khong lap ky luc "
                      f"(tot nhat {best}) -> hoi tu, dung", flush=True)
                break
            # CE cua MOT item dao dong 0,02-2,46 tuy item (do that trong run
            # dau: 1.34/0.96/2.46/1.54/0.82/.../0.018). Lay ce cua dung buoc
            # val lam dieu kien dung = tung dong xu: ~8%/moc x 8 moc ~ 50%
            # kha nang dung som oan. Dung TRUNG BINH TRUOT 20 buoc gan nhat.
            recent = [c for _, c in results["train_loss"][-20:]]
            ce_ma = sum(recent) / max(len(recent), 1)
            print(f"    ce trung binh 20 buoc = {ce_ma:.4f}"
                  + (f" (ce buoc nay {cev:.4f})" if cev is not None else "")
                  + f" | da bo qua {n_skip} buoc", flush=True)
            if ce_ma < args.ce_floor:
                print(f"CE_FLOOR (trung binh {ce_ma:.4f} < {args.ce_floor}) "
                      "-> dung som", flush=True)
                break

    # val cuoi CHI chay khi buoc cuoi khong roi dung moc val — neu khong se
    # lap lai y het val vua chay xong. Moi val ~10 phut (gsm8k sinh 320 token
    # x 2 dieu kien), nen lan lap thua nay tốn that.
    if step % args.val_every != 0:
        vs, score = run_val(args.val_n)
        results["val"].append([step, score, vs])
        print(f"=== VAL cuoi: score={score} {vs}", flush=True)
        save_results()
        hf_up(out / "results.json", "results.json")
        save_ckpt("last")
        if score > best:
            save_ckpt("best")
    print("E9_JOINT_EXIT", flush=True)


if __name__ == "__main__":
    main()
