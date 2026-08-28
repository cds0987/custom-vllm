"""E6 v3 — mapper 4B->27B train bang CE-GOLD tren mien that (user duyet 2026-08-23).

Thay doi cot loi so v2 (user chi ra KL-only chua du):
  L = CE(gold)  +  BETA*KL(teacher||student)  +  lam(t)*auxMSE  +  GAMMA*dense
  - CE tinh TREN TOKEN DAP AN THAT (response-only — dung bai
    train_on_responses_only cua Unsloth: context nam trong cache nen moi vi
    tri duoc cham diem deu la gold token; bo CONV_WARM vi tri dau).
  - Chon checkpoint bang VALIDATION CHUC NANG (hit/valid/needle), khong bang
    loss (luat error-placement). Test NIEM PHONG mo dung 1 lan cuoi.
  - Chap nhan narrow-domain (user chot): train dung mien se dung.

Data (khao sat schema that 2026-08-23):
  - BFCL exec_simple: 20 item dau = TEST NIEM PHONG (y het E6); phan con lai
    + BFCL_v3_simple.json -> train/val. Gold = "fn_name(".
  - ifstruct (LiquidAI/ifstruct-v1.0, split test): KHONG co gold — dung
    PSEUDO-GOLD = greedy output cua chinh 27B tu doc (muc tieu user: bao dam
    dau ra cua 27B). Val metric = validator chuong trinh (parse YAML/JSON,
    check code-block/top_level_key theo spec cua tung item).
  - ParseBench (config parse-bench, split table): input goc la PDF -> khong
    dung thang; dung expected_markdown lam CONTEXT: "bang trong cache ->
    trich dung hang thu k", gold cat regex tu chinh bang — suite
    structured-retrieval.
  - Needle tong hop (khuon e6.make_train_ids): giu truc truy xuat dai.

v3.3 (user duyet 2026-08-24 — toc do + ctx dai + chinh xac):
  TOC DO: (1) bo 2 deepcopy ~600MB/buoc (clone_cache_struct trong e5);
  (2) PHASE B1 tien tinh teacher MOT LAN/item (top-64 logp + dense caps
  4 layer/item, fp16 ra dia) -> vong train khong con teacher feed-forward
  + khong deepcopy tch_ext; (3) aux chi tinh khi lam>0.
  CTX DAI: (4) mapper.ckpt = checkpoint map_attn (autograd fp32 ~1GB@2K);
  (5) --gdn-bf16 (thu nghiem, can sanity); (6) needle curriculum train
  700/1200/1600/2000 + val bucket @2000 de thay vach da lui theo thoi gian.
  CHINH XAC: (7) BFCL them parallel+multiple; (8) trong so token-XUONG
  (xuong cu phap `([{":` x2) cho ifstruct/pbtable — bai hoc CONV_WARM:
  token quyet dinh bi bo doi la chet; (9) run_val DUMP text sinh ra vao
  results (mo no ifstruct/pbtable 0 diem — v3.2 khong luu gi de mo).

v3.4-long (user duyet 2026-08-25 — tang sequence length, triet ly Unsloth
tu chinh cho co may cua ta):
  (10) prefill THEO KHUC (e5.prefill_chunked) — transient khong phu thuoc T;
  (11) B1 luu them BO XUONG template (e5.cache_meta — shape/int, khong
  tensor) -> vong train DUNG LAI template bang zeros (build_template_from_
  meta) khi lam=0: KHONG con teacher prefill moi buoc — chi phi buoc train
  gan nhu khong phu thuoc do dai context; (12) needle buckets keo dai
  3000/4000(/8000/16000 sau ladder) qua --max-ctx; (13) --ladder do
  s/buoc + peak + GB-dia/item tung nac truoc khi chon tran; (14)
  --tpl-check: doi chieu logits template-that vs template-xuong (bat buoc
  truoc khi tin duong moi).

Modes:
  --dry-data       : local, khong GPU — dung + in data mix de user duyet
  --ladder L1,L2.. : sandbox rieng — do tung nac do dai
  --tpl-check N    : kiem tra duong template-xuong tren N item train
  (mac dinh)       : Colab — phase A (4B spill) -> B0 pseudo-gold ->
                     B1 teacher-precompute -> train+val -> test niem phong
"""

import argparse
import gc
import importlib.util
import json
import random
import re
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "e6_suite", Path(__file__).parent / "e6_suite.py")
e6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e6)
e5, e2 = e6.e5, e6.e2

TRAIN_MAX = 1024     # phong bi da kiem chung; 1536/2048 OOM (5.15 luu GDN
                     # state fp32 — cache 27B nang gap doi runtime cu)
NK_MAXLEN = 4096     # needle khong bao gio duoc cat (bug needle2k cu)
GOLD_MAX = 64        # fallback
# v3.2: gold/gen rieng tung loai — ifstruct/pbtable 0 diem vi gold 64 cat cut
GMAX = {"bfcl": 16, "needle": 12, "ifstruct": 96, "pbtable": 64}
# (128/96 OOM: feed 132 vi tri x fp32-states 5.15 trat phong bi L4)
GEN_LEN = {"bfcl": 24, "needle": 16, "ifstruct": 160, "pbtable": 120}
# --- 2026-08-28: mien MOI (gen_data.py). User: "co train mapper cho math/
# reasoning ko? neu chi training cho bfcl thi failed cho task khac la dung
# roi" -- kiem code cho thay data cu 0 mau math/reasoning VA moi gold deu
# ngan-trich 12-16 token. gsm8k gold = loi giai day du chinh la tin hieu
# "giu cache song qua sinh dai" con thieu. GEN_LEN bam theo ext_bench.N_NEW
# de val luc train do DUNG cai benchmark cuoi se do.
GMAX.update({"gsm8k": 256, "bbh": 24, "musr": 8,
             "suite_rag": 16, "suite_mid": 16, "suite_math": 16,
             "suite_swe": 16})
GEN_LEN.update({"gsm8k": 320, "bbh": 48, "musr": 24,
                "suite_rag": 24, "suite_mid": 24, "suite_math": 24,
                "suite_swe": 24})
NEW_KINDS = {"gsm8k", "bbh", "musr",
             "suite_rag", "suite_mid", "suite_math", "suite_swe"}
_GD = None


def _gen_data():
    """Nap gen_data.py mot lan (no keo theo ext_bench -> cham, chi nap khi
    that su co item mien moi)."""
    global _GD
    if _GD is None:
        spec2 = importlib.util.spec_from_file_location(
            "gen_data", Path(__file__).parent / "gen_data.py")
        _GD = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(_GD)
    return _GD


def merge_new_data(data, path):
    """Tron data mien moi vao data cu. GIU LAI data cu (bfcl/needle/ifstruct/
    pbtable) — khong de mapper QUEN nang luc da co 18/20 + 15/15; do chinh la
    thu duy nhat dang ban duoc luc nay."""
    from collections import Counter
    extra = json.loads(Path(path).read_text())
    before = len(data["train"]), len(data["val"])
    data["train"] += extra["train"]
    data["val"] += extra["val"]
    random.Random(SEED).shuffle(data["train"])
    print(f"tron data moi: train {before[0]}->{len(data['train'])}, "
          f"val {before[1]}->{len(data['val'])}")
    print("  train:", dict(Counter(x["kind"] for x in data["train"])))
    print("  val  :", dict(Counter(x["kind"] for x in data["val"])))
    return data
CONV_WARM = e5.CONV_WARM   # (v3.0 — giu cho tham chieu)
# v3.1 (user duyet 2026-08-24): CONV_WARM skip 4 vi tri dau cua GOLD = khong
# bao gio day token quyet dinh (fn name dau). Giao thuc moi: cache cat tai
# T-WARM_P, warm conv bang WARM_P token CUOI PROMPT (token that, khong phai
# dap an) -> CE cham TRON 100% gold, token dau trong so FIRST_W.
WARM_P = 5
FIRST_W = 3.0
BETA = 0.3           # trong so KL phu
CE_FLOOR = 0.2       # Unsloth: train loss <0,2 = overfit -> dung (user chot)
GAMMA = 0.05         # dense supervision
N_NEEDLE_TRAIN = 200
VAL_EVERY = 250   # v3.2: val 55 mau ~10 phut/lan, 2000 buoc = 8 moc
SEED = 7
K_TOP = 64        # v3.3: KL tren top-64 logits teacher (du 99%+ mass, luu dia)
N_CAP = 4         # v3.3: dense loss lay mau 4/16 layer, xoay vong theo item
SKEL = set('([{"\':,|`')   # token-xuong cu phay: trong so x2 (ifstruct/pbtable)
SKEL_W = 2.0


# ------------------------------ data ----------------------------------------

def bfcl_load(fname, n):
    # KHONG dung load_dataset: datasets ban moi (Colab py3.13) parse JSON
    # kieu khac -> "Trailing data". Tai file tho + tu parse (JSON hoac JSONL).
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                           fname, repo_type="dataset")
    txt = Path(path).read_text(encoding="utf-8")
    try:
        ds = json.loads(txt)
    except ValueError:
        ds = [json.loads(l) for l in txt.splitlines() if l.strip()]
    items = []
    for ex in ds:
        q = ex.get("question")
        if isinstance(q, list):
            try:
                q = q[0][0]["content"]
            except Exception:
                q = str(q)
        fns = ex.get("function") or []
        if isinstance(fns, dict):
            fns = [fns]
        fname_ = fns[0].get("name") if fns else None
        if not q or not fname_:
            continue
        prompt = ("You can call these functions:\n" + json.dumps(fns)[:3000]
                  + f"\nUser request: {q}\nRespond with one function call.\nCall: ")
        items.append({"kind": "bfcl", "prompt": prompt,
                      "gold": fname_ + "(", "fn": fname_})
        if len(items) >= n:
            break
    return items


def ifstruct_load(n):
    from datasets import load_dataset
    ds = load_dataset("LiquidAI/ifstruct-v1.0", split="test")
    items = []
    for ex in ds:
        p = ex.get("prompt")
        if not (isinstance(p, str) and p.strip()):
            continue
        items.append({"kind": "ifstruct", "prompt": p[:4000] + "\nOutput: ",
                      "gold": None,          # pseudo-gold: 27B tu sinh o phase B0
                      "spec": {"fmt": ex.get("output_format"),
                               "key": ex.get("top_level_key"),
                               "code_block": ex.get("require_code_block")}})
        if len(items) >= n:
            break
    return items


def pbtable_load(n):
    from datasets import load_dataset
    ds = load_dataset("llamaindex/ParseBench", "parse-bench", split="table")
    rng = random.Random(SEED)
    items = []
    for ex in ds:
        md = ex.get("expected_markdown") or ""
        rows = re.findall(r"<tr>.*?</tr>", md, flags=re.S)
        if len(rows) < 3 or len(md) < 400:
            continue
        k = rng.randint(1, min(len(rows) - 1, 6))
        gold = re.sub(r"\s+", " ", rows[k]).strip()[:400]
        if len(gold) < 20:
            continue
        prompt = (md[:5000]
                  + f"\n\nRepeat EXACTLY row number {k+1} (the {k+1}-th <tr>...</tr>) "
                  + "of the table above, as one line.\nRow: ")
        items.append({"kind": "pbtable", "prompt": prompt, "gold": gold})
        if len(items) >= n:
            break
    return items


def needle_items(tok, n, seed0, ctx_tok=700):
    # MOT stream duy nhat roi cat lat (khuon E6). token_stream skip `seed`
    # document truoc khi lay token — seed lon = tai-va-vut hang van doc (bug
    # treo 20 phut da dinh 2026-08-23). Seed giu nho, xao tron bang rng.
    stream = e2.token_stream(tok, n * ctx_tok + ctx_tok, seed=seed0 % 400)
    items = []
    for i in range(n):
        rng = random.Random(seed0 + i)
        name = rng.choice(e2.NAMES) + str(rng.randint(0, 99))
        code = "".join(rng.choice("0123456789") for _ in range(6))
        ids = stream[i * ctx_tok:(i + 1) * ctx_tok]
        half = ctx_tok // 2
        prompt = (tok.decode(ids[:half])
                  + f"\nIMPORTANT: The secret code for project {name} is {code}.\n"
                  + tok.decode(ids[half:]) + "\n" + e2.build_q(name) + " ")
        items.append({"kind": "needle", "prompt": prompt, "gold": code + ".",
                      "code": code, "ctx": ctx_tok})
    return items


def build_data(tok=None, max_ctx=4096):
    """v3.4 (user duyet 2026-08-25): needle curriculum keo dai theo
    --max-ctx (ladder quyet tran); test niem phong cu GIU NGUYEN de so
    doi chieu, mien dai co bo niem phong MOI. tok=None -> bo needle."""
    rng = random.Random(SEED)
    exec_simple = bfcl_load("BFCL_v3_exec_simple.json", 100)
    test = exec_simple[:20]                      # NIEM PHONG — y het E6
    train = exec_simple[20:]
    simple = bfcl_load("BFCL_v3_simple.json", 400)
    test_prompts = {it["prompt"] for it in test}
    simple = [it for it in simple if it["prompt"] not in test_prompts]
    rng.shuffle(simple)
    val = simple[:15]
    train += simple[15:]
    # v3.3: category kho hon — nhieu ham ung vien / goi nhieu ham. Gold van
    # "fn dau tien(" (grading `fn in txt` giu nguyen thang do voi simple).
    for extra in ("BFCL_v3_parallel.json", "BFCL_v3_multiple.json"):
        try:
            xs = bfcl_load(extra, 100)
        except Exception as ex:            # ten file doi giua cac ban dataset
            print(f"WARN bo qua {extra}: {ex}")
            continue
        xs = [it for it in xs if it["prompt"] not in test_prompts]
        val += xs[:5]
        train += xs[5:]
    ifs = ifstruct_load(150)
    val += ifs[:15]
    train += ifs[15:]
    pbt = pbtable_load(120)
    val += pbt[:10]
    train += pbt[10:]
    if tok is not None:
        # v3.4: curriculum keo dai — dong o ngan, thua dan o dai (dia la
        # rang buoc: cache 4B ~T-tuyen tinh). Bucket > max_ctx bi bo.
        for ctx, n, seed in [(700, 150, 30000), (950, 80, 40000),
                             (1200, 60, 50000), (1600, 50, 60000),
                             (2000, 40, 70000), (3000, 30, 80000),
                             (4000, 20, 90000), (8000, 10, 100000),
                             (16000, 6, 110000)]:
            if ctx <= max_ctx:
                train += needle_items(tok, n, seed, ctx_tok=ctx)
        val += needle_items(tok, 10, 31000)
        val += needle_items(tok, 5, 41000, ctx_tok=1500)
        val += needle_items(tok, 5, 42000, ctx_tok=2000)
        for ctx, seed in [(4000, 43000), (8000, 44000), (16000, 45000)]:
            if ctx <= max_ctx:
                val += needle_items(tok, 3, seed, ctx_tok=ctx)
        test += needle_items(tok, 10, 32000, ctx_tok=2000)   # y het E6/v3.3
        if max_ctx > 2000:   # niem phong MOI cho mien dai (seed chua dung)
            test += needle_items(tok, 5, 33000, ctx_tok=max_ctx)
    rng.shuffle(train)
    return {"train": train, "val": val, "test": test}


def ifstruct_valid(text, spec):
    """Validator chuong trinh cho ifstruct (khong can gold)."""
    ok = True
    body = text
    if spec.get("code_block"):
        ok &= "```" in text
        m = re.search(r"```[a-z]*\n(.*?)(```|$)", text, flags=re.S)
        body = m.group(1) if m else text
    fmt = (spec.get("fmt") or "").lower()
    try:
        if fmt == "json":
            json.loads(body)
        elif fmt == "yaml":
            import yaml
            yaml.safe_load(body)
    except Exception:
        ok = False
    key = spec.get("key")
    if key:
        ok &= key in text
    return int(ok)


# ------------------------------ main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tgt-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="/content/mapper_v34.pt")
    ap.add_argument("--cache-dir", default="/content/v34_src")
    ap.add_argument("--results", default="/content/logs/e6v34_results.json")
    ap.add_argument("--init-from", default="/content/mapper_v33.pt",
                    help="v3.4: warm-start tu mapper v3.3 (giu BFCL 18/20 "
                         "lam von) — bo qua neu file khong ton tai")
    ap.add_argument("--no-ckpt-mapper", action="store_true",
                    help="tat checkpoint map_attn (mac dinh BAT tu v3.3)")
    ap.add_argument("--gdn-bf16", action="store_true",
                    help="thu nghiem: ep GDN state bf16 sau prefill "
                         "(-~300MB/cache) — chi bat sau khi sanity 20 buoc")
    ap.add_argument("--tgt-cpu-offload", action="store_true",
                    help="(user 2026-08-27) nap tgt-model qua "
                         "e5.load_4bit_cpu_offload_io thay load_4bit — "
                         "tu tay day embed_tokens+lm_head xuong CPU, tiet "
                         "kiem ~4.85GiB, da do mo khoa max-ctx=8192 cho 27B "
                         "tren L4 (baseline OOM cung o do). 2 lan goi doc "
                         "lap lien tiep da kiem tra OK (an toan cho train).")
    ap.add_argument("--sanity", type=int, default=0,
                    help="chay N buoc train roi dung + in VRAM (luot do 1)")
    ap.add_argument("--max-ctx", type=int, default=4096,
                    help="v3.4: bucket needle dai nhat dua vao train/val")
    ap.add_argument("--ladder", default="",
                    help="v3.4: '4096,8192,16384' — do tung nac roi dung")
    ap.add_argument("--tpl-check", type=int, default=0,
                    help="v3.4: doi chieu logits template that vs xuong")
    ap.add_argument("--no-tpl", action="store_true",
                    help="v3.4: tat duong template-xuong (prefill that)")
    ap.add_argument("--v35-eval", action="store_true",
                    help="v3.5: do benh sinh-dai ifstruct/pbtable — 4 dieu "
                         "kien (self/map x greedy/rep-penalty), khong train")
    ap.add_argument("--rep-pen", type=float, default=1.3)
    ap.add_argument("--hf-repo", default="gunnybd01/qwen35-kv-mapper-4b-27b",
                    help="tu upload checkpoint/results moi moc val (quy tac "
                         "6d — hoc phi 2 lan). Rong = tat. Token doc tu env "
                         "HF_TOKEN hoac .env o root repo (KHONG commit .env)")
    ap.add_argument("--hf-prefix", default="",
                    help="thu muc con trong hf-repo (mac dinh: suy tu ten "
                         "--out, vd mapper_v49.pt -> 'v49'). TRUOC day GHIM "
                         "CUNG 'v34/' cho moi target — hoc phi 2026-08-26: "
                         "chien dich 4->9 se de len mapper 4->27B neu khong "
                         "tach thu muc.")
    ap.add_argument("--data-file", default="",
                    help="2026-08-28: tron them data DA DANG do gen_data.py "
                         "sinh (gsm8k/bbh/musr tach khoi tap test niem phong "
                         "+ 4 ho suite_gen). Rong = giu nguyen data cu.")
    ap.add_argument("--new-maxlen", type=int, default=2048,
                    help="tran token cho item mien moi (musr dai ~1000-1500 "
                         "token; TRAIN_MAX=1024 cu se cat cut narrative)")
    ap.add_argument("--dry-data", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--train-check", action="store_true",
                    help="falsification: nap mapper .last (thuoc long), "
                         "greedy tren 30 mau TRAIN — phai ra dap an neu "
                         "memorization la that")
    args = ap.parse_args()

    # sanity dung SANDBOX rieng: data bi cat/xep lai theo do dai -> chi so
    # item khong khop voi cache cua run that (train{i}.pt keyed by index)
    if args.sanity and args.cache_dir == "/content/v34_src":
        args.cache_dir = "/content/v34_sanity"
        args.out = "/content/mapper_sanity.pt"
        print(f"SANITY -> cache-dir {args.cache_dir}")

    # v3.4: needle khong bao gio duoc cat — tran token hoa theo max-ctx
    global NK_MAXLEN
    NK_MAXLEN = max(4096, args.max_ctx + 128)

    # token flow (user chot 2026-08-25): .env o root repo, dung truc tiep moi
    # moi truong; .env nam trong .gitignore — TUYET DOI khong commit (HF tu
    # revoke token lo trong repo public = mat duong upload).
    import os as _os
    if not _os.environ.get("HF_TOKEN"):
        for _p in (Path(__file__).resolve().parents[3] / ".env",
                   Path("/content/custom-vllm/.env")):
            try:
                for _l in _p.read_text().splitlines():
                    if _l.strip().startswith("HF_TOKEN="):
                        _os.environ["HF_TOKEN"] = _l.split("=", 1)[1].strip()
                        break
            except OSError:
                continue
            if _os.environ.get("HF_TOKEN"):
                break
    have_token = bool(_os.environ.get("HF_TOKEN"))
    hf_prefix = args.hf_prefix or Path(args.out).stem.replace("mapper_", "")
    if args.hf_repo and not have_token and not args.dry_data:
        print("CANH BAO 6d: khong tim thay HF_TOKEN (env/.env) — se KHONG "
              "auto-upload duoc; checkpoint chi nam tren runtime!")

    def hf_up(local, dest):
        """Best-effort upload — that bai khong duoc lam do train."""
        if not args.hf_repo or not have_token:
            return
        try:
            from huggingface_hub import HfApi
            # 401 hoc phi 2026-08-26 (c2c_sem): HfApi() khong tu doc HF_TOKEN
            # tren moi phien ban/runtime — luon truyen token= tuong minh.
            HfApi(token=_os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(local), path_in_repo=dest,
                repo_id=args.hf_repo)
            print(f"HF-UP {dest}")
        except Exception as ex:
            print(f"HF-UP FAIL {dest}: {type(ex).__name__}")

    if args.dry_data:
        data = build_data(tok=None, max_ctx=args.max_ctx)
        if args.data_file:
            data = merge_new_data(data, args.data_file)
        for split, items in data.items():
            from collections import Counter
            cnt = Counter(it["kind"] for it in items)
            print(f"{split}: {len(items)} items  {dict(cnt)}")
        for kind in ("bfcl", "ifstruct", "pbtable"):
            it = next(x for x in data["train"] if x["kind"] == kind)
            print(f"\n===== SAMPLE {kind} =====")
            print("PROMPT:", it["prompt"][:400].replace("\n", " | "))
            print("GOLD:", (it["gold"] or "(pseudo-gold tu 27B)")[:200])
        print(f"\n(v3.4: needle buckets 700..{args.max_ctx} them vao khi chay"
              " that — so luong theo build_data; test giu needle@2K cu"
              f" + 5 niem phong moi @{args.max_ctx})")
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig

    # transformers 5.15: update_recurrent_state lam .copy_() IN-PLACE len
    # tensor state ("static address for cudagraphs") — pha vo autograd khi
    # initial_state la tensor mapper mang grad (fla backward: "modified by an
    # inplace operation"). Rebind khi co grad; giu copy_ cho moi ca khac.
    try:
        from transformers.cache_utils import LinearAttentionLayer
        _orig_urs = LinearAttentionLayer.update_recurrent_state

        def _urs(self, recurrent_states, state_idx=0, **kw):
            cur = (self.recurrent_states.get(state_idx)
                   if isinstance(self.recurrent_states, dict) else None)
            if cur is not None and (cur.requires_grad
                                    or recurrent_states.requires_grad):
                self.recurrent_states[state_idx] = recurrent_states
                return recurrent_states
            return _orig_urs(self, recurrent_states, state_idx, **kw)

        LinearAttentionLayer.update_recurrent_state = _urs
        print("patched update_recurrent_state (rebind khi co grad)")
    except ImportError:
        pass

    # ---------------- LADDER (v3.4): do tung nac do dai roi dung -------------
    if args.ladder:
        Ls = [int(x) for x in args.ladder.split(",")]
        NK_MAXLEN = max(Ls) + 128
        ldir = Path(args.cache_dir + "_ladder")
        ldir.mkdir(parents=True, exist_ok=True)
        N_IT = 6
        if not (ldir / "LA_DONE").exists():
            tok_s, model_s = e5.load_4bit(args.src_model)
            items = {}
            for L in Ls:
                its = needle_items(tok_s, N_IT, 200000 + L, ctx_tok=L)
                items[str(L)] = its
                for j, it in enumerate(its):
                    pth = ldir / f"lad{L}_{j}.pt"
                    if pth.exists():
                        continue
                    enc = tok_s(it["prompt"], return_tensors="pt",
                                truncation=True, max_length=NK_MAXLEN).to("cuda")
                    past = e5.prefill_chunked(model_s,
                                              enc["input_ids"][:, :-WARM_P])
                    e5.spill_cache(past, pth)
                    del past
                    torch.cuda.empty_cache()
                print(f"LA {L} xong")
            (ldir / "items.json").write_text(json.dumps(items))
            del model_s
            gc.collect(); torch.cuda.empty_cache()
            (ldir / "LA_DONE").touch()
        items = json.loads((ldir / "items.json").read_text())
        theta_s = e5.e1.get_rope_theta(
            AutoConfig.from_pretrained(args.src_model).get_text_config())
        load_fn = e5.load_4bit_cpu_offload_io if args.tgt_cpu_offload else e5.load_4bit
        tok_t, model_t = load_fn(args.tgt_model)
        theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
        with torch.no_grad():
            probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                            use_cache=True, logits_to_keep=1).past_key_values
        src0 = e5.load_cache(ldir / f"lad{Ls[0]}_0.pt")
        a_s, g_s = e5.split_layers(src0)
        Hs = next(iter(g_s.values())).recurrent_states.shape[1]
        attn_dim = (next(iter(a_s.values())).keys.shape[1]
                    * next(iter(a_s.values())).keys.shape[3])
        del src0
        torch.cuda.empty_cache()
        a_t, g_t = e5.split_layers(probe)
        Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
        mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim,
                           theta_s, theta_t)
        mapper.ckpt = True
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(mapper.params, lr=args.lr)
        for L in Ls:
            try:
                sizes = [(ldir / f"lad{L}_{j}.pt").stat().st_size
                         for j in range(N_IT)]
                t0 = time.time()
                tks = []
                for j, it in enumerate(items[str(L)]):
                    enc = tok_t(it["prompt"], return_tensors="pt",
                                truncation=True, max_length=NK_MAXLEN).to("cuda")
                    ids = enc["input_ids"]
                    cut, warm2 = ids[:, :-WARM_P], ids[:, -WARM_P:]
                    gold_ids = tok_t(it["gold"], add_special_tokens=False,
                                     return_tensors="pt")["input_ids"][
                                     :, :GMAX["needle"]].to("cuda")
                    feed = torch.cat([warm2, gold_ids[:, :-1]], 1)
                    with torch.no_grad():
                        past = e5.prefill_chunked(model_t, cut)
                        tpl_meta = e5.cache_meta(past)
                        logp = torch.log_softmax(
                            model_t(input_ids=feed, past_key_values=past,
                                    use_cache=True).logits[:, WARM_P - 1:]
                            .float(), -1)
                        tl, ti2 = logp.topk(K_TOP, -1)
                    del past, logp
                    torch.cuda.empty_cache()
                    tks.append((ti2.long(), tl.float(), tpl_meta,
                                gold_ids, feed))
                t_b1 = (time.time() - t0) / N_IT
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0, n = time.time(), 0
                for _rep in range(2):
                    for j, (ti2, tl, tpl_meta, gold_ids, feed) in enumerate(tks):
                        with torch.no_grad():
                            tch = e5.build_template_from_meta(probe, tpl_meta)
                        src = e5.load_cache(ldir / f"lad{L}_{j}.pt")
                        sp = e5.build_student_past(tch, src, mapper)
                        out = model_t(input_ids=feed, past_key_values=sp,
                                      use_cache=True)
                        logp = torch.log_softmax(
                            out.logits[:, WARM_P - 1:].float(), -1)
                        nll = -logp.gather(2, gold_ids.unsqueeze(-1)).squeeze(-1)
                        s_top = logp.gather(2, ti2)
                        p_t = tl.exp()
                        rest_t = (1 - p_t.sum(-1)).clamp_min(1e-6)
                        rest_s = (1 - s_top.exp().sum(-1)).clamp_min(1e-6)
                        kl = ((p_t * (tl - s_top)).sum(-1)
                              + rest_t * (rest_t.log() - rest_s.log())).mean()
                        loss = nll.mean() + BETA * kl
                        opt.zero_grad(); loss.backward(); opt.step()
                        del tch, src, sp, out, logp, loss
                        torch.cuda.empty_cache()
                        n += 1
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"LADDER {L}: spill {sum(sizes)/N_IT/2**20:.0f} MB/item"
                      f" | B1 {t_b1:.1f} s/item"
                      f" | {(time.time()-t0)/n:.2f} s/buoc (template-path)"
                      f" | peak {peak:.2f} GiB")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as ex:
                # Triton OOM cua fla kernel nem RuntimeError thuong —
                # phai bat rong de nac sau van duoc do (hoc phi recon 1)
                print(f"LADDER {L}: {type(ex).__name__} — TRAN o nac nay"
                      f" ({str(ex)[:90]})")
                gc.collect(); torch.cuda.empty_cache()
        print("E6V34_LADDER_DONE")
        return

    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    results = {"val_curve": [], "config": {k: v for k, v in vars(args).items()}}

    def save_results():
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results, "w") as fh:
            json.dump(results, fh, indent=1)

    # ---------------- PHASE A: 4B mot minh — spill cache moi item ------------
    marker = cdir / "PHASE_A_DONE"
    tok = None
    if not marker.exists():
        tok_s, model_s = e5.load_4bit(args.src_model)
        tok = tok_s
        data = build_data(tok_s, args.max_ctx)
        if args.data_file:
            data = merge_new_data(data, args.data_file)
        if args.sanity:
            # do ca XAU NHAT: chi giu cac item train DAI nhat (needle 2000
            # dung dau) + val rut gon — du de do s/buoc va peak VRAM
            data["train"].sort(key=lambda x: -len(x["prompt"]))
            data["train"] = data["train"][:max(2 * args.sanity, 30)]
            data["val"] = data["val"][:5]
        with open(cdir / "data.json", "w") as fh:
            json.dump(data, fh)
        with torch.no_grad():
            for split in ("train", "val", "test"):
                for i, it in enumerate(data[split]):
                    pth = cdir / f"{split}{i}.pt"
                    if pth.exists():
                        continue
                    ml = (NK_MAXLEN if it["kind"] == "needle" else
                          args.new_maxlen if it["kind"] in NEW_KINDS
                          else TRAIN_MAX)   # PHAI khop _maxlen() ben duoi:
                    # cache 4B spill o day va prompt 27B doc lai phai CUNG
                    # do dai, lech mot token la cache lech vi tri hoan toan
                    enc = tok_s(it["prompt"], return_tensors="pt", truncation=True,
                                max_length=ml).to("cuda")
                    # v3.1: cache cat tai T-WARM_P (GDN khong tua nguoc duoc)
                    # v3.4: prefill theo khuc — item 16K khong lam no transient
                    past = e5.prefill_chunked(model_s,
                                              enc["input_ids"][:, :-WARM_P])
                    e5.spill_cache(past, pth)
                    del past
                    torch.cuda.empty_cache()
                    if i % 50 == 0:
                        print(f"A {split} {i}")
        del model_s
        gc.collect(); torch.cuda.empty_cache()
        marker.touch()
        print("PHASE_A_DONE")

    # ---------------- PHASE B: 27B mot minh ---------------------------------
    theta_s = e5.e1.get_rope_theta(
        AutoConfig.from_pretrained(args.src_model).get_text_config())
    # user 2026-08-27 "training-test 4-27 ... voi ctx 8192 training": CPU-
    # offload thu cong (load_4bit_cpu_offload_io) da do THAT mo khoa 8192
    # cho 27B tren L4 (baseline OOM cung o day) -- xem STATUS.md muc MAPPER
    # 4B->27B CPU offload thu cong: ket qua cuoi.
    load_fn = e5.load_4bit_cpu_offload_io if args.tgt_cpu_offload else e5.load_4bit
    tok_t, model_t = load_fn(args.tgt_model)
    if tok is None:
        tok = tok_t
    theta_t = e5.e1.get_rope_theta(model_t.config.get_text_config())
    data = json.loads((cdir / "data.json").read_text())

    def _maxlen(it):
        if it["kind"] == "needle":
            return NK_MAXLEN
        if it["kind"] in NEW_KINDS:
            return args.new_maxlen
        return TRAIN_MAX

    def enc_item(it):
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=_maxlen(it)).to("cuda")
        return enc["input_ids"][:, :-1], enc["input_ids"][:, -1:]

    def enc_cut(it):
        """v3.1: (cache_ids cat tai T-WARM_P, warm = WARM_P token cuoi prompt)."""
        enc = tok(it["prompt"], return_tensors="pt", truncation=True,
                  max_length=_maxlen(it)).to("cuda")
        ids = enc["input_ids"]
        return ids[:, :-WARM_P], ids[:, -WARM_P:]

    # --- B0: pseudo-gold cho ifstruct (27B tu doc, greedy GOLD_MAX) ---
    pg_path = cdir / "pseudo_gold.json"
    if not pg_path.exists():
        pg = {}
        with torch.no_grad():
            for split in ("train", "val"):
                for i, it in enumerate(data[split]):
                    if it["kind"] != "ifstruct":
                        continue
                    pre, last = enc_item(it)
                    past = model_t(input_ids=pre, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    cur, gen, inp = past, [], last
                    for _ in range(GMAX["ifstruct"]):
                        o = model_t(input_ids=inp, past_key_values=cur,
                                    use_cache=True)
                        cur = o.past_key_values
                        inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                        gen.append(int(inp))
                    pg[f"{split}{i}"] = tok.decode(gen)
                    del past, cur
                    torch.cuda.empty_cache()
        pg_path.write_text(json.dumps(pg))
        print(f"B0: {len(pg)} pseudo-gold")
    pg = json.loads(pg_path.read_text())
    for split in ("train", "val"):
        for i, it in enumerate(data[split]):
            if it["kind"] == "ifstruct":
                it["gold"] = pg.get(f"{split}{i}", "")

    # --- mapper + dense hooks (nguyen khuon e6) ---
    src0 = e5.load_cache(cdir / "train0.pt")
    a_s, g_s = e5.split_layers(src0)
    Hs = next(iter(g_s.values())).recurrent_states.shape[1]
    attn_dim = (next(iter(a_s.values())).keys.shape[1]
                * next(iter(a_s.values())).keys.shape[3])
    del src0
    with torch.no_grad():
        probe = model_t(input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
                        use_cache=True, logits_to_keep=1).past_key_values
    a_t, g_t = e5.split_layers(probe)
    Ht = e5._get(next(iter(g_t.values())).recurrent_states).shape[1]
    mapper = e5.Mapper(len(a_t), len(g_t), Hs, Ht, attn_dim, theta_s, theta_t)
    mapper.ckpt = not args.no_ckpt_mapper
    if mapper.ckpt:
        print("mapper.ckpt BAT (map_attn recompute trong backward)")
    if args.skip_train:
        mapper.load(args.out)
    elif Path(args.out + ".last").exists():
        mapper.load(args.out + ".last")
        print("RESUME tu checkpoint .last")
    elif args.init_from and Path(args.init_from).exists():
        mapper.load(args.init_from)   # v3.4: warm-start tu v3.3
        print(f"WARM-START tu {args.init_from}")

    captured = []
    cap_on = {"v": False}   # hook chi ghi khi bat — val/teacher-ctx-prefill
    hooks = []              # tung nhet 16x160MB/luot vao captured -> OOM @VAL
    def _hook(mod, inp, out):
        if cap_on["v"]:
            captured.append(out[0] if isinstance(out, tuple) else out)
    for name, mod in model_t.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls and "Linear" not in cls and hasattr(mod, "o_proj"):
            hooks.append(mod.register_forward_hook(_hook))

    def _gold_feed(it, i_train=None):
        """(cut, warm, gold_ids, feed) — DUY NHAT mot dinh nghia cho ca B1
        lan train: lech protocol giua hai noi = teacher cache sai lech ngam."""
        cut, warm = enc_cut(it)
        gm = GMAX.get(it["kind"], GOLD_MAX)
        if cut.shape[1] > 850:      # phong thu chu dong: ctx dai x gold dai
            gm = min(gm, 32)
        gold_ids = tok(it["gold"], add_special_tokens=False,
                       return_tensors="pt")["input_ids"][:, :gm].to("cuda")
        feed = torch.cat([warm, gold_ids[:, :-1]], 1)
        return cut, warm, gold_ids, feed

    # --- B1 (v3.3): tien tinh teacher MOT LAN/item -> vong train khong con
    # teacher feed-forward + deepcopy tch_ext. Luu: top-K logp + dense caps
    # N_CAP layer xoay vong (fp16, ~4MB/item). Idempotent tung file — song
    # sot restart nhu moi phase khac; item lam B1 chet (OOM cung) -> stub
    # .skip de khong livelock.
    n_train = len(data["train"])
    b1a = cdir / "b1_attempt.txt"
    if b1a.exists():
        dead = int(b1a.read_text())
        (cdir / f"tk{dead}.skip").touch()
        print(f"B1 OOM-skip item {dead}")
        b1a.unlink()
    b1_marker = cdir / "PHASE_B1_DONE"
    if not b1_marker.exists() and not args.skip_train \
            and not args.train_check and not args.v35_eval:
        nh = len(hooks)
        done = 0
        for i, it in enumerate(data["train"]):
            tkp = cdir / f"tk{i}.pt"
            if tkp.exists() or (cdir / f"tk{i}.skip").exists() \
                    or not it.get("gold"):
                continue
            b1a.write_text(str(i))
            try:
                with torch.no_grad():
                    cut, warm, gold_ids, feed = _gold_feed(it)
                    if gold_ids.shape[1] < 2:
                        b1a.unlink()
                        continue
                    past = e5.prefill_chunked(model_t, cut)   # v3.4: theo khuc
                    tpl_meta = e5.cache_meta(past)   # v3.4: bo xuong template
                    captured.clear()
                    cap_on["v"] = True
                    logits = model_t(input_ids=feed, past_key_values=past,
                                     use_cache=True).logits[:, WARM_P - 1:]
                    cap_on["v"] = False
                    logp = torch.log_softmax(logits.float(), -1)
                    tl, ti = logp.topk(K_TOP, dim=-1)
                    cidx = sorted({(i + o * nh // N_CAP) % nh
                                   for o in range(N_CAP)})
                    caps = [captured[j].detach().to(torch.float16).cpu()
                            for j in cidx]
                    torch.save({"ti": ti.to(torch.int32).cpu(),
                                "tl": tl.to(torch.float16).cpu(),
                                "cidx": cidx, "caps": caps, "tpl": tpl_meta,
                                "flen": feed.shape[1]}, tkp)
                    del past, logits, logp
            except torch.cuda.OutOfMemoryError:
                print(f"B1 OOM item {i} -> skip")
                (cdir / f"tk{i}.skip").touch()
            captured.clear()
            gc.collect(); torch.cuda.empty_cache()
            b1a.unlink(missing_ok=True)
            done += 1
            if done % 25 == 0:
                print(f"B1 {done} items (toi {i}/{n_train})")
        b1_marker.touch()
        print("PHASE_B1_DONE")

    # --- tpl-check (v3.4): duong template-xuong co cho logits Y HET duong
    # teacher-prefill that khong? PHAI pass truoc khi train tin no. ---
    if args.tpl_check:
        checked = 0
        with torch.no_grad():
            for i, it in enumerate(data["train"]):
                tkp = cdir / f"tk{i}.pt"
                if not tkp.exists() or not it.get("gold"):
                    continue
                tk = torch.load(tkp, map_location="cpu")
                if "tpl" not in tk:
                    continue
                cut, warm, gold_ids, feed = _gold_feed(it)
                if tk["flen"] != feed.shape[1]:
                    continue
                src = e5.load_cache(cdir / f"train{i}.pt")
                real = e5.prefill_chunked(model_t, cut)
                o1 = model_t(input_ids=feed,
                             past_key_values=e5.build_student_past(
                                 real, src, mapper),
                             use_cache=True).logits.float()
                del real
                torch.cuda.empty_cache()
                syn = e5.build_template_from_meta(probe, tk["tpl"])
                o2 = model_t(input_ids=feed,
                             past_key_values=e5.build_student_past(
                                 syn, src, mapper),
                             use_cache=True).logits.float()
                agree = float((o1.argmax(-1) == o2.argmax(-1)).float().mean())
                print(f"tpl-check item {i} ({it['kind']}, ctx {cut.shape[1]}):"
                      f" max|dlogit| {float((o1 - o2).abs().max()):.4f}"
                      f" | argmax-agree {agree:.3f}")
                del o1, o2, syn, src
                torch.cuda.empty_cache()
                checked += 1
                if checked >= args.tpl_check:
                    break
        print("E6V34_TPLCHECK_DONE")
        return

    # --- v3.5 (user chot 2026-08-25): mo benh SINH-DAI bang thuoc decode-time
    # Cau hoi 1: TRAN teacher — 27B-self co qua noi validator khong? (nghi van
    # pseudo-gold <think>). Cau hoi 2: repetition penalty co pha vong lap?
    if args.v35_eval:
        if not Path(args.out).exists():
            from huggingface_hub import hf_hub_download
            import shutil
            shutil.copy(hf_hub_download(args.hf_repo,
                                        f"v34/{Path(args.out).name}"), args.out)
            print("mapper tai tu HF v34/")
        mapper.load(args.out)
        sc, dumps = {}, []
        with torch.no_grad():
            for i, it in enumerate(data["val"]):
                if it["kind"] not in ("ifstruct", "pbtable"):
                    continue
                cut, warm = enc_cut(it)
                gl = GEN_LEN[it["kind"]]
                for cond in ("self", "self-rp", "map", "map-rp"):
                    rp = args.rep_pen if cond.endswith("rp") else 1.0
                    if cond.startswith("self"):
                        past = e5.prefill_chunked(model_t, cut)
                    else:
                        src = e5.load_cache(cdir / f"val{i}.pt")
                        tpl = e5.prefill_chunked(model_t, cut)
                        past = e5.build_student_past(tpl, src, mapper)
                        del src, tpl
                    o = model_t(input_ids=warm, past_key_values=past,
                                use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen_ids = [int(inp)]
                    for _ in range(gl - 1):
                        o = model_t(input_ids=inp, past_key_values=cur,
                                    use_cache=True)
                        cur = o.past_key_values
                        logits = o.logits[:, -1, :].float()
                        if rp != 1.0:
                            idx = torch.tensor(sorted(set(gen_ids)),
                                               device="cuda")
                            sel = logits[0, idx]
                            logits[0, idx] = torch.where(sel > 0, sel / rp,
                                                         sel * rp)
                        inp = logits.argmax(-1, keepdim=True)
                        gen_ids.append(int(inp))
                    txt = tok.decode(gen_ids)
                    if it["kind"] == "ifstruct":
                        hit = ifstruct_valid(txt, it["spec"])
                    else:
                        gh = re.sub(r"\s+", " ", it["gold"])[:60]
                        hit = int(gh[:30] in re.sub(r"\s+", " ", txt))
                    sc.setdefault((it["kind"], cond), []).append(hit)
                    dumps.append({"i": i, "kind": it["kind"], "cond": cond,
                                  "hit": hit, "txt": txt[:200]})
                    del past, cur
                    torch.cuda.empty_cache()
        for k, v in sorted(sc.items()):
            print(f"V35 {k[0]:9s} {k[1]:8s} {sum(v)}/{len(v)}")
        results["v35"] = {f"{k[0]}|{k[1]}": f"{sum(v)}/{len(v)}"
                          for k, v in sc.items()}
        results["v35_dumps"] = dumps
        save_results()
        hf_up(args.results, "v35/e6v35_decode.json")
        print("E6V35_DONE")
        return

    def student_forward(it, sid, feed):
        src = e5.load_cache(cdir / f"{sid}.pt")
        pre, _ = enc_item(it)
        with torch.no_grad():
            tpl = model_t(input_ids=pre, use_cache=True,
                          logits_to_keep=1).past_key_values
        past = e5.build_student_past(tpl, src, mapper)
        del src
        return model_t(input_ids=feed, past_key_values=past, use_cache=True), tpl

    def run_val(tag, split="val", limit=None):
        """Validation CHUC NANG qua duong mapper (split='train' = kiem
        thuoc-long: greedy tren chinh mau da train phai ra dap an)."""
        import statistics
        sc = {"bfcl": [], "needle": [], "ifstruct": [], "pbtable": []}
        dumps = []   # v3.3: luu text sinh ra — khong co gi de mo la mu
        with torch.no_grad():
            for i, it in enumerate(data[split][:limit]):
                sid = f"{split}{i}"
                cut, warm = enc_cut(it)
                src = e5.load_cache(cdir / f"{sid}.pt")
                tpl = e5.prefill_chunked(model_t, cut)
                past = e5.build_student_past(tpl, src, mapper)
                del src, tpl
                o = model_t(input_ids=warm, past_key_values=past, use_cache=True)
                cur = o.past_key_values
                inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                gen = [int(inp)]
                for _ in range(GEN_LEN.get(it["kind"], GOLD_MAX) - 1):
                    o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                txt = tok.decode(gen)
                if it["kind"] in NEW_KINDS:
                    # uy quyen cho CHINH grader cua benchmark cuoi (ext_bench.
                    # score_text) — val luc train khong the troi khoi bao cao
                    hit = _gen_data().score_item(it, txt)
                    sc.setdefault(it["kind"], []).append(hit)
                elif it["kind"] == "bfcl":
                    hit = int(it["fn"] in txt)
                    sc["bfcl"].append(hit)
                elif it["kind"] == "needle":
                    hit = int(it["code"] in re.sub(r"\D", "", txt))
                    sc["needle"].append(hit)
                elif it["kind"] == "ifstruct":
                    hit = ifstruct_valid(txt, it["spec"])
                    sc["ifstruct"].append(hit)
                else:
                    gold_head = re.sub(r"\s+", " ", it["gold"])[:60]
                    hit = int(gold_head[:30] in re.sub(r"\s+", " ", txt))
                    sc["pbtable"].append(hit)
                dumps.append({"i": i, "kind": it["kind"], "hit": hit,
                              "ctx": it.get("ctx"), "txt": txt[:240]})
                del past, cur
                torch.cuda.empty_cache()
        out = {k: f"{sum(v)}/{len(v)}" for k, v in sc.items() if v}
        score = sum(sum(v) for v in sc.values())
        results.setdefault("val_dumps", {})[tag] = dumps
        print(f"VAL[{tag}]: {out} -> score {score}")
        return score, out

    if args.train_check:
        mapper.load(args.out + ".last")
        print("TRAIN-CHECK: mapper .last (CE 0.008)")
        score, detail = run_val("train-check", split="train", limit=30)
        results["train_check"] = detail
        save_results()
        print("E6V3_TRAINCHECK_DONE")
        return

    if not args.skip_train:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(mapper.params, lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        n_train = len(data["train"])
        best, stale = -1, 0
        ce_hist = []
        # tu hoc mau-OOM qua cac lan restart: attempt.txt ghi step dang thu;
        # neu khoi dong ma file con do = lan truoc CHET tai step ay -> skip
        # item do vinh vien (chong livelock resume dam lai dung mau to).
        att_f, skip_f = cdir / "attempt.txt", cdir / "skip.json"
        skip_ids = set(json.loads(skip_f.read_text())) if skip_f.exists() else set()
        if att_f.exists():
            crashed = int(att_f.read_text())
            skip_ids.add(crashed % n_train)
            skip_f.write_text(json.dumps(sorted(skip_ids)))
            print(f"OOM-skip: item {crashed % n_train} (tong {len(skip_ids)})")
            att_f.unlink()
        # step BEN qua restart: OOM moi ~200 buoc < VAL_EVERY -> khong co step
        # toan cuc thi val/cosine/hoan-thanh khong bao gio dat
        gstep_f = cdir / "gstep.txt"
        start_step = int(gstep_f.read_text()) if gstep_f.exists() else 0
        if start_step:
            print(f"RESUME step toan cuc {start_step}")
            for _ in range(start_step):
                sched.step()
        t0 = time.time()
        # moc val kieu "nguong": crash giua moc -> resume van bat kip
        next_val = (start_step // VAL_EVERY) * VAL_EVERY + VAL_EVERY - 1
        for step in range(start_step, args.steps):
            gc.collect(); torch.cuda.empty_cache()
            # VAL o DAU vong lap: khong bi item-skip (continue) nuot moc nua
            if step >= next_val:
                next_val += VAL_EVERY
                score, detail = run_val(f"step{step}")
                results["val_curve"].append({"step": step, "score": score,
                                             **detail})
                save_results()
                # 6d: moi moc val la mot lan cuu ho — best + .last + results
                hf_up(args.results, f"{hf_prefix}/{Path(args.results).name}")
                if Path(args.out + ".last").exists():
                    hf_up(args.out + ".last", f"{hf_prefix}/{Path(args.out).name}.last")
                if score > best:
                    best, stale = score, 0
                    torch.save(mapper.state_dict(), args.out)
                    print(f"  best-by-val {best} -> saved")
                    hf_up(args.out, f"{hf_prefix}/{Path(args.out).name}")
                else:
                    stale += 1
                    if stale >= 3:
                        print("VAL dung im 3 moc — DUNG SOM (ky luat E8)")
                        break
            if step % 50 == 49:   # day an toan: checkpoint bat ke val
                torch.save(mapper.state_dict(), args.out + ".last")
                gstep_f.write_text(str(step + 1))
            idx = step % n_train
            it = data["train"][idx]
            sid = f"train{idx}"
            tkp = cdir / f"tk{idx}.pt"
            if not it.get("gold") or idx in skip_ids or not tkp.exists():
                continue
            att_f.write_text(str(step))   # neu chet o buoc nay -> skip lan sau
            cut, warm, gold_ids, feed = _gold_feed(it)
            if gold_ids.shape[1] < 2:
                continue
            tk = torch.load(tkp, map_location="cpu")
            if tk["flen"] != feed.shape[1]:   # protocol lech giua B1 va train
                print(f"WARN tk{idx} flen {tk['flen']} != {feed.shape[1]} -> skip")
                continue
            ti = tk["ti"].to("cuda").long()
            tl = tk["tl"].to("cuda").float()
            lam = max(0.0, 1.0 - step / (0.2 * args.steps))
            # v3.4: khi lam=0 khong can teacher that nua — dung template-XUONG
            # (zeros dung shape/vi tri, moi tensor deu bi thay/zero) -> bo han
            # teacher prefill: chi phi buoc train ~doc lap voi do dai context.
            # Duong nay PHAI qua --tpl-check truoc khi tin.
            # item DAI di duong template tu buoc 0: prefill-that + aux +
            # student la combo nang nhat (2 OOM dau v3.4 deu o day) — hy
            # sinh aux cho rieng item dai, CE/KL van day du
            use_tpl = (not args.no_tpl and "tpl" in tk
                       and (lam <= 0 or cut.shape[1] > 2500))
            with torch.no_grad():
                if use_tpl:
                    tch_past = e5.build_template_from_meta(probe, tk["tpl"])
                else:
                    tch_past = e5.prefill_chunked(model_t, cut)
                    if args.gdn_bf16:
                        e5.force_state_dtype(tch_past, torch.bfloat16)
            src = e5.load_cache(cdir / f"{sid}.pt")
            student_past = e5.build_student_past(tch_past, src, mapper)
            # v3.3: aux chi tinh khi con trong so (va chi co nghia voi teacher
            # that — template xuong toan zeros)
            aux = (e5.aux_mse(student_past, tch_past)
                   if (lam > 0 and not use_tpl)
                   else torch.zeros((), device="cuda"))
            # giai phong cache teacher TRUOC forward student (GDN fp32 5.15
            # ~604MB/ban 27B — template da clone-struct, khong con tham chieu)
            del tch_past
            torch.cuda.empty_cache()
            captured.clear()
            cap_on["v"] = True
            out = model_t(input_ids=feed, past_key_values=student_past,
                          use_cache=True)
            cap_on["v"] = False
            stu_caps = [captured[j] for j in tk["cidx"]]
            logp = torch.log_softmax(out.logits[:, WARM_P - 1:].float(), -1)
            nll = -logp.gather(2, gold_ids.unsqueeze(-1)).squeeze(-1)
            wts = torch.ones_like(nll)
            wts[:, 0] = FIRST_W          # token quyet dinh — trong so x3
            if it["kind"] in ("ifstruct", "pbtable"):
                # v3.3: token-XUONG cu phap x2 — bai hoc CONV_WARM ap vao
                # output co cau truc (xuong sai la validator/match rot ngay)
                pieces = tok.convert_ids_to_tokens(gold_ids[0].tolist())
                for pi, pc in enumerate(pieces):
                    if pc and any(ch in SKEL for ch in pc):
                        wts[0, pi] = max(float(wts[0, pi]), SKEL_W)
            ce = (nll * wts).sum() / wts.sum()
            # v3.3: KL tren top-K cua teacher + duoi gop 1 gio (thay full-vocab)
            s_top = logp.gather(2, ti)                     # (1, P, K)
            p_t = tl.exp()
            rest_t = (1 - p_t.sum(-1)).clamp_min(1e-6)
            rest_s = (1 - s_top.exp().sum(-1)).clamp_min(1e-6)
            kl = ((p_t * (tl - s_top)).sum(-1)
                  + rest_t * (rest_t.log() - rest_s.log())).mean()
            tch_caps = [c.to("cuda").float() for c in tk["caps"]]
            dense = sum((s.float() - t).pow(2).mean()
                        / (t.pow(2).mean() + 1e-6)
                        for s, t in zip(stu_caps, tch_caps)) / max(len(tch_caps), 1)
            loss = ce + BETA * kl + lam * aux + GAMMA * dense
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            cev, klv = float(ce), float(kl)
            del (student_past, out, logp, src, ce, kl, ti, tl, tk,
                 dense, loss, stu_caps, tch_caps)
            captured.clear()
            torch.cuda.empty_cache()
            att_f.unlink(missing_ok=True)   # buoc nay song sot
            ce_hist.append(cev)
            n_done = len(ce_hist)
            if step % 25 == 0:
                print(f"step {step}/{args.steps} CE {cev:.3f} KL {klv:.3f} "
                      f"lam {lam:.2f} ({time.time()-t0:.0f}s)")
            if args.sanity and n_done >= args.sanity:
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"SANITY: {n_done} buoc, {(time.time()-t0)/n_done:.2f}"
                      f" s/buoc (gom ca overhead khoi dong), peak {peak:.2f} GiB"
                      f" (ckpt={mapper.ckpt} gdn_bf16={args.gdn_bf16})")
                print("E6V33_SANITY_DONE")
                return
            if step > 150 and len(ce_hist) >= 50 \
                    and sum(ce_hist[-50:]) / 50 < CE_FLOOR:
                print(f"CE trung binh 50 buoc < {CE_FLOOR} — nguy co overfit "
                      "(quy tac Unsloth, user chot) -> chay VAL roi DUNG")
                score, detail = run_val(f"cefloor-step{step}")
                results["val_curve"].append({"step": step, "score": score,
                                             "stop": "ce_floor", **detail})
                save_results()
                if score > best:
                    best = score
                    torch.save(mapper.state_dict(), args.out)
                    hf_up(args.out, f"{hf_prefix}/{Path(args.out).name}")
                break
        print("TRAIN_DONE")
        mapper.load(args.out)

    # ---------------- TEST NIEM PHONG (mo dung 1 lan) ------------------------
    for h in hooks:
        h.remove()
    test_res = {"bfcl": {"self": 0, "mapped": 0, "no_ctx": 0, "n": 0},
                "needle2k": {"self": 0, "mapped": 0, "no_ctx": 0, "n": 0}}
    with torch.no_grad():
        for i, it in enumerate(data["test"]):
            key = "bfcl" if it["kind"] == "bfcl" else "needle2k"
            test_res[key]["n"] += 1
            pre, last = enc_item(it)
            for cond in ("self", "mapped", "no_ctx"):
                if cond == "self":
                    past = e5.prefill_chunked(model_t, pre)
                    cur, gen, inp = past, [], last
                elif cond == "no_ctx":
                    past = model_t(input_ids=last, use_cache=True,
                                   logits_to_keep=1).past_key_values
                    cur, gen, inp = past, [], last
                else:   # v3.1: warm conv bang WARM_P token cuoi prompt
                    cut, warm = enc_cut(it)
                    src = e5.load_cache(cdir / f"test{i}.pt")
                    tpl = e5.prefill_chunked(model_t, cut)
                    past = e5.build_student_past(tpl, src, mapper)
                    del src, tpl
                    o = model_t(input_ids=warm, past_key_values=past,
                                use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen = [int(inp)]
                for _ in range(24 - len(gen)):
                    o = model_t(input_ids=inp, past_key_values=cur, use_cache=True)
                    cur = o.past_key_values
                    inp = o.logits[:, -1, :].argmax(-1, keepdim=True)
                    gen.append(int(inp))
                txt = tok.decode(gen)
                hit = (int(it["fn"] in txt) if it["kind"] == "bfcl"
                       else int(it["code"] in re.sub(r"\D", "", txt)))
                test_res[key][cond] += hit
                del past, cur
                torch.cuda.empty_cache()
            print(f"TEST {i+1}/{len(data['test'])}: {test_res}")
    results["test"] = test_res
    save_results()
    print("===== E6V3 KET QUA NIEM PHONG =====")
    print(json.dumps(test_res, indent=1))
    # 6d: cuu ho cuoi chien dich — moi thu quy len HF ngay trong phien
    hf_up(args.out, f"{hf_prefix}/{Path(args.out).name}")
    if Path(args.out + ".last").exists():
        hf_up(args.out + ".last", f"{hf_prefix}/{Path(args.out).name}.last")
    hf_up(args.results, f"{hf_prefix}/{Path(args.results).name}")
    for extra in ("pseudo_gold.json", "data.json"):
        if (cdir / extra).exists():
            hf_up(cdir / extra, f"{hf_prefix}/{extra}")
    print("E6V3_DONE")


if __name__ == "__main__":
    main()
