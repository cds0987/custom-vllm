#!/usr/bin/env bash
# Sinh lai pseudo-gold CoT cho 4 ho QUAN HE (musr, suite_rag/mid/swe) — user
# duyet 2026-09-01: dung chinh 9B lam giao vien, mo ngan sach 24->200 token
# de bat duoc CA BUOC SUY LUAN chu khong chi dap an ngan bi cat cut.
#
# BAT BUOC chay SAU khi suite_gen.score da vay (commit da co) — bo loc
# "giu dau ra CHAM DIEM DUNG" dung dung ham nay cho suite_*; sinh truoc khi
# vay se giu nham CoT co dap an garble-trung-may-man lam "dung".
#
#   bash run_pseudo_cot.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs

for pkg in vllm; do
  python3 -c "import $pkg" 2>/dev/null || { echo "CHUA CO vllm, chay setup_env truoc"; exit 1; }
done

echo "=== xac nhan suite_gen.score DA VA (khong con la ban cu) ==="
python3 -c "
import re
src = open('suite_gen.py', encoding='utf-8').read()
assert 'nxt.isdigit()' in src, 'suite_gen.score CHUA VA -- dung lai'
print('OK: suite_gen.score da va')
"

echo "=== sinh pseudo-gold CoT (musr,suite_rag,suite_mid,suite_swe), split=ca ==="
python3 -u gen_pseudo_vllm.py \
  --data /content/train_items.json \
  --out /content/pseudo_gold_cot.json \
  --model Qwen/Qwen3.5-9B --quant bnb \
  --kinds musr,suite_rag,suite_mid,suite_swe \
  --split ca \
  --hf-prefix joint49_cot || exit 1

echo "=== gop voi pseudo_gold.json cu (giu nguyen bbh/gsm8k, chi thay 4 ho quan he) ==="
python3 -c "
import json, pathlib
old = json.loads(pathlib.Path('/content/pseudo_gold.json').read_text())
new = json.loads(pathlib.Path('/content/pseudo_gold_cot.json').read_text())
merged = dict(old)
merged.update(new)   # CoT moi de sau -> thang the cho 4 ho quan he
pathlib.Path('/content/pseudo_gold_v2.json').write_text(
    json.dumps(merged, ensure_ascii=False))
print(f'gop: cu={len(old)} + moi={len(new)} -> tong={len(merged)} '
      f'(trung {len(old)+len(new)-len(merged)})')
"

echo "RUN_PSEUDO_COT_EXIT"
