#!/usr/bin/env bash
# 1) cham dung/sai cua fla (truoc/sau khi cai) 2) cai that 3) do lai toc do
# eba_grpo.py --sanity de xac nhan loi ich, truoc khi dung cho vong compare.
#   bash run_fla_check.sh
set -u
cd "$(dirname "$0")"
mkdir -p /content/logs
for pkg in peft bitsandbytes; do
  python3 -c "import $pkg" 2>/dev/null || pip install -q "$pkg" 2>&1 | tail -1
done

echo "=== [1/4] check_fla TRUOC khi cai fla ==="
python3 -u check_fla.py --tag truoc --n 8 || exit 1

echo "=== [2/4] cai flash-linear-attention ==="
pip install -q "flash-linear-attention[cuda]" 2>&1 | tail -20
python3 -c "import fla; print('fla', getattr(fla, '__version__', '?'), 'OK')" || \
  echo "CANH BAO: import fla that bai sau khi cai -- xem log tren"

echo "=== [3/4] check_fla SAU khi cai fla ==="
python3 -u check_fla.py --tag sau --n 8 || exit 1
python3 -u check_fla.py --diff truoc sau

echo "=== [4/4] do lai toc do eba_grpo.py (sanity 20 buoc) ==="
NAME="${CK:-joint49cc}"
[ -f "/content/$NAME/mapper_best.pt" ] || python3 - "$NAME" <<'PYEOF' || true
import os, sys, shutil, pathlib
from huggingface_hub import snapshot_download
name = sys.argv[1]
try:
    p = snapshot_download("gunnybd01/qwen35-kv-mapper-4b-27b",
                          allow_patterns=[f"{name}/*"],
                          local_dir=f"/content/_hf_{name}",
                          token=os.environ.get("HF_TOKEN"))
    src = pathlib.Path(p) / name
    if any(src.glob("mapper_*.pt")):
        shutil.copytree(src, f"/content/{name}", dirs_exist_ok=True)
        print("KEO VE", name)
except Exception as e:
    print(f"khong lay duoc {name}:", type(e).__name__, str(e)[:80])
PYEOF
python3 -u eba_grpo.py \
  --init-mapper "/content/$NAME/mapper_best.pt" \
  --init-lora "/content/$NAME/lora_best" \
  --init-lora-t "/content/$NAME/lorat_best" \
  --k 6 --sanity 20 --out /content/_fla_speed_test \
  2>&1 | tee /content/logs/eba_grpo_fla_speed.log

python3 - <<'PYEOF' || true
import os, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
for f in ["/content/logs/fla_check_truoc.json", "/content/logs/fla_check_sau.json",
         "/content/logs/eba_grpo_fla_speed.log"]:
    p = pathlib.Path(f)
    if p.exists():
        try:
            api.upload_file(path_or_fileobj=f, repo_id="gunnybd01/qwen35-kv-mapper-4b-27b",
                            path_in_repo="evalbig/" + p.name)
            print("HF-UP", p.name)
        except Exception as e:
            print("HF-UP FAIL", p.name, type(e).__name__, str(e)[:80])
PYEOF
echo "RUN_FLA_CHECK_EXIT"
