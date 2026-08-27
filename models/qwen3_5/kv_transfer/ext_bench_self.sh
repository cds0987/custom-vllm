#!/bin/bash
# ext_bench_self.sh -- baseline 27B THUAN (khong mapper) tren MuSR + AIME_2024
# + nvidia/compute-eval (build+test THAT). Khong can mapper -- chay TRUOC,
# doc lap voi chien dich train 4->27B @ctx8192 dang cho.
# User 2026-08-27: "thu thuan 27B ... de thay tong quat kha nang cua mapper"
# (sau nay se doi chieu voi cross qua ext_bench.py mode=cross).
set -u
cd /content/custom-vllm || exit 9
git pull -q
export HF_TOKEN=$(grep -oP '(?<=HF_TOKEN=).*' .env)
E=models/qwen3_5/kv_transfer/ext_bench.py
N_COMPUTE="${1:-60}"

pip install -q datasets 2>&1 | tail -2
python -u "$E" check-nvcc || true

if [ ! -f /content/ext_bench_items.json ]; then
  python -u "$E" gen --bench musr,aime,compute --n-compute "$N_COMPUTE" || exit 4
fi

hfup() {
  python - "$1" <<'EOF' || echo HF_UP_SKIP
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
for f in glob.glob('/content/logs/extbench_*.json') + ['/content/ext_bench_items.json']:
    try:
        a.upload_file(path_or_fileobj=f, path_in_repo=sys.argv[1] + '/' + os.path.basename(f),
                      repo_id='gunnybd01/qwen35-kv-mapper-4b-27b')
    except Exception as e:
        print('up-fail', os.path.basename(f), e)
print('HF-UP', sys.argv[1])
EOF
}

# musr (756) + aime (30) nhanh (gen ngan, khong build) -> 1 lan
if [ ! -f /content/logs/extbench_self_qa.json ]; then
  python -u "$E" self --bench musr,aime --tgt-model Qwen/Qwen3.5-27B --max-len 8192 \
    2>&1 | tee /content/logs/extbench_self_qa_run.log
  mv /content/logs/extbench_self.json /content/logs/extbench_self_qa.json 2>/dev/null
  hfup extbench
fi

# compute-eval: build+test THAT tren MOI item -> cham, chia wave 10/lan
NC=$(python -c "import json; print(sum(1 for it in json.load(open('/content/ext_bench_items.json')) if it['bench']=='compute'))")
echo "COMPUTE_TOTAL=$NC"
for a in $(seq 0 10 $((NC - 1))); do
  b=$((a + 10)); [ $b -gt $NC ] && b=$NC
  [ -f "/content/logs/extbench_self_${a}_${b}.json" ] && { echo "WAVE_${a}_SKIP"; continue; }
  echo "WAVE_$a"
  python -u "$E" self --bench compute --tgt-model Qwen/Qwen3.5-27B --max-len 8192 --slice "$a:$b" \
    2>&1 | tee -a /content/logs/extbench_self_compute_run.log
  hfup extbench
done

python -u "$E" agg
hfup extbench
echo EXTBENCH_SELF_EXIT
