#!/bin/bash
# C2c — scope NGU NGHIA cross 4B->9B tren serving vLLM (N=60 @8K, 3 wave x 20).
# Ke thua nguyen van driver C2b-N (10 wave 2h 0 nga): dieu kien serving da chot
# bang do — bf16 KV (KV_DTYPE=auto), block-align 1056, L1 20GB, lmcache CUNG
# torch voi vLLM (source /tmp/vllm_env.sh), producer 4B bf16, consumer champion.
# Chay tren Colab qua Popen: bash models/qwen3_5/kv_transfer/c2c_sem.sh
set -u
cd /content/custom-vllm || exit 9
git pull -q
export HF_TOKEN=$(grep -oP '(?<=HF_TOKEN=).*' .env)
export KV_DTYPE=auto
source /tmp/vllm_env.sh 2>/dev/null

PROD='--kv-transfer-config={"kv_connector":"LMCacheMPConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}} --enforce-eager'
CONS='--kv-transfer-config={"kv_connector":"LMCacheMPConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}} --enforce-eager'

gpuwait() { for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1500 ] && return 0; sleep 5; done; echo GPU_STUCK; exit 3; }
chk() { curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null && echo PORT_OK_$1 || { echo PORT_FAIL_$1; tail -20 /content/logs/serve.log; exit 2; }; }
lmc_restart() {
  # kill theo CHU CONG (khong pkill pattern — bai hoc con song sot giu cong)
  for p in $(ss -tlnp 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in *lmcach[e]*) kill -9 $p;; esac
  done
  sleep 3
  nohup lmcache server --chunk-size 1056 --separate-object-groups \
    --l1-size-gb 20 --eviction-policy LRU --http-port 8081 \
    > /content/logs/lmc_server.log 2>&1 &
  sleep 20
  grep -q 'ZMQ cache server is running' /content/logs/lmc_server.log \
    && echo LMC_UP || { echo LMC_DEAD; exit 1; }
}

# gen: giu lai bo 240 needle cu (neu con), can datasets cho wikitext
cp /content/c2b_prompts.json /content/c2b_prompts_n240.json 2>/dev/null
python -c "import datasets" 2>/dev/null || pip -q install datasets 2>&1 | tail -1
python models/qwen3_5/kv_transfer/c2b_gates.py gen-sem --n 60 --ctx 8000 || exit 4

# baseline self: 9B champion, KHONG connector — 60 prompt 1 luot
export EXTRA_FLAGS='--enforce-eager'
gpuwait; bash run.sh serve 9b 2>&1 | tail -3; chk base; echo SEM_BASE_UP
python models/qwen3_5/kv_transfer/c2b_gates.py sembase || echo BASE_PARTIAL
bash run.sh stop; gpuwait

# 3 wave x 20 (kho L1 20GB ~30 vo 8K — 20/wave an toan)
for w in 0 1 2; do
  a=$((w*20)); b=$((a+20)); echo WAVE_$a
  lmc_restart
  export EXTRA_FLAGS="$PROD" FRAME_4B_REPO=Qwen/Qwen3.5-4B MODELS_DIR=/content/models3
  bash run.sh serve 4b >/dev/null 2>&1; chk p$a
  python models/qwen3_5/kv_transfer/c2b_gates.py produce --slice $a:$b || echo PROD_PARTIAL_$a
  bash run.sh stop; gpuwait; unset FRAME_4B_REPO MODELS_DIR
  export EXTRA_FLAGS="$CONS"
  bash run.sh serve 9b >/dev/null 2>&1; chk c$a
  python models/qwen3_5/kv_transfer/c2b_gates.py semcross --slice $a:$b || echo CROSS_PARTIAL_$a
  bash run.sh stop; gpuwait
done

python models/qwen3_5/kv_transfer/c2b_gates.py agg-sem

# quy tac 6d: ket qua len HF ngay trong phien
python - <<'EOF'
import glob
import os
from huggingface_hub import HfApi
a = HfApi()
files = (glob.glob('/content/logs/c2b_sembase*.json')
         + glob.glob('/content/logs/c2b_semcross*.json')
         + ['/content/c2b_prompts.json'])
for f in files:
    a.upload_file(path_or_fileobj=f, path_in_repo='c2c_sem/' + os.path.basename(f),
                  repo_id='gunnybd01/qwen35-kv-mapper-4b-27b')
print('HF-UP c2c_sem', len(files))
EOF
echo C2C_SEM_EXIT
