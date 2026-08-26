#!/bin/bash
# C2-SUITE — chien dich do LON cho cross 4B->9B tren vLLM, 4 ho de
# (rag / mid-information / reasoning-math / swe) x 3 do dai (2K/4K/8K).
# User chot 2026-08-26: "test luon vai ngan prompts ... target 80-90 nhu
# normal decode". Ke thua nguyen dieu kien serving da chot bang do:
# bf16 KV, block-align 1056, L1 lon, lmcache CUNG torch, vLLM ghim 0.27.1.
#
# Chay:  bash c2suite.sh [N] [CTXS] [FAMILIES]
#   vd:  bash c2suite.sh 1200 2000,4000,8000 rag,mid,math,swe
# Resume: chay lai la bo qua wave da co file ket qua (idempotent).
set -u
N="${1:-1200}"; CTXS="${2:-2000,4000,8000}"; FAMS="${3:-rag,mid,math,swe}"
cd /content/custom-vllm || exit 9
git pull -q
export HF_TOKEN=$(grep -oP '(?<=HF_TOKEN=).*' .env)
export KV_DTYPE=auto
G=models/qwen3_5/kv_transfer/c2b_gates.py
L1GB="${L1GB:-32}"

bash run.sh setup 2>&1 | tail -3
source /tmp/vllm_env.sh 2>/dev/null || true
pip install -q 'lmcache>=0.5.2' 2>&1 | tail -1
pip install -q --force-reinstall --no-deps 'lmcache>=0.5.2' 2>&1 | tail -1
python -c "import torch,vllm,lmcache;print('STACK torch',torch.__version__,'vllm',vllm.__version__,'lmcache',lmcache.__version__)" || { echo STACK_FAIL; exit 6; }
R=$(python -c 'import lmcache,os;print(os.path.dirname(lmcache.__file__))')
grep -q qwen35-shared "$R/integration/vllm/lmcache_mp_connector.py" || sed -i 's/model_name=vllm_config.model_config.model,/model_name="qwen35-shared",/' "$R/integration/vllm/lmcache_mp_connector.py" "$R/integration/vllm/lmcache_mp_connector_0201.py"
grep -q qwen35-shared "$R/integration/vllm/lmcache_mp_connector.py" && echo SUITE_PATCHED || { echo PATCH_FAIL; exit 5; }

PROD='--kv-transfer-config={"kv_connector":"LMCacheMPConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}} --enforce-eager'
CONS='--kv-transfer-config={"kv_connector":"LMCacheMPConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5555}} --enforce-eager'

gpuwait() { for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 1500 ] && return 0; sleep 5; done; echo GPU_STUCK; exit 3; }
chk() { curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null && echo PORT_OK_$1 || { echo PORT_FAIL_$1; tail -20 /content/logs/serve.log; exit 2; }; }
lmc_restart() {
  for p in $(ss -tlnp 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in *lmcach[e]*) kill -9 $p;; esac
  done
  sleep 3
  nohup lmcache server --chunk-size 1056 --separate-object-groups \
    --l1-size-gb "$L1GB" --eviction-policy LRU --http-port 8081 \
    > /content/logs/lmc_server.log 2>&1 &
  sleep 20
  grep -q 'ZMQ cache server is running' /content/logs/lmc_server.log \
    && echo LMC_UP || { echo LMC_DEAD; exit 1; }
}
hfup() {  # quy tac 6d: day ket qua len HF NGAY moi wave
  python - "$1" <<'EOF' || echo HF_UP_SKIP
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
for f in glob.glob('/content/logs/c2b_suite*.json') + glob.glob('/content/c2b_prompts*.json'):
    try:
        a.upload_file(path_or_fileobj=f, path_in_repo=sys.argv[1] + '/' + os.path.basename(f),
                      repo_id='gunnybd01/qwen35-kv-mapper-4b-27b')
    except Exception as e:
        print('up-fail', os.path.basename(f), e); break
print('HF-UP', sys.argv[1])
EOF
}

# ---- de bai (chi sinh 1 lan; resume dung lai dung bo cu) ----
if [ ! -f /content/c2b_prompts_waves.json ]; then
  python "$G" gen-suite --n "$N" --ctxs "$CTXS" --families "$FAMS" || exit 4
fi
NW=$(python -c "import json;print(len(json.load(open('/content/c2b_prompts_waves.json'))))")
echo "SUITE_PLAN N=$N waves=$NW ctxs=$CTXS fams=$FAMS L1=${L1GB}GB"

# ---- self baseline: 9B champion, KHONG connector, 1 lan boot ----
if [ ! -f /content/logs/c2b_suitebase.json ]; then
  export EXTRA_FLAGS='--enforce-eager'
  gpuwait; bash run.sh serve 9b 2>&1 | tail -3; chk base; echo SUITE_BASE_UP
  python "$G" suitebase || echo BASE_PARTIAL
  bash run.sh stop; gpuwait; hfup suite
fi

# ---- cross: moi wave = 4B produce -> 9B consume ----
for w in $(seq 0 $((NW - 1))); do
  RANGE=$(python -c "import json;a,b=json.load(open('/content/c2b_prompts_waves.json'))[$w];print(f'{a}:{b} {a} {b}')")
  SL=$(echo "$RANGE" | cut -d' ' -f1); A=$(echo "$RANGE" | cut -d' ' -f2); B=$(echo "$RANGE" | cut -d' ' -f3)
  [ -f "/content/logs/c2b_suitecross_${A}_${B}.json" ] && { echo "WAVE_${A}_SKIP"; continue; }
  echo "WAVE_$A of $NW ($SL)"
  lmc_restart
  export EXTRA_FLAGS="$PROD" FRAME_4B_REPO=Qwen/Qwen3.5-4B MODELS_DIR=/content/models3
  bash run.sh serve 4b >/dev/null 2>&1; chk p$A
  python "$G" produce --slice "$SL" || echo PROD_PARTIAL_$A
  bash run.sh stop; gpuwait; unset FRAME_4B_REPO MODELS_DIR
  export EXTRA_FLAGS="$CONS"
  bash run.sh serve 9b >/dev/null 2>&1; chk c$A
  python "$G" suitecross --slice "$SL" || echo CROSS_PARTIAL_$A
  bash run.sh stop; gpuwait
  hfup suite
  python "$G" agg-suite || true
done

python "$G" agg-suite
hfup suite
echo C2SUITE_EXIT
