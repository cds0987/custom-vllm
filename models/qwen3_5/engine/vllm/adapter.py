"""vLLM engine adapter for Qwen3.5 — serve configs measured on L4 + patch list.

Everything here was MEASURED, not guessed; provenance is STATUS.md.
run.sh reads these configs conceptually (kept in sync by tests/test_structure.py
checking this file exists); humans read them as the source of truth.
"""

ADAPTER = {
    "axis": "engine",
    "variant": "vllm",
    "supports": ["qwen3_5-dense-0.8b/2b/4b/9b/27b"],
    "tested_versions": ["0.26", "0.27.1"],
    "patches_dir": "patches",  # 19 idempotent scripts, applied by loading/setup_env.sh
}

# Tuned serve configs (L4 22.5GB — see ../hardware/l4.py for the numbers behind them)
SERVE_CONFIGS = {
    "9b": {
        "max_model_len": 65536,          # Q2c: 49152 is 5.2% WORSE at the 8-session operating point
        "max_num_batched_tokens": 1088,  # SLA floor: newcomers' prefill must not starve decoders
        "gpu_memory_utilization": 0.97,  # util sweep 2026-08-14: KV 404K->560K (+38.5%); 1.0 dies
        "extra": ["--enable-prefix-caching", "--mamba-cache-mode align",
                   "--kv-cache-dtype fp8_e4m3"],
    },
    "27b": {
        "max_model_len": 8192,           # KV budget: 12,288 tokens total at util 0.97
        "max_num_batched_tokens": 512,
        "max_num_seqs": 8,               # Mamba cache blocks cap at 19 on this VRAM budget
        "gpu_memory_utilization": 0.97,
        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8],
                                "max_cudagraph_capture_size": 8},  # eager costs -47% decode
        "extra": ["--enable-prefix-caching", "--mamba-cache-mode align",
                   "--kv-cache-dtype fp8_e4m3"],
    },
}
