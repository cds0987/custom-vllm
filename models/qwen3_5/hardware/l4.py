"""NVIDIA L4 (23GB, 72W-capped, sm_89) — measured operating envelope for Qwen3.5.

Every number below was measured on Colab L4 (vLLM 0.27.1 unless noted);
provenance: STATUS.md. This file is the machine-readable MANIFEST of the
hardware axis; change nothing here without a fresh measurement.
"""

ADAPTER = {
    "axis": "hardware",
    "variant": "l4",
    "gpu": "NVIDIA L4 22.5GiB sm_89, power-capped 72W (real compute ~45% of datasheet)",
}

MEASURED = {
    "9b_champion": {
        "ppl_99_swebench": 4.7637,          # bf16 baseline: 5.13
        "decode_conc1_tok_s": 36.3,
        "prefill_tok_s": (2789, 2934),
        "serve_throughput_30k_prefix": 390,
        "agent_loop_operating_point": {"sessions": 12, "tasks_per_hr": "308 cold / 358 warm", "gpu_util": 0.97},
        "kv_at_65536": "560,380 tokens (fp8, util 0.97; 404K at old 0.85)",
        "speculative_ngram": "OFF: -36% tasks/hr at 8 sessions (compute-saturated); only +10% at pure conc1",
    },
    "27b_frame_w4a16": {                     # apolo13x frame, GDN pre-quantized
        "ppl_99_swebench": 4.1484,           # beats 9B champion by 12.3%
        "decode_conc1_tok_s": 15.8,          # bandwidth ceiling ~16 for 18.6GB weights
        "ttft_p50_conc1_s": 0.24,            # 95.3% prefix hit, 4K prefix
        "kv_tokens_total": 12288,
        "operating_point": "1-2 users, quality tier (vs 9B throughput tier)",
        "notes": "eager mode costs -47% decode (8.4 tok/s); util 0.97 required",
    },
    "cpu": "never the bottleneck: GPU flat 100% at conc32 while vLLM uses ~8% of one core",
}
