---
name: benchmark
description: Cách lấy dataset và chạy benchmark cho model - bench decode/prefill, chất lượng (ppl), agent-loop nhiều user, workload BFCL/public-test/SWE-bench và cách chấm điểm. Nạp khi cần đo hiệu năng hoặc chất lượng của một checkpoint.
---

# Cách lấy và chạy benchmark

Mọi bench chạy trên Colab (repo đã clone ở `/content/custom_vllm`). Kết quả PHẢI ghi
vào `STATUS.md` + commit — số chỉ nằm trên runtime là sẽ mất.

## 1. Sanity + tốc độ cơ bản (checkpoint mới bắt buộc chạy trước)

```bash
# load + decode ngắn (bảng chuẩn: bench_load ~35.35s load / 646.0 tok tổng)
python bench/bench_load.py --model /content/champion
# prefill sạch cache (đo prefill thật, không dính prefix cache)
python bench/prefill_bench.py --model /content/champion
```
Số chuẩn để so (vLLM 0.27.1): ppl 4.7637, decode ~390 tok/s server / ~520 offline
(prefix 30K), prefill 2789–2934 tok/s.

## 2. Chất lượng — perplexity

```bash
python bench/eval_quality_swebench.py --model /content/champion        # ppl nhanh
# eval_quality_swebench.py cần datasets → dựng env với CUSTOM_VLLM_TOOLS=1
```

## 3. Serving bench (server thật, qua HTTP)

Serve bằng cell SERVE (config chuẩn trong CLAUDE.md), rồi:

```bash
python bench/bench_serving.py --host 127.0.0.1 --port 8000 ...
python bench/bench_skills.py --synthetic-prefix-tokens 30000 --seed 0   # prefix chia sẻ tổng hợp
```

## 4. Agent-loop nhiều user (workload chính của dự án)

```bash
python bench/bench_agent_loop.py \
    --sessions 8 --turns 6 --synthetic-prefix-tokens 30000 \
    --max-model-len 65536
```
- Cờ stress: `--tool-latency-tail`, `--tool-result-spike`, `--context-overflow-policy`,
  `--toolcall-invalid-rate`, `--burst-sync`, `--abandon-rate`, `--mixed-chat`.
- Bẫy đã dính: KHÔNG có `--synthetic-prefix-tokens` thì prefix chỉ ~1-1.5K token —
  hit rate thấp bất thường là dấu hiệu đo sai kịch bản.
- Mô hình capacity: scheduler đặt chỗ worst-case → concurrency = KV_tokens/max_model_len.
  Điểm vận hành đã đo: 8 session (328 task/giờ); 16 session là lỗ.

## 5. Workload thật: BFCL + public-test + SWE-bench

```bash
# BFCL đã tải sẵn ở datasets/bfcl/ (gitignored) — tải lại nếu runtime mới
python bench/workload/prepare_agent_workload.py --mix "bfcl:50,public-test:30,swebench:20" \
    --out /content/workload.jsonl
python bench/bench_agent_loop.py --workload /content/workload.jsonl ...
python bench/workload/score_bfcl.py --results <ket-qua>   # chấm AST-lite cho tool call
```

## Kỷ luật đo

- Đo từng biến một; mỗi mức tải đo RIÊNG (đo chồng pha từng cho 97% KV ảo).
- So sánh phải cùng runtime + cùng phiên bản vLLM (0.26 → 0.27.1 từng làm lệch chuẩn).
- Suy luận nội suy không thay được số đo (đã sai với chunk trend và fp8⇒int8).
