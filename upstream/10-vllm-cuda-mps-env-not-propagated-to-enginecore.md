# [Observed behavior] CUDA_MPS_* environment variables are not propagated from the API server process to spawned EngineCore workers, preventing MPS SM-partitioning

**Target repo:** vllm-project/vllm
**Severity:** Low (not a crash; forecloses an optimization rather than breaking functionality — filed as an observed-behavior report, not a bug, since we have not fully traced whether this is fixable without a code change or is an inherent consequence of the multiprocessing spawn boundary)
**Affected versions:** vllm 0.26
**Local fix:** None implemented — this is a report of the limitation, not a patch.
**Duplicate check:** No existing issue found specifically about `CUDA_MPS_*` propagation to `EngineCore`. Adjacent-but-distinct: general vLLM multi-instance/MPS discussions exist in the ecosystem, but we found none addressing this specific propagation gap.

## Summary

We attempted to use NVIDIA MPS (Multi-Process Service) SM-partitioning (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and related `CUDA_MPS_*` variables) to give a latency-sensitive "chat" vLLM instance a guaranteed slice of SM resources while a throughput-oriented "long document" instance shares the same GPU. This did not work: the `CUDA_MPS_*` environment variables reach the top-level `APIServer` process (since that's where the shell that launched `vllm serve` set them), but vLLM's `EngineCore` — the process that actually allocates and holds GPU context — is spawned via Python multiprocessing (`spawn` start method) as a *separate* process, and does not inherit or re-receive these variables. MPS SM-pinning is a property of the CUDA context created in the process that talks to the MPS control daemon, so setting the variable in the parent process has no effect on the child that actually matters.

## Reproduction

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 vllm serve Qwen/Qwen3.5-2B ...
```

The `APIServer` process sees `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` in its own environment (confirmed via `/proc/<pid>/environ` inspection), but the spawned `EngineCore` process — which is where `torch.cuda` context creation and all GPU work actually happens — does not have it set, and MPS SM-partitioning has no observable effect on the instance's GPU resource share.

## What we did instead, and what it's worth

Since MPS pinning isn't currently usable this way, we benchmarked the practical alternative — running the two workloads as **two separate vLLM instances** on the same GPU (each capped at `--gpu-memory-utilization 0.40`, relying on the driver's default time-slicing rather than MPS) — against a single instance handling both workloads on one queue. Chat-instance tail latency while the document-instance is under sustained load:

| | 1 instance (mixed) | 2 instances (separate) |
|---|---|---|
| TTFT max | 1.985 s | **0.327 s** (6× lower) |
| ITL max | 1.257 s | **0.083 s** (15× lower) |

Weights for a 2B model are small enough (~1.8 GiB) that doubling them (one copy per instance) is a reasonable trade for a 6–15× tail-latency improvement, so we'd recommend this as the practical production pattern regardless of whether MPS propagation gets fixed. We're filing the MPS gap anyway because SM-partitioning would in principle let both instances share GPU compute more precisely than gpu-memory-utilization + time-slicing can, if it worked.

## What would be needed to actually fix it

We have not implemented this — it would require vLLM's `EngineCore` worker-bootstrap code (wherever it spawns/execs the engine-core subprocess for the `multiprocessing` executor) to explicitly forward `CUDA_MPS_*` variables (or a broader allowlist / all of `os.environ`) into the spawned process's environment, or to read them itself before creating the CUDA context. We haven't traced the exact spawn call site closely enough to propose a specific patch, which is why this is filed as an observed-behavior report rather than a fix.
