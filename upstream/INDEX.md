# Upstream draft index

Twelve draft reports (11 bug reports + 1 PR proposal) produced from the
Qwen3.5-GGUF-on-vLLM investigation recorded in `STATUS.md` and the anchor-based
source patches in `scripts/patch_*.py`. All local-only — nothing in this
directory has been posted to GitHub. Posting is a separate, user-approved step.

Duplicate/overlap status checked live against GitHub on 2026-08-08 (see each
draft's "Duplicate check" section for the specific issues/PRs and the exact
delta). Nothing we found was already fixed upstream since our investigation —
every issue checked is still open, and the one relevant PR (vllm#38140) is
still unmerged.

| # | File | Target repo | Title | Severity | Overlaps with |
|---|---|---|---|---|---|
| 1 | `01-vllm-gguf-plugin-trailing-dot-drops-bare-params.md` | vllm-gguf-plugin | Trailing-dot bug silently drops bare GGUF parameters (A_log) | Critical | None found |
| 2 | `02-vllm-gguf-plugin-qwen35-weight-transforms-not-inverted.md` | vllm-gguf-plugin | Qwen3.5 GGUF weight transforms (A_log, norms, conv1d, V-head tiling) never inverted on load | Critical | vllm-gguf-plugin#80 (tracking issue, no fix); no direct duplicate |
| 3 | `03-vllm-gguf-plugin-tuple-shard-id-weight-type.md` | vllm-gguf-plugin | Tuple shard-ids collapse per-shard GGUF weight types (Q5_K read as Q4_K) | High | None found |
| 4 | `04-vllm-gguf-plugin-qwen35-support-batch.md` | vllm-gguf-plugin | Qwen3.5 GGUF support: 6 correctness/compatibility gaps (naming, vision depth, multimodality detection, conv1d shape, `_bias` stripping, M-RoPE) | High | vllm#38122 / PR#38140 (covers 2 of the 6 items, but against the pre-plugin in-tree loader — delta documented); vllm#36456 (adjacent, different bug in the same user journey, no overlap); vllm-gguf-plugin#80 (tracking issue) |
| 5 | `05-vllm-gguf-plugin-bf16-unquantized-branch-broken.md` | vllm-gguf-plugin | BF16/unquantized GGUF branch loads ~0.03 GiB then crashes in torch.compile | High | None found. No fix included — filed as bug report only (root cause not isolated) |
| 6 | `06-vllm-gguf-plugin-hybrid-kernel-dispatch-pr-proposal.md` | vllm-gguf-plugin | PR proposal: dispatch GGUF matmuls by shape (fused Triton decode / dequant+cuBLAS prefill) | N/A (enhancement) | vllm#36802 / PR#43047 (different problem — Triton shmem OOM, not speed; no overlap, noted for context) |
| 7 | `07-vllm-qwen35-causallm-not-registered.md` | vllm | Qwen3_5ForCausalLM/Qwen3_5MoeForCausalLM defined but never registered; silently resolves to the multimodal class | Critical | vllm#38122/#38140/#36456 (upstream-of, in the same load sequence, no direct overlap) |
| 8 | `08-vllm-qwen35-text-only-missing-hybrid.md` | vllm | Text-only Qwen3.5 missing IsHybrid + mamba-state classmethods | High | None found (unreachable until #7's fix) |
| 9 | `09-vllm-qwen35-embed-tokens-missing-quant-config.md` | vllm | Qwen3_5Model.embed_tokens built without quant_config | Medium | None found |
| 10 | `10-vllm-cuda-mps-env-not-propagated-to-enginecore.md` | vllm | CUDA_MPS_* env vars not propagated to spawned EngineCore workers | Low (observed behavior, no fix) | None found |
| 11 | `11-transformers-qwen35-config-missing-vocab-size.md` | transformers | Qwen3_5Config missing top-level vocab_size (strict dataclass rejects @property fix) | Medium | None found |
| 12 | `12-gguf-py-missing-dt-bias-tensor-template.md` | ggml-org/llama.cpp (gguf-py) | Missing tensor template for qwen3.5's linear_attn.dt_bias | Medium | None found |

## Repos

- **vllm-project/vllm-gguf-plugin** — items 1–6 (the bulk). GGUF loading moved
  out-of-tree from vllm core into this plugin as of vllm 0.26 (see commit
  `1534218`), so this is the primary target for most of the correctness work.
- **vllm-project/vllm** — items 7–10 (core model registry/architecture and one
  low-severity infra observation).
- **huggingface/transformers** — item 11 (`Qwen3_5Config`).
- **ggml-org/llama.cpp** (source of the `gguf` PyPI package) — item 12.

## Duplicate-check detail

Checked live via `gh api`/`gh issue view` against the exact issue/PR numbers
supplied, plus a full open+closed issue listing for vllm-project/vllm-gguf-plugin:

- **vllm#38122** ("[Bug]: Qwen 3.5 fails to load from GGUF", open) and
  **PR#38140** ("[Bugfix] Fix Qwen 3.5 GGUF loading...", open, unmerged) —
  cover exactly two of the bugs in draft 4 (model_type naming, vision-config
  `depth` fallback), but patch `vllm/model_executor/model_loader/gguf_loader.py`,
  the in-tree loader that GGUF support has since moved out of into
  vllm-gguf-plugin. The plugin has its own independent copy of this logic with
  the identical two bugs, which #38122/#38140 do not touch — draft 4 is scoped
  to the plugin and states this delta explicitly, and covers 4 additional
  items (`is_multimodal` detection, conv1d shape, `_bias` suffix stripping,
  M-RoPE) that #38122/#38140 never mention.
- **vllm#30023** ("[Feature]: Support qwen3next with GGUF?", open) — same
  `model_type` underscore-vs-none naming class of bug (`qwen3next` vs
  `qwen3next`... actually `qwen3_next` vs `qwen3next` per that issue), but for
  a different architecture and a different code path (core vllm's own
  `gguf_loader.py` naming table, not the plugin). Confirms the same naming
  pattern recurs across architectures; no direct file overlap with any draft
  here since it's neither the plugin nor covers Qwen3.5.
- **vllm#36456** ("[Bug]: Local GGUF path fails with 'architecture qwen35 is
  not supported yet' even when --hf-config-path is provided", open) — a
  distinct bug (`maybe_override_with_speculators` reads the raw `.gguf` file
  path through transformers' own GGUF config loader instead of the supplied
  `--hf-config-path`, and transformers' loader has no `qwen35` entry at all —
  consistent with what we document in draft 2). Upstream of everything in
  drafts 4/7/8/9 in the "load a local Qwen3.5 GGUF" sequence, but a different
  bug in a different function; noted, not merged into any draft, since fixing
  it doesn't touch our patched files. The issue's documented workaround
  (`repo:quant` HF-hub syntax instead of a local path) is exactly the syntax
  our own `setup_env.sh`/`serve_test.sh` already use (see commit `1534218`),
  which is presumably why we never hit this one ourselves.
- **vllm#29739** ("[Feature]: GGUF model with architecture qwen3vlmoe is not
  supported yet.", **closed**) — same naming-gap shape, different
  architecture, already closed/resolved for its own architecture; no overlap.
- **vllm#36802** ("[Bug]: Tesla T4 GPU - triton.runtime.errors.OutOfResources:
  out of resource: shared memory...", open) and **PR#43047** ("[Core] Add
  shmem-aware autotune pruner for non-H100 Triton kernels", open, unmerged) —
  a different failure mode (Triton kernel launch fails outright with an
  `OutOfResources` shared-memory error on non-H100 GPUs, for FLA/gated-delta-net
  kernels) than what draft 6 addresses (fused Triton kernels launch fine but
  are pathologically *slow*, not resource-exhausted, on sm75). Both concern
  Triton behavior for hybrid/SSM kernels on non-flagship GPUs, which is why
  we checked, but they are different bugs in different code (PR#43047 patches
  `vllm/model_executor/layers/fla/ops/*.py` autotune configs; draft 6 patches
  vllm-gguf-plugin's GGUF matmul kernel *selection*). No overlap; noted for
  context in draft 6.
- **vllm#47549** ("[Bug]: REGRESSION: FP8 KV cache FlashInfer no longer
  available as attention backend on SM75 (Turing) in v0.24.0", open) — a
  regression report for fp8 KV cache on Turing, related in spirit to our
  T4-dead-ends note in STATUS.md (`--kv-cache-dtype fp8*` hard-blocked, needs
  SM89+) but describing a *regression* (previously worked, broke in v0.24)
  rather than the hard architectural requirement we observed. Not the same
  claim; we did not build a draft around it since it isn't one of our
  findings, and note it here only because it was in the requested duplicate
  check list.
- **vllm-gguf-plugin#80** ("GGUF Qwen 3.5/3.6 MoE architecture not
  supported", open) — the plugin's own tracking issue for Qwen3.5/3.6 support.
  No fix has landed upstream. One comment links a third-party fork
  (`localweights/vllm-gguf-plugin`) reportedly adding support for 3.6-27B/35B
  including MTP; we have not audited that fork's code, so we cannot state
  whether it overlaps with drafts 1–4's specific fixes — flagged here as a
  fork to be aware of, not verified as covering (or not covering) our
  findings. Drafts 1–4 (plus 5) are offered as the concrete, itemized fix list
  against this tracking issue for the upstream-maintained package.
- No open or closed issue in vllm-gguf-plugin's full issue list (#101, #93,
  #88, #80, #76, #75, #25, #11, #2 — the complete set at time of writing)
  covers the trailing-dot bug (draft 1), the weight-transform inversions
  (draft 2), the tuple shard-id bug (draft 3), or the BF16 branch (draft 5).

## Notes on scope decisions

- Draft 4 intentionally bundles six smaller items into one "Qwen3.5 GGUF
  support" report rather than filing six separate issues, per the task's
  guidance that a combined draft referencing the individual patch scripts is
  acceptable — each item is individually small and none is independently
  useful (Qwen3.5 GGUF loading requires clearing all of them together).
- Draft 3 (tuple shard-id weight type) is kept separate from draft 4 even
  though the task description mentions "weight-type params routed through the
  fused loader" in both the item-3 and item-4 descriptions — we scoped it as
  its own report (it has a distinct crash signature, a distinct silent-miscompute
  failure mode, and its own patch script, `patch_gguf_weight_type_loader.py`)
  and draft 4 cross-references it rather than duplicating its content.
- Draft 10 (CUDA_MPS_*) is filed as an "observed behavior" report rather than
  a bug report with a fix, per the task's framing ("low severity") — we did
  not trace the exact EngineCore spawn call site closely enough to propose a
  specific code change, and say so explicitly in the draft.
- Draft 5 (BF16 branch) is filed as a bug report with a full, twice-reproduced
  repro but no fix — root cause was not isolated within the scope of this
  investigation (matches STATUS.md's own note that bug #16 remains unfixed).
