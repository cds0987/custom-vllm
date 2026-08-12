# Upstream draft index

Twenty draft reports (17 bug/observation reports + 1 PR proposal + 1 feature
request + 1 status question) produced from the Qwen3.5-GGUF-on-vLLM
optimization campaign recorded in `STATUS.md`, `docs/routing-research.md`,
and the anchor-based source patches in `scripts/patch_*.py` /
`scripts/graft_*.py` / `scripts/gguf2marlin.py`. All local-only — **nothing
in this directory has been posted to GitHub. Posting is a separate step that
requires explicit owner approval before any of these are filed.**

Drafts 1–12 were written 2026-08-08 from the first phase of the campaign.
Drafts 13–20 were added after three more days of optimization work (TASK
G/G2a, K/K2-K4, L, M, N1-N6b, F2/F2b/F2c, C2, P1, and the A/B attention
backend investigation) surfaced substantially more upstream-worthy findings.
Duplicate/overlap status for drafts 1–12 was checked live against GitHub on
2026-08-08 (see each draft's "Duplicate check" section). **Drafts 13–20 have
NOT had a live duplicate-check pass against current GitHub issues/PRs** —
this environment had no network access during the 2026-08-11 writing
session; each new draft's own "Duplicate check" section says so explicitly
and should be re-verified live before filing.

## Table

| # | File | Target repo | Title | Severity | Local fix? | Ready to send? |
|---|---|---|---|---|---|---|
| 1 | `01-vllm-gguf-plugin-trailing-dot-drops-bare-params.md` | vllm-gguf-plugin | Trailing-dot bug silently drops bare GGUF parameters (A_log) | Critical | Yes, 1-line | Yes |
| 2 | `02-vllm-gguf-plugin-qwen35-weight-transforms-not-inverted.md` | vllm-gguf-plugin | Qwen3.5 GGUF weight transforms (A_log, norms, conv1d, V-head tiling) never inverted on load | Critical | Yes | Yes |
| 3 | `03-vllm-gguf-plugin-tuple-shard-id-weight-type.md` | vllm-gguf-plugin | Tuple shard-ids collapse per-shard GGUF weight types (Q5_K read as Q4_K) | High | Yes | Yes |
| 4 | `04-vllm-gguf-plugin-qwen35-support-batch.md` | vllm-gguf-plugin | Qwen3.5 GGUF support: 6 correctness/compatibility gaps | High | Yes | Yes |
| 5 | `05-vllm-gguf-plugin-bf16-unquantized-branch-broken.md` | vllm-gguf-plugin | BF16/unquantized GGUF branch loads ~0.03 GiB then crashes | High | No (root cause not isolated) | Yes, as repro-only report |
| 6 | `06-vllm-gguf-plugin-hybrid-kernel-dispatch-pr-proposal.md` | vllm-gguf-plugin | PR proposal: shape-based dispatch for GGUF matmuls (fused/dequant) | N/A (enhancement) | Yes, working PR-shaped patch | Yes |
| 7 | `07-vllm-qwen35-causallm-not-registered.md` | vllm | Qwen3_5*ForCausalLM defined but never registered | Critical | Yes | Yes |
| 8 | `08-vllm-qwen35-text-only-missing-hybrid.md` | vllm | Text-only Qwen3.5 missing IsHybrid + mamba-state classmethods | High | Yes | Yes |
| 9 | `09-vllm-qwen35-embed-tokens-missing-quant-config.md` | vllm | Qwen3_5Model.embed_tokens built without quant_config | Medium | Yes | Yes |
| 10 | `10-vllm-cuda-mps-env-not-propagated-to-enginecore.md` | vllm | CUDA_MPS_* env vars not propagated to spawned EngineCore workers | Low | No (observation only) | Yes, as observation |
| 11 | `11-transformers-qwen35-config-missing-vocab-size.md` | transformers | Qwen3_5Config missing top-level vocab_size | Medium | Yes | Yes |
| 12 | `12-gguf-py-missing-dt-bias-tensor-template.md` | gguf-py (llama.cpp) | Missing tensor template for qwen3.5's linear_attn.dt_bias | Medium | Yes | Yes |
| 13 | `13-vllm-gguf-plugin-pypi-wheel-abi-mismatch-silent-triton-fallback.md` | vllm-gguf-plugin | PyPI wheel's `_C_gguf` ABI-mismatches torch, silently falls back to Triton (up to 3.9x slower) | Critical | Yes (build-from-sdist workaround) | Yes, but re-run live duplicate check first |
| 14 | `14-vllm-mergedcolumnparallellinear-packed-gdn-load.md` | vllm | `MergedColumnParallelLinear` can't load compressed-tensors packed shards for fused layers; `weight_shape` silently corrupted | High | Yes | Yes, but re-run live duplicate check first |
| 15 | `15-compressed-tensors-find-matched-target-substring-catchall.md` | compressed-tensors | `find_matched_target` matches catch-all `["Linear"]` by substring before fused-component reconciliation | High | No (config-authoring workaround only) | **No — needs live duplicate check + line-number re-verification against current main first (see caveat in file)** |
| 16 | `16-vllm-parallel-lm-head-no-quantization-path.md` | vllm | `ParallelLMHead`/`VocabParallelEmbedding` have no compressed-tensors quantization path | N/A (feature request) | No | Yes, but re-run live duplicate check first |
| 17 | `17-vllm-flashinfer-cascade-attention-hardcoded-false.md` | vllm | FlashInfer `use_cascade_attention()` hardcoded False; status/plan unknown | Medium | No (not patchable downstream) | Yes, but re-run live duplicate check first |
| 18 | `18-vllm-mamba-cache-mode-silent-override-and-opaque-block-size-assert.md` | vllm | `--mamba-cache-mode align` silently reverts to `none`; opaque `block_size<=max_num_batched_tokens` assert | Low-Medium | No (learned the rules, no patch) | Yes, but re-run live duplicate check first |
| 19 | `19-vllm-gguf-plugin-datasets-downgrades-huggingface-hub-import-crash.md` | vllm-gguf-plugin | `pip install datasets` downgrades huggingface_hub, breaks import of ANY model | High | Yes (re-pin workaround in setup_env.sh) | Yes, but re-run live duplicate check first |
| 20 | `20-vllm-attention-backend-env-var-silently-ignored.md` | vllm | `VLLM_ATTENTION_BACKEND` env var silently ignored (moved to `--attention-backend`) | Medium | No (switched to new flag) | Yes, but re-run live duplicate check first; re-capture exact log text |

## Repos

- **vllm-project/vllm-gguf-plugin** — items 1–6, 13, 19 (the bulk). GGUF
  loading moved out-of-tree from vllm core into this plugin as of vllm 0.26
  (commit `1534218`), so this remains the primary target for correctness
  work, joined now by a critical silent-performance-regression report (13)
  and a dependency-conflict report (19).
- **vllm-project/vllm** — items 7–10, 14, 16, 17, 18, 20 (core model
  registry/architecture, the compressed-tensors integration surface, the
  scheduler/config UX gaps, and one env-var deprecation UX gap).
- **vllm-project/compressed-tensors** — item 15 (the `find_matched_target`
  substring-catch-all bug). New target repo as of this update — not covered
  by drafts 1–12, since prior work hadn't yet touched quantized-checkpoint
  authoring for fused layers.
- **huggingface/transformers** — item 11 (`Qwen3_5Config`).
- **ggml-org/llama.cpp** (source of the `gguf` PyPI package) — item 12.

## What changed since 2026-08-08 (drafts 13–20)

The campaign's next three days moved from "make Qwen3.5 GGUF load and run
correctly" (drafts 1–12) into quantization-format exploration (AWQ/GPTQ/
GGUF-to-Marlin transcoding, GDN-mixer quantization grafting), long-context/
scheduler tuning, and infrastructure hardening. That surfaced bugs in
different parts of the stack than before: vLLM's own compressed-tensors
integration (14, 15, 16), scheduler/config diagnostics (18, 20), a
packaging/distribution bug specific to how the plugin is normally installed
(13), and a dependency-resolution bug that isn't GGUF-specific at all (19).
Item 17 is closer to a status inquiry than a bug report.

## Considered but NOT written up — with reasons

The task's candidate list (from STATUS.md/routing-research.md) included a
few items that were deliberately **not** turned into drafts:

- **"gguf TensorNameMap returns a WRONG candidate (not 'no candidate') for
  Qwen3.5's GDN tensors and q_norm/k_norm, due to a shortest-name tie-break
  mixing in aliases from other architectures."** Investigated in depth
  (`scripts/gguf2marlin.py`'s own docstring, "LIMITATIONS" section, TASK
  K3/K5). **Excluded because the ambiguity lives in OUR OWN inversion
  heuristic, not in an upstream defect.** `gguf-py`'s `tensor_mapping.py` is
  designed and used by the `gguf` package for one purpose: recognizing which
  HF-side name convention a tensor came from when *writing* a GGUF file. Our
  own `gguf2marlin.py` reuses that table for the opposite, unsupported
  purpose — reverse-inferring "the" canonical modern HF name for an
  arbitrary ggml tensor role via a "shortest plausible name" tie-break we
  invented — and the table legitimately contains multiple valid HF aliases
  per ggml role because multiple real architectures use that role
  differently. That is a limitation of our script's generic-mapper design
  (already documented and hardcoded around in `gguf2marlin.py` and
  `graft_gguf_gdn.py`), not a data-quality bug in `gguf-py`'s own table for
  its own intended use. Filing it upstream as a `gguf-py` bug would be
  asking the wrong project to fix our tie-break. (Note: this is distinct
  from draft 12, which IS a genuine gguf-py gap — a template that is simply
  *absent*, not one that is ambiguous.)
- **The vLLM scheduler behavior where a newly-admitted request's chunked
  prefill competes for the same per-step token budget as running requests'
  decode steps** (documented in detail in `docs/routing-research.md` Part 1c,
  and empirically observed as TASK F2b/F2c's "`num_requests_waiting=0` but
  still slow" finding). **Excluded because it is not a bug** — it is
  documented, intentional scheduler design (`vllm/v1/core/sched/
  scheduler.py`'s two-phase `schedule()`), and vLLM already exposes the
  exact knobs needed to control it (`--long-prefill-token-threshold`,
  `--max-num-partial-prefills`, `--max-num-batched-tokens`) — which is what
  TASK F2c's `--max-num-batched-tokens 1088` tuning actually used to fix our
  own SLA problem. Filing "the scheduler doesn't prioritize decode over new
  prefill" as a bug against a system that already ships the config to do
  exactly that would not be accurate; nothing here rises above "our own
  tuning note," which stays in `docs/routing-research.md`.
- **Everything in `docs/routing-research.md` Part 2 (P1-P5, the
  TTFT-predictor gateway / two-tier priority queue / KV-aware multi-instance
  routing proposals).** These are proposed **additions to our own gateway
  layer**, explicitly designed to sit in front of vLLM without any vLLM code
  change (P2 does suggest vLLM's existing, already-shipped `--policy
  priority` flag, which needs no upstream work at all). None of them
  describe a vLLM defect or gap; they're application-level design work, not
  upstream material.
- **TASK A/B's `FLASH_ATTN` + fp8 KV cache incompatibility**
  (`ValueError: kv_cache_dtype not supported` at startup). **Excluded
  because this already fails loudly and correctly** — it's a clear,
  immediate, correctly-attributed error at server startup, not a silent or
  confusing failure. Nothing to report; the auto-selection logic that steers
  users away from this combination (offering only `['FLASHINFER',
  'TRITON_ATTN']`) is already doing the right thing.
- **TASK N1's finding that `--max-num-seqs` is a poor SLA lever under
  client-side burst load.** Excluded — this is expected queueing-theory
  behavior of any admission cap facing an already-saturating client, not a
  vLLM defect; the finding is operational guidance for our own deployment,
  not a report against vLLM.
- **TASK C2's finding that `OffloadingConnector`/CPU KV offload cannot
  increase concurrent *session* count for hybrid models (only extends
  context/prefix reach), because mamba-state has no offload tier.** This
  is architecturally true and clearly documented as a design property in
  the connector's own code (single-tier KV offload; mamba/conv state lives
  in a separate, non-offloaded pool) rather than a bug — it reads as a
  legitimate feature gap (mamba-state offload doesn't exist anywhere yet,
  for any consumer), but we have not exercised it enough — no live
  reproduction beyond static/architectural reading, no measured "this breaks
  a real workflow" case as concrete as drafts 16/17 have — to justify a
  confident feature-request draft yet. Flagged here as a candidate for a
  future draft if the campaign returns to CPU-offload work with a concrete
  failing scenario in hand.

## Duplicate-check detail (drafts 1–12, checked live 2026-08-08)

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
  `model_type` underscore-vs-none naming class of bug, but for a different
  architecture and a different code path. Confirms the pattern recurs; no
  direct file overlap with any draft here.
- **vllm#36456** ("[Bug]: Local GGUF path fails with 'architecture qwen35 is
  not supported yet' even when --hf-config-path is provided", open) — a
  distinct bug, upstream of everything in drafts 4/7/8/9 in the "load a
  local Qwen3.5 GGUF" sequence, but a different bug in a different function;
  noted, not merged into any draft.
- **vllm#29739** (closed) — same naming-gap shape, different architecture,
  already resolved for its own architecture; no overlap.
- **vllm#36802** ("[Bug]: Tesla T4 ... OutOfResources: out of resource:
  shared memory...", open) and **PR#43047** (open, unmerged) — a different
  failure mode (hard crash on non-H100 GPUs) than draft 6 (fused kernels
  launch fine but are pathologically slow on sm75, not resource-exhausted).
  No overlap; noted for context in draft 6, and again in draft 13 (both
  concern Triton/GGUF-plugin behavior on non-flagship GPUs).
- **vllm#47549** ("[Bug]: REGRESSION: FP8 KV cache FlashInfer no longer
  available as attention backend on SM75 (Turing)...", open) — related in
  spirit to our T4 dead-ends note but describes a regression, not the hard
  architectural requirement we observed. Not the same claim; noted only
  because it was in the original duplicate-check list.
- **vllm-gguf-plugin#80** ("GGUF Qwen 3.5/3.6 MoE architecture not
  supported", open) — the plugin's own tracking issue for Qwen3.5/3.6
  support. No fix landed upstream. Drafts 1–6 (plus 13, 19 as of this
  update) are offered as the concrete, itemized fix/observation list
  against this tracking issue.
- No open or closed issue in vllm-gguf-plugin's full issue list at time of
  the 2026-08-08 check (#101, #93, #88, #80, #76, #75, #25, #11, #2) covered
  the trailing-dot bug (draft 1), the weight-transform inversions (draft 2),
  the tuple shard-id bug (draft 3), or the BF16 branch (draft 5).

**Drafts 13–20 have not had this live pass run against them** — the
2026-08-11 writing session had no network access. Each of those drafts'
"Duplicate check" section says so and should be re-run (`gh issue list`/
`gh api` against vllm-project/vllm, vllm-project/vllm-gguf-plugin, and
vllm-project/compressed-tensors) before any of them are filed.

## Notes on scope decisions

- Draft 4 intentionally bundles six smaller items into one "Qwen3.5 GGUF
  support" report rather than filing six separate issues — each item is
  individually small and none is independently useful (Qwen3.5 GGUF loading
  requires clearing all of them together).
- Draft 3 is kept separate from draft 4 despite topical overlap — it has a
  distinct crash signature, a distinct silent-miscompute failure mode, and
  its own patch script; draft 4 cross-references it rather than duplicating.
- Draft 10 is filed as an "observed behavior" report rather than a bug
  report with a fix — the exact EngineCore spawn call site was not traced
  closely enough to propose a specific code change, and the draft says so.
- Draft 5 is filed as a bug report with a full, twice-reproduced repro but
  no fix — root cause was not isolated within scope (STATUS.md's bug #16,
  still unfixed as of this update).
- Draft 14 and draft 15 are two DIFFERENT bugs in the SAME user journey
  (quantizing Qwen3.5's GDN mixer with compressed-tensors): 15 is why a
  correctly-written `config_groups` entry for a fused layer gets silently
  ignored (a compressed-tensors routing bug, reachable BEFORE loading even
  starts); 14 is why, once that's worked around and loading is attempted,
  vLLM's own merged-column loader mishandles packed shards (a vllm loader
  bug, reachable AFTER 15 is worked around). Filed separately because they
  are different bugs in different repos with different fixes, even though
  discovering them required the exact same reproduction session (TASK M-exec).
- Draft 13's severity (Critical) reflects that it is a *silent* correctness-
  adjacent regression (no error, no warning) affecting the DEFAULT
  installation method (`pip install vllm-gguf-plugin`) — every performance
  number collected before this was found, across this entire campaign's
  earlier phase, was unknowingly measuring the slow fallback path.
- Draft 17 is filed as a status question rather than a bug report, since
  the hardcoded `False` reads as deliberate (it's commented) rather than an
  oversight — we don't have visibility into why, and say so.

## Next step

None of drafts 1–20 have been filed on GitHub. **Filing requires explicit
approval from the project owner** — this index and the drafts it points to
are a preparation artifact, not a queue that gets submitted automatically.
Before filing drafts 13–20 specifically: re-run a live duplicate check
against current GitHub state (network access was unavailable during their
writing session), and for draft 15 specifically, re-verify the exact
`compressed-tensors` file/line references against current `main` before
submitting.
