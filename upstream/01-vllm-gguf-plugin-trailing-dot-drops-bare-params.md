# [Bug] `find_hf_name_in_tensor_map()` appends a trailing dot to bare (suffixless) parameters, silently dropping them — breaks every SSM/hybrid architecture

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** Critical (silent correctness failure, not a crash — affects every SSM/hybrid GGUF model the plugin claims to support)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26, gguf 0.19.0
**Local fix:** `scripts/patch_gguf_empty_suffix.py`
**Duplicate check:** No existing issue found in vllm-project/vllm-gguf-plugin covering this (checked #101, #93, #88, #80, #76, #75, #25, #11, #2). Not covered by vllm-project/vllm#38122 / PR #38140 either — those only cover `model_type` naming and vision-config `depth`.

## Summary

`vllm_gguf_plugin/weights_adapter/default.py`'s `find_hf_name_in_tensor_map()` splits an HF parameter name into `(base_name, suffix)`, where `suffix` is the empty string for a *bare* parameter (one with no `.weight`/`.bias` tail — e.g. a standalone bias-like tensor such as an SSM decay-gate log parameter). It then unconditionally rejoins:

```python
return gguf_name + "." + suffix
```

For a bare parameter this produces `"blk.1.ssm_a."` — a trailing dot — while the GGUF file actually stores the tensor under `"blk.1.ssm_a"` (no dot). The string mismatch means nothing in the model ever loads this tensor from the checkpoint; it silently keeps its framework-default initialized value.

## Why it is silent

The plugin has exactly one integrity check after building the name map: it asserts that every HF parameter name appears in `gguf_to_hf_name_map.values()`. That check passes, because the mapping *is* added — just under the wrong (dotted) key. The lookup that happens later, at actual weight-load time, is the one that misses, and nothing checks that every entry that was *added* was also *consumed*. So:

- No exception.
- No warning.
- No "failed to map GGUF parameters" error (that's the one guard that exists, and it's satisfied).
- The server starts, the model answers requests, output is fluent, plausible-looking token sequences — right up until it degenerates.

## Reproduction

Serve any GGUF checkpoint of an SSM/hybrid architecture whose per-layer decay-gate parameter has no `.weight`/`.bias` suffix in HF naming — Mamba, Jamba, Qwen3-Next, Qwen3.5 (gated-delta-net layers) are all affected the same way, since this parameter is bare in all of them:

```bash
vllm serve unsloth/Qwen3.5-2B-GGUF:Q4_K_M \
  --hf-config-path Qwen/Qwen3.5-2B \
  --tokenizer Qwen/Qwen3.5-2B
```

The server starts cleanly. Send any prompt:

```
User: What is the capital of France?
Model: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

## Root cause and mechanism

For Qwen3.5, the casualty is `linear_attn.A_log` in all 24 gated-delta-net layers. vllm computes the SSM decay gate as:

```python
g = -A_log.float().exp() * softplus(a + dt_bias)
```

`A_log` is never loaded, so it keeps its zero-initialized value, and every layer computes `-exp(0) = -1` uniformly, for every head, in every gated-delta-net layer. That destroys the recurrence in three-quarters of the network (Qwen3.5 interleaves gated-delta-net layers with full-attention layers). The remaining attention layers still function, so the model doesn't crash or emit malformed tensors — it emits well-formed logits, and the degenerate distribution collapses to argmax over token 0, printing `"!"` forever. This is a worse failure mode than a crash: it looks like a working, if broken, model, and could easily be misattributed to quantization noise or a different bug entirely.

## The fix

In `vllm_gguf_plugin/weights_adapter/default.py`, only append the separator+suffix when a suffix actually exists:

```python
# before
return gguf_name + "." + suffix

# after
return gguf_name + "." + suffix if suffix else gguf_name
```

That's the entire fix — one conditional expression. Verified against `scripts/patch_gguf_empty_suffix.py` in our repro repo (custom_vllm), which applies this anchor-based, idempotent patch directly to the installed package.

## Verification

After the fix, `A_log` (and `dt_bias`, and every other bare per-layer scalar/vector parameter) loads correctly. We cross-checked the loaded values numerically against the original HF checkpoint (after also inverting llama.cpp's GGUF-specific weight transforms — see the companion report "GGUF checkpoints of Qwen3.5 are not byte-faithful HF weights"): `A_log`, `dt_bias`, `conv1d`, and norm weights match the HF reference exactly; quantized tensors differ by 0.004–0.07, consistent with expected quantization noise. The model produces coherent, on-topic responses instead of `"!!!!"`.

## Impact

This is not Qwen3.5-specific. Every hybrid/SSM architecture the plugin advertises support for — Mamba, Jamba, Qwen3-Next, and any future architecture with a bare (non-`.weight`/`.bias`) per-layer parameter — is silently broken by this bug whenever that parameter's HF name has no suffix. We'd recommend, in addition to the one-line fix, adding a positive-coverage assertion (every added map entry gets read back at least once during weight loading) so an equivalent bug can't reach production silently again.
