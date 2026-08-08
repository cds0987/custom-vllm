# [Bug] Qwen3_5Model.embed_tokens is constructed without quant_config, breaking quantized (GGUF) token-embedding loading

**Target repo:** vllm-project/vllm
**Severity:** Medium (crash, but only for checkpoints whose embedding happens to be stored in a quantized ggml type rather than F16/F32)
**Affected versions:** vllm 0.26
**Local fix:** `scripts/patch_vllm_qwen35_embed.py`
**Duplicate check:** No existing issue found.

## Summary

`vllm/model_executor/models/qwen3_5.py`'s `Qwen3_5Model.__init__` constructs the token embedding without `quant_config`:

```python
self.embed_tokens = VocabParallelEmbedding(
    self.vocab_size,
    config.hidden_size,
)
```

`VocabParallelEmbedding` accepts an optional `quant_config` that makes it quantization-aware — specifically, able to receive the `.qweight_type` metadata GGUF weight loading attaches to quantized parameters. Without it, loading a GGUF checkpoint whose token embedding happens to be stored in an actually-quantized ggml type (some quant recipes keep the embedding table in F16/F32 for quality reasons and only quantize it in others — e.g. it varies by which llama.cpp quantization profile produced the file) fails with:

```
ValueError: There is no module or parameter named 'embed_tokens.qweight_type' in Qwen3_5Model.
```

vllm's own `Qwen3Model` (via `Qwen2Model`) already passes `quant_config` for exactly this reason; `Qwen3_5Model` (and its parent `Qwen3NextModel`, which has the identical gap) simply omits it — this looks like an oversight during Qwen3.5's addition rather than an intentional divergence, since every neighboring model in the same family handles it correctly.

## Reproduction

Depends on which specific GGUF quant profile is used — reproduces when the embedding table is quantized rather than kept in F16/F32:

```bash
vllm serve <a Qwen3.5 GGUF repo/quant where embed_tokens is quantized> \
  --hf-config-path Qwen/Qwen3.5-2B --tokenizer Qwen/Qwen3.5-2B
```

## The fix

```python
self.embed_tokens = VocabParallelEmbedding(
    self.vocab_size,
    config.hidden_size,
    quant_config=self.quant_config,
)
```

One added keyword argument, matching the pattern already used in `Qwen2Model`/`Qwen3Model`. Also applies to `Qwen3NextModel`, which has the same gap for the same reason (not filed separately since it's the identical one-line change in a sibling class — we scoped our fix and testing to the Qwen3.5 path only, but the maintainers may want to fix both in one PR).
