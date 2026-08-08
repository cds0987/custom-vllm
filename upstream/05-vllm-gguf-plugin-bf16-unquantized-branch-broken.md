# [Bug] BF16 (unquantized) GGUF branch is broken — loads ~0.03 GiB instead of full weights, then crashes inside torch.compile at a zero-sized split

**Target repo:** vllm-project/vllm-gguf-plugin
**Severity:** High (feature is advertised — GGUF's F16/BF16/F32 storage types are meant to be "unquantized" passthroughs — but is unusable; no workaround short of using a quantized GGUF or a non-GGUF checkpoint)
**Affected versions:** vllm-gguf-plugin 0.0.4, vllm 0.26, gguf 0.19.0
**Status:** No fix yet — reproduced identically on two different GPU architectures (sm75 / Tesla T4 and sm89 / L4), which rules out a hardware-specific torch.compile codegen issue and points at something structural in how the plugin sizes/loads unquantized tensors.

## Summary

Serving a BF16-quantized GGUF (i.e. an *unquantized* GGUF — BF16 is `gguf`'s "no real quantization, just a narrower float" storage class) of Qwen3.5 loads a suspiciously small amount of data (~0.03 GiB reported, versus ~4–5 GiB expected for an unquantized 2B–4B model) and then crashes during `torch.compile` graph capture with a shape mismatch inside a `split` call whose target size includes a zero dimension.

## Reproduction

```bash
vllm serve unsloth/Qwen3.5-2B-GGUF:BF16 \
  --hf-config-path Qwen/Qwen3.5-2B --tokenizer Qwen/Qwen3.5-2B
```

Observed on:
- Tesla T4 (sm75, Colab)
- L4 (sm89, Colab)

Same failure signature on both, byte-for-byte as far as the visible symptoms go — model claims to load `0.03 GiB` of weights (versus the multi-GiB checkpoint on disk), then crashes shortly after, during `torch.compile`, at:

```
RuntimeError: ... split(size=(s72, 0)) ...
```

(exact operator/line varies slightly by vllm build, but the shape signature — a split whose target includes a literal `0`-sized chunk — is consistent across both GPUs).

## Analysis

The `0.03 GiB` figure is the tell: that is far too small to be any meaningful fraction of even a 2B-parameter model at 2 bytes/param (~4 GiB expected). Something in the BF16-specific code path is either failing to walk the full tensor list, or is misreading tensor shapes/counts for the unquantized storage class specifically — quantized GGUF branches (Q4_K_M, Q5_K_M, Q8_0, etc., see the companion draft's GGUF format table) do not exhibit this and load the expected weight volume. The subsequent `torch.compile` crash, with a `0` literal appearing in a `split()` target size, is consistent with a downstream consumer (e.g. a QKV/gate-up split during graph capture) receiving a tensor whose shape metadata says one dimension is `0` — i.e. a shape that was never correctly populated from the (mostly-unread) GGUF tensor data.

We have not isolated the exact code path responsible (this would require instrumenting the plugin's unquantized-tensor loader, which is out of scope for this report), but the facts established are:
- The failure is reproducible and identical across two GPU architectures, ruling out a compile-target-specific codegen bug.
- The failure is specific to the unquantized (BF16) storage class — quantized GGUF loading through the same plugin works correctly (see companion reports 01–04 once their fixes are applied).
- The `0.03 GiB` loaded-weight figure strongly suggests a loading-completeness bug (most tensors never actually read) rather than a purely numerical one.

## Impact

Anyone who wants to serve a GGUF specifically *because* they want the unquantized reference weights in GGUF container format (e.g. to compare against a quantized variant, or because that's the only format a checkpoint was published in) cannot currently do so through vllm-gguf-plugin. The workaround is to use a quantized GGUF variant or fall back to safetensors — neither of which is available for every published Qwen3.5 GGUF repo.

## What we'd need to file a fix

We were not able to isolate root cause within the scope of this investigation (it likely requires tracing the unquantized-tensor path in the plugin's weight loader/adapter against the actual GGUF F16/BF16/F32 tensor-info parsing in the `gguf` package, comparing tensor count and byte offsets read versus expected). We're filing this as a bug report with full repro rather than a bugfix so it's tracked and so anyone already familiar with the loader's internals can bisect faster than we could from the outside.
