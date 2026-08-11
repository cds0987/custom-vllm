#!/usr/bin/env python3
"""
TASK N5b -- int8-quantize lm_head on top of an existing champion graft.

BACKGROUND
----------
The champion checkpoint (and every graft built on top of it by
graft_gguf_gdn.py) keeps lm_head fp16/bf16 -- it sits in the compressed-
tensors `ignore` list (a literal "lm_head" entry, not a regex). This script
tests whether int8-quantizing lm_head buys extra speed/VRAM without an
unacceptable ppl hit. Unlike graft_gguf_gdn.py's GDN in_proj_* grafts, lm_head
does NOT need a GGUF source: lm_head is never touched by graft_gguf_gdn.py's
GGUF-sourced grafting (it isn't one of GDN_SUFFIXES) and RedHatAI's own GGUF
release quantizes it too, but with llama.cpp's own vocab-tiling that isn't
worth re-deriving for one tensor -- the frame checkpoint ALREADY carries a
full-precision lm_head.weight tensor, which is exactly what
`encode_int8_module` (graft_gguf_gdn.py, reused via importlib here, not
copy-pasted) needs as input: a plain (out_features, in_features) fp32-castable
array. No dequant-from-GGUF step is needed for this one tensor -- the "GGUF
source" gate at the top of graft_gguf_gdn.py's DECISION 1 doesn't apply here.

lm_head IS quality-sensitive (it directly produces every output-token
logit) -- this script's own docstring makes no promises about ppl; run
eval_quality_swebench.py after serving and DISCARD this checkpoint outright
if it WARNs/FAILs, no regrets (per campaign policy).

Scope decision: embed_tokens is a plain nn.Embedding, not nn.Linear -- it is
NOT covered by compressed-tensors' Linear-targeting `ignore`/`config_groups`
matching at all (confirmed: it has no entry in the champion's `ignore` list,
unlike the literal "lm_head" entry), and CompressedTensorsWNA16's WNA16
scheme (the on-disk layout graft_gguf_gdn.py already reverse-engineered) is a
Linear-layer scheme, not an Embedding-layer one. Quantizing embed_tokens
would need a genuinely different vLLM code path this project hasn't
reverse-engineered -- out of scope for this script; lm_head only.

REUSED, NOT REIMPLEMENTED
--------------------------
Imports graft_gguf_gdn.py via importlib (same pattern that script itself
uses for gguf2marlin.py) and calls its `encode_int8_module` (RTN quantize +
compressed-tensors on-disk pack, byte-for-byte the same math already used
for the GDN in_proj_* grafts) and `resolve_frame_dtype` /
`narrow_ignore_list` / `_is_equal_or_regex_match` helpers directly -- no
quantization math duplicated here.

USAGE
-----
    python scripts/graft_lm_head_int8.py --frame <champion_graft_dir> \\
        --out <out_dir> [--group-size 32]

--frame must be an existing compressed-tensors checkpoint (a champion graft
or the champion itself) with an un-ignored... no, WITH lm_head still in its
`ignore` list and an lm_head.weight tensor present on disk. Everything else
(every other tensor, tokenizer files, chat template) is copied through
unmodified, exactly like graft_gguf_gdn.py's own pass 4/5.

Runs entirely on CPU (numpy/torch/safetensors), no GPU required.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("graft_gguf_gdn", SCRIPTS_DIR / "graft_gguf_gdn.py")
ggg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ggg)

MARLIN_SUPPORTED_GROUP_SIZES = (32, 64, 128)
LM_HEAD_NAME = "lm_head"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True, help="dir of the champion (graft) checkpoint to quantize lm_head in")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--group-size", type=int, default=32, choices=MARLIN_SUPPORTED_GROUP_SIZES)
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    args = ap.parse_args(argv)

    ggg.BITS = args.bits
    group_size = args.group_size

    frame_dir = Path(args.frame)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = frame_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    quant_cfg = cfg.get("quantization_config")
    if not quant_cfg or "config_groups" not in quant_cfg:
        raise ggg.GraftConfigError(
            "--frame's config.json has no quantization_config.config_groups -- "
            "this script quantizes lm_head INTO an already-quantized "
            "compressed-tensors checkpoint, it does not create one from scratch."
        )
    ignore_list = list(quant_cfg.get("ignore", []))
    if LM_HEAD_NAME not in ignore_list:
        raise ggg.GraftConfigError(
            f"{LM_HEAD_NAME!r} not found (as a literal entry) in --frame's "
            f"quantization_config.ignore -- expected the champion's own "
            f"convention (a literal 'lm_head' entry); refusing to guess how "
            f"to narrow a differently-shaped ignore rule."
        )

    text_cfg = cfg.get("text_config", cfg)
    frame_dtype = ggg.resolve_frame_dtype(cfg, text_cfg)
    print(f"[graft_lm_head_int8] frame weight_scale dtype: {frame_dtype}", file=sys.stderr)

    weight_map = ggg.load_frame_weight_map(frame_dir)
    lm_head_key = f"{LM_HEAD_NAME}.weight"
    if lm_head_key not in weight_map:
        raise ggg.GraftError(f"{lm_head_key!r} not found in --frame's weight map -- nothing to quantize.")
    shard_name = weight_map[lm_head_key]

    print(f"[graft_lm_head_int8] reading {lm_head_key} from {shard_name}", file=sys.stderr)
    with safe_open(frame_dir / shard_name, framework="pt") as f:
        w_t = f.get_tensor(lm_head_key)
    w = w_t.to(torch.float32).numpy()
    out_features, in_features = w.shape
    print(f"[graft_lm_head_int8] lm_head shape (out_features, in_features) = {w.shape}, "
          f"dtype on disk = {w_t.dtype}", file=sys.stderr)

    weight_packed, weight_scale, weight_shape, rel_rms = ggg.encode_int8_module(w, group_size)
    print(f"[graft_lm_head_int8] rel RMS error dequant(fp) vs dequant(int8 g{group_size}): "
          f"{rel_rms:.6f} (target ~0.005)", file=sys.stderr)

    quantized_tensors = {
        f"{LM_HEAD_NAME}.weight_packed": torch.from_numpy(weight_packed),
        f"{LM_HEAD_NAME}.weight_scale": torch.from_numpy(weight_scale.astype(np.float32)).to(frame_dtype),
        f"{LM_HEAD_NAME}.weight_shape": torch.from_numpy(weight_shape),
    }

    # ---- config.json surgery: narrow ignore, add one config_group ----------
    probe_names = [f"{LM_HEAD_NAME}_probe_never_matches", "model.layers.0.self_attn.q_proj"]
    new_ignore = ggg.narrow_ignore_list(ignore_list, [LM_HEAD_NAME], probe_names)
    new_group = ggg.build_new_config_group(quant_cfg, group_size)
    new_group["targets"] = [LM_HEAD_NAME]

    new_quant_cfg = dict(quant_cfg)
    new_quant_cfg["ignore"] = new_ignore
    new_config_groups = dict(quant_cfg["config_groups"])
    new_config_groups["graft_lm_head_int8"] = new_group
    new_quant_cfg["config_groups"] = new_config_groups

    new_cfg = dict(cfg)
    new_cfg["quantization_config"] = new_quant_cfg
    (out_dir / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- stream-copy every other tensor, shard by shard --------------------
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)

    index_path = frame_dir / "model.safetensors.index.json"
    is_sharded = index_path.exists()
    new_weight_map: dict[str, str] = {}
    for shard, keys in shards.items():
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(frame_dir / shard, framework="pt") as f:
            for key in keys:
                if key == lm_head_key:
                    continue
                out_tensors[key] = f.get_tensor(key)
        if shard == shard_name:
            out_tensors.update(quantized_tensors)
        save_file(out_tensors, out_dir / shard, metadata={"format": "pt"})
        for key in out_tensors:
            new_weight_map[key] = shard

    if is_sharded:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"] = new_weight_map
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    skip_names = {"config.json", "model.safetensors.index.json"} | set(shards.keys())
    for path in frame_dir.iterdir():
        if path.name in skip_names or path.is_dir():
            continue
        shutil.copy2(path, out_dir / path.name)

    manifest = {
        "source_frame": str(frame_dir),
        "group_size": group_size,
        "bits": ggg.BITS,
        "quantized_module": LM_HEAD_NAME,
        "rel_rms_error": rel_rms,
    }
    (out_dir / "lm_head_graft_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("LM_HEAD GRAFT REPORT")
    print("=" * 72)
    print(f"quantized: {LM_HEAD_NAME} -> int8 g{group_size}, rel_rms_error={rel_rms:.6f} (target ~0.005)")
    print(f"wrote {out_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
