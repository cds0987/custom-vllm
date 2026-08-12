# [Bug] Qwen3_5Config does not expose vocab_size at the top level, breaking any code that constructs a model from the composite config directly

## Tóm tắt cho người không chuyên

Cấu hình của Qwen3.5 giấu kích thước từ vựng bên trong một cấu hình con
(phần "văn bản"), nhưng đoạn mã dựng mô hình lại đi tìm nó ở cấp ngoài cùng
— chỗ không tồn tại — nên bị lỗi ngay khi ai đó dựng mô hình theo cách thông
thường nhất (đưa cả cấu hình tổng vào hàm dựng có sẵn). Cách sửa đơn giản là
thêm một "cổng chuyển tiếp" để cấp ngoài cùng cũng đọc được giá trị từ cấu
hình con, nhưng bản vá gặp thêm rắc rối kỹ thuật vì kiểu dữ liệu dataclass
chặt chẽ không cho thêm thuộc tính kiểu đó dễ dàng.

**Target repo:** huggingface/transformers
**Severity:** Medium (crash for any external caller that follows the otherwise-normal pattern of building `AutoModelForCausalLM.from_config(the_full_config)`)
**Affected versions:** transformers 5.13.1
**Local fix:** `scripts/patch_transformers_qwen35.py`
**Duplicate check:** No existing issue found for this specific attribute gap on `Qwen3_5Config`.

## Summary

`Qwen3_5Config` is a composite config (separate `text_config`/`vision_config` sub-configs, following the same pattern as other Qwen-VL configs) and does not expose `vocab_size` at the top level — only `config.text_config.vocab_size` exists. But `Qwen3_5TextModel.__init__` reads `config.vocab_size` directly, which crashes:

```
AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'
```

This fires whenever *any* code — not just vllm-gguf-plugin, which is where we hit it (building a dummy model on the meta device to inspect its `state_dict()` shape/names, in order to build a GGUF-to-HF tensor name map) — constructs a `Qwen3_5ForConditionalGeneration` (or the text-only variant) from the full composite config instead of from `config.text_config`. Every neighboring Qwen-VL config we checked in the same family (`Qwen3VLConfig`, etc.) forwards `vocab_size` from `text_config` for exactly this reason; `Qwen3_5Config` is the outlier.

## Reproduction

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
# AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'
```

## Why the obvious fix (a @property) doesn't work

The natural fix — add `vocab_size` as a `@property` forwarding to `self.text_config.vocab_size` — does not work here, because `PreTrainedConfig` is a "strict dataclass" (via `huggingface_hub.dataclasses`) that type-checks every attribute against its declared type on assignment, and `vocab_size: int` is already declared as a real dataclass field somewhere up the MRO. The strict-dataclass `__init__` tries to assign the *property object itself* as `vocab_size`'s value during construction and fails type validation before the property would ever get a chance to be read:

```python
# does NOT work — strict dataclass rejects assigning a property object
# to a field declared as `int`
@property
def vocab_size(self):
    return self.text_config.vocab_size
```

## The fix

Assign a real `int` to `self.vocab_size` inside `__post_init__`, once `text_config` has been resolved, instead of using a property:

```python
elif self.text_config is None:
    self.text_config = self.sub_configs["text_config"]()

self.vocab_size = self.text_config.vocab_size   # <-- added

super().__post_init__(**kwargs)
```

This satisfies the strict-dataclass type check (it's a plain `int` assignment, same as any other field) and makes `config.vocab_size` available immediately after construction, matching how every other composite Qwen-VL config in the same module already behaves for this field.

## Verification

After the fix, `AutoModelForCausalLM.from_config(config)` (and vllm-gguf-plugin's meta-device dummy-model construction, which depends on it — see the companion "Qwen3.5 GGUF support batch" report, item 3) succeeds without the `AttributeError`.
