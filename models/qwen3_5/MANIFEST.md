# Qwen3.5 (dense) — adapter kiến trúc ĐÃ THUẦN HÓA ✅

Hợp đồng: user chọn BẤT KỲ checkpoint nào đúng kiến trúc Qwen3.5 dense
(hybrid GDN + attention: 0.8B / 2B / 4B / 9B / 27B) — các entrypoint nhận
`--frame/--gguf/--model`, không hardcode size.

## Ma trận hỗ trợ (mọi số đều ĐÃ ĐO — nguồn: STATUS.md)

| Trục | Biến thể | Trạng thái |
|---|---|---|
| engine | vllm (0.26, 0.27.1) | ✅ 19 patch trong `engine/vllm/patches/` |
| engine | sglang, tensorrt_llm | ⬜ PLANNED — mỗi engine sẽ tự mang `patches/` riêng |
| load | `gguf_to_marlin` | ✅ đường champion 9B: ppl 4,7637, decode 36,3 tok/s |
| load | `pytorch_tensor` | ✅ đường 27B: ppl 4,1484, decode 15,8 tok/s |
| load | `pure_gguf` | ✅ chạy mọi GGUF, chậm ~2× (cần engine=vllm) |
| hardware | l4 | ✅ bộ số đầy đủ trong `hardware/l4.py` |
| hardware | t4 / a100 / h100 | ⬜ PLANNED (t4 đắt nhất: Marlin đòi sm_80+) |

## Checkpoint khuyến nghị

- **9B multi-session**: `gunnybd01/qwen35-9b-champion` (prebuilt, pull ~1 phút)
  — serve config trong `engine/vllm/adapter.py` mục "9b".
- **27B chất lượng cao 1-2 user**: `apolo13x/Qwen3.5-27B-quantized.w4a16`
  — serve config mục "27b". KHÔNG cần graft (GDN đã quantize sẵn).

## Cảnh báo sống còn

- ⚠ `utils/fix_qwen35_hf_checkpoint.py` CẤM chạy lên frame multimodal (làm hỏng frame).
- RMS graft 9,4%/11,4% là BÌNH THƯỜNG với bits=4 (0,55% là số của bits=8) — báo động giả kinh niên.
- Script offline: `LLM()` phải nằm trong `if __name__ == "__main__":` (vLLM spawn).
