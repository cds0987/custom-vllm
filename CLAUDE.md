# custom-vllm

**Một lệnh cho mọi thứ**: `bash run.sh serve 9b|27b` (Colab = đúng 1 cell — xem `run.sh help`).
**Registry đệ quy**: `python register.py --flat` in toàn bộ ma trận hỗ trợ; mỗi folder có
`register.py` riêng, folder không có = vô hình. Cấu trúc: `models/` (kiến trúc đã thuần hóa,
khuôn transformers) + `sdk/ loading/ logging/ utils/` (tầng ngoài) + `bench/` + `tests/`.

## Tầm nhìn sản phẩm

**custom-vllm KHÔNG phải dự án tối ưu một model.** Nó là sản phẩm thương mại: layer
adapter hóa LLM serving để business dùng ngay, theo 4 trục (sơ đồ user, 2026-08-12):

```
custom-vllm
├── Adapter-Architecture   → Qwen3.5 ✅, Gemma4 ⬜, nhiều kiến trúc khác ⬜
├── Adapter-Engine         → vLLM ✅, SGLang ⬜
├── Adapter-Hardware       → Single GPU (L4 ✅, T4 ⬜), Multi-Node K8s ⬜
├── Adapter-Format         → GGUF ✅ (→ compressed-tensors W4A16), format khác ⬜
└── Cross-model KV transfer→ 4B→9B copy ✅ (TTFT ×1,66), x→27B mapper 🔶, 0.8B/2B ❌ (đóng bằng đo)
```

Giá trị bán: khách chọn 1 điểm trong không gian (kiến trúc × engine × phần cứng ×
format), nhận một serving stack đã tối ưu, đã đo, dùng ngay.

## Instance #1 đã chứng minh: Qwen3.5-9B + vLLM + L4 + GGUF

Đã đi hết vòng convert → patch → serve → bench → tune (số liệu xác nhận trên vLLM 0.27.1):

- **Champion v2**: frame `RedHatAI/Qwen3.5-9B-quantized.w4a16` + trọng số GDN trích từ
  `unsloth/Qwen3.5-9B-GGUF Q4_K_M`, requantize int4 group-size 32, compressed-tensors
  (kernel Marlin). Build: `models/qwen3_5/load/gguf/gguf_to_marlin.py --bits 4 --group-size 32`.
  Prebuilt trên HF: `gunnybd01/qwen35-9b-champion` (pull ~1 phút).
- ppl **4.7637** (baseline bf16: 5.13). Decode ~390 tok/s server / ~520 offline trên
  prefix 30K. Prefill 2789–2934 tok/s.
- Serve chuẩn: `vllm serve <champion> --max-model-len 32768 --max-num-batched-tokens 1088
  --enable-prefix-caching --mamba-cache-mode align --kv-cache-dtype fp8_e4m3
  --gpu-memory-utilization 0.85`
- Báo động giả tái diễn: RMS graft 9.4%/11.4% là BÌNH THƯỜNG với bits=4; `LLM()` ở
  module level chết vì spawn (bọc `if __name__ == "__main__":`); KHÔNG chạy
  `fix_qwen35_hf_checkpoint.py` lên frame.

## Instance #2 (nghiên cứu đã chốt): cross-model KV transfer trong họ Qwen3.5

Chiến dịch E0→E8 (2026-08-14→16, chi tiết STATUS.md, code `models/qwen3_5/kv_transfer/`):

- **4B→9B: bê nguyên cache CHẠY** (không cần mapper — needle 100% tới 30K, decode
  parity, TTFT 30K ×1,66 hai-GPU / ×1,15-1,2 đồng trú E3C). Scope an toàn:
  chat/QA/RAG; function-calling hụt biên mỏng (9/20 vs 19/20 bf16) — cần polisher.
- **Định luật ghép đôi (E7)**: attention thẳng hàng toàn họ; số phận cặp nằm 100%
  ở GDN — CCA-GDN ≥0,9 bê được / ~0,8 học được (4B→27B) / ~0,23 tường (0.8B/2B).
- **E8 đóng**: phương ngữ GDN nhóm nhỏ KHÔNG sửa được bằng adapter nhẹ (3 đòn
  LoRA/loss đều 0/5 dù gate thông tin sáng 5/5). 4B = prefill-helper duy nhất của 9B.
- 2 luật đã kiểm ≥4 lần: **error-placement** (R²/NLL/nMSE không dự đoán chức năng —
  chỉ tin thí nghiệm chức năng) và **nhất quán nội tại** (vá suffix càng vá càng hỏng).
- Cửa sản phẩm còn lại: **Phase C** = KVConnector vLLM (`kv_transfer_config`) + đo
  lại trên W4A16 Marlin thật.

## Quy tắc cứng — luôn áp dụng

- Mọi thứ chạy trên GPU: đưa plan → user duyệt → mới chạy. Cấm tự làm.
- Chỉ notebook A (server `colab-mcp`). Không subagent, không B/C — trừ khi user ra lệnh đích danh.
- Không tự kết luận "hết đường"; không ghi HF token vào file/commit/log.
- **Mọi sản phẩm làm ra (checkpoint/mapper/data — kể cả kết quả âm) phải save lên
  HF trong CÙNG PHIÊN**, trừ khi user nói khác — runtime recycle là mất (quy tắc 6d).
- Bản đầy đủ 13 quy tắc: `.claude/rules/quy-tac.md` — đọc trước khi bắt đầu làm việc.

## Trạng thái phiên — nạp cùng file này

@TRANG-THAI.md

## Nạp thêm khi cần

- Cách dùng colab-mcp (kết nối, cell, bootstrap) → skill `colab-mcp`
- Cách dùng HuggingFace (login, tải nhanh, repo trung chuyển) → skill `huggingface`
- Cách lấy dataset và chạy benchmark → skill `benchmark`
- Nhật ký đo đạc chi tiết → `STATUS.md`; bug đã gặp → `docs/03-bug-va-cach-sua.md`
