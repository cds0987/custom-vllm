# 04 — Kết quả cuối và cách dùng

## Cấu hình vô địch hiện tại (champion v2)

**Checkpoint:** khung `RedHatAI/Qwen3.5-9B-quantized.w4a16` + phần GDN ghép từ
`unsloth/Qwen3.5-9B-GGUF:Q4_K_M`, nén int4 nhóm 32.

Tái tạo bằng một lệnh (~5 phút CPU, kết quả tất định):

```bash
python scripts/graft_gguf_gdn.py \
    --frame  <thư mục RedHatAI đã tải, GIỮ NGUYÊN, không chạy fix nào> \
    --gguf   <file Qwen3.5-9B-Q4_K_M.gguf> \
    --out    <thư mục đích> \
    --bits 4 --group-size 32
```

## Lệnh chạy máy chủ (traffic tương tác)

```bash
python scripts/patch_vllm_gdn_quant_load.py     # bắt buộc: mở khóa nạp GDN đã nén

vllm serve <thư mục champion> \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-batched-tokens 1088

python scripts/warmup_prefix.py --prefix-file skills_pack/system_prefix.txt --verify
```

Giải thích từng cờ:

| Cờ | Vì sao |
|---|---|
| `--enable-prefix-caching` | **Bắt buộc.** Nguồn sống của hệ (99,4% dùng lại). Lưu ý: bản này KHÔNG tự bật |
| `--mamba-cache-mode align` | Cần cho kiến trúc lai khi bật cache; đã kiểm đúng từng byte |
| `--kv-cache-dtype fp8_e4m3` | Thắng miễn phí: cuốn sổ tra cứu nhẹ đi một nửa |
| `--max-num-batched-tokens 1088` | Chia nhịp để người mới đến không giẫm chân người đang được phục vụ. **1.088 là sàn cứng** — thấp hơn không khởi động được |
| `warmup_prefix.py` | Trả trước chi phí nguội (10-60 giây) trước khi mở cửa cho khách |

## Lệnh chạy theo lô (tự động hóa, chạy đêm) — nhanh hơn 35%

Không dựng máy chủ; gọi thẳng động cơ trong tiến trình:

```python
from vllm import LLM
llm = LLM(model="<champion>", enable_prefix_caching=True,
          kv_cache_dtype="fp8_e4m3", max_num_batched_tokens=1088,
          max_model_len=32768)
outputs = llm.chat(danh_sach_hoi_thoai)   # đưa cả lô vào một lần
```

Lô **74-100 việc là đủ chạm trần**; gom nhiều hơn chỉ tăng độ trễ, không tăng
sản lượng.

## Bảng năng lực (tất cả là số đo thật, trên bộ 74 câu của bạn)

| Kịch bản | Số người cùng lúc | Chờ chữ đầu | Năng suất |
|---|---|---|---|
| Nền kỹ năng 30.000 chữ | **~30** | 0,2-0,6s (p95 2,75s) | 390 chữ/giây |
| Chạy theo lô | — | — | **520 chữ/giây · ~31.500 việc/đêm** |
| Nền 128.000 chữ | ~6-10 | ~1,0s | thấp hơn (mỗi người ~8-22 chữ/giây) |
| Không dùng nền chung | ~3 yêu cầu/phút | = độ dài tài liệu ÷ 1.400 | — |

**Cam kết dịch vụ khuyến nghị:** 0,3 yêu cầu/giây với p95 dưới 3 giây (0% vi
phạm khi đã dùng cấu hình trên).

## Công tắc theo tải

- **MTP (đoán trước):** bật khi ≤4 người → mỗi người nhanh hơn ~29%. **Tắt khi
  ≥8 người** (chờ chữ đầu nổ lên 4,6-17 giây).
- **Chế độ "turbo"** (bản nén sâu phần GDN, +36% tốc độ nhưng chất lượng sụt
  15%): chỉ dùng cho tác vụ chấp nhận được, mặc định KHÔNG dùng.

## Danh sách kiểm khi triển khai thật

1. ☐ Prefix caching đã bật (kiểm bằng `warmup_prefix.py --verify`)
2. ☐ Có làm nóng trước khi nhận traffic (blue-green: server mới warm xong mới
   chuyển traffic)
3. ☐ Giới hạn tải đặt ở **cổng vào**, không phải ở máy chủ (đã chứng minh chặn
   ở máy chủ làm mọi thứ tệ hơn)
4. ☐ Proxy không gom buffer luồng SSE (lỗi kinh điển biến stream mượt thành
   giật cục)
5. ☐ Checkpoint nằm sẵn trên đĩa máy chủ, không tải từ mạng lúc khởi động
6. ☐ Giám sát: tỷ lệ dùng lại cache, số yêu cầu đang chờ, p95 — cảnh báo khi
   cache tụt (dấu hiệu sớm nhất của mọi sự cố)
7. ☐ Đã chạy thử tải dài nhiều giờ trên máy đích (dự án này chưa làm được vì
   máy ảo bị xóa liên tục)

## Công cụ trong repo

| Script | Dùng để |
|---|---|
| `setup_env.sh` | Dựng lại toàn bộ môi trường bằng một lệnh |
| `graft_gguf_gdn.py` | Tạo checkpoint vô địch |
| `warmup_prefix.py` | Làm nóng trước khi mở cửa |
| `bench_skills.py` | Đo trên bộ câu hỏi thật + nền dùng chung |
| `bench_sla_prefix.py` | Đo cam kết dịch vụ theo nhịp khách đến thật |
| `bench_agent_loop.py` | Đo vòng lặp agent gọi tool (+8 chế độ gây stress) |
| `prepare_agent_workload.py` | Trộn BFCL + câu hỏi thật + SWE-bench thành tải mô phỏng |
| `eval_quality_swebench.py` | Cổng chất lượng — chạy trước mọi lần đổi checkpoint |

➡️ Muốn hiểu tầng dưới (CUDA → vLLM → serving):
[05-hoc-cuda-vllm-serving.md](05-hoc-cuda-vllm-serving.md)
