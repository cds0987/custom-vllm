# Nghiên cứu: Tuyên bố của Unsloth (30x speed, accuracy) và khả năng ứng dụng cho serving Qwen3.5-9B

## 1. Tóm tắt (5 dòng, trả lời thẳng câu hỏi user)

"30x faster" là con số **training** từ blog gốc tháng 12/2023 ("Introducing Unsloth"), so sánh Unsloth **Max** (bản trả phí thời đó, nay không còn tồn tại như một SKU riêng) với HuggingFace Transformers gốc trên **1 GPU Tesla T4**, dataset Alpaca — không liên quan gì tới inference/serving [nguồn: https://unsloth.ai/introducing]. Bản mã nguồn mở (free) chỉ được quảng cáo "2x faster, 50% less memory" trong chính bài đó. "0% loss in accuracy" là tuyên bố riêng, độc lập với con số tốc độ, dựa trên lý do kỹ thuật là Unsloth không dùng phép xấp xỉ (Triton kernel viết lại chính xác toán học của backprop) [nguồn: https://huggingface.co/blog/unsloth-trl]. README GitHub hiện tại (8/2026) không còn nhắc "30x" — con số chủ đạo bây giờ là "2x faster, 70% less VRAM" cho training/RL nói chung và "12x faster" riêng cho MoE training [nguồn: https://github.com/unslothai/unsloth]. Quan trọng nhất cho dự án: repo `unsloth/Qwen3.5-9B-GGUF` **có tồn tại thật** và có file `Qwen3.5-9B-UD-Q4_K_XL.gguf` (dynamic quant) bên cạnh `Qwen3.5-9B-Q4_K_M.gguf` chuẩn — đây là ứng viên đáng thử làm nguồn ghép GDN thay thế [nguồn: https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF]. Không có bằng chứng nào cho thấy kernel Triton fast-dequantize của Unsloth được thiết kế hay tối ưu cho serving batch lớn — nó phục vụ forward+backward của LoRA/QLoRA training, khác hẳn mục đích của Marlin (GEMM W4A16 tối ưu riêng cho decode batch nhỏ-vừa trong serving).

## 2. Bảng tuyên bố → đo cái gì → điều kiện → nguồn

| Tuyên bố | Đo gì | Baseline so sánh | Điều kiện | Nguồn URL |
|---|---|---|---|---|
| "30x faster. Alpaca takes 3 hours instead of 85." | Training (fine-tune) | HuggingFace Transformers gốc | Unsloth **Max** (tier trả phí, blog 12/2023), GPU không nêu rõ trong câu này | https://unsloth.ai/introducing |
| "Huggingface's original implementation takes...23 hours and 15m, whilst our Max offering takes 2 hours and 34m, which is 8.8x faster" | Training | HF | Max tier, 1x Tesla T4, dataset Alpaca | https://unsloth.ai/introducing |
| "LAION's Chip2 dataset takes 164 hours whilst ours takes 5 hours (31x faster)" | Training | HF | Max tier, 2x Tesla T4 (DDP) | https://unsloth.ai/introducing |
| "SlimOrca, 1301 hours...is slashed to 54 hours or 24x faster" | Training | HF | Max tier, 2x Tesla T4 (DDP) | https://unsloth.ai/introducing |
| "free open source version makes finetuning 2x faster" + "50% less memory" | Training | HF (ngầm định) | Bản OSS/free, không nêu GPU/model cụ thể trong câu này | https://unsloth.ai/introducing |
| "peak memory usage is slashed to 6.9GB from 16.7GB (59% less)" | VRAM training | HF | Max tier, 1x A10 bfloat16, dataset Open Assistant | https://unsloth.ai/introducing |
| "0% loss in accuracy or +20% increased accuracy with our Max offering" | Accuracy | QLoRA chuẩn | Max tier (blog 2023) | https://unsloth.ai/introducing |
| "0% loss in accuracy - no approximation methods - all exact" | Accuracy | QLoRA/HF chuẩn | Áp dụng chung, lý do: kernel Triton tính chính xác toán học, không xấp xỉ | https://huggingface.co/blog/unsloth-trl |
| "Train and RL 500+ models up to 2x faster with 70% less VRAM; MoE up to 12x faster" | Training/RL | Không nêu rõ baseline cụ thể trong câu | README hiện tại (8/2026), chung cho mọi model | https://github.com/unslothai/unsloth |
| "New RoPE & MLP Triton Kernels & Padding Free + Packing: 3x faster training & 30% less VRAM" | Training | Bản Unsloth trước khi có các kernel này | Kernel RoPE/MLP mới | https://unsloth.ai/docs/blog/3x-faster-training-packing |
| "Get up to 50% more accurate tool-calling with self-healing tool calls" | Tool-calling accuracy (không phải quant accuracy) | Không nêu baseline rõ | Trang chủ hiện tại | https://unsloth.ai/ |

**Kết luận về "30% + 30x" mà user có thể nhầm**: Không tìm thấy một cặp con số "30x" và "30%" nào được Unsloth công bố song song trong cùng ngữ cảnh. Các con số phần trăm liên quan gần nhất là: "30% less VRAM" (RoPE/MLP Triton kernel, training) và "20% increased accuracy" (Max tier, blog 2023) — đây là hai tuyên bố tách biệt, không phải một con số kép. Rất có thể user đang trộn "30x faster" (2023, training, Max tier) với "0% loss in accuracy" (tuyên bố accuracy riêng biệt, áp dụng cho QLoRA training nói chung) thành một câu duy nhất.

## 3. Đánh giá độc lập / phê bình

- **Hacker News thread (12/2023)** trên bài "80% faster, 50% less memory, 0% loss of accuracy" — không lấy được nội dung comment cụ thể do WebFetch bị rate-limit (HTTP 429) khi truy cập https://news.ycombinator.com/item?id=38492652. **Không xác minh được** nội dung chi tiết các bình luận phê bình, chỉ xác nhận được thread tồn tại.

- **Paper "Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth"** (Arjun S. Nair, arXiv 2601.02609, công bố 6/1/2026) — đây là phê bình định lượng, độc lập, mạnh nhất tìm được:
  - Full fine-tune: framework của họ đạt 41.184 tokens/s so với Unsloth 11.736 tokens/s → 3,51x nhanh hơn Unsloth (không phải HF gốc). Đo trên GPU A100-40GB, model Qwen2.5-0.5B [nguồn: https://arxiv.org/abs/2601.02609].
  - LoRA rank 32: 11.699 vs 2.857 tokens/s → 4,10x [nguồn: https://arxiv.org/abs/2601.02609].
  - **Phát hiện then chốt về phương pháp benchmark**: "Unsloth's reported 46,000 tokens/second benchmark exhibited zero gradient norms--the model was not training." Nghĩa là một số con số throughput cao mà Unsloth từng công bố được đo trên một cấu hình mà mô hình thực chất KHÔNG học (gradient norm bằng 0) — một cảnh báo nghiêm túc về việc verify benchmark trước khi tin số liệu [nguồn: https://arxiv.org/abs/2601.02609].

- **GitHub issue #1176** (`unslothai/unsloth`) — user đặt câu hỏi liệu tốc độ/memory nhanh hơn của "full finetune" trong Unsloth có đánh đổi bằng độ chính xác kém hơn không, so sánh với HF trên Llama 3.2 1B. Ghi nhận sơ bộ "hf has a lower loss" (HF có loss thấp hơn) ở early training, nhưng issue **không được giải quyết dứt khoát** (còn mở, chưa có kết luận cuối) [nguồn: https://github.com/unslothai/unsloth/issues/1176].

- Không tìm thấy thảo luận cụ thể trên Reddit r/LocalLLaMA phê bình baseline yếu (chỉ so với FA2 chưa optimize, batch size nhỏ) qua tìm kiếm — **không xác minh được** mảng này bằng nguồn cụ thể, dù về mặt logic các con số 2023 (so với FA2/HF baseline không rõ có được optimize tối đa hay không, GPU T4 cũ) là dấu hiệu cảnh báo hợp lý mà không có nguồn phê bình trực tiếp xác nhận.

## 4. Phần chuyển giao được cho SERVING

### 4a. Kernel fast-dequantize Triton vs Marlin

Kernel `fast_dequantize()` trong `unsloth/kernels/fast_lora.py` (và các file kernel liên quan) được dùng **cả forward lẫn backward pass** của LoRA/QLoRA: ví dụ dòng `QW = fast_dequantize(QW.t(), QW_quant)` xuất hiện trong `LoRA_QKV.backward()` và `LoRA_W.backward()` [nguồn: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py]. Các autograd function (`LoRA_MLP`, `LoRA_QKV`, `LoRA_W`) đều triển khai backward pass đầy đủ với `@torch_amp_custom_bwd`, và có comment kỹ thuật riêng cho hạn chế của bitsandbytes/TorchInductor khi xử lý tensor 3D trong backward — tức là thiết kế nhắm vào **training**, không phải phục vụ decode-serving batch lớn tùy ý.

So sánh với **Marlin** (kernel GEMM W4A16 dùng trong vLLM, do Neural Magic/Red Hat phát triển): Marlin được thiết kế chuyên cho **inference decode ở batch nhỏ-vừa** — W4A16 có throughput lý thuyết cao hơn khi batch size < 78 vì GEMM trong LLM bị memory-bound ở batch nhỏ, và footprint bộ nhớ nhỏ hơn của W4A16 giúp hiệu năng tốt hơn trong vùng này [nguồn: https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/]. Ở batch lớn (vd batch=64), kernel dạng mixed-precision như Marlin có thể **giảm hiệu năng tới 20,3%** so với kernel FP16×FP16 thuần — nghĩa là ngay cả Marlin cũng có vùng hoạt động tối ưu riêng (batch nhỏ/vừa), khác hẳn tối ưu cho throughput training [nguồn tìm qua WebSearch, trích dẫn không có URL bài viết cụ thể — **không xác minh được nguồn gốc chính xác của con số 20,3% này**, cần kiểm tra lại nếu dùng để quyết định kỹ thuật].

**Kết luận 4a**: Kernel Triton fast-dequantize của Unsloth **không phải** lựa chọn phù hợp cho serving — nó được viết, test và tối ưu cho training loop (forward+backward, LoRA adapter), không có bằng chứng nào về việc được benchmark hay khuyến nghị cho inference batch lớn. Marlin (đã dùng trong vLLM của dự án) đúng mục đích serving hơn nhiều.

### 4b. Dynamic quants (UD-*) — cơ chế, bằng chứng chất lượng, và danh sách file thật

**Cơ chế**: Theo blog "Unsloth Dynamic v2.0 GGUFs", thay vì chỉ chỉnh một tập con layer như bản Dynamic 1.0 (chỉ hiệu quả cho MoE), Dynamic 2.0 "dynamically adjust[s] the quantization type of every possible layer, and the combinations will differ for each layer and model" — tức là phân tích độ nhạy của TỪNG layer rồi chọn bit-depth phù hợp riêng cho layer đó, áp dụng được cho mọi kiến trúc (kể cả MoE) [nguồn: https://unsloth.ai/blog/dynamic-v2]. Tài liệu Qwen3.5 xác nhận cơ chế cụ thể: "4-bit has important layers upcasted to 8 or 16-bit" [nguồn: https://unsloth.ai/docs/models/qwen3.5].

**Bằng chứng chất lượng (đo trên Gemma 3 12B, không phải Qwen3.5)**:
- 5-shot MMLU: Dynamic 2.0 Q2_K_XL đạt 67,07% so với baseline BF16 67,15% — chênh lệch chỉ 0,08 điểm phần trăm [nguồn: https://unsloth.ai/blog/dynamic-v2].
- KL Divergence (so với baseline GGUF quant thường, thấp hơn = tốt hơn):
  - Q2_K_XL: 0,2297 → 0,2209 (giảm ~3,8%)
  - Q3_K_XL: 0,0878 → 0,0806 (giảm ~8,2%)
  - Q4_K_XL: 0,0249 → 0,0237 (giảm ~4,8%)
  [nguồn: https://unsloth.ai/blog/dynamic-v2]

Lưu ý: các số liệu này đo trên **Gemma 3 12B**, không phải Qwen3.5-9B — **không xác minh được** con số tương đương cho riêng Qwen3.5-9B qua các nguồn đã truy cập.

**Danh sách file thật trong `unsloth/Qwen3.5-9B-GGUF`** (lấy từ API chính thức https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF, truy cập 2026-08-11):
```
Qwen3.5-9B-BF16.gguf
Qwen3.5-9B-IQ4_NL.gguf
Qwen3.5-9B-IQ4_XS.gguf
Qwen3.5-9B-Q3_K_M.gguf
Qwen3.5-9B-Q3_K_S.gguf
Qwen3.5-9B-Q4_0.gguf
Qwen3.5-9B-Q4_1.gguf
Qwen3.5-9B-Q4_K_M.gguf   <-- nguồn hiện tại dự án đang dùng để ghép GDN
Qwen3.5-9B-Q4_K_S.gguf
Qwen3.5-9B-Q5_K_M.gguf
Qwen3.5-9B-Q5_K_S.gguf
Qwen3.5-9B-Q6_K.gguf
Qwen3.5-9B-Q8_0.gguf
Qwen3.5-9B-UD-IQ2_M.gguf
Qwen3.5-9B-UD-IQ2_XXS.gguf
Qwen3.5-9B-UD-IQ3_XXS.gguf
Qwen3.5-9B-UD-Q2_K_XL.gguf
Qwen3.5-9B-UD-Q3_K_XL.gguf
Qwen3.5-9B-UD-Q4_K_XL.gguf   <-- ứng viên thay thế, "dynamic quant" tương đương group Q4
Qwen3.5-9B-UD-Q5_K_XL.gguf
Qwen3.5-9B-UD-Q6_K_XL.gguf
Qwen3.5-9B-UD-Q8_K_XL.gguf
imatrix_unsloth.gguf_file
mmproj-BF16.gguf / mmproj-F16.gguf / mmproj-F32.gguf (multimodal projector, không liên quan)
```
[nguồn: https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF, xác nhận chéo qua https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main]

Tài liệu chính thức của Unsloth khuyến nghị: "For Qwen3.5-9B, ... UD-Q4_K_XL" là lựa chọn chính, và "UD-Q2_K_XL" cho trường hợp cần tiết kiệm dung lượng [nguồn: https://unsloth.ai/docs/models/qwen3.5]. Riêng model này cũng có repo phụ `unsloth/Qwen3.5-9B-MTP-GGUF` (multi-token prediction variant) — chưa kiểm tra danh sách file của repo này, ngoài phạm vi câu hỏi.

### 4c. Tuyên bố chất lượng GGUF unsloth vs GGUF thường

Unsloth so sánh Dynamic 2.0 GGUF của họ với "baseline" GGUF/BF16 chuẩn qua KL-divergence và MMLU như mục 4b, và khẳng định phương pháp "set new benchmarks on 5-shot MMLU and KL Divergence" so với các phương pháp quant khác [nguồn: https://x.com/UnslothAI/status/1915476692786962441, xác nhận lại qua https://unsloth.ai/blog/dynamic-v2]. Tuy nhiên, đây là số liệu do chính Unsloth tự đo và công bố (không phải bên thứ ba độc lập kiểm chứng lại) — **không tìm thấy** một bài đánh giá độc lập (bên thứ ba) nào so sánh trực tiếp GGUF chuẩn llama.cpp K-quant vs UD-* của Unsloth trên Qwen3.5. Vì vậy nên coi các con số KL-div/MMLU ở trên là "self-reported, đáng tham khảo nhưng chưa được kiểm chứng độc lập".

### 4d. Bug fix liên quan Qwen3.5/Qwen

- Tài liệu Qwen3.5 của Unsloth ghi rõ: "Tool-calling improved following our chat template fixes. Fix is universal and applies to any Qwen3.5 format and any uploader" — tức là bug về chat template ảnh hưởng tool-calling, và fix này áp dụng cho MỌI phiên bản GGUF của Qwen3.5 (không riêng gì của Unsloth) [nguồn: https://unsloth.ai/docs/models/qwen3.5].
- Bug tương tự đã từng xảy ra với **Qwen3** (dòng trước): "A bug was found in Qwen's chat template, and Unsloth updated all the Qwen3 GGUFs and safetensors with the fixed chat template" [nguồn: https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/14].
- GitHub issue mở: "[Bug] Qwen3-Instruct model does not load chat_template properly" [nguồn: https://github.com/unslothai/unsloth/issues/3552] — chưa xác minh được trạng thái hiện tại (đã đóng hay chưa).
- HF discussion: "unsloth/Qwen3.5-35B-A3B-GGUF · LM Studio does not support the newly updated chat template" — cho thấy các fix chat template của Unsloth đôi khi gây vấn đề tương thích ngược với tool bên thứ ba (LM Studio) [nguồn: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/18].
- Bug fix cho **Qwen 2.5** (không phải Qwen3.5 nhưng cùng họ Qwen): pad_token không được là `<|endoftext|>` (gây generate vô hạn khi fine-tune), và `<|im_start|>`/`<|im_end|>` chưa được train trên base model nên không nên dùng cho chat template khi infer trên base model [nguồn: https://unsloth.ai/blog/qwen-coder].
- Đây đều là các fix liên quan **chat template / tokenizer**, không liên quan tới độ chính xác số học của quantization hay GDN — quan trọng để lưu ý khi dự án dùng tokenizer/chat template nhưng không trực tiếp ảnh hưởng tới việc ghép trọng số GDN.

## 5. Khuyến nghị hành động cho dự án

1. **Đáng thử**: Lấy `Qwen3.5-9B-UD-Q4_K_XL.gguf` từ `unsloth/Qwen3.5-9B-GGUF` làm nguồn GDN thay thế cho `Q4_K_M`, rồi lặp lại quy trình graft + đo perplexity, giữ nguyên phần khung RedHatAI W4A16. Lý do: UD-Q4_K_XL cùng "họ" Q4 (kích thước tương đương) nhưng có cơ chế giữ layer nhạy ở bit cao hơn — về lý thuyết CÓ THỂ cho GDN weights chất lượng tốt hơn nếu các layer GDN nằm trong nhóm được Unsloth đánh giá là nhạy cảm. Đây là thử nghiệm rẻ (chỉ tải thêm 1 file GGUF, chạy lại pipeline ghép có sẵn), rủi ro thấp, đáng làm.
2. **Không nên tin ngay số liệu KL-div/MMLU của Unsloth (mục 4b) là đại diện cho case cụ thể của dự án** — số liệu đó đo trên Gemma 3 12B, không phải Qwen3.5-9B, và là self-reported. Phải tự đo perplexity/benchmark sau khi ghép UD-Q4_K_XL, giống cách đã làm với Q4_K_M, để có con số thật cho project.
3. **Không đáng đầu tư thời gian**: kernel Triton `fast_dequantize` của Unsloth cho mục đích tăng tốc serving/inference trong vLLM — nó được thiết kế và benchmark cho training LoRA/QLoRA (forward+backward), không có bằng chứng tối ưu cho batch-serving. Marlin (đã tích hợp sẵn trong vLLM) đúng use-case hơn nhiều; không cần khảo sát thêm hướng này.
4. **Không đáng tin tuyệt đối vào các con số "Nx faster" quảng cáo của Unsloth cho quyết định kỹ thuật** — các con số 2023 (30x, 8.8x, 31x...) đo trên GPU T4 cũ, bản Max/Pro không còn tồn tại nguyên dạng, và ít nhất một nghiên cứu độc lập (Chronicals, arXiv 2601.02609) phát hiện một số benchmark tốc độ cao của Unsloth từng có gradient-norm bằng 0 (model không thực sự học) — dấu hiệu cần thận trọng khi dùng số liệu marketing của Unsloth làm căn cứ.
5. **Cân nhắc phụ**: kiểm tra repo `unsloth/Qwen3.5-9B-MTP-GGUF` (multi-token-prediction variant) xem có ích cho pipeline serving không — ngoài phạm vi nghiên cứu này, chưa xem chi tiết.
6. Bug fix chat-template của Unsloth cho Qwen3.5 (mục 4d) đáng lưu ý nếu dự án dùng tokenizer/chat template convert từ checkpoint unsloth — đảm bảo dùng đúng bản đã fix, tránh vấn đề tool-calling.

## Phụ lục: tất cả nguồn đã dùng

- https://unsloth.ai/introducing (blog gốc "Introducing Unsloth: 30x faster LLM training", 12/2023)
- https://github.com/unslothai/unsloth (README chính, truy cập 2026-08-11)
- https://unsloth.ai/ (trang chủ, truy cập 2026-08-11)
- https://unsloth.ai/docs/blog/3x-faster-training-packing
- https://huggingface.co/blog/unsloth-trl ("Make LLM Fine-tuning 2x faster with Unsloth and TRL", HuggingFace blog)
- https://unsloth.ai/blog/dynamic-v2 ("Unsloth Dynamic v2.0 GGUFs")
- https://x.com/UnslothAI/status/1915476692786962441 (tweet giới thiệu Dynamic v2.0)
- https://huggingface.co/posts/danielhanchen/766411311755038
- https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- https://unsloth.ai/docs/models/qwen3.5 ("Qwen3.5 - How to Run Locally")
- https://huggingface.co/api/models/unsloth/Qwen3.5-9B-GGUF (API danh sách file thật, truy cập 2026-08-11)
- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main (xác nhận chéo)
- https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF (repo phụ, chưa khảo sát sâu)
- https://arxiv.org/abs/2601.02609 / https://arxiv.org/pdf/2601.02609 (paper "Chronicals", Arjun S. Nair, 6/1/2026 — phê bình định lượng độc lập)
- https://github.com/unslothai/unsloth/issues/1176 (thảo luận về full-finetune accuracy, chưa có kết luận)
- https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py (mã nguồn kernel fast_dequantize)
- https://github.com/unslothai/unsloth/issues/3552 (bug chat_template Qwen3-Instruct)
- https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/14 (fix chat template Qwen3)
- https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/18 (LM Studio không tương thích chat template mới)
- https://unsloth.ai/blog/qwen-coder ("Qwen 2.5 Coder Fine-tuning with Unsloth" — bug pad_token)
- https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/ (bối cảnh Marlin/W4A16 batch behavior)
- https://news.ycombinator.com/item?id=38492652 (HN thread — KHÔNG lấy được nội dung do rate limit, chỉ xác nhận tồn tại)

**Các mục "không xác minh được"** cần lưu ý riêng:
- Nội dung chi tiết bình luận Hacker News (429 rate limit).
- Con số "20,3% performance degradation" của Marlin ở batch=64 — tìm thấy qua tóm tắt search, chưa truy xuất được URL bài viết gốc cụ thể để trích dẫn chính xác.
- Số liệu KL-div/MMLU của Dynamic 2.0 cho riêng Qwen3.5-9B (chỉ có số liệu cho Gemma 3 12B).
- Đánh giá độc lập bên thứ ba so sánh trực tiếp UD-* GGUF vs K-quant chuẩn trên Qwen3.5 — không tìm thấy.
