# Báo cáo dự án — từ đầu đến nay

> **Đọc file này nếu bạn mới tham gia.** Nó kể lại toàn bộ dự án theo thứ tự
> logic, giả định bạn chưa biết gì về GPU, CUDA hay mô hình ngôn ngữ.
> Đọc hết mất khoảng 25 phút. Cập nhật: 12/08/2026.

---

## Phần 1 — Dự án này là gì

### 1.1. Bài toán

Chạy một mô hình AI **9 tỷ tham số** (Qwen3.5-9B) trên **một card đồ họa L4
giá rẻ** (loại thuê được trên Colab), sao cho:

- phục vụ được càng nhiều người cùng lúc càng tốt,
- mà mô hình **không trả lời kém đi**,
- và mọi thứ **tái tạo được** trên máy khác.

### 1.2. Vì sao khó

Ba ràng buộc chồng lên nhau:

| Ràng buộc | Hệ quả |
|---|---|
| Card L4 chỉ có **23GB** bộ nhớ và bị **giới hạn điện 72W** | Mô hình gốc không vừa; sức tính thực chỉ bằng ~45% con số quảng cáo |
| Qwen3.5 dùng kiến trúc **lai (hybrid GDN)** quá mới | Phần mềm phục vụ chưa hỗ trợ đầy đủ → phải tự vá ~20 chỗ |
| Máy ảo Colab **bị xóa trắng bất cứ lúc nào** | Đã mất 9 lần; mọi thứ buộc phải dựng lại được bằng một lệnh |

### 1.3. Cách làm việc

Viết mã ở máy cá nhân → đẩy lên GitHub → máy ảo Colab kéo về và chạy. Mọi kết
quả ghi vào `STATUS.md` ngay sau khi đo, không để dồn — đây là lý do 9 lần mất
máy ảo không làm mất một kết quả nào.

---

## Phần 2 — Nền tảng tối thiểu cần hiểu

Bốn khái niệm này đủ để đọc phần còn lại.

### 2.1. Mô hình là một đống số, và vấn đề là *khuân vác*

Mô hình 9 tỷ tham số nghĩa là 9 tỷ con số. Mỗi khi sinh ra **một chữ**, máy phải
kéo gần như toàn bộ đống số đó từ bộ nhớ ra. Không phải tính toán khó — mà là
vận chuyển nặng.

> **Ví von dùng xuyên suốt:** GPU là một đầu bếp cực nhanh nhưng kho nguyên liệu
> ở xa. Đầu bếp không bao giờ là nút thắt — **đường vận chuyển mới là nút thắt.**

### 2.2. Hai giai đoạn ngược nhau: ĐỌC và VIẾT

| | **Đọc câu hỏi** (prefill) | **Viết câu trả lời** (decode) |
|---|---|---|
| Làm gì | Nhai cả câu hỏi một lượt | Sinh từng chữ, lần lượt |
| Nghẽn ở đâu | **Sức tính toán** | **Băng thông bộ nhớ** |
| Thuốc chữa | Tính nhanh hơn | Khuân ít byte hơn |

Nhầm lẫn hai giai đoạn này là nguồn gốc của nhiều sai lầm trong dự án — kể cả
một sai lầm lớn ở mục 4.3.

### 2.3. Nén mô hình (quantization)

Mỗi con số vốn ghi bằng 16 bit; nén xuống 4 bit là ghi "3,14" thay vì
"3,14159265". File nhẹ đi 4 lần → khuân nhanh hơn 4 lần. Cái giá: nén ẩu thì mô
hình "lú" đi. **Cả dự án xoay quanh câu hỏi: nén sâu tới đâu mà chất lượng không
sứt mẻ?**

Quan trọng: cùng 4 bit vẫn có nhiều *cách đóng gói*, và chúng chênh nhau **gấp
đôi** về tốc độ (mục 4.2).

### 2.4. Bộ nhớ đệm — mánh quan trọng nhất

Hai thứ khác nhau, đừng lẫn:

- **KV cache (cuốn sổ tra cứu):** khi viết chữ thứ 100, mô hình cần nhớ 99 chữ
  trước. Nó ghi sẵn một bảng ghi chú. Hội thoại càng dài, sổ càng dày, mỗi chữ
  mới càng tốn công tra, và sổ chiếm bộ nhớ → giới hạn số người phục vụ.
- **Prefix cache (dùng chung phần đầu):** nếu mọi người đều bắt đầu bằng cùng
  một đoạn văn bản (bộ quy tắc/kỹ năng ~30.000 chữ), ta đọc **một lần rồi dùng
  chung** thay vì đọc lại cho từng người.

Prefix cache là đòn lớn nhất của dự án: **chờ chữ đầu từ 10,5 giây xuống 0,2
giây.**

---

## Phần 3 — Hành trình, theo đúng thứ tự nhân quả

Mỗi vòng sinh ra từ câu hỏi mà vòng trước để lại.

### Vòng 0 — Làm cho nó chạy được

Mô hình quá mới; phần mềm phục vụ chưa đăng ký kiến trúc này, cấu hình sai khớp,
tên tham số lệch. Phải viết **~20 miếng vá** mới nạp nổi.

Quyết định quan trọng: gói tất cả vào một script dựng môi trường. **Quyết định
này về sau cứu dự án 9 lần.**

### Vòng 1 — Nén tới đâu thì gãy?

Chạy thang bit từ cao xuống thấp, mỗi mức đo *cả tốc độ lẫn chất lượng*, chất
lượng chấm bằng bộ đề thật chứ không phải vài câu mẫu.

Hai điều bất ngờ: dưới 4 bit **mất chất lượng** (mô hình nói lảm nhảm không biết
dừng) **và mất cả tốc độ** (274 → 236 → 219 → 167 tok/s) — vì cách đóng gói bit
thấp rắc rối hơn, bóc hộp tốn công hơn.

➡️ **4 bit là sàn.** Muốn nhanh hơn phải tìm đường khác.

### Vòng 2 — Cùng 4 bit, đóng gói kiểu khác

| Định dạng (đều 4 bit) | Tốc độ | Chất lượng |
|---|---|---|
| GGUF Q4_K_M (sau khi vắt kiệt) | 274 | đạt |
| **W4A16 + kernel Marlin** | **563** | **tốt nhất** |
| AWQ tự nén | 616 | ✗ rớt |

Chênh gấp đôi vì GGUF đóng gói kiểu "hộp trong hộp" phải bóc lúc dùng, còn W4A16
xếp sẵn đúng khuôn GPU cần.

### Vòng 3 — Trần vật lý ở đâu?

Card ghi 121 TFLOPS, nhưng đo thật chỉ đạt **~55** — vì bị **giới hạn điện 72W**
nên xung nhịp chỉ chạy ~50%. Đây là trần cứng, không sửa được (đã xác minh:
Colab không cho quyền chỉnh giới hạn điện).

➡️ Nếu không thể tính nhanh hơn, thì phải **làm ít việc đi**.

### Vòng 4 — Đừng đọc lại thứ đã đọc

Bật prefix cache cho kịch bản "nền kỹ năng 30.000 chữ dùng chung".

**Cổng kiểm tra đầu tiên là tính đúng đắn, không phải tốc độ:** cùng câu hỏi, so
câu trả lời khi có cache và khi không — phải **giống nhau từng byte**. Đạt.

Kết quả: chờ chữ đầu **10,5s → 0,2-1,4s**, tỷ lệ dùng lại **99,4%**.

### Vòng 5 — Vì sao nén phần GDN làm hỏng chất lượng?

Checkpoint tốt nhất thế giới chỉ nén 25% mô hình, để nguyên 75% phần GDN. Nén
nốt có nhanh hơn không?

- **Thử 1:** tự nén bằng GPTQ → nhanh hơn 36% nhưng **chất lượng rớt 15%**.
- **Thử 2:** nén nhẹ tay hơn (8 bit) → chất lượng **y hệt**, mà còn chậm hơn.
  ⟹ Giả thuyết "nén sâu quá" **sai**; thủ phạm là *cách GPTQ ước lượng độ quan
  trọng* trên phần GDN.
- **Thử 3 — nước cờ quyết định:** nếu GPTQ hỏng ở đó, hãy **mượn phần GDN từ bản
  GGUF** (llama.cpp nén theo kiểu khác), ghép vào khung W4A16.

| | Tốc độ | Chất lượng (thấp = tốt) |
|---|---|---|
| Champion cũ | 563 | 5,132 |
| **Ghép int4 = champion hiện tại** | **~611** | **4,778 — tốt hơn cả bản gốc** |

➡️ Bài học: **khi một phương pháp hỏng ở một chỗ cụ thể, hãy thay đúng chỗ đó
thay vì bỏ cả phương pháp.**

*(Chi tiết phản trực giác: bản int4 có sai số kỹ thuật cao gấp 17 lần bản int8
nhưng chất lượng lại tốt hơn — vì nó đồng nhất kiểu nén với phần còn lại của
khung. **Sai số thô không phải chỉ báo tốt cho chất lượng.**)*

### Vòng 6 — Cam kết dịch vụ thật là bao nhiêu?

Đo kiểu "đổ cả rổ 32 yêu cầu cùng lúc" cho **1,3 yêu cầu/giây**. Đo lại theo kiểu
khách đến rải rác như thực tế: **0,2** — lạc quan gấp 6 lần.

Truy nguyên bằng số liệu hệ thống: hàng đợi rỗng, bộ nhớ mới dùng 37% → thủ phạm
là **người mới đến chen phần "đọc" vào giữa các nhịp "viết" của người đang được
phục vụ**. Thuốc: giới hạn mỗi nhịp chỉ nhai tối đa 1.088 chữ → **0,3 yêu
cầu/giây, 0% vi phạm**.

Một thuốc **thất bại**: chặn bớt khách ở phía máy chủ → tệ hơn hẳn (đuôi chờ nổ
lên 64 giây). ➡️ **Kiểm soát tải phải làm ở cổng vào, không phải ở máy chủ.**

### Vòng 7 — Ba loại workload, ba cấu hình

**(a) Xử lý theo lô (chạy đêm):** gọi thẳng động cơ trong tiến trình, bỏ HTTP →
**+34,8%** (522 tok/s ≈ **31.500 việc/đêm**). Batch 74-100 việc là đủ chạm trần;
gom nhiều hơn chỉ tăng độ trễ.

**(b) Vòng lặp agent gọi tool** (đọc yêu cầu → gọi công cụ → đọc kết quả → …):

- Cache **sống dai**: phiên nằm im chờ công cụ tới **60 giây** vẫn được nhớ
  nguyên vẹn, nối lại gần như miễn phí.
- Kiến trúc nền-dùng-chung **được xác nhận bằng số học** (bộ nhớ tăng dưới tuyến
  tính đúng như mô hình lý thuyết, khớp 4/4 điểm đo).
- **Điểm vận hành là 8 phiên đồng thời** (328 việc/giờ), không phải 1 phiên như
  chỉ tiêu "chờ dưới 3 giây" gợi ý — vì người dùng agent vốn đã chờ công cụ chạy
  sau mỗi lượt. 16 phiên là **lỗ vốn rõ ràng** (ít việc hơn 42%, mỗi việc lâu gấp
  3,5 lần).
- **Luật cấu hình riêng cho agent:** prompt **lớn dần mỗi lượt**, nên mức trần
  ngữ cảnh phải tính `nền + số lượt × cỡ kết quả công cụ + biên`. Đặt thiếu là
  vỡ giữa chừng.

**(c) fp8 — công cụ chuyên dụng:** thắng **+35% ở khâu đọc**, nhưng thua ở khâu
viết, thua sức chứa (−44%), và **thua chất lượng** (rơi vùng cảnh báo). ⟹ Không
phong ngôi; dùng riêng cho việc nuốt tài liệu lạ.

---

## Phần 4 — Kết quả hiện tại

### 4.1. Champion và cách tái tạo

Khung `RedHatAI/Qwen3.5-9B-quantized.w4a16` + phần GDN bóc từ
`unsloth/Qwen3.5-9B-GGUF:Q4_K_M`, nén lại int4 nhóm 32. **Một lệnh, 5 phút CPU,
tất định:**

```bash
python scripts/graft_gguf_gdn.py \
    --frame <thư mục RedHatAI, GIỮ NGUYÊN, không chạy fix nào> \
    --gguf  <file Qwen3.5-9B-Q4_K_M.gguf> \
    --out   <thư mục đích> --bits 4 --group-size 32
```

### 4.2. Lệnh phục vụ và ý nghĩa từng tham số

```bash
python scripts/patch_vllm_gdn_quant_load.py      # mở khoá nạp GDN đã nén

vllm serve <champion> \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8_e4m3 \
    --max-num-batched-tokens 1088

python scripts/warmup_prefix.py --prefix-file <nền> --verify
```

| Tham số | Vì sao |
|---|---|
| `--enable-prefix-caching` | **Bắt buộc** — nguồn sống của hệ; bản này KHÔNG tự bật |
| `--mamba-cache-mode align` | Cần cho kiến trúc lai khi bật cache; đã kiểm đúng từng byte |
| `--kv-cache-dtype fp8_e4m3` | Thắng miễn phí: cuốn sổ tra cứu nhẹ đi một nửa |
| `--max-num-batched-tokens 1088` | Chia nhịp để người mới không giẫm chân người cũ. **1.088 là sàn cứng** |
| `warmup_prefix.py` | Trả trước chi phí nguội trước khi mở cửa cho khách |

⚠️ **Mức trần ngữ cảnh ăn thẳng vào số phiên phục vụ được:**
`số phiên ≈ tổng bộ nhớ KV ÷ mức trần khai báo`. Khai rộng rãi "cho chắc" là tự
vứt bỏ dung lượng. Đặt đúng nhu cầu thật.

### 4.3. Bảng năng lực (số đo thật, trên dữ liệu thật)

| Kịch bản | Sức chứa | Chờ chữ đầu | Năng suất |
|---|---|---|---|
| Chat, nền 30K | ~30 người | 0,2-0,6s | 390 tok/s |
| Xử lý theo lô | — | — | **520 tok/s ≈ 31.500 việc/đêm** |
| Vòng lặp agent | **8 phiên** | — | **328 việc/giờ** |
| Nền 128K | 6-10 người | ~1,0s | thấp hơn |

---

## Phần 5 — Bài học phương pháp (phần đáng giá nhất)

Dự án này có **6 lần một kết luận sụp đổ khi được đo lại tử tế**. Mẫu hình quá
rõ để bỏ qua:

| Bẫy | Biểu hiện |
|---|---|
| Cache thổi phồng số | Bài đo lặp lại câu cũ → máy trả lời "nhanh giả" |
| Đổ-cả-rổ vs khách-đến-rải-rác | Lệch **6 lần** |
| Bỏ bước làm nóng | Mức đo đầu tiên gánh hết chi phí nguội → kết luận đảo ngược |
| Dùng mốc cũ khác phiên bản | So số của hai thế giới khác nhau |
| Nền tổng hợp sai cỡ (1K thay vì 30K) | **Đo nhầm hẳn kịch bản** |
| Các mức tải chồng pha | Một con số 97% gây hoảng, hoá ra là ảo |

Và ba sai lầm suy luận, đều của người điều phối:

1. **Tuyên bố "trần đọc là vật lý"** dựa trên một con số nền **sai gấp đôi** —
   suýt đóng vĩnh viễn một hướng cho **+35%**.
2. **Suy "fp8 thua ở khâu viết ⇒ thua luôn ở khâu đọc"** — không phân biệt hai
   chế độ nghẽn khác nhau (mục 2.2).
3. **Tin một smoke test 2 câu** làm bằng chứng chất lượng — suýt phong ngôi cho
   cấu hình sụt chất lượng 14%.

➡️ Hai luật vận hành ra đời từ đó, và cả hai đã sinh lời ngay lần đầu áp dụng:

> **Luật 1 — Đo trước khi tin.** Kết quả nào *quá đẹp* hoặc *phản trực giác* →
> nghi ngờ phép đo trước, nghi ngờ thế giới sau.
>
> **Luật 2 — Tranh luận trước khi đóng.** Không ai được tự tuyên bố "hết cách";
> phải có một bên *tìm đường* và một bên *phản biện bằng chứng cứ*, rồi mới kết.

---

## Phần 6 — Còn dang dở

| Việc | Kỳ vọng |
|---|---|
| Quét mức trần ngữ cảnh | **+1,6× số phiên miễn phí** |
| Ablation nguồn ghép (bf16 / Q4_K_M / dynamic-quant) | Có cơ ra **champion v3** |
| Truy thủ phạm khiến chuyển đổi GGUF toàn phần ra rác | Mở khoá "quăng file GGUF nào vào cũng chạy nhanh" — giá trị cho cả hệ sinh thái |
| Stress test + tràn ngữ cảnh | Tìm kịch bản phá cam kết dịch vụ |
| CPU có phải nút thắt không | Quyết định cấu hình máy production |

**Nợ kỹ thuật đã biết, cần nói thẳng:**

1. **Cổng kiểm chất lượng có điểm mù** — chỉ kiểm *một request tuần tự*, nên
   không thể bắt lỗi chỉ xuất hiện khi nhiều người dùng cùng lúc.
2. **Chưa từng chạy tải dài nhiều giờ** — bài đo dài nhất ~10 phút (do máy ảo bị
   xoá liên tục). Chưa biết hệ có rò bộ nhớ sau 8 tiếng không.
3. **Chưa từng đo với mạng thật** — mọi số đều qua localhost.
4. **Chưa biết CPU có phải nút thắt không.**
5. **9 lần mất máy ảo, chưa có bản sao lưu** — rủi ro vận hành lớn nhất, khắc
   phục chỉ tốn 30 giây (`huggingface-cli login`).

---

## Phần 7 — Bản đồ tài liệu và công cụ

**Tài liệu** (đọc theo thứ tự nếu muốn đào sâu):

| File | Nội dung |
|---|---|
| [00-doc-tu-dau.md](00-doc-tu-dau.md) | Bản đồ + tóm tắt 2 phút |
| [01-kien-thuc-nen.md](01-kien-thuc-nen.md) | Khái niệm nền, chi tiết hơn Phần 2 |
| [02-hanh-trinh-toi-uu.md](02-hanh-trinh-toi-uu.md) | Hành trình, chi tiết hơn Phần 3 |
| [03-bug-va-cach-sua.md](03-bug-va-cach-sua.md) | 20+ bug kể bằng ngôn ngữ thường |
| [04-ket-qua-va-cach-dung.md](04-ket-qua-va-cach-dung.md) | Cấu hình + checklist triển khai |
| [05-hoc-cuda-vllm-serving.md](05-hoc-cuda-vllm-serving.md) | **Học nghề**: CUDA → vLLM → serving, 6 tầng, lộ trình 6 tuần |
| [agent-loop-playbook.md](agent-loop-playbook.md) | 20 kịch bản hệ agent gọi tool |
| [routing-research.md](routing-research.md) | Thuật toán định tuyến + 5 đề xuất cải tiến |
| `STATUS.md` (thư mục gốc) | Sổ ghi chép kỹ thuật gốc — chi tiết, khô khan |

**Công cụ chính:**

| Script | Dùng để |
|---|---|
| `setup_env.sh` | Dựng lại toàn bộ môi trường bằng một lệnh |
| `graft_gguf_gdn.py` | Tạo checkpoint vô địch |
| `warmup_prefix.py` | Làm nóng trước khi mở cửa |
| `bench_skills.py` | Đo trên nền dùng chung (có chế độ tổng hợp, chạy được không cần dữ liệu riêng) |
| `bench_sla_prefix.py` | Đo cam kết dịch vụ theo nhịp khách đến thật |
| `bench_agent_loop.py` | Đo vòng lặp agent + 8 chế độ gây stress |
| `prefill_bench.py` | Đo riêng khâu đọc, có chống cache |
| `eval_quality_swebench.py` | **Cổng chất lượng — chạy trước mọi lần đổi checkpoint** |

---

## Một câu để mang theo

Phần lớn công sức của dự án này **không phải làm cho máy tính nhanh hơn**, mà là
**giảm số byte phải khuân**, **tránh làm lại việc đã làm**, và — quan trọng nhất
— **không tin bất kỳ con số nào cho tới khi hiểu vì sao nó ra như vậy.**
