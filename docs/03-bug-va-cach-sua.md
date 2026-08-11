# 03 — Những con bug đã gặp, kể bằng ngôn ngữ thường

Hơn 20 lỗi thật đã được tìm ra và sửa. File này chọn những con đáng học nhất,
xếp theo **mức độ nguy hiểm — mà nguy hiểm nhất luôn là loại KHÔNG báo lỗi**.

---

## Loại 1 — Nguy hiểm nhất: hỏng trong im lặng

### 1.1. Thư viện phát hành sai, âm thầm dùng đường chậm gấp 4

**Chuyện gì:** thư viện chạy GGUF có một phần "tăng tốc" viết bằng CUDA. Bản
phát hành công khai được biên dịch lệch phiên bản, nên khi nạp thì **im lặng
thất bại** và tự rơi về đường dự phòng chậm hơn.

**Vì sao khó phát hiện:** không có thông báo lỗi. Mọi thứ *chạy đúng*, chỉ là
chậm. Suốt một thời gian dài mọi số đo GGUF của dự án đều là số của đường chậm.

**Cách phát hiện:** thấy tốc độ thấp bất thường ở batch nhỏ → thử nạp thủ công
phần CUDA đó → lộ ra lỗi ký hiệu.

**Cách sửa:** tự biên dịch lại từ mã nguồn cho đúng máy. **Nhanh lên 3,9 lần**
ở tình huống một người dùng.

> **Bài học:** khi một con số "hơi tệ hơn kỳ vọng", đừng chấp nhận. Hãy hỏi
> "liệu thứ tôi nghĩ đang chạy có thật sự đang chạy không?"

### 1.2. Ghép cấu hình sai — quy tắc chung nuốt mất quy tắc riêng

**Chuyện gì:** ta viết cấu hình "phần GDN dùng kiểu nén A", nhưng hệ thống lại
có sẵn quy tắc chung "mọi lớp Linear dùng kiểu nén B". Khi tìm quy tắc áp dụng,
nó so **tên kiểu** bằng cách "có chứa chữ" — mà tên lớp GDN *có chứa* chữ
"Linear" → quy tắc chung thắng, quy tắc riêng bị bỏ qua **không một lời cảnh báo**.

**Hậu quả:** checkpoint tưởng đã nén theo cách ta muốn, thực tế nén kiểu khác,
rồi vỡ ở bước sau với thông báo lỗi chẳng liên quan gì.

**Cách sửa:** đặt tên mục tiêu chính xác thay vì dựa vào so khớp mờ.

### 1.3. Trọng số bị "gài số" theo kiểu riêng của llama.cpp

**Chuyện gì:** khi chuyển mô hình sang định dạng GGUF, llama.cpp *biến đổi* vài
tham số theo quy ước riêng (cộng 1 vào một loại tham số, đổi dấu-mũ ở tham số
khác, xáo thứ tự vài hàng). Nếu bóc ra dùng mà không đảo ngược các phép đó,
mô hình vẫn **chạy được** nhưng **nói năng sai lệch**.

**Vì sao đáng sợ:** nó không sập. Nếu chỉ kiểm bằng "server có lên không" thì
qua hết.

**Cách sửa:** đảo ngược đủ 4 phép biến đổi, và viết bài kiểm tra kiểu *tạo dữ
liệu chuẩn → biến đổi xuôi → đảo lại → so khớp*, kèm phép thử chứng minh **nếu
quên đảo thì bài kiểm phải trượt** (nếu không, bài kiểm vô dụng).

### 1.4. Bài kiểm tra tự đồng thuận với chính nó

**Chuyện gì:** bộ kiểm thử tự tạo dữ liệu giả với **cùng một lỗi đặt tên** như
mã đang kiểm → hai bên "đồng ý" với nhau, bài kiểm PASS trong khi mã sai.

**Bài học:** dữ liệu kiểm thử phải đối chiếu với **định dạng thật**, không phải
với giả định của chính mình.

## Loại 2 — Bug đo lường: số đúng nhưng nghĩa sai

### 2.1. Bộ nhớ đệm làm đẹp số (bạn là người phát hiện)

Bài đo cũ xoay vòng 500 câu hỏi; chạy quá 500 lượt là bắt đầu lặp lại câu cũ →
máy trả lời nhanh giả vì đã nhớ. **Sửa:** gắn nhãn duy nhất ở *đầu* mỗi câu.

### 2.2. "Đổ cả rổ" khác hẳn "khách đến rải rác"

Đo bằng cách bắn 32 yêu cầu cùng lúc cho **1,3 yêu cầu/giây**. Đo lại theo nhịp
đến thực tế: **0,2**. Lạc quan gấp 6 lần. Mọi cam kết dịch vụ về sau đều phải
đo theo kiểu thứ hai.

### 2.3. Bỏ bước làm nóng → mức đo đầu tiên gánh hết chi phí nguội

Kết quả ra "0,4 tệ hơn 0,5" — đảo ngược trực giác. Không phải phát hiện gì cả,
chỉ là thứ tự chạy. **Luật mới:** luôn làm nóng trước khi đo.

### 2.4. Dùng lại số cũ làm mốc so sánh

Một lần so chất lượng dùng con số chép từ sổ (máy cũ đã bị xóa). Kết luận vẫn
đúng, nhưng phải chạy lại toàn bộ mốc **trên cùng một máy** mới dám phong vô
địch. **Luật:** mốc so sánh phải đo tươi cùng phiên.

### 2.5. Một con số báo sai vì đọc nhầm

Có lúc báo "nén xong trong 13 phút" — thực tế là ~160 phút. Phát hiện khi soi
lại dấu thời gian trong nhật ký. Đã đính chính công khai trong sổ.

> **Bài học chung của loại 2:** phần lớn "phát hiện lớn" hóa ra là lỗi đo. Quy
> tắc: kết quả nào *quá đẹp* hoặc *phản trực giác* → nghi ngờ phép đo trước,
> nghi ngờ thế giới sau.

## Loại 3 — Bug tài liệu và môi trường

### 3.1. Cái bẫy trong chính hướng dẫn của mình

Một công cụ tự in ra gợi ý "hãy chạy thêm bước sửa checkpoint" — nhưng với loại
checkpoint đó, bước này **phá hỏng** mô hình. Nó đã lừa được cả một trợ lý cẩn
thận. **Sửa:** đổi dòng gợi ý thành lời cảnh báo ngược lại.

### 3.2. Cài thư viện A âm thầm hạ cấp thư viện B

Thêm một dòng cài đặt vô hại làm tụt phiên bản một thư viện nền → **mọi lần
khởi động đều chết ngay từ bước nạp**, kể cả với mô hình chẳng liên quan.

### 3.3. Đổi tên chỉ số giữa các phiên bản

Công cụ đo báo "không có dữ liệu bộ nhớ đệm" suốt nhiều lần — hóa ra phiên bản
mới đổi tên chỉ số (thêm hậu tố). Cache vẫn chạy tốt 99%, chỉ là đọc nhầm chỗ.

## Loại 4 — Giới hạn thật, không phải bug (nhưng phải chứng minh)

Ba thứ ta muốn làm mà **không làm được**, mỗi thứ đều có bằng chứng từ mã nguồn
chứ không phải phỏng đoán:

| Muốn làm | Vì sao không được |
|---|---|
| Nén lớp phát chữ (lm_head) | vLLM không có đường nén cho loại lớp này |
| Chia sẻ lượt tra sổ giữa nhiều người (cascade) | Chính vLLM tắt cứng kèm ghi chú "chưa chạy được" |
| Dùng kernel attention khác | Không tương thích với định dạng bộ nhớ đệm đang dùng |

> **Bài học:** "không làm được" cũng là kết quả có giá trị — miễn là kèm bằng
> chứng để người sau khỏi đào lại.

---

## Ba nguyên tắc rút ra

1. **Lỗi im lặng nguy hiểm hơn lỗi ồn ào.** Ưu tiên thiết kế sao cho sai thì
   *sập ngay*, đừng để sai mà vẫn chạy.
2. **Không tin số cho tới khi hiểu vì sao nó ra như vậy.** Mọi con số đẹp bất
   thường trong dự án này đều từng là lỗi đo.
3. **Ghi lại cả thất bại.** Một nửa giá trị của sổ ghi chép là danh sách "đã
   thử, không được, đây là lý do".

➡️ Tiếp theo: [04-ket-qua-va-cach-dung.md](04-ket-qua-va-cach-dung.md)
