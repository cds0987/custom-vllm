# Qwen3.5-MoE (122B-A10B, 35B-A3B) — ⬜ PLANNED, CHƯA HỖ TRỢ

Folder giữ chỗ theo sơ đồ sản phẩm. Chưa có dòng code nào — đừng để cấu trúc
đẹp tạo ảo giác đã hỗ trợ.

## Việc thật sự phải làm để thuần hóa (ước lượng từ kinh nghiệm dense)

1. Lớp MoE (expert weights) chưa từng qua tay graft/patch nào của repo này.
2. Khảo sát checkpoint: RedHatAI có FP8-dynamic cho cả 122B-A10B lẫn 35B-A3B;
   unsloth có GGUF + NVFP4. 35B-A3B (~3B active) là ứng viên L4 duy nhất.
3. Đo lại từ đầu trên hardware đích — không suy diễn từ số của dense.
