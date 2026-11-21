# Risk Log — Dự án Multi-Modal Product Classification

> **Cập nhật lần cuối:** Ngày 7 (Cuối Tuần 1 — Review Gate M3)  
> **Phụ trách cập nhật:** Hà (PM / Data Engineering) & Tân (Deep Learning)  
> **Quy ước:** `XS` = Xác suất | `TĐ` = Tác động (`Cao` / `TB` / `Thấp`)

---

## 1. Tóm tắt trạng thái sau Tuần 1

* **Đã xử lý xong (5 rủi ro):** `R1`, `R2`, `R3`, `R7`, `R8`
* **Ưu tiên kiểm soát cao ở Tuần 2 (2 rủi ro):** `R9` (chống rò rỉ tập Test), `R12` (tiến độ báo cáo & slide)
* **Đang theo dõi định kỳ (5 rủi ro):** `R4`, `R5`, `R6`, `R10`, `R11`

---

## 2. Bảng theo dõi chi tiết rủi ro (R1 – R12)

| ID | Mô tả rủi ro | XS | TĐ | Giảm thiểu / Phương án dự phòng | Owner | Trạng thái (Ngày 7) | Ghi chú thực tế sau Tuần 1 |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **R1** | Dataset thiếu `description`/`price` so với charter | Cao | TB | Dùng `productDisplayName` làm Text; dùng 7 cột metadata làm Tabular; ghi rõ khác biệt trong báo cáo. | Hà | ✅ Đã xử lý xong | Đã chốt trong Data Contract; loại `usage` khỏi Tabular vì là target label (4 lớp). |
| **R2** | Rò rỉ nhãn qua text → Text-only quá cao, Fusion không nổi bật | Cao | Cao | Kiểm tra ở Ngày 2; chạy thêm biến thể che từ khóa nhãn `[MASK]`; báo cáo minh bạch cả hai bản. | Hà | ✅ Đã xử lý xong | Tân đã chạy đối chứng Text gốc (`Macro-F1: 0.8832`) và Text masked (`0.8541`, giảm `-0.0291`). |
| **R3** | Fusion không vượt Text-only / Image-only | TB | Cao | Debug Checklist: LayerNorm/BatchNorm trước concat, tuning lr/dropout, class weights, kiểm tra căn chỉnh ID. | Tân | ✅ Đã xử lý xong | **Gate M3 = GO.** Fusion v1 (Weighted/Tuned) đạt `Val Macro-F1 = 0.9066`, vượt baseline tốt nhất (`0.8832`) là `+0.0234`. |
| **R4** | Lỗi mount Google Drive, giới hạn RAM Colab hoặc mất kết nối | TB | TB | Timebox 4h; lưu checkpoint Parquet/Delta sau mỗi bước ETL trên Drive; dự phòng chạy Spark trên Kaggle. | Hà | 🟡 Đang theo dõi | ETL Bronze → Silver → Gold đã chạy ổn định. Sẽ đóng gói chạy end-to-end idempotent ở Ngày 8 (`H24`). |
| **R5** | Hết quota Kaggle GPU / session timeout | TB | TB | Trích xuất offline features `.pt` 1 lần; lưu checkpoint mỗi epoch; upload Kaggle Dataset; dự phòng Colab. | Tân | 🟡 Đang theo dõi | Đã trích xuất xong 6 file `.pt` lên Kaggle Dataset. Tiếp tục tắt GPU ngay sau mỗi phiên Tuning/Ablation Tuần 2. |
| **R6** | Out-of-Memory (OOM) khi fine-tune end-to-end | TB | Thấp | Dùng batch 16–32, bật AMP, freeze các tầng đầu; Stage B ở mức Should-have nên có thể cắt nếu thiếu tài nguyên. | Tân | 🟡 Đang theo dõi | Chưa triển khai (dự kiến đánh giá ở Ngày 9 theo ưu tiên MoSCoW). |
| **R7** | Mất cân bằng lớp nặng (`Casual` ~78%) | Cao | TB | Áp dụng class weights; dùng `Macro-F1` làm thước đo chính thay vì chỉ nhìn `Accuracy`. | Hà | ✅ Đã xử lý xong | Đã thử nghiệm H18 (`Weighted Sqrt` vs `Raw`); Tân áp dụng `Smoothed Sqrt Class Weights` thành công vào Fusion v1. |
| **R8** | Lệch split / schema giữa Spark và PyTorch | TB | Cao | Đóng băng `split_ids.csv` (stratified 70/15/15, `seed=42`) + Data Contract; kiểm tra count & khớp ID ở M1. | Hà | ✅ Đã xử lý xong | Đã xác nhận khớp 100% ID giữa `gold_metadata.parquet` và 6 file features `.pt`. |
| **R9** | Rò rỉ tập Test (tune siêu tham số trên Test) | Thấp | Cao | Chỉ tune trên tập Val; tập Test chỉ chạy đúng 1 lần ở Ngày 10 qua test-harness (`H27`). | Hà | 🔴 Ưu tiên Tuần 2 | Tập Test vẫn đang đóng băng hoàn toàn. Sẽ viết test-harness khóa quy trình chạy ở Ngày 9. |
| **R10** | Một thành viên ốm / bận (đội chỉ 2 người) | TB | Cao | Cross-review hàng ngày, đồng bộ code/kết quả qua GitHub & Excel chung, giữ buffer Ngày 14, áp dụng MoSCoW. | Cả hai | 🟡 Đang theo dõi | Tiến độ Tuần 1 bám sát kế hoạch; cả 2 đã đồng bộ xong kết quả Giai đoạn 1–2 trên file Excel chung. |
| **R11** | Scope creep (cross-attention, hierarchical, deploy phức tạp) | TB | TB | Bám sát MoSCoW; mọi thay đổi phạm vi phải ghi Decision Log và đánh đổi bằng 1 task khác. | Hà | 🟡 Đang theo dõi | Chưa phát sinh ngoài phạm vi. Tập trung hoàn thiện Must/Should trước khi cân nhắc hạng mục Could. |
| **R12** | Dồn báo cáo và slide về những ngày cuối | Cao | TB | Viết báo cáo song song từ Ngày 6; mỗi chương gắn liền với từng milestone. | Cả hai | 🔴 Ưu tiên Tuần 2 | Hiện còn nợ `H19` (Ch.1–2) và `H21` (Ch.3.1). Cần viết cuốn chiếu trong Ngày 7–8 để kịp mốc M5 (Ngày 11). |

---

## 3. Nhật ký cập nhật

* **Ngày 1:** Khởi tạo danh sách 12 rủi ro (`R1–R12`) theo Kế hoạch thực hiện 2 tuần.
* **Ngày 7 (M3 Review):** Đóng 5 rủi ro đã xử lý thành công (`R1`, `R2`, `R3`, `R7`, `R8`); nâng mức ưu tiên theo dõi cho `R9` và `R12` trong Tuần 2.