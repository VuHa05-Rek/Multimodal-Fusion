# DATA CONTRACT & DICTIONARY (SILVER LAYER)
**Dự án:** Multi-modal Fusion for E-Commerce Product Classification
**Ngày cập nhật:** 14/11/2026 | **Người chịu trách nhiệm:** Nguyễn Vũ Hà

## 1. Thông tin Dataset
* **Nguồn (Bronze):** `Fashion Dataset.csv` (Myntra Dataset).
* **Định dạng đầu ra (Silver):** Delta Lake.
* **Tổng số dòng:** ~12,745.
* **Số lớp Target Label:** 7 lớp (`Casual`, `Daily`, `Ethnic`, `Festive`, `Party`, `Traditional`, `Western`).

## 2. Cấu trúc Schema (Data Dictionary)

| Cột | Kiểu Dữ Liệu | Nguồn Trích Xuất | Ý Nghĩa / Cách Tiền Xử Lý |
| :--- | :--- | :--- | :--- |
| `id` | String | Index `_c0` (raw) | Khóa chính của sản phẩm. |
| `image_file` | String | Sinh tự động | Tên file ảnh (vd: `id` + `.jpg`). Khớp 100% với file vật lý. |
| `text` | String | `name` + `description` | Văn bản đã xóa toàn bộ thẻ HTML, lowercase, bỏ khoảng trắng thừa. |
| `price` | Double | `price` | Mức giá gốc (Điền `0.0` nếu NULL). |
| `avg_rating` | Double | `avg_rating` | Điểm đánh giá trung bình. |
| `ratingCount` | Double | `ratingCount` | Số lượng lượt đánh giá. |
| `colour` | String | `colour` | Nhóm màu chủ đạo. Đã Index thành `colour_idx`. |
| `brand` | String | `brand` | Tên thương hiệu. Đã Index thành `brand_idx`. |
| `fabric` | String | `p_attributes` | Trích xuất JSON bằng UDF. Đã Index thành `fabric_idx`. |
| `pattern` | String | `p_attributes` | Hoa văn. Trích xuất JSON bằng UDF. Đã Index thành `pattern_idx`. |
| `wash_care` | String | `p_attributes` | HD giặt ủi. Trích xuất JSON bằng UDF. Đã Index thành `wash_care_idx`. |
| `sustainable` | String | `p_attributes` | Tính bền vững. Trích xuất JSON bằng UDF. Đã Index thành `sustainable_idx`. |
| `label` | String | `p_attributes` | **Target Label (Occasion)**. Đã Index thành `label_idx` (0-6). |
| `split` | String | `split_ids.csv` | Tập dữ liệu phân chia với tỷ lệ Train (70) / Val (15) / Test (15). |

## 3. Xác nhận Thỏa thuận (SLA)
- Các cột có hậu tố `_idx` (ví dụ: `label_idx`, `fabric_idx`) được mã hóa liên tục bắt đầu từ `0` bằng `PySpark StringIndexer`.
- Code Deep Learning của Tân sẽ sử dụng trực tiếp các cột `_idx` đưa vào PyTorch `nn.Embedding`.
- Không có bất kỳ giá trị `NULL` nào tồn tại trong bảng Delta.