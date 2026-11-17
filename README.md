# Dự Án Big Data & Deep Learning: Multi-Modal Fusion cho Phân Loại Sản Phẩm Thương Mại Điện Tử (Myntra Fashion)

## 1. Giới thiệu Dự Án
Dự án xây dựng hệ thống **Multi-Modal Deep Learning kết hợp Big Data Lakehouse** nhằm phân loại thuộc tính dịp sử dụng (`Occasion` — 7 lớp: *Casual, Daily, Ethnic, Festive, Party, Traditional, Western*) trên bộ dữ liệu **Myntra Fashion Product Dataset** (`12,659` sản phẩm hợp lệ sau làm sạch).

Hệ thống kết hợp 3 phương thức dữ liệu (Modalities):
- **Hình ảnh (Image):** Trích xuất đặc trưng `2048-d` bằng mạng **ResNet-50** (ImageNet1K_V2).
- **Văn bản (Text):** Trích xuất đặc trưng `768-d` (`[CLS]` token) từ `name + description` bằng **DistilBERT** (`distilbert-base-uncased`).
- **Dữ liệu Bảng (Tabular):** Mã hóa đặc trưng có cấu trúc (`colour`, `brand`, `price`, `avg_rating`, `ratingCount`) qua **Tabular MLP Encoder** (`64-d`).

---

## 2. Cấu trúc Thư mục & Các Script / Notebook (Hoàn thành Tuần 1 — Milestone M1, M2, M3)

| Tên File / Thư mục | Nhiệm vụ (Task ID) | Mô tả chi tiết |
| :--- | :---: | :--- |
| `01_eda_preliminary.ipynb` | `H01–H04`, `B01` | Khám phá dữ liệu sơ bộ (EDA), kiểm tra phân phối nhãn `Occasion` và chất lượng ảnh/văn bản. |
| `d2-bigdata.ipynb` | `T01, T02, T04–T06` | Benchmark GPU T4, xây dựng `PyTorch MultimodalDataset` và `DataLoader`, kiểm tra ghép nối `clean_v0.csv` + `split_ids.csv` + `label_map.json`. |
| `T03_LyThuyet_MultimodalFusion.docx` | `T03` | Tài liệu cơ sở lý thuyết Chương 2: ResNet-50, DistilBERT, Tabular MLP và kiến trúc Early/Late/Intermediate Multi-Modal Fusion. |
| `03-feature-extraction.ipynb` | `T07–T12` | Trích xuất Offline Features trên GPU T4 và kiểm tra khớp `id` 100% với `split_ids.csv` (`train`: 8,861 / `val`: 1,899 / `test`: 1,899). |
| `image_feats_{train,val,test}.pt` | `T07, T08` | Tensor đặc trưng ảnh ResNet-50 kích thước `(N, 2048)`, kiểu `float32`. |
| `text_feats_{train,val,test}.pt` | `T09, T10` | Tensor đặc trưng văn bản DistilBERT kích thước `(N, 768)`, kiểu `float32`. |
| `evaluate.py` | `H13` | Module chuẩn hóa đánh giá mô hình: tính `Accuracy`, `Macro Precision`, `Macro Recall`, `Macro F1-Score` và xuất biểu đồ `Confusion Matrix`. |
| `models_and_train.py` | `T14, T15` | Module độc lập chứa kiến trúc `TextBaselineHead`, `TabularEncoder`, `MultiModalFusionHead` và hàm huấn luyện `train_model()`. |
| `05-baselines-and-fusion.ipynb` | `T13–T19`, `H14–H18` | Huấn luyện & đánh giá toàn bộ 3 Single-Modal Baselines, xử lý mất cân bằng lớp (`Class Weights`), huấn luyện `Fusion v1`, vẽ `Learning Curves` và tối ưu `MultiModalFusionGuided` vượt Gate M3. |
| `day5_6_7_outputs/` | `M2, M3 Gate` | Chứa toàn bộ trọng số `.pth` tốt nhất và thư mục `results/` (9 biểu đồ Confusion Matrix, Learning Curves, và các file CSV tổng hợp kết quả). |

---

## 3. Kết quả Thực nghiệm Tuần 1 (Tập Validation — 1,899 mẫu)

| STT | Mô hình (Model) | Luồng dữ liệu đầu vào | Val Accuracy | Macro Precision | Macro Recall | Val Macro-F1 | Trạng thái Gate M3 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model 1: Tabular-Only (MLP)** | Tabular (`232-d -> 64-d`) | `76.04%` | `0.5664` | `0.6190` | `0.5852` | Baseline 1 |
| 2 | **Model 2: Image-Only (ResNet50)** | Image (`2048-d`) | `49.71%` | `0.1581` | `0.1489` | `0.1423` | Baseline 2 |
| 3 | **Model 3: Text-Only (DistilBERT)** | Text (`768-d`) | `84.89%` | `0.7289` | `0.7579` | `0.7403` | Baseline 3 (Mạnh nhất) |
| 4 | **Model 4: Multi-Modal Fusion v1 (Naive)** | Concat (`2048 + 768 + 64 = 2880-d`) | `78.15%` | `0.6123` | `0.6391` | `0.6218` | Bị nhiễu bởi chiều Image |
| 5 | **Model 5: Multi-Modal Fusion Guided (Ours)** | **Image + Text + Tabular (Guided)** | **`85.62%`** | **`0.7424`** | **`0.7633`** | **`0.7517`** | **ĐẠT GATE M3 (Vượt cả 3 Baselines)** |

---

## 4. Hướng dẫn Chạy lại (Reproducibility)
1. Đặt các file dữ liệu (`clean_v0.csv`, `split_ids.csv`, `label_map.json`) và 6 file `.pt` vào cùng thư mục hoặc gắn vào Kaggle Input.
2. Chạy `05-baselines-and-fusion.ipynb` (hoặc import từ `models_and_train.py` và `evaluate.py`).
3. Toàn bộ kết quả đánh giá và biểu đồ sẽ tự động lưu vào thư mục `results/`.
