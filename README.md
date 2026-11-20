# Dự Án Big Data & Deep Learning: Multi-Modal Fusion cho Phân Loại Sản Phẩm Thương Mại Điện Tử (Fashion Product Images — 43,982 Mẫu)

## 1. Giới thiệu Dự Án
Dự án xây dựng hệ thống **Multi-Modal Deep Learning kết hợp Big Data Lakehouse** nhằm phân loại mục đích sử dụng sản phẩm (`usage` — 4 lớp: *Casual, Ethnic, Formal, Sports*) trên bộ dữ liệu **Fashion Product Images Dataset** (`paramaggarwal/fashion-product-images-dataset`, gồm `43,982` sản phẩm hợp lệ sau làm sạch qua PySpark ETL: `train = 30,787`, `val = 6,597`, `test = 6,598`).

Hệ thống kết hợp 3 phương thức dữ liệu (Modalities):
- **Hình ảnh (Image):** Trích xuất đặc trưng `2048-d` bằng mạng **ResNet-50** (`ResNet50_Weights.DEFAULT`).
- **Văn bản (Text):** Trích xuất đặc trưng `768-d` (`[CLS]` token) từ cột `text_masked` (đã che từ khóa lộ nhãn để chống Data Leakage) bằng **DistilBERT** (`distilbert-base-uncased`).
- **Dữ liệu Bảng (Tabular):** Mã hóa các thuộc tính chuẩn Data Contract (`gender`, `masterCategory`, `subCategory`, `baseColour`, `season`, `year` $\rightarrow$ `107-d`) qua **Tabular MLP Encoder** (`107-d -> 64-d`).

---

## 2. Cấu trúc Thư mục & Các Script / Notebook (Hoàn thành Ngày 1 – Ngày 8)

| Tên File / Thư mục | Nhiệm vụ (Task ID) | Mô tả chi tiết |
| :--- | :---: | :--- |
| `01_eda_preliminary.ipynb` | `H01–H04`, `B01` | Khám phá dữ liệu sơ bộ (EDA), kiểm tra phân phối nhãn và chất lượng ảnh/văn bản. |
| `notebooks/02_dataset_dataloader_gpu_test.ipynb` | `T01, T02, T04–T06` | Benchmark GPU T4, xây dựng `PyTorch FashionMultimodalDataset` và `DataLoader`, kiểm tra ghép nối `clean_v0.csv` + `split_ids.csv` + `label_map.json` (`43,982` mẫu). |
| `reports/T03_LyThuyet_MultimodalFusion.docx` | `T03` | Tài liệu cơ sở lý thuyết Chương 2: ResNet-50, DistilBERT, Tabular MLP và kiến trúc Early/Late/Intermediate Multi-Modal Fusion. |
| `notebooks/03_feature_extraction.ipynb` | `T07–T12` | Trích xuất Offline Features trên GPU T4 và kiểm tra khớp `id` 100% với `split_ids.csv` (`train`: 30,787 / `val`: 6,597 / `test`: 6,598). |
| `src/evaluate.py` | `H13` | Module chuẩn hóa đánh giá mô hình: tính `Accuracy`, `Macro Precision`, `Macro Recall`, `Macro F1-Score` và xuất biểu đồ `Confusion Matrix`. |
| `src/models_and_train.py` | `T14, T15` | Module độc lập chứa kiến trúc `TextBaselineHead`, `TabularEncoder`, `MultiModalFusionHead` (`Concat 2880-d -> 512 -> 256 -> 4`) và hàm huấn luyện `train_model()`. |
| `notebooks/05_baselines_and_fusion.ipynb` | `T13–T22`, `H14–H23` | Huấn luyện & đánh giá 3 Single-Modal Baselines, xử lý mất cân bằng lớp (`Class Weights`), huấn luyện `Multi-Modal Fusion`, Tuning (`T21`), 3-Seed `Mean ± Std` (`T22`), và Ablation Study (`H22, H23`). |
| `results/` | `M2, M3, Day 8` | Chứa các bảng CSV tổng hợp (`val_comparison_final.csv`, `val_tuning_results.csv`, `val_3seeds_summary.csv`, `val_ablation_study.csv`), biểu đồ `learning_curves_fusion_v1.png`, `ablation_study_chart.png` và thư mục `confusion_matrix/`. |

---

## 3. Kết quả Thực nghiệm (Tập Validation — 6,597 mẫu)

### 3.1. Bảng So sánh 3 Single-Modal Baselines vs. Multi-Modal Fusion (Milestone M2 & M3 Gate)

| STT | Mô hình (Model) | Luồng dữ liệu đầu vào | Val Accuracy | Macro Precision | Macro Recall | Val Macro-F1 | Trạng thái Gate M3 |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Model 1: Tabular-Only (MLP)** | Tabular (`107-d -> 64-d`) | `76.75%` | `0.5622` | `0.7037` | `0.6093` | Baseline 1 |
| 2 | **Model 2: Image-Only (ResNet50)** | Image (`2048-d`) | `91.59%` | `0.8154` | `0.9038` | `0.8551` | Baseline 2 |
| 3 | **Model 3: Text-Only (DistilBERT)** | Text (`768-d`, `text_masked`) | `91.36%` | `0.8086` | `0.9153` | `0.8541` | Baseline 3 |
| 4 | **Model 4: Multi-Modal Fusion (Ours)** | **Concat (`2048 + 768 + 64 = 2880-d`)** | **`94.63%`** | **`0.8768`** | **`0.9425`** | **`0.9066`** | **ĐẠT GATE M3 (Vượt xa cả 3 Baselines: +3.04% Acc, +5.15% Macro-F1)** |

---

### 3.2. Độ ổn định trên 3 Seed độc lập `[42, 100, 2026]` (Task T22)

| Chỉ số (Metric) | Seed 42 | Seed 100 | Seed 2026 | Mean ± Std |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Accuracy** | `0.9469` | `0.9459` | `0.9476` | **`0.9468 ± 0.0009`** |
| **Macro Precision** | `0.8778` | `0.8761` | `0.8827` | **`0.8789 ± 0.0034`** |
| **Macro Recall** | `0.9450` | `0.9427` | `0.9414` | **`0.9430 ± 0.0018`** |
| **Macro F1-Score** | `0.9082` | `0.9062` | `0.9094` | **`0.9079 ± 0.0016`** |

---

### 3.3. Kết quả Nghiên cứu Loại bỏ Từng phần — Ablation Study (Task H22 & H23)

| Tổ hợp Phương thức (Combination) | Nhánh sử dụng | Val Accuracy | Macro Precision | Macro Recall | Val Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0a. Tabular-Only (Baseline)` | `Tab` | `76.75%` | `0.5622` | `0.7037` | `0.6093` |
| `0b. Image-Only (Baseline)` | `Img` | `91.59%` | `0.8154` | `0.9038` | `0.8551` |
| `0c. Text-Only (Baseline)` | `Txt` | `91.36%` | `0.8086` | `0.9153` | `0.8541` |
| `1. Image + Tabular (w/o Text)` | `Img + Tab` | `92.27%` | `0.8289` | `0.9194` | `0.8689` |
| `2. Image + Text (w/o Tabular)` | `Img + Txt` | `94.42%` | `0.8746` | `0.9406` | `0.9046` |
| `3. Text + Tabular (w/o Image)` | `Txt + Tab` | `91.94%` | `0.8191` | `0.9358` | `0.8666` |
| `4. All 3 Modalities (Full Fusion)` | **`Img + Txt + Tab`** | **`94.59%`** | **`0.8749`** | **`0.9470`** | **`0.9073`** |
