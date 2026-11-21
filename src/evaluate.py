import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, class_names, model_name="Model", save_dir="results/"):
    """
    Hàm đánh giá chuẩn dùng chung cho tất cả các mô hình Single-modal và Multi-modal.
    """
    # 1. Tính toán các Metrics cơ bản
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"=== KẾT QUẢ ĐÁNH GIÁ: {model_name} ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1-Score  : {macro_f1:.4f}\n")
    
    # 2. In Classification Report chi tiết từng lớp (chống crash khi thiếu lớp)
    labels = list(range(len(class_names)))
    print("--- Báo cáo phân loại chi tiết ---")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    
    # 3. Vẽ và lưu Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Thực tế (True)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.tight_layout()
    
    # Đảm bảo thư mục lưu tồn tại
    os.makedirs(save_dir, exist_ok=True)
    
    cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(cm_path, dpi=300)
    print(f"Đã lưu biểu đồ Confusion Matrix tại: {cm_path}")
    plt.close()
    
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1
    }

# ==========================================
# CÁCH SỬ DỤNG (Cho Hà và Tân gọi trong notebook):
# import json
# from evaluate import evaluate_model
#
# # Đọc trực tiếp từ label_map.json để đảm bảo đúng thứ tự index:
# with open('label_map.json', 'r') as f:
#     label_map = json.load(f)
# class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
# # Hoặc khai báo trực tiếp: ['Casual', 'Ethnic', 'Formal', 'Sports']
#
# metrics = evaluate_model(y_true, y_pred, class_names, model_name="Tabular_Baseline")
# ==========================================