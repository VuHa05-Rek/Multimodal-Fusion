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
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1-Score  : {macro_f1:.4f}\n")
    
    # 2. In Classification Report chi tiết từng lớp
    print("--- Báo cáo phân loại chi tiết ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # 3. Vẽ và lưu Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Thực tế (True)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.tight_layout()
    
    # Đảm bảo thư mục lưu tồn tại
    import os
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
# from evaluate import evaluate_model
# class_names = ['Casual', 'Daily', 'Ethnic', 'Festive', 'Party', 'Traditional', 'Western']
# metrics = evaluate_model(y_true, y_pred, class_names, model_name="Tabular_Baseline")
# ==========================================