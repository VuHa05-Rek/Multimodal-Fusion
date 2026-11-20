import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_predictions(y_true, y_pred, class_names, model_name="Model", save_dir="/kaggle/working/results"):
    os.makedirs(save_dir, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"=== KẾT QUẢ ĐÁNH GIÁ: {model_name} ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1-Score  : {macro_f1:.4f}")
    print("\n--- Báo cáo phân loại chi tiết ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=13, fontweight='bold')
    plt.ylabel('Thực tế (True)', fontsize=11)
    plt.xlabel('Dự đoán (Predicted)', fontsize=11)
    plt.tight_layout()
    img_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(img_path, dpi=200)
    plt.show()
    plt.close()
    print(f"Đã lưu biểu đồ Confusion Matrix tại: {img_path}\n")

    return {"accuracy": acc, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1}
