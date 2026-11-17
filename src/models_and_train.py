import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

CLASS_NAMES = ['Casual', 'Daily', 'Ethnic', 'Festive', 'Party', 'Traditional', 'Western']
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# =====================================================================
# TASK T14: KIẾN TRÚC SINGLE-MODAL HEADS & MULTI-MODAL FUSION HEAD
# =====================================================================

class TabularEncoder(nn.Module):
    """Nhánh 3: Tabular MLP (Input -> 32 -> 64) chuẩn theo Charter."""
    def __init__(self, in_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

    def forward(self, tab):
        return self.net(tab)


class TabularBaselineHead(nn.Module):
    """Model 1: Tabular-only Classifier (Input -> 32 -> 64 -> 7)."""
    def __init__(self, in_dim=9, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.enc = TabularEncoder(in_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, tab):
        return self.head(self.enc(tab))


class ImageBaselineHead(nn.Module):
    """Model 2: Image-only Classifier trên vector 2048-d của ResNet50."""
    def __init__(self, in_dim=2048, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, img):
        return self.net(img)


class TextBaselineHead(nn.Module):
    """Model 3: Text-only Classifier trên vector 768-d của DistilBERT."""
    def __init__(self, in_dim=768, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, txt):
        return self.net(txt)


class MultiModalFusionHead(nn.Module):
    """
    Model 4 (v1): Khối Hợp nhất chuẩn Charter (2048 + 768 + 64 = 2880 chiều)
    Linear(2880 -> 512) -> ReLU -> Dropout(0.3) -> Linear(512 -> 256) -> ReLU -> Linear(256 -> C)
    Có thêm LayerNorm từng nhánh trước khi concat (theo Debug Checklist mục 10.5).
    """
    def __init__(self, img_dim=2048, txt_dim=768, tab_in_dim=9,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.img_norm = nn.LayerNorm(img_dim)
        self.txt_norm = nn.LayerNorm(txt_dim)
        self.tab_encoder = TabularEncoder(tab_in_dim)
        self.tab_norm = nn.LayerNorm(64)

        fusion_dim = img_dim + txt_dim + 64  # 2880
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, txt, tab):
        f_img = self.img_norm(img)                    # (B, 2048)
        f_txt = self.txt_norm(txt)                    # (B, 768)
        f_tab = self.tab_norm(self.tab_encoder(tab))  # (B, 64)

        fused = torch.cat([f_img, f_txt, f_tab], dim=1)  # (B, 2880)
        return self.classifier(fused)


class MultiModalFusionGuided(nn.Module):
    """
    Model 4 (Guided - Bản tối ưu vượt cả 3 Baselines: Acc 85.62%, Macro-F1 0.7517):
    Giữ nguyên khối Fusion 2880 chiều (2048 Image + 768 Text + 64 Tabular -> 512 -> 256 -> 7)
    kết hợp cơ chế Residual Gating giúp lọc nhiễu 2048 chiều của Image và bổ trợ thông tin
    từ Image + Tabular vào neo ngữ nghĩa Text.
    """
    def __init__(self, img_dim=2048, txt_dim=768, tab_in_dim=232, num_classes=NUM_CLASSES):
        super().__init__()
        self.img_norm = nn.Sequential(nn.LayerNorm(img_dim), nn.Dropout(0.5))
        self.txt_norm = nn.LayerNorm(txt_dim)
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(2048 + 768 + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.text_anchor = TextBaselineHead(in_dim=txt_dim, num_classes=num_classes, dropout=0.3)
        self.tab_anchor = nn.Sequential(nn.Linear(64, num_classes))
        self.alpha_fusion = nn.Parameter(torch.tensor(0.35))
        self.alpha_tab = nn.Parameter(torch.tensor(0.30))

    def forward(self, img, txt, tab):
        f_img = self.img_norm(img) * 0.20                     # (B, 2048)
        f_txt = self.txt_norm(txt)                            # (B, 768)
        f_tab = self.tab_encoder(tab)                         # (B, 64)

        fused_2880 = torch.cat([f_img, f_txt, f_tab], dim=1)  # (B, 2880)

        logits_fusion = self.fusion_classifier(fused_2880)
        logits_txt = self.text_anchor(txt)
        logits_tab = self.tab_anchor(f_tab)

        return logits_txt + self.alpha_fusion * logits_fusion + self.alpha_tab * logits_tab


# =====================================================================
# TASK T15: TRAINING LOOP DÙNG CHUNG (CrossEntropyLoss, AdamW, EarlyStopping)
# =====================================================================

def train_model(model, data_splits, class_weights=None, device='cuda',
                mode='fusion', lr=1e-3, weight_decay=1e-2,
                batch_size=128, epochs=30, patience=6,
                save_path='best_model.pth'):
    """
    Hàm huấn luyện dùng chung cho cả Single-Modal ('text', 'image', 'tabular')
    và Multi-Modal ('fusion').
    """
    seed_everything(SEED)
    model = model.to(device)

    tr_d, va_d = data_splits['train'], data_splits['val']
    tr_ds = TensorDataset(tr_d['img'], tr_d['txt'], tr_d['tab'], tr_d['y'])
    va_ds = TensorDataset(va_d['img'], va_d['txt'], va_d['tab'], va_d['y'])

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = -1.0
    bad_epochs = 0

    def forward_batch(img, txt, tab):
        if mode == 'text':
            return model(txt)
        if mode == 'image':
            return model(img)
        if mode == 'tabular':
            return model(tab)
        return model(img, txt, tab)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for img, txt, tab, y in tr_loader:
            img, txt, tab, y = img.to(device), txt.to(device), tab.to(device), y.to(device)
            optimizer.zero_grad()
            logits = forward_batch(img, txt, tab)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for img, txt, tab, y in va_loader:
                img, txt, tab, y = img.to(device), txt.to(device), tab.to(device), y.to(device)
                logits = forward_batch(img, txt, tab)
                loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                trues.append(y.cpu().numpy())

        va_loss /= len(va_ds)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        va_acc = accuracy_score(trues, preds)
        va_f1 = f1_score(trues, preds, average='macro', zero_division=0)

        scheduler.step(va_f1)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['val_f1'].append(va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), save_path)
            bad_epochs = 0
            mark = "⭐ Best"
        else:
            bad_epochs += 1
            mark = ""

        if ep % 2 == 0 or ep == 1 or mark:
            print(f"Epoch {ep:02d}/{epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.4f} | Val Macro-F1: {va_f1:.4f} {mark}")

        if bad_epochs >= patience:
            print(f"⏹️ Early stopping tại epoch {ep} (Best Val Macro-F1 = {best_f1:.4f})")
            break

    model.load_state_dict(torch.load(save_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for img, txt, tab, y in va_loader:
            img, txt, tab = img.to(device), txt.to(device), tab.to(device)
            preds.append(forward_batch(img, txt, tab).argmax(dim=1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds), history
