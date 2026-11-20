import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

NUM_CLASSES = 4   # ['Casual', 'Ethnic', 'Formal', 'Sports']
TAB_IN_DIM  = 107 # OneHot(gender, masterCategory, subCategory, baseColour, season) + Scaled(year)

class TabularEncoder(nn.Module):
    """Mã hóa vector Tabular đầu vào (107-d) -> 64-d chuẩn thiết kế Charter."""
    def __init__(self, in_dim=TAB_IN_DIM, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class TabularBaselineModel(nn.Module):
    """Model 1: Tabular-Only (Input 107-d -> 64-d -> 4 classes)."""
    def __init__(self, in_dim=TAB_IN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = TabularEncoder(in_dim, 64, dropout=0.2)
        self.head = nn.Linear(64, num_classes)
    def forward(self, img, txt, tab):
        return self.head(self.encoder(tab))

class ImageBaselineModel(nn.Module):
    """Model 2: Image-Only (ResNet50 2048-d -> 256 -> 4 classes)."""
    def __init__(self, in_dim=2048, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, img, txt, tab):
        return self.net(img)

class TextBaselineHead(nn.Module):
    """Model 3: Text-Only (DistilBERT 768-d -> 256 -> 4 classes)."""
    def __init__(self, in_dim=768, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, img, txt=None, tab=None):
        x = img if txt is None else txt
        return self.net(x)

class MultiModalFusionHead(nn.Module):
    """
    Model 4: Multi-Modal Fusion chuẩn thiết kế:
    Image (2048-d) + Text (768-d) + Tabular (64-d) -> Concat (2880-d) -> 512 -> 256 -> 4 classes
    """
    def __init__(self, img_dim=2048, txt_dim=768, tab_in_dim=TAB_IN_DIM, num_classes=NUM_CLASSES,
                 hidden_dims=(512, 256), dropout=0.35, use_img=True, use_txt=True, use_tab=True):
        super().__init__()
        self.use_img, self.use_txt, self.use_tab = use_img, use_txt, use_tab
        h1, h2 = hidden_dims

        self.img_norm = nn.Sequential(nn.LayerNorm(img_dim), nn.Dropout(dropout))
        self.txt_norm = nn.Sequential(nn.LayerNorm(txt_dim), nn.Dropout(dropout * 0.7))
        self.tab_enc  = TabularEncoder(tab_in_dim, 64, dropout=0.2)

        self.fusion_net = nn.Sequential(
            nn.Linear(img_dim + txt_dim + 64, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(max(0.1, dropout - 0.1)),
            nn.Linear(h2, num_classes)
        )

    def forward(self, img, txt, tab):
        f_img = self.img_norm(img) if self.use_img else torch.zeros_like(img)
        f_txt = self.txt_norm(txt) if self.use_txt else torch.zeros_like(txt)
        f_tab = self.tab_enc(tab)  if self.use_tab else torch.zeros(img.size(0), 64, device=img.device)

        fused_2880 = torch.cat([f_img, f_txt, f_tab], dim=1)   # (B, 2880)
        return self.fusion_net(fused_2880)


def train_model(model, data_splits, class_weights, device, epochs=25, lr=5e-4,
                weight_decay=1e-2, label_smoothing=0.0, patience=6, save_path='best_model.pth'):
    model = model.to(device)
    tr_d, va_d = data_splits['train'], data_splits['val']
    tr_loader = DataLoader(TensorDataset(tr_d['img'], tr_d['txt'], tr_d['tab'], tr_d['y']), batch_size=256, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_d['img'], va_d['txt'], va_d['tab'], va_d['y']), batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = -1.0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for img, txt, tab, y in tr_loader:
            img, txt, tab, y = img.to(device), txt.to(device), tab.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(img, txt, tab)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        va_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for img, txt, tab, y in va_loader:
                img, txt, tab, y = img.to(device), txt.to(device), tab.to(device), y.to(device)
                logits = model(img, txt, tab)
                loss = criterion(logits, y)
                va_loss += loss.item() * y.size(0)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                trues.append(y.cpu().numpy())

        va_loss /= len(va_loader.dataset)
        preds, trues = np.concatenate(preds), np.concatenate(trues)
        va_acc = accuracy_score(trues, preds)
        va_f1  = f1_score(trues, preds, average='macro', zero_division=0)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['val_f1'].append(va_f1)
        scheduler.step(va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(torch.load(save_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for img, txt, tab, y in va_loader:
            img, txt, tab = img.to(device), txt.to(device), tab.to(device)
            preds.append(model(img, txt, tab).argmax(dim=1).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds), history
