import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import TripletMarginLoss
import librosa
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
root_dir = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/speakers"
model_dir = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary"
model_save_root = os.path.join(model_dir, "comp3_others_XinYuan_mining")
os.makedirs(model_save_root, exist_ok=True)

embed_dim = 64
batch_size = 16
num_epochs = 100
lr = 1e-3
margin = 1.0
val_ratio = 0.2

# ==================== Dataset with Anchor Augmentation ====================
class TripletMiningDataset(Dataset):
    def __init__(self, root_dir, n_mfcc=20, max_len=100):
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.margin = margin
        self.anchor_label = 1  # XiaoYuan
        self.data = {}

        label_map = {'XiaoXin': 0, 'XiaoYuan': 1, 'other_speaker': 2}
        for name, label in label_map.items():
            wavs = []
            folder = os.path.join(root_dir, name)
            for dirpath, _, filenames in os.walk(folder):
                for f in filenames:
                    if f.endswith('.wav'):
                        wavs.append(os.path.join(dirpath, f))
            self.data[label] = wavs

        self.positives = self.data[self.anchor_label]
        self.negatives = [f for k, v in self.data.items() if k != self.anchor_label for f in v]
        # 复制 anchor：原始一半 + 增强一半
        self.augmented_anchors = [(p, False) for p in self.positives] + [(p, True) for p in self.positives]
        random.shuffle(self.augmented_anchors)

    def __len__(self):
        return len(self.augmented_anchors)

    def augment_audio(self, y, sr):
        if random.random() < 0.5:
            y += 0.005 * np.random.randn(len(y))
        if random.random() < 0.5:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
        return y

    def wav_to_tensor(self, path, augment=False):
        y, sr = librosa.load(path, sr=16000)
        if augment:
            y = self.augment_audio(y, sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        if mfcc.shape[1] < self.max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :self.max_len]
        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        anchor_path, do_augment = self.augmented_anchors[idx]
        anchor_tensor = self.wav_to_tensor(anchor_path, augment=do_augment)

        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.positives)
        positive_tensor = self.wav_to_tensor(positive_path)

        candidate_neg_paths = random.sample(self.negatives, min(10, len(self.negatives)))
        d_ap = F.pairwise_distance(anchor_tensor.view(1, -1), positive_tensor.view(1, -1)).item()
        semi_hard_neg = None
        for neg_path in candidate_neg_paths:
            neg_tensor = self.wav_to_tensor(neg_path)
            d_an = F.pairwise_distance(anchor_tensor.view(1, -1), neg_tensor.view(1, -1)).item()
            if d_ap < d_an < d_ap + self.margin:
                semi_hard_neg = neg_tensor
                break
        if semi_hard_neg is None:
            semi_hard_neg = self.wav_to_tensor(random.choice(self.negatives))

        return anchor_tensor, positive_tensor, semi_hard_neg

# ==================== Model ====================
class MFCCEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

# ==================== Training ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_dataset = TripletMiningDataset(root_dir)
n_val = int(len(full_dataset) * val_ratio)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

model = MFCCEmbeddingNet(embed_dim).to(device)
criterion = TripletMarginLoss(margin=margin, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_loss = float('inf')
train_loss_history, val_loss_history = [], []

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0
    for anchor, positive, negative in train_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch:03d} | Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss <= best_loss:
        best_loss = avg_val_loss
        save_path = os.path.join(model_save_root, f"embedding_model_epoch{epoch:03d}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  ↳ New best model saved at {save_path}")

# ==================== Plot ====================
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Triplet Loss Over Epochs (Anchor: 50% raw + 50% augmented)")
plt.savefig(os.path.join(model_save_root, "loss_curve.png"))
plt.close()