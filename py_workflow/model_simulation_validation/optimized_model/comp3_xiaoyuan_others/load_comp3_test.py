import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
model_path = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp3_others_XinYuan_mining/embedding_model_epoch079.pth"
data_root  = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/speakers"
n_mfcc = 20
max_len = 100
embed_dim = 64
threshold = 0.75

# ==================== Model ====================
class MFCCEmbeddingNet(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)

# ==================== Helper Functions ====================
def wav_to_tensor(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ==================== Evaluation Dataset ====================
label_map = {'XiaoYuan': 1, 'XiaoXin': 0, 'other_speaker': 0}
all_data = []
for label_name, label in label_map.items():
    folder = os.path.join(data_root, label_name)
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                all_data.append((path, label))

# ==================== Load Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MFCCEmbeddingNet(embed_dim=embed_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==================== Embedding & Prediction ====================
X, y = [], []
with torch.no_grad():
    for path, label in all_data:
        x = wav_to_tensor(path).to(device)
        emb = model(x).cpu().numpy().squeeze()
        X.append(emb)
        y.append(label)

X, y = np.stack(X), np.array(y)
gallery = X[y == 1]
preds = []
for v in X:
    sims = [cos_sim(v, g) for g in gallery]
    pred = int(np.mean(sims) > threshold)
    preds.append(pred)

# ==================== Metrics ====================
acc = accuracy_score(y, preds)
cm = confusion_matrix(y, preds)
print(f"\n🎯 XiaoYuan Recognition Accuracy: {acc*100:.2f}%")
ConfusionMatrixDisplay(cm, display_labels=["Others", "XiaoYuan"]).plot()
plt.savefig(os.path.join(os.path.dirname(model_path), "confusion_matrix_eval79.png"))
plt.close()
print("Confusion matrix saved.")
