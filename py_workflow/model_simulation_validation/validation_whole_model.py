import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
import librosa

# ======== CONFIG ========

LABELS = ["noise", "XiaoXin", "XiaoYuan", "other_speaker"]
n_mfcc = 20
max_len = 100
threshold = 0.75
embed_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



test_root = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/test_set"

path_speech_model   = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp1_noise_speech/fold5_best.pth"
path_xiaoxin_model  = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp2_others_XinYuan_mining/embedding_model_epoch095.pth"
path_xiaoyuan_model = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp3_others_XinYuan_mining/embedding_model_epoch083.pth"

gallery_xiaoxin  = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/gallery/XiaoXin"     # comp2 gallery
gallery_xiaoyuan = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/gallery/XiaoYuan"     # comp3 gallery

# ======== 网络结构 ========
class MFCCBinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

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

# ======== 工具函数 ========
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

def load_gallery_embeddings(folder, model):
    embeddings = []
    with torch.no_grad():
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".wav"):
                    x = wav_to_tensor(os.path.join(root, f)).to(device)
                    emb = model(x).cpu().numpy().squeeze()
                    embeddings.append(emb)
    return embeddings

# ======== 加载模型 ========
model1 = MFCCBinaryCNN().to(device)
model1.load_state_dict(torch.load(path_speech_model, map_location=device))
model1.eval()

model_xin = MFCCEmbeddingNet(embed_dim=embed_dim).to(device)
model_xin.load_state_dict(torch.load(path_xiaoxin_model, map_location=device))
model_xin.eval()

model_yuan = MFCCEmbeddingNet(embed_dim=embed_dim).to(device)
model_yuan.load_state_dict(torch.load(path_xiaoyuan_model, map_location=device))
model_yuan.eval()

gallery_emb_xin = load_gallery_embeddings(gallery_xiaoxin, model_xin)
gallery_emb_yuan = load_gallery_embeddings(gallery_xiaoyuan, model_yuan)

# ======== 推理函数 ========
def classify_audio(audio_path):
    x = wav_to_tensor(audio_path).to(device)
    with torch.no_grad():
        out1 = model1(x)
        prob_speech = F.softmax(out1, dim=1)[0, 1].item()
        if prob_speech < threshold:
            return "noise"

        emb = model_xin(x).cpu().numpy().squeeze()
        if np.mean([cos_sim(emb, g) for g in gallery_emb_xin]) > threshold:
            return "XiaoXin"

        emb = model_yuan(x).cpu().numpy().squeeze()
        if np.mean([cos_sim(emb, g) for g in gallery_emb_yuan]) > threshold:
            return "XiaoYuan"

    return "other_speaker"

# ======== 评估函数 ========
def evaluate_on_folder(test_root):
    y_true, y_pred = [], []
    test_dir = Path(test_root)

    for true_label in LABELS:
        label_folder = test_dir / true_label
        if not label_folder.exists():
            continue
        for wav_path in label_folder.rglob("*.wav"):
            pred = classify_audio(str(wav_path))
            y_true.append(true_label)
            y_pred.append(pred)
            print(f"{wav_path.name} | True: {true_label} | Pred: {pred}")
    return y_true, y_pred

# ======== 主程序入口 ========
if __name__ == "__main__":
    y_true, y_pred = evaluate_on_folder(test_root)

    # 输出准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ 准确率 Accuracy: {acc*100:.2f}%")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/confusion_matrix_final.png")
    print("✅ 混淆矩阵保存为 confusion_matrix_final.png")

    # 保存预测结果
    pd.DataFrame({"True": y_true, "Pred": y_pred}).to_csv("/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/prediction_log.csv", index=False)
    print("📄 预测结果保存为 prediction_log.csv")
