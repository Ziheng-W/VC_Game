import torch, librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import pandas as pd

# ========= CONFIG =========
n_mfcc = 20
max_len = 100
threshold = 0.75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
path_speech_model   = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp1_noise_speech/fold5_best.pth"
path_xiaoxin_model  = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp2_others_XinYuan_mining/embedding_model_epoch095.pth"
path_xiaoyuan_model = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp3_others_XinYuan_mining/embedding_model_epoch083.pth"

gallery_xiaoxin  = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/gallery/XiaoXin"     # comp2 gallery
gallery_xiaoyuan = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/gallery/XiaoYuan"     # comp3 gallery

# ========= 数据处理函数 =========
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

# ========= 模型定义 =========
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

# ========= 加载模型 =========
model1 = MFCCBinaryCNN().to(device)
model1.load_state_dict(torch.load(path_speech_model, map_location=device))
model1.eval()

model_xin = MFCCEmbeddingNet().to(device)
model_xin.load_state_dict(torch.load(path_xiaoxin_model, map_location=device))
model_xin.eval()

model_yuan = MFCCEmbeddingNet().to(device)
model_yuan.load_state_dict(torch.load(path_xiaoyuan_model, map_location=device))
model_yuan.eval()

# ========= 加载 gallery =========
def load_gallery_embeddings(folder, model):
    gallery_embeddings = []
    with torch.no_grad():
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".wav"):
                    x = wav_to_tensor(os.path.join(root, f)).to(device)
                    emb = model(x).cpu().numpy().squeeze()
                    gallery_embeddings.append(emb)
    return gallery_embeddings #返回Xiaoxin和Xiaoyuan的特征向量

gallery_emb_xin = load_gallery_embeddings(gallery_xiaoxin, model_xin)
gallery_emb_yuan = load_gallery_embeddings(gallery_xiaoyuan, model_yuan)

# ========= 推理函数 =========
def classify_audio(audio_path):
    x = wav_to_tensor(audio_path).to(device)

    # Step 1: speech vs noise
    with torch.no_grad():
        out1 = model1(x)
        prob_speech = F.softmax(out1, dim=1)[0, 1].item()
        if prob_speech < threshold:
            return "Noise"

    # Step 2: xiaoxin vs others
    with torch.no_grad():
        emb = model_xin(x).cpu().numpy().squeeze()
        sims = [cos_sim(emb, g) for g in gallery_emb_xin]
        if np.mean(sims) > threshold:
            return "XiaoXin"

    # Step 3: xiaoyuan vs others
    with torch.no_grad():
        emb = model_yuan(x).cpu().numpy().squeeze()
        sims = [cos_sim(emb, g) for g in gallery_emb_yuan]
        if np.mean(sims) > threshold:
            return "XiaoYuan"

    return "Other"

# ========= 测试 =========
if __name__ == "__main__":


    test_dir = Path("/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/test_set")

    results = []  # 用于存储所有音频的识别结果

    for wav_path in test_dir.rglob("*.wav"):   # 支持递归地搜索所有子文件夹
        result = classify_audio(str(wav_path))
        print(f"{wav_path.name} → 分类结果: {result}")
        results.append((wav_path.name, result))

    df = pd.DataFrame(results, columns=["Filename", "Prediction"])
    df.to_csv("/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/model_test/classification_results.csv", index=False)
    print("\n✅ 所有结果已保存到 classification_results.csv")

