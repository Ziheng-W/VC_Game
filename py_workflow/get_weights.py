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
path_speech_model   = "./model_simulation_validation/optimized_model/comp1_noise_speech/fold5_best.pth"
path_xiaoxin_model  = "./model_simulation_validation/optimized_model/comp2_xiaoxin_others/embedding_model_epoch095.pth"
path_xiaoyuan_model = "./model_simulation_validation/optimized_model/comp3_xiaoyuan_others/embedding_model_epoch083.pth"

gallery_xiaoxin  = "./model_simulation_validation/model_test/gallery/XiaoXin"     # comp2 gallery
gallery_xiaoyuan = "./model_simulation_validation/model_test/gallery/XiaoYuan"     # comp3 gallery

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
model1.load_state_dict(torch.load(path_speech_model, map_location=device, weights_only=True))
model1.eval()

model_xin = MFCCEmbeddingNet().to(device)
model_xin.load_state_dict(torch.load(path_xiaoxin_model, map_location=device, weights_only=True))
model_xin.eval()

model_yuan = MFCCEmbeddingNet().to(device)
model_yuan.load_state_dict(torch.load(path_xiaoyuan_model, map_location=device, weights_only=True))
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

def export_model_parameters(model, output_path="model_params.h"):
    with open(output_path, "w") as f:
        macro_guard = os.path.basename(output_path).replace(".", "_").upper()
        f.write(f"#ifndef __{macro_guard}__\n")
        f.write(f"#define __{macro_guard}__\n\n")

        layer_id = 0

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                layer_id += 1
                prefix = f"layer{layer_id}_{module.__class__.__name__}"

                param_dict = {}

                # 卷积和线性层的参数
                for pname, p in module.named_parameters(recurse=False):
                    param_dict[pname] = p.detach().cpu().numpy()

                # BatchNorm 额外的 running stats
                if isinstance(module, torch.nn.BatchNorm2d):
                    param_dict["running_mean"] = module.running_mean.cpu().numpy()
                    param_dict["running_var"] = module.running_var.cpu().numpy()

                # 写出每个参数
                for key, val in param_dict.items():
                    arr = val.flatten()
                    dim = arr.size
                    varname = f"{prefix}_{key}"
                    f.write(f"float {varname}_yuan[{dim}] = {{\n    ")
                    for i, v in enumerate(arr):
                        f.write(f"{v:.6f}")
                        if i != dim - 1:
                            f.write(", ")
                        if (i + 1) % 512 == 0:
                            f.write("\n    ")
                    f.write("\n};\n\n")

        f.write(f"#endif // __{macro_guard}__\n")
export_model_parameters(model1, "yuan.h")