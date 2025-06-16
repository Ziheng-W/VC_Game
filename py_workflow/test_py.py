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
path_speech_model   = "./py_workflow/model_simulation_validation/optimized_model/comp1_noise_speech/fold5_best.pth"
path_xiaoxin_model  = "./py_workflow/model_simulation_validation/optimized_model/comp2_xiaoxin_others/embedding_model_epoch095.pth"
path_xiaoyuan_model = "./py_workflow/model_simulation_validation/optimized_model/comp3_xiaoyuan_others/embedding_model_epoch083.pth"

gallery_xiaoxin  = "./py_workflow/model_simulation_validation/model_test/gallery/XiaoXin"     # comp2 gallery
gallery_xiaoyuan = "./py_workflow/model_simulation_validation/model_test/gallery/XiaoYuan"     # comp3 gallery

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
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

# ========= 加载模型 =========
model1 = MFCCBinaryCNN().to(device)
model1.load_state_dict(torch.load(path_speech_model, map_location=device, weights_only=True))
model1.eval()

# 推理输入
input_tensor = torch.ones((1, 1, 20, 100), dtype=torch.float32).to(device)
# input_tensor = torch.zeros((1, 1, 20, 100), dtype=torch.float32).to(device)
out1 = model1(input_tensor)
print(out1)

# intermediate_output = {}

# def hook_fn(module, input, output):
#     intermediate_output["inter"] = output.detach()

# # 注册 hook 到第二个 MaxPool2d（在 features 中是第 8 层，索引为 7）
# model1.features[1].register_forward_hook(hook_fn)

# print("MaxPool1 输出形状:", intermediate_output["inter"].shape)
# # print("输出示例:", intermediate_output["inter"][0, 0, 5:, :50])  # 打印前 5×5 patch
# # 假设 intermediate_output["inter"] 已经存储了你 hook 的那层输出
# sum_val = intermediate_output["inter"].sum().item()
# print("该层输出所有元素的和为:", sum_val)
