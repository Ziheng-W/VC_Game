import torch, torchaudio, librosa, numpy as np
import torch.nn as nn
from torch.quantization import quantize_dynamic
import torch.nn.utils.prune as prune
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class MFCCBinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                              # (32,10,50)

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                              # (64, 5,25)

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                  # (128,1,1)
        )
        self.classifier = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.features(x).flatten(1)                   # (B,128)
        return self.classifier(x)

def wav_to_tensor(path, n_mfcc=20, max_len=100, device="cpu"):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = F.pad(torch.tensor(mfcc), (0, max_len - mfcc.shape[1]))
    else:
        start = (mfcc.shape[1] - max_len) // 2
        mfcc = torch.tensor(mfcc[:, start:start + max_len])
    return mfcc.unsqueeze(0).unsqueeze(0).float().to(device)


if __name__ == "__main__":
    trained_model_path = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp1_noise_speech/fold5_best.pth"
    INPUT_PATH = "/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/Dataset"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MFCCBinaryCNN().to(device)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    label_map = {'noise': 0, 'speakers_pcphone': 1}

    for label_name, label in label_map.items():
        folder = Path(INPUT_PATH) / label_name
        for p in folder.rglob("*.wav"):
            x = wav_to_tensor(p, device=device)
            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, 1)[0].cpu().numpy()
                pred   = int((probs[1] > 0.5))
            y_true.append(label)
            y_pred.append(pred)
            print(f"{p.name} → Predict: {pred}  (noise={probs[0]:.2f}, speech={probs[1]:.2f})")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    ConfusionMatrixDisplay(cm, display_labels=["noise", "speech"]).plot()
    plt.savefig("/Users/tracytracy/Desktop/VeriHealthi_QEMU_SDK.202505_preliminary/comp1_noise_speech/binary_confusion_matrix5.png")
    print("\n✅ Confusion matrix saved as binary_confusion_matrix.png")
