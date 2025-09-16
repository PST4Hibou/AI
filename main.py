import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchaudio
from tqdm import tqdm

# -----------------------------
# 1. Define CRNN model
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, n_mels=64, rnn_hidden=128, rnn_layers=1, dropout=0.3):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d((2,2))

        self.rnn = nn.GRU(input_size=(n_mels // 8) * 128,
                          hidden_size=rnn_hidden,
                          num_layers=rnn_layers,
                          batch_first=True,
                          bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden * 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        b, c, m, t = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c * m)
        rnn_out, _ = self.rnn(x)
        out = torch.mean(rnn_out, dim=1)
        out = self.dropout(out)
        out = self.classifier(out)
        return torch.sigmoid(out)

# -----------------------------
# 2. Data preprocessing helpers
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=64
).to(device)
db_transform = torchaudio.transforms.AmplitudeToDB()

def preprocess_waveform(waveform, sr, target_len=16000):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.size(-1) < target_len:
        pad = target_len - waveform.size(-1)
        waveform = F.pad(waveform, (0, pad))
    elif waveform.size(-1) > target_len:
        start = torch.randint(0, waveform.size(-1) - target_len + 1, (1,)).item()
        waveform = waveform[:, start:start+target_len]
    return waveform

def collate_fn(batch):
    waves, labels = [], []
    for sample in batch:
        waveform = torch.tensor(sample["audio"]["array"]).unsqueeze(0)
        sr = sample["audio"]["sampling_rate"]
        waveform = preprocess_waveform(waveform, sr)
        waves.append(waveform)
        labels.append(sample["label"])
    waves = torch.stack(waves)
    labels = torch.tensor(labels, dtype=torch.float32)
    return waves, labels

# -----------------------------
# 3. Training loop
# -----------------------------

def train_model():
    dataset = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=2000, shuffle=True, collate_fn=collate_fn, num_workers=16, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=2000, shuffle=False, collate_fn=collate_fn, num_workers=16, pin_memory=True)

    model = CRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/10")
        model.train()
        total_loss = 0
        train_progress = tqdm(train_dl, desc=f"Training Epoch {epoch+1}", leave=False)
        for waves, labels in train_progress:
            waves, labels = waves.to(device), labels.to(device)
            specs = mel_transform(waves)
            specs_db = db_transform(specs)
            specs_db = (specs_db - specs_db.mean()) / (specs_db.std() + 1e-9)
            inputs = specs_db
            preds = model(inputs).squeeze(1)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_dl)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        val_progress = tqdm(val_dl, desc=f"Validating Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for waves, labels in val_progress:
                waves, labels = waves.to(device), labels.to(device)
                specs = mel_transform(waves)
                specs_db = db_transform(specs)
                specs_db = (specs_db - specs_db.mean()) / (specs_db.std() + 1e-9)
                inputs = specs_db
                preds = model(inputs).squeeze(1)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                preds_cls = (preds > 0.5).float()
                correct += (preds_cls == labels).sum().item()
                total += labels.size(0)
                val_progress.set_postfix(loss=loss.item())
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss/len(val_dl):.4f}, Val Acc={val_acc:.4f}")

    torch.save(model.state_dict(), "crnn_drone_detector_2.pt")
    print("Model saved: crnn_drone_detector_2.pt")

# Run training if executed directly
if __name__ == "__main__":
    train_model()
