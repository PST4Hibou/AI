import os, glob, random
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.metrics import classification_report
from main import CRNN, mel_transform, db_transform

# ✅ Load trained model
def load_model(model_path="crnn_drone_detector_2.pt", device="cuda"):
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

import soundfile as sf
def preprocess_file(filepath, target_len=16000):
    waveform, sr = sf.read(filepath, dtype="float32")  # returns np.array
    waveform = torch.tensor(waveform).T  # shape (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.size(-1) < target_len:
        pad = target_len - waveform.size(-1)
        waveform = F.pad(waveform, (0, pad))
    elif waveform.size(-1) > target_len:
        waveform = waveform[:, :target_len]
    return waveform


def evaluate_realworld(model, base_path, max_per_class=100, device="cuda"):
    # Collect real-world test files
    drone_files = glob.glob(f"{base_path}/drone/*.wav")
    no_drone_files = glob.glob(f"{base_path}/no drone/*.wav")
    print(f"Found {len(drone_files)} drone files and {len(no_drone_files)} no-drone files")

    random.seed(42)
    if len(drone_files) > max_per_class:
        drone_files = random.sample(drone_files, max_per_class)
    if len(no_drone_files) > max_per_class:
        no_drone_files = random.sample(no_drone_files, max_per_class)

    print(f"Selected {len(drone_files)} drone files and {len(no_drone_files)} no-drone files for testing")

    files = drone_files + no_drone_files
    labels = [1]*len(drone_files) + [0]*len(no_drone_files)

    preds = []
    with torch.no_grad():
        for filepath in files:
            waveform = preprocess_file(filepath).to(device)
            specs = mel_transform(waveform.to(device))  # (1, n_mels, time)
            specs_db = db_transform(specs)
            specs_db = (specs_db - specs_db.mean()) / (specs_db.std() + 1e-9)
            specs_db = specs_db.unsqueeze(0)  # add batch → (1, 1, n_mels, time)
            output = model(specs_db).squeeze().item()

            preds.append(1 if output > 0.5 else 0)

    print(classification_report(labels, preds, target_names=["No Drone", "Drone"]))

# Example usage
if __name__ == "__main__":
    base_path = "/home/pierre/Documents/Projects/DroneDetection/data/Recorded Audios/Real World Testing"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("crnn_drone_detector.pt", device)
    evaluate_realworld(model, base_path, max_per_class=50, device=device)
