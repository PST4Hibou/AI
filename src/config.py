import torch
import numpy as np
import random

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Audio processing parameters
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 64

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 24
EPOCHS = 10
LEARNING_RATE = 1e-3

# Model parameters
N_CLASSES = 2  # drone, other

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset parameters
TRAIN_TEST_SPLIT = 0.3
VALID_TEST_SPLIT = 0.5

# Model checkpoint path
MODEL_CHECKPOINT_PATH = None

print(f"Using device: {DEVICE}")
