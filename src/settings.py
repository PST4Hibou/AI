from dataclasses import dataclass
import torch
import numpy as np
import random
from src.arguments import args


@dataclass
class Settings:
    SEED: int = 42
    SAMPLE_RATE: int = 16000
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 16
    EPOCHS: int = 10
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_TEST_SPLIT: int = 0.3
    VALID_TEST_SPLIT: int = 0.5
    MODEL_CHECKPOINT_PATH: str | None = None
    LOG_LEVEL: str = "DEBUG"


SETTINGS = Settings()

torch.manual_seed(SETTINGS.SEED)
np.random.seed(SETTINGS.SEED)
random.seed(SETTINGS.SEED)
