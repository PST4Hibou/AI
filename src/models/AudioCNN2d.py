from torch import nn

from src.data import Preprocessor
from src.data.preprocess import CollateFn
from src.settings import SETTINGS


class Model(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


preprocessor = Preprocessor(
    [
        Preprocessor.mel_transform(
            sample_rate=SETTINGS.SAMPLE_RATE,
            n_fft=2048,
            hop_length=256,
            n_mels=64,
        ),
        Preprocessor.db_transform(),
    ]
)

collate_fn = (CollateFn(preprocessor).process_mel_and_pad,)
