"""Data preprocessing functions for audio data."""

from typing import Union, List, Tuple
import torch
import torchaudio
from torch import Tensor

from ..config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


class AudioPreprocessor:
    """Audio preprocessing class for converting audio to mel spectrograms."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, 
                 hop_length: int = HOP_LENGTH, n_mels: int = N_MELS):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
    
    def preprocess(self, item: Union[dict, torch.Tensor]) -> Tensor:
        """
        Preprocess audio data to mel spectrogram.
        
        Args:
            item: Either a dict with 'audio' key or a tensor
            
        Returns:
            Mel spectrogram in dB scale
        """
        if isinstance(item, dict):
            audio = item["audio"]
            waveform = torch.tensor(audio).float()
        else:
            waveform = item

        # Convert stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)

        return mel_db


# Global preprocessor instance
preprocessor = AudioPreprocessor()


def preprocess(item: Union[dict, torch.Tensor]) -> Tensor:
    """Convenience function using global preprocessor."""
    return preprocessor.preprocess(item)


def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for batching audio data.
    
    Args:
        batch: List of dictionaries containing audio data and labels
        
    Returns:
        Tuple of (batched_spectrograms, labels)
    """
    xs = []
    ys = []

    for b in batch:
        mel_db = preprocess(b)
        xs.append(mel_db)
        ys.append(b["label"])

    # Pad to max length in batch
    max_len = max(x.shape[-1] for x in xs)
    xs_padded = torch.zeros((len(xs), 1, N_MELS, max_len))
    for i, x in enumerate(xs):
        xs_padded[i, 0, :, :x.shape[-1]] = x

    return xs_padded, torch.tensor(ys)
