from typing import Union, List, Tuple, Callable, TypeVar
import torch
import torchaudio
from torch import Tensor

from src.settings import SETTINGS


class Preprocessor:
    """
    A utility class for handling preprocessing operations used in audio
    transformation and data processing pipelines.

    :type sequences: List[Callable] A list of callable preprocessing transformations to
        sequentially apply to the data.
    """

    def __init__(
        self,
        sequences: List[Callable],
    ):
        pass

    @classmethod
    def mel_transform(
        cls,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    @classmethod
    def db_transform(cls):
        return torchaudio.transforms.AmplitudeToDB()

    def process(self, item: dict | torch.Tensor):
        """

        :param item:
        :return:
        """
        data: torch.Tensor = item
        # When an item is a dict, only take the audio
        if isinstance(item, dict):
            audio = item["audio"]
            data = torch.tensor(audio).float()

        # Convert stereo to mono
        if data.ndim > 1:
            waveform = data.mean(dim=0)

        for sequence in self.sequences:
            data = sequence(data)

        return data


class CollateFn:
    """
    CollateFn is a class used to call differents collate functions to be used in PyTorch dataloaders

    :ivar processor: The preprocessor instance is used to preprocess input data.
                     Can be of type ``Preprocessor`` or ``None``.
    """

    def __init__(self, processor: Preprocessor | None):
        self.processor = processor

    def process_mel_and_pad(
        self, batch: List[dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []

        for b in batch:
            mel_db = self.processor.process(b)
            xs.append(mel_db)
            ys.append(b["label"])

        # Pad to max length in batch
        max_len = max(x.shape[-1] for x in xs)
        xs_padded = torch.zeros((len(xs), 1, 64, max_len))
        for i, x in enumerate(xs):
            xs_padded[i, 0, :, : x.shape[-1]] = x

        return xs_padded, torch.tensor(ys)


#
#
# class AudioPreprocessor:
#     """Audio preprocessing class for converting audio to mel spectrograms."""
#
#     def __init__(
#         self,
#         sample_rate: int = SETTINGS.SAMPLE_RATE,
#         n_fft: int = SETTINGS.N_FFT,
#         hop_length: int = SETTINGS.HOP_LENGTH,
#         n_mels: int = SETTINGS.N_MELS,
#     ):
#         self.sample_rate = sample_rate
#         self.mel_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             n_mels=n_mels,
#         )
#         self.db_transform = torchaudio.transforms.AmplitudeToDB()
#
#     def preprocess(self, item: Union[dict, torch.Tensor]) -> Tensor:
#         """
#         Preprocess audio data to mel spectrogram.
#
#         Args:
#             item: Either a dict with 'audio' key or a tensor
#
#         Returns:
#             Mel spectrogram in dB scale
#         """
#         if isinstance(item, dict):
#             audio = item["audio"]
#             waveform = torch.tensor(audio).float()
#         else:
#             waveform = item
#
#         # Convert stereo → mono
#         if waveform.ndim > 1:
#             waveform = waveform.mean(dim=0)
#
#         mel = self.mel_transform(waveform)
#         mel_db = self.db_transform(mel)
#
#         return mel_db
#
#
# # Global preprocessor instance
# preprocessor = AudioPreprocessor()
#
#
# def preprocess(item: Union[dict, torch.Tensor]) -> Tensor:
#     """Convenience function using global preprocessor."""
#     return preprocessor.preprocess(item)
#
#
# def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Collate function for batching audio data.
#
#     Args:
#         batch: List of dictionaries containing audio data and labels
#
#     Returns:
#         Tuple of (batched_spectrograms, labels)
#     """
#     xs = []
#     ys = []
#
#     for b in batch:
#         mel_db = preprocess(b)
#         xs.append(mel_db)
#         ys.append(b["label"])
#
#     # Pad to max length in batch
#     max_len = max(x.shape[-1] for x in xs)
#     xs_padded = torch.zeros((len(xs), 1, N_MELS, max_len))
#     for i, x in enumerate(xs):
#         xs_padded[i, 0, :, : x.shape[-1]] = x
#
#     return xs_padded, torch.tensor(ys)
