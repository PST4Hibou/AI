"""
Data loading and preprocessing module for acoustic drone detection
"""

from dataclasses import dataclass
from typing import List, Callable
from src.data.preprocess import Preprocessor
from src.settings import SETTINGS


@dataclass
class KnownDataset:
    path: str
    split: str | None = None


known_datasets: dict[str, KnownDataset] = {
    "df_462700_2": KnownDataset(
        path="Usernameeeeee/df_462700_2",
        split="train",
    ),
    "drone_test": KnownDataset(
        path="Usernameeeeee/drone_test",
        split="test",
    ),
}
