from dataclasses import dataclass
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler

from src.data.preprocess import Preprocessor
from typing import Callable


@dataclass
class ModelItem:
    name: str
    model: nn.Module
    preprocessor: Preprocessor
    collate_fn: Callable
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: LRScheduler
    metric_acc: Callable
