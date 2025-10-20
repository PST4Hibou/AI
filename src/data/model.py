import torch
from torch import nn

from src.models.model_item import ModelItem


def save_model(model_item: ModelItem, is_checkpoint: bool = False):
    torch.save(
        model_item.model.state_dict(),
        f"models/{"checkpoints/" if is_checkpoint else ""}{model_item.name.lower()}.pth",
    )


def load_model_from_file(model_item: ModelItem, is_checkpoint: bool = False):
    model_item.model.load_state_dict(
        torch.load(
            f"models/{"checkpoints/" if is_checkpoint else ""}{model_item.name.lower()}.pth"
        )
    )
