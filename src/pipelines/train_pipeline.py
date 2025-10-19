import logging
from src.data.datasets import load_dataset
from datasets import DatasetDict

import src.models.AudioCNN2d as AudioCNN2d
from src.settings import SETTINGS
from torch.utils.data import DataLoader
from typing import Callable


class TrainingPipeline:
    def __init__(
        self,
        model_name: str,
        dataset_key: str,
        epochs: int,
        checkpoint_path: str,
        device: str,
        resume_from_checkpoint: bool,
        evaluate_after_training: bool,
    ):
        self.model_name = model_name
        self.dataset_key = dataset_key
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.resume_from_checkpoint = resume_from_checkpoint
        self.evaluate_after_training = evaluate_after_training

        if self.checkpoint_path is None:
            logging.debug("No checkpoint path provided. Using default path.")
            self.checkpoint_path = f"models/checkpoints/{self.model_name}.pt"

    def _get_loaders(self, ds: DatasetDict, collate_fn: Callable):
        train_loader = DataLoader(
            ds["train"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=True,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            ds["valid"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            ds["test"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        return train_loader, valid_loader, test_loader

    def run(self):
        ds, labels = load_dataset(self.dataset_key)

        # TODO: Autoselect of model
        selected_model = AudioCNN2d

        logging.info(f"Loading model  {self.model_name}")
        logging.debug(f"Loading model: {model}")
        model = selected_model.Model(n_classes=len(labels))

        logging.debug("Loading loaders")
        train_loader, valid_loader, test_loader = self._get_loaders(
            ds, selected_model.collate_fn
        )
