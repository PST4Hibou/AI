import logging

import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

from src.data.datasets import load_dataset, validate_model_dataset_compatibility
from datasets import DatasetDict

import src.models.AudioCNN2d as AudioCNN2d
from src.data.model import save_model, load_model_from_file
from src.models import select_model
from src.models.model_item import ModelItem
from src.settings import SETTINGS
from torch.utils.data import DataLoader
from typing import Callable

from src.utils.metrics import log_confusion_matrix


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
        self.model_item: ModelItem | None = None

        self.train_loader: DataLoader | None = None
        self.train_len = 0
        self.val_loader: DataLoader | None = None
        self.val_len = 0
        self.test_loader: DataLoader | None = None
        self.test_len = 0
        self.labels: list[str] = []

        if self.checkpoint_path is None:
            logging.debug("No checkpoint path provided. Using default path.")
            self.checkpoint_path = f"models/checkpoints/{self.model_name}.pt"

    def _create_loaders(self, ds: DatasetDict, collate_fn: Callable):
        self.train_loader = DataLoader(
            ds["train"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=True,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        self.train_len = len(ds["train"])
        self.valid_loader = DataLoader(
            ds["valid"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        self.val_len = len(ds["valid"])
        self.test_loader = DataLoader(
            ds["test"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        self.test_len = len(ds["test"])

    def _train(self):
        """
        Training the model
        """
        best_val_acc = 0

        for epoch in range(self.epochs):
            self.model_item.model.train()

            train_loss, train_acc = 0, 0

            for x, y in tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Train]"
            ):
                x, y = x.to(self.device), y.to(self.device)
                self.model_item.optimizer.zero_grad()
                out = self.model_item.model(x)
                loss = self.model_item.criterion(out, y)
                loss.backward()
                self.model_item.optimizer.step()

                train_loss += loss.item() * x.size(0)
                train_acc += self.model_item.metric_acc(out, y) * x.size(0)

            self.model_item.scheduler.step()
            train_loss /= self.train_len
            train_acc = train_acc / self.train_len

            # Validation
            self.model_item.model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for x, y in tqdm(
                    self.valid_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Valid]"
                ):
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model_item.model(x)
                    loss = self.model_item.criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    val_acc += self.model_item.metric_acc(out, y) * x.size(0)

            val_loss /= self.val_len
            val_acc = val_acc / self.val_len

            logging.info(
                f"Epoch {epoch + 1}/{self.epochs} | Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"Saving next best model.")
                # Save the checkpoints
                save_model(self.model_item, is_checkpoint=True)
                # Save the best model
                save_model(self.model_item, is_checkpoint=False)

    def _evaluate(self):
        load_model_from_file(self.model_item, is_checkpoint=False)
        self.model_item.model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Testing"):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model_item.model(x)
                preds = out.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        log_confusion_matrix(cm, self.labels)

    def run(self):
        ds, self.labels = load_dataset(self.dataset_key)

        logging.info(f"Loading model  {self.model_name}")

        self.model_item = select_model(self.model_name).get_model_item(
            labels=self.labels, device=self.device
        )
        
        # Validate that model and dataset have compatible labels
        validate_model_dataset_compatibility(self.labels, self.dataset_key)
        self.model_item.metric_acc.to(self.device)

        logging.debug("Loading loaders")
        self._create_loaders(ds, self.model_item.collate_fn)

        self._train()
        self._evaluate()
