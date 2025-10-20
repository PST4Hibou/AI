import logging
from torch.utils.data import DataLoader
import torch

from src.data.datasets import load_dataset, validate_model_dataset_compatibility
from src.models import select_model
from src.models.model_item import ModelItem
from src.settings import SETTINGS
from datasets import DatasetDict
from typing import Callable
from sklearn.metrics import confusion_matrix

from src.utils.metrics import log_confusion_matrix
from src.data.model import load_model_from_file


class InferPipeline:

    def __init__(self, model_name, dataset_key: str, checkpoint_path, device="cpu"):
        self.model_name = model_name
        self.dataset_key = dataset_key
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model_item: ModelItem | None = None

        self.test_loader: DataLoader | None = None
        self.test_len = 0
        self.labels: list[str] = []

        if self.checkpoint_path is None:
            logging.debug("No checkpoint path provided. Using default path.")
            self.checkpoint_path = f"models/{self.model_name}.pth"

    def _create_loaders(self, ds: DatasetDict, collate_fn: Callable):
        self.test_loader = DataLoader(
            ds["test"],
            batch_size=SETTINGS.BATCH_SIZE,
            shuffle=False,
            num_workers=SETTINGS.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        self.test_len = len(ds["test"])

    def _infer(self):
        self.model_item.model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model_item.model(x)
                preds = out.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        log_confusion_matrix(cm, self.labels)

    def _load_model(self):
        logging.info(f"Loading checkpoint from {self.checkpoint_path}")
        try:
            self.model_item.model.load_state_dict(torch.load(self.checkpoint_path))
            logging.info("Model checkpoint loaded successfully!")
        except FileNotFoundError:
            logging.error(f"Checkpoint not found at {self.checkpoint_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            raise

    def run(self):
        ds, self.labels = load_dataset(self.dataset_key)

        logging.info(f"Loading model  {self.model_name}")
        self.model_item = select_model(self.model_name).get_model_item(
            labels=self.labels, device=self.device
        )
        
        # Validate that model and dataset have compatible labels
        validate_model_dataset_compatibility(self.labels, self.dataset_key)

        self._load_model()
        self._create_loaders(ds, self.model_item.collate_fn)
        self._infer()
