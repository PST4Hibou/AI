from datasets import load_dataset as load_dataset_from_huggingface, DatasetDict
from src.data import known_datasets
from src.settings import SETTINGS
import logging


def load_dataset(dataset_key: str):
    if dataset_key not in known_datasets.keys():
        raise ValueError(
            "Dataset not found, please check your datasets and dataset key"
        )
    known_dataset = known_datasets[dataset_key]

    ds = load_dataset_from_huggingface(known_dataset.path)
    labels = ds[known_dataset.split].features["label"].names
    if known_dataset.split == "train":
        ds_split = ds["train"].train_test_split(
            test_size=SETTINGS.TRAIN_TEST_SPLIT, seed=SETTINGS.SEED
        )
        test_and_valid = ds_split["test"].train_test_split(
            test_size=SETTINGS.VALID_TEST_SPLIT, seed=SETTINGS.SEED
        )

        ds = DatasetDict(
            {
                "train": ds_split["train"],
                "valid": test_and_valid["train"],
                "test": test_and_valid["test"],
            }
        )

    logging.debug(ds)
    logging.debug(f"Label names: {labels}")

    return ds, labels
