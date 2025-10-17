"""Data loading and dataset management functions."""

import os
import librosa
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader

from ..config import SEED, TRAIN_TEST_SPLIT, VALID_TEST_SPLIT, BATCH_SIZE, NUM_WORKERS
from .preprocess import collate_fn


def load_hibou_audio_files(directory, sr=44100, n_mfcc=13):
    """
    Load audio files from hibou directory structure.
    
    Args:
        directory: Path to directory containing class subdirectories
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        features, labels as numpy arrays
    """
    features = []
    labels = []
    class_names = sorted(os.listdir(directory))

    for class_name in class_names:
        for filename in os.listdir(os.path.join(directory, class_name)):
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, class_name, filename)
                audio, _ = librosa.load(filepath, sr=sr)
                features.append(audio)
                labels.append(class_name)

    return np.array(features), np.array(labels)


def load_huggingface_dataset(dataset_name: str = "Usernameeeeee/df_462700_2") -> DatasetDict:
    """
    Load and split the HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        
    Returns:
        DatasetDict with train, valid, and test splits
    """
    # Load the main dataset
    ds = load_dataset(dataset_name)
    
    # Split the dataset
    ds_split = ds["train"].train_test_split(test_size=TRAIN_TEST_SPLIT, seed=SEED)
    test_and_valid = ds_split["test"].train_test_split(test_size=VALID_TEST_SPLIT, seed=SEED)

    ds_final = DatasetDict({
        "train": ds_split["train"],
        "valid": test_and_valid["train"],
        "test": test_and_valid["test"],
    })
    
    print("Dataset splits:", {k: v.shape for k, v in ds_final.items()})
    return ds_final


def create_data_loaders(dataset: DatasetDict, batch_size: int = BATCH_SIZE, 
                       num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        dataset: DatasetDict containing train, valid, and test splits
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    train_loader = DataLoader(
        dataset["train"], 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        dataset["valid"], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        dataset["test"], 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, test_loader


def load_test_datasets() -> Tuple[Dataset]:
    """
    Load additional test datasets for evaluation.
    """
    ds_test = load_dataset("Usernameeeeee/drone_test", split="test")
    
    return  (ds_test, )


def create_local_test_recordings(base_dir: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Create a list of local test recordings with their labels.
    
    Args:
        base_dir: Base directory containing class subdirectories
        
    Returns:
        List of (file_path, class_label) tuples
    """
    base_dir = Path(base_dir)
    recordings = []
    class_folders = ["drone", "other"]
    
    for class_folder in class_folders:
        class_path = base_dir / class_folder
        if class_path.exists():
            for file_path in class_path.glob("*.wav"):
                recordings.append((str(file_path), class_folder))
    
    return recordings