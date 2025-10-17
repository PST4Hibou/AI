"""Inference functions for the audio classification model."""

import torch
from torch import nn, Tensor
import torchaudio
import pandas as pd
from typing import Union, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

from ..config import DEVICE
from ..data.preprocess import preprocess
from .model import AudioCNN2D


def infer_from_waveform(model: nn.Module, waveform: Tensor, 
                       labels: Optional[object] = None) -> Tuple[str, float]:
    """
    Perform inference on a single waveform.
    
    Args:
        model: Trained model
        waveform: Audio waveform tensor
        labels: Label encoder object with int2str method
        
    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    mel_db = preprocess(waveform)
    mel_db = mel_db.unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(mel_db)
        pred_idx = outputs.argmax(dim=1).item()
        prob = torch.softmax(outputs, dim=1)[0, pred_idx].item()
        
        if labels is not None:
            pred_label = labels.int2str(pred_idx)
        else:
            pred_label = str(pred_idx)
    
    return pred_label, prob


def infer(model: nn.Module, path: Optional[str] = None, 
          audio_array: Optional[List] = None, audio_decoder: Optional[List] = None,
          labels: Optional[object] = None) -> Tuple[str, float]:
    """
    Perform inference on audio data from various sources.
    
    Args:
        model: Trained model
        path: Path to audio file
        audio_array: Audio data as array
        audio_decoder: Audio data from decoder
        labels: Label encoder object
        
    Returns:
        Tuple of (predicted_label, confidence_score)
    """
    # Load waveform from different sources
    if path:
        waveform, _ = torchaudio.load(path)
    elif audio_array is not None:
        waveform = torch.tensor(audio_array)
    elif audio_decoder is not None:
        waveform = torch.tensor(audio_decoder)
    else:
        raise ValueError("Must provide either path, audio_array, or audio_decoder")
    
    return infer_from_waveform(model, waveform, labels)


def infer_from_dataset(model: nn.Module, dataset: Union[Dataset, List], 
                      only_drone: bool = False, show_resume: bool = True, 
                      show_accuracy: bool = True, title: Optional[str] = None, 
                      highlight: bool = False, labels: Optional[object] = None) -> pd.DataFrame:
    """
    Perform inference on a dataset or list of audio files.
    
    Args:
        model: Trained model
        dataset: Dataset or list of (path, label) tuples
        only_drone: If True, only test drone samples
        show_resume: If True, print results summary
        show_accuracy: If True, calculate and show accuracy
        title: Title for the results display
        highlight: If True, show results with styling (for Jupyter)
        labels: Label encoder object
        
    Returns:
        DataFrame with inference results
    """
    results = []
    is_dataset_list = isinstance(dataset, list)
    
    # Set labels for list-type datasets
    if not is_dataset_list and labels is None:
        dataset_labels = dataset.features["label"]
    else:
        dataset_labels = labels
    print(only_drone)
    for item in tqdm(dataset, desc="Inference"):
        # Handle different dataset types
        if not is_dataset_list:
            # HuggingFace Dataset
            if only_drone and item["label"] != dataset_labels.str2int("drone"):
                continue
            pred_label, prob = infer(model, audio_decoder=item["audio"]["array"],
                                   labels=dataset_labels)
            true_label = dataset_labels.int2str(item["label"])
            filename = "N/A"
        else:
            # List of (path, label) tuples
            if only_drone and item[1] != "drone":
                continue
            pred_label, prob = infer(model, path=item[0], labels=dataset_labels)
            true_label = item[1]
            filename = str(Path(item[0]).name)

        results.append({
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(prob, 3),
            "correct": (pred_label == true_label),
            "filename": filename
        })

    df_results = pd.DataFrame(results)
    
    # Display results
    if title:
        print(title)
    
    if show_resume:
        print(df_results)
    
    if show_accuracy and len(df_results) > 0:
        accuracy = df_results["correct"].mean()
        dataset_type = "Drone" if only_drone else "Global"
        print(f"✅ {dataset_type} accuracy on dataset: {accuracy*100:.2f}%")
    
    if highlight:
        try:
            # This will work in Jupyter environments
            from IPython.display import display
            styled_df = (df_results.style
                        .background_gradient(subset=["confidence"], cmap="Blues")
                        .map(lambda v: "background-color:#aaffaa" if v else "background-color:#ffaaaa", 
                             subset=["correct"]))
            display(styled_df)
        except ImportError:
            print("Highlighting not available outside Jupyter environment")
    
    return df_results


class InferenceEngine:
    """Inference engine for batch processing and model management."""
    
    def __init__(self, model: AudioCNN2D, device: str = DEVICE):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_single(self, audio_input: Union[str, Tensor], 
                      labels: Optional[object] = None) -> Tuple[str, float]:
        """
        Predict on a single audio input.
        
        Args:
            audio_input: Path to audio file or tensor
            labels: Label encoder object
            
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if isinstance(audio_input, str):
            return infer(self.model, path=audio_input, labels=labels)
        else:
            return infer_from_waveform(self.model, audio_input, labels)
    
    def predict_batch(self, audio_inputs: List[Union[str, Tensor]], 
                     labels: Optional[object] = None) -> List[Tuple[str, float]]:
        """
        Predict on a batch of audio inputs.
        
        Args:
            audio_inputs: List of audio inputs (paths or tensors)
            labels: Label encoder object
            
        Returns:
            List of (predicted_label, confidence_score) tuples
        """
        results = []
        for audio_input in tqdm(audio_inputs, desc="Batch inference"):
            result = self.predict_single(audio_input, labels)
            results.append(result)
        return results
    
    def evaluate_dataset(self, dataset: Union[Dataset, List], 
                        labels: Optional[object] = None, **kwargs) -> pd.DataFrame:
        """
        Evaluate model performance on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            labels: Label encoder object
            **kwargs: Additional arguments for infer_from_dataset
            
        Returns:
            DataFrame with evaluation results
        """
        return infer_from_dataset(self.model, dataset, labels=labels, **kwargs)
