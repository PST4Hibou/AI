"""Inference pipeline orchestrator for the drone detection model."""

from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import pandas as pd
from datasets import Dataset

from ..config import DEVICE, MODEL_CHECKPOINT_PATH, N_CLASSES
from ..data.load_data import load_test_datasets, create_local_test_recordings
from ..models.registry import create_model
from ..models.model import load_model_checkpoint
from ..models.infer import InferenceEngine, infer_from_dataset


class InferencePipeline:
    """Complete inference pipeline for the drone detection model."""
    
    def __init__(self, model_name: str = "audiocnn2d",
                 checkpoint_path: str = None,
                 device: str = DEVICE, n_classes: int = N_CLASSES):
        """
        Initialize the inference pipeline.
        
        Args:
            model_name: Name of the model architecture to use
            checkpoint_path: Path to model checkpoint (if None, uses model-specific path)
            device: Device to run inference on
            n_classes: Number of classes in the model
        """
        self.model_name = model_name
        # Generate model-specific checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = f"models/{model_name}_best.pt"
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.n_classes = n_classes
        
        # Initialize components
        self.model = None
        self.inference_engine = None
        self.labels = None

        # Only tests the drone samples
        self.only_drone = False
    
    def load_model(self, labels: Optional[object] = None) -> None:
        """
        Load the trained model.
        
        Args:
            labels: Label encoder object
        """
        print(f"Loading {self.model_name} model...")
        self.model = create_model(self.model_name, n_classes=self.n_classes, device=self.device)
        self.model = load_model_checkpoint(self.model, self.checkpoint_path, self.device)
        self.labels = labels
        
        # Create inference engine
        self.inference_engine = InferenceEngine(self.model, self.device)
        print("Model loaded successfully.")
    
    def predict_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Predict on a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        if self.inference_engine is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        pred_label, confidence = self.inference_engine.predict_single(file_path, self.labels)
        
        return {
            'file_path': file_path,
            'predicted_label': pred_label,
            'confidence': confidence
        }
    
    def predict_batch_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Predict on a batch of audio files.
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            DataFrame with prediction results
        """
        if self.inference_engine is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = []
        for file_path in file_paths:
            try:
                result = self.predict_single_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'predicted_label': 'ERROR',
                    'confidence': 0.0
                })
        
        return pd.DataFrame(results)
    
    def evaluate_on_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Evaluate model on various test datasets.
        
        Returns:
            Dictionary containing evaluation results for each dataset
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = {}
        
        print("Loading test datasets...")
        try:
            ds_test_2_online = load_test_datasets()[0]
            
            results['online_test_2_drone'] = infer_from_dataset(
                self.model, ds_test_2_online, labels=self.labels,
                only_drone=self.only_drone, show_resume=False, show_accuracy=True,
                title="Online Test Dataset 2 "
            )
            

        except Exception as e:
            print(f"Error loading test datasets: {e}")
        
        return results
    
    def evaluate_on_custom_dataset(self, dataset: Union[Dataset, List], 
                                  dataset_name: str = "Custom Dataset",
                                  show_detailed_results: bool = False) -> pd.DataFrame:
        """
        Evaluate model on a custom dataset.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name for display purposes
            show_detailed_results: Whether to show detailed results
            
        Returns:
            DataFrame with evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Evaluating on {dataset_name}...")
        results = infer_from_dataset(
            self.model, dataset, labels=self.labels,
            only_drone=self.only_drone, show_resume=show_detailed_results,
            show_accuracy=True, title=dataset_name
        )
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all available test datasets.
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("Starting comprehensive evaluation...")
        
        # Evaluate on standard test datasets
        test_results = self.evaluate_on_test_datasets()
        
        # Calculate summary statistics
        summary = {}
        for dataset_name, results_df in test_results.items():
            if not results_df.empty:
                accuracy = results_df['correct'].mean()
                avg_confidence = results_df['confidence'].mean()
                summary[dataset_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'num_samples': len(results_df)
                }
        
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*60)
        for dataset_name, stats in summary.items():
            print(f"{dataset_name}:")
            print(f"  - Accuracy: {stats['accuracy']:.3f}")
            print(f"  - Avg Confidence: {stats['avg_confidence']:.3f}")
            print(f"  - Samples: {stats['num_samples']}")
        print("="*60)
        
        return {
            'detailed_results': test_results,
            'summary': summary
        }
    

def run_inference_pipeline(model_name: str = "audiocnn2d",
                          checkpoint_path: str = None,
                          device: str = DEVICE,
                          labels: Optional[object] = None) -> Dict[str, Any]:
    """
    Convenience function to run the complete inference pipeline.
    
    Args:
        model_name: Name of the model architecture to use
        checkpoint_path: Path to model checkpoint (if None, uses model-specific path)
        device: Device to run inference on
        labels: Label encoder object

    Returns:
        Dictionary containing inference results
    """
    pipeline = InferencePipeline(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Load model
    pipeline.load_model(labels)
    
    # Run comprehensive evaluation
    results = pipeline.run_comprehensive_evaluation()
    
    return results


def predict_on_files(file_paths: Union[str, List[str]], 
                    model_name: str = "audiocnn2d",
                    checkpoint_path: str = None,
                    device: str = DEVICE,
                    labels: Optional[object] = None) -> pd.DataFrame:
    """
    Simple function to predict on audio files.
    
    Args:
        file_paths: Single file path or list of file paths
        model_name: Name of the model architecture to use
        checkpoint_path: Path to model checkpoint (if None, uses model-specific path)
        device: Device to run inference on
        labels: Label encoder object
        
    Returns:
        DataFrame with prediction results
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    pipeline = InferencePipeline(
        model_name=model_name, 
        checkpoint_path=checkpoint_path, 
        device=device
    )
    pipeline.load_model(labels)
    
    return pipeline.predict_batch_files(file_paths)
