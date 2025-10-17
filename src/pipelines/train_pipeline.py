"""Training pipeline orchestrator for the drone detection model."""

from typing import Optional, Dict, Any
from pathlib import Path

from ..config import EPOCHS, DEVICE, MODEL_CHECKPOINT_PATH
from ..data.load_data import load_huggingface_dataset, create_data_loaders
from ..models.registry import create_model
from ..models.model import load_model_checkpoint
from ..models.train import train_model
from ..utils.metrics import MetricsTracker, evaluate_model_on_test_set


class TrainingPipeline:
    """Complete training pipeline for the drone detection model."""
    
    def __init__(self, model_name: str = "audiocnn2d",
                 dataset_name: str = "Usernameeeeee/df_462700_2",
                 checkpoint_path: str = None,
                 device: str = DEVICE):
        """
        Initialize the training pipeline.
        
        Args:
            model_name: Name of the model architecture to use
            dataset_name: Name of the HuggingFace dataset
            checkpoint_path: Path to save/load model checkpoints (if None, uses model-specific path)
            device: Device to train on
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        # Generate model-specific checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = f"models/{model_name}_best.pt"
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.metrics_tracker = MetricsTracker()
        
        # Initialize components
        self.dataset = None
        self.data_loaders = None
        self.model = None
        self.labels = None
    
    def load_data(self) -> None:
        """Load and prepare the dataset."""
        print("Loading dataset...")
        self.dataset = load_huggingface_dataset(self.dataset_name)
        self.labels = self.dataset["train"].features["label"]
        print(f"Label names: {self.labels.names}")
        
        print("Creating data loaders...")
        self.data_loaders = create_data_loaders(self.dataset)
        print("Data loaders created successfully.")
    
    def create_model(self) -> None:
        """Create the model."""
        print(f"Creating {self.model_name} model...")
        n_classes = len(self.labels.names) if self.labels else 2
        self.model = create_model(self.model_name, n_classes=n_classes, device=self.device)
        print("Model created successfully.")
    
    def train(self, epochs: int = EPOCHS, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if self.data_loaders is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if resume_from_checkpoint:
            print(f"Attempting to load checkpoint from {self.checkpoint_path}")
            self.model = load_model_checkpoint(self.model, self.checkpoint_path, self.device)
        
        print(f"Starting training for {epochs} epochs...")
        train_loader, valid_loader, _ = self.data_loaders
        
        history = train_model(
            model=self.model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=epochs,
            device=self.device,
            checkpoint_path=self.checkpoint_path
        )
        
        # Update metrics tracker
        for i in range(len(history['train_loss'])):
            self.metrics_tracker.update(
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i]
            )
        
        return history
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Test evaluation results
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if self.data_loaders is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Loading best model checkpoint...")
        self.model = load_model_checkpoint(self.model, self.checkpoint_path, self.device)
        
        print("Evaluating on test set...")
        _, _, test_loader = self.data_loaders
        target_names = self.labels.names if self.labels else None
        
        test_results = evaluate_model_on_test_set(
            model=self.model,
            test_loader=test_loader,
            target_names=target_names,
            device=self.device
        )
        
        self.metrics_tracker.set_test_results(test_results)
        return test_results
    
    def run_full_pipeline(self, epochs: int = EPOCHS, 
                         resume_from_checkpoint: bool = False,
                         evaluate_after_training: bool = True) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            epochs: Number of epochs to train
            resume_from_checkpoint: Whether to resume from existing checkpoint
            evaluate_after_training: Whether to evaluate on test set after training
            
        Returns:
            Dictionary containing training history and test results
        """
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create model
        self.create_model()
        
        # Step 3: Train model
        history = self.train(epochs, resume_from_checkpoint)
        
        # Step 4: Evaluate model (optional)
        test_results = None
        if evaluate_after_training:
            test_results = self.evaluate()
        
        # Step 5: Print summary
        self.metrics_tracker.print_summary()
        
        # Step 6: Plot training history
        self.metrics_tracker.plot_history()
        
        return {
            'history': history,
            'test_results': test_results,
            'model': self.model,
            'labels': self.labels,
            'metrics_tracker': self.metrics_tracker
        }


def run_training_pipeline(model_name: str = "audiocnn2d",
                         dataset_name: str = "Usernameeeeee/df_462700_2",
                         epochs: int = EPOCHS,
                         checkpoint_path: str = None,
                         device: str = DEVICE,
                         resume_from_checkpoint: bool = False,
                         evaluate_after_training: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run the complete training pipeline.
    
    Args:
        model_name: Name of the model architecture to use
        dataset_name: Name of the HuggingFace dataset
        epochs: Number of epochs to train
        checkpoint_path: Path to save/load model checkpoints (if None, uses model-specific path)
        device: Device to train on
        resume_from_checkpoint: Whether to resume from existing checkpoint
        evaluate_after_training: Whether to evaluate on test set after training

    Returns:
        Dictionary containing pipeline results
    """
    pipeline = TrainingPipeline(
        model_name=model_name,
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    results = pipeline.run_full_pipeline(
        epochs=epochs,
        resume_from_checkpoint=resume_from_checkpoint,
        evaluate_after_training=evaluate_after_training
    )
    
    return results
