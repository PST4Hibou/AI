"""Evaluation metrics and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import DEVICE


def calculate_accuracy(y_true: List, y_pred: List) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def generate_classification_report(y_true: List, y_pred: List, 
                                 target_names: Optional[List[str]] = None, 
                                 digits: int = 3) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the target classes
        digits: Number of digits for formatting output values
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names, digits=digits)


def plot_confusion_matrix(y_true: List, y_pred: List, 
                         target_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix - Drone Recognition",
                         figsize: Tuple[int, int] = (10, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of the target classes
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=target_names, yticklabels=target_names,
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()


def evaluate_model_on_test_set(model, test_loader: DataLoader, 
                              target_names: Optional[List[str]] = None,
                              device: str = DEVICE) -> Dict[str, Any]:
    """
    Evaluate model performance on test set and generate comprehensive metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        target_names: Names of the target classes
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    y_true, y_pred = [], []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    accuracy = calculate_accuracy(y_true, y_pred)
    report = generate_classification_report(y_true, y_pred, target_names)
    
    # Print results
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(report)
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, target_names)
    plt.show()
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def plot_training_history(history: Dict[str, List], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


class MetricsTracker:
    """Class to track and store metrics during training and evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_results': None
        }
    
    def update(self, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float):
        """
        Update metrics for one epoch.
        
        Args:
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
    
    def set_test_results(self, test_results: Dict[str, Any]):
        """
        Set test results.
        
        Args:
            test_results: Dictionary containing test evaluation results
        """
        self.metrics['test_results'] = test_results
    
    def get_best_epoch(self, metric: str = 'val_acc') -> int:
        """
        Get the epoch with the best performance for a given metric.
        
        Args:
            metric: Metric to optimize ('val_acc' or 'val_loss')
            
        Returns:
            Best epoch number (1-indexed)
        """
        if metric == 'val_loss':
            return np.argmin(self.metrics[metric]) + 1
        else:
            return np.argmax(self.metrics[metric]) + 1
    
    def get_best_score(self, metric: str = 'val_acc') -> float:
        """
        Get the best score for a given metric.
        
        Args:
            metric: Metric to get best score for
            
        Returns:
            Best score
        """
        if metric == 'val_loss':
            return min(self.metrics[metric])
        else:
            return max(self.metrics[metric])
    
    def plot_history(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training history."""
        return plot_training_history(self.metrics, save_path)
    
    def save_metrics(self, save_path: str):
        """Save metrics to a file."""
        pd.DataFrame(self.metrics).to_csv(save_path, index=False)
        print(f"Metrics saved to {save_path}")
    
    def print_summary(self):
        """Print a summary of the training metrics."""
        if not self.metrics['train_loss']:
            print("No training metrics available.")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total epochs: {len(self.metrics['train_loss'])}")
        print(f"Best validation accuracy: {self.get_best_score('val_acc'):.4f} (epoch {self.get_best_epoch('val_acc')})")
        print(f"Best validation loss: {self.get_best_score('val_loss'):.4f} (epoch {self.get_best_epoch('val_loss')})")
        
        if self.metrics['test_results']:
            print(f"Final test accuracy: {self.metrics['test_results']['accuracy']:.4f}")
        print("="*50)
