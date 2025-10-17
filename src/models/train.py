"""Training logic for the audio classification model."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from ..config import (
    DEVICE, EPOCHS, LEARNING_RATE, N_CLASSES, MODEL_CHECKPOINT_PATH
)
from .model import AudioCNN2D


class Trainer:
    """Trainer class for the AudioCNN2D model."""
    
    def __init__(self, model: AudioCNN2D, device: str = DEVICE):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
        # Metrics
        self.metric_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=N_CLASSES
        ).to(device)
        
        # Best validation accuracy for checkpointing
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{total_epochs} [Train]"
        )
        
        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_acc += self.metric_acc(outputs, y) * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{self.metric_acc(outputs, y):.4f}'
            })
        
        # Calculate averages
        train_loss /= len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        
        return train_loss, train_acc
    
    def validate_epoch(self, valid_loader: DataLoader, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            valid_loader: Validation data loader
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        progress_bar = tqdm(
            valid_loader, 
            desc=f"Epoch {epoch+1}/{total_epochs} [Valid]"
        )
        
        with torch.no_grad():
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                # Update metrics
                batch_size = x.size(0)
                val_loss += loss.item() * batch_size
                val_acc += self.metric_acc(outputs, y) * batch_size
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{self.metric_acc(outputs, y):.4f}'
                })
        
        # Calculate averages
        val_loss /= len(valid_loader.dataset)
        val_acc = val_acc / len(valid_loader.dataset)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, checkpoint_path: str = MODEL_CHECKPOINT_PATH):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        # Ensure the directory exists
        checkpoint_dir = Path(checkpoint_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✅ Saved new best model to {checkpoint_path}!")
    
    def train(self, train_loader: DataLoader, valid_loader: DataLoader, 
              epochs: int = EPOCHS, checkpoint_path: str = MODEL_CHECKPOINT_PATH) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            epochs: Number of epochs to train
            checkpoint_path: Path to save best model checkpoint
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs)
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(valid_loader, epoch, epochs)
            
            # Log results
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(checkpoint_path)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(float(train_acc))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(float(val_acc))
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.4f}")
        return history


def train_model(model: AudioCNN2D, train_loader: DataLoader, valid_loader: DataLoader,
                epochs: int = EPOCHS, device: str = DEVICE, 
                checkpoint_path: str = MODEL_CHECKPOINT_PATH) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        epochs: Number of epochs
        device: Device to train on
        checkpoint_path: Path to save best checkpoint
        
    Returns:
        Training history dictionary
    """
    trainer = Trainer(model, device)
    return trainer.train(train_loader, valid_loader, epochs, checkpoint_path)
