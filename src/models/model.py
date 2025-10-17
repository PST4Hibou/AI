"""Model architecture definitions for audio classification."""

import torch
from torch import nn
from typing import Optional

from ..config import N_CLASSES, DEVICE


class AudioCNN2D(nn.Module):
    """
    2D CNN architecture for audio classification using mel spectrograms.
    
    This model processes mel spectrograms through convolutional layers with
    batch normalization and adaptive pooling, followed by fully connected layers.
    """
    
    def __init__(self, n_classes: int = N_CLASSES):
        """
        Initialize the AudioCNN2D model.
        
        Args:
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_classes = n_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)
            
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)
    
    def get_num_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(n_classes: int = N_CLASSES, device: str = DEVICE) -> AudioCNN2D:
    """
    Create and initialize the AudioCNN2D model.
    
    Args:
        n_classes: Number of output classes
        device: Device to place the model on ('cuda' or 'cpu')
        
    Returns:
        Initialized AudioCNN2D model
    """
    model = AudioCNN2D(n_classes=n_classes)
    model = model.to(device)
    
    print(f"Model created with {model.get_num_parameters():,} parameters")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    print(f"Model device: {device}")
    
    return model


def load_model_checkpoint(model: AudioCNN2D, checkpoint_path: str, 
                         device: str = DEVICE) -> AudioCNN2D:
    """
    Load model weights from a checkpoint.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        Model with loaded weights
    """
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Successfully loaded model checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        print("Using randomly initialized model weights")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model weights")
    
    return model
