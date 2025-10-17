"""Additional model architectures for drone detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..config import N_CLASSES, DEVICE


class AudioCNN1D(nn.Module):
    """
    1D CNN architecture for audio classification working directly on waveforms.
    
    This model processes raw audio waveforms through 1D convolutional layers
    with batch normalization and adaptive pooling.
    """
    
    def __init__(self, n_classes: int = N_CLASSES, input_length: int = 16000):
        """
        Initialize the AudioCNN1D model.
        
        Args:
            n_classes: Number of output classes
            input_length: Expected input length (sample rate * seconds)
        """
        super().__init__()
        self.n_classes = n_classes
        self.input_length = input_length
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, time_samples) for raw audio
               or (batch_size, 1, n_mels, time_steps) for spectrograms
            
        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        # Handle both raw audio and spectrogram inputs
        if x.dim() == 4:  # Spectrogram input (batch, 1, n_mels, time)
            # Flatten mel bins into the channel dimension
            x = x.view(x.size(0), x.size(2), x.size(3))  # (batch, n_mels, time)
        elif x.dim() == 3 and x.size(1) == 1:  # Raw audio (batch, 1, time)
            x = x  # Keep as is
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


class AudioCNN2DDeep(nn.Module):
    """
    Deeper 2D CNN architecture with residual connections for complex feature extraction.
    """
    
    def __init__(self, n_classes: int = N_CLASSES):
        """
        Initialize the AudioCNN2DDeep model.
        
        Args:
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_classes = n_classes
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 256)
        self.res_block4 = self._make_res_block(256, 512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def _make_res_block(self, in_channels: int, out_channels: int):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.initial_conv(x)
        
        # Residual blocks with skip connections
        x = self._forward_res_block(x, self.res_block1)
        x = self._forward_res_block(x, self.res_block2)
        x = self._forward_res_block(x, self.res_block3)
        x = self._forward_res_block(x, self.res_block4)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def _forward_res_block(self, x: torch.Tensor, block: nn.Sequential) -> torch.Tensor:
        """Forward through residual block with skip connection."""
        identity = x
        out = block(x)
        
        # Adjust dimensions if needed for skip connection
        if identity.shape != out.shape:
            identity = F.adaptive_avg_pool2d(identity, (out.shape[2], out.shape[3]))
            if identity.shape[1] != out.shape[1]:
                identity = F.pad(identity, (0, 0, 0, 0, 0, out.shape[1] - identity.shape[1]))
        
        out = F.relu(out + identity)
        return out


class AudioCNNLightweight(nn.Module):
    """
    Lightweight CNN architecture for fast inference with minimal parameters.
    """
    
    def __init__(self, n_classes: int = N_CLASSES):
        """
        Initialize the lightweight model.
        
        Args:
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_classes = n_classes
        
        # Depthwise separable convolutions for efficiency
        self.features = nn.Sequential(
            # First depthwise separable conv
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Conv2d(1, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second depthwise separable conv
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third depthwise separable conv
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Lightweight classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class AudioAttentionCNN(nn.Module):
    """
    CNN with attention mechanism for better feature selection.
    """
    
    def __init__(self, n_classes: int = N_CLASSES):
        """
        Initialize the attention-based CNN model.
        
        Args:
            n_classes: Number of output classes
        """
        super().__init__()
        self.n_classes = n_classes
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Extract features
        features = self.backbone(x)
        
        # Generate attention weights
        attention_weights = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Global pooling and classification
        x = self.global_pool(attended_features)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Model parameter counts for reference
def get_model_info(model: nn.Module) -> dict:
    """Get information about model parameters and size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }
