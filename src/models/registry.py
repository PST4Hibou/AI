"""Model registry and factory for managing multiple model architectures."""

from typing import Dict, Type, Any, Optional, List
import torch.nn as nn

from ..config import N_CLASSES, DEVICE
from .model import AudioCNN2D
from .architectures import (
    AudioCNN1D, AudioCNN2DDeep, AudioCNNLightweight, 
    AudioAttentionCNN, get_model_info
)


class ModelRegistry:
    """Registry for managing different model architectures."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, Type[nn.Module]] = {}
        self._descriptions: Dict[str, str] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model architectures."""
        self.register(
            "audiocnn2d", 
            AudioCNN2D,
            "Original 2D CNN from notebook - balanced performance and accuracy"
        )
        
        self.register(
            "audiocnn1d", 
            AudioCNN1D,
            "1D CNN working on raw waveforms - good for temporal patterns"
        )
        
        self.register(
            "audiocnn2d_deep", 
            AudioCNN2DDeep,
            "Deeper 2D CNN with residual connections - high accuracy, more parameters"
        )
        
        self.register(
            "audiocnn_lightweight", 
            AudioCNNLightweight,
            "Lightweight CNN with depthwise separable convs - fast inference"
        )
        
        self.register(
            "audiocnn_attention", 
            AudioAttentionCNN,
            "CNN with attention mechanism - focuses on important features"
        )
    
    def register(self, name: str, model_class: Type[nn.Module], description: str = ""):
        """
        Register a new model architecture.
        
        Args:
            name: Unique name for the model
            model_class: Model class
            description: Description of the model
        """
        self._models[name] = model_class
        self._descriptions[name] = description
        print(f"Registered model: {name}")
    
    def get_model_class(self, name: str) -> Type[nn.Module]:
        """
        Get model class by name.
        
        Args:
            name: Model name
            
        Returns:
            Model class
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        return self._models[name]
    
    def create_model(self, name: str, n_classes: int = N_CLASSES, 
                    device: str = DEVICE, **kwargs) -> nn.Module:
        """
        Create a model instance by name.
        
        Args:
            name: Model name
            n_classes: Number of output classes
            device: Device to place model on
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Model instance
        """
        model_class = self.get_model_class(name)
        model = model_class(n_classes=n_classes, **kwargs)
        model = model.to(device)
        
        # Print model info
        info = get_model_info(model)
        print(f"Created {info['architecture']} model:")
        print(f"  - Parameters: {info['total_parameters']:,}")
        print(f"  - Model size: {info['model_size_mb']:.2f} MB")
        print(f"  - Device: {device}")
        
        return model
    
    def list_models(self) -> Dict[str, str]:
        """
        List all available models with descriptions.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        return self._descriptions.copy()
    
    def get_model_names(self) -> List[str]:
        """Get list of available model names."""
        return list(self._models.keys())
    
    def print_models(self):
        """Print all available models with descriptions."""
        print("Available Models:")
        print("=" * 50)
        for name, description in self._descriptions.items():
            print(f"• {name:<20} - {description}")
        print("=" * 50)
    
    def compare_models(self, n_classes: int = N_CLASSES) -> Dict[str, Dict[str, Any]]:
        """
        Compare all models by creating instances and getting their info.
        
        Args:
            n_classes: Number of output classes
            
        Returns:
            Dictionary with model comparison data
        """
        comparison = {}
        
        for name in self._models.keys():
            try:
                # Create model on CPU for comparison to avoid memory issues
                model = self.create_model(name, n_classes=n_classes, device="cpu")
                info = get_model_info(model)
                info['description'] = self._descriptions[name]
                comparison[name] = info
                
                # Clean up
                del model
            except Exception as e:
                print(f"Error creating model {name}: {e}")
                comparison[name] = {'error': str(e)}
        
        return comparison
    
    def print_model_comparison(self, n_classes: int = N_CLASSES):
        """Print a formatted comparison of all models."""
        comparison = self.compare_models(n_classes)
        
        print("\nModel Comparison:")
        print("=" * 100)
        print(f"{'Model':<20} {'Parameters':<12} {'Size (MB)':<10} {'Description':<50}")
        print("-" * 100)
        
        for name, info in comparison.items():
            if 'error' not in info:
                params = f"{info['total_parameters']:,}"
                size = f"{info['model_size_mb']:.2f}"
                desc = info['description'][:47] + "..." if len(info['description']) > 50 else info['description']
                print(f"{name:<20} {params:<12} {size:<10} {desc}")
            else:
                print(f"{name:<20} {'ERROR':<12} {'N/A':<10} {info['error']}")
        
        print("=" * 100)


# Global model registry instance
model_registry = ModelRegistry()


def create_model(name: str, n_classes: int = N_CLASSES, device: str = DEVICE, 
                **kwargs) -> nn.Module:
    """
    Convenience function to create a model using the global registry.
    
    Args:
        name: Model name
        n_classes: Number of output classes
        device: Device to place model on
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    return model_registry.create_model(name, n_classes, device, **kwargs)


def list_available_models() -> Dict[str, str]:
    """
    Convenience function to list available models.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return model_registry.list_models()


def print_available_models():
    """Print all available models."""
    model_registry.print_models()


def compare_all_models(n_classes: int = N_CLASSES):
    """Print comparison of all models."""
    model_registry.print_model_comparison(n_classes)


# Backward compatibility - keep the original create_model function working
def create_original_model(n_classes: int = N_CLASSES, device: str = DEVICE) -> AudioCNN2D:
    """Create the original AudioCNN2D model for backward compatibility."""
    return create_model("audiocnn2d", n_classes, device)
