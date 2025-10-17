# Migration Guide: Notebook to Structured Codebase

This document explains the migration from the Jupyter notebook `2dcnn_mel_clean.py.ipynb` to a structured Python codebase.

## 📁 New Structure

```
src/
├── config.py                    # Configuration constants and parameters
├── data/
│   ├── load_data.py             # Dataset loading and splitting
│   └── preprocess.py            # Audio preprocessing functions
├── models/
│   ├── model.py                 # AudioCNN2D model architecture
│   ├── train.py                 # Training logic and trainer class
│   └── infer.py                 # Inference functions and engine
├── pipelines/
│   ├── train_pipeline.py        # Complete training pipeline
│   └── infer_pipeline.py        # Complete inference pipeline
└── utils/
    ├── metrics.py               # Evaluation metrics and visualization
    └── logger.py                # Logging utilities (existing)
```

## 🔄 Migration Mapping

| Notebook Component | New Location | Description |
|-------------------|--------------|-------------|
| Constants (SEED, SAMPLE_RATE, etc.) | `src/config.py` | Centralized configuration |
| `preprocess()`, `collate_fn()` | `src/data/preprocess.py` | Audio preprocessing |
| Dataset loading and splitting | `src/data/load_data.py` | Data management |
| `AudioCNN2D` class | `src/models/model.py` | Model architecture |
| Training loop | `src/models/train.py` | Training logic |
| `infer()`, `infer_from_dataset()` | `src/models/infer.py` | Inference functions |
| Evaluation and plotting | `src/utils/metrics.py` | Metrics and visualization |
| Complete workflows | `src/pipelines/` | End-to-end pipelines |

## 🚀 Usage Examples

### 1. Command Line Interface

```bash
# Train the model
python main.py train --epochs 10 --device cuda

# Run inference on test datasets
python main.py infer

# Run inference on specific files
python main.py infer --files audio1.wav audio2.wav

# Resume training from checkpoint
python main.py train --resume --epochs 5
```

### 2. Training Pipeline

```python
from src.pipelines.train_pipeline import TrainingPipeline

# Create and run training pipeline
pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline(epochs=10)

# Access trained model and metrics
model = results['model']
history = results['history']
```

### 3. Inference Pipeline

```python
from src.pipelines.infer_pipeline import InferencePipeline

# Create and run inference pipeline
pipeline = InferencePipeline()
pipeline.load_model()
results = pipeline.run_comprehensive_evaluation()
```

### 4. Custom Usage

```python
from src.models.model import create_model, load_model_checkpoint
from src.models.infer import InferenceEngine
from src.data.load_data import load_huggingface_dataset

# Load data and model
dataset = load_huggingface_dataset()
labels = dataset["train"].features["label"]
model = create_model()
model = load_model_checkpoint(model, "best_model_2dcnn_clean.pt")

# Create inference engine
engine = InferenceEngine(model)
pred_label, confidence = engine.predict_single("audio.wav", labels)
```

## ✨ Key Improvements

1. **Modularity**: Code is organized into logical modules
2. **Reusability**: Components can be used independently
3. **Configuration**: Centralized configuration management
4. **Error Handling**: Better error handling and validation
5. **Documentation**: Comprehensive docstrings and type hints
6. **Testing**: Structure supports easy unit testing
7. **Extensibility**: Easy to add new features and models
8. **CLI Support**: Command-line interface for common tasks

## 📋 Original Notebook Features Preserved

- ✅ All preprocessing functions (`preprocess`, `collate_fn`)
- ✅ AudioCNN2D model architecture
- ✅ Training loop with validation and checkpointing
- ✅ Inference functions for single files and datasets
- ✅ Evaluation on multiple test datasets
- ✅ Metrics calculation and visualization
- ✅ Model checkpointing and loading
- ✅ Support for HuggingFace datasets
- ✅ Local file evaluation capabilities

## 🔧 Configuration

All configuration is centralized in `src/config.py`:

```python
# Audio processing
SAMPLE_RATE = 16000
N_MELS = 64

# Training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

# Model
N_CLASSES = 2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📊 Running the Original Workflow

To replicate the exact notebook workflow:

```python
from src.pipelines.train_pipeline import run_training_pipeline
from src.pipelines.infer_pipeline import run_inference_pipeline

# Complete training and evaluation
train_results = run_training_pipeline(epochs=10)
infer_results = run_inference_pipeline()
```

This provides the same functionality as the original notebook but with better organization, error handling, and extensibility.
