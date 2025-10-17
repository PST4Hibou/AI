# Multi-Model Testing Guide

This guide explains how to test and compare multiple model architectures for drone detection.

## 🏗️ Available Models

The system now supports 5 different model architectures:

| Model Name | Description | Best For | Parameters | Speed |
|------------|-------------|----------|------------|-------|
| `audiocnn2d` | Original 2D CNN from notebook | Balanced performance | ~100K | Medium |
| `audiocnn1d` | 1D CNN for raw waveforms | Temporal patterns | ~500K | Fast |
| `audiocnn2d_deep` | Deeper 2D CNN with residual connections | High accuracy | ~1M | Slow |
| `audiocnn_lightweight` | Lightweight with depthwise separable convs | Fast inference | ~20K | Very Fast |
| `audiocnn_attention` | CNN with attention mechanism | Feature selection | ~150K | Medium |

## 🚀 Quick Start

### List Available Models
```bash
python main.py list-models
```

### Train a Specific Model
```bash
# Train lightweight model for fast inference
python main.py train --model audiocnn_lightweight --epochs 10

# Train deep model for high accuracy
python main.py train --model audiocnn2d_deep --epochs 15
```

### Run Inference with Different Models
```bash
# Use lightweight model for inference
python main.py infer --model audiocnn_lightweight

# Use 1D model for inference
python main.py infer --model audiocnn1d --files audio1.wav audio2.wav
```

### Compare Models
```bash
# Quick architecture comparison
python main.py compare

# Full comparison with training (takes time!)
python main.py compare --quick --epochs 3

# Compare specific models
python main.py compare --quick --models audiocnn2d audiocnn_lightweight --epochs 5
```

## 📊 Model Comparison

### Architecture Comparison
```python
from src.models.registry import compare_all_models
compare_all_models()
```

### Full Comparison with Training
```python
from src.utils.model_comparison import quick_model_comparison

results = quick_model_comparison(
    model_names=["audiocnn2d", "audiocnn_lightweight", "audiocnn1d"],
    epochs=5
)
```

### Custom Comparison
```python
from src.utils.model_comparison import ModelComparator

comparator = ModelComparator(["audiocnn2d", "audiocnn_attention"])
comparator.load_data()

# Compare complexity
complexity_df = comparator.compare_model_complexity()

# Benchmark speed
speed_df = comparator.benchmark_inference_speed()

# Train and compare
training_df = comparator.train_and_compare(epochs=10)

# Plot results
comparator.plot_comparison_results(training_df)
```

## 🔧 Python API Usage

### Training Different Models
```python
from src.pipelines.train_pipeline import TrainingPipeline

# Train lightweight model
pipeline = TrainingPipeline(model_name="audiocnn_lightweight")
results = pipeline.run_full_pipeline(epochs=10)

# Train deep model
pipeline = TrainingPipeline(model_name="audiocnn2d_deep")
results = pipeline.run_full_pipeline(epochs=15)
```

### Inference with Different Models
```python
from src.pipelines.infer_pipeline import InferencePipeline

# Use attention model for inference
pipeline = InferencePipeline(model_name="audiocnn_attention")
pipeline.load_model()
results = pipeline.run_comprehensive_evaluation()
```

### Create Custom Models
```python
from src.models.registry import create_model

# Create different models
lightweight = create_model("audiocnn_lightweight")
deep_model = create_model("audiocnn2d_deep")
attention_model = create_model("audiocnn_attention")
```

## 📈 Benchmarking Results

### Model Complexity
- **Smallest**: `audiocnn_lightweight` (~20K parameters)
- **Largest**: `audiocnn2d_deep` (~1M parameters)
- **Balanced**: `audiocnn2d` (~100K parameters)

### Inference Speed (samples/second)*
- **Fastest**: `audiocnn_lightweight` (~2000 samples/s)
- **Medium**: `audiocnn2d`, `audiocnn_attention` (~1000 samples/s)
- **Slower**: `audiocnn2d_deep` (~500 samples/s)
- **Variable**: `audiocnn1d` (depends on input length)

*Results may vary based on hardware and input size

### Accuracy Comparison*
Results depend on dataset and training configuration:
- **High Accuracy**: `audiocnn2d_deep`, `audiocnn_attention`
- **Balanced**: `audiocnn2d`
- **Good for Speed**: `audiocnn_lightweight`
- **Specialized**: `audiocnn1d` (good for temporal patterns)

*Run your own comparison for dataset-specific results

## 🎯 Model Selection Guide

### Choose `audiocnn_lightweight` if:
- ✅ Fast inference is critical
- ✅ Limited computational resources
- ✅ Real-time processing needed
- ✅ Good enough accuracy is acceptable

### Choose `audiocnn2d` if:
- ✅ Balanced performance needed
- ✅ Proven architecture (from original notebook)
- ✅ Medium computational resources
- ✅ Good starting point

### Choose `audiocnn2d_deep` if:
- ✅ Maximum accuracy is priority
- ✅ Abundant computational resources
- ✅ Training time is not critical
- ✅ Complex patterns in data

### Choose `audiocnn1d` if:
- ✅ Working with raw audio waveforms
- ✅ Temporal patterns are important
- ✅ Different preprocessing approach
- ✅ Experimental comparisons

### Choose `audiocnn_attention` if:
- ✅ Feature selection is important
- ✅ Interpretability needed
- ✅ Modern architecture preferred
- ✅ Balanced accuracy and efficiency

## 🔄 Batch Model Training

### Train All Models (Bash)
```bash
#!/bin/bash
models=("audiocnn2d" "audiocnn1d" "audiocnn_lightweight" "audiocnn_attention" "audiocnn2d_deep")

for model in "${models[@]}"; do
    echo "Training $model..."
    python main.py train --model $model --epochs 10 --checkpoint "${model}_best.pt"
done
```

### Train All Models (Python)
```python
from src.pipelines.train_pipeline import TrainingPipeline

models = ["audiocnn2d", "audiocnn1d", "audiocnn_lightweight", "audiocnn_attention"]
results = {}

for model_name in models:
    print(f"Training {model_name}...")
    pipeline = TrainingPipeline(
        model_name=model_name,
        checkpoint_path=f"{model_name}_best.pt"
    )
    results[model_name] = pipeline.run_full_pipeline(epochs=10)
```

## 📋 Model Checkpoints

Each model saves its checkpoint with a unique name:
```
audiocnn2d_best.pt
audiocnn1d_best.pt
audiocnn2d_deep_best.pt
audiocnn_lightweight_best.pt
audiocnn_attention_best.pt
```

## 🔍 Troubleshooting

### Memory Issues
- Use `audiocnn_lightweight` for low memory
- Reduce batch size in config
- Use CPU instead of GPU for comparison

### Training Time
- Use fewer epochs for quick testing
- Start with `audiocnn_lightweight`
- Use `--quick` flag for comparisons

### Model Selection
- Run `python main.py compare` first
- Test with small epochs initially
- Consider your hardware limitations

## 🎛️ Advanced Configuration

### Custom Model Registration
```python
from src.models.registry import model_registry

# Register custom model
model_registry.register(
    "my_custom_model", 
    MyCustomModelClass,
    "Description of my custom model"
)

# Use custom model
model = create_model("my_custom_model")
```

### Model Comparison Configuration
```python
from src.utils.model_comparison import ModelComparator

comparator = ModelComparator(
    model_names=["model1", "model2"],
    device="cuda"
)

# Custom comparison parameters
complexity_df = comparator.compare_model_complexity()
speed_df = comparator.benchmark_inference_speed(num_samples=200)
training_df = comparator.train_and_compare(epochs=20)
```

## 📊 Results and Reports

The comparison system automatically generates:
- CSV files with detailed metrics
- Comparison plots and visualizations
- Markdown reports with key findings
- Model checkpoint files

Results are saved in:
- `model_comparison_checkpoints/` - Model checkpoints
- `results/` - Training results and plots
- `inference_results/` - Inference results
- `model_comparison_report.md` - Comparison report

## 🎉 Summary

You now have a complete multi-model testing framework that allows you to:
- ✅ Train and compare 5 different architectures
- ✅ Benchmark performance and speed
- ✅ Select optimal models for your use case
- ✅ Generate comprehensive comparison reports
- ✅ Use both CLI and Python API
- ✅ Extend with custom models

Start with `python main.py list-models` and `python main.py compare` to explore the available options!
