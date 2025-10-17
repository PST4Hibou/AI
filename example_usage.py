"""Example usage of the migrated drone detection model."""

from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.infer_pipeline import InferencePipeline
from src.models.model import create_model, load_model_checkpoint
from src.models.infer import InferenceEngine
from src.data.load_data import load_huggingface_dataset


def example_training():
    """Example of how to train the model using the new structure."""
    print("=== Training Example ===")
    
    # Option 1: Use the training pipeline (recommended)
    pipeline = TrainingPipeline()
    results = pipeline.run_full_pipeline(epochs=2)  # Short training for demo
    
    print("Training completed!")
    print(f"Best validation accuracy: {pipeline.metrics_tracker.get_best_score('val_acc'):.4f}")
    return results


def example_inference():
    """Example of how to run inference using the new structure."""
    print("=== Inference Example ===")
    
    # Option 1: Use the inference pipeline (recommended)
    pipeline = InferencePipeline()
    pipeline.load_model()
    
    # Run comprehensive evaluation
    results = pipeline.run_comprehensive_evaluation()
    
    print("Inference completed!")
    return results


def example_custom_usage():
    """Example of how to use individual components."""
    print("=== Custom Usage Example ===")
    
    # Load data
    print("Loading dataset...")
    dataset = load_huggingface_dataset()
    labels = dataset["train"].features["label"]
    
    # Create and load model
    print("Creating model...")
    model = create_model()
    model = load_model_checkpoint(model, "best_model_2dcnn_clean.pt")
    
    # Create inference engine
    print("Creating inference engine...")
    inference_engine = InferenceEngine(model)
    
    # Example prediction on a single file
    # pred_label, confidence = inference_engine.predict_single("path/to/audio.wav", labels)
    # print(f"Prediction: {pred_label} (confidence: {confidence:.3f})")
    
    print("Custom setup completed!")


def main():
    """Run all examples."""
    print("Drone Detection Model - Usage Examples")
    print("=" * 50)
    
    # Note: Uncomment the example you want to run
    
    # example_training()  # Uncomment to run training example
    # example_inference()  # Uncomment to run inference example
    example_custom_usage()  # This one is safe to run without training
    
    print("\nExamples completed!")


if __name__ == "__main__":
    main()
