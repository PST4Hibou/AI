"""Examples demonstrating multi-model capabilities."""

from src.models.registry import print_available_models, create_model, compare_all_models
from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.infer_pipeline import InferencePipeline
from src.utils.model_comparison import ModelComparator, quick_model_comparison


def example_list_models():
    """Example of listing available models."""
    print("=== Available Models ===")
    print_available_models()
    print()


def example_model_comparison():
    """Example of comparing model architectures without training."""
    print("=== Model Architecture Comparison ===")
    compare_all_models()
    print()


def example_create_different_models():
    """Example of creating different model architectures."""
    print("=== Creating Different Models ===")
    
    model_names = ["audiocnn2d", "audiocnn1d", "audiocnn_lightweight", "audiocnn_attention"]
    
    for model_name in model_names:
        try:
            print(f"\nCreating {model_name}...")
            model = create_model(model_name, device="cpu")  # Use CPU for demo
            print(f"✅ Successfully created {model_name}")
            
            # Clean up
            del model
            
        except Exception as e:
            print(f"❌ Error creating {model_name}: {e}")


def example_train_different_models():
    """Example of training different models."""
    print("=== Training Different Models ===")
    
    # List of models to train (reduced for demo)
    models_to_train = ["audiocnn2d", "audiocnn_lightweight"]
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            # Create training pipeline with specific model
            pipeline = TrainingPipeline(
                model_name=model_name,
                checkpoint_path=f"models/{model_name}_checkpoint.pt"
            )
            
            # Load data (comment out to avoid actual execution)
            # pipeline.load_data()
            # pipeline.create_model()
            # results = pipeline.train(epochs=2)  # Short training for demo
            
            print(f"✅ Would train {model_name} successfully")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")


def example_inference_with_different_models():
    """Example of running inference with different models."""
    print("=== Inference with Different Models ===")
    
    models_to_test = ["audiocnn2d", "audiocnn_lightweight"]
    
    for model_name in models_to_test:
        print(f"\n--- Inference with {model_name} ---")
        
        try:
            # Create inference pipeline with specific model
            pipeline = InferencePipeline(
                model_name=model_name,
                checkpoint_path=f"models/{model_name}_checkpoint.pt"
            )
            
            # Load model (comment out to avoid actual execution)
            # pipeline.load_model()
            # results = pipeline.run_comprehensive_evaluation()
            
            print(f"✅ Would run inference with {model_name} successfully")
            
        except Exception as e:
            print(f"❌ Error with inference using {model_name}: {e}")


def example_comprehensive_model_comparison():
    """Example of comprehensive model comparison with training."""
    print("=== Comprehensive Model Comparison ===")
    
    # Select models to compare (reduced for demo)
    models_to_compare = ["audiocnn2d", "audiocnn_lightweight"]
    
    print(f"Comparing models: {models_to_compare}")
    
    try:
        # Create comparator
        comparator = ModelComparator(model_names=models_to_compare)
        
        # Compare model complexity (this is safe to run)
        complexity_df = comparator.compare_model_complexity()
        print("\nModel Complexity Comparison:")
        print(complexity_df.to_string(index=False))
        
        # The following would do actual training and evaluation
        # Uncomment to run (will take significant time):
        
        # comparator.load_data()
        # 
        # # Benchmark inference speed
        # speed_df = comparator.benchmark_inference_speed(num_samples=50)
        # print("\nInference Speed Comparison:")
        # print(speed_df.to_string(index=False))
        # 
        # # Train and compare models
        # training_df = comparator.train_and_compare(epochs=2)
        # print("\nTraining Comparison:")
        # print(training_df.to_string(index=False))
        
        print("✅ Model comparison framework ready")
        
    except Exception as e:
        print(f"❌ Error in model comparison: {e}")


def example_quick_comparison():
    """Example of quick model comparison function."""
    print("=== Quick Model Comparison ===")
    
    # This would run a complete comparison including training
    # Uncomment to run (will take significant time):
    
    # results = quick_model_comparison(
    #     model_names=["audiocnn2d", "audiocnn_lightweight"],
    #     epochs=2
    # )
    # print("Quick comparison completed!")
    
    print("✅ Quick comparison function available")


def main():
    """Run all examples."""
    print("Multi-Model Drone Detection Examples")
    print("=" * 60)
    
    # Safe examples that don't require training
    example_list_models()
    example_model_comparison()
    example_create_different_models()
    
    # Examples that show structure but don't execute heavy operations
    example_train_different_models()
    example_inference_with_different_models()
    example_comprehensive_model_comparison()
    example_quick_comparison()
    
    print("\n" + "=" * 60)
    print("CLI Usage Examples:")
    print("=" * 60)
    print("# List available models")
    print("python main.py list-models")
    print()
    print("# Compare model architectures")
    print("python main.py compare")
    print()
    print("# Train a specific model")
    print("python main.py train --model audiocnn_lightweight --epochs 5")
    print()
    print("# Run inference with a specific model")
    print("python main.py infer --model audiocnn1d")
    print()
    print("# Quick comparison of specific models")
    print("python main.py compare --quick --models audiocnn2d audiocnn_lightweight --epochs 3")
    print()
    print("# Train multiple models (bash script)")
    print("for model in audiocnn2d audiocnn1d audiocnn_lightweight; do")
    print("    python main.py train --model $model --epochs 10")
    print("done")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
