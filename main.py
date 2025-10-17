"""Main entry point for the drone detection project."""

import argparse
from pathlib import Path

from src.pipelines.train_pipeline import run_training_pipeline
from src.pipelines.infer_pipeline import run_inference_pipeline, predict_on_files
from src.models.registry import print_available_models, compare_all_models
from src.utils.model_comparison import quick_model_comparison


def train_model(args):
    """Train the drone detection model."""
    print(f"Starting training pipeline with model: {args.model}")
    results = run_training_pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        device=args.device,
        resume_from_checkpoint=args.resume,
        evaluate_after_training=args.evaluate,
    )
    print("Training completed successfully!")
    return results


def run_inference(args):
    """Run inference on test datasets or custom files."""
    if args.files:
        # Predict on specific files
        file_paths = args.files
        print(f"Running inference on {len(file_paths)} files with model: {args.model}")
        results = predict_on_files(
            file_paths=file_paths,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

    else:
        # Run comprehensive evaluation
        print(f"Running comprehensive inference pipeline with model: {args.model}")
        results = run_inference_pipeline(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
        print("Inference completed successfully!")
    
    return results


def list_models(args):
    """List available models."""
    print_available_models()


def compare_models(args):
    """Compare multiple models."""
    if args.quick:
        print("Running quick model comparison...")
        results = quick_model_comparison(
            model_names=args.models,
            epochs=args.epochs
        )
        print("Model comparison completed!")
        return results
    else:
        print("Showing model complexity comparison...")
        compare_all_models()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Drone Detection Model")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--checkpoint", default="models/best_model_2dcnn_clean.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--model", default="audiocnn2d",
                       help="Model to use (audiocnn2d, audiocnn1d, audiocnn2d_deep, audiocnn_lightweight, audiocnn_attention)")
    
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", default="Usernameeeeee/df_462700_2",
                             help="HuggingFace dataset name")
    train_parser.add_argument("--epochs", type=int, default=10,
                             help="Number of training epochs")
    train_parser.add_argument("--resume", action="store_true",
                             help="Resume from checkpoint")
    train_parser.add_argument("--no-evaluate", dest="evaluate", action="store_false",
                             help="Skip evaluation after training")

    # Inference arguments
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--files", nargs="+", help="Audio files to predict on")

    # Model listing arguments
    list_parser = subparsers.add_parser("list-models", help="List available models")
    
    # Model comparison arguments
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", nargs="+", 
                               help="Specific models to compare (default: all models)")
    compare_parser.add_argument("--quick", action="store_true",
                               help="Run quick comparison with training")
    compare_parser.add_argument("--epochs", type=int, default=3,
                               help="Number of epochs for quick comparison")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        return train_model(args)
    elif args.mode == "infer":
        return run_inference(args)
    elif args.mode == "list-models":
        return list_models(args)
    elif args.mode == "compare":
        return compare_models(args)
    else:
        # Default behavior - show help
        parser.print_help()
        return None


if __name__ == "__main__":
    main()