import argparse

from src.pipelines.train_pipeline import TrainingPipeline
from src.arguments import args, parser
import logging
import src.logger


# from src.pipelines.infer_pipeline import run_inference_pipeline, predict_on_files
# from src.models.registry import print_available_models, compare_all_models
# from src.utils.model_comparison import quick_model_comparison
#
#
def train_model(_args):
    """Train the drone detection model."""
    logging.info(f"Starting training pipeline with model: {args.model}")
    pipeline = TrainingPipeline(
        model_name=_args.model,
        dataset_key=_args.dataset,
        epochs=_args.epochs,
        checkpoint_path=_args.checkpoint,
        device=_args.device,
        resume_from_checkpoint=_args.resume,
        evaluate_after_training=_args.evaluate,
    )
    results = pipeline.run()
    logging.info("Training completed successfully!")
    return results


#
#
def run_inference(_args):
    """Run inference on test datasets or custom files."""


#     if args.files:
#         # Predict on specific files
#         file_paths = args.files
#         print(f"Running inference on {len(file_paths)} files with model: {args.model}")
#         results = predict_on_files(
#             file_paths=file_paths,
#             model_name=args.model,
#             checkpoint_path=args.checkpoint,
#             device=args.device
#         )
#
#     else:
#         # Run comprehensive evaluation
#         print(f"Running comprehensive inference pipeline with model: {args.model}")
#         results = run_inference_pipeline(
#             model_name=args.model,
#             checkpoint_path=args.checkpoint,
#             device=args.device,
#         )
#         print("Inference completed successfully!")
#
#     return results
#
#
def list_models(_args):
    pass
    """List available models."""
    # print_available_models()


def main():
    if args.mode == "train":
        train_model(args)
    elif args.mode == "infer":
        run_inference(args)
    elif args.mode == "list-models":
        list_models(args)
    else:
        # Default behavior - show help
        parser.print_help()


if __name__ == "__main__":
    main()
