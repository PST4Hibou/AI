import argparse

parser = argparse.ArgumentParser(description="Drone Detection Model")
parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
parser.add_argument(
    "--model",
    default="audiocnn2d",
    help="Model to use (audiocnn2d, audiocnn1d, audiocnn2d_deep, audiocnn_lightweight, audiocnn_attention)",
)
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

subparsers = parser.add_subparsers(dest="mode", help="Mode to run")

# Training arguments
train_parser = subparsers.add_parser("train", help="Train the model")
train_parser.add_argument(
    "--dataset",
    default="df_462700_2",
    help="HuggingFace dataset name",
)
train_parser.add_argument(
    "--epochs", type=int, default=10, help="Number of training epochs"
)
train_parser.add_argument(
    "--resume", action="store_true", help="Resume from checkpoint"
)
train_parser.add_argument(
    "--no-evaluate",
    dest="evaluate",
    action="store_false",
    help="Skip evaluation after training",
)

# Inference arguments
infer_parser = subparsers.add_parser("infer", help="Run inference")
infer_parser.add_argument("--files", nargs="+", help="Audio files to predict on")

# Model listing arguments
list_parser = subparsers.add_parser("list-models", help="List available models")

# Model comparison arguments
compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
compare_parser.add_argument(
    "--models", nargs="+", help="Specific models to compare (default: all models)"
)
compare_parser.add_argument(
    "--quick", action="store_true", help="Run quick comparison with training"
)
compare_parser.add_argument(
    "--epochs", type=int, default=3, help="Number of epochs for quick comparison"
)

args = parser.parse_args()
