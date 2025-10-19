import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..settings import DEVICE, EPOCHS
from ..data.datasets import (
    load_huggingface_dataset,
    create_data_loaders,
    load_test_datasets,
)
from ..models.registry import model_registry, create_model
from ..models.train import train_model
from ..models.infer import infer_from_dataset
from ..models.model import load_model_checkpoint
from ..utils.metrics import MetricsTracker, evaluate_model_on_test_set


class ModelComparator:
    """Class for comparing multiple model architectures."""

    def __init__(self, model_names: Optional[List[str]] = None, device: str = DEVICE):
        """
        Initialize the model comparator.

        Args:
            model_names: List of model names to compare. If None, uses all available models.
            device: Device to run comparisons on
        """
        self.device = device
        self.model_names = model_names or model_registry.get_model_names()
        self.results = {}
        self.dataset = None
        self.data_loaders = None
        self.labels = None

    def load_data(self, dataset_name: str = "Usernameeeeee/df_462700_2"):
        """Load dataset for comparison."""
        print("Loading dataset for comparison...")
        self.dataset = load_huggingface_dataset(dataset_name)
        self.labels = self.dataset["train"].features["label"]
        self.data_loaders = create_data_loaders(self.dataset)
        print("Dataset loaded successfully.")

    def compare_model_complexity(self) -> pd.DataFrame:
        """
        Compare model complexity (parameters, size, etc.).

        Returns:
            DataFrame with model complexity comparison
        """
        print("Comparing model complexity...")
        comparison_data = []

        for model_name in self.model_names:
            try:
                # Create model on CPU to avoid memory issues
                model = create_model(model_name, device="cpu")

                # Calculate metrics
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

                comparison_data.append(
                    {
                        "model_name": model_name,
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_size_mb": model_size_mb,
                        "architecture": model.__class__.__name__,
                    }
                )

                # Clean up
                del model

            except Exception as e:
                print(f"Error creating model {model_name}: {e}")
                comparison_data.append(
                    {
                        "model_name": model_name,
                        "total_parameters": 0,
                        "trainable_parameters": 0,
                        "model_size_mb": 0,
                        "architecture": "ERROR",
                        "error": str(e),
                    }
                )

        return pd.DataFrame(comparison_data)

    def benchmark_inference_speed(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Benchmark inference speed for all models.

        Args:
            num_samples: Number of samples to use for benchmarking

        Returns:
            DataFrame with inference speed comparison
        """
        if self.data_loaders is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(f"Benchmarking inference speed on {num_samples} samples...")
        speed_data = []

        # Get a subset of test data
        test_loader = self.data_loaders[2]  # Test loader
        samples = []
        for i, (x, y) in enumerate(test_loader):
            if i * x.size(0) >= num_samples:
                break
            samples.append((x, y))

        for model_name in self.model_names:
            try:
                print(f"Benchmarking {model_name}...")

                # Create and prepare model
                model = create_model(model_name, device=self.device)
                model.eval()

                # Warm up
                with torch.no_grad():
                    for x, _ in samples[:2]:
                        x = x.to(self.device)
                        _ = model(x)

                # Actual benchmarking
                start_time = time.time()
                total_samples = 0

                with torch.no_grad():
                    for x, _ in samples:
                        x = x.to(self.device)
                        _ = model(x)
                        total_samples += x.size(0)

                end_time = time.time()

                # Calculate metrics
                total_time = end_time - start_time
                samples_per_second = total_samples / total_time
                ms_per_sample = (total_time * 1000) / total_samples

                speed_data.append(
                    {
                        "model_name": model_name,
                        "total_samples": total_samples,
                        "total_time_s": total_time,
                        "samples_per_second": samples_per_second,
                        "ms_per_sample": ms_per_sample,
                    }
                )

                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                speed_data.append(
                    {
                        "model_name": model_name,
                        "total_samples": 0,
                        "total_time_s": 0,
                        "samples_per_second": 0,
                        "ms_per_sample": 0,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(speed_data)

    def train_and_compare(
        self, epochs: int = 3, checkpoint_dir: str = "models/comparison"
    ) -> pd.DataFrame:
        """
        Train all models and compare their performance.

        Args:
            epochs: Number of epochs to train each model
            checkpoint_dir: Directory to save model checkpoints

        Returns:
            DataFrame with training comparison results
        """
        if self.data_loaders is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print(
            f"Training and comparing {len(self.model_names)} models for {epochs} epochs..."
        )

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)

        train_loader, valid_loader, test_loader = self.data_loaders
        comparison_data = []

        for model_name in self.model_names:
            try:
                print(f"\n{'='*50}")
                print(f"Training {model_name}")
                print(f"{'='*50}")

                # Create model
                model = create_model(model_name, device=self.device)
                model_checkpoint = checkpoint_path / f"{model_name}_best.pt"

                # Train model
                start_time = time.time()
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=epochs,
                    device=self.device,
                    checkpoint_path=str(model_checkpoint),
                )
                training_time = time.time() - start_time

                # Get best metrics
                best_train_acc = max(history["train_acc"])
                best_val_acc = max(history["val_acc"])
                final_train_loss = history["train_loss"][-1]
                final_val_loss = history["val_loss"][-1]

                # Evaluate on test set
                model = load_model_checkpoint(model, str(model_checkpoint), self.device)
                test_results = evaluate_model_on_test_set(
                    model, test_loader, self.labels.names, self.device
                )

                comparison_data.append(
                    {
                        "model_name": model_name,
                        "training_time_s": training_time,
                        "epochs": epochs,
                        "best_train_acc": best_train_acc,
                        "best_val_acc": best_val_acc,
                        "final_train_loss": final_train_loss,
                        "final_val_loss": final_val_loss,
                        "test_accuracy": test_results["accuracy"],
                        "checkpoint_path": str(model_checkpoint),
                    }
                )

                # Store detailed results
                self.results[model_name] = {
                    "history": history,
                    "test_results": test_results,
                    "model": model,
                    "checkpoint_path": str(model_checkpoint),
                }

                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                comparison_data.append(
                    {
                        "model_name": model_name,
                        "training_time_s": 0,
                        "epochs": epochs,
                        "best_train_acc": 0,
                        "best_val_acc": 0,
                        "final_train_loss": float("inf"),
                        "final_val_loss": float("inf"),
                        "test_accuracy": 0,
                        "error": str(e),
                    }
                )

        return pd.DataFrame(comparison_data)

    def plot_comparison_results(
        self, results_df: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Plot comparison results.

        Args:
            results_df: DataFrame with comparison results
            save_path: Path to save the plot
        """
        # Filter out error rows
        valid_results = results_df[~results_df.get("error", pd.Series(False))]

        if valid_results.empty:
            print("No valid results to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Comparison Results", fontsize=16, fontweight="bold")

        # 1. Model complexity (parameters vs accuracy)
        if (
            "total_parameters" in valid_results.columns
            and "test_accuracy" in valid_results.columns
        ):
            ax1 = axes[0, 0]
            scatter = ax1.scatter(
                valid_results["total_parameters"],
                valid_results["test_accuracy"],
                s=100,
                alpha=0.7,
            )
            ax1.set_xlabel("Total Parameters")
            ax1.set_ylabel("Test Accuracy")
            ax1.set_title("Parameters vs Accuracy")

            # Add model name annotations
            for idx, row in valid_results.iterrows():
                ax1.annotate(
                    row["model_name"],
                    (row["total_parameters"], row["test_accuracy"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        # 2. Training time vs accuracy
        if (
            "training_time_s" in valid_results.columns
            and "test_accuracy" in valid_results.columns
        ):
            ax2 = axes[0, 1]
            ax2.scatter(
                valid_results["training_time_s"],
                valid_results["test_accuracy"],
                s=100,
                alpha=0.7,
                color="orange",
            )
            ax2.set_xlabel("Training Time (seconds)")
            ax2.set_ylabel("Test Accuracy")
            ax2.set_title("Training Time vs Accuracy")

            for idx, row in valid_results.iterrows():
                ax2.annotate(
                    row["model_name"],
                    (row["training_time_s"], row["test_accuracy"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        # 3. Inference speed comparison
        if "samples_per_second" in valid_results.columns:
            ax3 = axes[1, 0]
            bars = ax3.bar(
                valid_results["model_name"], valid_results["samples_per_second"]
            )
            ax3.set_xlabel("Model")
            ax3.set_ylabel("Samples per Second")
            ax3.set_title("Inference Speed Comparison")
            ax3.tick_params(axis="x", rotation=45)

        # 4. Test accuracy comparison
        if "test_accuracy" in valid_results.columns:
            ax4 = axes[1, 1]
            bars = ax4.bar(valid_results["model_name"], valid_results["test_accuracy"])
            ax4.set_xlabel("Model")
            ax4.set_ylabel("Test Accuracy")
            ax4.set_title("Test Accuracy Comparison")
            ax4.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, acc in zip(bars, valid_results["test_accuracy"]):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    def generate_comparison_report(
        self, results_df: pd.DataFrame, output_path: str = "model_comparison_report.md"
    ):
        """
        Generate a markdown report with comparison results.

        Args:
            results_df: DataFrame with comparison results
            output_path: Path to save the report
        """
        with open(output_path, "w") as f:
            f.write("# Model Comparison Report\n\n")
            f.write(
                "This report contains the results of comparing multiple model architectures "
            )
            f.write("for drone detection.\n\n")

            f.write("## Model Overview\n\n")
            f.write(
                "| Model | Parameters | Size (MB) | Test Accuracy | Training Time (s) |\n"
            )
            f.write(
                "|-------|------------|-----------|---------------|------------------|\n"
            )

            for _, row in results_df.iterrows():
                if "error" not in row or pd.isna(row.get("error")):
                    params = (
                        f"{row.get('total_parameters', 0):,}"
                        if "total_parameters" in row
                        else "N/A"
                    )
                    size = (
                        f"{row.get('model_size_mb', 0):.2f}"
                        if "model_size_mb" in row
                        else "N/A"
                    )
                    acc = (
                        f"{row.get('test_accuracy', 0):.4f}"
                        if "test_accuracy" in row
                        else "N/A"
                    )
                    time = (
                        f"{row.get('training_time_s', 0):.1f}"
                        if "training_time_s" in row
                        else "N/A"
                    )

                    f.write(
                        f"| {row['model_name']} | {params} | {size} | {acc} | {time} |\n"
                    )

            f.write("\n## Key Findings\n\n")

            if "test_accuracy" in results_df.columns:
                best_acc_model = results_df.loc[results_df["test_accuracy"].idxmax()]
                f.write(f"- **Best Accuracy**: {best_acc_model['model_name']} ")
                f.write(f"({best_acc_model['test_accuracy']:.4f})\n")

            if "training_time_s" in results_df.columns:
                fastest_training = results_df.loc[
                    results_df["training_time_s"].idxmin()
                ]
                f.write(f"- **Fastest Training**: {fastest_training['model_name']} ")
                f.write(f"({fastest_training['training_time_s']:.1f}s)\n")

            if "samples_per_second" in results_df.columns:
                fastest_inference = results_df.loc[
                    results_df["samples_per_second"].idxmax()
                ]
                f.write(f"- **Fastest Inference**: {fastest_inference['model_name']} ")
                f.write(f"({fastest_inference['samples_per_second']:.1f} samples/s)\n")

            if "total_parameters" in results_df.columns:
                smallest_model = results_df.loc[results_df["total_parameters"].idxmin()]
                f.write(f"- **Smallest Model**: {smallest_model['model_name']} ")
                f.write(f"({smallest_model['total_parameters']:,} parameters)\n")

        print(f"Comparison report saved to {output_path}")


def quick_model_comparison(
    model_names: Optional[List[str]] = None, epochs: int = 3
) -> Dict[str, Any]:
    """
    Quick function to compare models with training and evaluation.

    Args:
        model_names: List of model names to compare
        epochs: Number of epochs to train each model

    Returns:
        Dictionary with comparison results
    """
    comparator = ModelComparator(model_names)
    comparator.load_data()

    # Compare model complexity
    complexity_df = comparator.compare_model_complexity()
    print("\nModel Complexity Comparison:")
    print(complexity_df.to_string(index=False))

    # Benchmark inference speed
    speed_df = comparator.benchmark_inference_speed()
    print("\nInference Speed Comparison:")
    print(speed_df.to_string(index=False))

    # Train and compare
    training_df = comparator.train_and_compare(epochs)
    print("\nTraining Comparison:")
    print(training_df.to_string(index=False))

    # Combine results
    combined_df = complexity_df.merge(speed_df, on="model_name", how="outer")
    combined_df = combined_df.merge(training_df, on="model_name", how="outer")

    # Plot results
    comparator.plot_comparison_results(combined_df)

    # Generate report
    comparator.generate_comparison_report(combined_df)

    return {
        "complexity": complexity_df,
        "speed": speed_df,
        "training": training_df,
        "combined": combined_df,
        "comparator": comparator,
    }
