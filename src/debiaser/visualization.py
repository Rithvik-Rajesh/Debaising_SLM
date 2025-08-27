"""
Visualization utilities for training metrics and bias analysis.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizer for training metrics and analysis."""

    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized visualizer with output directory: {output_dir}")

    def plot_training_history(
        self, history: Dict[str, List[float]], save_name: str = "training_history"
    ) -> None:
        """
        Plot training history metrics.

        Args:
            history: Dictionary containing training metrics
            save_name: Name for saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16, fontweight="bold")

        # Loss plot
        axes[0, 0].plot(history["train_loss"], label="Training Loss", linewidth=2)
        if "val_loss" in history and history["val_loss"]:
            axes[0, 0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy loss
        if "train_accuracy_loss" in history:
            axes[0, 1].plot(
                history["train_accuracy_loss"],
                label="Accuracy Loss",
                linewidth=2,
                color="orange",
            )
            axes[0, 1].set_title("Accuracy Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Consistency loss
        if "train_consistency_loss" in history:
            axes[1, 0].plot(
                history["train_consistency_loss"],
                label="Consistency Loss",
                linewidth=2,
                color="green",
            )
            axes[1, 0].set_title("Consistency Loss (Debiasing)")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Validation metrics
        if "val_accuracy" in history and history["val_accuracy"]:
            axes[1, 1].plot(
                history["val_accuracy"], label="Accuracy", linewidth=2, color="red"
            )
            if "val_f1" in history and history["val_f1"]:
                axes[1, 1].plot(
                    history["val_f1"], label="F1 Score", linewidth=2, color="blue"
                )
            axes[1, 1].set_title("Validation Metrics")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")

    def plot_interactive_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = "interactive_training_history",
    ) -> None:
        """
        Create interactive training history plot using Plotly.

        Args:
            history: Dictionary containing training metrics
            save_name: Name for saved plot
        """
        epochs = list(range(1, len(history["train_loss"]) + 1))

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Total Loss",
                "Component Losses",
                "Validation Metrics",
                "Loss Comparison",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}],
            ],
        )

        # Total loss
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                name="Training Loss",
                line=dict(width=3),
            ),
            row=1,
            col=1,
        )
        if "val_loss" in history and history["val_loss"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val_loss"],
                    name="Validation Loss",
                    line=dict(width=3),
                ),
                row=1,
                col=1,
            )

        # Component losses
        if "train_accuracy_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train_accuracy_loss"],
                    name="Accuracy Loss",
                    line=dict(width=2),
                ),
                row=1,
                col=2,
            )
        if "train_consistency_loss" in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train_consistency_loss"],
                    name="Consistency Loss",
                    line=dict(width=2),
                ),
                row=1,
                col=2,
            )

        # Validation metrics
        if "val_accuracy" in history and history["val_accuracy"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["val_accuracy"],
                    name="Accuracy",
                    line=dict(width=2),
                ),
                row=2,
                col=1,
            )
        if "val_f1" in history and history["val_f1"]:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=history["val_f1"], name="F1 Score", line=dict(width=2)
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

        # Loss comparison
        if len(history["train_loss"]) > 0:
            train_loss_normalized = np.array(history["train_loss"]) / max(
                history["train_loss"]
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_loss_normalized,
                    name="Train Loss (Normalized)",
                    line=dict(width=2),
                ),
                row=2,
                col=2,
            )
            if "val_loss" in history and history["val_loss"]:
                val_loss_normalized = np.array(history["val_loss"]) / max(
                    history["val_loss"]
                )
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=val_loss_normalized,
                        name="Val Loss (Normalized)",
                        line=dict(width=2),
                    ),
                    row=2,
                    col=2,
                )

        fig.update_layout(
            title_text="Interactive Training History", height=800, showlegend=True
        )

        # Save interactive plot
        save_path = os.path.join(self.output_dir, f"{save_name}.html")
        fig.write_html(save_path)
        logger.info(f"Interactive training history plot saved to {save_path}")

    def plot_loss_components(
        self, history: Dict[str, List[float]], save_name: str = "loss_components"
    ) -> None:
        """
        Plot detailed loss component analysis.

        Args:
            history: Dictionary containing training metrics
            save_name: Name for saved plot
        """
        if (
            "train_accuracy_loss" not in history
            or "train_consistency_loss" not in history
        ):
            logger.warning("Loss components not available in history")
            return

        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Stacked area plot
        ax1.fill_between(
            epochs,
            history["train_accuracy_loss"],
            alpha=0.7,
            label="Accuracy Loss",
            color="orange",
        )
        ax1.fill_between(
            epochs,
            history["train_accuracy_loss"],
            [
                a + c
                for a, c in zip(
                    history["train_accuracy_loss"], history["train_consistency_loss"]
                )
            ],
            alpha=0.7,
            label="Consistency Loss",
            color="green",
        )
        ax1.set_title("Loss Components (Stacked)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Ratio plot
        ratios = [
            c / (a + c) if (a + c) > 0 else 0
            for a, c in zip(
                history["train_accuracy_loss"], history["train_consistency_loss"]
            )
        ]
        ax2.plot(epochs, ratios, linewidth=3, color="purple", marker="o")
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Equal Weight")
        ax2.set_title("Consistency Loss Ratio")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Consistency Loss / Total Loss")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Loss components plot saved to {save_path}")

    def plot_evaluation_metrics(
        self, metrics: Dict[str, float], save_name: str = "evaluation_metrics"
    ) -> None:
        """
        Plot evaluation metrics as a radar chart.

        Args:
            metrics: Dictionary of evaluation metrics
            save_name: Name for saved plot
        """
        # Filter metrics for radar chart
        radar_metrics = {
            k.replace("eval_", "").replace("_", " ").title(): v
            for k, v in metrics.items()
            if k.startswith("eval_") and "loss" not in k
        }

        if not radar_metrics:
            logger.warning("No suitable metrics found for radar chart")
            return

        # Create radar chart
        fig = go.Figure()

        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name="Model Performance",
                line_color="rgb(32, 201, 151)",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Evaluation Metrics Radar Chart",
            showlegend=True,
        )

        save_path = os.path.join(self.output_dir, f"{save_name}.html")
        fig.write_html(save_path)
        logger.info(f"Evaluation metrics radar chart saved to {save_path}")

    def plot_bias_consistency_analysis(
        self,
        pro_predictions: List[float],
        anti_predictions: List[float],
        sentences: Optional[List[Tuple[str, str]]] = None,
        save_name: str = "bias_consistency",
    ) -> None:
        """
        Analyze bias consistency between pro and anti sentences.

        Args:
            pro_predictions: Predictions for pro-stereotyped sentences
            anti_predictions: Predictions for anti-stereotyped sentences
            sentences: Optional list of (pro, anti) sentence pairs
            save_name: Name for saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Bias Consistency Analysis", fontsize=16, fontweight="bold")

        # Scatter plot of predictions
        axes[0, 0].scatter(pro_predictions, anti_predictions, alpha=0.6, s=50)
        axes[0, 0].plot([0, 1], [0, 1], "r--", label="Perfect Consistency")
        axes[0, 0].set_xlabel("Pro-stereotyped Predictions")
        axes[0, 0].set_ylabel("Anti-stereotyped Predictions")
        axes[0, 0].set_title("Prediction Consistency")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Difference histogram
        differences = np.array(pro_predictions) - np.array(anti_predictions)
        axes[0, 1].hist(
            differences, bins=30, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[0, 1].axvline(
            x=0, color="red", linestyle="--", label="Perfect Consistency"
        )
        axes[0, 1].axvline(
            x=np.mean(differences),
            color="blue",
            linestyle="-",
            label=f"Mean: {np.mean(differences):.3f}",
        )
        axes[0, 1].set_xlabel("Prediction Difference (Pro - Anti)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Prediction Differences Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot comparison
        data = [pro_predictions, anti_predictions]
        axes[1, 0].boxplot(data, labels=["Pro-stereotyped", "Anti-stereotyped"])
        axes[1, 0].set_ylabel("Prediction Score")
        axes[1, 0].set_title("Prediction Distribution Comparison")
        axes[1, 0].grid(True, alpha=0.3)

        # Consistency score over examples
        consistency_scores = 1 - np.abs(differences)
        axes[1, 1].plot(consistency_scores, alpha=0.7, linewidth=1)
        axes[1, 1].axhline(
            y=np.mean(consistency_scores),
            color="red",
            linestyle="--",
            label=f"Mean Consistency: {np.mean(consistency_scores):.3f}",
        )
        axes[1, 1].set_xlabel("Example Index")
        axes[1, 1].set_ylabel("Consistency Score")
        axes[1, 1].set_title("Consistency Score per Example")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Bias consistency analysis saved to {save_path}")

        # Calculate and log statistics
        consistency_mean = np.mean(consistency_scores)
        consistency_std = np.std(consistency_scores)
        bias_magnitude = np.mean(np.abs(differences))

        stats = {
            "consistency_mean": consistency_mean,
            "consistency_std": consistency_std,
            "bias_magnitude": bias_magnitude,
            "perfect_consistency_pct": np.mean(np.abs(differences) < 0.01) * 100,
        }

        stats_path = os.path.join(self.output_dir, f"{save_name}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Consistency statistics: {stats}")

    def create_training_report(
        self,
        history: Dict[str, List[float]],
        final_metrics: Dict[str, float],
        config_dict: Dict[str, Any],
        save_name: str = "training_report",
    ) -> None:
        """
        Create a comprehensive training report.

        Args:
            history: Training history
            final_metrics: Final evaluation metrics
            config_dict: Training configuration
            save_name: Name for saved report
        """
        # Create all plots
        self.plot_training_history(history, f"{save_name}_history")
        self.plot_interactive_training_history(history, f"{save_name}_interactive")
        self.plot_loss_components(history, f"{save_name}_components")
        self.plot_evaluation_metrics(final_metrics, f"{save_name}_metrics")

        # Create summary report
        report = {
            "training_config": config_dict,
            "training_summary": {
                "total_epochs": len(history["train_loss"]),
                "final_train_loss": history["train_loss"][-1]
                if history["train_loss"]
                else None,
                "final_val_loss": history["val_loss"][-1]
                if "val_loss" in history and history["val_loss"]
                else None,
                "best_val_accuracy": max(history["val_accuracy"])
                if "val_accuracy" in history and history["val_accuracy"]
                else None,
                "final_val_accuracy": history["val_accuracy"][-1]
                if "val_accuracy" in history and history["val_accuracy"]
                else None,
            },
            "final_metrics": final_metrics,
            "training_history": history,
        }

        report_path = os.path.join(self.output_dir, f"{save_name}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Comprehensive training report saved to {report_path}")


class BiasAnalyzer:
    """Analyzer for bias patterns in model predictions."""

    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_occupation_bias(
        self,
        predictions: Dict[str, List[float]],
        occupations: List[str],
        save_name: str = "occupation_bias",
    ) -> None:
        """
        Analyze bias patterns across different occupations.

        Args:
            predictions: Dictionary with 'pro' and 'anti' prediction lists
            occupations: List of occupations corresponding to predictions
            save_name: Name for saved analysis
        """
        # Create DataFrame for analysis
        df = pd.DataFrame(
            {
                "occupation": occupations * 2,
                "prediction": predictions["pro"] + predictions["anti"],
                "stereotype": ["pro"] * len(predictions["pro"])
                + ["anti"] * len(predictions["anti"]),
            }
        )

        # Calculate bias scores per occupation
        bias_scores = []
        for occ in set(occupations):
            occ_data = df[df["occupation"] == occ]
            pro_mean = occ_data[occ_data["stereotype"] == "pro"]["prediction"].mean()
            anti_mean = occ_data[occ_data["stereotype"] == "anti"]["prediction"].mean()
            bias_score = abs(pro_mean - anti_mean)
            bias_scores.append(
                {
                    "occupation": occ,
                    "bias_score": bias_score,
                    "pro_mean": pro_mean,
                    "anti_mean": anti_mean,
                }
            )

        bias_df = pd.DataFrame(bias_scores).sort_values("bias_score", ascending=False)

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Bias score ranking
        top_biased = bias_df.head(10)
        axes[0, 0].barh(range(len(top_biased)), top_biased["bias_score"])
        axes[0, 0].set_yticks(range(len(top_biased)))
        axes[0, 0].set_yticklabels(top_biased["occupation"])
        axes[0, 0].set_xlabel("Bias Score")
        axes[0, 0].set_title("Top 10 Most Biased Occupations")

        # Distribution of bias scores
        axes[0, 1].hist(bias_df["bias_score"], bins=20, alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(
            bias_df["bias_score"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {bias_df['bias_score'].mean():.3f}",
        )
        axes[0, 1].set_xlabel("Bias Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Bias Scores")
        axes[0, 1].legend()

        # Pro vs Anti predictions by occupation
        sns.scatterplot(
            data=df, x="occupation", y="prediction", hue="stereotype", ax=axes[1, 0]
        )
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
        axes[1, 0].set_title("Predictions by Occupation and Stereotype")

        # Bias score vs prediction variance
        variance_scores = df.groupby("occupation")["prediction"].var().reset_index()
        variance_scores = variance_scores.merge(
            bias_df[["occupation", "bias_score"]], on="occupation"
        )
        axes[1, 1].scatter(variance_scores["prediction"], variance_scores["bias_score"])
        axes[1, 1].set_xlabel("Prediction Variance")
        axes[1, 1].set_ylabel("Bias Score")
        axes[1, 1].set_title("Bias Score vs Prediction Variance")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save detailed analysis
        analysis_path = os.path.join(self.output_dir, f"{save_name}_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(
                {
                    "overall_bias_score": bias_df["bias_score"].mean(),
                    "bias_score_std": bias_df["bias_score"].std(),
                    "most_biased_occupations": bias_df.head(10).to_dict("records"),
                    "least_biased_occupations": bias_df.tail(10).to_dict("records"),
                },
                f,
                indent=2,
            )

        logger.info(
            f"Occupation bias analysis saved to {save_path} and {analysis_path}"
        )


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample training history
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.25],
        "train_accuracy_loss": [0.5, 0.4, 0.3, 0.2, 0.15],
        "train_consistency_loss": [0.3, 0.2, 0.1, 0.1, 0.1],
        "val_loss": [0.7, 0.55, 0.45, 0.35, 0.3],
        "val_accuracy": [0.6, 0.7, 0.8, 0.85, 0.87],
        "val_f1": [0.58, 0.68, 0.78, 0.83, 0.85],
    }

    # Sample evaluation metrics
    metrics = {
        "eval_accuracy": 0.87,
        "eval_f1": 0.85,
        "eval_precision": 0.86,
        "eval_recall": 0.84,
    }

    # Initialize visualizer
    visualizer = TrainingVisualizer()

    # Create visualizations
    visualizer.plot_training_history(history)
    visualizer.plot_interactive_training_history(history)
    visualizer.plot_loss_components(history)
    visualizer.plot_evaluation_metrics(metrics)

    # Sample bias analysis
    pro_preds = np.random.uniform(0.4, 0.9, 100)
    anti_preds = np.random.uniform(0.1, 0.6, 100)
    visualizer.plot_bias_consistency_analysis(pro_preds, anti_preds)

    logger.info("Sample visualizations created!")
