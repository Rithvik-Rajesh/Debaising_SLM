"""
Complete pipeline for debiasing small language models.
This module integrates data processing, training, evaluation, and visualization.
"""

import os
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
import argparse

from src.preprocess.data_splitter import DataSplitter, split_and_save_data
from src.debiaser.trainer import DebiasTrainer, TrainingConfig, load_training_data
from src.debiaser.visualization import TrainingVisualizer, BiasAnalyzer
from src.debiaser.inference import DebiasedModelInference, load_test_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""

    # Data configuration
    input_data_path: str = "data/processed/dataset.json"
    splits_dir: str = "data/processed/splits"
    test_size: float = 0.2
    val_size: float = 0.1

    # Training configuration
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    alpha: float = 0.5  # Balance between accuracy and consistency loss

    # Output configuration
    output_dir: str = "outputs"
    models_dir: str = "outputs/models"
    reports_dir: str = "outputs/reports"
    visualizations_dir: str = "outputs/visualizations"

    # Runtime configuration
    device: str = "auto"
    seed: int = 42

    def to_training_config(self) -> TrainingConfig:
        """Convert to TrainingConfig."""
        return TrainingConfig(
            model_name=self.model_name,
            num_labels=self.num_labels,
            max_length=self.max_length,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            alpha=self.alpha,
            device=self.device,
            seed=self.seed,
            output_dir=self.models_dir,
        )


class DebiasePipeline:
    """Complete pipeline for debiasing small language models."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._setup_directories()

        # Initialize components
        self.data_splitter = DataSplitter(
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.seed,
        )

        self.visualizer = TrainingVisualizer(config.visualizations_dir)
        self.bias_analyzer = BiasAnalyzer(config.visualizations_dir)

        logger.info("Pipeline initialized successfully")

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config.output_dir,
            self.config.models_dir,
            self.config.reports_dir,
            self.config.visualizations_dir,
            self.config.splits_dir,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"Created directories: {directories}")

    def prepare_data(self, force_resplit: bool = False) -> None:
        """
        Prepare and split the data.

        Args:
            force_resplit: Whether to force re-splitting even if splits exist
        """
        logger.info("Starting data preparation...")

        # Check if splits already exist
        train_path = os.path.join(self.config.splits_dir, "train.json")
        val_path = os.path.join(self.config.splits_dir, "val.json")
        test_path = os.path.join(self.config.splits_dir, "test.json")

        if not force_resplit and all(
            os.path.exists(p) for p in [train_path, val_path, test_path]
        ):
            logger.info(
                "Data splits already exist. Use force_resplit=True to recreate."
            )
            return

        # Split data
        split_and_save_data(
            input_file=self.config.input_data_path,
            output_dir=self.config.splits_dir,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.seed,
        )

        logger.info("Data preparation completed")

    def train_model(self) -> Dict[str, Any]:
        """
        Train the debiasing model.

        Returns:
            Training history and final metrics
        """
        logger.info("Starting model training...")

        # Load training data
        train_data, val_data, _ = load_training_data(self.config.splits_dir)

        # Initialize trainer
        training_config = self.config.to_training_config()
        trainer = DebiasTrainer(training_config)

        # Train model
        history = trainer.train(train_data, val_data)

        # Evaluate on validation set
        val_dataloader = trainer.create_dataloader(val_data, shuffle=False)
        final_metrics = trainer.evaluate(val_dataloader)

        # Save training history
        history_path = os.path.join(self.config.reports_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save final metrics
        metrics_path = os.path.join(self.config.reports_dir, "final_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Training completed. History saved to {history_path}")

        return {"history": history, "final_metrics": final_metrics, "trainer": trainer}

    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Args:
            model_path: Path to model checkpoint. If None, uses best_model.

        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation...")

        # Determine model path
        if model_path is None:
            model_path = os.path.join(self.config.models_dir, "best_model")

        if not os.path.exists(model_path):
            raise ValueError(
                f"Model not found at {model_path}. Please train a model first."
            )

        # Load test data
        test_data_path = os.path.join(self.config.splits_dir, "test.json")
        test_pairs = load_test_data(test_data_path)

        # Initialize inference
        inference = DebiasedModelInference(model_path, device=self.config.device)

        # Evaluate bias consistency
        pro_sentences = [pair[0] for pair in test_pairs]
        anti_sentences = [pair[1] for pair in test_pairs]

        bias_metrics = inference.evaluate_bias_consistency(
            pro_sentences, anti_sentences
        )

        # Detailed analysis on a sample
        sample_size = min(100, len(test_pairs))
        sample_pairs = test_pairs[:sample_size]

        analysis_results = inference.batch_analyze_pairs(
            sample_pairs,
            os.path.join(self.config.reports_dir, "detailed_bias_analysis.json"),
        )

        # Get predictions for visualization
        pro_predictions, pro_probs = inference.predict_batch(
            pro_sentences, return_probabilities=True
        )
        anti_predictions, anti_probs = inference.predict_batch(
            anti_sentences, return_probabilities=True
        )

        # Convert probabilities to numpy for analysis
        import numpy as np

        pro_probs = np.array(pro_probs)[:, 1]  # Positive class probabilities
        anti_probs = np.array(anti_probs)[:, 1]

        # Create bias consistency visualization
        self.visualizer.plot_bias_consistency_analysis(
            pro_probs.tolist(), anti_probs.tolist(), save_name="test_bias_consistency"
        )

        evaluation_results = {
            "bias_metrics": bias_metrics,
            "sample_analysis": analysis_results,
            "model_path": model_path,
            "test_samples": len(test_pairs),
        }

        # Save evaluation results
        eval_path = os.path.join(self.config.reports_dir, "evaluation_results.json")
        with open(eval_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {
                "bias_metrics": bias_metrics,
                "model_path": model_path,
                "test_samples": len(test_pairs),
            }
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation completed. Results saved to {eval_path}")

        return evaluation_results

    def create_visualizations(
        self, training_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create comprehensive visualizations.

        Args:
            training_results: Results from training (history, metrics)
        """
        logger.info("Creating visualizations...")

        # Load training history if not provided
        if training_results is None:
            history_path = os.path.join(
                self.config.reports_dir, "training_history.json"
            )
            metrics_path = os.path.join(self.config.reports_dir, "final_metrics.json")

            if os.path.exists(history_path) and os.path.exists(metrics_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                with open(metrics_path, "r") as f:
                    final_metrics = json.load(f)

                training_results = {"history": history, "final_metrics": final_metrics}
            else:
                logger.warning(
                    "Training results not found. Skipping training visualizations."
                )
                return

        history = training_results["history"]
        final_metrics = training_results["final_metrics"]

        # Create training visualizations
        self.visualizer.plot_training_history(history, "training_history")
        self.visualizer.plot_interactive_training_history(
            history, "interactive_training_history"
        )
        self.visualizer.plot_loss_components(history, "loss_components")
        self.visualizer.plot_evaluation_metrics(final_metrics, "evaluation_metrics")

        # Create comprehensive report
        self.visualizer.create_training_report(
            history, final_metrics, self.config.__dict__, "comprehensive_report"
        )

        logger.info("Visualizations created successfully")

    def run_complete_pipeline(
        self, force_resplit: bool = False, skip_training: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from data preparation to evaluation.

        Args:
            force_resplit: Whether to force data re-splitting
            skip_training: Whether to skip training (use existing model)

        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete debiasing pipeline...")

        # Step 1: Prepare data
        self.prepare_data(force_resplit=force_resplit)

        # Step 2: Train model (unless skipped)
        if not skip_training:
            training_results = self.train_model()
        else:
            training_results = None
            logger.info("Skipping training as requested")

        # Step 3: Evaluate model
        try:
            evaluation_results = self.evaluate_model()
        except ValueError as e:
            logger.error(f"Evaluation failed: {e}")
            if not skip_training:
                raise
            else:
                logger.warning("Skipping evaluation due to missing model")
                evaluation_results = None

        # Step 4: Create visualizations
        self.create_visualizations(training_results)

        # Compile results
        pipeline_results = {
            "config": self.config.__dict__,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "pipeline_completed": True,
        }

        # Save complete results
        results_path = os.path.join(self.config.reports_dir, "pipeline_results.json")
        with open(results_path, "w") as f:
            # Create a JSON-serializable version
            serializable_results = {
                "config": self.config.__dict__,
                "training_completed": training_results is not None,
                "evaluation_completed": evaluation_results is not None,
                "pipeline_completed": True,
            }
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Complete pipeline finished! Results saved to {results_path}")

        return pipeline_results


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Debiasing Small Language Models Pipeline"
    )

    # Data arguments
    parser.add_argument(
        "--input-data",
        default="data/processed/dataset.json",
        help="Path to input dataset",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/processed/splits",
        help="Directory for data splits",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation set size (0.0-1.0)"
    )

    # Training arguments
    parser.add_argument(
        "--model-name", default="distilbert-base-uncased", help="Pre-trained model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Balance between accuracy and consistency loss",
    )

    # Output arguments
    parser.add_argument("--output-dir", default="outputs", help="Output directory")

    # Runtime arguments
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Pipeline control
    parser.add_argument(
        "--force-resplit", action="store_true", help="Force data re-splitting"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training step"
    )
    parser.add_argument(
        "--only-evaluate", action="store_true", help="Only run evaluation"
    )
    parser.add_argument(
        "--only-visualize", action="store_true", help="Only create visualizations"
    )

    args = parser.parse_args()

    # Create pipeline configuration
    config = PipelineConfig(
        input_data_path=args.input_data,
        splits_dir=args.splits_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        output_dir=args.output_dir,
        models_dir=os.path.join(args.output_dir, "models"),
        reports_dir=os.path.join(args.output_dir, "reports"),
        visualizations_dir=os.path.join(args.output_dir, "visualizations"),
        device=args.device,
        seed=args.seed,
    )

    # Initialize pipeline
    pipeline = DebiasePipeline(config)

    # Run pipeline based on arguments
    if args.only_evaluate:
        logger.info("Running evaluation only...")
        pipeline.evaluate_model()
    elif args.only_visualize:
        logger.info("Creating visualizations only...")
        pipeline.create_visualizations()
    else:
        logger.info("Running complete pipeline...")
        pipeline.run_complete_pipeline(
            force_resplit=args.force_resplit, skip_training=args.skip_training
        )


if __name__ == "__main__":
    # Example usage when called directly
    config = PipelineConfig(
        num_epochs=2,  # Reduced for testing
        batch_size=8,  # Smaller batch for testing
        alpha=0.3,
    )

    pipeline = DebiasePipeline(config)
    results = pipeline.run_complete_pipeline()

    logger.info("Pipeline execution completed!")
