"""
Advanced trainer for debiasing language models.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    alpha: float = 0.5  # Balance between accuracy and consistency loss
    device: str = "auto"
    seed: int = 42
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    output_dir: str = "outputs/models"


class BiasDataset(Dataset):
    """Dataset class for bias sentence pairs."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        labels: Optional[List[int]] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dictionaries with 'pro' and 'anti' keys
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            labels: Optional labels for supervised learning
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels if labels else [1] * len(data)  # Default positive labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Clean sentences by removing brackets
        pro_sentence = item["pro"].replace("[", "").replace("]", "")
        anti_sentence = item["anti"].replace("[", "").replace("]", "")

        # Tokenize both sentences
        pro_encoding = self.tokenizer(
            pro_sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        anti_encoding = self.tokenizer(
            anti_sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pro_input_ids": pro_encoding["input_ids"].flatten(),
            "pro_attention_mask": pro_encoding["attention_mask"].flatten(),
            "anti_input_ids": anti_encoding["input_ids"].flatten(),
            "anti_attention_mask": anti_encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class DebiasTrainer:
    """Advanced trainer for debiasing language models."""

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self._set_seed()
        self.device = self._get_device()

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels
        ).to(self.device)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(f"Initialized DebiasTrainer with device: {self.device}")

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def create_dataloader(
        self,
        data: List[Dict[str, str]],
        labels: Optional[List[int]] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader from data.

        Args:
            data: List of bias sentence pairs
            labels: Optional labels
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        dataset = BiasDataset(data, self.tokenizer, self.config.max_length, labels)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def compute_loss(
        self, pro_outputs, anti_outputs, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss (accuracy + consistency).

        Args:
            pro_outputs: Model outputs for pro sentences
            anti_outputs: Model outputs for anti sentences
            labels: True labels

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        pro_logits = pro_outputs.logits
        anti_logits = anti_outputs.logits

        # Consistency loss: minimize difference between pro and anti predictions
        consistency_loss = F.mse_loss(pro_logits, anti_logits)

        # Accuracy loss: standard cross-entropy
        combined_logits = torch.cat([pro_logits, anti_logits], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        accuracy_loss = F.cross_entropy(combined_logits, combined_labels)

        # Combined loss
        total_loss = (
            1 - self.config.alpha
        ) * accuracy_loss + self.config.alpha * consistency_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "accuracy_loss": accuracy_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }

        return total_loss, loss_dict

    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.

        Args:
            eval_dataloader: DataLoader for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy_loss = 0
        total_consistency_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass for pro sentences
                pro_outputs = self.model(
                    input_ids=batch["pro_input_ids"],
                    attention_mask=batch["pro_attention_mask"],
                )

                # Forward pass for anti sentences
                anti_outputs = self.model(
                    input_ids=batch["anti_input_ids"],
                    attention_mask=batch["anti_attention_mask"],
                )

                # Compute loss
                loss, loss_dict = self.compute_loss(
                    pro_outputs, anti_outputs, batch["labels"]
                )

                total_loss += loss_dict["total_loss"]
                total_accuracy_loss += loss_dict["accuracy_loss"]
                total_consistency_loss += loss_dict["consistency_loss"]

                # Collect predictions for metrics
                pro_predictions = torch.argmax(pro_outputs.logits, dim=-1)
                all_predictions.extend(pro_predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(eval_dataloader)
        avg_accuracy_loss = total_accuracy_loss / len(eval_dataloader)
        avg_consistency_loss = total_consistency_loss / len(eval_dataloader)

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        precision = precision_score(all_labels, all_predictions, average="weighted")
        recall = recall_score(all_labels, all_predictions, average="weighted")

        return {
            "eval_loss": avg_loss,
            "eval_accuracy_loss": avg_accuracy_loss,
            "eval_consistency_loss": avg_consistency_loss,
            "eval_accuracy": accuracy,
            "eval_f1": f1,
            "eval_precision": precision,
            "eval_recall": recall,
        }

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
        train_labels: Optional[List[int]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_data: Training data
            val_data: Validation data
            train_labels: Training labels
            val_labels: Validation labels

        Returns:
            Dictionary of training history
        """
        # Create dataloaders
        train_dataloader = self.create_dataloader(
            train_data, train_labels, shuffle=True
        )
        val_dataloader = None
        if val_data:
            val_dataloader = self.create_dataloader(val_data, val_labels, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training history
        history = {
            "train_loss": [],
            "train_accuracy_loss": [],
            "train_consistency_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        logger.info(f"Starting training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            total_accuracy_loss = 0
            total_consistency_loss = 0

            progress_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass for pro sentences
                pro_outputs = self.model(
                    input_ids=batch["pro_input_ids"],
                    attention_mask=batch["pro_attention_mask"],
                )

                # Forward pass for anti sentences
                anti_outputs = self.model(
                    input_ids=batch["anti_input_ids"],
                    attention_mask=batch["anti_attention_mask"],
                )

                # Compute loss
                loss, loss_dict = self.compute_loss(
                    pro_outputs, anti_outputs, batch["labels"]
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update running totals
                total_loss += loss_dict["total_loss"]
                total_accuracy_loss += loss_dict["accuracy_loss"]
                total_consistency_loss += loss_dict["consistency_loss"]

                self.global_step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_dict['total_loss']:.4f}",
                        "acc_loss": f"{loss_dict['accuracy_loss']:.4f}",
                        "cons_loss": f"{loss_dict['consistency_loss']:.4f}",
                    }
                )

                # Evaluation
                if val_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    logger.info(f"Step {self.global_step} - {eval_metrics}")

                    # Save best model
                    if eval_metrics["eval_loss"] < self.best_val_loss:
                        self.best_val_loss = eval_metrics["eval_loss"]
                        self.save_model("best_model")
                        logger.info(
                            f"New best model saved with validation loss: {self.best_val_loss:.4f}"
                        )

                    self.model.train()

                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint-{self.global_step}")

            # End of epoch evaluation
            avg_train_loss = total_loss / len(train_dataloader)
            avg_accuracy_loss = total_accuracy_loss / len(train_dataloader)
            avg_consistency_loss = total_consistency_loss / len(train_dataloader)

            history["train_loss"].append(avg_train_loss)
            history["train_accuracy_loss"].append(avg_accuracy_loss)
            history["train_consistency_loss"].append(avg_consistency_loss)

            if val_dataloader:
                eval_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(eval_metrics["eval_loss"])
                history["val_accuracy"].append(eval_metrics["eval_accuracy"])
                history["val_f1"].append(eval_metrics["eval_f1"])

                logger.info(
                    f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {eval_metrics['eval_loss']:.4f}, "
                    f"Val Acc: {eval_metrics['eval_accuracy']:.4f}"
                )

        # Save final model
        self.save_model("final_model")
        logger.info("Training completed!")

        return history

    def save_model(self, checkpoint_name: str) -> None:
        """
        Save model and tokenizer.

        Args:
            checkpoint_name: Name for the checkpoint
        """
        save_path = os.path.join(self.config.output_dir, checkpoint_name)
        os.makedirs(save_path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config
        config_path = os.path.join(save_path, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info(f"Model loaded from {checkpoint_path}")


def load_training_data(
    data_dir: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load training data splits.

    Args:
        data_dir: Directory containing train.json, val.json, test.json

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    splits = []
    for filename in ["train.json", "val.json", "test.json"]:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        splits.append(data)

    return tuple(splits)


if __name__ == "__main__":
    # Example usage
    from src.preprocess.data_splitter import split_and_save_data

    # Split data if not already done
    input_file = "data/processed/dataset.json"
    splits_dir = "data/processed/splits"

    if not os.path.exists(splits_dir):
        split_and_save_data(input_file, splits_dir)

    # Load data
    train_data, val_data, test_data = load_training_data(splits_dir)

    # Initialize trainer
    config = TrainingConfig(num_epochs=2, batch_size=8, learning_rate=2e-5, alpha=0.3)

    trainer = DebiasTrainer(config)

    # Train model
    history = trainer.train(train_data, val_data)

    # Evaluate on test set
    test_dataloader = trainer.create_dataloader(test_data, shuffle=False)
    test_metrics = trainer.evaluate(test_dataloader)
    logger.info(f"Test metrics: {test_metrics}")
