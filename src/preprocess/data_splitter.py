"""
Data splitting utilities for train/validation/test splits.
"""

import json
import os
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Handles data splitting for bias dataset."""

    def __init__(
        self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42
    ):
        """
        Initialize data splitter.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(
        self, data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], ...]:
        """
        Split data into train, validation, and test sets.

        Args:
            data: List of bias pairs with 'pro' and 'anti' keys

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"Splitting {len(data)} samples...")

        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )

        # Second split: separate validation from training
        train_data, val_data = train_test_split(
            train_val_data, test_size=self.val_size, random_state=self.random_state
        )

        logger.info(
            f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def save_splits(
        self,
        train_data: List[Dict[str, str]],
        val_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
        output_dir: str,
    ) -> None:
        """
        Save data splits to JSON files.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        splits = {
            "train.json": train_data,
            "val.json": val_data,
            "test.json": test_data,
        }

        for filename, data in splits.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} samples to {filepath}")

    def load_splits(self, data_dir: str) -> Tuple[List[Dict[str, str]], ...]:
        """
        Load data splits from JSON files.

        Args:
            data_dir: Directory containing split files

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


def split_and_save_data(
    input_file: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    Convenience function to split and save data.

    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save split files
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    """
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples from {input_file}")

    # Split data
    splitter = DataSplitter(
        test_size=test_size, val_size=val_size, random_state=random_state
    )
    train_data, val_data, test_data = splitter.split_data(data)

    # Save splits
    splitter.save_splits(train_data, val_data, test_data, output_dir)

    logger.info(f"Data splitting complete. Files saved to {output_dir}")


if __name__ == "__main__":
    # Split the processed dataset
    input_file = "data/processed/dataset.json"
    output_dir = "data/processed/splits"

    split_and_save_data(input_file, output_dir)
