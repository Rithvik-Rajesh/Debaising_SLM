#!/usr/bin/env python3

import argparse
import os
from src.main import train_model, evaluate_model, prepare_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Debias Small Language Models")
    parser.add_argument(
        "mode",
        choices=["prepare", "train", "evaluate", "all"],
        help="Mode to run: prepare data, train model, evaluate model, or run all",
    )
    parser.add_argument(
        "--model-path",
        default="outputs/models/best_model",
        help="Path to model for evaluation",
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_data()
        logger.info("Data preparation completed!")

    elif args.mode == "train":
        train_model()
        logger.info("Training completed!")

    elif args.mode == "evaluate":
        if not os.path.exists(args.model_path):
            logger.error(f"Model not found at {args.model_path}. Please train first.")
            return
        evaluate_model(args.model_path)
        logger.info("Evaluation completed!")

    elif args.mode == "all":
        prepare_data()
        train_model()
        evaluate_model()
        logger.info("Complete pipeline finished!")


if __name__ == "__main__":
    main()
