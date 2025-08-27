import os
from src.preprocess.data_splitter import split_and_save_data
from src.debiaser.trainer import DebiasTrainer, TrainingConfig, load_training_data
from src.debiaser.inference import DebiasedModelInference, load_test_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_data():
    input_file = "data/processed/dataset.json"
    splits_dir = "data/processed/splits"

    if not os.path.exists(os.path.join(splits_dir, "train.json")):
        logger.info("Splitting data...")
        split_and_save_data(input_file, splits_dir)
    else:
        logger.info("Data splits already exist")

    return splits_dir


def train_model():
    splits_dir = prepare_data()
    train_data, val_data, test_data = load_training_data(splits_dir)

    config = TrainingConfig(
        model_name="distilbert-base-uncased",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        alpha=0.5,
        output_dir="outputs/models",
    )

    trainer = DebiasTrainer(config)
    history = trainer.train(train_data, val_data)

    logger.info("Training completed!")
    return trainer, history


def evaluate_model(model_path="outputs/models/best_model"):
    test_data_path = "data/processed/splits/test.json"
    test_pairs = load_test_data(test_data_path)

    inference = DebiasedModelInference(model_path)

    pro_sentences = [pair[0] for pair in test_pairs]
    anti_sentences = [pair[1] for pair in test_pairs]

    bias_metrics = inference.evaluate_bias_consistency(pro_sentences, anti_sentences)

    logger.info(f"Bias consistency metrics: {bias_metrics}")
    return bias_metrics


if __name__ == "__main__":
    trainer, history = train_model()

    metrics = evaluate_model()

    logger.info("Pipeline completed successfully!")
