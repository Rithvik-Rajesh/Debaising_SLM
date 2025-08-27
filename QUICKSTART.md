# Quick Start Guide

## How to Run the Trainer and Finetune the Model

### 1. Simple Execution

Run the complete pipeline (data preparation, training, and evaluation):
```bash
python src/main.py
```

### 2. Using the CLI Tool

For more control, use the CLI tool:

```bash
# Prepare data only
python run.py prepare

# Train model only
python run.py train

# Evaluate existing model
python run.py evaluate --model-path outputs/models/best_model

# Run complete pipeline
python run.py all
```

### 3. Programmatic Usage

```python
from src.main import train_model, evaluate_model

# Train the model
trainer, history = train_model()

# Evaluate the model
metrics = evaluate_model("outputs/models/best_model")
```

### 4. Configuration

Modify training parameters in `src/main.py`:

```python
config = TrainingConfig(
    model_name="distilbert-base-uncased",  # Base model
    num_epochs=3,                          # Training epochs
    batch_size=16,                         # Batch size
    learning_rate=2e-5,                    # Learning rate
    alpha=0.5,                            # Balance between accuracy and consistency
    output_dir="outputs/models"           # Model save directory
)
```

### 5. Expected Outputs

- **Models**: Saved in `outputs/models/`
  - `best_model/` - Best validation model
  - `final_model/` - Final training model
  - `checkpoint-*/` - Training checkpoints

- **Data**: Split data in `data/processed/splits/`
  - `train.json` - Training data
  - `val.json` - Validation data  
  - `test.json` - Test data

### 6. Requirements

Ensure you have the required dependencies installed:
```bash
uv sync
```