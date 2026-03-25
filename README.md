# Debiasing Small Language Models

A Python toolkit for detecting and reducing **gender bias** in Small Language Models (SLMs) using contrastive fine-tuning. The model is trained to produce consistent predictions regardless of the gender of the subject in a sentence.

---

## Overview

Gender bias in language models manifests when they assign different probabilities or classifications to semantically equivalent sentences that differ only by gendered words (e.g., "he" vs "she"). This project:

1. **Preprocesses** the [WinoBias](https://github.com/uclanlp/corefBias) dataset — stereotyped and anti-stereotyped occupation sentence pairs.
2. **Fine-tunes** a `distilbert-base-uncased` classifier with a dual loss: standard cross-entropy for accuracy + MSE consistency loss to equalise predictions across gender-swapped pairs.
3. **Evaluates** bias reduction by comparing model confidence on *pro-stereotyped* vs *anti-stereotyped* sentences.
4. **Visualises** training progress and bias metrics.

---

## Project Structure

```
Debiasing_SLM/
├── CDA_trainer.ipynb            # Interactive Contrastive Debiasing Analysis notebook
├── data/
│   ├── raw/                     # WinoBias sentence pairs (sets 1–4)
│   │   ├── 1/ … 4/              # Each: anti_stereotyped_type1.txt.dev + pro_stereotyped_type1.txt.dev
│   │   ├── occupations.txt      # Full occupation list
│   │   ├── male_occupations.txt # Male-coded occupations
│   │   └── female_occupations.txt
│   └── processed/
│       ├── dataset.json         # Cleaned (anti, pro) sentence pairs — 3,160 pairs
│       └── splits/              # Generated train/val/test splits (gitignored)
├── output/
│   └── debiased_model/          # Saved PEFT adapter weights + tokenizer
├── src/
│   ├── debiaser/
│   │   ├── __init__.py
│   │   ├── trainer.py           # DebiasTrainer, TrainingConfig, BiasDataset
│   │   ├── inference.py         # DebiasedModelInference — batch prediction & evaluation
│   │   ├── visualization.py     # TrainingVisualizer, BiasAnalyzer (matplotlib + plotly)
│   │   └── masked_lm_demo.py    # Masked-LM bias probe using DistilBERT MLM
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── create_dataset.py    # Builds processed/dataset.json from raw files
│   │   ├── data_splitter.py     # DataSplitter — train/val/test JSON splits
│   │   └── utils.py             # load_file, save_json, clean_occupation helpers
│   ├── benchmarking/
│   │   ├── __init__.py
│   │   └── stereoset.py         # StereoSet evaluator — icat/lm/ss scores via DistilBERT
│   ├── examples/
│   │   ├── __init__.py
│   │   ├── masked_example.py    # Masked-LM prediction demo
│   │   ├── mc_example.py        # Multiple-choice classification demo
│   │   └── biasbios.py          # BiasBios dataset exploration demo
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py            # Shared logger factory (console + optional file)
│   └── pipeline.py              # DebiasePipeline — full end-to-end orchestration
├── run.py                       # CLI entry point
├── pyproject.toml               # Project metadata and dependencies (uv/pip)
├── uv.lock                      # Locked dependency versions
├── .python-version              # Python version pin
└── .gitignore
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. You can also use standard `pip`.

### With uv (recommended)
```bash
# Install uv if you don't have it
curl -Lsf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync
```

### With pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Usage

### Command-Line Interface

All pipeline operations are run through `run.py`:

```bash
# 1. Prepare data — build train/val/test splits from processed/dataset.json
python run.py prepare --input-data data/processed/dataset.json --splits-dir data/processed/splits

# 2. Train the debiasing model
python run.py train --model-name distilbert-base-uncased --num-epochs 3 --alpha 0.5

# 3. Evaluate a trained model
python run.py evaluate --model-path outputs/models/best_model

# 4. Run the full pipeline in one shot
python run.py all

# Extra flags
python run.py --force-resplit all       # Re-split data even if splits exist
python run.py --skip-training evaluate  # Evaluate an existing model
python run.py --only-visualize          # Regenerate plots from saved history
```

Run `python run.py --help` for the full list of arguments.

### Python API

```python
from src.pipeline import DebiasePipeline, PipelineConfig

config = PipelineConfig(
    model_name="distilbert-base-uncased",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    alpha=0.5,          # 0 = pure accuracy loss, 1 = pure consistency loss
    output_dir="outputs",
)

pipeline = DebiasePipeline(config)
results = pipeline.run_complete_pipeline()
```

### Rebuild the Dataset from Raw Files

If you modify the raw data, regenerate `data/processed/dataset.json`:

```bash
python -m src.preprocess.create_dataset
```

### Masked-LM Bias Probe

Before fine-tuning, you can probe raw DistilBERT for bias using masked-LM prediction:

```python
from src.debiaser.masked_lm_demo import predict_masked_word

predictions = predict_masked_word("The engineer fixed it. <mask> was very skilled.", top_k=5)
for word, prob in predictions:
    print(f"{word}: {prob:.4f}")
```

---

## Dataset

The dataset is derived from **WinoBias** (Zhao et al., 2018), a benchmark for gender bias in coreference resolution.

| Split | Samples |
|-------|---------|
| Train | ~2,270  |
| Val   | ~253    |
| Test  | ~632    |

Each sample is a pair:
- **`pro`** — a pro-stereotyped sentence (e.g., female nurse, male engineer)
- **`anti`** — an anti-stereotyped sentence (same template, swapped gender)

---

## Model & Training

| Component | Details |
|-----------|---------|
| Base model | `distilbert-base-uncased` |
| Task | Binary sequence classification |
| Loss | `(1-α) × CrossEntropy + α × MSE(pro_logits, anti_logits)` |
| α (alpha) | Controls bias/accuracy trade-off (default `0.5`) |
| Optimizer | AdamW with linear warmup schedule |
| Device | Auto-detected: MPS (Apple Silicon) → CUDA → CPU |

The **consistency loss** (MSE between pro and anti logit vectors) is the core debiasing mechanism — it penalises the model whenever it treats the two sentence versions differently.

---

## Outputs

After training, outputs are written to:

```
outputs/
├── models/
│   ├── best_model/          # Best checkpoint (lowest val loss)
│   └── final_model/         # Final epoch checkpoint
├── reports/
│   ├── training_history.json
│   ├── final_metrics.json
│   ├── evaluation_results.json
│   └── detailed_bias_analysis.json
└── visualizations/
    ├── training_history.png
    ├── loss_components.png
    ├── evaluation_metrics.png
    └── test_bias_consistency.png
```

---

## Benchmarking

The StereoSet benchmark measures bias using three scores:
- **LM score** — how well the model assigns higher probability to meaningful sentences over nonsense
- **SS score** (Stereotype Score) — how often the model prefers stereotyped over anti-stereotyped sentences (50% = unbiased)
- **iCAT score** — combined metric: `LM × min(SS, 100-SS) / 50` (higher is better)

Run the StereoSet evaluator:
```python
from src.benchmarking.stereoset import Bias

bias = Bias("gender")
bias.run()           # Runs DistilBERT on StereoSet gender subset
bias.save_report()   # Saves JSON report
```

---

## References

- Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018). **Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods.** NAACL.
- [WinoBias Dataset](https://github.com/uclanlp/corefBias)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased) — Sanh et al., 2019
