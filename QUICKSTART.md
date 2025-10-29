# Quick Start Guide

## Installation

```bash
pip install mlflow tensorflow keras numpy
```

## Running Scripts (Recommended Order)

### 1. Start with Basic Script (Recommended for beginners)

```bash
python reuters_mlflow.py
```

**What it does:** Trains a single experiment with default parameters. This is the main script from the lab assignment.

**Output:** Creates MLflow experiment "reuters_classification" with one run.

### 2. View Results in MLflow UI

```bash
mlflow ui
```

Then open: http://localhost:5000

### 3. Run Multiple Experiments (Optional)

```bash
python run_experiments.py
```

**What it does:** Runs 7 different experiments with various hyperparameters and compares them.

**Output:** All experiments logged to "reuters_classification" experiment.

### 4. Enhanced Version (Optional - Advanced)

```bash
python enhanced_reuters_mlflow.py
```

**What it does:** Runs two experiments (baseline and optimized) with comprehensive comparison and analysis.

**Output:** Creates "reuters-classification-comparison" experiment.

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `reuters_mlflow.py` | Basic MLflow integration (lab assignment) | **Start here** - Simple single experiment |
| `run_experiments.py` | Run multiple experiments with different hyperparameters | When you want to compare many configurations |
| `enhanced_reuters_mlflow.py` | Enhanced version with detailed tracking | When you need advanced features and comparisons |

## Understanding the Results

- **Test Accuracy**: Main metric to compare experiments
- **Test Loss**: Lower is better
- **Parameters**: Hyperparameters used in each experiment
- **Artifacts**: Saved model files, checkpoints, etc.

All metrics are automatically logged to MLflow and can be viewed in the UI.

## Troubleshooting

- **GPU not found**: Scripts will automatically fall back to CPU
- **MLflow UI not accessible**: Check if port 5000 is available
- **Import errors**: Make sure you're in the project directory when running scripts

