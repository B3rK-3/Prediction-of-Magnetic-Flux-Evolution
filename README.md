# Prediction of Magnetic Flux Evolution During Solar Active Region Emergence using Long Short-Term Memory Networks

[![arXiv](https://img.shields.io/badge/arXiv-2604.03507-b31b1b.svg)](https://arxiv.org/abs/2604.03507)

*This repository contains the official implementation of the paper, which will be published in the **Solar Physics** journal.*

This repository contains the implementation of LSTM models for predicting Solar Active Region (SAR) emergence patterns. The project is written in Python using PyTorch.


## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Wulver Cluster Setup](#wulver-cluster-setup)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Data Structure](#data-structure)
8. [Weights & Biases Integration](#weights--biases-integration)

## Environment Setup

1. **Clone the repository** and enter the directory:
   ```bash
   git clone https://github.com/B3rK-3/Prediction-of-Magnetic-Flux-Evolution.git
   cd Prediction-of-Magnetic-Flux-Evolution
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python3 -m venv sar-env
   source sar-env/bin/activate
   ```

3. **Install dependencies**:
   Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install all exhaustive dependencies (if needed for exact replication):
   ```bash
   pip install -r req.txt
   ```

## Wulver Cluster Setup

1. **Load required modules**:
   ```bash
   module load wulver
   ```

2. **Activate environment**:
   ```bash
   source /mmfs1/project/PI-ucid/your-ucid/sar-env/bin/activate
   ```

3. **Environment Configuration**:
   Create a `.env` file in the project root with:
   ```
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_ENTITY="jonastirona-new-jersey-institute-of-technology"
   WANDB_PROJECT="sar-emergence"
   ```

## Project Structure

```text
SAR_EMERGENCE_RESEARCH/
├── data/                    # Raw SAR data organized by AR number (Ignored in Git)
├── docs/                    # Development notes and change logs
├── scripts/                 # Utility scripts (e.g., check_scales.py)
├── lstm/                    # LSTM model implementations
│   ├── functions.py         # Core utilities, model classes, training functions
│   ├── grid_search.py       # Hyperparameter search with Ray Tune
│   ├── test_models_denormalized.py  # Denormalized evaluation
│   ├── scripts/             # HPC training scripts
│   ├── results/             # Saved model checkpoints and plots (Ignored)
│   └── models/              # Intermediate model artifacts (Ignored)
├── requirements.txt         # Core dependencies
├── req.txt                  # Full pip freeze dependencies
├── AGENTS.md                # AI agent instructions and conventions
└── README.md                # Project documentation
```

## Training Pipeline

### Hyperparameter Search

1. **Run `grid_search.py`**:
   ```bash
   cd lstm
   python grid_search.py <num_samples>
   ```

### HPC Training

1. **Run `grid_search.sh`**:
   ```bash
   cd lstm/scripts
   sbatch grid_search.sh
   ```
   This will run the training scripts on the NJIT Wulver HPC.

## Evaluation Pipeline

### LSTM Evaluation

To run batch evaluations:
```bash
python test_models_denormalized.py
```

## Data Structure

The project uses SAR data organized by Active Region (AR) number:
```text
data/
├── AR11698/
├── AR11726/
├── AR13165/
...
```

Each AR directory contains:
- Power maps (`mean_pmdop{AR}_flat.npz`)
- Magnetic flux data (`mean_mag{AR}_flat.npz`)
- Intensity data (`mean_int{AR}_flat.npz`)

## Model Outputs

### LSTM Models
Saved in: `lstm/results/` and `lstm/models/`
Format: `LSTM12_r{rid_of_top}_i{num_in}_n{num_layers}_h{hidden_size}_e{epochs}_lr{learning_rate}_d{dropout}.pth`

## Weights & Biases Integration

The project uses Weights & Biases for experiment tracking. Configure your W&B credentials in the `.env` file as shown in the [Wulver Cluster Setup](#wulver-cluster-setup) section.

Training metrics, model artifacts, and experiment results are automatically logged to your W&B project dashboard.