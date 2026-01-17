# MultiFrame-LPR

This repository contains the baseline and advanced solutions for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition** (Task 1).

The project implements **Multi-Frame OCR** architectures that utilize **Attention Fusion** mechanisms to effectively combine information from multiple low-resolution video frames (temporal sequences) to improve recognition accuracy.

🔗 **Contest:** [ICPR 2026 LRLPR Challenge](https://icpr26lrlpr.github.io/)

## 📌 Features

* **Multi-Frame Input Handling:** Processes sequences of 5 frames simultaneously to mitigate low-resolution artifacts.
* **Spatial Transformer Network (STN):** STN module predicts per-frame affine transformations to align and rectify input images before feature extraction.
* **Attention-Based Fusion:** A dedicated `AttentionFusion` module dynamically weights feature maps from different frames before sequence modeling.
* **Dual Architecture Support:**
* **CRNN (Baseline):** STN + CNN backbone + Bidirectional LSTM with CTC Loss.
* **ResTranOCR (Advanced):** STN + ResNet34 backbone + Transformer Encoder with CTC Loss.
* **Robust Data Pipeline:**
  * Handles both real LR images and synthetic LR (degraded HR) images.
  * Advanced augmentations using `Albumentations` (Affine, Perspective, HSV, CoarseDropout).
  * Configurable augmentation levels: `full` (heavy augmentation) or `light` (resize + normalize only).
  * **Scenario-B Aware Splitting:** Automatically prioritizes challenging scenarios for validation to prevent overfitting.
* **Optimized Training:** Implements Mixed Precision (AMP), Gradient Clipping, and OneCycleLR scheduler.
* **Ablation Study Support:** Automated script for running multiple experiments with different configurations.

## 🛠️ Model Architectures

### CRNN (Baseline)
The pipeline follows this flow:

1. **Input:** Tensor of shape `(Batch, 5, 3, 32, 128)`.
2. **STN Alignment:** Predicts per-frame affine transformations and applies spatial alignment to each frame independently `(B*5, 3, 32, 128)`.
3. **CNN Backbone:** Extracts features from each aligned frame independently `(B*5, 512, 1, W')`.
4. **Attention Fusion:** Computes attention scores across the temporal dimension and fuses features into `(B, 512, 1, W')`.
5. **Reshape:** Converts spatial features to sequential format `(B, W', 512)`.
6. **BiLSTM:** Captures contextual dependencies `(B, W', Hidden*2)`.
7. **CTC Decoder:** Outputs the final character sequence.

### ResTranOCR (Advanced)
The pipeline follows this flow:

1. **Input:** Tensor of shape `(Batch, 5, 3, 32, 128)`.
2. **STN Alignment:** Predicts per-frame affine transformations and applies spatial alignment to each frame independently `(B*5, 3, 32, 128)`.
3. **ResNet34 Backbone:** Extracts features from each aligned frame independently `(B*5, 512, 1, W')`.
4. **Attention Fusion:** Computes attention scores across the temporal dimension and fuses features into `(B, 512, 1, W')`.
5. **Reshape:** Converts spatial features to sequential format `(B, W', 512)`.
6. **Transformer Encoder:** Captures long-range dependencies with positional encoding `(B, W', 512)`.
7. **CTC Decoder:** Outputs the final character sequence.

## 📂 Project Structure

```text
.
├── configs/             # Configuration parameters (Hyperparameters, Paths)
│   └── config.py       # Main configuration dataclass
├── src/
│   ├── data/           # Dataset loading, Transforms, Degradation logic
│   │   ├── dataset.py   # MultiFrameDataset with Scenario-B aware splitting
│   │   └── transforms.py # Augmentation pipelines (full/light/val/degradation)
│   ├── models/          # Model architectures
│   │   ├── crnn.py      # CRNN baseline architecture
│   │   ├── restran.py   # ResTranOCR advanced architecture
│   │   └── components.py # Shared components (STNBlock, AttentionFusion, ResNet, PositionalEncoding)
│   ├── training/         # Training logic
│   │   └── trainer.py   # Trainer loop, Validation, Checkpointing
│   └── utils/           # Utilities
│       ├── common.py    # Seeding utilities
│       └── postprocess.py # CTC Decoding (Greedy)
├── train.py             # Main entry point for training
├── run_ablation.py      # Automated ablation study script
├── pyproject.toml       # Dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* CUDA-enabled GPU recommended.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/duongtruongbinh/MultiFrame-LPR.git
cd MultiFrame-LPR
```

2. **Install dependencies:**

Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install albumentations opencv-python matplotlib numpy pandas tqdm
```

## 📊 Usage

### Data Preparation

Ensure your dataset is organized with track folders containing images (`lr-*.png` or `hr-*.png`) and an `annotations.json` file.

Default config expects: `data/train`

### Training

Run the training script with default settings defined in `configs/config.py`:

```bash
python train.py
```

**Override hyperparameters via CLI:**

```bash
python train.py \
    --model restran \
    --experiment-name my_experiment \
    --data-root /path/to/your/dataset \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --aug-level full \
    --output-dir results
```

**Available CLI arguments:**
* `-m, --model`: Model architecture (`crnn` or `restran`)
* `-n, --experiment-name`: Experiment name for checkpoint/submission files
* `--data-root`: Root directory for training data
* `--batch-size`: Batch size for training
* `--epochs`: Number of training epochs
* `--lr, --learning-rate`: Learning rate
* `--seed`: Random seed for reproducibility
* `--num-workers`: Number of data loader workers
* `--hidden-size`: LSTM hidden size for CRNN
* `--transformer-heads`: Number of transformer attention heads
* `--transformer-layers`: Number of transformer encoder layers
* `--aug-level`: Augmentation level (`full` or `light`)
* `--output-dir`: Directory to save checkpoints and submission files (default: `results/`)

### Ablation Study

Run automated ablation experiments:

```bash
python run_ablation.py
```

This script runs multiple experiments with different configurations:
* `crnn_no_stn`: CRNN without STN, full augmentation
* `crnn_with_stn`: CRNN with STN, full augmentation
* `restran34_no_stn`: ResTranOCR (ResNet34) without STN, full augmentation
* `restran34_with_stn`: ResTranOCR (ResNet34) with STN, full augmentation

Results are saved in `experiments/` directory with logs and summary table.

### Outputs

* **Best Model:** Saved as `{experiment_name}_best.pth` in the output directory.
* **Submission File:** Generates `submission_{experiment_name}.txt` (Track ID, Predicted Text, Confidence).
* **Ablation Summary:** `experiments/ablation_summary.txt` with best accuracies for each experiment.

## ⚙️ Configuration

Key hyperparameters can be modified in `configs/config.py`: