# MultiFrame-LPR

This repository contains the baseline solution for the **ICPR 2026 Challenge on Low-Resolution License Plate Recognition** (Task 1).

The project implements a **Multi-Frame CRNN** architecture that utilizes an **Attention Fusion** mechanism to effectively combine information from multiple low-resolution video frames (temporal sequences) to improve recognition accuracy.

🔗 **Contest:** [ICPR 2026 LRLPR Challenge](https://icpr26lrlpr.github.io/)

## 📌 Features

* **Multi-Frame Input Handling:** Processes sequences of 5 frames simultaneously to mitigate low-resolution artifacts.
* **Attention-Based Fusion:** A dedicated `AttentionFusion` module dynamically weights feature maps from different frames before sequence modeling.
* **Hybrid Architecture:**
* **Backbone:** Custom deep CNN (VGG-style) for feature extraction.
* **Head:** Bidirectional LSTM with CTC Loss for sequence decoding.


* **Robust Data Pipeline:**
* Handles both real LR images and synthetic LR (degraded HR) images.
* Advanced augmentations using `Albumentations` (Affine, Perspective, HSV, CoarseDropout).
* **Scenario-B Aware Splitting:** Automatically prioritizes challenging scenarios for validation to prevent overfitting.


* **Optimized Training:** Implements Mixed Precision (AMP), Gradient Clipping, and OneCycleLR scheduler.

## 🛠️ Model Architecture

The pipeline follows this flow:

1. **Input:** Tensor of shape `(Batch, 5, 3, 32, 128)`.
2. **CNN Backbone:** Extracts features from each frame independently `(B*5, 512, H, W)`.
3. **Attention Fusion:** Computes attention scores across the temporal dimension and fuses features into `(B, 512, H, W)`.
4. **Reshape:** Converts spatial features to sequential format `(B, Width, Channels)`.
5. **BiLSTM:** Captures contextual dependencies.
6. **CTC Decoder:** Outputs the final character sequence.

## 📂 Project Structure

```text
.
├── configs/             # Configuration parameters (Hyperparameters, Paths)
├── src/
│   ├── data/            # Dataset loading, Transforms, Degradation logic
│   ├── models/          # CRNN Architecture, Attention Fusion, Backbones
│   ├── training/        # Trainer loop, Validation, Checkpointing
│   └── utils/           # Seeding, CTC Decoding, Post-processing
├── train.py             # Main entry point for training
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
You can use `pip` or `uv` based on the `pyproject.toml`:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
    --data-root /path/to/your/dataset \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --num-workers 8

```

### Outputs

* **Best Model:** Saved as `best_model.pth`.
* **Submission File:** Generates `submission.txt` (Track ID, Predicted Text, Confidence).

## ⚙️ Configuration

Key hyperparameters can be modified in `configs/config.py`:

* `IMG_HEIGHT`, `IMG_WIDTH`: 32x128 (Default)
* `CHARS`: Standard alphanumeric set (0-9, A-Z)
* `HIDDEN_SIZE`: LSTM hidden dimension (256)
* `SPLIT_RATIO`: Train/Val split (0.9)

