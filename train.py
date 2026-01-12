#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame CRNN for License Plate Recognition"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size (default: from config)"
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Initialize config with CLI overrides
    config = Config()
    
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root
    if args.seed is not None:
        config.SEED = args.seed
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.hidden_size is not None:
        config.HIDDEN_SIZE = args.hidden_size
    
    # Set random seeds
    seed_everything(config.SEED)
    
    print(f"🚀 Configuration:")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    
    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # Create datasets
    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED
    )
    
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED
    )
    
    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    else:
        print("⚠️ WARNING: Validation dataset is empty.")

    # Initialize model
    model = MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT
    ).to(config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    trainer.fit()
    
    print(f"\n✅ Training complete! Best accuracy: {trainer.best_acc:.2f}%")


if __name__ == "__main__":
    main()
