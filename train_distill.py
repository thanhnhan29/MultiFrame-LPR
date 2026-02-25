#!/usr/bin/env python3
"""Entry point for Teacher-Student Knowledge Distillation training.

Supports 3 phases:
  --phase teacher  : Pre-train Teacher model on HR images only
  --phase distill  : Train Student (LR) with frozen Teacher (HR) guidance
  --phase both     : Run Phase 1 then Phase 2 sequentially

Usage:
  # Phase 1: Pre-train teacher
  python train_distill.py --phase teacher --model restran --epochs 50

  # Phase 2: Distill to student
  python train_distill.py --phase distill --model restran \\
      --teacher-ckpt results/teacher_restran_best.pth --epochs 50

  # Combined
  python train_distill.py --phase both --model restran --epochs 50
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.data.paired_dataset import PairedMultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.training.distill_trainer import DistillationTrainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Teacher-Student Knowledge Distillation for Multi-Frame LPR"
    )
    parser.add_argument(
        "--phase", type=str, choices=["teacher", "distill", "both"], default="both",
        help="Training phase: 'teacher' (pre-train HR), 'distill' (train LR student), 'both'"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran"], default=None,
        help="Model architecture (default: from config)"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--teacher-ckpt", type=str, default=None,
        help="Path to pre-trained teacher checkpoint (for distill phase)"
    )
    parser.add_argument(
        "--no-stn", action="store_true",
        help="Disable Spatial Transformer Network"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results/)"
    )
    # Distillation hyperparameters
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Weight for feature alignment loss (default: from config)"
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Weight for KD logit loss (default: from config)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Softmax temperature for KD (default: from config)"
    )
    parser.add_argument(
        "--aug-level", type=str, choices=["full", "light"], default=None,
        help="Augmentation level for LR training images"
    )
    parser.add_argument(
        "--teacher-epochs", type=int, default=None,
        help="Override epochs for teacher phase only (when using --phase both)"
    )
    parser.add_argument(
        "--student-epochs", type=int, default=None,
        help="Override epochs for student distill phase only (when using --phase both)"
    )
    parser.add_argument(
        "--submission-mode", action="store_true",
        help="After distillation, generate submission on test data"
    )
    return parser.parse_args()


def apply_config_overrides(config: Config, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to config."""
    if args.model is not None:
        config.MODEL_TYPE = args.model
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root
    if args.no_stn:
        config.USE_STN = False
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.alpha is not None:
        config.DISTILL_ALPHA = args.alpha
    if args.beta is not None:
        config.DISTILL_BETA = args.beta
    if args.temperature is not None:
        config.DISTILL_TEMPERATURE = args.temperature
    if args.teacher_ckpt is not None:
        config.TEACHER_CHECKPOINT = args.teacher_ckpt
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def create_model(config: Config) -> torch.nn.Module:
    """Create model based on config."""
    if config.MODEL_TYPE == "restran":
        return ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        )
    else:
        return MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )


def run_teacher_phase(config: Config, args: argparse.Namespace) -> str:
    """Phase 1: Pre-train Teacher on HR images.
    
    Returns:
        Path to the saved teacher checkpoint.
    """
    print("\n" + "=" * 60)
    print("📚 PHASE 1: PRE-TRAINING TEACHER (HR Images)")
    print("=" * 60)
    
    # Override experiment name for teacher
    config.EXPERIMENT_NAME = args.experiment_name or f"teacher_{config.MODEL_TYPE}"
    
    # Common dataset params
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
    }
    
    # Create training dataset — use standard dataset but HR images will be
    # loaded as 'synthetic' samples (which are HR files with degradation applied).
    # For teacher training, we want CLEAN HR, so we use val_transforms on HR.
    # The simplest approach: use standard dataset which already indexes both LR and HR.
    # Teacher trained on ALL data (both real LR + synthetic LR from HR).
    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        **common_ds_params
    )
    
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        **common_ds_params
    )
    
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
    
    # Create teacher model
    teacher = create_model(config).to(config.DEVICE)
    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"📊 Teacher ({config.MODEL_TYPE}): {total_params:,} params")
    
    # Train teacher using standard Trainer
    trainer = Trainer(
        model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )
    trainer.fit()
    
    # Return path to best checkpoint
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
    print(f"\n✅ Teacher checkpoint: {ckpt_path}")
    return ckpt_path


def run_distill_phase(config: Config, args: argparse.Namespace, teacher_ckpt: str) -> None:
    """Phase 2: Train Student with Knowledge Distillation from Teacher.
    
    Args:
        teacher_ckpt: Path to pre-trained teacher checkpoint.
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 2: KNOWLEDGE DISTILLATION (Teacher → Student)")
    print("=" * 60)
    
    # Override experiment name for student
    config.EXPERIMENT_NAME = args.experiment_name or f"distill_{config.MODEL_TYPE}"
    
    # Load Teacher model (frozen)
    teacher = create_model(config).to(config.DEVICE)
    if os.path.exists(teacher_ckpt):
        print(f"📦 Loading teacher from: {teacher_ckpt}")
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=config.DEVICE))
    else:
        print(f"⚠️ WARNING: Teacher checkpoint not found at {teacher_ckpt}")
        print("   Teacher will use random weights (not recommended!)")
    teacher.eval()
    
    # Create Student model (new, trainable)
    student = create_model(config).to(config.DEVICE)
    
    total_params_t = sum(p.numel() for p in teacher.parameters())
    total_params_s = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"📊 Teacher: {total_params_t:,} params (frozen)")
    print(f"📊 Student: {total_params_s:,} params (trainable)")
    
    # Common dataset params
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
    }
    
    # Paired dataset for training (LR + HR)
    paired_train_ds = PairedMultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        augmentation_level=config.AUGMENTATION_LEVEL,
        **common_ds_params
    )
    
    paired_train_loader = DataLoader(
        paired_train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=PairedMultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Standard validation dataset (LR only, for fair comparison with baseline)
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        augmentation_level=config.AUGMENTATION_LEVEL,
        **common_ds_params
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
    
    if len(paired_train_ds) == 0:
        print("❌ ERROR: No paired samples found (need tracks with both LR and HR)!")
        sys.exit(1)
    
    # Distillation training
    distill_trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        train_loader=paired_train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )
    distill_trainer.fit()
    
    # Submission mode: generate predictions on test data
    if args.submission_mode and os.path.exists(config.TEST_DATA_ROOT):
        print("\n" + "=" * 60)
        print("📝 GENERATING SUBMISSION (Student Model)")
        print("=" * 60)
        
        test_ds = MultiFrameDataset(
            root_dir=config.TEST_DATA_ROOT,
            mode='val',
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            seed=config.SEED,
            is_test=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Load best student checkpoint
        exp_name = config.EXPERIMENT_NAME
        best_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_student_best.pth")
        if os.path.exists(best_path):
            print(f"📦 Loading best student: {best_path}")
            student.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
        
        # Use standard Trainer for inference convenience
        inference_trainer = Trainer(
            model=student,
            train_loader=paired_train_loader,  # dummy, not used for predict
            val_loader=None,
            config=config,
            idx2char=config.IDX2CHAR,
        )
        inference_trainer.predict_test(test_loader, f"submission_{exp_name}_student_final.txt")


def main():
    """Main entry point."""
    args = parse_args()
    config = Config()
    apply_config_overrides(config, args)
    seed_everything(config.SEED)
    
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)
    
    print(f"🚀 Configuration:")
    print(f"   PHASE: {args.phase}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   DISTILL_ALPHA: {config.DISTILL_ALPHA}")
    print(f"   DISTILL_BETA: {config.DISTILL_BETA}")
    print(f"   DISTILL_TEMPERATURE: {config.DISTILL_TEMPERATURE}")
    
    if args.phase == "teacher":
        # Phase 1 only
        epochs_override = args.teacher_epochs or config.EPOCHS
        config.EPOCHS = epochs_override
        run_teacher_phase(config, args)
        
    elif args.phase == "distill":
        # Phase 2 only (requires teacher checkpoint)
        teacher_ckpt = config.TEACHER_CHECKPOINT
        if not teacher_ckpt:
            print("❌ ERROR: --teacher-ckpt is required for distill phase")
            sys.exit(1)
        epochs_override = args.student_epochs or config.EPOCHS
        config.EPOCHS = epochs_override
        run_distill_phase(config, args, teacher_ckpt)
        
    elif args.phase == "both":
        # Phase 1 + Phase 2
        original_epochs = config.EPOCHS
        
        # Phase 1: Teacher
        config.EPOCHS = args.teacher_epochs or original_epochs
        teacher_ckpt = run_teacher_phase(config, args)
        
        # Phase 2: Distill
        config.EPOCHS = args.student_epochs or original_epochs
        args.experiment_name = None  # Reset so it auto-generates for student
        run_distill_phase(config, args, teacher_ckpt)


if __name__ == "__main__":
    main()
