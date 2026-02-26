#!/usr/bin/env python3
"""Pre-process LR images with Super Resolution to generate SR-enhanced images.

Scans all track folders, finds lr-*.png files, applies the Cgnet SR model,
and saves the results as sr-*.png in the same track folder.

Usage:
    python preprocess_sr.py --data-root data/train --sr-ckpt path/to/sr_model.pth
    python preprocess_sr.py --data-root data/public_test --sr-ckpt path/to/sr_model.pth

Supports both training and test data directories.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.sr_enhancer import (
    load_sr_model,
    preprocess_for_sr,
    enhance_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process LR images with Super Resolution")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing track folders")
    parser.add_argument("--sr-ckpt", type=str, required=True,
                        help="Path to Cgnet SR model checkpoint (.pth)")
    parser.add_argument("--sr-input-h", type=int, default=16,
                        help="SR model input height (default: 16)")
    parser.add_argument("--sr-input-w", type=int, default=48,
                        help="SR model input width (default: 48)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for SR inference (default: 32)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing sr-*.png files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load SR model
    sr_model = load_sr_model(args.sr_ckpt, device)
    
    # Find all track folders
    abs_root = os.path.abspath(args.data_root)
    search_path = os.path.join(abs_root, "**", "track_*")
    all_tracks = sorted(glob.glob(search_path, recursive=True))
    
    if not all_tracks:
        print(f"❌ No track folders found in: {abs_root}")
        sys.exit(1)
    
    print(f"📁 Found {len(all_tracks)} tracks in {abs_root}")
    
    total_processed = 0
    total_skipped = 0
    
    # Collect all LR files across all tracks
    all_lr_files = []
    for track_path in all_tracks:
        lr_files = sorted(
            glob.glob(os.path.join(track_path, "lr-*.png")) +
            glob.glob(os.path.join(track_path, "lr-*.jpg"))
        )
        for lr_path in lr_files:
            # Determine output path: lr-001.png → sr-001.png
            basename = os.path.basename(lr_path)
            sr_basename = basename.replace("lr-", "sr-", 1)
            # Keep same extension as input but save as PNG for quality
            sr_basename = os.path.splitext(sr_basename)[0] + ".png"
            sr_path = os.path.join(os.path.dirname(lr_path), sr_basename)
            
            if not args.overwrite and os.path.exists(sr_path):
                total_skipped += 1
                continue
            
            all_lr_files.append((lr_path, sr_path))
    
    if total_skipped > 0:
        print(f"⏭️  Skipped {total_skipped} existing SR files (use --overwrite to regenerate)")
    
    if not all_lr_files:
        print("✅ All SR files already exist. Nothing to process.")
        return
    
    print(f"🔄 Processing {len(all_lr_files)} LR images...")
    
    # Process in batches for GPU efficiency
    batch_size = args.batch_size
    
    for batch_start in tqdm(range(0, len(all_lr_files), batch_size), desc="SR Processing"):
        batch_items = all_lr_files[batch_start:batch_start + batch_size]
        
        # Load and preprocess batch
        tensors = []
        valid_items = []
        for lr_path, sr_path in batch_items:
            try:
                image = cv2.imread(lr_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = preprocess_for_sr(image, args.sr_input_h, args.sr_input_w)
                tensors.append(tensor)
                valid_items.append((lr_path, sr_path))
            except Exception as e:
                print(f"⚠️ Error loading {lr_path}: {e}")
                continue
        
        if not tensors:
            continue
        
        # Batch SR inference on GPU
        batch_tensor = torch.stack(tensors, dim=0).to(device)
        sr_batch = enhance_tensor(sr_model, batch_tensor)  # [B, 3, 2H, 2W]
        
        # Save results
        for i, (lr_path, sr_path) in enumerate(valid_items):
            sr_np = sr_batch[i].cpu().numpy().transpose(1, 2, 0)  # [2H, 2W, 3]
            sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)
            sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sr_path, sr_bgr)
            total_processed += 1
    
    print(f"\n✅ Done! Processed {total_processed} LR → SR images")
    print(f"   SR output size: {args.sr_input_h * 2} × {args.sr_input_w * 2}")


if __name__ == "__main__":
    main()
