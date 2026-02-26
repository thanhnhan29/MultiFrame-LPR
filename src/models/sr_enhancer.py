"""Utility module for loading and running the Cgnet Super Resolution model.

The Cgnet model (from LPSR) performs 2x upscaling on license plate images:
  LR [B, 3, H, W] → SR [B, 3, 2H, 2W]  (output in [0, 1] via sigmoid)

Usage:
    from src.models.sr_enhancer import load_sr_model, enhance_frames

    sr_model = load_sr_model("path/to/checkpoint.pth", device)
    sr_images = enhance_frames(sr_model, lr_images)  # [B, 3, 2H, 2W]
"""
import sys
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _register_lpsr_models():
    """Import lpsr models so they register with the models registry."""
    # Add lpsr to path so its internal imports work
    lpsr_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'lpsr')
    lpsr_dir = os.path.abspath(lpsr_dir)
    if lpsr_dir not in sys.path:
        sys.path.insert(0, lpsr_dir)


def load_sr_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the pre-trained Cgnet SR model from a checkpoint.
    
    The checkpoint structure (from LPSR training):
        {
            'model': {'name': 'cgnetV2_deformable', 'args': {...}, 'sd': state_dict},
            'state': rng_state
        }
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Target device.
        
    Returns:
        Frozen Cgnet model in eval mode.
    """
    _register_lpsr_models()
    import models as lpsr_models

    print(f"📦 Loading SR model from: {checkpoint_path}")
    sv_file = torch.load(checkpoint_path, map_location=device)
    
    # Create and load model via lpsr's registry
    model_spec = sv_file['model']
    sr_model = lpsr_models.make(model_spec, load_model=True)
    sr_model = sr_model.to(device)
    sr_model.eval()
    
    # Freeze all parameters
    for param in sr_model.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in sr_model.parameters())
    print(f"✅ SR model loaded: {total_params:,} params (frozen)")
    
    return sr_model


@torch.no_grad()
def enhance_tensor(
    sr_model: nn.Module,
    lr_tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply SR model to a batch of LR image tensors.
    
    Args:
        sr_model: Frozen Cgnet model.
        lr_tensor: [B, 3, H, W] in [0, 1] range.
        
    Returns:
        SR tensor [B, 3, 2H, 2W] in [0, 1] range.
    """
    sr = sr_model(lr_tensor)
    if isinstance(sr, tuple):
        sr = sr[0]
    return sr.clamp(0, 1)


def pad_to_aspect_ratio(
    image: np.ndarray,
    target_ratio: float = 3.0,
    tolerance: float = 0.15,
    background: Tuple[int, int, int] = (127, 127, 127)
) -> np.ndarray:
    """Pad image to maintain target aspect ratio (W/H).
    
    Matches the LPSR wrapper's padding logic.
    
    Args:
        image: HWC uint8 image.
        target_ratio: Target W/H ratio (default 3.0 for LP images).
        tolerance: Allowed deviation from target ratio.
        background: Padding fill color.
        
    Returns:
        Padded image.
    """
    h, w = image.shape[:2]
    ar = float(w) / h
    
    min_ratio = target_ratio - tolerance
    max_ratio = target_ratio + tolerance
    
    if min_ratio <= ar <= max_ratio:
        return image
    
    border_w = 0
    border_h = 0
    
    if ar < min_ratio:
        while float(w + border_w) / (h + border_h) < min_ratio:
            border_w += 1
    else:
        while float(w) / (h + border_h) > max_ratio:
            border_h += 1
    
    border_w //= 2
    border_h //= 2
    
    image = cv2.copyMakeBorder(
        image, border_h, border_h, border_w, border_w,
        cv2.BORDER_CONSTANT, value=background
    )
    return image


def preprocess_for_sr(
    image: np.ndarray,
    sr_input_h: int = 16,
    sr_input_w: int = 48,
) -> torch.Tensor:
    """Preprocess a raw LR image for the SR model.
    
    Steps: pad to AR 3:1 → resize to SR input → convert to [0,1] tensor.
    
    Args:
        image: RGB HWC uint8 image.
        sr_input_h: SR model input height (default 16).
        sr_input_w: SR model input width (default 48).
        
    Returns:
        Tensor [3, sr_input_h, sr_input_w] in [0, 1] range.
    """
    image = pad_to_aspect_ratio(image, target_ratio=float(sr_input_w) / sr_input_h)
    image = cv2.resize(image, (sr_input_w, sr_input_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor


def enhance_single_image(
    sr_model: nn.Module,
    image: np.ndarray,
    sr_input_h: int = 16,
    sr_input_w: int = 48,
    device: torch.device = None,
) -> np.ndarray:
    """Full pipeline: raw LR image → SR-enhanced image as numpy.
    
    Args:
        sr_model: Frozen Cgnet model.
        image: RGB HWC uint8 ndarray.
        sr_input_h: SR model input height.
        sr_input_w: SR model input width.
        device: Device (inferred from model if None).
        
    Returns:
        SR-enhanced RGB HWC uint8 ndarray at 2x resolution.
    """
    if device is None:
        device = next(sr_model.parameters()).device
    
    tensor = preprocess_for_sr(image, sr_input_h, sr_input_w)
    tensor = tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    sr = enhance_tensor(sr_model, tensor)  # [1, 3, 2H, 2W]
    
    sr_np = sr[0].cpu().numpy().transpose(1, 2, 0)  # [2H, 2W, 3]
    sr_np = (sr_np * 255).clip(0, 255).astype(np.uint8)
    return sr_np
