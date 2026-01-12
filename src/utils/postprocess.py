"""Post-processing utilities for OCR decoding."""
from typing import Dict, List, Tuple

import torch


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores.
    
    Args:
        preds: Log-softmax predictions of shape [Batch, Time, Classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    result_list: List[Tuple[str, float]] = []
    
    for b in range(preds.size(0)):
        pred_str = ""
        confidences: List[float] = []
        last_char = 0
        
        for t in range(preds.size(1)):
            c = indices[b, t].item()
            p = max_probs[b, t].item()
            if c != 0 and c != last_char:
                pred_str += idx2char.get(c, '')
                confidences.append(p)
            last_char = c
        
        score = sum(confidences) / len(confidences) if confidences else 0.0
        result_list.append((pred_str, score))
    
    return result_list
