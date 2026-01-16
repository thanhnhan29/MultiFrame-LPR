"""Post-processing utilities for OCR decoding."""
from collections import defaultdict
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding.
    
    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        # Group consecutive identical characters and filter blanks
        # groupby returns (key, group_iterator) pairs
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                # Get maximum probability from this group
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))
    
    return results


def decode_beam_search(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    beam_width: int = 10,
    blank: int = 0
) -> List[Tuple[str, float]]:
    """CTC beam search decoding.
    
    Args:
        preds: Log-softmax predictions [Batch, Time, Classes].
        idx2char: Index to character mapping.
        beam_width: Number of beams to maintain.
        blank: Index of the blank label.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    log_probs = preds.detach().cpu().numpy()
    batch_size, time_steps, num_classes = log_probs.shape
    
    results: List[Tuple[str, float]] = []
    
    for b in range(batch_size):
        # Each beam: (prefix_tuple, last_char, log_prob_blank, log_prob_non_blank)
        # prefix_tuple: tuple of char indices (excluding blanks and collapsed repeats)
        beams = {(): (0.0, float('-inf'))}  # {prefix: (prob_blank, prob_non_blank)}
        
        for t in range(time_steps):
            new_beams: Dict[tuple, Tuple[float, float]] = defaultdict(
                lambda: (float('-inf'), float('-inf'))
            )
            
            for prefix, (pb, pnb) in beams.items():
                # Total probability of this prefix
                p_total = np.logaddexp(pb, pnb)
                
                for c in range(num_classes):
                    log_p = log_probs[b, t, c]
                    
                    if c == blank:
                        # Blank extends prefix without adding character
                        key = prefix
                        old_pb, old_pnb = new_beams[key]
                        new_pb = np.logaddexp(old_pb, p_total + log_p)
                        new_beams[key] = (new_pb, old_pnb)
                    else:
                        # Non-blank character
                        if len(prefix) > 0 and prefix[-1] == c:
                            # Same as last character - need blank in between for new char
                            # Extend without adding new char (collapse)
                            key = prefix
                            old_pb, old_pnb = new_beams[key]
                            new_pnb = np.logaddexp(old_pnb, pnb + log_p)
                            new_beams[key] = (old_pb, new_pnb)
                            
                            # Add new char only if preceded by blank
                            new_key = prefix + (c,)
                            old_pb2, old_pnb2 = new_beams[new_key]
                            new_pnb2 = np.logaddexp(old_pnb2, pb + log_p)
                            new_beams[new_key] = (old_pb2, new_pnb2)
                        else:
                            # Different from last character - always add
                            new_key = prefix + (c,)
                            old_pb, old_pnb = new_beams[new_key]
                            new_pnb = np.logaddexp(old_pnb, p_total + log_p)
                            new_beams[new_key] = (old_pb, new_pnb)
            
            # Prune to top beam_width
            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:beam_width]
            beams = dict(sorted_beams)
        
        # Get best beam
        best_prefix = max(beams.keys(), key=lambda x: np.logaddexp(beams[x][0], beams[x][1]))
        best_prob = np.logaddexp(beams[best_prefix][0], beams[best_prefix][1])
        
        # Convert to string
        pred_str = ''.join(idx2char.get(c, '') for c in best_prefix)
        confidence = float(np.exp(best_prob / max(len(best_prefix), 1)))
        
        results.append((pred_str, confidence))
    
    return results
