"""Models module containing network architectures."""
from src.models.crnn import MultiFrameCRNN
from src.models.components import AttentionFusion

__all__ = ["MultiFrameCRNN", "AttentionFusion"]
