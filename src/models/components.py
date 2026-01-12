"""Reusable model components for multi-frame OCR."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Attention-based fusion module for combining multi-frame features.
    
    Computes attention weights across frames and performs weighted fusion.
    """
    
    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input feature channels.
        """
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B*num_frames, C, H, W].
        
        Returns:
            Fused tensor of shape [B, C, H, W].
        """
        b_frames, c, h, w = x.size()
        num_frames = 5
        b_size = b_frames // num_frames
        
        # Reshape for attention computation
        x_view = x.view(b_size, num_frames, c, h, w)
        scores = self.score_net(x).view(b_size, num_frames, 1, h, w)
        
        weights = F.softmax(scores, dim=1)
        return torch.sum(x_view * weights, dim=1)  # [B, C, H, W]


class CNNBackbone(nn.Module):
    """CNN feature extractor backbone for OCR."""
    
    def __init__(self, out_channels: int = 512):
        """
        Args:
            out_channels: Number of output channels (default 512).
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        return self.cnn(x)


class SequenceModeler(nn.Module):
    """Bidirectional LSTM for sequence modeling."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.25):
        """
        Args:
            input_size: Input feature dimension.
            hidden_size: LSTM hidden size.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence through LSTM."""
        output, _ = self.rnn(x)
        return output
