"""Multi-frame CRNN architecture (Baseline)."""
import torch
import torch.nn as nn

from src.models.components import AttentionFusion

class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline."""
    def __init__(self, out_channels=512):
        super().__init__()
        # Defined as a list of layers for clarity: Conv -> ReLU -> Pool
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5 (Map to sequence height 1)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)


class MultiFrameCRNN(nn.Module):
    """
    Standard CRNN architecture adapted for Multi-frame input.
    Pipeline: Input (5 frames) -> CNN Backbone -> Attention Fusion -> BiLSTM -> CTC Head
    """
    def __init__(self, num_classes: int, hidden_size: int = 256, rnn_dropout: float = 0.25):
        super().__init__()
        self.cnn_channels = 512
        
        # 1. Feature Extractor (CNN Backbone)
        self.backbone = CNNBackbone(out_channels=self.cnn_channels)
        
        # 2. Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 3. Sequence Modeling (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels, # Height is collapsed to 1, so input is just channels
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout
        )
        
        # 4. Prediction Head
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, t, c, h, w = x.size()
        
        # Flatten batch and frames to process in parallel
        x_flat = x.view(b * t, c, h, w)
        
        # Extract features
        features = self.backbone(x_flat) # [B*T, 512, 1, W']
        
        # Fuse frames
        fused = self.fusion(features)    # [B, 512, 1, W']
        
        # Prepare for RNN: [B, C, 1, W'] -> [B, W', C]
        # Squeeze height (1) and permute
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # RNN Modeling
        rnn_out, _ = self.rnn(seq_input) # [B, W', Hidden*2]
        
        # Classification
        out = self.head(rnn_out)         # [B, W', Num_Classes]
        
        return out.log_softmax(2)