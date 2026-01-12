"""Multi-frame CRNN architecture for license plate recognition."""
import torch
import torch.nn as nn

from src.models.components import AttentionFusion, CNNBackbone, SequenceModeler


class MultiFrameCRNN(nn.Module):
    """CNN-RNN architecture with multi-frame attention fusion.
    
    Architecture: CNN Backbone -> Attention Fusion -> BiLSTM -> FC -> CTC
    """
    
    def __init__(self, num_classes: int, hidden_size: int = 256, rnn_dropout: float = 0.25):
        """
        Args:
            num_classes: Number of output classes (including blank for CTC).
            hidden_size: LSTM hidden dimension.
            rnn_dropout: Dropout rate for LSTM layers.
        """
        super().__init__()
        self.cnn_channels = 512
        
        self.cnn = CNNBackbone(out_channels=self.cnn_channels)
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        self.rnn = SequenceModeler(
            input_size=self.cnn_channels,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=rnn_dropout
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, C, H, W] where T=5 frames.
        
        Returns:
            Log-softmax output of shape [B, W, num_classes].
        """
        b, t, c, h, w = x.size()
        
        # Process all frames through CNN
        x = x.view(b * t, c, h, w)
        feat = self.cnn(x)  # [B*5, 512, H_out, W_out]
        
        # Fuse multi-frame features
        fused = self.fusion(feat)  # [B, 512, H_out, W_out]
        
        # Reshape for RNN: (B, C, H, W) -> (B, W, C*H)
        b_out, c_out, h_f, w_f = fused.size()
        rnn_input = fused.permute(0, 3, 1, 2).reshape(b_out, w_f, c_out * h_f)
        
        # Sequence modeling and classification
        rnn_out = self.rnn(rnn_input)
        out = self.fc(rnn_out)
        
        return out.log_softmax(2)
