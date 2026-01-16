"""ResTranOCR: ResNet + Transformer architecture (Advanced)."""
import torch
import torch.nn as nn

from src.models.components import AttentionFusion, PositionalEncoding, ResNetFeatureExtractor

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using ResNet and Transformer.
    Pipeline: Input (5 frames) -> ResNet -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        resnet_layers: int = 18,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cnn_channels = 512
        
        # 1. Backbone: ResNet (Stronger than VGG)
        self.backbone = ResNetFeatureExtractor(layers=resnet_layers, pretrained=False)
        
        # 2. Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 3. Sequence Modeling: Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 4. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, t, c, h, w = x.size()
        
        # Flatten batch and frames
        x_flat = x.view(b * t, c, h, w)
        
        # Extract features
        features = self.backbone(x_flat) # [B*T, 512, 1, W']
        
        # Fuse frames
        fused = self.fusion(features)    # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        # Classification
        out = self.head(seq_out)              # [B, W', Num_Classes]
        
        return out.log_softmax(2)