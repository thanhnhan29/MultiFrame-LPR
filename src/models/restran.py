"""ResTranOCR: ResNet + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ResNetFeatureExtractor, AttentionFusion, PositionalEncoding, STNBlock

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ResNet and Transformer.
    Pipeline: Input (5 frames) -> [Optional STN] -> ResNet -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        resnet_layers: int = 18,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet
        self.backbone = ResNetFeatureExtractor(layers=resnet_layers, pretrained=False)
        
        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 4. Transformer Encoder
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
        
        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, t, c, h, w = x.size()
        
        if self.use_stn:
            # 1. Compute temporal mean to find "stable" features
            x_mean = torch.mean(x, dim=1) # [B, 3, H, W]
            
            # 2. Predict affine matrix from the mean frame
            theta = self.stn(x_mean)      # [B, 2, 3]
            
            # 3. Prepare input for grid_sample (Flatten batch and time)
            x_flat = x.view(b * t, c, h, w)
            
            # 4. Repeat theta for all frames (Temporal consistency)
            # [B, 2, 3] -> [B, T, 2, 3] -> [B*T, 2, 3]
            theta_repeated = theta.unsqueeze(1).repeat(1, t, 1, 1).view(b * t, 2, 3)
            
            # 5. Apply Warp
            grid = F.affine_grid(theta_repeated, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            # Skip STN, directly flatten frames
            x_aligned = x.view(b * t, c, h, w)
        
        features = self.backbone(x_aligned) # [B*T, 512, 1, W']
        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.head(seq_out)              # [B, W', Num_Classes]
        
        return out.log_softmax(2)