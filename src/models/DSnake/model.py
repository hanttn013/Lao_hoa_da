import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for p in [SRC_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.unet.unet_model import UNet
except ImportError:
    from unet.unet_model import UNet

from models.DSnake.dsconv import DSnakeBlock
from models.AttentionUNet.model import CrossAttentionFusion

class DSnakeUNet(nn.Module):
    """
    DSnake Model (Our Proposed Architecture - DS-SAUNet):
    - Integrates Dynamic Snake Convolutions (DSConv - Qi et al., ICCV 2023)
      at multiple feature scales (input, mid-level, and output refinement)
      to follow continuous, winding, curvilinear wrinkle structures.
    - Cross-Attention Fusion at the bottleneck (512 channels).
    - Effectively eliminates wrinkle fragmentation ("đứt gãy rãnh nhăn").
    - Input: 4 channels [RGB + Texture Map]
    - Output: 2 classes [Background, Wrinkle]
    """
    def __init__(self, in_channels=4, out_channels=2, bilinear=True):
        super().__init__()
        self.name = "DSnake_OurModel"
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)
        
        # 1. Dynamic Snake Block for high-resolution edge tracking (64 ch)
        self.snake_in = DSnakeBlock(64, 64)
        
        # 2. Dynamic Snake Block for mid-level curvilinear features (256 ch)
        self.snake_mid = DSnakeBlock(256, 256)
        
        # 3. Cross-Attention Fusion at bottleneck (512 ch)
        self.cross_attn = CrossAttentionFusion(in_channels=512)
        
        # 4. Dynamic Snake Block for final contour continuity refinement (64 ch)
        self.snake_refine = DSnakeBlock(64, 64)

    def forward(self, x):
        x1 = self.unet.inc(x)
        x1 = self.snake_in(x1)
        
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x3 = self.snake_mid(x3)
        
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)
        
        # Bottleneck Cross-Attention
        x5 = self.cross_attn(x5, x5)
        
        x = self.unet.up1(x5, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        
        # Refinement with DSnake
        x = self.snake_refine(x)
        logits = self.unet.outc(x)
        return logits
