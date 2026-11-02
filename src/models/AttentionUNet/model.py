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

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion at Bottleneck:
    Directs focus toward fine, elongated wrinkle features across spatial coordinates.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_feat, texture_feat=None):
        if texture_feat is None:
            texture_feat = spatial_feat
        B, C, H, W = spatial_feat.shape
        proj_query = self.query_conv(texture_feat).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(texture_feat).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(spatial_feat).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + spatial_feat

class AttentionUNet(nn.Module):
    """
    Attention U-Net with Bottleneck Attention Fusion:
    - Input: 4 channels [RGB + Texture Map]
    - Output: 2 classes [Background, Wrinkle]
    - Parameters: 17,591,939
    """
    def __init__(self, in_channels=4, out_channels=2, bilinear=True):
        super().__init__()
        self.name = "Attention_UNet"
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)
        self.cross_attn = CrossAttentionFusion(in_channels=512)

    def forward(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)
        
        # Bottleneck Cross-Attention
        x5 = self.cross_attn(x5, x5)
        
        x = self.unet.up1(x5, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        logits = self.unet.outc(x)
        return logits
