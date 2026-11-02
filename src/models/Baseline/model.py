import os
import sys
import torch
import torch.nn as nn

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

class BaselineUNet(nn.Module):
    """
    Baseline U-Net (ICPR 2024 Reference):
    - Input: 4 channels [RGB + Texture Map]
    - Output: 2 classes [Background, Wrinkle]
    - Parameters: 17,263,618
    """
    def __init__(self, in_channels=4, out_channels=2, bilinear=True):
        super().__init__()
        self.name = "Baseline_UNet"
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=bilinear)

    def forward(self, x):
        return self.unet(x)
