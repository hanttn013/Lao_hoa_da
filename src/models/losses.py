import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, 2, H, W], targets: [B, H, W]
        probs = F.softmax(logits, dim=1)[:, 1, :, :] # foreground probability
        targets_f = targets.float()
        
        intersection = (probs * targets_f).sum(dim=(1, 2))
        cardinality = (probs + targets_f).sum(dim=(1, 2))
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

class BoundaryAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 Laplacian kernel
        kernel = torch.tensor([[[[0.0, 1.0, 0.0],
                                 [1.0, -4.0, 1.0],
                                 [0.0, 1.0, 0.0]]]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1:2, :, :]
        targets_f = targets.float().unsqueeze(1)
        
        kernel = self.laplacian_kernel.to(logits.device)
        pred_boundary = torch.abs(F.conv2d(probs, kernel, padding=1))
        target_boundary = torch.abs(F.conv2d(targets_f, kernel, padding=1))
        
        return F.l1_loss(pred_boundary, target_boundary)

class CombinedLoss(nn.Module):
    def __init__(self, use_boundary=True, boundary_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        if use_boundary:
            self.boundary = BoundaryAwareLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        loss = ce_loss + dice_loss
        if self.use_boundary:
            b_loss = self.boundary(logits, targets)
            loss = loss + self.boundary_weight * b_loss
        return loss

