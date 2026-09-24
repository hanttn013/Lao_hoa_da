import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion:
    - Stream A (Texture / Frequency) đóng vai trò Query/Key định hướng các rãnh nhăn mảnh.
    - Stream B (Spatial RGB) đóng vai trò Value cung cấp ngữ cảnh màu sắc toàn cục.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_feat, texture_feat):
        B, C, H, W = spatial_feat.shape
        proj_query = self.query_conv(texture_feat).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(texture_feat).view(B, -1, H * W)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value_conv(spatial_feat).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + spatial_feat

class BoundaryAwareLoss(nn.Module):
    """
    Boundary Loss:
    - Nhạy cảm với đường viền biên rãnh nếp nhăn (1-2 pixels).
    - Kết hợp với Dice Loss & Cross Entropy Loss để tối ưu mIoU và HD95 (Hausdorff Distance 95).
    """
    def __init__(self):
        super().__init__()
        # Kernel Laplacian cố định dùng để trích xuất gradient cạnh biên
        kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, pred_logits, target_mask):
        pred_probs = torch.softmax(pred_logits, dim=1)[:, 1:2, :, :]
        target_f = target_mask.float().unsqueeze(1)
        
        # Bắt biên của prediction và target qua toán tử Laplacian
        pred_boundary = F.conv2d(pred_probs, self.laplacian_kernel, padding=1)
        target_boundary = F.conv2d(target_f, self.laplacian_kernel, padding=1)
        
        pred_boundary = torch.abs(pred_boundary)
        target_boundary = torch.abs(target_boundary)
        
        # L1 Loss giữa hai bản đồ đường biên
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)
        return boundary_loss
