import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSnakeConv2d(nn.Module):
    """
    Dynamic Snake Convolution (DSConv) - Qi et al., ICCV 2023
    Optimized for tubular and curvilinear structure segmentation (e.g., facial wrinkles).
    Extracts elongated features along flexible snake trajectories along X and Y axes.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, extend_scope=1.0, morph=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph # 0: x-axis snake, 1: y-axis snake
        
        # Offset generator generates offsets for (kernel_size - 1) positions along snake path
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2 * kernel_size, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Aggregation convolution after sampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Predict iterative offsets
        raw_offsets = self.offset_conv(x) * self.extend_scope # [B, 2*K, H, W]
        
        # 2. Construct sampling grid with accumulated snake offsets
        # Base grid: normalized coords in [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]
        
        # Step size in normalized coordinates
        step_x = 2.0 / max(1, W - 1)
        step_y = 2.0 / max(1, H - 1)
        
        sampled_features = []
        half_k = self.kernel_size // 2
        
        # Cumulative trajectory offsets
        accum_offset_x = torch.zeros(B, H, W, device=device)
        accum_offset_y = torch.zeros(B, H, W, device=device)
        
        for i in range(self.kernel_size):
            rel_idx = i - half_k
            idx_offset = i * 2
            
            # Predict delta
            delta_x = raw_offsets[:, idx_offset, :, :] * step_x
            delta_y = raw_offsets[:, idx_offset + 1, :, :] * step_y
            
            if self.morph == 0:
                # X-axis snake: propagates along x with flexible y deviations
                curr_grid_x = base_grid[..., 0] + rel_idx * step_x + delta_x
                accum_offset_y = accum_offset_y + delta_y
                curr_grid_y = base_grid[..., 1] + accum_offset_y
            else:
                # Y-axis snake: propagates along y with flexible x deviations
                accum_offset_x = accum_offset_x + delta_x
                curr_grid_x = base_grid[..., 0] + accum_offset_x
                curr_grid_y = base_grid[..., 1] + rel_idx * step_y + delta_y
                
            curr_grid = torch.stack((curr_grid_x, curr_grid_y), dim=-1) # [B, H, W, 2]
            
            # Sample feature with bilinear interpolation
            sampled = F.grid_sample(x, curr_grid, mode='bilinear', padding_mode='border', align_corners=True)
            sampled_features.append(sampled)
            
        # Average / fuse along snake trajectory
        fused = torch.mean(torch.stack(sampled_features, dim=0), dim=0) # [B, C, H, W]
        out = self.conv(fused)
        out = self.norm(out)
        return self.act(out)

class DSnakeBlock(nn.Module):
    """
    DSnake Block combining X-axis and Y-axis Dynamic Snake Convolutions
    with residual connection and zero-initialized learnable gate for smooth convergence.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.dsconv_x = DynamicSnakeConv2d(in_channels, mid_channels, kernel_size=9, morph=0)
        self.dsconv_y = DynamicSnakeConv2d(in_channels, mid_channels, kernel_size=9, morph=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        feat_x = self.dsconv_x(x)
        feat_y = self.dsconv_y(x)
        fused = self.fuse(torch.cat([feat_x, feat_y], dim=1))
        return self.res(x) + self.gamma * fused
