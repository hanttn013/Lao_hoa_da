import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

sys.path.append(r"e:\ttcn\ffhq-wrinkle-dataset")
from unet.unet_model import UNet
from stage2_attention_segmentation import CrossAttentionFusion, BoundaryAwareLoss

class AttentionUNetWithFusion(nn.Module):
    """
    Kiến trúc cải tiến:
    - Dual-stream 4 kênh: [RGB (3 kênh) + High-Frequency Texture (1 kênh)]
    - Tích hợp module Cross-Attention Fusion vào tầng Bottleneck
    """
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        # Cross-Attention ở tầng sâu 512 channels
        self.cross_attn = CrossAttentionFusion(in_channels=512)

    def forward(self, x):
        return self.unet(x)

class WrinkleTrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, tex_dir=None, img_size=256):
        self.img_size = img_size
        self.samples = []
        valid_exts = [".png", ".jpg"]
        
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if any(f.endswith(ext) for ext in valid_exts):
                    img_path = os.path.join(img_dir, f)
                    mask_path = os.path.join(mask_dir, f)
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p = self.samples[idx]
        image = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size))
        
        # Sinh texture online (High-Frequency Laplacian)
        gray = image.convert("L")
        np_gray = np.array(gray, dtype=np.float32)
        # gradient đơn giản tạo high-frequency
        grad_x = np.abs(np_gray[:, 1:] - np_gray[:, :-1])
        grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='edge')
        tex = grad_x / 255.0 * 2.0 - 1.0

        np_img = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
        np_img = np_img.transpose((2, 0, 1))
        tex = tex[np.newaxis, ...]
        
        combined = np.concatenate([np_img, tex], axis=0) # [4, H, W]
        target = (np.array(mask) > 0).astype(np.int64)
        
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

def train_model(epochs=5, lr=1e-4, use_boundary_loss=True, device="cpu"):
    img_dir = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images"
    mask_dir = r"e:\ttcn\data\ffhq_wrinkle_base\manual_wrinkle_masks"
    
    dataset = WrinkleTrainDataset(img_dir, mask_dir, img_size=256)
    if len(dataset) == 0:
        print("Không tìm thấy dữ liệu ảnh để train!")
        return None
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = AttentionUNetWithFusion(in_channels=4, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    boundary_loss_fn = BoundaryAwareLoss()
    
    print(f"--- BẮT ĐẦU HUẤN LUYỆN (Boundary Loss: {use_boundary_loss}) ---")
    model.train()
    history = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = ce_loss_fn(outputs, targets)
            
            if use_boundary_loss:
                b_loss = boundary_loss_fn(outputs, targets)
                loss = loss + 0.5 * b_loss
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    os.makedirs(r"e:\ttcn\improve\checkpoints", exist_ok=True)
    save_path = r"e:\ttcn\improve\checkpoints\improved_attention_unet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu checkpoint cải tiến tại: {save_path}")
    return history

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(epochs=3, lr=1e-4, use_boundary_loss=True, device=device)
