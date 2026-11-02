import os
import sys
import argparse

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "ffhq-wrinkle-dataset"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unet.unet_model import UNet
from stage2_attention_segmentation import CrossAttentionFusion, BoundaryAwareLoss

class BaselineUNet(nn.Module):
    """
    Mô hình gốc (ICPR 2024 Baseline):
    - Nhận 4 kênh [RGB + Texture Map]
    - Kiến trúc U-Net tiêu chuẩn
    """
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)

    def forward(self, x):
        return self.unet(x)

class AttentionUNetWithFusion(nn.Module):
    """
    Kiến trúc cải tiến:
    - Dual-stream 4 kênh: [RGB (3 kênh) + High-Frequency Texture (1 kênh)]
    - Tích hợp module Cross-Attention Fusion vào tầng Bottleneck (512 channels)
    """
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.unet = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        self.cross_attn = CrossAttentionFusion(in_channels=512)

    def forward(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        x5 = self.unet.down4(x4)
        
        # Nhúng Cross-Attention vào tầng bottleneck để tập trung vào các rãnh nhăn mảnh
        x5 = self.cross_attn(x5, x5)
        
        x = self.unet.up1(x5, x4)
        x = self.unet.up2(x, x3)
        x = self.unet.up3(x, x2)
        x = self.unet.up4(x, x1)
        logits = self.unet.outc(x)
        return logits

class WrinkleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_size = img_size
        self.samples = []
        valid_exts = [".png", ".jpg"]
        
        if os.path.exists(img_dir):
            for f in sorted(os.listdir(img_dir)):
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
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Sinh texture online (High-Frequency Laplacian)
        gray = image.convert("L")
        np_gray = np.array(gray, dtype=np.float32)
        grad_x = np.abs(np_gray[:, 1:] - np_gray[:, :-1])
        grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='edge')
        tex = grad_x / 255.0 * 2.0 - 1.0

        np_img = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
        np_img = np_img.transpose((2, 0, 1))
        tex = tex[np.newaxis, ...]
        
        combined = np.concatenate([np_img, tex], axis=0) # [4, H, W]
        target = (np.array(mask) > 0).astype(np.int64)
        
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

def compute_metrics(model, val_loader, device, ce_loss_fn, boundary_loss_fn=None):
    model.eval()
    val_loss = 0.0
    total_dice = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = ce_loss_fn(outputs, targets)
            if boundary_loss_fn:
                b_loss = boundary_loss_fn(outputs, targets)
                loss = loss + 0.5 * b_loss
            val_loss += loss.item()
            
            # Dice score
            pred = torch.argmax(outputs, dim=1)
            intersection = ((pred == 1) & (targets == 1)).sum().float()
            union = (pred == 1).sum().float() + (targets == 1).sum().float()
            dice = (2.0 * intersection) / (union + 1e-6)
            total_dice += dice.item()
            
    avg_loss = val_loss / max(1, num_batches)
    avg_dice = total_dice / max(1, num_batches)
    return avg_loss, avg_dice

def train_model(model_type="improved", epochs=100, lr=1e-4, batch_size=8, img_size=256, patience=10, device="cuda"):
    img_dir = os.path.join(PROJECT_ROOT, "data", "ffhq_wrinkle_base", "masked_face_images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "ffhq_wrinkle_base", "manual_wrinkle_masks")
    
    full_dataset = WrinkleDataset(img_dir, mask_dir, img_size=img_size)
    total_samples = len(full_dataset)
    if total_samples == 0:
        print("Lỗi: Không tìm thấy dữ liệu ảnh để train!", flush=True)
        return None
        
    # Phân chia 85% Train (674 ảnh), 15% Validation (119 ảnh)
    val_size = max(1, int(total_samples * 0.15))
    train_size = total_samples - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Tổng mẫu: {total_samples} | Train: {train_size} | Validation: {val_size}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device=="cuda"))
    
    if model_type == "baseline":
        model = BaselineUNet(in_channels=4, out_channels=2).to(device)
        boundary_loss_fn = None
        save_name = "baseline_unet.pth"
        title = "BASELINE U-NET (Mô hình gốc ICPR 2024)"
    else:
        model = AttentionUNetWithFusion(in_channels=4, out_channels=2).to(device)
        boundary_loss_fn = BoundaryAwareLoss().to(device)
        save_name = "improved_attention_unet.pth"
        title = "IMPROVED ATTENTION U-NET (Dual-Stream + Bottleneck Attention + Boundary Loss)"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, save_name)
    
    print("=" * 90, flush=True)
    print(f" BẮT ĐẦU HUẤN LUYỆN: {title}", flush=True)
    print(f" Cấu hình: Epochs={epochs} | Batch Size={batch_size} | Kích thước={img_size}x{img_size} | Device={device}", flush=True)
    print(f" Early Stopping: Dừng nếu Val Dice không tăng sau {patience} epochs liên tiếp", flush=True)
    print("=" * 90, flush=True)
    
    best_dice = -1.0
    epochs_no_improve = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = ce_loss_fn(outputs, targets)
            
            if boundary_loss_fn:
                b_loss = boundary_loss_fn(outputs, targets)
                loss = loss + 0.5 * b_loss
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / total_batches
        
        # Đánh giá trên tập Validation
        val_loss, val_dice = compute_metrics(model, val_loader, device, ce_loss_fn, boundary_loss_fn)
        scheduler.step(val_dice)
        
        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice
        })
        
        # Kiểm tra cải thiện của Dice Score
        is_best = val_dice > best_dice + 1e-4
        if is_best:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            status_mark = f"🔥 TĂNG TRƯỞNG! Đã lưu Best Model ({best_dice:.4f})"
        else:
            epochs_no_improve += 1
            status_mark = f"⏳ Không tăng ({epochs_no_improve}/{patience})"
            
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Best Dice: {best_dice:.4f} | {status_mark}", flush=True)
        
        # Kích hoạt Early Stopping
        if epochs_no_improve >= patience:
            print("\n" + "=" * 90, flush=True)
            print(f"🛑 [EARLY STOPPING KÍCH HOẠT]: Chỉ số Validation Dice không tăng sau {patience} epochs liên tiếp!", flush=True)
            print(f"   Dừng huấn luyện tại Epoch {epoch}. Best Validation Dice đạt được: {best_dice:.4f}", flush=True)
            print(f"   Trọng số tối ưu nhất đã được lưu tại: {save_path}", flush=True)
            print("=" * 90 + "\n", flush=True)
            break
            
    print("=" * 90, flush=True)
    print(f" [HOÀN TẤT] Model: {save_name} | Best Val Dice: {best_dice:.4f}", flush=True)
    print("=" * 90, flush=True)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân đoạn nếp nhăn với Early Stopping")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "improved"],
                        help="Chọn loại mô hình: 'baseline' hoặc 'improved'")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs tối đa (Mặc định: 100)")
    parser.add_argument("--batch_size", type=int, default=8, help="Kích thước batch (Mặc định: 8)")
    parser.add_argument("--patience", type=int, default=10, help="Số epochs dừng khi không tăng (Mặc định: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tốc độ học")
    parser.add_argument("--img_size", type=int, default=256, help="Kích thước ảnh")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model_type=args.model, epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, patience=args.patience,
                img_size=args.img_size, device=device)
