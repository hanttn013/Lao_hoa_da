import os
import sys
import time
import torch
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.dataset import get_dataloaders
from models.Baseline.model import BaselineUNet
from models.losses import CombinedLoss
from models.metrics import compute_metrics

def train_baseline(epochs=5, batch_size=4, lr=3e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [BASELINE U-NET] Starting Training on {device} ===")
    
    train_loader, val_loader, test_loader, full_loader = get_dataloaders(SRC_DIR, batch_size=batch_size)
    
    model = BaselineUNet().to(device)
    criterion = CombinedLoss(use_boundary=False) # standard baseline loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_dice = 0.0
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "baseline_unet.pth")
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                tg = targets.cpu().numpy()
                for p, t in zip(preds, tg):
                    m = compute_metrics(p, t)
                    val_dice += m["dice"]
        val_dice /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  --> Saved new best baseline model to {best_ckpt_path} (Val Dice: {val_dice:.4f})")
            
    print(f"Baseline Training Completed in {time.time() - start_time:.2f}s. Best Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    train_baseline()
