import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

def compute_texture_gradient(image_pil):
    """
    Computes horizontal high-frequency texture gradient map for the 4th channel.
    Matches the official preprocessing protocol.
    """
    gray = image_pil.convert("L")
    np_gray = np.array(gray, dtype=np.float32)
    grad_x = np.abs(np_gray[:, 1:] - np_gray[:, :-1])
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    tex = grad_x / 255.0 * 2.0 - 1.0 # [-1, 1]
    return tex

class WrinkleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list=None, img_size=256, is_train=False):
        self.img_size = img_size
        self.is_train = is_train
        self.samples = []
        valid_exts = [".png", ".jpg"]
        
        if file_list is not None:
            for f in file_list:
                img_path = os.path.join(img_dir, f)
                mask_path = os.path.join(mask_dir, f)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path, f))
        elif os.path.exists(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if any(f.endswith(ext) for ext in valid_exts):
                    img_path = os.path.join(img_dir, f)
                    mask_path = os.path.join(mask_dir, f)
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, filename = self.samples[idx]
        image = Image.open(img_p).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_p).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        
        tex = compute_texture_gradient(image)
        np_img = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0 # [-1, 1]
        np_img = np_img.transpose((2, 0, 1))
        tex = tex[np.newaxis, ...]
        
        combined = np.concatenate([np_img, tex], axis=0) # [4, H, W]
        target = (np.array(mask) > 0).astype(np.int64)
        
        return torch.tensor(combined, dtype=torch.float32), torch.tensor(target, dtype=torch.long), filename

def get_dataloaders(project_root, img_size=256, batch_size=8, seed=42):
    possible_data_dirs = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "..", "data"),
        os.path.abspath("data"),
        os.path.abspath("../data")
    ]
    data_dir = None
    for p in possible_data_dirs:
        candidate_img = os.path.join(p, "ffhq_wrinkle_base", "masked_face_images")
        if os.path.exists(candidate_img):
            data_dir = p
            break
            
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")

    img_dir = os.path.join(data_dir, "ffhq_wrinkle_base", "masked_face_images")
    mask_dir = os.path.join(data_dir, "ffhq_wrinkle_base", "manual_wrinkle_masks")
    
    if not os.path.exists(img_dir):
        print(f"Warning: Dataset path {img_dir} does not exist yet. Please download dataset.")
        return None, None, None, None
        
    all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and os.path.exists(os.path.join(mask_dir, f))])
    
    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))
    
    # 50 train, 10 val, 13 test (out of 73 total available ground truth pairs)
    train_idx = indices[:50]
    val_idx = indices[50:60]
    test_idx = indices[60:]
    
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    test_files = [all_files[i] for i in test_idx]
    
    train_ds = WrinkleDataset(img_dir, mask_dir, train_files, img_size=img_size, is_train=True)
    val_ds = WrinkleDataset(img_dir, mask_dir, val_files, img_size=img_size, is_train=False)
    test_ds = WrinkleDataset(img_dir, mask_dir, test_files, img_size=img_size, is_train=False)
    all_test_ds = WrinkleDataset(img_dir, mask_dir, all_files, img_size=img_size, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    full_eval_loader = DataLoader(all_test_ds, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader, full_eval_loader
