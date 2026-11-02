import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "improve"))
from train import BaselineUNet, AttentionUNetWithFusion

def prepare_input_tensor(image_pil, img_size=256):
    img_resized = image_pil.convert("RGB").resize((img_size, img_size))
    gray = img_resized.convert("L")
    np_gray = np.array(gray, dtype=np.float32)
    grad_x = np.abs(np_gray[:, 1:] - np_gray[:, :-1])
    grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='edge')
    tex = grad_x / 255.0 * 2.0 - 1.0

    np_img = np.array(img_resized, dtype=np.float32) / 255.0 * 2.0 - 1.0
    np_img = np_img.transpose((2, 0, 1))
    tex = tex[np.newaxis, ...]
    combined = np.concatenate([np_img, tex], axis=0)
    return torch.tensor(combined, dtype=torch.float32).unsqueeze(0)

def make_sharp_overlay(base_bgr, mask_bin, color_bgr=(0, 255, 0), alpha=0.75):
    """Tạo lớp phủ màu sắc nét chuẩn y khoa (giữ nguyên độ sáng vùng da bình thường)"""
    overlay = base_bgr.copy().astype(np.float32)
    mask_bool = (mask_bin == 1)
    color_arr = np.array(color_bgr, dtype=np.float32)
    overlay[mask_bool] = (1.0 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)

def run_test_evaluation(max_visual_samples=73):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 95)
    print(f"🚀 BẮT ĐẦU CHẠY SUY LUẬN & XUẤT ẢNH TRỰC QUAN ĐỐI ĐẦU TRÊN TẬP TEST (Device: {device})")
    print("=" * 95)
    
    # 1. Tải danh sách file test chính thức từ test_file_lists.txt
    test_list_file = os.path.join(PROJECT_ROOT, "ffhq-wrinkle-dataset", "test_file_lists.txt")
    masked_dir = os.path.join(PROJECT_ROOT, "data", "ffhq_wrinkle_base", "masked_face_images")
    mask_gt_dir = os.path.join(PROJECT_ROOT, "data", "ffhq_wrinkle_base", "manual_wrinkle_masks")
    
    test_ids = []
    if os.path.exists(test_list_file):
        with open(test_list_file, "r") as f:
            for line in f:
                item_id = line.strip()
                if item_id and os.path.exists(os.path.join(masked_dir, f"{item_id}.png")):
                    test_ids.append(item_id)
                    
    if len(test_ids) == 0:
        all_imgs = sorted([f.replace(".png", "") for f in os.listdir(masked_dir) if f.endswith(".png")])
        test_ids = all_imgs[-50:]
        
    print(f"✅ Tìm thấy {len(test_ids)} ảnh trong tập Test chính thức.")
    
    # 2. Khởi tạo và nạp trọng số 2 mô hình
    ckpt_dir = os.path.join(PROJECT_ROOT, "improve", "checkpoints")
    base_ckpt = os.path.join(ckpt_dir, "baseline_unet.pth")
    imp_ckpt = os.path.join(ckpt_dir, "improved_attention_unet.pth")
    
    base_model = BaselineUNet().to(device)
    base_model.load_state_dict(torch.load(base_ckpt, map_location=device, weights_only=True))
    base_model.eval()
    
    imp_model = AttentionUNetWithFusion().to(device)
    imp_model.load_state_dict(torch.load(imp_ckpt, map_location=device, weights_only=True))
    imp_model.eval()
    
    # Thư mục lưu kết quả test
    out_base_dir = os.path.join(PROJECT_ROOT, "improve", "results", "test_predictions", "baseline")
    out_imp_dir = os.path.join(PROJECT_ROOT, "improve", "results", "test_predictions", "improved")
    out_vis_dir = os.path.join(PROJECT_ROOT, "improve", "results", "test_visual_comparisons")
    os.makedirs(out_base_dir, exist_ok=True)
    os.makedirs(out_imp_dir, exist_ok=True)
    os.makedirs(out_vis_dir, exist_ok=True)
    
    base_dices, imp_dices = [], []
    base_ious, imp_ious = [], []
    
    print("\n" + "=" * 95)
    print(f"{'Image ID':<10} | {'Dice (Baseline)':<18} | {'Dice (Improved)':<18} | {'IoU (Baseline)':<18} | {'IoU (Improved)':<18}")
    print("-" * 95)
    
    for idx, item_id in enumerate(test_ids):
        img_path = os.path.join(masked_dir, f"{item_id}.png")
        gt_path = os.path.join(mask_gt_dir, f"{item_id}.png")
        
        orig_img = Image.open(img_path)
        tensor_in = prepare_input_tensor(orig_img, img_size=256).to(device)
        
        # Load Ground Truth (Unicode-safe)
        gt_mask_pil = Image.open(gt_path).convert('L')
        gt_mask_256 = np.array(gt_mask_pil.resize((256, 256), Image.NEAREST))
        gt_bin = (gt_mask_256 > 0).astype(np.uint8)
        
        # Dự đoán Baseline & Improved
        with torch.no_grad():
            out_base = base_model(tensor_in)
            pred_base_256 = torch.argmax(out_base, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            out_imp = imp_model(tensor_in)
            pred_imp_256 = torch.argmax(out_imp, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
        # Tính Dice & IoU
        def calc_metrics(p, t):
            inter = np.sum((p == 1) & (t == 1))
            u_dice = np.sum(p == 1) + np.sum(t == 1)
            dice = (2.0 * inter) / (u_dice + 1e-6) if u_dice > 0 else 1.0
            u_iou = np.sum((p == 1) | (t == 1))
            iou = inter / (u_iou + 1e-6) if u_iou > 0 else 1.0
            return dice, iou
            
        d_base, iou_base = calc_metrics(pred_base_256, gt_bin)
        d_imp, iou_imp = calc_metrics(pred_imp_256, gt_bin)
        
        base_dices.append(d_base)
        imp_dices.append(d_imp)
        base_ious.append(iou_base)
        imp_ious.append(iou_imp)
        
        # Lưu mask dự đoán phân giải gốc 1024x1024 (Unicode-safe)
        orig_w, orig_h = orig_img.size
        pred_base_full = Image.fromarray((pred_base_256 * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        pred_imp_full = Image.fromarray((pred_imp_256 * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        pred_base_full.save(os.path.join(out_base_dir, f"{item_id}_mask.png"))
        pred_imp_full.save(os.path.join(out_imp_dir, f"{item_id}_mask.png"))
        
        # Xuất ảnh ghép so sánh trực quan độ nét cao (4 khung hình)
        if idx < max_visual_samples:
            img_np = cv2.cvtColor(np.array(orig_img.resize((256, 256))), cv2.COLOR_RGB2BGR)
            
            # 1. Ground Truth Overlay (Xanh ngọc BGR: 0, 255, 128)
            gt_overlay = make_sharp_overlay(img_np, gt_bin, color_bgr=(50, 220, 50), alpha=0.75)
            # 2. Baseline Overlay (Đỏ cam BGR: 40, 60, 255)
            base_overlay = make_sharp_overlay(img_np, pred_base_256, color_bgr=(40, 60, 255), alpha=0.75)
            # 3. Improved Overlay (Tím neon / Hồng BGR: 255, 50, 180)
            imp_overlay = make_sharp_overlay(img_np, pred_imp_256, color_bgr=(255, 50, 180), alpha=0.75)
            
            # Tạo thanh header ghi tiêu đề chuyên nghiệp
            row_images = np.hstack([img_np, gt_overlay, base_overlay, imp_overlay])
            canvas = np.zeros((256 + 40, 256 * 4, 3), dtype=np.uint8)
            canvas[40:, :] = row_images
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            titles = [
                f"Face Input ({item_id})",
                "Ground Truth (Bac si)",
                f"Baseline U-Net (Dice: {d_base:.3f})",
                f"Improved Attention (Dice: {d_imp:.3f})"
            ]
            colors_tit = [
                (255, 255, 255),
                (100, 255, 100),
                (100, 150, 255),
                (255, 120, 220)
            ]
            for col_i in range(4):
                cv2.putText(canvas, titles[col_i], (col_i * 256 + 10, 27), font, 0.48, colors_tit[col_i], 1, cv2.LINE_AA)
            
            comp_path = os.path.join(out_vis_dir, f"compare_{item_id}.png")
            Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(comp_path)
            
        print(f"{item_id:<10} | {d_base:<18.4f} | {d_imp:<18.4f} | {iou_base:<18.4f} | {iou_imp:<18.4f}")
        
    print("=" * 95)
    print(f"\n[KẾT QUẢ TỔNG HỢP TRÊN TẬP TEST ({len(test_ids)} ẢNH)]:")
    print(f"  * Baseline U-Net:          Mean Dice = {np.mean(base_dices):.4f} | Mean IoU = {np.mean(base_ious):.4f}")
    print(f"  * Improved Attention U-Net: Mean Dice = {np.mean(imp_dices):.4f} | Mean IoU = {np.mean(imp_ious):.4f}")
    print(f"\n[XUẤT TOÀN BỘ ẢNH SO SÁNH THÀNH CÔNG]:")
    print(f"  -> Đã lưu đầy đủ {min(len(test_ids), max_visual_samples)} ảnh so sánh 4 khung tại: {out_vis_dir}")
    print("=" * 95)

if __name__ == "__main__":
    run_test_evaluation(max_visual_samples=73)
