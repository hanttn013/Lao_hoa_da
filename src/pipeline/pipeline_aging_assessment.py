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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline.stage1_roi_alignment import align_and_crop_face_rois
from pipeline.stage3_morphometry import extract_morphometry
from pipeline.stage4_clinical_scoring_xai import compute_clinical_aging_score, generate_xai_heatmap
from models.Baseline.model import BaselineUNet
from models.AttentionUNet.model import AttentionUNet
from models.DSnake.model import DSnakeUNet

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

def run_aging_evaluation(image_path, model_type="dsnake", output_dir=None):
    """
    Quy trinh Danh gia Lao hoa Da Lam sang Toan dien (End-to-End):
    - Stage 1: Can chinh Face Mesh MediaPipe & xac dinh 4 vung giai phau (Tran, 2 Duoi mat, Ranh cuoi)
    - Stage 2: Phan doan Nep nhan bang DSnake / Attention U-Net / Baseline
    - Stage 3: Do luong hinh thai hoc dinh luong (Do dai skeletonized, mat do WAF)
    - Stage 4: Chan doan cap do lao hoa lam sang (Thang Lemperle/Merz 0-4) & Xuat ban do giai thich XAI
    """
    if output_dir is None:
        output_dir = os.path.join(SRC_DIR, "..", "results")
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Loi: Khong tim thay file anh tai {image_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_orig, w_orig = image_bgr.shape[:2]

    print("=" * 85)
    print("      HE THONG DANH GIA NEP NHAN & MUC DO LAO HOA DA LAM SANG (END-TO-END)")
    print(f"      Mo hinh: {model_type.upper()} | Thiet bi: {device}")
    print("=" * 85)

    # 1. Stage 1: Landmark Face Mesh & 4 ROIs
    rois, status = align_and_crop_face_rois(image_bgr)
    print(f"\n[Stage 1] Landmark Face Mesh: {status}")
    if rois:
        print("          Da phan tich thanh cong 4 phan vung giai phau:")
        for name in rois.keys():
            print(f"          + ROI '{name.upper()}': {rois[name].shape[1]}x{rois[name].shape[0]} px")

    # 2. Stage 2: Model Inference
    print(f"\n[Stage 2] Segmentation: Dang suy luan phan doan nep nhan...")
    if model_type == "baseline":
        ckpt_path = os.path.join(SRC_DIR, "models", "Baseline", "checkpoints", "baseline_unet.pth")
        model = BaselineUNet().to(device)
    elif model_type == "attention":
        ckpt_path = os.path.join(SRC_DIR, "models", "AttentionUNet", "checkpoints", "improved_attention_unet.pth")
        model = AttentionUNet().to(device)
    else:
        ckpt_path = os.path.join(SRC_DIR, "models", "DSnake", "checkpoints", "dsnake_unet.pth")
        model = DSnakeUNet().to(device)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    orig_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    tensor_in = prepare_input_tensor(orig_pil, img_size=256).to(device)

    with torch.no_grad():
        out = model(tensor_in)
        pred_256 = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask = cv2.resize(pred_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # 3. Stage 3: Quantitative Morphometry
    face_area = int(np.sum(np.any(image_bgr > 0, axis=-1)))
    metrics = extract_morphometry(mask, face_area)

    print("\n[Stage 3] Do luong hinh thai hoc (Quantitative Morphometry):")
    print(f"          + Dien tich da mat khao sat: {face_area:,} pixels")
    print(f"          + So pixel nep nhan boc tach: {metrics['wrinkle_pixels']:,} pixels")
    print(f"          + Tong chieu dai ranh nhan (Skeletonized): {metrics['wrinkle_length']:,} pixels")
    print(f"          + Ty le nep nhan (WAF): {metrics['area_fraction']*100:.3f}%")
    print(f"          + Mat do nep nhan: {metrics['wrinkle_density']:.5f}")

    # 4. Stage 4: Clinical Scoring & XAI Heatmap
    score, description = compute_clinical_aging_score(metrics)
    print("\n[Stage 4] Chan doan cap do lao hoa lam sang (Lemperle/Merz Scale):")
    print(f"          ⭐ CAP DO LAO HOA: {score} / 4")
    print(f"          Mo ta: {description}")

    # Generate and save XAI Heatmap
    heatmap_overlay = generate_xai_heatmap(image_bgr, mask)
    out_img_path = os.path.join(output_dir, "xai_wrinkle_heatmap.png")
    cv2.imwrite(out_img_path, heatmap_overlay)
    print(f"\n[XAI] Ban do giai thich nhiet (Overlay Heatmap) da duoc luu tai:\n      --> {out_img_path}")
    print("=" * 85)

if __name__ == "__main__":
    test_img = os.path.join(SRC_DIR, "..", "data", "ffhq_wrinkle_base", "masked_face_images", "00001.png")
    if os.path.exists(test_img):
        run_aging_evaluation(test_img, model_type="dsnake")
    else:
        print("Please supply an image path to test.")
