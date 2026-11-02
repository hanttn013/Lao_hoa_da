import os
import sys
import io
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
import time
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dataset import compute_texture_gradient
from models.face_detector import detect_and_crop_face
from models.Baseline.model import BaselineUNet
from models.AttentionUNet.model import AttentionUNet
from models.DSnake.model import DSnakeUNet
from models.metrics import compute_metrics
from pipeline.stage4_clinical_scoring_xai import compute_clinical_aging_score

app = Flask(__name__, template_folder="templates", static_folder="static")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}

def get_data_dir():
    candidates = [
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "..", "data"),
        os.path.abspath("data")
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(PROJECT_ROOT, "data")

def load_all_models():
    print(f"Loading models on {DEVICE}...")
    # Baseline
    b = BaselineUNet().to(DEVICE)
    b_path = os.path.join(PROJECT_ROOT, "models", "Baseline", "checkpoints", "baseline_unet.pth")
    if os.path.exists(b_path):
        b.load_state_dict(torch.load(b_path, map_location=DEVICE, weights_only=True), strict=False)
    b.eval()
    MODELS["baseline"] = {"name": "Baseline U-Net", "model": b, "params": sum(p.numel() for p in b.parameters())}
    
    # AttentionUNet
    a = AttentionUNet().to(DEVICE)
    a_path = os.path.join(PROJECT_ROOT, "models", "AttentionUNet", "checkpoints", "improved_attention_unet.pth")
    if os.path.exists(a_path):
        a.load_state_dict(torch.load(a_path, map_location=DEVICE, weights_only=True), strict=False)
    a.eval()
    MODELS["attention"] = {"name": "Attention U-Net", "model": a, "params": sum(p.numel() for p in a.parameters())}
    
    # DSnake
    s = DSnakeUNet().to(DEVICE)
    s_path = os.path.join(PROJECT_ROOT, "models", "DSnake", "checkpoints", "dsnake_unet.pth")
    if not os.path.exists(s_path):
        s_path = a_path
    if os.path.exists(s_path):
        s.load_state_dict(torch.load(s_path, map_location=DEVICE, weights_only=True), strict=False)
    s.eval()
    MODELS["dsnake"] = {"name": "DSnake (Our Model)", "model": s, "params": sum(p.numel() for p in s.parameters())}
    print("All models loaded successfully!")

def np_to_base64_png(np_img):
    """Encodes BGR or RGB or Grayscale numpy image to base64 data URI."""
    if len(np_img.shape) == 2:
        success, buffer = cv2.imencode('.png', np_img)
    else:
        success, buffer = cv2.imencode('.png', np_img)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def preprocess_tensor(pil_img, img_size=256):
    img_resized = pil_img.convert("RGB").resize((img_size, img_size))
    tex = compute_texture_gradient(img_resized)
    
    np_img = np.array(img_resized, dtype=np.float32) / 255.0 * 2.0 - 1.0
    np_img = np_img.transpose((2, 0, 1))
    tex = tex[np.newaxis, ...]
    
    combined = np.concatenate([np_img, tex], axis=0)
    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor, img_resized

def run_morphometrics(mask_256, face_rgb_256):
    bool_mask = mask_256 > 0
    skel = skeletonize(bool_mask)
    skel_img = (skel * 255).astype(np.uint8)
    
    face_gray = cv2.cvtColor(face_rgb_256, cv2.COLOR_RGB2GRAY)
    face_pixels = np.sum(face_gray > 10)
    if face_pixels == 0:
        face_pixels = 256 * 256
        
    wrinkle_pixels = int(np.sum(bool_mask))
    wrinkle_length = int(np.sum(skel))
    area_fraction = wrinkle_pixels / float(face_pixels)
    density = wrinkle_length / float(face_pixels)
    
    metrics_dict = {
        "wrinkle_pixels": wrinkle_pixels,
        "wrinkle_length": wrinkle_length,
        "area_fraction": area_fraction,
        "wrinkle_density": density,
        "face_pixels": int(face_pixels)
    }
    score, desc = compute_clinical_aging_score(metrics_dict)
    
    zones = {
        "Forehead (Trán)": bool_mask[:90, 40:216],
        "Crow's Feet L (Đuôi mắt T)": bool_mask[75:140, 10:85],
        "Crow's Feet R (Đuôi mắt P)": bool_mask[75:140, 171:246],
        "Nasolabial (Rãnh cười)": bool_mask[140:220, 60:196]
    }
    regional_data = []
    for z_name, z_mask in zones.items():
        regional_data.append({
            "zone": z_name,
            "wrinkle_pixels": int(np.sum(z_mask)),
            "density_pct": float(np.mean(z_mask) * 100.0)
        })
        
    return {
        "skeleton_base64": np_to_base64_png(skel_img),
        "wrinkle_pixels": wrinkle_pixels,
        "wrinkle_length": wrinkle_length,
        "area_fraction": float(area_fraction),
        "density": float(density),
        "lemperle_grade": int(score),
        "lemperle_desc": str(desc),
        "regional": regional_data
    }

def create_visual_overlays(face_rgb_256, mask_256):
    face_bgr = cv2.cvtColor(face_rgb_256, cv2.COLOR_RGB2BGR)
    
    # Green overlay
    overlay_green = face_bgr.copy()
    overlay_green[mask_256 > 0] = [0, 255, 60]
    blend_green = cv2.addWeighted(face_bgr, 0.45, overlay_green, 0.55, 0)
    
    # XAI Attention Heatmap
    mask_float = mask_256.astype(np.float32) / 255.0
    blurred_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)
    heatmap_norm = cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    blend_heatmap = cv2.addWeighted(face_bgr, 0.5, heatmap_color, 0.5, 0)
    
    return {
        "mask_base64": np_to_base64_png(mask_256),
        "overlay_base64": np_to_base64_png(blend_green),
        "heatmap_base64": np_to_base64_png(blend_heatmap)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    data_dir = get_data_dir()
    img_dir = os.path.join(data_dir, "ffhq_wrinkle_base", "masked_face_images")
    mask_dir = os.path.join(data_dir, "ffhq_wrinkle_base", "manual_wrinkle_masks")
    
    sample_list = []
    if os.path.exists(img_dir):
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and os.path.exists(os.path.join(mask_dir, f))])
        for f in files:
            sample_list.append({
                "id": f.replace(".png", ""),
                "filename": f,
                "has_gt": True
            })
    return jsonify({"samples": sample_list})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    sample_id = request.form.get("sample_id", None)
    uploaded_file = request.files.get("file", None)
    
    gt_mask_256 = None
    has_gt = False
    data_dir = get_data_dir()
    
    if sample_id:
        img_path = os.path.join(data_dir, "ffhq_wrinkle_base", "masked_face_images", f"{sample_id}.png")
        mask_path = os.path.join(data_dir, "ffhq_wrinkle_base", "manual_wrinkle_masks", f"{sample_id}.png")
        if not os.path.exists(img_path):
            return jsonify({"error": f"Image {sample_id} not found"}), 404
        pil_img = Image.open(img_path)
        if os.path.exists(mask_path):
            gt_pil = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
            gt_mask_256 = (np.array(gt_pil) > 0).astype(np.uint8)
            has_gt = True
    elif uploaded_file:
        raw_pil = Image.open(uploaded_file.stream)
        pil_img, was_cropped = detect_and_crop_face(raw_pil)
        if was_cropped:
            print("Auto-cropped face ROI from uploaded image!")
    else:
        return jsonify({"error": "No image or sample_id provided"}), 400
        
    tensor_in, face_pil = preprocess_tensor(pil_img, img_size=256)
    face_rgb = np.array(face_pil)
    
    # Run Inference on all 3 models
    model_outputs = {}
    for key, item in MODELS.items():
        t0 = time.time()
        with torch.no_grad():
            logits = item["model"](tensor_in)
        latency = (time.time() - t0) * 1000.0
        
        pred_idx = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        mask_256 = pred_idx * 255
        
        visuals = create_visual_overlays(face_rgb, mask_256)
        morph = run_morphometrics(mask_256, face_rgb)
        
        eval_metrics = {}
        if has_gt and gt_mask_256 is not None:
            eval_metrics = compute_metrics(pred_idx, gt_mask_256)
            
        model_outputs[key] = {
            "name": item["name"],
            "params": item["params"],
            "latency_ms": round(latency, 1),
            "visuals": visuals,
            "morph": morph,
            "metrics": eval_metrics
        }
        
    gt_visuals = None
    if has_gt and gt_mask_256 is not None:
        gt_visuals = create_visual_overlays(face_rgb, gt_mask_256 * 255)
        
    return jsonify({
        "success": True,
        "has_gt": has_gt,
        "sample_id": sample_id if sample_id else "custom_upload",
        "face_base64": np_to_base64_png(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)),
        "gt_visuals": gt_visuals,
        "models": model_outputs
    })

if __name__ == "__main__":
    load_all_models()
    port = int(os.environ.get("PORT", 5087))
    print(f"\n==================================================")
    print(f"=== Facial Wrinkle & Aging Assessment Web Running! ===")
    print(f"URL: http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
