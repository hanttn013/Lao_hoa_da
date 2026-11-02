import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def build_hq_preprocessing_pipeline():
    # 1. Load image and landmarks
    img_path = 'docs/figures/raw_portrait_sample.png'
    lm_path = 'docs/figures/true_landmarks.json'
    
    img_pil = Image.open(img_path).convert('RGB')
    W, H = img_pil.size
    
    with open(lm_path) as f:
        lms = json.load(f)
    
    # Coordinates array
    coords = np.array([[p['x'] * W, p['y'] * H] for p in lms])
    
    # -------------------------------------------------------------
    # Panel (a): Raw Facial Portrait (Crop face focused 800x800)
    # Center around nose bridge (pt 168 or pt 1)
    center_x = int(coords[168, 0])
    center_y = int(coords[168, 1])
    box_size = int(min(W, H) * 0.78)
    half_box = box_size // 2
    
    x1 = max(0, center_x - half_box)
    y1 = max(0, center_y - int(half_box * 0.85)) # slightly shift down to cover neck
    x2 = min(W, x1 + box_size)
    y2 = min(H, y1 + box_size)
    
    # Re-adjust if out of bounds
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    crop_img = img_pil.crop((x1, y1, x2, y2)).resize((512, 512), Image.Resampling.LANCZOS)
    scale_x = 512.0 / crop_w
    scale_y = 512.0 / crop_h
    
    # Projected coordinates on 512x512 crop
    pts_512 = np.zeros_like(coords)
    pts_512[:, 0] = (coords[:, 0] - x1) * scale_x
    pts_512[:, 1] = (coords[:, 1] - y1) * scale_y
    
    # -------------------------------------------------------------
    # Panel (b): 468-pt MediaPipe Mesh & Roll Normalization
    panel_b = crop_img.copy()
    overlay_b = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw_b = ImageDraw.Draw(overlay_b)
    
    tesselation = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    contours = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    
    # 1. Draw fine tesselation mesh
    for edge in tesselation:
        p1 = (pts_512[edge.start, 0], pts_512[edge.start, 1])
        p2 = (pts_512[edge.end, 0], pts_512[edge.end, 1])
        draw_b.line([p1, p2], fill=(0, 240, 255, 75), width=1)
        
    # 2. Draw anatomical contours
    for edge in contours:
        p1 = (pts_512[edge.start, 0], pts_512[edge.start, 1])
        p2 = (pts_512[edge.end, 0], pts_512[edge.end, 1])
        draw_b.line([p1, p2], fill=(0, 255, 230, 200), width=1)
        
    # 3. Draw key landmarks points
    for i in range(468):
        x, y = pts_512[i, 0], pts_512[i, 1]
        draw_b.ellipse([x-1, y-1, x+1, y+1], fill=(200, 255, 255, 180))
        
    # 4. Inter-pupillary axis (Pupil centers: Left pt 468, Right pt 473 if 478, or left_eye center)
    # MediaPipe iris centers: left iris=468, right iris=473
    if len(lms) >= 478:
        left_eye_pt = pts_512[468]
        right_eye_pt = pts_512[473]
    else:
        left_eye_pt = pts_512[33]
        right_eye_pt = pts_512[263]
        
    # Draw horizontal reference line and eye axis
    # Inter-pupillary axis
    draw_b.line([(left_eye_pt[0] - 25, left_eye_pt[1]), (right_eye_pt[0] + 25, right_eye_pt[1])], fill=(255, 40, 40, 255), width=3)
    
    # Pupil landmarks (Glowing red circles)
    for p in [left_eye_pt, right_eye_pt]:
        draw_b.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill=(255, 255, 255, 255), outline=(255, 0, 0, 255), width=2)
        draw_b.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(255, 0, 0, 255))
        
    panel_b = Image.alpha_composite(panel_b.convert('RGBA'), overlay_b).convert('RGB')
    
    # -------------------------------------------------------------
    # Panel (c): Anatomical Skin Mask (Pruning Non-Skin)
    # Create mask using face oval, subtracting eyes and lips
    mask_c = Image.new('L', (512, 512), 0)
    draw_c = ImageDraw.Draw(mask_c)
    
    # Face oval points (indices from MediaPipe)
    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    face_polygon = [(pts_512[idx, 0], pts_512[idx, 1]) for idx in face_oval_idx]
    draw_c.polygon(face_polygon, fill=255)
    
    # Natural curved forehead directly from MediaPipe face oval points
    # (points 109, 67, 10, 338, 297 form the natural continuous hairline arc)
    
    # Prune Left Eye
    left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    draw_c.polygon([(pts_512[idx, 0], pts_512[idx, 1]) for idx in left_eye_idx], fill=0)
    
    # Prune Right Eye
    right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    draw_c.polygon([(pts_512[idx, 0], pts_512[idx, 1]) for idx in right_eye_idx], fill=0)
    
    # Prune Lips / Mouth Cavity
    lips_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    draw_c.polygon([(pts_512[idx, 0], pts_512[idx, 1]) for idx in lips_idx], fill=0)
    
    # Convert mask to 3-channel visual
    mask_np = np.array(mask_c)
    panel_c = Image.fromarray(np.stack([mask_np]*3, axis=-1))
    
    # -------------------------------------------------------------
    # Panel (d): Directional Texture Gx (High-Freq Fissures)
    gray = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # Directional gradient Gx = |I(x+1, y) - I(x, y)|
    gx = cv2.Sobel(gray_clahe, cv2.CV_64F, 1, 0, ksize=3)
    gx_abs = np.abs(gx)
    gx_norm = (gx_abs / (gx_abs.max() + 1e-6) * 255.0).astype(np.uint8)
    
    # Mask out non-skin
    gx_masked = np.where(mask_np > 0, gx_norm, 0)
    
    # Colormap with Inferno/Magma for aesthetic micro-fissure visualization
    gx_color = cv2.applyColorMap(gx_masked, cv2.COLORMAP_MAGMA)
    # Set background outside skin to black
    gx_color[mask_np == 0] = [5, 5, 12]
    panel_d = Image.fromarray(cv2.cvtColor(gx_color, cv2.COLOR_BGR2RGB))
    
    # -------------------------------------------------------------
    # Panel (e): Dual-Stream Input [RGB (3ch) + Gx (1ch)]
    # Visual blend: RGB face with skin-mask, highlighted texture in cyan-green
    crop_np = np.array(crop_img).astype(np.float32)
    # Apply skin mask softly
    skin_alpha = (mask_np.astype(np.float32) / 255.0)[:, :, None]
    face_skin = crop_np * skin_alpha
    
    # Overlay Gx texture as radiant aesthetic groove channel
    texture_glow = np.zeros_like(face_skin)
    texture_glow[:, :, 0] = 0.0                      # Red
    texture_glow[:, :, 1] = gx_masked.astype(np.float32) * 1.3  # Green
    texture_glow[:, :, 2] = gx_masked.astype(np.float32) * 1.5  # Blue
    
    dual_stream_vis = np.clip(face_skin + texture_glow * 0.7, 0, 255).astype(np.uint8)
    panel_e = Image.fromarray(dual_stream_vis)
    
    # -------------------------------------------------------------
    # Assemble 5 panels into a cohesive 300 DPI figure
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), dpi=300)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.06)
    
    panels = [
        (crop_img, "(a) Raw Facial Portrait\n(High-Res Portrait)"),
        (panel_b, "(b) 468-pt MediaPipe Mesh\n& Roll Normalization ($\\theta=0^\\circ$)"),
        (panel_c, "(c) Anatomical Skin Mask\n(Pruning Non-Skin & Cavities)"),
        (panel_d, "(d) Directional Texture $\\mathbf{G}_x$\n(High-Frequency Fissures)"),
        (panel_e, "(e) Dual-Stream Input\n[RGB (3ch) + $\\mathbf{G}_x$ (1ch)]")
    ]
    
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8, color='#0f172a')
        ax.axis('off')
        
    out_path = 'docs/figures/preprocessing_pipeline.png'
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print('SUCCESS: Rendered HQ Preprocessing Pipeline to', out_path)

if __name__ == '__main__':
    build_hq_preprocessing_pipeline()
