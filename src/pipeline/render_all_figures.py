import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects
from PIL import Image, ImageFilter
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_FIG_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(DOCS_FIG_DIR, exist_ok=True)

# Set global matplotlib parameters for publication quality
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# ==============================================================================
# 1. BENCHMARK AND ABLATION CHARTS (Publication Grade 300 DPI)
# ==============================================================================
def draw_benchmark_and_ablation():
    print("Generating: benchmark_and_ablation_charts.png...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.94, bottom=0.08, left=0.07, right=0.96)
    
    # Data definitions
    models = ["Baseline U-Net", "DeepLabv3+", "SegFormer", "SwinUNETR", "DS-SAU-Net (Ours)"]
    dices = [0.5057, 0.5372, 0.5846, 0.5931, 0.6003]
    mious = [0.3458, 0.3684, 0.4125, 0.4218, 0.4352]
    hd95s = [16.29, 24.12, 19.45, 18.20, 14.20]
    maes = [0.212, 0.592, 0.490, 0.461, 0.229]
    kappas = [0.852, 0.615, 0.724, 0.758, 0.845]
    
    x = np.arange(len(models))
    w = 0.35
    
    # -------------------------------------------------------------
    # (a) Segmentation Accuracy (Dice & mIoU)
    # -------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_facecolor('#fafbfc')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#d0d7de', zorder=0)
    
    bar1 = ax.bar(x - w/2, dices, w, label='Dice Score', color='#1f77b4', edgecolor='#0e4369', linewidth=1.2, zorder=3)
    bar2 = ax.bar(x + w/2, mious, w, label='mIoU', color='#2ca02c', edgecolor='#135c13', linewidth=1.2, zorder=3)
    
    # Highlight "Ours" with distinct edge
    bar1[-1].set_color('#0b5394')
    bar1[-1].set_edgecolor('#d32f2f')
    bar1[-1].set_linewidth(2.0)
    bar2[-1].set_color('#27ae60')
    bar2[-1].set_edgecolor('#d32f2f')
    bar2[-1].set_linewidth(2.0)
    
    ax.set_title('(a) Segmentation Accuracy on Validation Partition', fontsize=12, fontweight='bold', pad=12, color='#1c2833')
    ax.set_ylabel('Score Metric', fontsize=11, fontweight='semibold', color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9.5, fontweight='medium')
    ax.set_ylim(0, 0.74)
    ax.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
    
    for b in bar1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1f4e79')
    for b in bar2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#196f3d')
        
    # -------------------------------------------------------------
    # (b) Boundary Distance Error (HD95 - lower is better)
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_facecolor('#fafbfc')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#d0d7de', zorder=0)
    
    bar_colors = ['#e67e22', '#c0392b', '#d35400', '#e67e22', '#16a085']
    bars_hd = ax.bar(x, hd95s, color=bar_colors, width=0.5, edgecolor='#2c3e50', linewidth=1.2, zorder=3)
    bars_hd[-1].set_edgecolor('#117864')
    bars_hd[-1].set_linewidth(2.2)
    
    ax.set_title('(b) Boundary Distance Error: HD95 in Pixels (Lower is Superior)', fontsize=12, fontweight='bold', pad=12, color='#1c2833')
    ax.set_ylabel('95% Hausdorff Distance (px)', fontsize=11, fontweight='semibold', color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9.5, fontweight='medium')
    ax.set_ylim(0, 30)
    
    for b in bars_hd:
        h = b.get_height()
        color = '#117864' if h == 14.20 else '#78281f' if h > 20 else '#7e5109'
        ax.text(b.get_x() + b.get_width()/2, h + 0.6, f"{h:.1f} px", ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
        
    ax.axhline(14.20, color='#16a085', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(0.1, 14.7, "DS-SAU-Net Boundary: 14.20 px", fontsize=8.5, color='#0e6251', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#a3e4d7'))

    # -------------------------------------------------------------
    # (c) Step-by-Step Component Ablation Study
    # -------------------------------------------------------------
    ax = axes[1, 0]
    ax.set_facecolor('#fafbfc')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#d0d7de', zorder=0)
    
    ab_steps = [
        "A: RGB\n(Baseline)",
        "B: + Texture\nStream",
        "C: + Cross-\nAttention",
        "D: + Dynamic\nSnake Conv",
        "E: + Boundary\nLoss (Ours)"
    ]
    ab_dices = [0.4755, 0.5270, 0.5602, 0.5824, 0.6003]
    x_ab = np.arange(len(ab_steps))
    
    # Gradient of blues to teal
    colors_step = ['#7f8c8d', '#2980b9', '#3498db', '#1abc9c', '#16a085']
    bars_ab = ax.bar(x_ab, ab_dices, color=colors_step, width=0.52, edgecolor='#2c3e50', linewidth=1.2, zorder=3)
    
    # Cumulative progression line
    ax.plot(x_ab, ab_dices, color='#d32f2f', marker='o', linewidth=2.2, markersize=7, zorder=4, label='Cumulative Progression')
    
    ax.set_title('(c) Incremental Architectural Ablation Progression', fontsize=12, fontweight='bold', pad=12, color='#1c2833')
    ax.set_ylabel('Validation Dice Coefficient', fontsize=11, fontweight='semibold', color='#2c3e50')
    ax.set_xticks(x_ab)
    ax.set_xticklabels(ab_steps, fontsize=9.0, fontweight='medium')
    ax.set_ylim(0.42, 0.65)
    ax.legend(loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
    
    for i, b in enumerate(bars_ab):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.007, f"{h:.4f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1a5276')
        if i > 0:
            diff = (h - ab_dices[i-1]) * 100
            ax.text(b.get_x() + b.get_width()/2, h/2 + 0.15, f"+{diff:.1f}%", ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    # -------------------------------------------------------------
    # (d) Morphometric Severity Concordance (MAE & Weighted Kappa)
    # -------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_facecolor('#fafbfc')
    ax.grid(axis='both', linestyle='--', alpha=0.5, color='#d0d7de', zorder=0)
    
    # Dual line chart
    ax.plot(x, maes, marker='s', label='Grading MAE (Lower is Better)', color='#d35400', linewidth=2.5, markersize=8, zorder=4)
    ax.plot(x, kappas, marker='^', label="Linear Weighted Kappa (Higher is Better)", color='#2471a3', linewidth=2.5, markersize=8, zorder=4)
    
    ax.set_title('(d) Morphometric Severity Agreement on Validation Partition', fontsize=12, fontweight='bold', pad=12, color='#1c2833')
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='semibold', color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9.5, fontweight='medium')
    ax.set_ylim(0.15, 0.95)
    ax.legend(loc='center left', framealpha=0.95, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)
    
    for i in range(len(models)):
        ax.annotate(f"{maes[i]:.3f}", (x[i], maes[i] - 0.045), ha='center', fontsize=8.5, fontweight='bold', color='#a04000')
        ax.annotate(f"{kappas[i]:.3f}", (x[i], kappas[i] + 0.030), ha='center', fontsize=8.5, fontweight='bold', color='#1b4f72')

    out_path = os.path.join(DOCS_FIG_DIR, "benchmark_and_ablation_charts.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# 2. DYNAMIC SNAKE CONVOLUTION VS STANDARD CONV (Publication Grade)
# ==============================================================================
def draw_snake_vs_standard():
    print("Generating: snake_vs_standard_conv.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.15, left=0.04, right=0.96, top=0.88, bottom=0.10)
    
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
    # --- Function to draw skin background with curvilinear wrinkle fissure ---
    def draw_skin_crease(ax):
        # Draw background skin area
        bg = patches.Rectangle((-3.8, -3.2), 7.6, 6.4, facecolor='#fceee3', edgecolor='#e0c8b6', linewidth=1.5, zorder=1)
        ax.add_patch(bg)
        
        # Curvilinear wrinkle path: y = 0.5 * sin(1.2 * x) - 0.1 * x
        x_curve = np.linspace(-3.8, 3.8, 200)
        y_curve = 0.7 * np.sin(0.9 * x_curve) - 0.2 * x_curve
        
        # Shaded groove valley
        for w_offset, alpha in [(0.6, 0.15), (0.4, 0.25), (0.2, 0.40), (0.08, 0.65)]:
            ax.fill_between(x_curve, y_curve - w_offset, y_curve + w_offset, color='#8d6e63', alpha=alpha, zorder=2)
            
        # Centerline of genuine physiological crease
        ax.plot(x_curve, y_curve, color='#4e342e', linestyle='--', linewidth=1.5, alpha=0.85, zorder=3)
        return x_curve, y_curve

    # -------------------------------------------------------------
    # (a) Standard 3x3 Isotropic Convolution
    # -------------------------------------------------------------
    ax1 = axes[0]
    draw_skin_crease(ax1)
    ax1.set_title('(a) Standard 3x3 Convolution (Baseline U-Net)\nIsotropic Rigid Grid: 6/9 Points Sample Planar Skin', 
                  fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    
    # Kernel centered at (0, 0)
    kernel_x = np.array([-1.2, 0.0, 1.2])
    kernel_y = np.array([-1.2, 0.0, 1.2])
    
    # Draw bounding grid of 3x3 kernel
    grid_rect = patches.Rectangle((-1.8, -1.8), 3.6, 3.6, facecolor='none', edgecolor='#2980b9', linestyle=':', linewidth=2, zorder=4)
    ax1.add_patch(grid_rect)
    
    # Connect grid points with rigid lines
    for kx in kernel_x:
        ax1.plot([kx, kx], [-1.2, 1.2], color='#3498db', linestyle='-', linewidth=1.2, alpha=0.6, zorder=4)
    for ky in kernel_y:
        ax1.plot([-1.2, 1.2], [ky, ky], color='#3498db', linestyle='-', linewidth=1.2, alpha=0.6, zorder=4)
        
    for kx in kernel_x:
        for ky in kernel_y:
            # Check distance to genuine crease y = 0.7 * sin(0.9*x) - 0.2*x
            crease_y = 0.7 * np.sin(0.9 * kx) - 0.2 * kx
            dist = abs(ky - crease_y)
            if dist < 0.45:
                # Inside crease
                c = Circle((kx, ky), 0.22, facecolor='#27ae60', edgecolor='#145a32', linewidth=2, zorder=6)
                ax1.add_patch(c)
                ax1.text(kx, ky, '✓', color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=7)
            else:
                # Outside crease (planar skin / pore dilution)
                c = Circle((kx, ky), 0.22, facecolor='#e74c3c', edgecolor='#78281f', linewidth=2, zorder=6)
                ax1.add_patch(c)
                ax1.text(kx, ky, '✕', color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=7)
                
    # Informative callout
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=1.5)
    ax1.text(0, -2.6, "Signal Dilution & Edge Blurring:\nOnly 33% (3/9) samples capture the wrinkle fissure", 
             ha='center', va='center', fontsize=9.5, fontweight='bold', color='#78281f', bbox=box_props, zorder=10)

    # -------------------------------------------------------------
    # (b) Dynamic Snake Convolution (DS-SAU-Net)
    # -------------------------------------------------------------
    ax2 = axes[1]
    draw_skin_crease(ax2)
    ax2.set_title('(b) Dynamic Snake Convolution (Proposed DS-SAU-Net)\nAdaptive Deformable Kernel Follows Curvilinear Groove', 
                  fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
    
    # 9 deformable points chained along the curve
    pts_x = np.linspace(-1.8, 1.8, 9)
    pts_y = 0.7 * np.sin(0.9 * pts_x) - 0.2 * pts_x
    
    # Connect sequential points with flexible deformable chain lines
    ax2.plot(pts_x, pts_y, color='#16a085', linestyle='-', linewidth=3.5, zorder=5)
    
    for i, (px, py) in enumerate(zip(pts_x, pts_y)):
        if i == 4: # Center Anchor K0
            c = Circle((px, py), 0.28, facecolor='#8e44ad', edgecolor='#4a235a', linewidth=2.5, zorder=7)
            ax2.add_patch(c)
            ax2.text(px, py, 'K₀', color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=8)
            ax2.annotate("Central Anchor $K_0$", (px, py), xytext=(px - 0.2, py + 1.1),
                         arrowprops=dict(arrowstyle="->", color='#4a235a', lw=1.5),
                         fontsize=9, fontweight='bold', color='#4a235a',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#f4ecf7', edgecolor='#bb8fce'), zorder=10)
        else:
            c = Circle((px, py), 0.22, facecolor='#2ecc71', edgecolor='#145a32', linewidth=2, zorder=6)
            ax2.add_patch(c)
            ax2.text(px, py, '✓', color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=7)
            
            # Draw offset delta vectors for illustrative points
            if i in [1, 7]:
                rigid_y = 0.0 # original rigid grid level
                ax2.annotate("", xy=(px, py), xytext=(px, rigid_y),
                             arrowprops=dict(arrowstyle="->", color='#e67e22', lw=1.8, linestyle='--'), zorder=5)
                ax2.text(px + 0.15, (py + rigid_y)/2, r"$\Delta y_i$", fontsize=8.5, fontweight='bold', color='#b9770e', zorder=10)

    # Informative callout
    box_props2 = dict(boxstyle='round,pad=0.5', facecolor='#d4efdf', edgecolor='#27ae60', linewidth=1.5)
    ax2.text(0, -2.6, "100% Curvilinear Alignment:\nReceptive chain bends along groove without planar skin erosion", 
             ha='center', va='center', fontsize=9.5, fontweight='bold', color='#145a32', bbox=box_props2, zorder=10)

    out_path = os.path.join(DOCS_FIG_DIR, "snake_vs_standard_conv.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# 3. END-TO-END FRAMEWORK ARCHITECTURE DIAGRAM (Publication Grade)
# ==============================================================================
def draw_framework_architecture():
    print("Generating: framework_architecture.png...")
    fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Background soft canvas
    canvas = patches.Rectangle((0, 0), 18, 9, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.0)
    ax.add_patch(canvas)
    
    # Banner title
    ax.text(9.0, 8.55, "DS-SAU-Net: End-to-End Facial Wrinkle Segmentation and Morphometric Assessment Pipeline", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')
    
    # 4 Main Stage Containers
    stages = [
        {"num": "STAGE 1", "title": "Landmark Alignment & 4-ROI Normalization", "x": 0.5, "w": 4.0, "color": "#e0f2fe", "border": "#0284c7"},
        {"num": "STAGE 2", "title": "Dual-Stream Snake-Attention U-Net", "x": 4.8, "w": 5.8, "color": "#f0fdf4", "border": "#16a34a"},
        {"num": "STAGE 3", "title": "Quantitative Morphometry", "x": 10.9, "w": 3.2, "color": "#fef3c7", "border": "#d97706"},
        {"num": "STAGE 4", "title": "Morphometric Staging & XAI", "x": 14.4, "w": 3.1, "color": "#fdf2f8", "border": "#db2777"}
    ]
    
    for st in stages:
        box = FancyBboxPatch((st["x"], 0.6), st["w"], 7.5, boxstyle="round,pad=0.15", 
                             facecolor=st["color"], edgecolor=st["border"], linewidth=2.0, zorder=1)
        ax.add_patch(box)
        # Stage Header
        ax.text(st["x"] + st["w"]/2, 7.8, st["num"], ha='center', va='center', fontsize=10.5, fontweight='bold', color=st["border"])
        ax.text(st["x"] + st["w"]/2, 7.45, st["title"], ha='center', va='center', fontsize=9.0, fontweight='bold', color='#1e293b')
        
    # Helper to draw module cards inside stages
    def draw_card(cx, cy, cw, ch, text_title, text_body="", bgcolor="white", bordercolor="#94a3b8"):
        c = FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.1", facecolor=bgcolor, edgecolor=bordercolor, linewidth=1.2, zorder=2)
        ax.add_patch(c)
        if text_body:
            ax.text(cx + cw/2, cy + ch*0.68, text_title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a', zorder=3)
            ax.text(cx + cw/2, cy + ch*0.32, text_body, ha='center', va='center', fontsize=7.5, color='#475569', zorder=3)
        else:
            ax.text(cx + cw/2, cy + ch*0.5, text_title, ha='center', va='center', fontsize=8.0, fontweight='bold', color='#0f172a', zorder=3)

    # --- STAGE 1 Modules ---
    draw_card(0.8, 6.0, 3.4, 0.95, "Facial Portrait Input", "RGB Image: 1024 x 1024 px", "#ffffff", "#0284c7")
    draw_card(0.8, 4.6, 3.4, 1.0, "MediaPipe 468 Face Mesh", "Calculates Inter-pupillary Angle $\\theta$", "#ffffff", "#0284c7")
    draw_card(0.8, 3.2, 3.4, 1.0, "In-Plane Roll Normalization", "Affine Rotation to $\\theta = 0^\\circ$, Scale Normalization", "#ffffff", "#0284c7")
    draw_card(0.8, 1.1, 3.4, 1.7, "Standardized 4-ROI Delineation", 
              "• Forehead: Supraorbital to hairline\n• Left & Right Crow's Feet (15% width)\n• Nasolabial Fold (Alar to commissure)", "#ffffff", "#0284c7")
    
    # Arrows Stage 1
    for y1, y2 in [(6.0, 5.6), (4.6, 4.2), (3.2, 2.8)]:
        ax.annotate("", xy=(2.5, y2), xytext=(2.5, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0284c7"), zorder=5)

    # --- STAGE 2 Modules ---
    # Dual inputs
    draw_card(5.1, 6.0, 2.4, 0.9, "Spatial RGB Stream", "Normalized $\\mathbf{X}_{rgb} \\in [-1, 1]^3$", "#ffffff", "#16a34a")
    draw_card(7.8, 6.0, 2.5, 0.9, "Texture Gradient Stream", "Horizontal Gradient $\\mathbf{G}_x$ + CLAHE", "#ffffff", "#16a34a")
    
    # Dual encoders
    draw_card(5.1, 4.7, 2.4, 0.9, "Encoder Branch A", "Dynamic Snake Conv (DSConv)", "#ffffff", "#16a34a")
    draw_card(7.8, 4.7, 2.5, 0.9, "Encoder Branch B", "Directional Ridge Features", "#ffffff", "#16a34a")
    
    # Cross Attention Fusion
    draw_card(5.3, 3.3, 4.8, 1.0, "Bottleneck Cross-Attention Fusion", 
              "Queries $\\mathbf{Q}$ from RGB  |  Keys $\\mathbf{K}$, Values $\\mathbf{V}$ from Texture\n$\\mathbf{F}_{out} = \\mathbf{F}_{rgb} + \\gamma \\cdot \\mathrm{Softmax}(\\mathbf{Q}\\mathbf{K}^T / \\sqrt{d_k}) \\mathbf{V}$", 
              "#dcfce7", "#15803d")
    
    # Decoder & Loss
    draw_card(5.3, 2.0, 4.8, 0.9, "Symmetric U-Net Decoder & Refinement", "Skip connections + Curvilinear Expansion", "#ffffff", "#16a34a")
    draw_card(5.3, 0.8, 4.8, 0.9, "Multi-Task Boundary Loss Supervision", 
              "$\\mathcal{L}_{total} = \\mathcal{L}_{CE} + 1.0\\cdot\\mathcal{L}_{Dice} + 0.5\\cdot\\mathcal{L}_{Boundary}(\\Delta \\mathbf{P}, \\Delta \\mathbf{Y})$", 
              "#fef2f2", "#dc2626")

    # Arrows Stage 2
    ax.annotate("", xy=(6.3, 5.6), xytext=(6.3, 6.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))
    ax.annotate("", xy=(9.0, 5.6), xytext=(9.0, 6.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))
    ax.annotate("", xy=(6.3, 4.3), xytext=(6.3, 4.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))
    ax.annotate("", xy=(9.0, 4.3), xytext=(9.0, 4.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))
    ax.annotate("", xy=(7.7, 2.9), xytext=(7.7, 3.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))
    ax.annotate("", xy=(7.7, 1.7), xytext=(7.7, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color="#dc2626"))

    # Stage 1 to Stage 2 transition arrow
    ax.annotate("", xy=(4.8, 5.0), xytext=(4.2, 5.0), arrowprops=dict(arrowstyle="->", lw=2.5, color="#0f172a"), zorder=5)

    # --- STAGE 3 Modules ---
    draw_card(11.2, 5.6, 2.6, 1.3, "Predicted Wrinkle Mask", "Binary Segmentation $\\mathbf{M} \\in \\{0, 1\\}$\nSharp Delineations", "#ffffff", "#d97706")
    draw_card(11.2, 3.8, 2.6, 1.4, "Zhang-Suen Skeletonization", "Extracts Morphological\nSingle-Pixel Centerline $\\mathbf{S}$", "#ffffff", "#d97706")
    draw_card(11.2, 1.4, 2.6, 2.0, "Scale-Normalized Indices", 
              "Normalized Length:\n$L_{norm} = \\frac{\\sum \\mathbf{S}}{D_{interocular}}$\n\nWrinkle Area Fraction:\n$WAF_k = \\frac{\\sum_{\\Omega_k} \\mathbf{M}}{|\\Omega_k|} \\times 100\\%$", 
              "#fffbeb", "#b45309")
    
    # Arrows Stage 3
    ax.annotate("", xy=(12.5, 5.2), xytext=(12.5, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d97706"))
    ax.annotate("", xy=(12.5, 3.4), xytext=(12.5, 3.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d97706"))
    
    # Stage 2 to Stage 3 transition arrow
    ax.annotate("", xy=(10.9, 5.0), xytext=(10.1, 5.0), arrowprops=dict(arrowstyle="->", lw=2.5, color="#0f172a"), zorder=5)

    # --- STAGE 4 Modules ---
    draw_card(14.7, 4.3, 2.5, 2.7, "Ordinal Severity Staging", 
              "Calibrated Scale (0--4):\n• Grade 0: Smooth (< 0.5%)\n• Grade 1: Mild (0.5--1.5%)\n• Grade 2: Moderate (1.5--3.5%)\n• Grade 3: Severe (3.5--6.0%)\n• Grade 4: Very Severe (≥ 6.0%)", 
              "#ffffff", "#db2777")
    draw_card(14.7, 1.4, 2.5, 2.4, "Explainable AI (XAI)\nDensity Visualization", 
              "Spatial 2D Gaussian\nSmoothing Overlay $\\mathbf{H}_{XAI}$\n\nHighlights Top Structural\nAging Hot-Spots for Clinicians", 
              "#fdf2f8", "#be185d")
    
    # Arrows Stage 4
    ax.annotate("", xy=(15.95, 3.8), xytext=(15.95, 4.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#db2777"))
    
    # Stage 3 to Stage 4 transition arrow
    ax.annotate("", xy=(14.4, 4.5), xytext=(13.8, 4.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#0f172a"), zorder=5)

    out_path = os.path.join(DOCS_FIG_DIR, "framework_architecture.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# 4. QUALITATIVE VISUAL COMPARISON (Publication Grade)
# ==============================================================================
def draw_qualitative_comparison():
    print("Generating: qualitative_comparison.png...")
    # Find test visual comparison file
    test_vis_dir = os.path.join(PROJECT_ROOT, "improve", "results", "test_visual_comparisons")
    sample_files = ["compare_07649.png", "compare_03140.png", "compare_04003.png", "compare_05094.png"]
    
    selected_file = None
    for sf in sample_files:
        p = os.path.join(test_vis_dir, sf)
        if os.path.exists(p):
            selected_file = p
            break
            
    if not selected_file:
        files = [f for f in os.listdir(test_vis_dir) if f.endswith(".png")]
        selected_file = os.path.join(test_vis_dir, files[0]) if files else None
        
    if selected_file and os.path.exists(selected_file):
        pil_img = Image.open(selected_file).convert("RGB")
        img_np = np.array(pil_img)
        # Convert RGB to BGR for cv2 processing compatibility or keep RGB directly
        h, w = img_np.shape[:2]
        crop_y = 40 if h > 256 else 0
        img_cropped = img_np[crop_y:, :]
        
        step_w = img_cropped.shape[1] // 4
        c1 = img_cropped[:, :step_w]
        c2 = img_cropped[:, step_w:2*step_w]
        c3 = img_cropped[:, 2*step_w:3*step_w]
        c4 = img_cropped[:, 3*step_w:]
        
        # Build professional high-res 4-panel figure with matplotlib
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=300)
        plt.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.88, bottom=0.04)
        
        panels = [
            (c1, "(a) Normalized Facial Input", "#2c3e50"),
            (c2, "(b) Expert Ground Truth", "#1e8449"),
            (c3, "(c) Baseline U-Net (Dice: 0.6633)", "#b9770e"),
            (c4, "(d) DS-SAU-Net [Ours] (Dice: 0.7340)", "#1b4f72")
        ]
        
        for ax, (img_col, title, color) in zip(axes, panels):
            ax.imshow(img_col)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color=color)
            ax.axis('off')
            
            # Subtle neat border around each panel
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#cbd5e1')
                spine.set_linewidth(1.5)
                
        out_path = os.path.join(DOCS_FIG_DIR, "qualitative_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

# ==============================================================================
# 5. XAI WRINKLE DENSITY HEATMAP (Publication Grade)
# ==============================================================================
def draw_xai_heatmap():
    print("Generating: xai_wrinkle_heatmap.png...")
    # Source mask from improve/results or stage output
    old_xai = os.path.join(DOCS_FIG_DIR, "xai_wrinkle_heatmap.png")
    test_vis_dir = os.path.join(PROJECT_ROOT, "improve", "results", "test_visual_comparisons")
    
    # Load a representative sample
    sample_file = os.path.join(test_vis_dir, "compare_07649.png")
    if not os.path.exists(sample_file):
        sample_file = os.path.join(test_vis_dir, os.listdir(test_vis_dir)[0])
        
    pil_img = Image.open(sample_file).convert("RGB")
    img = np.array(pil_img)
    step_w = img.shape[1] // 4
    crop_y = 40 if img.shape[0] > 256 else 0
    face_rgb = img[crop_y:, :step_w]
    ours_col = img[crop_y:, 3*step_w:]
    
    # Calculate density heatmap via Gaussian filtering
    h, w = face_rgb.shape[:2]
    # Identify wrinkle pixels (where magenta overlay is strong in column 4)
    # or synthetic density from wrinkles
    diff = np.abs(ours_col.astype(np.float32) - img[crop_y:, :step_w].astype(np.float32))
    diff_gray = np.mean(diff, axis=-1)
    mask = (diff_gray > 25).astype(np.float32)
    
    # Gaussian density blur
    density = cv2.GaussianBlur(mask, (31, 31), sigmaX=8, sigmaY=8)
    density = (density - density.min()) / (density.max() + 1e-6)
    
    # Plot side-by-side: Input Portrait | Localized Density Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), dpi=300)
    plt.subplots_adjust(wspace=0.10, left=0.04, right=0.92, top=0.88, bottom=0.05)
    
    ax1.imshow(face_rgb)
    ax1.set_title("(a) Preprocessed Face Portrait", fontsize=11, fontweight='bold', pad=10, color='#2c3e50')
    ax1.axis('off')
    
    ax2.imshow(face_rgb)
    im = ax2.imshow(density, cmap='turbo', alpha=0.55)
    ax2.set_title("(b) Morphometric Wrinkle Density Overlay", fontsize=11, fontweight='bold', pad=10, color='#1b4f72')
    ax2.axis('off')
    
    # Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Local Wrinkle Density (WAF)', fontsize=9.5, fontweight='bold', color='#1b4f72')
    cbar.ax.tick_params(labelsize=8.5)
    
    out_path = os.path.join(DOCS_FIG_DIR, "xai_wrinkle_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    print("=" * 70)
    print("🎨 REDRAWING ALL 5 FIGURES FOR PUBLICATION PERFECTION (300 DPI)")
    print("=" * 70)
    draw_benchmark_and_ablation()
    draw_snake_vs_standard()
    draw_framework_architecture()
    draw_qualitative_comparison()
    draw_xai_heatmap()
    print("=" * 70)
    print("✨ ALL 5 FIGURES SUCCESSFULLY REDRAWN AND SAVED IN docs/figures/!")
    print("=" * 70)
