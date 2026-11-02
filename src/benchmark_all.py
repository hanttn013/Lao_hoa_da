import os
import sys
import io
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
import time
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.dataset import get_dataloaders
from models.Baseline.model import BaselineUNet
from models.AttentionUNet.model import AttentionUNet
from models.DSnake.model import DSnakeUNet
from models.metrics import compute_metrics

def run_benchmark():
    print("=== [BENCHMARK & ABLATION STUDY] Running Comprehensive Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    _, _, _, full_loader = get_dataloaders(SRC_DIR, batch_size=1)
    if full_loader is None:
        print("Dataset not found. Please ensure data/ exists.")
        return
        
    print(f"Loaded {len(full_loader)} benchmark test image pairs.")
    
    models_dict = {
        "Baseline U-Net": {
            "model": BaselineUNet().to(device),
            "ckpt": os.path.join(SRC_DIR, "models", "Baseline", "checkpoints", "baseline_unet.pth"),
            "color": "#64748b" # slate gray
        },
        "Attention U-Net": {
            "model": AttentionUNet().to(device),
            "ckpt": os.path.join(SRC_DIR, "models", "AttentionUNet", "checkpoints", "improved_attention_unet.pth"),
            "color": "#3b82f6" # royal blue
        },
        "DSnake (Our Model)": {
            "model": DSnakeUNet().to(device),
            "ckpt": os.path.join(SRC_DIR, "models", "DSnake", "checkpoints", "dsnake_unet.pth"),
            "color": "#10b981" # emerald green
        }
    }
    
    # Load checkpoints
    for name, item in models_dict.items():
        if os.path.exists(item["ckpt"]):
            sd = torch.load(item["ckpt"], map_location=device, weights_only=True)
            item["model"].load_state_dict(sd, strict=False)
            print(f"Loaded weights for {name} from {item['ckpt']}")
        else:
            print(f"Warning: Checkpoint not found for {name}: {item['ckpt']}")
        item["model"].eval()
        item["param_count"] = sum(p.numel() for p in item["model"].parameters())
        
    benchmark_results = {}
    detailed_per_image = {}
    
    for name, item in models_dict.items():
        print(f"\nEvaluating {name}...")
        model = item["model"]
        dices, ious, precs, recs, hd95s, wafs = [], [], [], [], [], []
        latencies = []
        
        with torch.no_grad():
            for inputs, targets, filenames in full_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                filename = filenames[0]
                
                t0 = time.time()
                logits = model(inputs)
                latency = (time.time() - t0) * 1000.0 # ms
                latencies.append(latency)
                
                pred_binary = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                target_binary = targets.squeeze(0).cpu().numpy()
                
                m = compute_metrics(pred_binary, target_binary)
                dices.append(m["dice"])
                ious.append(m["iou"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                hd95s.append(m["hd95"])
                wafs.append(m["waf"])
                
                if filename not in detailed_per_image:
                    detailed_per_image[filename] = {}
                detailed_per_image[filename][name] = {
                    "dice": m["dice"],
                    "iou": m["iou"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "hd95": m["hd95"]
                }
                
        benchmark_results[name] = {
            "parameters": item["param_count"],
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "hd95_mean": float(np.mean(hd95s)),
            "latency_ms": float(np.mean(latencies[2:]) if len(latencies) > 2 else np.mean(latencies)),
            "color": item["color"]
        }
        print(f"-> {name}: Dice={benchmark_results[name]['dice_mean']:.4f} | IoU={benchmark_results[name]['iou_mean']:.4f} | HD95={benchmark_results[name]['hd95_mean']:.2f}px | Latency={benchmark_results[name]['latency_ms']:.1f}ms")

    results_dir = os.path.join(SRC_DIR, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(results_dir, "benchmark_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": benchmark_results,
            "per_image": detailed_per_image
        }, f, indent=4)
    print(f"\nSaved benchmark JSON: {json_path}")
    
    # Render High-Resolution Chart
    chart_path = os.path.join(results_dir, "benchmark_comparison_chart.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    plt.subplots_adjust(wspace=0.3)
    
    names = list(benchmark_results.keys())
    colors = [benchmark_results[n]["color"] for n in names]
    
    # Subplot 1: Dice & IoU
    dices = [benchmark_results[n]["dice_mean"] for n in names]
    ious = [benchmark_results[n]["iou_mean"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width/2, dices, width, label='Dice Score', color='#3b82f6', alpha=0.9, edgecolor='black')
    axes[0].bar(x + width/2, ious, width, label='IoU Score', color='#10b981', alpha=0.9, edgecolor='black')
    axes[0].set_title('Segmentation Accuracy (Dice & IoU)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10, rotation=15)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left')
    for i in range(len(names)):
        axes[0].text(x[i] - width/2, dices[i] + 0.02, f"{dices[i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        axes[0].text(x[i] + width/2, ious[i] + 0.02, f"{ious[i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        
    # Subplot 2: HD95 Error Distance
    hd95s = [benchmark_results[n]["hd95_mean"] for n in names]
    axes[1].bar(x, hd95s, 0.45, color=colors, alpha=0.9, edgecolor='black')
    axes[1].set_title('Boundary Distance HD95 (Pixels, Lower is better)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=10, rotation=15)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i in range(len(names)):
        axes[1].text(x[i], hd95s[i] + 0.5, f"{hd95s[i]:.2f} px", ha='center', fontsize=10, fontweight='bold')
        
    # Subplot 3: Precision & Recall Trade-off
    precs = [benchmark_results[n]["precision_mean"] for n in names]
    recs = [benchmark_results[n]["recall_mean"] for n in names]
    axes[2].bar(x - width/2, precs, width, label='Precision', color='#f59e0b', alpha=0.9, edgecolor='black')
    axes[2].bar(x + width/2, recs, width, label='Recall', color='#8b5cf6', alpha=0.9, edgecolor='black')
    axes[2].set_title('Precision vs Recall Trade-off', fontsize=13, fontweight='bold', pad=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, fontsize=10, rotation=15)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper left')
    for i in range(len(names)):
        axes[2].text(x[i] - width/2, precs[i] + 0.02, f"{precs[i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        axes[2].text(x[i] + width/2, recs[i] + 0.02, f"{recs[i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved benchmark Chart: {chart_path}")
    print("=== Benchmark Finished Successfully! ===")

if __name__ == "__main__":
    run_benchmark()
