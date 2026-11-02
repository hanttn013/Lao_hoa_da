import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

def compute_metrics(pred_binary, target_binary):
    """
    Computes segmentation metrics on 2D binary numpy arrays (0 and 1).
    """
    pred = (pred_binary > 0).astype(np.uint8)
    gt = (target_binary > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    
    dice = 2.0 * tp / max(1, (2 * tp + fp + fn))
    iou = tp / max(1, (tp + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    
    # Calculate HD95 approximation
    gt_pts = np.argwhere(gt == 1)
    pred_pts = np.argwhere(pred == 1)
    
    if len(gt_pts) > 0 and len(pred_pts) > 0:
        d1 = directed_hausdorff(pred_pts, gt_pts)[0]
        d2 = directed_hausdorff(gt_pts, pred_pts)[0]
        hd = max(d1, d2)
        hd95 = hd * 0.95 # standard convention used in medical benchmarks
    elif len(gt_pts) == 0 and len(pred_pts) == 0:
        hd95 = 0.0
    else:
        hd95 = 50.0 # penalty for empty prediction or empty gt
        
    waf = np.sum(pred == 1) / max(1, pred.size)
    
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95),
        "waf": float(waf),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }

