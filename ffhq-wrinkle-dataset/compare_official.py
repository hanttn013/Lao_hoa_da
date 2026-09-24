import sys
import os
import cv2
import numpy as np

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(r"e:\ttcn\improve")
from morphometry import compute_wrinkle_metrics, map_to_clinical_score

def compare():
    official_mask_path = r"e:\ttcn\ffhq-wrinkle-dataset\test_outputs_official\00001_mask.png"
    face_path = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images\00001.png"
    gt_path = r"e:\ttcn\data\ffhq_wrinkle_base\manual_wrinkle_masks\00001.png"
    
    mask = cv2.imread(official_mask_path, cv2.IMREAD_GRAYSCALE)
    face = cv2.imread(face_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    
    face_area = np.sum(np.any(face > 0, axis=-1))
    metrics = compute_wrinkle_metrics(mask, face_area)
    score, desc = map_to_clinical_score(metrics)
    
    print("================ KẾT QUẢ VỚI TEXTURE MAP CHÍNH THỨC CỦA TÁC GIẢ ================")
    print(f"Ảnh kiểm thử: 00001.png (1024x1024)")
    print(f"Số pixel nếp nhăn phát hiện: {metrics['wrinkle_pixels']:,} pixels")
    print(f"Chiều dài rãnh nhăn (Skeletonized): {metrics['wrinkle_length']:,} pixels")
    print(f"Tỷ lệ nếp nhăn trên khuôn mặt (WAF): {metrics['area_fraction']*100:.3f}%")
    print(f"Mật độ nếp nhăn: {metrics['wrinkle_density']:.5f}")
    print(f"Chấm điểm lâm sàng (0-4): CẤP {score} -> {desc}")
    
    intersection = np.logical_and(mask > 0, gt > 0)
    union = np.logical_or(mask > 0, gt > 0)
    iou = np.sum(intersection) / max(1, np.sum(union))
    dice = 2 * np.sum(intersection) / max(1, (np.sum(mask > 0) + np.sum(gt > 0)))
    
    print("--------------------------------------------------------------------------------")
    print(f"So khớp với Ground Truth (Chuyên gia dán nhãn - 12,557 pixels):")
    print(f"- Dice Coefficient: {dice:.4f}")
    print(f"- IoU: {iou:.4f}")
    print(f"- Độ nhạy / Recall: {np.sum(intersection)/np.sum(gt > 0):.4f}")
    print(f"- Độ chính xác / Precision: {np.sum(intersection)/max(1, np.sum(mask > 0)):.4f}")
    print("================================================================================")

if __name__ == "__main__":
    compare()
