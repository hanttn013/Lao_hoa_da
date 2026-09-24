import os
import sys
import cv2
import numpy as np

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import các stage cải tiến từ thư mục improve
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stage1_roi_alignment import align_and_crop_face_rois, extract_dual_streams
from stage3_morphometry import extract_morphometry
from stage4_clinical_scoring_xai import compute_clinical_aging_score, generate_xai_heatmap

def run_aging_evaluation(image_path, wrinkle_mask_path, output_dir="e:/ttcn/improve/results"):
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    mask = cv2.imread(wrinkle_mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print("Lỗi: Không tìm thấy file ảnh hoặc file mask kiểm thử.")
        return

    print("================================================================================")
    print("        HỆ THỐNG ĐÁNH GIÁ NẾP NHĂN & MỨC ĐỘ LÃO HÓA DA (END-TO-END)            ")
    print("================================================================================")
    
    # 1. Stage 1: Kiểm tra căn chỉnh ROI
    rois, status = align_and_crop_face_rois(image)
    if rois:
        print(f"[Stage 1] Landmark Face Mesh: Đã phân tích thành công 4 vùng giải phẫu:")
        for name in rois.keys():
            print(f"          + ROI: {name.upper()} (Kích thước: {rois[name].shape[1]}x{rois[name].shape[0]})")
    
    # 2. Stage 3: Phân tích hình thái học
    face_area = np.sum(np.any(image > 0, axis=-1))
    metrics = extract_morphometry(mask, face_area)
    
    print("\n[Stage 3] Đo lường hình thái học (Quantitative Morphometry):")
    print(f"          + Diện tích da mặt khảo sát: {face_area:,} pixels")
    print(f"          + Số pixel nếp nhăn bóc tách: {metrics['wrinkle_pixels']:,} pixels")
    print(f"          + Chiều dài rãnh nhăn (Skeletonized): {metrics['wrinkle_length']:,} pixels")
    print(f"          + Tỷ lệ nếp nhăn (WAF): {metrics['area_fraction']*100:.3f}%")
    print(f"          + Mật độ nếp nhăn: {metrics['wrinkle_density']:.5f}")
    
    # 3. Stage 4: Chẩn đoán lâm sàng & XAI
    score, desc = compute_clinical_aging_score(metrics)
    print("\n[Stage 4] Đánh giá cấp độ lão hóa lâm sàng (Lemperle/Merz Scale 0-4):")
    print(f"          >>> KẾT QUẢ: CẤP {score}")
    print(f"          >>> CHẨN ĐOÁN: {desc}")
    
    # Xuất bản đồ giải thích XAI
    heatmap = generate_xai_heatmap(image, mask)
    heatmap_out = os.path.join(output_dir, "xai_wrinkle_heatmap.png")
    cv2.imwrite(heatmap_out, heatmap)
    print(f"\n[XAI] Đã lưu bản đồ giải thích trực quan tại: {heatmap_out}")
    print("================================================================================")

if __name__ == "__main__":
    test_face = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images\00001.png"
    test_mask = r"e:\ttcn\ffhq-wrinkle-dataset\test_outputs_official\00001_mask.png"
    run_aging_evaluation(test_face, test_mask)
