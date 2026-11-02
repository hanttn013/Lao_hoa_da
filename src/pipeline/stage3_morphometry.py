import numpy as np
from skimage.morphology import skeletonize

def extract_morphometry(binary_mask, roi_area):
    """
    Stage 3: Topological Thinning & Quantitative Morphometry Extraction
    1. Skeletonization (Topological thinning):
       Rút gọn nếp nhăn về đường trung tâm dày chính xác 1-pixel.
    2. Chỉ số định lượng:
       - Wrinkle Area Fraction (WAF): Tỷ lệ % diện tích nếp nhăn / Diện tích vùng ROI.
       - Total Wrinkle Length: Tổng độ dài đường nếp nhăn (pixels / mm).
       - Wrinkle Density: Mật độ chiều dài nếp nhăn trên diện tích vùng da.
    """
    binary = binary_mask > 0
    wrinkle_pixels = np.sum(binary)
    
    # Rút gọn topological về xương sống nếp nhăn (Skeleton)
    skeleton = skeletonize(binary)
    wrinkle_length = np.sum(skeleton)
    
    area_fraction = wrinkle_pixels / max(1, roi_area)
    wrinkle_density = wrinkle_length / max(1, roi_area)
    
    return {
        "wrinkle_pixels": int(wrinkle_pixels),
        "wrinkle_length": int(wrinkle_length),
        "area_fraction": float(area_fraction),
        "wrinkle_density": float(wrinkle_density)
    }
