import cv2
import numpy as np

def compute_clinical_aging_score(morphometry_dict):
    """
    Stage 4: Clinically-Aligned Scoring
    Chuyển đổi đặc trưng hình thái học sang thang điểm lâm sàng da liễu (Lemperle/Merz Scale 0 - 4):
    - Cấp 0: Hoàn toàn không nhăn (Da mịn màng)
    - Cấp 1: Lão hóa độ 1 (Nếp nhăn li ti / nông bắt đầu xuất hiện)
    - Cấp 2: Lão hóa độ 2 (Nếp nhăn nông rõ rệt)
    - Cấp 3: Lão hóa độ 3 (Nếp nhăn sâu trung bình)
    - Cấp 4: Lão hóa độ 4 (Nếp nhăn sâu và dày đặc)
    """
    waf = morphometry_dict["area_fraction"]
    density = morphometry_dict["wrinkle_density"]
    
    # Chỉ số tích hợp giữa tỷ lệ diện tích và mật độ rãnh nhăn
    combined_index = waf * 0.5 + density * 0.5
    
    if combined_index < 0.005:
        score = 0
        desc = "Lão hóa độ 0 (Da phẳng mịn, không phát hiện rãnh nhăn rõ)"
    elif combined_index < 0.02:
        score = 1
        desc = "Lão hóa độ 1 (Nếp nhăn li ti / thoáng qua bắt đầu xuất hiện)"
    elif combined_index < 0.05:
        score = 2
        desc = "Lão hóa độ 2 (Nếp nhăn nông nhìn thấy rõ trên bề mặt)"
    elif combined_index < 0.10:
        score = 3
        desc = "Lão hóa độ 3 (Nếp nhăn sâu trung bình, rãnh nhăn hằn rõ)"
    else:
        score = 4
        desc = "Lão hóa độ 4 (Nếp nhăn rất sâu, mạng lưới rãnh nhăn dày đặc)"
        
    return score, desc

def generate_xai_heatmap(image_bgr, wrinkle_mask):
    """
    XAI (Explainable AI):
    Trực quan hóa bản đồ nhiệt (Overlay Heatmap) giải thích cho bác sĩ/người dùng
    chính xác vị trí những đường rãnh nhăn dẫn tới điểm số lão hóa đó.
    """
    overlay = image_bgr.copy()
    
    # Tạo màu đỏ nổi bật cho các đường nhăn phát hiện
    colored_wrinkles = np.zeros_like(image_bgr)
    colored_wrinkles[wrinkle_mask > 0] = [0, 0, 255] # Màu đỏ (BGR)
    
    # Hòa trộn bản đồ nhiệt lên ảnh mặt gốc
    alpha = 0.6
    cv2.addWeighted(colored_wrinkles, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay
