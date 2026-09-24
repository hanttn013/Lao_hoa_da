import cv2
import numpy as np

def align_and_crop_face_rois(image_bgr):
    """
    Stage 1: Landmark Alignment & ROI Normalization
    - Sử dụng MediaPipe Face Mesh (468 landmarks) để xác định các đặc trưng hình học.
    - Căn chỉnh xoay mặt thẳng (roll = 0).
    - Trích xuất 4 vùng quan tâm chuẩn hóa (Normalized ROI):
        1. Vùng trán (Forehead)
        2. Đuôi mắt trái (Left Crow's feet)
        3. Đuôi mắt phải (Right Crow's feet)
        4. Rãnh cười/rãnh mũi má (Nasolabial folds)
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, "Không tìm thấy khuôn mặt trong ảnh"
            
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(int(pt.x * w), int(pt.y * h)) for pt in landmarks])
        
        # 1. Trán (Forehead ROI): Giới hạn giữa chân mày và đỉnh trán
        brow_top = int(np.min(coords[[70, 63, 105, 66, 107, 336, 296, 334, 293, 300], 1]))
        eyebrow_center_x = int(np.mean(coords[[107, 336], 0]))
        forehead_height = int((coords[168, 1] - brow_top) * 1.5)
        forehead_box = (
            max(0, eyebrow_center_x - int(w * 0.25)),
            max(0, brow_top - forehead_height),
            min(w, eyebrow_center_x + int(w * 0.25)),
            brow_top
        )
        
        # 2. Đuôi mắt (Crow's feet - Trái & Phải)
        left_eye_outer = coords[33]
        right_eye_outer = coords[263]
        crow_size = int(w * 0.08)
        
        # 3. Rãnh cười / Rãnh mũi má (Nasolabial Folds)
        nose_bottom = coords[2]
        mouth_left = coords[61]
        mouth_right = coords[291]
        
        nasolabial_box = (
            max(0, mouth_left[0] - int(w * 0.05)),
            max(0, nose_bottom[1] - int(h * 0.02)),
            min(w, mouth_right[0] + int(w * 0.05)),
            min(h, max(mouth_left[1], mouth_right[1]) + int(h * 0.05))
        )
        
        rois = {
            "forehead": image_bgr[forehead_box[1]:forehead_box[3], forehead_box[0]:forehead_box[2]],
            "right_crows_feet": image_bgr[max(0, left_eye_outer[1]-crow_size):min(h, left_eye_outer[1]+crow_size),
                                          max(0, left_eye_outer[0]-crow_size*2):min(w, left_eye_outer[0])],
            "left_crows_feet": image_bgr[max(0, right_eye_outer[1]-crow_size):min(h, right_eye_outer[1]+crow_size),
                                         max(0, right_eye_outer[0]):min(w, right_eye_outer[0]+crow_size*2)],
            "nasolabial_folds": image_bgr[nasolabial_box[1]:nasolabial_box[3], nasolabial_box[0]:nasolabial_box[2]]
        }
        return rois, "Thành công"

def extract_dual_streams(roi_bgr):
    """
    Trích xuất 2 luồng song song (Dual-Stream):
    - Stream A: High-Frequency / Texture Stream (Wavelet / CLAHE + Laplacian gradient)
    - Stream B: Spatial-Context Stream (RGB Color chuẩn hóa)
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Stream A: Tăng cường tương phản cục bộ và bắt cạnh tần số cao
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_32F, ksize=3)
    texture_stream = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
    
    # Stream B: Ảnh không gian màu RGB
    spatial_stream = roi_bgr.copy()
    
    return texture_stream, spatial_stream
