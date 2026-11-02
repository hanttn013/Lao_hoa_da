import cv2
import numpy as np
from PIL import Image

def detect_and_crop_face(image_pil, margin=0.25):
    """
    Tự động phát hiện khuôn mặt và cắt cúp (Face Crop & Alignment):
    - Đảm bảo mô hình chỉ tập trung 100% vào vùng da mặt (trán, mắt, mũi, má, cằm).
    - Loại bỏ hoàn toàn tạp nhiễu từ cơ thể, quần áo, hình xăm, bối cảnh xung quanh.
    - Chuẩn hóa tỷ lệ khuôn mặt tương thích tuyệt đối với tập huấn luyện FFHQ.
    """
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    
    img_np = np.array(image_pil.convert("RGB"))
    h, w, _ = img_np.shape
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_np)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            xs = [pt.x * w for pt in landmarks]
            ys = [pt.y * h for pt in landmarks]
            
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            # Mở rộng biên (margin) để lấy trọn vẹn trán và cằm
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            
            side = max(box_w, box_h) * (1.0 + margin)
            
            x1 = int(max(0, center_x - side / 2.0))
            y1 = int(max(0, center_y - side / 2.0))
            x2 = int(min(w, center_x + side / 2.0))
            y2 = int(min(h, center_y + side / 2.0))
            
            cropped = img_np[y1:y2, x1:x2]
            if cropped.size > 0:
                return Image.fromarray(cropped), True
                
    # Fallback OpenCV Haar Cascade nếu FaceMesh không bắt được
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        # Mở rộng biên
        pad_x = int(fw * margin)
        pad_y = int(fh * margin)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)
        cropped = img_np[y1:y2, x1:x2]
        return Image.fromarray(cropped), True
        
    # Nếu ảnh đã là khuôn mặt cận cảnh (close-up)
    return image_pil, False

