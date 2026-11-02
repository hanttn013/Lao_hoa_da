import os
import sys
import shutil

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Đặt cache Kaggle sang ổ D:\ để không chiếm dung lượng ổ C:\
os.environ["KAGGLEHUB_CACHE"] = os.path.join(BASE_DIR, "data", ".kaggle_cache")

import kagglehub

def download_and_link_to_data():
    target_dir = os.path.join(BASE_DIR, "data", "ffhq_raw_images", "kaggle_ffhq")
    os.makedirs(target_dir, exist_ok=True)
    
    print(">>> Bắt đầu tải dataset 'arnaud58/flickrfaceshq-dataset-ffhq' (~19.5 GB)...")
    path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
    print(f">>> Tải hoàn tất tại bộ nhớ đệm: {path}")
    
    print(f">>> Đang đồng bộ dữ liệu vào thư mục đích: {target_dir}...")
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                print(f"Sao chép thư mục {item}...")
                shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
            
    print(f">>> ĐÃ HOÀN TẤT! Dữ liệu đã sẵn sàng tại: {target_dir}")

if __name__ == "__main__":
    download_and_link_to_data()