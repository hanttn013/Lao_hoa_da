import os
import zipfile
import gdown

DATA_DIR = r"e:\ttcn\data"
os.makedirs(DATA_DIR, exist_ok=True)

# Google Drive IDs for wrinkle masks
# manual_wrinkle_masks: ~2.8MB (1000 ground truth masks)
# weak_wrinkle_masks: ~6.94GB (50000 texture/weak masks)
# face_parsed_labels: ~5.1MB (1000 facial parsing segmentations)
FILES = {
    "manual_wrinkle_masks.zip": "1vzbx847hq1hgKjMtMmxLzmzLrYb4t0QX",
    "face_parsed_labels.zip": "1Oje4gsfgsScBfLMwP-OobgoFK-DHOGqc",
    # "weak_wrinkle_masks.zip": "1dLl7_ChsJNrZSQIbNGj21sm45hmH0ZxC" # (6.94GB)
}

def download_and_extract():
    for filename, file_id in FILES.items():
        zip_path = os.path.join(DATA_DIR, filename)
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, zip_path, quiet=False)
        
        if os.path.exists(zip_path):
            print(f"Extracting {filename} to {DATA_DIR}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
            print(f"Done with {filename}!")

if __name__ == "__main__":
    download_and_extract()
