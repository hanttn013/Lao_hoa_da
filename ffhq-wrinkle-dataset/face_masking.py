import os
import sys
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm

def process_npy_and_png(npy_directory, png_directory, output_directory):
    png_files = [f for f in os.listdir(png_directory) if f.endswith('.png')]
    
    for png_file in tqdm(png_files, desc="Processing files"):
        png_path = os.path.join(png_directory, png_file)
        npy_file = png_file.replace('.png', '.npy')
        npy_path = os.path.join(npy_directory, npy_file)

        if os.path.exists(npy_path):
            npy_data = np.load(npy_path)
            image = Image.open(png_path)
            image_array = np.array(image)
            
            # Tự động điều chỉnh kích thước nhãn phân đoạn theo kích thước ảnh
            scale_y = image_array.shape[0] / npy_data.shape[0]
            scale_x = image_array.shape[1] / npy_data.shape[1]
            if scale_y != 1.0 or scale_x != 1.0:
                npy_resized = zoom(npy_data, (scale_y, scale_x), order=0)
            else:
                npy_resized = npy_data
                
            mask = (npy_resized == 1) | (npy_resized == 10)
            result_array = image_array.copy()
            result_array[~mask] = 0  
            result_image = Image.fromarray(result_array)
            output_path = os.path.join(output_directory, png_file)
            result_image.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python face_masking.py <face_parsed_npy_folder (ex. face_parsed_labels)> <face_images_folder (ex. face_images)> <output_masked_face_images_folder (ex. masked_face_images)>")
        sys.exit(1)
    npy_directory = sys.argv[1]
    png_directory = sys.argv[2]
    output_directory = sys.argv[3]
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    process_npy_and_png(npy_directory, png_directory, output_directory)
