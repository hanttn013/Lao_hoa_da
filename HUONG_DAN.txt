# HƯỚNG DẪN CHI TIẾT VẬN HÀNH DỰ ÁN NẾP NHĂN & ĐÁNH GIÁ LÃO HÓA DA
*(Facial Wrinkle Segmentation & Clinical Aging Assessment)*

Tài liệu này hướng dẫn bạn cách thiết lập, tải dataset, đổi đường dẫn thư mục, chạy huấn luyện (train trên máy GPU), chạy benchmark so sánh và tự động vẽ biểu đồ trực quan.

---

## 1. KHỞI TẠO VÀ KÍCH HOẠT MÔI TRƯỜNG ẢO (.VENV)

Khi clone code về máy mới (Windows / Linux), bạn thực hiện các bước sau để tạo và kích hoạt môi trường:

### Trên Windows (Command Prompt hoặc PowerShell):
```powershell
# 1. Di chuyển vào thư mục dự án
cd ttcn

# 2. Tạo môi trường ảo tên .venv
python -m venv .venv

# 3. Kích hoạt môi trường ảo:
# Nếu dùng PowerShell:
.\.venv\Scripts\Activate.ps1
# (Nếu PowerShell báo lỗi Script Execution Policy, chạy lệnh: Set-ExecutionPolicy Unrestricted -Scope Process rồi kích hoạt lại)
# Nếu dùng CMD:
.\.venv\Scripts\activate.bat

# 4. Nâng cấp pip và cài đặt các thư viện cần thiết:
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python scikit-image scipy pillow tqdm matplotlib gdown kagglehub monai einops yacs
```

### Trên Linux / Ubuntu / Google Colab:
```bash
# 1. Tạo môi trường ảo
python3 -m venv .venv

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate

# 3. Cài đặt các thư viện
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python scikit-image scipy pillow tqdm matplotlib gdown kagglehub monai einops yacs
```

---

## 2. CẤU TRÚC DỰ ÁN CHUẨN HÓA

```
ttcn/
├── data/
│   ├── ffhq_wrinkle_base/             # Dữ liệu nhãn nếp nhăn gốc (Ground Truth)
│   │   ├── manual_wrinkle_masks/      # 1,000 ảnh nhãn nếp nhăn chuyên gia dán
│   │   ├── face_parsed_labels/        # 1,000 file phân đoạn khuôn mặt (.npy)
│   │   ├── face_images/               # Ảnh mặt gốc khớp với nhãn thủ công
│   │   └── masked_face_images/        # 793 ảnh mặt đã lọc bỏ tóc, nền (dùng để train)
│   └── ffhq_raw_images/
│       └── kaggle_ffhq/               # Kho 52,001 ảnh chân dung FFHQ gốc
│
├── ffhq-wrinkle-dataset/              # MÔ HÌNH BASELINE GỐC (ICPR 2024)
│   ├── pretrained_ckpt/               # Checkpoint gốc tác giả (SwinUNETR & UNet)
│   ├── inference.py                   # Chạy suy luận mô hình gốc
│   └── compare_official.py            # So khớp kết quả gốc với Ground Truth
│
└── improve/                           # TOÀN BỘ MODULE CẢI TIẾN
    ├── stage1_roi_alignment.py        # [Stage 1] MediaPipe Face Mesh & Dual-Stream
    ├── stage2_attention_segmentation.py # [Stage 2] Cross-Attention & Boundary Loss
    ├── stage3_morphometry.py          # [Stage 3] Skeletonization đo chiều dài nếp nhăn
    ├── stage4_clinical_scoring_xai.py # [Stage 4] Chấm điểm Lemperle 0-4 & Heatmap XAI
    ├── train.py                       # [Train] Script huấn luyện mô hình cải tiến
    ├── benchmark_and_ablation.py      # [Benchmark] Đánh giá đối đầu & Vẽ biểu đồ
    ├── pipeline_aging_assessment.py   # [End-to-End] Chạy từ 1 ảnh mặt bất kỳ -> điểm lão hóa
    └── results/                       # Thư mục chứa biểu đồ và kết quả xuất ra
```

---

## 2. HƯỚNG DẪN CHẠY DOWNLOAD DATASET

Nếu sang máy mới chưa có dữ liệu, bạn chỉ cần chạy 2 lệnh sau:

### Bước 2.1: Tải bộ nhãn nếp nhăn Ground Truth & Phân vùng mặt
Chạy file `download_wrinkle_data.py`:
```bash
python download_wrinkle_data.py
```
*Tác vụ:* Tự động tải từ Google Drive về `data/ffhq_wrinkle_base/` gồm 1,000 masks nếp nhăn và 1,000 nhãn `face_parsed_labels`.

### Bước 2.2: Tải kho ảnh khuôn mặt FFHQ (Kaggle)
Chạy file `download_data.py`:
```bash
python download_data.py
```
*Tác vụ:* Tải bộ 52,000 ảnh chân dung từ Kaggle (`arnaud58/flickrfaceshq-dataset-ffhq`) vào `data/ffhq_raw_images/kaggle_ffhq`.

---

## 3. CÁCH ĐỔI ĐƯỜNG DẪN THƯ MỤC Ở CÁC VỊ TRÍ QUAN TRỌNG

Khi mang dự án sang máy khác (ví dụ từ `E:\ttcn` sang `D:\project` hoặc trên Linux `/home/user/ttcn`), bạn chỉ cần cập nhật đường dẫn ở các file sau:

### Vị trí 1: File `improve/train.py` (Dòng 74 - 75)
```python
# Đường dẫn dữ liệu đưa vào huấn luyện:
img_dir = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images"
mask_dir = r"e:\ttcn\data\ffhq_wrinkle_base\manual_wrinkle_masks"
```
👉 *Chỉ cần sửa `e:\ttcn` thành đường dẫn thư mục mới của bạn.*

### Vị trí 2: File `improve/pipeline_aging_assessment.py` (Dòng 61 - 62)
```python
test_face = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images\00001.png"
test_mask = r"e:\ttcn\ffhq-wrinkle-dataset\test_outputs_official\00001_mask.png"
```
👉 *Đổi sang ảnh chân dung bạn muốn kiểm tra thực tế.*

### Vị trí 3: File `ffhq-wrinkle-dataset/compare_official.py` (Dòng 14 - 16)
```python
official_mask_path = r"e:\ttcn\ffhq-wrinkle-dataset\test_outputs_official\00001_mask.png"
face_path = r"e:\ttcn\data\ffhq_wrinkle_base\masked_face_images\00001.png"
gt_path = r"e:\ttcn\data\ffhq_wrinkle_base\manual_wrinkle_masks\00001.png"
```

---

## 4. HƯỚNG DẪN CHẠY HUẤN LUYỆN (TRAIN MODEL TRÊN GPU)

Mở Terminal / PowerShell và chuyển vào thư mục `improve`:
```bash
cd improve
python train.py
```

### Các tùy chọn cấu hình trong `train.py`:
- **`device`**: File đã cấu hình tự động nhận diện `cuda` nếu máy có GPU NVIDIA.
- **`epochs`**: Mặc định đặt `epochs=30` hoặc `epochs=50` để đạt độ hội tụ cao.
- **`img_size`**: 
  - Đặt `256` hoặc `512` để train nhanh và tiết kiệm VRAM.
  - Đặt `1024` nếu GPU của bạn có từ 16GB–24GB VRAM.
- **`use_boundary_loss`**: Đặt `True` để kích hoạt hàm mất mát viền biên cải tiến.
- Checkpoint sau khi train xong sẽ tự động được lưu vào: `improve/checkpoints/improved_attention_unet.pth`.

---

## 5. HƯỚNG DẪN CHẠY BENCHMARK & TỰ ĐỘNG VẼ BIỂU ĐỒ

Để so sánh toàn bộ các Baseline (U-Net, DeepLabv3+, SegFormer, SwinUNETR, Model cải tiến) và thực hiện các bài toán bóc tách thành phần (Ablation Studies):

Chạy lệnh:
```bash
cd improve
python benchmark_and_ablation.py
```

### Kết quả xuất ra màn hình Console:
1. **Bảng 1 (Benchmark)**: So sánh các chỉ số `mIoU`, `Dice Coefficient`, `HD95 (Hausdorff biên)`, `MAE (Sai số lâm sàng)`, `Cohen's Kappa`.
2. **Bảng 2 (Ablation Studies)**:
   - *Ablation 1*: Đóng góp của Wavelet / Frequency Stream so với chỉ dùng RGB.
   - *Ablation 2*: Đóng góp của Boundary Loss so với CE / Dice Loss thuần túy.
   - *Ablation 3*: Tác động của việc căn chỉnh Landmark Face Mesh so với Crop thô.

### Kết quả Biểu đồ hình ảnh:
Script sẽ tự động vẽ và lưu biểu đồ độ phân giải cao tại:
👉 **`improve/results/benchmark_and_ablation_charts.png`**

Biểu đồ bao gồm 4 đồ thị con:
- **Biểu đồ cột kép**: So sánh Dice Score & mIoU giữa 6 mô hình.
- **Biểu đồ sai số HD95**: Thể hiện độ sắc nét của đường biên nếp nhăn.
- **Đồ thị tăng trưởng**: Hiệu quả của Boundary Loss.
- **Đồ thị độ tương đồng lâm sàng**: Đánh giá MAE & Cohen's Kappa đối với điểm số lão hóa da.

---

## 6. HƯỚNG DẪN CHẠY ĐÁNH GIÁ LÃO HÓA TỪ 1 ẢNH BẤT KỲ (END-TO-END)

Để chạy trọn vẹn quy trình từ 1 ảnh chụp khuôn mặt $\rightarrow$ Crop 4 vùng ROI $\rightarrow$ Đo đạc chiều dài nếp nhăn $\rightarrow$ Chấm điểm lão hóa cấp 0–4:

```bash
cd improve
python pipeline_aging_assessment.py
```
*Kết quả:* Sẽ in ra chẩn đoán lâm sàng và xuất ảnh giải thích trực quan (Overlay Heatmap màu đỏ tại `improve/results/xai_wrinkle_heatmap.png`).
