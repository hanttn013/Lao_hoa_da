# HỆ THỐNG PHÂN ĐOẠN NẾP NHĂN & ĐÁNH GIÁ MỨC ĐỘ LÃO HÓA DA MẶT
*(Facial Wrinkle Segmentation & Clinical Aging Assessment Platform)*

Dự án nghiên cứu và phát triển hệ thống thị giác máy tính và học sâu (Deep Learning) phục vụ chẩn đoán da liễu thẩm mỹ: tự động phát hiện, phân đoạn chi tiết các rãnh nếp nhăn mảnh, đo đạc chỉ số hình thái học định lượng và xếp hạng cấp độ lão hóa lâm sàng theo thang đo Lemperle / Merz (Cấp độ 0 – 4).

Dự án đề xuất kiến trúc **DS-SAUNet (DSnake)** tích hợp mạng tích chập uốn lượn động (**Dynamic Snake Convolution - DSConv**) kết hợp cơ chế **Cross-Attention** và hàm mất mát tối ưu đường viền **Boundary-Aware Loss**.

---

## 📑 MỤC LỤC
1. [Cấu trúc Thư mục Dự án](#1-cấu-trúc-thư-mục-dự-án)
2. [Yêu cầu Hệ thống & Cài đặt Môi trường](#2-yêu-cầu-hệ-thống--cài-đặt-môi-trường)
3. [Hướng dẫn Chạy Giao diện Web Trực quan (Web Demo)](#3-hướng-dẫn-chạy-giao-diện-web-trực-quan-web-demo)
4. [Hướng dẫn Chạy Đánh giá So sánh Đối đầu (Benchmark)](#4-hướng-dẫn-chạy-đánh-giá-so-sánh-đối-đầu-benchmark)
5. [Hướng dẫn Chạy Đánh giá Đơn lẻ qua Terminal (CLI Pipeline)](#5-hướng-dẫn-chạy-đánh-giá-đơn-lẻ-qua-terminal-cli-pipeline)
6. [Hướng dẫn Huấn luyện Mô hình (Training)](#6-hướng-dẫn-huấn-luyện-mô-hình-training)
7. [Tải Dữ liệu Dataset](#7-tải-dữ-liệu-dataset)
8. [Tóm tắt Kết quả Thực nghiệm](#8-tóm-tắt-kết-quả-thực-nghiệm)

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN

Toàn bộ mã nguồn cốt lõi và các mô hình đã huấn luyện hoàn chỉnh được tổ chức độc lập trong thư mục `src/`:

```text
├── Bao_Cao_Thuc_Tap_Tot_Nghiep_HUFLIT.docx   # Báo cáo thực tập tốt nghiệp chính thức
├── README.md                                 # Hướng dẫn chi tiết vận hành dự án
├── requirements.txt                          # Danh sách các thư viện phụ thuộc
├── run_app.py                                # Launcher khởi động nhanh ứng dụng Web
├── benchmark_all.py                          # Launcher chạy benchmark đối đầu
├── download_data.py                          # Script tải ảnh chân dung FFHQ
├── download_wrinkle_data.py                  # Script tải bộ nhãn nếp nhăn Ground Truth
├── data/                                     # Dữ liệu ảnh và nhãn nếp nhăn (tùy chọn tải)
│   └── ffhq_wrinkle_base/
├── results/                                  # Biểu đồ so sánh và kết quả thực nghiệm
│   ├── benchmark_comparison_chart.png
│   └── benchmark_comparison.json
└── src/                                      # TOÀN BỘ MÃ NGUỒN VÀ MODEL ĐÃ HUẤN LUYỆN
    ├── models/                               # Kiến trúc và trọng số mạng học sâu
    │   ├── Baseline/                         # Mô hình nền tảng Baseline U-Net (ICPR 2024)
    │   │   ├── model.py, train.py
    │   │   └── checkpoints/baseline_unet.pth
    │   ├── AttentionUNet/                    # Mô hình Attention U-Net
    │   │   ├── model.py, train.py
    │   │   └── checkpoints/improved_attention_unet.pth
    │   ├── DSnake/                           # DS-SAUNet (Mô hình cải tiến đề xuất)
    │   │   ├── dsconv.py, model.py, train.py
    │   │   └── checkpoints/dsnake_unet.pth
    │   ├── unet/                             # Các khối chức năng UNet chuẩn
    │   ├── dataset.py                        # Xử lý dữ liệu đa kênh (RGB + Texture Gradient)
    │   ├── face_detector.py                  # Cắt cúp & chuẩn hóa mặt MediaPipe Face Mesh
    │   ├── losses.py                         # Boundary-Aware Loss & Combined Loss
    │   └── metrics.py                        # Các thang đo Dice, IoU, HD95, Precision, Recall
    ├── pipeline/                             # Quy trình 4 tầng đánh giá lâm sàng
    │   ├── stage1_roi_alignment.py           # Stage 1: Phân tách 4 vùng giải phẫu (ROI)
    │   ├── stage2_attention_segmentation.py  # Stage 2: Phân đoạn nếp nhăn Attention
    │   ├── stage3_morphometry.py             # Stage 3: Đo lường hình thái học & Skeletonization
    │   ├── stage4_clinical_scoring_xai.py    # Stage 4: Chấm điểm Lemperle 0-4 & Heatmap XAI
    │   └── pipeline_aging_assessment.py      # Script chạy toàn diện End-to-End từ 1 ảnh
    ├── web_app/                              # Ứng dụng Web Demonstration (Flask)
    │   ├── static/ (css, js, placeholder)
    │   ├── templates/index.html
    │   └── server.py
    ├── run_app.py                            # Entrypoint chạy web từ trong src
    ├── benchmark_all.py                      # Entrypoint chạy benchmark từ trong src
    ├── test_file_lists.txt                   # Danh sách 100 ảnh kiểm thử chuẩn hóa
    └── requirements.txt
```

---

## 2. YÊU CẦU HỆ THỐNG & CÀI ĐẶT MÔI TRƯỜNG

### Yêu cầu tối thiểu:
- **Hệ điều hành:** Windows 10/11, Ubuntu 20.04+, hoặc macOS.
- **Python:** Phiên bản 3.10 đến 3.12.
- **Phần cứng:** Tối thiểu 8GB RAM. Có GPU NVIDIA (CUDA 11.8 / 12.1) để tăng tốc độ suy luận (hệ thống tự động chuyển sang CPU nếu máy không có GPU).

### Các bước cài đặt:

#### Bước 2.1: Mở Terminal / PowerShell tại thư mục dự án
```powershell
cd Lao_hoa_da
```

#### Bước 2.2: Khởi tạo môi trường ảo Python (`.venv`)
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu PowerShell báo lỗi Execution Policy, chạy lệnh: `Set-ExecutionPolicy Unrestricted -Scope Process` rồi kích hoạt lại).*

- **Trên Windows (CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Bước 2.3: Nâng cấp pip và cài đặt thư viện
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Tùy chọn: Nếu máy có card màn hình NVIDIA và muốn dùng GPU CUDA 12.1, chạy lệnh:*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*)*

---

## 3. HƯỚNG DẪN CHẠY GIAO DIỆN WEB TRỰC QUAN (WEB DEMO)

Hệ thống tích hợp sẵn giao diện Web tương tác hoàn chỉnh, cho phép tải ảnh chân dung bất kỳ hoặc chọn ảnh trong kho dữ liệu kiểm thử để phân tích ngay lập tức.

### Cách khởi động:
Chạy lệnh sau tại thư mục gốc:
```bash
python run_app.py
```
*(Hoặc chạy từ trong `src`: `cd src && python run_app.py`)*

### Quá trình vận hành:
1. Console sẽ tự động tải 3 mô hình (`Baseline U-Net`, `Attention U-Net`, `DSnake`).
2. Trình duyệt mặc định sẽ tự động mở địa chỉ: **`http://127.0.0.1:5087`**.

### Các tính năng chính trên Web:
- **Tải ảnh chân dung từ máy tính:** Tự động phát hiện khuôn mặt bằng MediaPipe Face Mesh, loại bỏ nền xung quanh, xoay thẳng trục và cắt cúp đúng chuẩn FFHQ.
- **Chọn ảnh kiểm thử có sẵn:** Duyệt danh sách các ảnh trong tập Ground Truth để xem nhãn thật (Ground Truth Mask).
- **So sánh đối đầu song song 3 mô hình:**
  - Hiển thị trực quan mặt nạ nếp nhăn (Mask) và ảnh phủ đường nhăn (Overlay).
  - So sánh tốc độ xử lý (Inference Latency - ms) và số lượng tham số (Parameters).
  - Đối chiếu chỉ số định lượng thời gian thực: `Dice`, `IoU (Jaccard)`, `Precision`, `Recall`, `HD95`.
- **Đo lường hình thái học chuyên sâu:**
  - Thuật toán **Skeletonization (Topological thinning)**: Rút gọn các đường rãnh nhăn về xương sống 1-pixel để đo tổng chiều dài nếp nhăn (pixels).
  - Tính toán tỷ lệ diện tích nếp nhăn (**Wrinkle Area Fraction - WAF**) và mật độ rãnh nhăn.
  - Phân tích chi tiết theo 4 vùng giải phẫu da mặt: Vùng trán (Forehead), Đuôi mắt trái, Đuôi mắt phải (Crow's feet) và Rãnh cười (Nasolabial folds).
- **Chẩn đoán cấp độ lão hóa lâm sàng:**
  - Quy đổi tự động sang thang đo chuẩn da liễu **Lemperle / Merz Scale** từ Cấp độ 0 (không lão hóa/da mịn màng) đến Cấp độ 4 (lão hóa rất nặng/nếp nhăn sâu).
- **Bản đồ nhiệt giải thích XAI (Explainable AI Attention Heatmap):**
  - Trực quan hóa bản đồ nhiệt màu Jet để chỉ ra chính xác các rãnh nhăn trọng yếu dẫn đến kết luận chấm điểm lão hóa.

---

## 4. HƯỚNG DẪN CHẠY ĐÁNH GIÁ SO SÁNH ĐỐI ĐẦU (BENCHMARK)

Để kiểm chứng hiệu năng định lượng của cả 3 mô hình trên toàn bộ tập ảnh kiểm thử Ground Truth độc lập:

Chạy lệnh:
```bash
python benchmark_all.py
```
*(Hoặc: `cd src && python benchmark_all.py`)*

### Kết quả xuất ra:
1. **Bảng thống kê tại Terminal:**
   - Số lượng tham số (Model Parameters).
   - Điểm số phân đoạn: `Dice Score (F1)` và `mIoU (Jaccard)`.
   - Độ chính xác đường biên: `HD95 (Hausdorff Distance 95)` (pixel - càng nhỏ càng sắc nét).
   - Cân bằng `Precision` và `Recall` (độ nhạy phát hiện rãnh mảnh).
   - Thời gian suy luận trung bình mỗi ảnh (`Latency ms`).
2. **File JSON chi tiết:** Được lưu tại `results/benchmark_comparison.json`.
3. **Biểu đồ độ phân giải cao:** Tự động vẽ và lưu tại `results/benchmark_comparison_chart.png` gồm 3 đồ thị:
   - Biểu đồ cột kép so sánh Dice & IoU.
   - Biểu đồ so sánh sai số khoảng cách biên HD95.
   - Biểu đồ cân bằng giữa Precision và Recall.

---

## 5. HƯỚNG DẪN CHẠY ĐÁNH GIÁ ĐƠN LẺ QUA TERMINAL (CLI PIPELINE)

Để chạy độc lập quy trình 4 tầng trên 1 ảnh khuôn mặt bất kỳ mà không cần bật Web:

Chạy lệnh:
```bash
python src/pipeline/pipeline_aging_assessment.py
```

### Các bước quy trình được thực thi:
1. **Stage 1 (Landmark Alignment & 4 ROIs):** Nhận diện 468 điểm mốc Face Mesh, bóc tách 4 vùng giải phẫu (Trán, 2 Đuôi mắt, Rãnh mũi má).
2. **Stage 2 (Segmentation):** Đưa ảnh qua mô hình **DS-SAUNet** trích xuất rãnh nếp nhăn.
3. **Stage 3 (Quantitative Morphometry):** Rút gọn topological skeleton, tính tổng độ dài và mật độ rãnh nhăn.
4. **Stage 4 (Clinical Scoring & XAI):** Chẩn đoán cấp độ lão hóa 0–4 và xuất file ảnh nhiệt giải thích tại `results/xai_wrinkle_heatmap.png`.

---

## 6. HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH (TRAINING)

Các mô hình đều đã có sẵn trọng số tốt nhất được lưu trong thư mục `src/models/*/checkpoints/`. Nếu muốn huấn luyện lại từ đầu:

- **Huấn luyện mô hình Baseline U-Net:**
  ```bash
  python src/models/Baseline/train.py
  ```
- **Huấn luyện mô hình Attention U-Net:**
  ```bash
  python src/models/AttentionUNet/train.py
  ```
- **Huấn luyện mô hình đề xuất DS-SAUNet (DSnake):**
  ```bash
  python src/models/DSnake/train.py
  ```

---

## 7. TẢI DỮ LIỆU DATASET

*Lưu ý: Các checkpoint mô hình đã được tích hợp sẵn trong mã nguồn. Bạn chỉ cần tải dữ liệu khi muốn chạy lại quá trình huấn luyện hoặc mở rộng tập kiểm thử.*

1. **Tải bộ nhãn Ground Truth (1,000 nhãn chuyên gia gán và nhãn phân vùng mặt):**
   ```bash
   python download_wrinkle_data.py
   ```
   Dữ liệu sẽ được tự động tải từ Google Drive về `data/ffhq_wrinkle_base/`.

2. **Tải bộ ảnh chân dung gốc FFHQ từ Kaggle (~19.5 GB):**
   ```bash
   python download_data.py
   ```

---

## 8. TÓM TẮT KẾT QUẢ THỰC NGHIỆM

Đánh giá thực nghiệm trên tập Ground Truth gồm 73 ảnh chân dung FFHQ độ phân giải $1024 \times 1024$ (chuyên gia dán nhãn thủ công):

| Kiến trúc Mô Hình | Số Tham Số (Params) | Dice Score (F1) ↑ | IoU (Jaccard) ↑ | HD95 (Pixels) ↓ | Recall ↑ | Tốc độ (GPU Latency) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** (ICPR 2024) | 17,263,618 | 0.4428 | 0.2844 | 14.86 px | 0.3812 | ~18.5 ms |
| **Attention U-Net** | 17,591,939 | 0.4715 | 0.3085 | 12.40 px | 0.4250 | ~21.2 ms |
| **DSnake (DS-SAUNet - Đề xuất)** | **17,738,407** | **0.5082** | **0.3406** | **9.75 px** | **0.4890** | **~24.8 ms** |

### Nhận xét chuyên môn:
- **Khắc phục hiện tượng đứt gãy rãnh nhăn:** Nhờ các khối tích chập uốn lượn động (**DSConv**) bám sát theo các đường cong sinh học của da mặt, mô hình DSnake giúp tăng mạnh chỉ số **Recall (+10.78%)** so với Baseline, hạn chế tối đa việc bỏ sót các rãnh nhăn mảnh ở đuôi mắt và khóe cười.
- **Tối ưu hóa độ sắc nét đường biên:** Sự kết hợp giữa **Cross-Attention** tại Bottleneck và **Boundary Loss** giúp thu hẹp sai số khoảng cách viền biên **HD95 từ 14.86 px xuống 9.75 px** (giảm 34.4% sai số biên).
