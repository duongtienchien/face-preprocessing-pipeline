# Nghiên cứu & Ứng dụng Xử lý ảnh số (DIP) kết hợp CNN trong Nhận diện Cảm xúc Khuôn mặt (FER)

Dự án nghiên cứu giải pháp tối ưu hóa độ chính xác cho bài toán **Nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition - FER)** bằng cách kết hợp chuỗi tiền xử lý hình ảnh số (Digital Image Processing - DIP) chuyên sâu trước khi nạp vào mạng nơ-ron tích chập (Convolutional Neural Network).

---

## 👥 Phân công nhiệm vụ (Team Roles)

| Thành viên | Vai trò | Nhiệm vụ chính | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Hiếu** | **Lead - CNN & Model Pipeline** | - Thiết kế kiến trúc CNN (Custom CNN / MobileNetV2 / Mini-Xception)<br>- Thiết lập pipeline train/val, Loss function chống lệch nhãn<br>- Tối ưu siêu tham số, xuất Confusion Matrix | `train.py`, file trọng số tối ưu (`.pt` / `.onnx`), biểu đồ Loss/Acc, báo cáo phân tích lỗi |
| **Chiến** | **DIP Specialist (Xử lý ảnh)** | - Face Detection & Alignment (xoay trục mắt về phương ngang)<br>- Pipeline DIP: Bilateral Filter khử nhiễu + CLAHE trên không gian màu LAB<br>- Batch processing làm sạch dữ liệu, đo chỉ số DIP độc lập | `preprocess.py`, 2 tập dữ liệu sạch (`dataset_no_dip`, `dataset_with_dip`), bảng số liệu thực nghiệm DIP |
| **Linh** | **System Integration & App** | - Tích hợp Webcam realtime (Đọc frame $\rightarrow$ Ní 2 xử lý $\rightarrow$ Ní 1 dự đoán $\rightarrow$ HUD)<br>- Giao diện trực quan (Streamlit / OpenCV HUD bar chart)<br>- Quản lý tiến độ, hoàn thiện slide & báo cáo kỹ thuật | `app.py`, Slide thuyết trình, File báo cáo kỹ thuật (Word/LaTeX) |

---

## 🔬 Pipeline tiền xử lý hình ảnh (DIP Pipeline)

Chuỗi xử lý dữ liệu được thiết kế nhằm chuẩn hóa hình học khuôn mặt và làm nổi bật các biến dạng vi mô trên cơ mặt (micro-expressions):

1. **Face Detection & Alignment:** Dùng MediaPipe trích xuất landmark 2 mắt, tính góc nghiêng $\theta$ và xoay ảnh bằng phép biến đổi Affine để trục mắt luôn nằm ngang.
2. **Face Cropping & Padding:** Cắt vùng mặt theo bounding box chuẩn có tỉ lệ đệm (padding) 20% tránh cụt cằm/trán.
3. **Bilateral Filtering:** Khử nhiễu làm mịn bề mặt da nhưng bảo tồn độ sắc nét của viền mắt, sống mũi và khuôn miệng.
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Cân bằng sáng cục bộ trên kênh L (không gian màu LAB) để tái hiện nếp nhăn cơ mặt trong vùng bóng tối mà không làm biến đổi sắc thái màu da.
5. **Resize Standard:** Quy chuẩn toàn bộ ảnh về kích thước đồng nhất **112 × 112**.

---

## 📊 Báo cáo làm sạch dữ liệu & Thực nghiệm DIP

### 1. Thống kê làm sạch dữ liệu (Dataset: KDEF)
* **Tổng số ảnh thô ban đầu:** 2.938 ảnh (7 nhãn cảm xúc đa góc chụp).
* **Số lượng ảnh xử lý thành công:** **2.697 ảnh (91.8%)**.
* **Số lượng ảnh loại bỏ (Discarded):** 241 ảnh (8.2% do góc quay ngang $90^\circ$ hoặc cơ mặt bị khuất).

### 2. Đánh giá chất lượng ảnh định lượng (DIP Metrics)

| Chỉ số đánh giá | Tập đối chứng (Không DIP) | Tập đề xuất (Có DIP) | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **Số lượng ảnh sạch** | 2.697 | 2.697 | Chuẩn hóa 100% |
| **RMS Contrast (Độ tương phản)** | 47.04 | **55.26** | **+17.5%** |
| **Laplacian Variance (Độ nét biên)** | 447.85 | **813.54** | **+81.7%** |

---

## 📁 Cấu trúc thư mục dự án

```text
XuLyAnh/
├── preprocess.py          # [Ní 2] Pipeline tiền xử lý ảnh và hàm batch processing
├── evaluate_metrics.py    # [Ní 2] Đo chỉ số RMS Contrast và Laplacian Variance
├── cleaning_report.txt    # [Ní 2] Log chi tiết các mẫu dữ liệu lỗi bị loại bỏ
│
├── models/                # [Ní 1] Định nghĩa các kiến trúc CNN
│   └── cnn_model.py
├── train.py               # [Ní 1] Huấn luyện, kiểm thử và vẽ đồ thị
├── weights/               # [Ní 1] Chứa file trọng số mô hình tốt nhất (.pt / .onnx)
│
├── app.py                 # [Ní 3] Ứng dụng đọc webcam realtime hiển thị kết quả
├── docs/                  # [Ní 3] Slide báo cáo và tài liệu kỹ thuật
│
├── requirements.txt       # Danh sách thư viện phụ thuộc
└── .gitignore
Chiến lược phân nhánh Git (Branching Workflow)

    main: Nhánh chính chứa mã nguồn ổn định nhất.

    feat/cnn-model-training: Nhánh làm việc của Ní 1 để code kiến trúc mô hình và huấn luyện.

    feat/app-integration: Nhánh làm việc của Ní 3 để ghép nối webcam và xây dựng giao diện.

Hướng dẫn cho thành viên nhóm:

    Clone repo về máy:
    Bash

    git clone <REPO_URL>
    cd XuLyAnh

    Chuyển sang nhánh tính năng được phân công:
    Bash

    # Dành cho Ní 1:
    git checkout feat/cnn-model-training

    # Dành cho Ní 3:
    git checkout feat/app-integration

    Sau khi hoàn thành tính năng, đẩy code lên nhánh của mình và tạo Pull Request vào main.

🚀 Hướng dẫn cài đặt & Chạy
1. Cài đặt môi trường
Bash

python3 -m venv venv
source venv/bin/activate   # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Tải dữ liệu đã qua xử lý sạch

Dữ liệu đã qua làm sạch được chia sẻ qua Google Drive để tránh làm nặng repository Git:

    Link tải Dataset Cleaned (Google Drive)

        Giải nén dataset_no_dip/ và dataset_with_dip/ vào thư mục gốc dự án.

3. Chạy đo đạc chỉ số DIP
Bash

python evaluate_metrics.py


---

