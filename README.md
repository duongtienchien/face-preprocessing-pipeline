# Nghiên cứu & Ứng dụng Xử lý ảnh số (DIP) kết hợp CNN trong Nhận diện Cảm xúc Khuôn mặt (FER)

Dự án nghiên cứu giải pháp tối ưu hóa độ chính xác cho bài toán **Nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition - FER)** bằng cách kết hợp chuỗi tiền xử lý hình ảnh số (Digital Image Processing - DIP) chuyên sâu trước khi nạp vào mạng nơ-ron tích chập (Convolutional Neural Network).

---

## 👥 Phân công nhiệm vụ (Team Roles)

| Thành viên | Vai trò | Nhiệm vụ chính | Sản phẩm bàn giao (Deliverables) |
| :--- | :--- | :--- | :--- |
| **Hiếu** | **Lead - CNN & Model Pipeline** | - Thiết kế kiến trúc CNN (Custom CNN / MobileNetV2 / Mini-Xception)<br>- Thiết lập pipeline train/val, Loss function chống lệch nhãn<br>- Tối ưu siêu tham số, xuất Confusion Matrix | `train.py`, file trọng số tối ưu (`.pt` / `.onnx`), biểu đồ Loss/Acc, báo cáo phân tích lỗi |
| **Chiến** | **DIP Specialist (Xử lý ảnh)** | - Face Detection & Alignment (xoay trục mắt về phương ngang)<br>- Pipeline DIP: Bilateral Filter khử nhiễu + CLAHE trên không gian màu LAB<br>- Batch processing làm sạch dữ liệu, đo chỉ số DIP độc lập | `preprocess.py`, 2 tập dữ liệu sạch (`dataset_no_dip`, `dataset_with_dip`), bảng số liệu thực nghiệm DIP |
| **Linh** | **System Integration & App** | - Tích hợp Webcam realtime (Đọc frame → Ní 2 xử lý → Ní 1 dự đoán → HUD)<br>- Giao diện trực quan (Streamlit / OpenCV HUD bar chart)<br>- Quản lý tiến độ & báo cáo kỹ thuật | `app.py`, File báo cáo kỹ thuật (Word/LaTeX) |

---

## 🤝 Quy ước dữ liệu giữa các module (đọc kỹ trước khi code)

Đầu ra của `preprocess.py` (cả 2 tập dataset lẫn hàm `FacePreprocessor.process()`):

| Thuộc tính | Giá trị |
| :--- | :--- |
| Thứ tự kênh màu | **BGR** (chuẩn OpenCV, *không phải* RGB) |
| Kích thước | **112 × 112 × 3** |
| Kiểu dữ liệu | **uint8**, pixel 0..255 (chưa chuẩn hoá) |
| Cấu trúc thư mục | `dataset_xxx/<tên_nhãn>/<ảnh>.jpg` (7 nhãn: angry, disgust, fear, happy, neutral, sad, surprise) |
| Không thấy mặt | `process()` trả về `None` |

**Ní 1 (train):**
- Nếu dùng model pretrained trên ImageNet (MobileNetV2…), nhớ `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` rồi mới normalize.
- **Chia train/val/test theo người (subject), không chia ngẫu nhiên.** KDEF có nhiều ảnh của cùng một người; chia ngẫu nhiên sẽ khiến cùng một người nằm ở cả train và val → accuracy ảo.
- Thực nghiệm chính của đề tài: train **cùng một model, cùng siêu tham số, cùng cách chia** trên `dataset_no_dip` và `dataset_with_dip`, so sánh accuracy / macro-F1.

**Ní 3 (app):** bắt buộc dùng đúng pipeline lúc train, không tự viết lại:
```python
from preprocess import FacePreprocessor

pre = FacePreprocessor(static_image_mode=False)   # webcam: model short-range, nhanh hơn
face = pre.process(frame_bgr, apply_dip=True)      # cùng apply_dip với model đang dùng
if face is not None:
    ...  # đưa vào model của Ní 1
```

---

## 🔬 Pipeline tiền xử lý hình ảnh (DIP Pipeline)

Chuỗi xử lý dữ liệu được thiết kế nhằm chuẩn hóa hình học khuôn mặt và làm nổi bật các biến dạng vi mô trên cơ mặt (micro-expressions):

1. **Face Detection & Alignment:** Dùng MediaPipe trích xuất landmark 2 mắt, tính góc nghiêng $\theta$ và xoay ảnh bằng phép biến đổi Affine để trục mắt luôn nằm ngang. Nếu sau lần xoay đầu trục mắt vẫn lệch > 2° (thường gặp khi mặt nghiêng nhiều), thực hiện thêm một vòng xoay bù.
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

Số ảnh bị loại **không đều giữa các nhãn** (cần lưu ý khi xử lý lệch nhãn):

| angry | disgust | fear | happy | neutral | sad | surprise |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 32 | 24 | 36 | 23 | 32 | 41 | 53 |

### 2. Đánh giá chất lượng ảnh định lượng (DIP Metrics, chỉ số phụ)

| Chỉ số đánh giá | Tập đối chứng (Không DIP) | Tập đề xuất (Có DIP) | Thay đổi |
| :--- | :---: | :---: | :---: |
| **Số lượng ảnh sạch** | 2.697 | 2.697 | Chuẩn hóa 100% |
| **RMS Contrast (Độ tương phản)** | 47.04 | **55.26** | **+17.5%** |
| **Laplacian Variance (Độ nét biên)** | 447.85 | **813.54** | **+81.7%** |

> ⚠️ Đây là chỉ số chất lượng ảnh, **không trực tiếp chứng minh** DIP giúp nhận diện tốt hơn. Laplacian Variance tăng một phần do CLAHE khuếch đại cả nhiễu/texture. Bằng chứng chính là bảng dưới.

### 3. Hiệu quả của DIP lên mô hình CNN (Ní 1 cập nhật)

| Tập huấn luyện | Model | Accuracy | Macro-F1 |
| :--- | :--- | :---: | :---: |
| `dataset_no_dip` | _TBD_ | _TBD_ | _TBD_ |
| `dataset_with_dip` | _TBD_ | _TBD_ | _TBD_ |

---

## 📁 Cấu trúc thư mục dự án

```text
face-preprocessing-pipeline/
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
```

---

## 🌿 Chiến lược phân nhánh Git (Branching Workflow)

* `main`: Nhánh chính chứa mã nguồn ổn định nhất.
* `feat/cnn-model-training`: Nhánh làm việc của Ní 1 để code kiến trúc mô hình và huấn luyện.
* `feat/app-integration`: Nhánh làm việc của Ní 3 để ghép nối webcam và xây dựng giao diện.

**Hướng dẫn cho thành viên nhóm:**

1. Clone repo về máy:
   ```bash
   git clone https://github.com/duongtienchien/face-preprocessing-pipeline.git
   cd face-preprocessing-pipeline
   ```
2. Chuyển sang nhánh tính năng được phân công, và kéo code mới nhất từ `main`:
   ```bash
   # Dành cho Ní 1:
   git checkout feat/cnn-model-training
   # Dành cho Ní 3:
   git checkout feat/app-integration

   git merge main
   ```
3. Khi thêm thư viện mới, **ghi vào `requirements.txt`** ở mục của mình.
4. Sau khi hoàn thành tính năng, đẩy code lên nhánh của mình và tạo Pull Request vào `main`.

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt môi trường

> ⚠️ Yêu cầu **Python 3.9 – 3.12**. Python 3.13 chỉ cài được mediapipe bản mới, bản này đã bỏ API `mp.solutions` mà code đang dùng.
> Không cài thêm `opencv-python`: mediapipe đã tự cài `opencv-contrib-python`, cài cả hai sẽ bị xung đột.

```bash
python3.12 -m venv venv
source venv/bin/activate   # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tải dữ liệu đã qua xử lý sạch

Dữ liệu đã qua làm sạch được chia sẻ qua Google Drive để tránh làm nặng repository Git:

* [Link tải Dataset Cleaned (Google Drive)](https://drive.google.com/file/d/10muBvWrnbhZPrxPxtAhEEL3Lw96A5mSp/view?usp=sharing)

Giải nén vào thư mục gốc của repo để có `dataset_no_dip/` và `dataset_with_dip/`.

### 3. (Tuỳ chọn) Chạy lại tiền xử lý từ dữ liệu thô

Đặt ảnh gốc vào `dataset_raw/<tên_nhãn>/`, rồi:
```bash
python preprocess.py
```

### 4. Chạy đo đạc chỉ số DIP

```bash
python evaluate_metrics.py
```
