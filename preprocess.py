import cv2
import mediapipe as mp
import numpy as np
import os

class FacePreprocessor:
    def __init__(self, target_size=(112, 112), padding_ratio=0.2):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        
        # Khởi tạo MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1: tối ưu ảnh chân dung/khoảng cách trung bình
            min_detection_confidence=0.5
        )
        
        # Khởi tạo bộ cân bằng sáng cục bộ CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect_and_crop(self, image_bgr):
        """Phát hiện và crop khuôn mặt kèm padding an toàn"""
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)

        if not results.detections:
            return None

        # Lấy khuôn mặt đầu tiên
        box = results.detections[0].location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # Mở rộng biên (padding) để không bị xén trán hoặc cằm
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)

        xmin = max(0, x - pad_x)
        ymin = max(0, y - pad_y)
        xmax = min(w, x + bw + pad_x)
        ymax = min(h, y + bh + pad_y)

        return image_bgr[ymin:ymax, xmin:xmax]

    def dip_pipeline(self, face_bgr):
        """Pipeline DIP: Khử nhiễu (Bilateral) + Cân bằng sáng (CLAHE) + Resize"""
        if face_bgr is None:
            return None

        # 1. Khử nhiễu giữ cạnh bằng Bilateral Filter
        denoised = cv2.bilateralFilter(face_bgr, d=7, sigmaColor=50, sigmaSpace=50)

        # 2. Cân bằng tương phản bằng CLAHE trên kênh L (không gian màu LAB)
        # Cách này cải thiện sáng tối tự nhiên mà không làm biến dạng màu da
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 3. Chuẩn hóa kích thước
        resized = cv2.resize(enhanced_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    def process(self, image_input, apply_dip=True):
        """
        Hàm wrapper nhận đường dẫn file hoặc mảng numpy
        apply_dip=True: Chạy full pipeline DIP
        apply_dip=False: Chỉ crop & resize (phục vụ đối sánh thực nghiệm)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                print(f"[Lỗi] File không tồn tại: {image_input}")
                return None
            image = cv2.imread(image_input)
        else:
            image = image_input

        if image is None:
            return None

        face_crop = self.detect_and_crop(image)
        if face_crop is None:
            return None

        if apply_dip:
            return self.dip_pipeline(face_crop)
        else:
            return cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    processor = FacePreprocessor(target_size=(112, 112))
    input_file = "test.png"

    # 1. Chạy ảnh có xử lý DIP
    face_with_dip = processor.process(input_file, apply_dip=True)
    
    # 2. Chạy ảnh gốc chỉ crop (dùng cho bảng số liệu thực nghiệm đối sánh)
    face_no_dip = processor.process(input_file, apply_dip=False)

    if face_with_dip is not None and face_no_dip is not None:
        cv2.imwrite("result_with_dip.jpg", face_with_dip)
        cv2.imwrite("result_no_dip.jpg", face_no_dip)
        
        # Ghép 2 ảnh cạnh nhau phóng to lên để mắt thường dễ quan sát so sánh
        comparison = np.hstack((face_no_dip, face_with_dip))
        comparison_display = cv2.resize(comparison, (448, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("comparison.jpg", comparison_display)
        
        print("Xử lý thành công!")
        print("-> Đã xuất 'result_no_dip.jpg', 'result_with_dip.jpg' và 'comparison.jpg'")
    else:
        print("Không tìm thấy khuôn mặt trong ảnh test.jpg. Vui lòng thử ảnh khác.")