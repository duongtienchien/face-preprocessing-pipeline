"""
Ní 2 - Pipeline tiền xử lý ảnh (DIP) cho bài toán nhận diện cảm xúc khuôn mặt.

ĐẦU RA CHUẨN (Ní 1 và Ní 3 dựa vào đây):
    - Ảnh màu BGR (thứ tự kênh của OpenCV, KHÔNG phải RGB)
    - Kích thước 112 x 112
    - Kiểu uint8, giá trị pixel 0..255 (chưa chuẩn hoá)

Cách dùng trong code khác (vd app.py):
    from preprocess import FacePreprocessor
    pre = FacePreprocessor(static_image_mode=False)   # webcam
    face = pre.process(frame_bgr, apply_dip=True)      # None nếu không thấy mặt

Chạy batch làm sạch dữ liệu:
    python preprocess.py
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm import tqdm


def _largest(detections):
    """Chọn khuôn mặt có diện tích bbox lớn nhất nếu ảnh có nhiều người."""
    return max(
        detections,
        key=lambda d: d.location_data.relative_bounding_box.width
        * d.location_data.relative_bounding_box.height,
    )


class FacePreprocessor:
    def __init__(self, target_size=(112, 112), padding_ratio=0.2, static_image_mode=True):
        """
        static_image_mode=True : ảnh tĩnh trong dataset (mặt có thể ở xa) -> model full-range (1)
        static_image_mode=False: webcam, mặt ở gần camera -> model short-range (0), nhanh hơn
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1 if static_image_mode else 0,
            min_detection_confidence=0.5,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _detect_largest(self, img_bgr):
        res = self.detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return _largest(res.detections) if res.detections else None

    @staticmethod
    def _eye_angle(detection, w, h):
        """Góc nghiêng (độ) của trục 2 mắt, chuẩn hoá về [-180, 180), và trung điểm 2 mắt."""
        kp = detection.location_data.relative_keypoints
        right_eye = (kp[0].x * w, kp[0].y * h)  # Mắt phải người trong ảnh (bên trái ảnh)
        left_eye = (kp[1].x * w, kp[1].y * h)   # Mắt trái người trong ảnh (bên phải ảnh)
        angle = np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
        angle = (angle + 180) % 360 - 180
        center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        return float(angle), center

    @staticmethod
    def _rotate(img_bgr, center, angle):
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

    def align_and_crop(self, img_bgr):
        h, w = img_bgr.shape[:2]
        best_detection = self._detect_largest(img_bgr)
        if best_detection is None:
            return None

        # 1. Alignment: tính góc trục 2 mắt, xoay ảnh quanh trung điểm 2 mắt
        angle, eyes_center = self._eye_angle(best_detection, w, h)
        rotated = self._rotate(img_bgr, eyes_center, angle)

        # 2. Phát hiện lại trên ảnh đã xoay (để lấy bbox mới + kiểm tra độ lệch còn lại)
        det_rot = self._detect_largest(rotated)

        # Khi mặt nghiêng nhiều (> ~15 độ), keypoint mắt của MediaPipe kém chính xác
        # nên vòng 1 có thể xoay lố. Đo lại độ lệch trên ảnh đã xoay, nếu còn > 2 độ
        # thì xoay bù vòng 2 (ảnh lúc này gần thẳng nên keypoint chính xác hơn).
        if det_rot is not None:
            residual, _ = self._eye_angle(det_rot, w, h)
            if abs(residual) > 2.0:
                angle += residual
                rotated = self._rotate(img_bgr, eyes_center, angle)
                det_rot = self._detect_largest(rotated)

        if det_rot is None:
            # Fallback nếu sau khi xoay không bắt lại được mặt
            bbox = best_detection.location_data.relative_bounding_box
        else:
            bbox = det_rot.location_data.relative_bounding_box

        # Tính tọa độ crop kèm padding
        bw, bh = int(bbox.width * w), int(bbox.height * h)
        bx, by = int(bbox.xmin * w), int(bbox.ymin * h)
        pad_x, pad_y = int(bw * self.padding_ratio), int(bh * self.padding_ratio)

        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None
        return rotated[y1:y2, x1:x2]

    def resize(self, face_bgr):
        return cv2.resize(face_bgr, self.target_size, interpolation=cv2.INTER_AREA)

    def dip_pipeline(self, face_bgr):
        if face_bgr is None:
            return None
        # Khử nhiễu giữ biên cạnh
        denoised = cv2.bilateralFilter(face_bgr, d=7, sigmaColor=50, sigmaSpace=50)
        # Cân bằng tương phản trên kênh L
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_lab = cv2.merge((self.clahe.apply(l), a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return self.resize(enhanced_bgr)

    def process(self, img_bgr, apply_dip=True):
        """Dùng cho app / train: trả về 1 ảnh 112x112 BGR uint8, hoặc None."""
        face_aligned = self.align_and_crop(img_bgr)
        if face_aligned is None:
            return None
        if apply_dip:
            return self.dip_pipeline(face_aligned)
        return self.resize(face_aligned)

    def process_both(self, img_bgr):
        """
        Dùng cho batch: detect + align + crop MỘT lần, rồi tách 2 nhánh.
        Trả về (face_no_dip, face_with_dip) hoặc (None, None).
        """
        face_aligned = self.align_and_crop(img_bgr)
        if face_aligned is None:
            return None, None
        return self.resize(face_aligned), self.dip_pipeline(face_aligned)


def run_batch(raw_dir="dataset_raw", out_no_dip="dataset_no_dip", out_with_dip="dataset_with_dip",
              report_path="cleaning_report.txt"):
    processor = FacePreprocessor(static_image_mode=True)
    failed_files = []

    # Quét tất cả file ảnh trong cây thư mục
    image_paths = glob.glob(os.path.join(raw_dir, "**", "*.*"), recursive=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = sorted(p for p in image_paths if p.lower().endswith(valid_exts))

    if not image_paths:
        print(f"Không tìm thấy ảnh nào trong '{raw_dir}'. Kiểm tra lại đường dẫn.")
        return

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")
    for p in tqdm(image_paths):
        rel_path = os.path.relpath(p, raw_dir)
        img = cv2.imread(p)
        if img is None:
            failed_files.append((p, "Lỗi đọc file"))
            continue

        face_no_dip, face_with_dip = processor.process_both(img)
        if face_with_dip is None:
            failed_files.append((p, "Không nhận diện được khuôn mặt"))
            continue

        # Lưu ảnh đồng bộ cấu trúc thư mục
        p1 = os.path.join(out_no_dip, rel_path)
        p2 = os.path.join(out_with_dip, rel_path)
        os.makedirs(os.path.dirname(p1), exist_ok=True)
        os.makedirs(os.path.dirname(p2), exist_ok=True)
        cv2.imwrite(p1, face_no_dip)
        cv2.imwrite(p2, face_with_dip)

    # Thống kê số ảnh bị loại theo từng nhãn (thư mục con đầu tiên)
    per_label = {}
    for path, _ in failed_files:
        label = os.path.relpath(path, raw_dir).split(os.sep)[0]
        per_label[label] = per_label.get(label, 0) + 1

    # encoding utf-8 để file tiếng Việt không lỗi font trên Windows
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Tổng số: {len(image_paths)}\n"
                f"Thành công: {len(image_paths) - len(failed_files)}\n"
                f"Thất bại: {len(failed_files)}\n\n")
        f.write("Số ảnh bị loại theo nhãn:\n")
        for label, n in sorted(per_label.items()):
            f.write(f"  {label}: {n}\n")
        f.write("\n")
        for path, reason in failed_files:
            f.write(f"{path} -> {reason}\n")
    print(f"Hoàn tất Batch Processing. Đã xuất log tại {report_path}.")


if __name__ == "__main__":
    run_batch()
