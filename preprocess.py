import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm import tqdm

class FacePreprocessor:
    def __init__(self, target_size=(112, 112), padding_ratio=0.2):
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def align_and_crop(self, img_bgr):
        h, w, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)

        if not results.detections:
            return None

        # Chọn khuôn mặt có diện tích lớn nhất nếu ảnh có nhiều người
        best_detection = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height
        )

        # 1. Trích xuất landmark 2 mắt để thực hiện Alignment
        keypoints = best_detection.location_data.relative_keypoints
        right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h)) # Mắt phải người trong ảnh
        left_eye  = (int(keypoints[1].x * w), int(keypoints[1].y * h)) # Mắt trái người trong ảnh

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Tâm xoay là trung điểm 2 mắt
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

        # 2. Phát hiện lại vị trí bbox trên ảnh đã xoay thẳng
        rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        results_rot = self.detector.process(rotated_rgb)
        if not results_rot.detections:
            # Fallback nếu sau khi xoay không bắt lại được mặt
            bbox = best_detection.location_data.relative_bounding_box
        else:
            bbox = results_rot.detections[0].location_data.relative_bounding_box

        # Tính tọa độ crop kèm padding
        bw, bh = int(bbox.width * w), int(bbox.height * h)
        bx, by = int(bbox.xmin * w), int(bbox.ymin * h)
        pad_x, pad_y = int(bw * self.padding_ratio), int(bh * self.padding_ratio)

        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)

        return rotated[y1:y2, x1:x2]

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
        return cv2.resize(enhanced_bgr, self.target_size, interpolation=cv2.INTER_AREA)

    def process(self, img_bgr, apply_dip=True):
        face_aligned = self.align_and_crop(img_bgr)
        if face_aligned is None:
            return None
        if apply_dip:
            return self.dip_pipeline(face_aligned)
        return cv2.resize(face_aligned, self.target_size, interpolation=cv2.INTER_AREA)


def run_batch(raw_dir="dataset_raw", out_no_dip="dataset_no_dip", out_with_dip="dataset_with_dip"):
    processor = FacePreprocessor()
    failed_files = []
    
    # Quét tất cả file ảnh trong cây thư mục
    image_paths = glob.glob(f"{raw_dir}/**/*.*", recursive=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_exts)]
    
    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")
    for p in tqdm(image_paths):
        rel_path = os.path.relpath(p, raw_dir)
        img = cv2.imread(p)
        if img is None:
            failed_files.append((p, "Lỗi đọc file"))
            continue

        face_no_dip = processor.process(img, apply_dip=False)
        face_with_dip = processor.process(img, apply_dip=True)

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

    with open("cleaning_report.txt", "w") as f:
        f.write(f"Tổng số: {len(image_paths)}\nThành công: {len(image_paths) - len(failed_files)}\nThất bại: {len(failed_files)}\n\n")
        for item in failed_files:
            f.write(f"{item[0]} -> {item[1]}\n")
    print("Hoàn tất Batch Processing. Đã xuất log tại cleaning_report.txt.")