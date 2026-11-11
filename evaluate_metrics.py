"""
Ní 2 - Đo chỉ số chất lượng ảnh (RMS Contrast, Laplacian Variance) trên 2 tập dữ liệu.

Lưu ý khi viết báo cáo: đây là CHỈ SỐ PHỤ. Laplacian Variance tăng một phần do CLAHE
khuếch đại cả nhiễu/texture, nên không tự nó chứng minh ảnh "tốt hơn". Bằng chứng chính
là accuracy / F1 của cùng một model CNN train trên dataset_no_dip vs dataset_with_dip.

Chạy:
    python evaluate_metrics.py
"""
import cv2
import numpy as np
import glob
import os


def rms_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()


def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_metrics(folder):
    files = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
    valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    rms_list, sharp_list = [], []
    for f in valid_files:
        img = cv2.imread(f)
        if img is not None:
            rms_list.append(rms_contrast(img))
            sharp_list.append(laplacian_sharpness(img))

    if not rms_list:
        raise FileNotFoundError(f"Không đọc được ảnh nào trong '{folder}'. Đã chạy preprocess.py chưa?")
    return np.mean(rms_list), np.mean(sharp_list), len(rms_list)


def pct(new, old):
    return (new - old) / old * 100


def main(no_dip_dir="dataset_no_dip", with_dip_dir="dataset_with_dip"):
    print("Đang tính toán chỉ số định lượng...")
    rms_no, sharp_no, total_no = calculate_metrics(no_dip_dir)
    rms_dip, sharp_dip, total_dip = calculate_metrics(with_dip_dir)

    print("\n" + "=" * 75)
    print("BẢNG SỐ LIỆU CHẤT LƯỢNG ẢNH CỦA PIPELINE DIP")
    print("=" * 75)
    print(f"{'Chỉ số đánh giá':<30} | {'Không DIP':>10} | {'Có DIP':>10} | {'Thay đổi':>9}")
    print("-" * 75)
    print(f"{'Số lượng ảnh sạch':<30} | {total_no:>10} | {total_dip:>10} |")
    print(f"{'RMS Contrast (Độ tương phản)':<30} | {rms_no:>10.2f} | {rms_dip:>10.2f} | {pct(rms_dip, rms_no):>+8.1f}%")
    print(f"{'Laplacian Var (Độ nét biên)':<30} | {sharp_no:>10.2f} | {sharp_dip:>10.2f} | {pct(sharp_dip, sharp_no):>+8.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
