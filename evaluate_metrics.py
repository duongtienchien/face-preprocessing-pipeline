import cv2
import numpy as np
import glob

def rms_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.std()

def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_metrics(folder):
    files = glob.glob(f"{folder}/**/*.*", recursive=True)
    valid_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    rms_list, sharp_list = [], []
    for f in valid_files:
        img = cv2.imread(f)
        if img is not None:
            rms_list.append(rms_contrast(img))
            sharp_list.append(laplacian_sharpness(img))
            
    return np.mean(rms_list), np.mean(sharp_list), len(valid_files)

print("Đang tính toán chỉ số định lượng...")
rms_no, sharp_no, total_no = calculate_metrics("dataset_no_dip")
rms_dip, sharp_dip, total_dip = calculate_metrics("dataset_with_dip")

print("\n" + "="*60)
print("BẢNG SỐ LIỆU CHỨNG MINH HIỆU QUẢ CỦA PIPELINE DIP")
print("="*60)
print(f"{'Chỉ số đánh giá':<25} | {'Không DIP':<15} | {'Có DIP (Đề xuất)':<15}")
print("-"*60)
print(f"{'Số lượng ảnh sạch':<25} | {total_no:<15} | {total_dip:<15}")
print(f"{'RMS Contrast (Độ tương phản)':<25} | {rms_no:<15.2f} | {rms_dip:<15.2f} (+{(rms_dip-rms_no)/rms_no*100:.1f}%)")
print(f"{'Laplacian Var (Độ nét biên)':<25} | {sharp_no:<15.2f} | {sharp_dip:<15.2f}")
print("="*60)