
import os

DATASETS = {
    "KHONG DIP": "XuLyAnh_dataset/dataset_no_dip",
    "CO DIP": "XuLyAnh_dataset/dataset_with_dip"
}

EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

for name, path in DATASETS.items():

    print("\n" + "=" * 50)
    print("DATASET:", name)
    print("=" * 50)

    total = 0

    for emotion in EMOTIONS:
        folder = os.path.join(path, emotion)

        if not os.path.exists(folder):
            print(emotion, ": KHONG TIM THAY")
            continue

        count = 0

        for file in os.listdir(folder):
            if file.lower().endswith(IMAGE_EXTENSIONS):
                count += 1

        total += count
        print(f"{emotion:10s}: {count:5d} anh")

    print("-" * 50)
    print("TONG SO ANH:", total)

print("\nDA KIEM TRA XONG!")
