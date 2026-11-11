
import torch
import torch.nn as nn


# So luong lop cam xuc
NUM_CLASSES = 7

# Kich thuoc anh dau vao
IMAGE_SIZE = 112

# Thu tu nhan cam xuc
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


class CustomCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super(CustomCNN, self).__init__()

        # Khoi CNN 1: 3 kenh mau -> 32 kenh
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Khoi CNN 2: 32 -> 64 kenh
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Khoi CNN 3: 64 -> 128 kenh
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Khoi CNN 4: 128 -> 256 kenh
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Dua kich thuoc dac trung ve 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Phan phan loai cam xuc
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avgpool(x)

        x = self.classifier(x)

        return x


# Kiem tra mo hinh
if __name__ == "__main__":

    model = CustomCNN()

    # Tao 2 anh gia lap kich thuoc 112x112
    test_images = torch.randn(2, 3, 112, 112)

    # Dua anh qua CNN
    outputs = model(test_images)

    print("Kiem tra Custom CNN")
    print("So luong tham so:",
          sum(p.numel() for p in model.parameters()))

    print("Kich thuoc dau vao:", test_images.shape)
    print("Kich thuoc dau ra:", outputs.shape)

    print("Danh sach cam xuc:", EMOTION_LABELS)

    print("Mo hinh hoat dong thanh cong!")
