
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CLASSES = 7

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]


class MobileNetV2FER(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        # Tai MobileNetV2
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)

        # Thay lop phan loai cu bang 7 lop cam xuc
        input_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_features, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)


# Kiem tra mo hinh
if __name__ == "__main__":
    import torch

    model = MobileNetV2FER(pretrained=False)
    test_image = torch.randn(2, 3, 112, 112)

    output = model(test_image)

    print("So luong tham so:", sum(p.numel() for p in model.parameters()))
    print("Kich thuoc dau vao:", test_image.shape)
    print("Kich thuoc dau ra:", output.shape)
    print("Danh sach cam xuc:", EMOTION_LABELS)
    print("MobileNetV2 hoat dong thanh cong!")
