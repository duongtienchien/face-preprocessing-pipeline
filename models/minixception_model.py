
import torch
import torch.nn as nn


NUM_CLASSES = 7


# Block convolution tach biet
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Block Mini-Xception
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            SeparableConv(in_channels, out_channels),
            SeparableConv(out_channels, out_channels),
            nn.MaxPool2d(2)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# Mo hinh Mini-Xception nhan dien 7 cam xuc
class MiniXception(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.block1 = XceptionBlock(32, 64)
        self.block2 = XceptionBlock(64, 128)
        self.block3 = XceptionBlock(128, 256)
        self.block4 = XceptionBlock(256, 512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)


# Kiem tra mo hinh
if __name__ == "__main__":
    model = MiniXception()
    sample = torch.randn(2, 3, 112, 112)

    output = model(sample)

    print("Dau vao:", sample.shape)
    print("Dau ra:", output.shape)
    print("So tham so:", sum(p.numel() for p in model.parameters()))
    print("Mini-Xception hoat dong thanh cong!")
