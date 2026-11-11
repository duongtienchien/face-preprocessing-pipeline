
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from models.minixception_model import MiniXception


# ==================== CAU HINH ====================

CLASSES = ["angry", "disgust", "fear", "happy",
           "neutral", "sad", "surprise"]

SIZE = 112
BATCH = 16
LR = 0.0001

NO_DIP = "XuLyAnh_dataset/dataset_no_dip"
WITH_DIP = "XuLyAnh_dataset/dataset_with_dip"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ==================== DOC DU LIEU ====================

def get_data(path):
    data = []

    # Kiem tra bo du lieu con lai
    other_path = WITH_DIP if path == NO_DIP else NO_DIP

    for label, emotion in enumerate(CLASSES):
        folder = os.path.join(path, emotion)
        other_folder = os.path.join(other_path, emotion)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Khong tim thay: {folder}")

        if not os.path.isdir(other_folder):
            raise FileNotFoundError(f"Khong tim thay: {other_folder}")

        # Lay danh sach ten anh, sap xep tang dan
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        other_files = sorted(
            f for f in os.listdir(other_folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        # Dam bao No-DIP va With-DIP co cung ten anh
        if files != other_files:
            raise ValueError(
                f"Ten anh khong khop giua hai bo du lieu: {emotion}"
            )

        for file in files:
            data.append((os.path.join(folder, file), label))

    if len(data) == 0:
        raise ValueError("Dataset khong co anh!")

    return data


# ==================== CHIA DU LIEU ====================

def split_data(data):
    train, val, test = [], [], []

    for label, emotion in enumerate(CLASSES):

        # Lay danh sach ten anh cua tung lop
        files = [
            os.path.basename(path)
            for path, file_label in data
            if file_label == label
        ]

        files.sort()

        # Tron dung seed nhu Custom CNN va MobileNetV2
        random.Random(42 + label).shuffle(files)

        # Chia 70% - 15% - 15%
        a = int(len(files) * 0.7)
        b = a + int(len(files) * 0.15)

        folder = os.path.join(NO_DIP, emotion)

        train += [
            (os.path.join(folder, f), label)
            for f in files[:a]
        ]

        val += [
            (os.path.join(folder, f), label)
            for f in files[a:b]
        ]

        test += [
            (os.path.join(folder, f), label)
            for f in files[b:]
        ]

    return train, val, test


# ==================== DATASET ====================

class EmotionDataset(Dataset):
    def __init__(self, data, is_train=False):
        self.data = data

        self.transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            *([transforms.RandomHorizontalFlip()] if is_train else []),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, label = self.data[index]

        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))

        return image, label


# ==================== HUAN LUYEN ====================

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ==================== DANH GIA ====================

def evaluate(model, loader, folder):
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(1).cpu()

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred, labels=list(range(7)),
        average="macro", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print("\n========== KET QUA TEST ==========")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Macro F1: {macro_f1 * 100:.2f}%")
    print(report)

    with open(os.path.join(folder, "report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report)

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(7))
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Mini-Xception Confusion Matrix")
    plt.xticks(range(7), CLASSES, rotation=45)
    plt.yticks(range(7), CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(7):
        for j in range(7):
            plt.text(
                j, i, matrix[i, j],
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()

    return accuracy, macro_f1


# ==================== VE BIEU DO ====================

def save_plots(train_loss, val_loss, train_acc, val_acc, folder):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "training_history.png"))
    plt.close()


# ==================== CHUONG TRINH CHINH ====================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["no_dip", "with_dip"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("So epoch phai lon hon 0")

    # Co dinh random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Chon duong dan dataset
    dataset_path = NO_DIP if args.dataset == "no_dip" else WITH_DIP

    result_folder = os.path.join(
        "results", "minixception_" + args.dataset
    )

    os.makedirs("weights", exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # Doc va chia du lieu
    data = get_data(dataset_path)
    train_data, val_data, test_data = split_data(data)

    print("====================================")
    print("HUAN LUYEN MINI-XCEPTION")
    print("Dataset:", args.dataset)
    print("Device:", DEVICE)
    print("====================================")

    print("\nSo luong du lieu:")
    print("Train:", len(train_data))
    print("Validation:", len(val_data))
    print("Test:", len(test_data))

    # In so luong anh cua tung lop
    print("\nSO LUONG ANH THEO TUNG LOP:")

    for label, emotion in enumerate(CLASSES):
        total_class = sum(1 for x in data if x[1] == label)
        train_class = sum(1 for x in train_data if x[1] == label)
        val_class = sum(1 for x in val_data if x[1] == label)
        test_class = sum(1 for x in test_data if x[1] == label)

        print(
            f"{emotion}: Total={total_class} | "
            f"Train={train_class} | "
            f"Val={val_class} | "
            f"Test={test_class}"
        )

    # Tao DataLoader
    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        EmotionDataset(train_data, True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=pin
    )

    val_loader = DataLoader(
        EmotionDataset(val_data),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=pin
    )

    test_loader = DataLoader(
        EmotionDataset(test_data),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=pin
    )

    # Khoi tao mo hinh
    model = MiniXception().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    weight_path = os.path.join(
        "weights", f"best_minixception_{args.dataset}.pth"
    )

    # Huan luyen
    for epoch in range(args.epochs):

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # Luu mo hinh tot nhat
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print("Da luu mo hinh tot nhat!")

    # Nap checkpoint tot nhat
    if not os.path.exists(weight_path):
        raise FileNotFoundError("Khong tim thay trong so mo hinh!")

    model.load_state_dict(
        torch.load(
            weight_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    print("\nBest Validation Accuracy:", f"{best_acc * 100:.2f}%")

    # Luu bieu do
    save_plots(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        result_folder
    )

    # Danh gia tap Test
    accuracy, macro_f1 = evaluate(
        model, test_loader, result_folder
    )

    print("\n========== HOAN THANH ==========")
    print("Model:", weight_path)
    print("Test Accuracy:", accuracy)
    print("Test Macro F1:", macro_f1)
    print("Ket qua:", result_folder)


if __name__ == "__main__":
    main()
