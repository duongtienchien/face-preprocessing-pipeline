
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models.mobilenet_model import MobileNetV2FER


# Cau hinh
SIZE = 112
BATCH = 16
LR = 0.0001
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NO_DIP = "XuLyAnh_dataset/dataset_no_dip"
WITH_DIP = "XuLyAnh_dataset/dataset_with_dip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Chia du lieu 70% - 15% - 15%
def split_data():
    train, val, test = [], [], []

    for label, emotion in enumerate(EMOTIONS):
        folder = os.path.join(NO_DIP, emotion)
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))

        other = sorted(f for f in os.listdir(os.path.join(WITH_DIP, emotion))
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))

        if files != other:
            raise ValueError(f"Ten anh khong khop: {emotion}")

        random.Random(42 + label).shuffle(files)
        a = int(len(files) * 0.7)
        b = a + int(len(files) * 0.15)

        train += [(emotion, f, label) for f in files[:a]]
        val += [(emotion, f, label) for f in files[a:b]]
        test += [(emotion, f, label) for f in files[b:]]

    return train, val, test


# Doc va xu ly anh
class EmotionDataset(Dataset):
    def __init__(self, files, root, training=False):
        self.files = files
        self.root = root

        steps = [transforms.Resize((SIZE, SIZE))]
        if training:
            steps.append(transforms.RandomHorizontalFlip())

        steps += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        emotion, filename, label = self.files[i]
        path = os.path.join(self.root, emotion, filename)
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


# Huan luyen hoac kiem tra mot Epoch
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

    return loss_sum / total, correct / total


# Danh gia tap Test
def evaluate(model, loader):
    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predicted.extend(outputs.argmax(1).cpu().numpy())
            actual.extend(labels.numpy())

    acc = accuracy_score(actual, predicted)
    macro_f1 = classification_report(
        actual, predicted, labels=range(7),
        target_names=EMOTIONS, output_dict=True, zero_division=0
    )["macro avg"]["f1-score"]

    print("\nAccuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print(classification_report(
        actual, predicted, labels=range(7),
        target_names=EMOTIONS, zero_division=0
    ))

    return acc, macro_f1, confusion_matrix(actual, predicted, labels=range(7))


# Luu bieu do
def save_plots(history, cm, folder):
    os.makedirs(folder, exist_ok=True)

    for key, title in [("loss", "Loss"), ("acc", "Accuracy")]:
        plt.figure()
        plt.plot(history[f"train_{key}"], label="Train")
        plt.plot(history[f"val_{key}"], label="Validation")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(f"{folder}/{key}.png")
        plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(7), EMOTIONS, rotation=45)
    plt.yticks(range(7), EMOTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(7):
        for j in range(7):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{folder}/confusion_matrix.png")
    plt.close()


# Chuong trinh chinh
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["no_dip", "with_dip"], default="no_dip")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epoch phai lon hon 0")

    root = NO_DIP if args.dataset == "no_dip" else WITH_DIP
    train, val, test = split_data()

    print("Dataset:", args.dataset, "| Device:", DEVICE)
    print("Train:", len(train), "| Validation:", len(val), "| Test:", len(test))

    train_loader = DataLoader(EmotionDataset(train, root, True),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(EmotionDataset(val, root),
                            batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(EmotionDataset(test, root),
                             batch_size=BATCH, shuffle=False, num_workers=0)

    model = MobileNetV2FER(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    weight = f"weights/best_mobilenet_{args.dataset}.pth"
    best_acc = 0

    os.makedirs("weights", exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        for key, value in zip(history, [train_loss, val_loss, train_acc, val_acc]):
            history[key].append(value)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight)
            print("Da luu mo hinh tot nhat!")

    model.load_state_dict(torch.load(weight, map_location=DEVICE, weights_only=True))
    print("\nBest Validation Accuracy:", best_acc)

    acc, f1, cm = evaluate(model, test_loader)
    save_plots(history, cm, f"results/mobilenet_{args.dataset}")

    print("\nHOAN THANH")
    print("Model:", weight)
    print("Test Accuracy:", acc)
    print("Test Macro F1:", f1)


if __name__ == "__main__":
    main()
