
import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.cnn_model import CustomCNN


# ==================== CAU HINH ====================

IMAGE_SIZE = 112
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001

EMOTIONS = ["angry", "disgust", "fear", "happy",
            "neutral", "sad", "surprise"]

DATASET_NO_DIP = "XuLyAnh_dataset/dataset_no_dip"
DATASET_WITH_DIP = "XuLyAnh_dataset/dataset_with_dip"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==================== CHIA DU LIEU ====================

def split_dataset():
    train_files, val_files, test_files = [], [], []

    for label, emotion in enumerate(EMOTIONS):
        folder_no = os.path.join(DATASET_NO_DIP, emotion)
        folder_yes = os.path.join(DATASET_WITH_DIP, emotion)

        if not os.path.isdir(folder_no) or not os.path.isdir(folder_yes):
            raise FileNotFoundError(f"Khong tim thay thu muc: {emotion}")

        files_no = sorted(
            f for f in os.listdir(folder_no)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        files_yes = sorted(
            f for f in os.listdir(folder_yes)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        if files_no != files_yes:
            raise ValueError(f"Ten anh hai bo du lieu khong khop: {emotion}")

        random.Random(42 + label).shuffle(files_no)

        n = len(files_no)
        train_end = int(n * 0.70)
        val_end = train_end + int(n * 0.15)

        train_files.extend(
            (emotion, f, label) for f in files_no[:train_end]
        )
        val_files.extend(
            (emotion, f, label) for f in files_no[train_end:val_end]
        )
        test_files.extend(
            (emotion, f, label) for f in files_no[val_end:]
        )

    return train_files, val_files, test_files


# ==================== DATASET ====================

class KDEF_Dataset(Dataset):
    def __init__(self, files, dataset_root, is_train=False):
        self.files = files
        self.dataset_root = dataset_root

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *([transforms.RandomHorizontalFlip(p=0.5)] if is_train else []),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        emotion, filename, label = self.files[index]

        path = os.path.join(self.dataset_root, emotion, filename)
        image = Image.open(path).convert("RGB")

        return self.transform(image), label


# ==================== TRAIN / VALIDATION ====================

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0
    total_correct = 0
    total_images = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_images += images.size(0)

    return total_loss / total_images, total_correct / total_images


# ==================== DANH GIA TEST ====================

def evaluate_test(model, loader):
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = np.mean(all_labels == all_predictions)

    cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=int)
    np.add.at(cm, (all_labels, all_predictions), 1)

    precision_list, recall_list, f1_list = [], [], []

    for i in range(len(EMOTIONS)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_f1 = np.mean(f1_list)

    print("\n========== KET QUA TEST ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nChi tiet tung lop:")
    for i, emotion in enumerate(EMOTIONS):
        print(
            f"{emotion:10s} "
            f"Precision: {precision_list[i]:.4f} "
            f"Recall: {recall_list[i]:.4f} "
            f"F1: {f1_list[i]:.4f}"
        )

    return accuracy, macro_f1, cm


# ==================== LUU BIEU DO ====================

def save_results(history, cm, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Loss va Accuracy
    for key, title, filename in [
        ("loss", "Training and Validation Loss", "loss.png"),
        ("acc", "Training and Validation Accuracy", "accuracy.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history[f"train_{key}"], label="Train")
        plt.plot(history[f"val_{key}"], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(key.upper())
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    # Confusion Matrix
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)

    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

    print("\nDa luu bieu do tai:", output_folder)


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["no_dip", "with_dip"],
                        default="no_dip")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("So epoch phai lon hon 0")

    dataset_root = (
        DATASET_NO_DIP if args.dataset == "no_dip"
        else DATASET_WITH_DIP
    )

    print("====================================")
    print("HUAN LUYEN CUSTOM CNN")
    print("Dataset:", args.dataset)
    print("Device:", DEVICE)
    print("Epochs:", args.epochs)
    print("====================================")

    train_files, val_files, test_files = split_dataset()

    print("\nSo luong du lieu:")
    print("Train:", len(train_files))
    print("Validation:", len(val_files))
    print("Test:", len(test_files))

    train_data = KDEF_Dataset(train_files, dataset_root, True)
    val_data = KDEF_Dataset(val_files, dataset_root)
    test_data = KDEF_Dataset(test_files, dataset_root)

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )

    model = CustomCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    os.makedirs("weights", exist_ok=True)

    weight_path = f"weights/best_custom_cnn_{args.dataset}.pth"
    result_folder = f"results/custom_cnn_{args.dataset}"

    best_val_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weight_path)
            print("Da luu mo hinh tot nhat!")

    model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE, weights_only=True)
    )

    print("\nBest Validation Accuracy:", best_val_acc)

    test_acc, macro_f1, cm = evaluate_test(model, test_loader)
    save_results(history, cm, result_folder)

    print("\n========== HOAN THANH ==========")
    print("Model:", weight_path)
    print("Test Accuracy:", test_acc)
    print("Test Macro F1:", macro_f1)


if __name__ == "__main__":
    main()
