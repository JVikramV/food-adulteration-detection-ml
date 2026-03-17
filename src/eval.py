import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from models import get_model  # your custom model loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model_path="models/chilli_model.pth",
                   data_dir="data/processed_chilli/val",
                   batch_size=16):

    print(f"Using device: {DEVICE}")

    # -----------------------------
    # 1️⃣ TRANSFORMS
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # -----------------------------
    # 2️⃣ DATASET & DATALOADER
    # -----------------------------
    val_dataset = datasets.ImageFolder(data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    class_names = val_dataset.classes
    print("Classes:", class_names)

    # -----------------------------
    # 3️⃣ LOAD MODEL
    # -----------------------------
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    # -----------------------------
    # 4️⃣ EVALUATION LOOP
    # -----------------------------
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # -----------------------------
    # 5️⃣ METRICS
    # -----------------------------
    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=class_names)
    print(report)

    # Save to text file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_report.txt", "w") as f:
        f.write(report)

    # -----------------------------
    # 6️⃣ CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Chilli Adulteration Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    print("\nSaved confusion matrix to outputs/confusion_matrix.png")
    print("Saved evaluation report to outputs/eval_report.txt")

if __name__ == "__main__":
    evaluate_model()
