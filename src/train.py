import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

DATA_DIR = "data/processed_chilli"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001

# -----------------------------
# 1. Data transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# -----------------------------
# 2. Load pretrained model
# -----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 3. Training loop
# -----------------------------
print("Training started...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            _, predicted = torch.max(preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Val Accuracy: {acc:.2f}%")

# -----------------------------
# 4. Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/chilli_model.pth")

print("\n🎉 Training complete! Model saved to models/chilli_model.pth")
print("Classes:", train_data.classes)
