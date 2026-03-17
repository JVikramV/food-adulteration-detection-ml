# src/train_milk.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np

CSV_PATH = os.path.abspath(r"E:\food-adulteration-proto\data\raw\milk\milk quality.csv")
# If you prefer the uploaded file path during testing, use "/mnt/data/milk quality.csv"

# -------------------------
# Dataset helper (simple)
# -------------------------
class MilkSpectraDataset(torch.utils.data.Dataset):
    def __init__(self, df, features, labels):
        self.X = df[features].astype("float32").values
        self.y = labels.astype("int64").values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# -------------------------
# Load dataframe & prepare
# -------------------------
df = pd.read_csv(CSV_PATH, encoding="latin1")  # using latin1 if utf-8 errors
# Label column
label_col = "Ingredient"

# Identify spectral columns (those starting with "SPC")
feature_cols = [c for c in df.columns if str(c).startswith("SPC")]
if len(feature_cols) == 0:
    raise RuntimeError("No spectral columns (SPC...) found in CSV. Check column names.")

# Map labels to indices
labels_cat = df[label_col].astype("category")
label_to_idx = {cat: idx for idx, cat in enumerate(labels_cat.cat.categories)}
idx_to_label = {v: k for k, v in label_to_idx.items()}
y = labels_cat.map(label_to_idx)

# Split and scale
X = df[feature_cols].astype("float32").values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and feature list for later
os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/milk_scaler.pkl")
joblib.dump(feature_cols, "artifacts/milk_features.pkl")
joblib.dump(label_to_idx, "artifacts/milk_label_to_idx.pkl")
joblib.dump(idx_to_label, "artifacts/milk_idx_to_label.pkl")
print("[INFO] Saved scaler, feature list and label maps to artifacts/")

# Train / val split
train_idx, val_idx = train_test_split(
    np.arange(len(X_scaled)), test_size=0.2, random_state=42, stratify=y
)

train_df = pd.DataFrame(X_scaled[train_idx], columns=feature_cols)
val_df = pd.DataFrame(X_scaled[val_idx], columns=feature_cols)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_val = y.iloc[val_idx].reset_index(drop=True)

train_dataset = MilkSpectraDataset(train_df, feature_cols, y_train)
val_dataset = MilkSpectraDataset(val_df, feature_cols, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------
# Model
# -------------------------
input_dim = len(feature_cols)
num_classes = len(label_to_idx)

class MilkNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MilkNet(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# -------------------------
# Training
# -------------------------
EPOCHS = 40
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            preds = model(Xb)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "artifacts/milk_model.pth")

# Save final artifacts
torch.save(model.state_dict(), "artifacts/milk_model_final.pth")
joblib.dump(scaler, "artifacts/milk_scaler.pkl")  # re-save to be safe
joblib.dump(feature_cols, "artifacts/milk_features.pkl")
joblib.dump(label_to_idx, "artifacts/milk_label_to_idx.pkl")
joblib.dump(idx_to_label, "artifacts/milk_idx_to_label.pkl")
print(f"Model saved as artifacts/milk_model_final.pth (best val acc {best_acc:.4f})")
