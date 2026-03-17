import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MilkSpectraDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # Label mapping
        self.labels = df["Ingredient"].astype("category")
        self.label_to_idx = {cat: idx for idx, cat in enumerate(self.labels.cat.categories)}
        self.idx_to_label = {idx: cat for cat, idx in self.label_to_idx.items()}
        self.labels = self.labels.map(self.label_to_idx)

        # Spectral columns
        self.feature_cols = [c for c in df.columns if c.startswith("SPC")]

        # Convert to float
        X = df[self.feature_cols].values.astype("float32")

        # -------- CRITICAL: SCALE FEATURES ----------
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X = X_scaled.astype("float32")
        self.y = self.labels.values.astype("int64")

        print(f"[INFO] Loaded dataset: {len(self.X)} samples")
        print(f"[INFO] Features: {len(self.feature_cols)} scaled")
        print(f"[INFO] Classes: {self.label_to_idx}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
