# src/predict_milk.py
import os
import joblib
import torch
import pandas as pd
import numpy as np

ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))


# Load artifacts once
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "milk_scaler.pkl"))
feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "milk_features.pkl"))
idx_to_label = joblib.load(os.path.join(ARTIFACT_DIR, "milk_idx_to_label.pkl"))

# Build same model architecture and load weights
import torch.nn as nn
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

# instantiate model and load weights
num_classes = len(idx_to_label)
input_dim = len(feature_cols)
model = MilkNet(input_dim, num_classes)
model_path = os.path.join(ARTIFACT_DIR, "milk_model_final.pth")
if not os.path.exists(model_path):
    model_path = os.path.join(ARTIFACT_DIR, "milk_model.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

def _prepare_df_for_prediction(df):
    """
    Accepts a pandas DataFrame (possibly containing extra columns).
    Returns a NumPy array of shape (n_samples, n_features) ordered as feature_cols.
    """
    # Ensure all expected feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required spectral columns in uploaded CSV: {missing[:5]} ... (total {len(missing)})")

    X = df[feature_cols].astype("float32").values
    Xs = scaler.transform(X)
    return Xs

def predict_milk(file_obj):
    """
    file_obj: a gradio File-like object (has .name) or a path string
    Returns: list of predicted label names (one per row)
    """
    # read the csv
    if hasattr(file_obj, "name"):
        csv_path = file_obj.name
    else:
        csv_path = str(file_obj)

    df = pd.read_csv(csv_path, encoding="latin1")
    Xs = _prepare_df_for_prediction(df)

    with torch.no_grad():
        X_tensor = torch.tensor(Xs, dtype=torch.float32)
        outputs = model(X_tensor)
        _, pred_idx = outputs.max(1)
        preds = [idx_to_label[int(i)] for i in pred_idx]

    # If single-row input, return single string
    if len(preds) == 1:
        return preds[0]
    return preds
