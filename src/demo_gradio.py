import gradio as gr
from predict import predict_image, predict_with_heatmap   # chilli prediction
from predict_milk import predict_milk, model, scaler, feature_cols   # milk prediction
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ---------------- CHILLI ----------------
def classify_chilli_with_heatmap(img_path):
    try:
        label, heatmap = predict_with_heatmap(img_path)
        return label, heatmap
    except Exception as e:
        return f"Error: {e}", None


# ---------------- MILK (UPGRADED UI) ----------------
def classify_milk_ui(csv_file):
    try:
        # Load CSV
        df = pd.read_csv(csv_file.name, encoding="latin1")

        # Preview first 10 rows
        preview = df.head(10)

        # Extract spectral columns
        spc_cols = [c for c in df.columns if c.startswith("SPC")]
        if len(spc_cols) == 0:
            raise ValueError("No SPC columns found.")

        # Plot spectral curve (first row)
        x = np.arange(len(spc_cols))
        y = df[spc_cols].iloc[0].values

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Milk Spectral Curve (Sample 1)")
        ax.set_xlabel("SPC Index")
        ax.set_ylabel("Intensity")

        # Run prediction
        label = predict_milk(csv_file)

        # Confidence (softmax on first row)
        X = df[feature_cols].astype("float32").values
        Xs = scaler.transform(X)
        X_tensor = torch.tensor(Xs[0:1], dtype=torch.float32)

        probs = F.softmax(model(X_tensor), dim=1)
        confidence = float(probs.max().item())

        return preview, fig, label, f"{confidence*100:.2f}%"

    except Exception as e:
        return pd.DataFrame([["Error", str(e)]]), None, "Error", "N/A"


# ---------------- UI ----------------
with gr.Blocks(title="Food Adulteration Detection") as demo:
    gr.Markdown("# 🍲 Food Adulteration Detection System")

    with gr.Tabs():
        
        # ======== CHILLI TAB =========
        with gr.Tab("🌶️ Chilli Powder"):
            gr.Markdown("Upload an image of chilli powder")

            img_in = gr.Image(type="filepath", label="Input Chilli Image")

            with gr.Row():
                pred_out = gr.Textbox(label="Prediction")
                heatmap_out = gr.Image(label="Grad-CAM Heatmap")

            gr.Button("Predict Chilli").click(
                classify_chilli_with_heatmap,
                inputs=img_in,
                outputs=[pred_out, heatmap_out]
            )

        # ======== MILK TAB (UPDATED) =========
        with gr.Tab("🥛 Milk Sample (CSV)"):
            gr.Markdown("Upload a CSV containing spectral values (SPC columns)")

            csv_in = gr.File(label="Milk CSV File")

            with gr.Row():
                csv_preview = gr.Dataframe(label="CSV Preview (First 10 Rows)")
                spectrum_plot = gr.Plot(label="Spectral Curve")

            pred_out = gr.Textbox(label="Prediction")
            conf_out = gr.Textbox(label="Confidence (%)")

            gr.Button("Predict Milk").click(
                classify_milk_ui,
                inputs=csv_in,
                outputs=[csv_preview, spectrum_plot, pred_out, conf_out]
            )

demo.launch()
