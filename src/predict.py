import torch
from torchvision import transforms, models
from PIL import Image
import os
import cv2
import numpy as np
from gradcam import GradCAM

MODEL_PATH = "models/chilli_model.pth"

classes = None  # will load automatically

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_model():
    global classes

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Load class names
    classes = sorted(os.listdir("data/processed_chilli/train"))

    return model


# -----------------------------
# ORIGINAL PREDICTION FUNCTION
# -----------------------------
def predict_image(image_path: str):
    model = load_model()
    img = Image.open(image_path)

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


# -----------------------------------
# HEATMAP SUPPORT (Grad-CAM)
# -----------------------------------

def apply_heatmap_on_image(img_path, cam):
    """Overlay heatmap on original image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.4 * heatmap + 0.6 * img).astype(np.uint8)
    return overlay


def predict_with_heatmap(image_path: str):
    """Return (predicted_class, heatmap_image)"""

    model = load_model()

    # pick ResNet50 last conv layer
    target_layer = model.layer4[2].conv3

    # Load image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    # Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    # Overlay
    heatmap_img = apply_heatmap_on_image(image_path, cam)

    # Normal prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    return label, heatmap_img


# CLI test
if __name__ == "__main__":
    path = input("Enter image path: ")
    cls, hm = predict_with_heatmap(path)
    print("Prediction:", cls)

