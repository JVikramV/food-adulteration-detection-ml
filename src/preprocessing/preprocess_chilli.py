import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw/chilli/Dataset for Adulterated Red Chilli Powder with Brick Powder"
OUTPUT_DIR = "data/processed_chilli"
IMAGE_SIZE = 224

# ---------------------------------------------
# 1. Detect adulteration class from folder name
# ---------------------------------------------
def get_class_from_path(path):
    name = path.lower()

    if "normal" in name or "pure" in name:
        return "normal"

    if "brick only" in name or "_brick" in name:
        return "brick_only"

    # Detect "5_", "10_", "15_" type adulteration
    for pct in ["5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]:
        if f"{pct}_" in name or f"{pct} %" in name or f"{pct} " in name:
            return f"adulterated_{pct}"

    return None


# ---------------------------------------------------
# 2. Collect all JPEG/PNG images recursively
# ---------------------------------------------------
def collect_all_images():
    image_paths = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(("jpg", "jpeg", "png")):
                full = os.path.join(root, f)
                image_paths.append(full)
    return image_paths


# ---------------------------------------------------
# 3. Preprocess, resize, rename + move images
# ---------------------------------------------------
def preprocess_and_save(images):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_path in images:
        img_class = get_class_from_path(img_path)

        if img_class is None:
            print("Skipping (no class detected):", img_path)
            continue

        class_dir = os.path.join(OUTPUT_DIR, img_class)
        os.makedirs(class_dir, exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            print("Corrupted:", img_path)
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        new_name = img_class + "_" + str(len(os.listdir(class_dir)) + 1) + ".jpg"
        save_path = os.path.join(class_dir, new_name)
        cv2.imwrite(save_path, img)


# ---------------------------------------------------
# 4. Create train/val split
# ---------------------------------------------------
def split_dataset():
    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    classes = [
        d for d in os.listdir(OUTPUT_DIR)
        if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d not in ["train", "val"]
    ]

    for cls in classes:
        full_class_path = os.path.join(OUTPUT_DIR, cls)
        images = os.listdir(full_class_path)
        images = [os.path.join(full_class_path, img) for img in images]

        # NEW FIX : skip small classes and continue
        if len(images) < 5:
            print(f"⚠ Skipping class '{cls}' — only {len(images)} samples.")
            shutil.rmtree(full_class_path)  # delete small unusable class
            continue   # <-- THIS WAS MISSING

        # Safe splitting
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        # create class folders inside train/ and val/
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        for img in train_imgs:
            shutil.move(img, os.path.join(train_class_dir, os.path.basename(img)))

        for img in val_imgs:
            shutil.move(img, os.path.join(val_class_dir, os.path.basename(img)))

        # delete original class folder after moving
        shutil.rmtree(full_class_path)

    print("\n✅ Dataset successfully split into train/ and val/!")



# ---------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------
if __name__ == "__main__":
    print("\n📌 Collecting images...")
    images = collect_all_images()
    print(f"Found {len(images)} total images")

    print("\n📌 Preprocessing + resizing + organizing...")
    preprocess_and_save(images)

    print("\n📌 Splitting into train/val...")
    split_dataset()

    print("\n🎉 DONE! Processed dataset saved at:", OUTPUT_DIR)
