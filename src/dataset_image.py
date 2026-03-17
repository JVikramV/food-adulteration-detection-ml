# src/dataset_image.py
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as T

class FoodImageDataset(Dataset):
    def __init__(self, csv_or_folder, transform=None):
        # csv_or_folder: if string path to CSV labels, else path to folder
        if os.path.isdir(csv_or_folder):
            # assume subfolders per class
            self.samples = []
            for cls in os.listdir(csv_or_folder):
                cls_dir = os.path.join(csv_or_folder, cls)
                if os.path.isdir(cls_dir):
                    for f in os.listdir(cls_dir):
                        self.samples.append((os.path.join(cls_dir,f), cls))
            self.df = None
        else:
            self.df = pd.read_csv(csv_or_folder)
            self.samples = list(zip(self.df['image_path'], self.df['label']))
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.class_to_idx = {c:i for i,c in enumerate(sorted({s[1] for s in self.samples}))}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label_idx = self.class_to_idx[label]
        return img, label_idx
