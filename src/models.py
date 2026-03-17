# src/models.py
import torch
import torch.nn as nn
import timm

class ImageBackbone(nn.Module):
    def __init__(self, model_name='mobilenetv2_100', pretrained=True, out_features=1280):
        super().__init__()
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.backbone = m
        feat_dim = m.num_features
        self.head = nn.Linear(feat_dim, 512)

    def forward(self, x):
        f = self.backbone.forward_features(x)  # [B, feat_dim]
        f = f.view(f.size(0), -1)
        f = self.head(f)
        return f  # [B,512]

class SpectraMLP(nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Linear(hid, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class FusionNet(nn.Module):
    def __init__(self, img_backbone:ImageBackbone, spectra_in_dim:int, num_classes=2, regression=False):
        super().__init__()
        self.img = img_backbone
        self.spec = SpectraMLP(spectra_in_dim, hid=128, out_dim=128)
        self.fc = nn.Sequential(
            nn.Linear(512+128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes if not regression else 1)
        )
        self.regression = regression

    def forward(self, img, spec):
        fi = self.img(img)
        fs = self.spec(spec)
        x = torch.cat([fi, fs], dim=1)
        out = self.fc(x)
        if self.regression:
            return out.squeeze(1)
        return out
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Loads a pretrained ResNet18 and replaces the final layer 
    with your number of classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
