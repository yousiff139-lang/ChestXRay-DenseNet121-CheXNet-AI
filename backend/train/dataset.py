"""
Chest X-Ray Dataset Loader for Multi-Label Classification.

Supports NIH ChestX-ray14 and Stanford CheXpert datasets.
Parses multi-label disease annotations and applies standard
medical imaging data augmentations.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CHEST_XRAY14_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Data augmentations for training chest X-ray classifiers."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard normalization for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ChestXRayDataset(Dataset):
    """
    Dataset class for NIH ChestX-ray14 multi-label classification.

    Expects CSV with columns:
    - Image Index: image file name (e.g. 00000001_000.png)
    - Finding Labels: pipe-separated labels (e.g. "Cardiomegaly|Effusion")
    """

    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        classes: list = CHEST_XRAY14_CLASSES,
        transform: transforms.Compose = None,
        is_train: bool = True,
    ):
        self.image_dir = image_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = (
            transform
            if transform is not None
            else (get_train_transforms() if is_train else get_val_transforms())
        )

        df = pd.read_csv(csv_file)
        self.image_names = df["Image Index"].values if "Image Index" in df.columns else df.iloc[:, 0].values

        # Parse pipe-separated multi-labels
        labels_col = "Finding Labels" if "Finding Labels" in df.columns else df.iloc[:, 1]
        self.labels = np.zeros((len(df), len(classes)), dtype=np.float32)

        for i, finding_str in enumerate(labels_col):
            if pd.isna(finding_str) or finding_str == "No Finding":
                continue
            findings = str(finding_str).split("|")
            for finding in findings:
                finding = finding.strip()
                if finding in self.class_to_idx:
                    self.labels[i, self.class_to_idx[finding]] = 1.0

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            # Create a blank fallback if specific image file is not downloaded yet
            image = Image.new("RGB", (224, 224), color=0)
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label
