"""
Training & Fine-Tuning Script for Chest X-Ray Classification.

Fulfills Seminar 16 Requirements:
- Architecture: DenseNet-121 (pretrained on ImageNet)
- Baseline: ResNet-50 (pretrained on ImageNet)
- Multi-label loss: BCEWithLogitsLoss
- Transfer learning: Freeze early blocks, fine-tune denseblock3, 4 + classifier
- Metrics: Mean ROC-AUC, Mean Average Precision (mAP), Per-class sensitivity, Inference latency
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure backend root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.classification import CHEST_XRAY_CLASSES
from train.dataset import ChestXRayDataset, get_train_transforms, get_val_transforms


def build_densenet121(num_classes: int = 14, freeze_early_layers: bool = True):
    """Build DenseNet-121 with custom multi-label head."""
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    if freeze_early_layers:
        # Freeze initial conv and early denseblocks
        for name, param in model.named_parameters():
            if "denseblock3" not in name and "denseblock4" not in name and "classifier" not in name:
                param.requires_grad = False

    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def build_resnet50(num_classes: int = 14, freeze_early_layers: bool = True):
    """Build ResNet-50 baseline with custom multi-label head."""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    if freeze_early_layers:
        # Freeze initial conv and layer1, layer2
        for name, param in model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """Compute the 4 evaluation metrics required by Seminar 16."""
    roc_aucs = []
    aps = []
    sensitivities = []

    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred_probs[:, i]

        if len(np.unique(yt)) > 1:
            try:
                roc_aucs.append(roc_auc_score(yt, yp))
            except Exception:
                pass
            try:
                aps.append(average_precision_score(yt, yp))
            except Exception:
                pass

        yp_bin = (yp >= threshold).astype(int)
        tp = ((yp_bin == 1) & (yt == 1)).sum()
        fn = ((yp_bin == 0) & (yt == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivities.append(sens)

    return {
        "mean_roc_auc": float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "mean_sensitivity": float(np.mean(sensitivities)) if sensitivities else 0.0,
        "per_class_auc": {cls: float(roc_aucs[i]) if i < len(roc_aucs) else 0.0 for i, cls in enumerate(CHEST_XRAY_CLASSES)},
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    total_time_ms = (time.perf_counter() - start_time) * 1000
    avg_inference_ms = total_time_ms / len(loader.dataset)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / len(loader.dataset)
    metrics["avg_inference_time_ms"] = float(round(avg_inference_ms, 2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Seminar 16: Chest X-Ray Transfer Learning & Benchmarking")
    parser.add_argument("--csv", type=str, default="data/Data_Entry_2017.csv", help="Path to labels CSV")
    parser.add_argument("--img_dir", type=str, default="data/images", help="Path to images directory")
    parser.add_argument("--model", type=str, choices=["densenet", "resnet", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="models/weights", help="Directory to save weights")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Seminar 16: Chest X-Ray Disease Classification")
    print(f" Device: {device} | Target Classes: {len(CHEST_XRAY_CLASSES)}")
    print(f"=======================================================\n")

    # If dataset CSV doesn't exist, print instructions and exit gracefully
    if not os.path.exists(args.csv):
        print(f"[!] Dataset CSV not found at '{args.csv}'.")
        print("To run fine-tuning on ChestX-ray14:")
        print("1. Download ChestX-ray14 from NIH or Kaggle.")
        print("2. Place Data_Entry_2017.csv in backend/data/ and images in backend/data/images/")
        print("3. Re-run: python train/train_chest_xray.py --csv data/Data_Entry_2017.csv --img_dir data/images/")
        print("\nNote: The backend API already ships with pretrained ImageNet & ChestX-ray weights for instant demo!")
        return

    # Load dataset
    print("[+] Loading dataset...")
    dataset = ChestXRayDataset(args.csv, args.img_dir, is_train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.BCEWithLogitsLoss()

    models_to_train = ["densenet", "resnet"] if args.model == "both" else [args.model]
    results = {}

    for model_name in models_to_train:
        print(f"\n>>> Fine-tuning {model_name.upper()} with Transfer Learning...")
        if model_name == "densenet":
            model = build_densenet121().to(device)
            save_path = os.path.join(args.save_dir, "densenet121_chestxray.pth")
        else:
            model = build_resnet50().to(device)
            save_path = os.path.join(args.save_dir, "resnet50_chestxray.pth")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        best_auc = 0.0
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            epoch_time = time.time() - t0

            print(f"[{model_name.upper()}] Epoch {epoch+1:02d}/{args.epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                  f"Mean ROC-AUC: {val_metrics['mean_roc_auc']:.4f} | mAP: {val_metrics['mAP']:.4f} | "
                  f"Inference: {val_metrics['avg_inference_time_ms']}ms | Time: {epoch_time:.1f}s")

            if val_metrics["mean_roc_auc"] > best_auc:
                best_auc = val_metrics["mean_roc_auc"]
                torch.save(model.state_dict(), save_path)
                print(f"    --> Saved best model weights to {save_path}")

        results[model_name] = val_metrics

    # Save benchmark results
    benchmark_path = os.path.join(args.save_dir, "benchmark_summary.json")
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Benchmarking complete. Summary saved to {benchmark_path}")


if __name__ == "__main__":
    main()
