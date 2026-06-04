"""
02_speech_training.py

Train a speech scene classifier using log-mel spectrograms.

Labels:
  no_speech       — silence / non-speech segments
  single_speaker  — one active speaker
  multi_speaker   — two or more concurrent speakers

V2 upgrades (matching environment model):
  - SE2d channel attention blocks
  - Residual skip connections
  - Focal loss
  - Random hyperparameter search

Input:  data/speech/processed/features/<clip_id>.npy  (128, T) log-mel
Output: output/speech/best_speech_model.pt
"""

import json
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_SPEECH_DIR = PROJECT_ROOT / "data" / "speech"
OUTPUT_DIR      = PROJECT_ROOT / "output" / "speech"

MANIFEST_PATH = DATA_SPEECH_DIR / "manifests" / "speech_manifest.csv"
FEAT_DIR      = DATA_SPEECH_DIR / "processed" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed settings ────────────────────────────────────────────────────────────

NUM_EPOCHS   = 25
VAL_FRACTION = 0.15
USE_AMP      = True
NUM_WORKERS  = 4
PIN_MEMORY   = True
PRINT_EVERY  = 20
N_TRIALS     = 8

LABEL_NAMES  = ["no_speech", "single_speaker", "multi_speaker"]
LABEL_TO_ID  = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL  = {i: name for name, i in LABEL_TO_ID.items()}
NUM_CLASSES  = len(LABEL_NAMES)

# Log-mel shape
N_MELS = 128
# T varies by clip length; we'll crop/pad to fixed width
TARGET_FRAMES = 313   # ~10s at hop=512, sr=16000: floor(10*16000/512)+1 = 313

# ── Hyperparameter search space ───────────────────────────────────────────────

HPARAM_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "batch_size":    [64, 128],
    "dropout":       [0.3, 0.4, 0.5],
    "focal_gamma":   [1.0, 2.0],
    "se_reduction":  [8, 16],
    "base_channels": [32, 64],
}


def sample_hparams() -> dict:
    return {k: random.choice(v) for k, v in HPARAM_SPACE.items()}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, using CPU.")
        return torch.device("cpu")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.zeros(1).cuda()
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"[WARN] GPU not usable ({e}), falling back to CPU.")
        return torch.device("cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_index() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    print(f"[INFO] Manifest rows: {len(df)}")

    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df = df[df["label_id"].notna()].copy()
    df["label_id"] = df["label_id"].astype(int)

    # Build feature paths
    df["feature_path"] = df["clip_id"].apply(lambda x: str(FEAT_DIR / f"{x}.npy"))

    # Filter to existing features
    before = len(df)
    df = df[df["feature_path"].apply(os.path.isfile)].reset_index(drop=True)
    print(f"[INFO] Rows with features: {len(df)}/{before}")

    print("\n[INFO] Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    return df[["feature_path", "label_id", "label"]]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label_id = int(row["label_id"])

        feat = np.load(row["feature_path"]).astype(np.float32)  # (128, T)

        # Pad or trim time axis to TARGET_FRAMES
        T = feat.shape[1]
        if T < TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, TARGET_FRAMES - T)))
        else:
            feat = feat[:, :TARGET_FRAMES]

        # SpecAugment-style augmentation
        if self.augment:
            # Time masking — mask up to 30 frames
            if np.random.rand() < 0.5:
                t0 = np.random.randint(0, TARGET_FRAMES - 30)
                feat[:, t0:t0 + np.random.randint(1, 30)] = feat.min()
            # Frequency masking — mask up to 20 mel bands
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, N_MELS - 20)
                feat[f0:f0 + np.random.randint(1, 20), :] = feat.min()

        # Add channel dim for Conv2d: (1, 128, T)
        x = torch.from_numpy(feat).unsqueeze(0)
        return x, torch.tensor(label_id, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class SE2d(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps (B, C, H, W)."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = x.mean(dim=[2, 3])         # (B, C)
        s = self.fc(s)                 # (B, C)
        return x * s[:, :, None, None]


class ResConv2dBlock(nn.Module):
    """Residual Conv2d block with SE attention."""
    def __init__(self, channels: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SE2d(channels, reduction=se_reduction)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class SpeechCNN(nn.Module):
    """
    2D CNN over log-mel spectrograms for speech scene classification.
    Input: (B, 1, 128, T)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 64,
        dropout: float = 0.4,
        se_reduction: int = 8,
    ):
        super().__init__()

        # Stem: (B, 1, 128, T) -> (B, base_channels, 64, T//2)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Residual blocks with SE
        self.block1 = ResConv2dBlock(base_channels, se_reduction)

        # Downsample + widen
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block2 = ResConv2dBlock(base_channels * 2, se_reduction)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)    # (B, C, 64, T//2)
        x = self.block1(x)  # residual + SE
        x = self.down1(x)   # (B, 2C, 32, T//4)
        x = self.block2(x)  # residual + SE
        x = self.down2(x)   # (B, 4C, 16, T//8)
        x = self.gap(x)     # (B, 4C, 1, 1)
        x = x.flatten(1)    # (B, 4C)
        x = self.drop(x)
        return self.fc(x)   # (B, num_classes)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts  = np.bincount(df["label_id"].values, minlength=NUM_CLASSES)
    total   = counts.sum()
    weights = total / (counts + 1e-6)
    weights = weights / weights.mean()
    print(f"[INFO] Class counts:  {counts}")
    print(f"[INFO] Class weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"  Train {epoch+1}", leave=False)
    for i, (x, y) in enumerate(pbar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        if i % PRINT_EVERY == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    all_logits, all_targets = [], []
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        running_loss += criterion(logits, y).item()
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    if not all_logits:
        return 0.0, 0.0, 0.0
    preds      = torch.cat(all_logits).argmax(1).numpy()
    targets_np = torch.cat(all_targets).numpy()
    return (
        running_loss / len(loader),
        accuracy_score(targets_np, preds),
        f1_score(targets_np, preds, average="macro"),
    )


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(train_df, val_df, hparams, device, trial_idx) -> Tuple[float, str]:
    batch_size    = hparams["batch_size"]
    lr            = hparams["learning_rate"]
    dropout       = hparams["dropout"]
    focal_gamma   = hparams["focal_gamma"]
    se_reduction  = hparams["se_reduction"]
    base_channels = hparams["base_channels"]

    train_ds = SpeechDataset(train_df, augment=True)
    val_ds   = SpeechDataset(val_df,   augment=False)

    class_weights = compute_class_weights(train_df).to(device)
    criterion     = FocalLoss(gamma=focal_gamma, weight=class_weights)

    counts         = np.bincount(train_df["label_id"].values, minlength=NUM_CLASSES)
    sample_weights = 1.0 / (counts[train_df["label_id"].values] + 1e-6)
    sampler        = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model     = SpeechCNN(base_channels=base_channels, dropout=dropout,
                          se_reduction=se_reduction).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_f1   = -1.0
    ckpt_path = str(OUTPUT_DIR / f"trial_{trial_idx}_best.pt")

    for epoch in range(NUM_EPOCHS):
        train_loss            = train_one_epoch(model, train_loader, optimizer, device,
                                                criterion, scaler, epoch)
        val_loss, val_acc, f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
                "hparams": hparams,
            }, ckpt_path)

    return best_f1, ckpt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[INFO] Device: {device}")

    df = load_index()
    train_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=42,
        stratify=df["label_id"],
    )
    print(f"[INFO] Train: {len(train_df)}  Val: {len(val_df)}")

    search_results = []
    for trial in range(N_TRIALS):
        hparams = sample_hparams()
        print(f"\n{'='*60}\nTRIAL {trial+1}/{N_TRIALS}  hparams={hparams}\n{'-'*60}")
        best_f1, ckpt_path = run_trial(train_df, val_df, hparams, device, trial)
        search_results.append({"trial": trial+1, "best_macro_f1": best_f1,
                                "checkpoint": ckpt_path, **hparams})
        print(f"  Trial {trial+1} best macro_f1: {best_f1:.4f}")

    results_df = pd.DataFrame(search_results).sort_values("best_macro_f1", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "hparam_search_results.csv", index=False)

    best_row  = results_df.iloc[0]
    best_ckpt = best_row["checkpoint"]
    print(f"\n{'='*60}")
    print(f"[INFO] Best trial: {int(best_row['trial'])}  macro_f1={best_row['best_macro_f1']:.4f}")
    print(results_df[["trial","best_macro_f1","learning_rate","batch_size",
                       "dropout","focal_gamma","se_reduction","base_channels"]].to_string(index=False))

    final_path = OUTPUT_DIR / "best_speech_model.pt"
    shutil.copy(best_ckpt, final_path)
    print(f"[INFO] Best model -> {final_path}")

    # Final report
    ckpt  = torch.load(final_path, map_location=device)
    model = SpeechCNN(
        base_channels=int(best_row["base_channels"]),
        dropout=float(best_row["dropout"]),
        se_reduction=int(best_row["se_reduction"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds     = SpeechDataset(val_df, augment=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(model(x.to(device)).cpu())
            all_targets.append(y)

    preds      = torch.cat(all_logits).argmax(1).numpy()
    targets_np = torch.cat(all_targets).numpy()
    report     = classification_report(targets_np, preds,
                                       target_names=LABEL_NAMES, digits=4)

    report_path = OUTPUT_DIR / "speech_classification_report.txt"
    report_path.write_text(report)
    print(f"\n[INFO] Classification report -> {report_path}")
    print(report)

    with open(OUTPUT_DIR / "speech_training_config.json", "w") as f:
        json.dump({
            "num_epochs": NUM_EPOCHS, "val_fraction": VAL_FRACTION,
            "num_classes": NUM_CLASSES, "label_names": LABEL_NAMES,
            "best_hparams": {k: float(best_row[k]) if hasattr(best_row[k], "item")
                             else best_row[k] for k in HPARAM_SPACE},
            "best_macro_f1": float(best_row["best_macro_f1"]),
        }, f, indent=2)


if __name__ == "__main__":
    main()
