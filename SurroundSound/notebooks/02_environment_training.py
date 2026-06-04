"""
02_environment_training.py

Train a single-label environment classifier using precomputed VGGish embeddings.

V2 upgrades:
- Squeeze-and-Excitation (SE) channel attention blocks (1D)
- Residual skip connections
- Focal loss for class imbalance
- Random search hyperparameter tuning

Input shape: (10, 128) -> transposed to (128, 10) for Conv1d
Output: 8-class softmax
"""

import json
import os
import random
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


def get_device() -> torch.device:
    """Try CUDA, fall back to CPU gracefully if sm_120 or other issues."""
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, using CPU.")
        return torch.device("cpu")
    try:
        # Suppress the sm_120 incompatibility warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.zeros(1).cuda()
            del t
        print(f"[INFO] GPU available and functional: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"[WARN] GPU detected but not usable ({e}). Falling back to CPU.")
        return torch.device("cpu")

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_ENV_DIR   = PROJECT_ROOT / "data" / "environment"
OUTPUT_ENV_DIR = PROJECT_ROOT / "output" / "environment"

INDEX_PATH_PARQUET = DATA_ENV_DIR / "data_index.parquet"
INDEX_PATH_CSV     = DATA_ENV_DIR / "data_index.csv"
LABEL_TO_ID_PATH   = DATA_ENV_DIR / "label_to_id.json"
ID_TO_LABEL_PATH   = DATA_ENV_DIR / "id_to_label.json"

OUTPUT_ENV_DIR.mkdir(parents=True, exist_ok=True)

# Fixed training settings
VAL_FRACTION = 0.15
NUM_EPOCHS   = 25
USE_AMP      = True
NUM_WORKERS  = 4
PIN_MEMORY   = True
PRINT_EVERY  = 20

# VGGish embedding shape
N_FRAMES  = 10
EMBED_DIM = 128

# ── Hyperparameter search space ───────────────────────────────────────────────

HPARAM_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "batch_size":    [64, 128, 256],
    "dropout":       [0.3, 0.4, 0.5],
    "focal_gamma":   [1.0, 2.0, 3.0],
    "se_reduction":  [4, 8, 16],
    "base_channels": [128, 256, 512],
}

N_SEARCH_TRIALS = 8   # number of random configs to try


def print_config(hparams: dict):
    print("CONFIG:")
    print(f"  PROJECT_ROOT   = {PROJECT_ROOT}")
    print(f"  NUM_EPOCHS     = {NUM_EPOCHS}")
    print(f"  VAL_FRACTION   = {VAL_FRACTION}")
    print(f"  N_TRIALS       = {N_SEARCH_TRIALS}")
    print(f"  HPARAMS        = {hparams}")
    print("-" * 60)


def sample_hparams() -> dict:
    return {k: random.choice(v) for k, v in HPARAM_SPACE.items()}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_label_dicts() -> Tuple[dict, dict]:
    with open(LABEL_TO_ID_PATH) as f:
        label_to_id = json.load(f)
    with open(ID_TO_LABEL_PATH) as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    return label_to_id, id_to_label


def load_environment_index() -> Tuple[pd.DataFrame, dict, dict]:
    if INDEX_PATH_PARQUET.exists():
        df = pd.read_parquet(INDEX_PATH_PARQUET)
        print(f"[INFO] Loaded parquet: {df.shape}")
    elif INDEX_PATH_CSV.exists():
        df = pd.read_csv(INDEX_PATH_CSV)
        print(f"[INFO] Loaded CSV: {df.shape}")
    else:
        raise RuntimeError(f"No data index found.")

    label_to_id, id_to_label = load_label_dicts()
    num_classes = len(label_to_id)

    if "y" in df.columns:
        df["label_id"] = df["y"].astype(int)
    elif "primary_label" in df.columns:
        df["label_id"] = df["primary_label"].map(label_to_id)
    else:
        raise RuntimeError("No label column found.")

    features_root = DATA_ENV_DIR / "processed" / "features"
    df["feature_path"] = df["feature_path"].apply(
        lambda p: str(features_root / Path(str(p)).name)
    )

    df = df[df["label_id"].notna()].copy()
    df["label_id"] = df["label_id"].astype(int)
    df = df[df["label_id"].between(0, num_classes - 1)].reset_index(drop=True)

    for sp in df["feature_path"].head(3):
        status = "OK" if os.path.isfile(sp) else "MISSING"
        print(f"[INFO] Sample [{status}]: {sp}")

    print(f"[INFO] Final index: {len(df)} rows, {num_classes} classes")
    return df[["feature_path", "label_id"]], label_to_id, id_to_label


# ── Dataset ───────────────────────────────────────────────────────────────────

class EnvironmentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label_id = int(row["label_id"])
        emb      = np.load(row["feature_path"]).astype(np.float32)  # (10, 128)

        # Ensure correct shape
        if emb.shape[0] < N_FRAMES:
            pad = np.zeros((N_FRAMES - emb.shape[0], EMBED_DIM), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=0)
        else:
            emb = emb[:N_FRAMES]

        if self.augment:
            # Random time shift
            if np.random.rand() < 0.5:
                shift = np.random.randint(1, N_FRAMES)
                emb = np.roll(emb, shift, axis=0)
            # Random frame dropout (zero out up to 2 random time steps)
            if np.random.rand() < 0.4:
                n_mask = np.random.randint(1, 3)
                idxs   = np.random.choice(N_FRAMES, n_mask, replace=False)
                emb[idxs] = 0.0

        # (128, 10) for Conv1d — embedding dim as channels, time as length
        x = torch.from_numpy(emb.T)
        return x, torch.tensor(label_id, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class SE1d(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps (B, C, T)."""
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
        # x: (B, C, T)
        s = x.mean(dim=2)          # squeeze: (B, C)
        s = self.fc(s)             # excite:  (B, C)
        return x * s.unsqueeze(2)  # scale:   (B, C, T)


class ResConv1dBlock(nn.Module):
    """
    Residual 1D conv block with SE attention.
    Input and output have the same number of channels.
    """
    def __init__(self, channels: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.se    = SE1d(channels, reduction=se_reduction)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)  # residual skip connection


class EnvironmentCNN(nn.Module):
    """
    1D CNN over VGGish embeddings with:
    - SE channel attention
    - Residual skip connections
    - Input: (B, 128, 10)
    - Output: (B, num_classes)
    """
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        base_channels: int = 256,
        dropout: float = 0.4,
        se_reduction: int = 8,
    ):
        super().__init__()

        # Project embedding dim -> base_channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(embed_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        # Two residual blocks with SE attention (same channels, no pooling needed
        # since T=10 is already very short)
        self.block1 = ResConv1dBlock(base_channels, se_reduction=se_reduction)
        self.block2 = ResConv1dBlock(base_channels, se_reduction=se_reduction)

        # Downsample + narrow
        self.down = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.ReLU(),
        )

        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels // 2, num_classes)

    def forward(self, x):
        x = self.input_proj(x)   # (B, base_channels, 10)
        x = self.block1(x)       # (B, base_channels, 10) — residual + SE
        x = self.block2(x)       # (B, base_channels, 10) — residual + SE
        x = self.down(x)         # (B, base_channels//2, 10)
        x = self.gap(x)          # (B, base_channels//2, 1)
        x = x.squeeze(-1)        # (B, base_channels//2)
        x = self.drop(x)
        return self.fc(x)        # (B, num_classes)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    class_weights acts as alpha per class.
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # per-class alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts  = np.bincount(df["label_id"].values, minlength=num_classes)
    total   = counts.sum()
    weights = total / (counts + 1e-6)
    weights = weights / weights.mean()
    print(f"[INFO] Class counts:  {counts}")
    print(f"[INFO] Class weights: {np.round(weights, 3)}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, device, criterion,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    epoch: int = 0,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches  = len(loader)
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
        if i % PRINT_EVERY == 0 or i == num_batches:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float, float]:
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

    preds      = torch.cat(all_logits).argmax(dim=1).numpy()
    targets_np = torch.cat(all_targets).numpy()

    return (
        running_loss / len(loader),
        accuracy_score(targets_np, preds),
        f1_score(targets_np, preds, average="macro"),
    )


# ── Single training run ───────────────────────────────────────────────────────

def run_trial(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_id: dict,
    id_to_label: dict,
    hparams: dict,
    device: torch.device,
    trial_idx: int,
) -> Tuple[float, str]:
    """Train one model with given hparams. Returns (best_macro_f1, checkpoint_path)."""

    num_classes   = len(label_to_id)
    batch_size    = hparams["batch_size"]
    lr            = hparams["learning_rate"]
    dropout       = hparams["dropout"]
    focal_gamma   = hparams["focal_gamma"]
    se_reduction  = hparams["se_reduction"]
    base_channels = hparams["base_channels"]

    train_ds = EnvironmentDataset(train_df, augment=True)
    val_ds   = EnvironmentDataset(val_df,   augment=False)

    class_weights = compute_class_weights(train_df, num_classes).to(device)
    criterion     = FocalLoss(gamma=focal_gamma, weight=class_weights)

    counts         = np.bincount(train_df["label_id"].values, minlength=num_classes)
    sample_weights = 1.0 / (counts[train_df["label_id"].values] + 1e-6)
    sampler        = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = EnvironmentCNN(
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
        se_reduction=se_reduction,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=lr * 0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_f1    = -1.0
    ckpt_path  = str(OUTPUT_ENV_DIR / f"trial_{trial_idx}_best.pt")

    for epoch in range(NUM_EPOCHS):
        train_loss             = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler, epoch)
        val_loss, val_acc, f1  = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(
            f"  Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"acc={val_acc:.4f} | f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "hparams": hparams,
                "n_frames": N_FRAMES,
                "embed_dim": EMBED_DIM,
            }, ckpt_path)

    return best_f1, ckpt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[INFO] Device: {device}")

    df_index, label_to_id, id_to_label = load_environment_index()
    num_classes = len(label_to_id)

    train_df, val_df = train_test_split(
        df_index,
        test_size=VAL_FRACTION,
        random_state=42,
        shuffle=True,
        stratify=df_index["label_id"],
    )
    print(f"[INFO] Train: {len(train_df)}  Val: {len(val_df)}")

    # ── Random hyperparameter search ──────────────────────────────────────────
    search_results = []

    for trial in range(N_SEARCH_TRIALS):
        hparams = sample_hparams()
        print(f"\n{'='*60}")
        print(f"TRIAL {trial+1}/{N_SEARCH_TRIALS}")
        print_config(hparams)

        best_f1, ckpt_path = run_trial(
            train_df, val_df, label_to_id, id_to_label,
            hparams, device, trial_idx=trial,
        )

        search_results.append({
            "trial": trial + 1,
            "best_macro_f1": best_f1,
            "checkpoint": ckpt_path,
            **hparams,
        })
        print(f"  Trial {trial+1} best macro_f1: {best_f1:.4f}")

    # ── Pick best trial ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(search_results).sort_values("best_macro_f1", ascending=False)
    results_df.to_csv(OUTPUT_ENV_DIR / "hparam_search_results.csv", index=False)

    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(results_df[["trial", "best_macro_f1", "learning_rate", "batch_size",
                       "dropout", "focal_gamma", "se_reduction", "base_channels"]].to_string(index=False))

    best_row  = results_df.iloc[0]
    best_ckpt = best_row["checkpoint"]
    print(f"\n[INFO] Best trial: {int(best_row['trial'])}  macro_f1={best_row['best_macro_f1']:.4f}")
    print(f"[INFO] Best hparams: lr={best_row['learning_rate']}  batch={best_row['batch_size']}  "
          f"dropout={best_row['dropout']}  gamma={best_row['focal_gamma']}  "
          f"se_red={best_row['se_reduction']}  base_ch={best_row['base_channels']}")

    # Copy best checkpoint to final path
    import shutil
    final_path = OUTPUT_ENV_DIR / "best_environment_model.pt"
    shutil.copy(best_ckpt, final_path)
    print(f"[INFO] Best model -> {final_path}")

    # ── Final classification report ───────────────────────────────────────────
    ckpt  = torch.load(final_path, map_location=device)
    model = EnvironmentCNN(
        num_classes=num_classes,
        base_channels=int(best_row["base_channels"]),
        dropout=float(best_row["dropout"]),
        se_reduction=int(best_row["se_reduction"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_ds     = EnvironmentDataset(val_df, augment=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(model(x.to(device)).cpu())
            all_targets.append(y)

    preds       = torch.cat(all_logits).argmax(dim=1).numpy()
    targets_np  = torch.cat(all_targets).numpy()
    label_names = [id_to_label[i] for i in range(num_classes)]
    report      = classification_report(targets_np, preds, target_names=label_names, digits=4)

    report_path = OUTPUT_ENV_DIR / "environment_classification_report.txt"
    report_path.write_text(report)
    print(f"\n[INFO] Classification report -> {report_path}")
    print(report)

    # Save final config
    with open(OUTPUT_ENV_DIR / "environment_training_config.json", "w") as f:
        json.dump({
            "num_epochs": NUM_EPOCHS,
            "val_fraction": VAL_FRACTION,
            "num_classes": num_classes,
            "n_frames": N_FRAMES,
            "embed_dim": EMBED_DIM,
            "best_hparams": {k: float(best_row[k]) if hasattr(best_row[k], "item") else best_row[k] for k in HPARAM_SPACE},
            "best_macro_f1": float(best_row["best_macro_f1"]),
        }, f, indent=2)


if __name__ == "__main__":
    main()