"""
02_events_training.py

Train a multi-label event classifier on precomputed log-mel spectrogram features
(from FSD50K), using the canonical taxonomy from 01_events_setup.py.

V2 upgrades:
- SE2d channel attention blocks
- Residual skip connections
- Focal loss (binary) replacing BCEWithLogitsLoss
- Random hyperparameter search
- GPU fallback for sm_120 incompatibility
- Fixed deprecated torch.amp warnings
- SpecAugment data augmentation
"""

import json
import os
import random
import shutil
import warnings
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_EVENTS_DIR   = PROJECT_ROOT / "data" / "events"
OUTPUT_EVENTS_DIR = PROJECT_ROOT / "output" / "events"

DATA_INDEX_PARQ  = DATA_EVENTS_DIR / "data_index.parquet"
DATA_INDEX_CSV   = DATA_EVENTS_DIR / "data_index.csv"
LABEL_TO_ID_PATH = DATA_EVENTS_DIR / "label_to_id.json"
ID_TO_LABEL_PATH = DATA_EVENTS_DIR / "id_to_label.json"

OUTPUT_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed settings ────────────────────────────────────────────────────────────

NUM_EPOCHS    = 40
VAL_FRACTION  = 0.15
USE_AMP       = True
NUM_WORKERS   = 4
PIN_MEMORY    = True
N_TRIALS      = 6

THRESH_DEFAULT = 0.5
THRESH_GRID    = np.linspace(0.1, 0.9, 17)

DB_MIN = -80.0
DB_MAX = 0.0

# ── Hyperparameter search space ───────────────────────────────────────────────

HPARAM_SPACE = {
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "batch_size":    [32, 64],
    "dropout":       [0.3, 0.4, 0.5],
    "focal_gamma":   [1.0, 2.0],
    "se_reduction":  [8, 16],
    "base_channels": [64, 128],
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


# ── Label loading ─────────────────────────────────────────────────────────────

def load_label_mappings():
    with open(LABEL_TO_ID_PATH) as f:
        label_to_id = json.load(f)
    with open(ID_TO_LABEL_PATH) as f:
        id_to_label = {int(k): v for k, v in json.load(f).items()}
    return label_to_id, id_to_label


def parse_label_ids_cell(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return [int(v) for v in x]
        except Exception:
            return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [int(z) for z in v]
            return [int(v)]
        except Exception:
            try:
                return [int(s)]
            except Exception:
                return []
    return []


# ── Dataset ───────────────────────────────────────────────────────────────────

class EventsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_classes: int, augment: bool = False):
        self.df          = df.reset_index(drop=True)
        self.num_classes = num_classes
        self.augment     = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label_ids = row["label_ids"]

        spec = np.load(row["feature_path"]).astype(np.float32)  # (128, T)

        # Normalize dB range -> [0, 1]
        spec = np.clip((spec - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)

        # SpecAugment
        if self.augment:
            T = spec.shape[1]
            if np.random.rand() < 0.5:
                t0 = np.random.randint(0, max(1, T - 30))
                spec[:, t0:t0 + np.random.randint(1, 30)] = 0.0
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, 108)  # 128 - 20
                spec[f0:f0 + np.random.randint(1, 20), :] = 0.0

        # (1, 128, T)
        x = torch.from_numpy(spec).unsqueeze(0)

        # Multi-hot target
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for cid in label_ids:
            if 0 <= cid < self.num_classes:
                y[cid] = 1.0

        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────

class SE2d(nn.Module):
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
        s = x.mean(dim=[2, 3])
        s = self.fc(s)
        return x * s[:, :, None, None]


class ResConv2dBlock(nn.Module):
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


class EventsCNN(nn.Module):
    """
    2D CNN for multi-label event classification.
    Input: (B, 1, 128, T)
    Output: (B, num_classes) — raw logits for sigmoid
    """
    def __init__(
        self,
        num_classes: int,
        base_channels: int = 64,
        dropout: float = 0.4,
        se_reduction: int = 8,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 1 + residual SE
        self.block1 = ResConv2dBlock(base_channels, se_reduction)

        # Widen
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2 + residual SE
        self.block2 = ResConv2dBlock(base_channels * 2, se_reduction)

        # Widen further
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 3 + residual SE (deeper than environment model for 62 classes)
        self.block3 = ResConv2dBlock(base_channels * 4, se_reduction)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


# ── Focal Loss (binary) ───────────────────────────────────────────────────────

class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.
    FL(p) = -alpha * (1 - pt)^gamma * log(pt)
    pos_weight acts as alpha per class.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt   = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


# ── Positive class weights ────────────────────────────────────────────────────

def compute_pos_weight(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for ids in train_df["label_ids"]:
        for cid in ids:
            if 0 <= cid < num_classes:
                counts[cid] += 1
    total      = len(train_df)
    eps        = 1e-3
    pos_weight = (total - counts) / (counts + eps)
    print(f"[INFO] pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    return torch.tensor(pos_weight, dtype=torch.float32)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, loss_fn, scaler, epoch) -> float:
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"  Train {epoch}", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, threshold=0.5) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_targets = [], []
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        running_loss += loss_fn(logits, y).item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    probs   = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds   = (probs >= threshold).astype(np.int32)
    micro   = f1_score(targets, preds, average="micro", zero_division=0)
    macro   = f1_score(targets, preds, average="macro", zero_division=0)
    return running_loss / max(len(loader), 1), micro, macro, probs, targets


def tune_threshold(model, loader, device, loss_fn) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    best_t, best_micro, best_macro = THRESH_DEFAULT, -1.0, -1.0
    best_probs, best_targets = None, None
    print("[THRESHOLD TUNING]")
    for t in THRESH_GRID:
        _, micro, macro, probs, targets = evaluate(model, loader, device, loss_fn, threshold=t)
        print(f"  t={t:.2f}: micro={micro:.4f}  macro={macro:.4f}")
        if micro > best_micro:
            best_micro, best_macro, best_t = micro, macro, t
            best_probs, best_targets = probs, targets
    print(f"  Best: t={best_t:.2f}  micro={best_micro:.4f}  macro={best_macro:.4f}")
    return best_t, best_micro, best_macro, best_probs, best_targets


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(
    train_df, val_df, num_classes, hparams, device, trial_idx
) -> Tuple[float, str]:
    batch_size    = hparams["batch_size"]
    lr            = hparams["learning_rate"]
    dropout       = hparams["dropout"]
    focal_gamma   = hparams["focal_gamma"]
    se_reduction  = hparams["se_reduction"]
    base_channels = hparams["base_channels"]

    train_ds = EventsDataset(train_df, num_classes, augment=True)
    val_ds   = EventsDataset(val_df,   num_classes, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    pos_weight = compute_pos_weight(train_df, num_classes).to(device)
    loss_fn    = BinaryFocalLoss(gamma=focal_gamma, pos_weight=pos_weight)

    model     = EventsCNN(num_classes, base_channels, dropout, se_reduction).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    best_micro = -1.0
    ckpt_path  = str(OUTPUT_EVENTS_DIR / f"trial_{trial_idx}_best.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss              = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler, epoch)
        val_loss, micro, macro, _, _ = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | train={train_loss:.4f} "
              f"val={val_loss:.4f} micro={micro:.4f} macro={macro:.4f}")
        if micro > best_micro:
            best_micro = micro
            torch.save({
                "model_state": model.state_dict(),
                "hparams": hparams,
                "num_classes": num_classes,
            }, ckpt_path)

    return best_micro, ckpt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"[INFO] Device: {device}")

    label_to_id, id_to_label = load_label_mappings()
    num_classes = len(label_to_id)
    print(f"[INFO] Classes: {num_classes}")

    # Load index
    if DATA_INDEX_PARQ.exists():
        df = pd.read_parquet(DATA_INDEX_PARQ)
        print(f"[INFO] Loaded parquet: {df.shape}")
    elif DATA_INDEX_CSV.exists():
        df = pd.read_csv(DATA_INDEX_CSV)
        print(f"[INFO] Loaded CSV: {df.shape}")
    else:
        raise FileNotFoundError("No data_index.parquet or data_index.csv found.")

    df["label_ids"] = df["label_ids"].apply(parse_label_ids_cell)
    before = len(df)
    df = df[df["label_ids"].map(len) > 0].reset_index(drop=True)
    print(f"[INFO] Rows with labels: {len(df)}/{before}")

    # Remap feature paths to current machine
    features_root = DATA_EVENTS_DIR / "processed" / "features"
    df["feature_path"] = df["feature_path"].apply(
        lambda p: str(features_root / Path(str(p)).name)
    )
    df = df[df["feature_path"].apply(os.path.isfile)].reset_index(drop=True)
    print(f"[INFO] Rows with existing features: {len(df)}")

    train_df, val_df = train_test_split(
        df, test_size=VAL_FRACTION, random_state=42, shuffle=True
    )
    print(f"[INFO] Train: {len(train_df)}  Val: {len(val_df)}")

    # ── Random hyperparameter search ──────────────────────────────────────────
    search_results = []
    for trial in range(N_TRIALS):
        hparams = sample_hparams()
        print(f"\n{'='*60}\nTRIAL {trial+1}/{N_TRIALS}  hparams={hparams}\n{'-'*60}")
        best_micro, ckpt_path = run_trial(train_df, val_df, num_classes, hparams, device, trial)
        search_results.append({"trial": trial+1, "best_micro_f1": best_micro,
                                "checkpoint": ckpt_path, **hparams})
        print(f"  Trial {trial+1} best micro_f1: {best_micro:.4f}")

    results_df = pd.DataFrame(search_results).sort_values("best_micro_f1", ascending=False)
    results_df.to_csv(OUTPUT_EVENTS_DIR / "hparam_search_results.csv", index=False)

    best_row  = results_df.iloc[0]
    best_ckpt = best_row["checkpoint"]
    print(f"\n{'='*60}")
    print(f"[INFO] Best trial: {int(best_row['trial'])}  micro_f1={best_row['best_micro_f1']:.4f}")
    print(results_df[["trial","best_micro_f1","learning_rate","batch_size",
                       "dropout","focal_gamma","se_reduction","base_channels"]].to_string(index=False))

    final_path = OUTPUT_EVENTS_DIR / "best_events_model.pt"
    shutil.copy(best_ckpt, final_path)
    print(f"[INFO] Best model -> {final_path}")

    # ── Final threshold tuning + report ──────────────────────────────────────
    ckpt  = torch.load(final_path, map_location=device)
    model = EventsCNN(
        num_classes    = num_classes,
        base_channels  = int(best_row["base_channels"]),
        dropout        = float(best_row["dropout"]),
        se_reduction   = int(best_row["se_reduction"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    pos_weight = compute_pos_weight(train_df, num_classes).to(device)
    loss_fn    = BinaryFocalLoss(gamma=float(best_row["focal_gamma"]), pos_weight=pos_weight)

    val_ds     = EventsDataset(val_df, num_classes, augment=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    best_t, best_micro, best_macro, best_probs, best_targets = tune_threshold(
        model, val_loader, device, loss_fn
    )

    preds  = (best_probs >= best_t).astype(np.int32)
    report = classification_report(
        best_targets, preds,
        target_names=[id_to_label[i] for i in range(num_classes)],
        zero_division=0,
    )

    report_path = OUTPUT_EVENTS_DIR / "events_classification_report.txt"
    report_path.write_text(
        f"BEST_THRESHOLD = {best_t:.3f}\n"
        f"MICRO_F1 = {best_micro:.4f}\n"
        f"MACRO_F1 = {best_macro:.4f}\n\n"
        + report
    )
    print(f"[INFO] Classification report -> {report_path}")
    print(report[:1500])

    with open(OUTPUT_EVENTS_DIR / "events_training_config.json", "w") as f:
        json.dump({
            "num_epochs": NUM_EPOCHS, "val_fraction": VAL_FRACTION,
            "num_classes": num_classes, "best_threshold": best_t,
            "best_micro_f1": float(best_row["best_micro_f1"]),
            "best_hparams": {k: float(best_row[k]) if hasattr(best_row[k], "item")
                             else best_row[k] for k in HPARAM_SPACE},
        }, f, indent=2)

    print("\n[INFO] DONE.")


if __name__ == "__main__":
    main()