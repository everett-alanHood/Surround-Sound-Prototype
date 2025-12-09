"""
02_events_training.py

Train a multi-label event classifier on precomputed log-mel spectrogram features
(from FSD50K), using the canonical 62-class taxonomy created in 01_events_setup.py.

Assumes 01_events_setup.py has already created:
- data/events/data_index.parquet (or .csv)
- data/events/label_to_id.json
- data/events/id_to_label.json
- .npy feature files under data/events/processed/features
"""

import os
import json
from ast import literal_eval
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_EVENTS_DIR   = PROJECT_ROOT / "data" / "events"
OUTPUT_EVENTS_DIR = PROJECT_ROOT / "output" / "events"

DATA_INDEX_PARQ   = DATA_EVENTS_DIR / "data_index.parquet"
DATA_INDEX_CSV    = DATA_EVENTS_DIR / "data_index.csv"
LABEL_TO_ID_PATH  = DATA_EVENTS_DIR / "label_to_id.json"
ID_TO_LABEL_PATH  = DATA_EVENTS_DIR / "id_to_label.json"

# Training hyperparameters
NUM_EPOCHS   = 40
BATCH_SIZE   = 64
LEARNING_RATE = 2e-4
VAL_FRACTION = 0.15

# Multi-label specifics
THRESH_DEFAULT = 0.5
THRESH_GRID    = np.linspace(0.1, 0.9, 17)  # for tuning after training

# Feature normalization range (log-mel in dB)
DB_MIN = -80.0
DB_MAX = 0.0

# Data loader / performance
NUM_WORKERS = 4
PIN_MEMORY  = True
USE_AMP     = True  # mixed precision on CUDA

# Fast mode for experiments (set FAST_MODE=False for full dataset)
FAST_MODE = False
FAST_N    = 15000  # used only if FAST_MODE=True

os.makedirs(OUTPUT_EVENTS_DIR, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def log_config():
    print("CONFIG:")
    print(f"  PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"  DATA_EVENTS_DIR   = {DATA_EVENTS_DIR}")
    print(f"  OUTPUT_EVENTS_DIR = {OUTPUT_EVENTS_DIR}")
    print(f"  NUM_EPOCHS        = {NUM_EPOCHS}")
    print(f"  BATCH_SIZE        = {BATCH_SIZE}")
    print(f"  LEARNING_RATE     = {LEARNING_RATE}")
    print(f"  VAL_FRACTION      = {VAL_FRACTION}")
    print(f"  FAST_MODE         = {FAST_MODE} FAST_N = {FAST_N}")
    print("------------------------------------------------------------")


def parse_label_ids_cell(x):
    """
    Robustly convert a cell to a list[int]:
    - Already a list/tuple/ndarray -> cast to int list.
    - String like "[1, 2, 3]" or "1" -> parse with literal_eval or int().
    - Anything else -> empty list.
    This fixes the bug where all rows were being dropped as empty.
    """
    import numpy as _np  # local alias to avoid confusion

    # Already list-like
    if isinstance(x, (list, tuple, _np.ndarray)):
        try:
            return [int(v) for v in x]
        except Exception:
            return []

    # String representation
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Try literal_eval (handles "[1, 2, 3]" or "1")
        try:
            v = literal_eval(s)
            if isinstance(v, (list, tuple, _np.ndarray)):
                return [int(z) for z in v]
            else:
                return [int(v)]
        except Exception:
            # Fallback: maybe it's a single int like "12"
            try:
                return [int(s)]
            except Exception:
                return []

    # Anything else, treat as missing
    return []


def load_label_mappings():
    with open(LABEL_TO_ID_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    with open(ID_TO_LABEL_PATH, "r", encoding="utf-8") as f:
        id_to_label = json.load(f)
    # Ensure keys are ints for id_to_label
    id_to_label_int = {int(k): v for k, v in id_to_label.items()}
    return label_to_id, id_to_label_int


# ============================================================
# DATASET
# ============================================================

class EventsDataset(Dataset):
    def __init__(self, df, num_classes):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row["feature_path"]
        label_ids = row["label_ids"]

        spec = np.load(feature_path)  # shape: [n_mels, time]
        spec = spec.astype(np.float32)

        # Normalize from [DB_MIN, DB_MAX] -> [0, 1]
        spec = (spec - DB_MIN) / (DB_MAX - DB_MIN)
        spec = np.clip(spec, 0.0, 1.0)

        # Add channel dim: [1, n_mels, time]
        spec = np.expand_dims(spec, axis=0)

        # Multi-hot targets
        y = np.zeros(self.num_classes, dtype=np.float32)
        for cid in label_ids:
            if 0 <= cid < self.num_classes:
                y[cid] = 1.0

        return torch.from_numpy(spec), torch.from_numpy(y)


def make_dataloaders(df_index, num_classes, device):
    # Optionally subsample for fast mode
    if FAST_MODE:
        print(f"[INFO] FAST_MODE: {len(df_index)} -> {FAST_N} samples.")
        df_index = df_index.sample(n=min(FAST_N, len(df_index)), random_state=42).reset_index(drop=True)

    # Train/val split
    train_df, val_df = train_test_split(
        df_index,
        test_size=VAL_FRACTION,
        random_state=42,
        shuffle=True,
    )

    print(f"[INFO] Train size: {len(train_df)}")
    print(f"[INFO] Val size:   {len(val_df)}")

    # Compute positive class weights for BCEWithLogitsLoss
    label_counts = np.zeros(num_classes, dtype=np.float64)
    for ids in train_df["label_ids"]:
        for cid in ids:
            if 0 <= cid < num_classes:
                label_counts[cid] += 1

    total_samples = len(train_df)
    # Avoid divide-by-zero: pos_weight = (N - n_pos) / (n_pos + eps)
    eps = 1e-3
    pos_weight = (total_samples - label_counts) / (label_counts + eps)
    pos_weight = torch.from_numpy(pos_weight.astype(np.float32))

    print(f"[INFO] Using class weights (pos_weight) with shape: {pos_weight.shape}")

    train_ds = EventsDataset(train_df, num_classes)
    val_ds   = EventsDataset(val_df,   num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches:   {len(val_loader)}")

    pos_weight = pos_weight.to(device)
    return train_loader, val_loader, pos_weight, train_df, val_df


# ============================================================
# MODEL
# ============================================================

class EventsCNN(nn.Module):
    """
    2D CNN similar in spirit to the environment model, but used for multi-label events.
    Conv width: 64 -> 128 -> 256 -> 256, with MaxPooling and GAP.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1,   64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.drop  = nn.Dropout(p=0.4)

        self.fc    = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [B, 64, H/2, W/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [B,128, H/4, W/4]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> [B,256, H/8, W/8]
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> [B,256, H/16, W/16]

        x = self.gap(x)       # [B,256,1,1]
        x = x.view(x.size(0), -1)  # [B,256]
        x = self.drop(x)
        logits = self.fc(x)   # [B,num_classes]
        return logits


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, use_amp=True):
    model.train()
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=loss.item())

    return running_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, threshold=THRESH_DEFAULT, use_amp=True):
    model.eval()
    all_logits = []
    all_targets = []
    running_loss = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
        else:
            logits = model(x)
            loss = loss_fn(logits, y)

        running_loss += loss.item()
        n_batches += 1

        all_logits.append(torch.sigmoid(logits).cpu())
        all_targets.append(y.cpu())

    avg_loss = running_loss / max(1, n_batches)
    probs = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    preds = (probs >= threshold).astype(np.int32)

    micro = f1_score(targets, preds, average="micro", zero_division=0)
    macro = f1_score(targets, preds, average="macro", zero_division=0)

    return avg_loss, micro, macro, probs, targets


def tune_threshold(model, loader, device, loss_fn, thresholds=THRESH_GRID, use_amp=True):
    best_t = None
    best_micro = -1.0
    best_macro = -1.0
    best_probs = None
    best_targets = None

    print("[THRESHOLD TUNING] evaluating thresholds...")
    for t in thresholds:
        val_loss, micro, macro, probs, targets = evaluate(
            model, loader, device, loss_fn, threshold=t, use_amp=use_amp
        )
        print(f"  t={t:.3f}: micro_f1={micro:.4f}, macro_f1={macro:.4f}")
        if micro > best_micro:
            best_micro = micro
            best_macro = macro
            best_t = t
            best_probs = probs
            best_targets = targets

    print(
        f"[THRESHOLD TUNING] best_t={best_t:.3f}, "
        f"micro_f1={best_micro:.4f}, macro_f1={best_macro:.4f}"
    )
    return best_t, best_micro, best_macro, best_probs, best_targets


# ============================================================
# MAIN
# ============================================================

def main():
    log_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load label mappings
    label_to_id, id_to_label = load_label_mappings()
    num_classes = len(label_to_id)
    print(f"[INFO] Loaded {num_classes} event classes.")

    # Load index (prefer Parquet)
    if DATA_INDEX_PARQ.exists():
        print(f"[INFO] Loading data index from Parquet: {DATA_INDEX_PARQ}")
        df_index = pd.read_parquet(DATA_INDEX_PARQ)
    elif DATA_INDEX_CSV.exists():
        print(f"[INFO] Loading data index from CSV: {DATA_INDEX_CSV}")
        df_index = pd.read_csv(DATA_INDEX_CSV)
    else:
        raise FileNotFoundError("No data_index.parquet or data_index.csv found in events directory.")

    # Robustly parse label_ids (fix for the 'all rows dropped' bug)
    print(f"[INFO] Full df_index shape: {df_index.shape}")
    df_index["label_ids"] = df_index["label_ids"].apply(parse_label_ids_cell)

    before = len(df_index)
    df_index = df_index[df_index["label_ids"].map(len) > 0]
    after = len(df_index)
    print(f"[INFO] Dropped rows with empty label_ids: {before} -> {after}")

    if len(df_index) == 0:
        raise RuntimeError(
            "After parsing label_ids, df_index has 0 rows. "
            "Check that data_index.* actually contains label_ids for events."
        )

    # Make loaders + pos_weight
    train_loader, val_loader, pos_weight, train_df, val_df = make_dataloaders(
        df_index, num_classes, device
    )

    # Model, optimizer, loss, scheduler
    model = EventsCNN(num_classes=num_classes).to(device)
    print(model)

    # BCEWithLogitsLoss with positive class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )

    # Cosine annealing over epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
        verbose=False,
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"[INFO] Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("[INFO] Starting training...")
    print("------------------------------------------------------------")

    best_micro_f1 = -1.0
    best_epoch = -1
    history = []

    best_model_path = OUTPUT_EVENTS_DIR / "best_events_model.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler if use_amp else None,
            device=device,
            loss_fn=loss_fn,
            use_amp=use_amp,
        )

        # Step scheduler per epoch
        scheduler.step()

        # Eval at default threshold 0.5 (for checkpointing)
        val_loss, micro_f1, macro_f1, _, _ = evaluate(
            model, val_loader, device, loss_fn, threshold=THRESH_DEFAULT, use_amp=use_amp
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "lr": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"micro_f1={micro_f1:.4f} | "
            f"macro_f1={macro_f1:.4f}"
        )

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_micro_f1": best_micro_f1,
                },
                best_model_path,
            )
            print(f"  -> New best model saved (epoch {epoch}, micro_f1={micro_f1:.4f})")

    print("------------------------------------------------------------")
    print(f"[FINAL] best_micro_f1: {best_micro_f1:.4f} (epoch {best_epoch})")

    # Save history + config
    hist_df = pd.DataFrame(history)
    hist_path = OUTPUT_EVENTS_DIR / "events_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"[INFO] Saved history to: {hist_path}")

    cfg = {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "val_fraction": VAL_FRACTION,
        "fast_mode": FAST_MODE,
        "fast_n": FAST_N,
        "use_amp": USE_AMP,
        "num_workers": NUM_WORKERS,
    }
    cfg_path = OUTPUT_EVENTS_DIR / "events_training_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Saved training config to: {cfg_path}")

    # ============================
    # Final evaluation + threshold tuning
    # ============================
    if not best_model_path.exists():
        print("[WARN] Best model not found; skipping final evaluation.")
        return

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # First evaluate at default t=0.5
    val_loss, micro_f1, macro_f1, probs, targets = evaluate(
        model, val_loader, device, loss_fn, threshold=THRESH_DEFAULT, use_amp=use_amp
    )
    print(
        f"[FINAL DEFAULT] threshold={THRESH_DEFAULT:.2f}, "
        f"val_loss={val_loss:.4f}, micro_f1={micro_f1:.4f}, macro_f1={macro_f1:.4f}"
    )

    # Then tune threshold on validation set
    best_t, best_micro, best_macro, best_probs, best_targets = tune_threshold(
        model, val_loader, device, loss_fn, thresholds=THRESH_GRID, use_amp=use_amp
    )

    print(
        f"[FINAL TUNED] threshold={best_t:.3f}, "
        f"val_loss={val_loss:.4f}, micro_f1={best_micro:.4f}, macro_f1={best_macro:.4f}"
    )

    # Classification report at tuned threshold
    preds = (best_probs >= best_t).astype(np.int32)
    report = classification_report(
        best_targets,
        preds,
        target_names=[id_to_label[i] for i in range(num_classes)],
        zero_division=0,
    )
    print("[CLASSIFICATION REPORT (truncated)]:")
    print(report[:1000])  # just keep console readable

    report_path = OUTPUT_EVENTS_DIR / "events_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"BEST_THRESHOLD = {best_t:.3f}\n"
            f"MICRO_F1 = {best_micro:.4f}\n"
            f"MACRO_F1 = {best_macro:.4f}\n\n"
        )
        f.write(report)

    print(f"[INFO] Saved classification report to: {report_path}")


if __name__ == "__main__":
    main()
