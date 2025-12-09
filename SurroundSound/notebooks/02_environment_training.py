"""
02_environment_training.py

Train a single-label environment classifier using precomputed log-mel features.

Requires:
- data/environment/data_index.parquet
- data/environment/label_to_id.json
- data/environment/id_to_label.json
- data/environment/processed/features/*.npy

Outputs:
- output/environment/best_environment_model.pt
- output/environment/environment_history.csv
- output/environment/environment_training_config.json
- output/environment/environment_classification_report.txt
"""

import os
import json
import math
import ast
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ENV_DIR = PROJECT_ROOT / "data" / "environment"
OUTPUT_ENV_DIR = PROJECT_ROOT / "output" / "environment"

INDEX_PATH_PARQUET = DATA_ENV_DIR / "data_index.parquet"
LABEL_TO_ID_PATH = DATA_ENV_DIR / "label_to_id.json"
ID_TO_LABEL_PATH = DATA_ENV_DIR / "id_to_label.json"

OUTPUT_ENV_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparams (tweak these if needed)
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
VAL_FRACTION = 0.15
USE_AMP = True           # mixed precision on 3060 Ti
NUM_WORKERS = 4
PIN_MEMORY = True

PRINT_EVERY = 20


def print_config():
    print("CONFIG:")
    print(f"  PROJECT_ROOT      = {PROJECT_ROOT}")
    print(f"  DATA_ENV_DIR      = {DATA_ENV_DIR}")
    print(f"  OUTPUT_ENV_DIR    = {OUTPUT_ENV_DIR}")
    print(f"  NUM_EPOCHS        = {NUM_EPOCHS}")
    print(f"  BATCH_SIZE        = {BATCH_SIZE}")
    print(f"  LEARNING_RATE     = {LEARNING_RATE}")
    print(f"  VAL_FRACTION      = {VAL_FRACTION}")
    print("------------------------------------------------------------")


# ============================================================
# LABEL / INDEX LOADING (ROBUST TO YOUR EXISTING FORMAT)
# ============================================================

def _load_label_dicts() -> Tuple[dict, dict]:
    with open(LABEL_TO_ID_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    with open(ID_TO_LABEL_PATH, "r", encoding="utf-8") as f:
        id_to_label = json.load(f)
    # keys of id_to_label might be strings
    id_to_label = {int(k): v for k, v in id_to_label.items()}
    return label_to_id, id_to_label


def _infer_feature_column(df: pd.DataFrame) -> str:
    candidates = [
        "feature_path",
        "spec_path",
        "npy_path",
        "feature_file",
        "path"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find a feature path column in data_index. "
        f"Looked for: {candidates}. Found columns: {list(df.columns)}"
    )


def _try_parse_label_from_object(
    value, label_to_id: dict, num_classes: int
) -> Optional[int]:
    """
    Try to turn a single cell (object/str) into an integer label index.
    We support:
      - plain integer strings: "3"
      - label name strings: "office"
      - JSON-like lists: '["office"]' or "[3]" (take first element)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    # Already an int?
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if 0 <= v < num_classes:
            return v
        return None

    # String cases
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        # pure integer string?
        if s.isdigit():
            v = int(s)
            if 0 <= v < num_classes:
                return v
            return None

        # Try to parse as Python literal (e.g., '["office"]' or "[3]")
        try:
            obj = ast.literal_eval(s)
        except Exception:
            obj = s

        if isinstance(obj, list) and obj:
            obj = obj[0]

        if isinstance(obj, (int, np.integer)):
            v = int(obj)
            if 0 <= v < num_classes:
                return v
            return None

        if isinstance(obj, str):
            label_name = obj
            if label_name in label_to_id:
                return int(label_to_id[label_name])
            return None

        return None

    # Unknown type
    return None


def _infer_label_column(df: pd.DataFrame, label_to_id: dict) -> Tuple[pd.Series, str]:
    """
    Try to extract a single integer label (0..num_classes-1) per row
    from one of several possible columns in environment data_index.
    """
    num_classes = len(label_to_id)
    candidate_cols = [
        "label_id",
        "label",
        "label_name",
        "env_label",
        "y",
        "class_id",
        "target",
    ]

    existing_candidate_cols = [c for c in candidate_cols if c in df.columns]
    if not existing_candidate_cols:
        raise RuntimeError(
            f"Could not find any label column in data_index. "
            f"Looked for: {candidate_cols}. Found columns: {list(df.columns)}"
        )

    print(f"[INFO] Candidate label columns present: {existing_candidate_cols}")

    for col in existing_candidate_cols:
        s = df[col]
        print(f"[INFO] Trying label column: {col} (dtype={s.dtype})")

        # Case 1: already integer-like
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            arr = s.astype(int)
            uniq = pd.unique(arr)
            uniq_valid = [u for u in uniq if 0 <= u < num_classes]
            if len(uniq_valid) > 0:
                print(
                    f"[INFO] Using integer label column '{col}' "
                    f"(unique valid labels: {sorted(set(uniq_valid))})"
                )
                return arr, col

        # Case 2: float but integer-valued
        if pd.api.types.is_float_dtype(s):
            # keep only those that are nearly ints
            def _float_to_int_or_none(v):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return None
                rv = round(float(v))
                if abs(rv - float(v)) < 1e-6 and 0 <= rv < num_classes:
                    return int(rv)
                return None

            ids = s.map(_float_to_int_or_none)
            if ids.notna().sum() > 0:
                uniq = sorted(set(ids.dropna().astype(int).tolist()))
                print(
                    f"[INFO] Using float->int label column '{col}' "
                    f"(unique valid labels: {uniq})"
                )
                return ids.astype("Int64"), col

        # Case 3: object/string: try parse names or literals
        if s.dtype == object:
            ids = s.map(lambda v: _try_parse_label_from_object(v, label_to_id, num_classes))
            valid_count = ids.notna().sum()
            if valid_count > 0:
                uniq = sorted(set(ids.dropna().astype(int).tolist()))
                print(
                    f"[INFO] Using parsed string label column '{col}' "
                    f"(valid rows: {valid_count}, unique labels: {uniq})"
                )
                return ids.astype("Int64"), col

        print(f"[INFO] Column '{col}' could not be parsed as labels, trying next...")

    raise RuntimeError(
        "Could not infer a valid label column from any of the candidates. "
        "Please inspect your data_index.parquet and adjust the script."
    )


def load_environment_index() -> Tuple[pd.DataFrame, dict, dict]:
    print(f"[INFO] Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"[INFO] Loading data index from Parquet: {INDEX_PATH_PARQUET}")

    df = pd.read_parquet(INDEX_PATH_PARQUET)
    print(f"[INFO] Raw index shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    label_to_id, id_to_label = _load_label_dicts()
    num_classes = len(label_to_id)
    print(f"[INFO] Loaded label_to_id with {num_classes} classes.")

    # Infer feature column
    feature_col = _infer_feature_column(df)
    print(f"[INFO] Using feature column: {feature_col}")

    # Infer label column/values
    label_ids_series, label_col = _infer_label_column(df, label_to_id)

    # Drop rows with invalid labels
    before = len(df)
    mask_valid = label_ids_series.notna()
    df_valid = df.loc[mask_valid].copy()
    df_valid["label_id"] = label_ids_series[mask_valid].astype(int)

    # remap feature paths to the *current* features dir
    features_root = DATA_ENV_DIR / "processed" / "features"

    def _map_feature_path(p: str) -> str:
        # keep only the filename and put it under features_root
        fname = Path(str(p)).name
        return str(features_root / fname)

    df_valid["feature_path"] = df_valid[feature_col].apply(_map_feature_path)
    df_final = df_valid[["feature_path", "label_id"]]

    after = len(df_final)
    print(f"[INFO] Dropped rows with invalid/empty labels: {before} -> {after}")
    print(f"[INFO] Final df_index shape: {df_final.shape}")

    # sanity check a couple of files exist
    sample_paths = df_final["feature_path"].head(3).tolist()
    for sp in sample_paths:
        if not os.path.isfile(sp):
            print(f"[WARN] Sample feature path does NOT exist: {sp}")
        else:
            print(f"[INFO] Sample feature path OK: {sp}")

    return df_final, label_to_id, id_to_label



# ============================================================
# DATASET
# ============================================================

class EnvironmentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_classes: int):
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row["feature_path"]
        label_id = int(row["label_id"])

        spec = np.load(feature_path).astype(np.float32)
        # Expect (freq, time); add channel dimension
        if spec.ndim == 2:
            spec = np.expand_dims(spec, axis=0)  # (1, F, T)
        elif spec.ndim == 3:
            # assume already (C, F, T)
            pass
        else:
            raise ValueError(f"Unexpected spec shape {spec.shape} at {feature_path}")

        y = label_id
        return torch.from_numpy(spec), torch.tensor(y, dtype=torch.long)


# ============================================================
# MODEL (simple CNN, lighter than events model)
# ============================================================

class EnvironmentCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = 32  # smaller than events model (which used 64/128/256)

        self.conv1 = nn.Conv2d(1, base, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base)

        self.conv2 = nn.Conv2d(base, base * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2)

        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base * 4)

        self.conv4 = nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base * 4)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(base * 4, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (B, 32, F/2, T/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (B, 64, F/4, T/4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (B, 128, F/8, T/8)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> (B, 128, F/16, T/16)
        x = self.gap(x)                                 # -> (B, 128, 1, 1)
        x = x.view(x.size(0), -1)                       # -> (B, 128)
        x = self.drop(x)
        x = self.fc(x)                                  # -> (B, num_classes)
        return x


# ============================================================
# TRAIN / EVAL
# ============================================================

def compute_class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = np.bincount(df["label_id"].values, minlength=num_classes)
    print(f"[INFO] Class counts: {counts}")
    total = counts.sum()
    # inverse freq
    weights = total / (counts + 1e-6)
    weights = weights / weights.mean()
    print(f"[INFO] Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    epoch: int = 0,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Train {epoch+1}", leave=False)
    for i, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if i % PRINT_EVERY == 0 or i == num_batches:
            pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    all_logits = []
    all_targets = []
    running_loss = 0.0
    num_batches = len(loader)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    avg_loss = running_loss / num_batches
    logits_full = torch.cat(all_logits, dim=0)
    targets_full = torch.cat(all_targets, dim=0)

    preds = logits_full.argmax(dim=1).numpy()
    targets_np = targets_full.numpy()

    acc = accuracy_score(targets_np, preds)
    macro_f1 = f1_score(targets_np, preds, average="macro")

    return avg_loss, acc, macro_f1


# ============================================================
# MAIN
# ============================================================

def main():
    print_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load index
    df_index, label_to_id, id_to_label = load_environment_index()
    num_samples = len(df_index)
    if num_samples == 0:
        raise RuntimeError(
            "No valid samples found in environment data_index after label parsing. "
            "Please check your data_index and label_to_id."
        )

    num_classes = len(label_to_id)
    print(f"[INFO] Num samples: {num_samples}")
    print(f"[INFO] Num classes: {num_classes}")

    # Train/val split
    train_df, val_df = train_test_split(
        df_index,
        test_size=VAL_FRACTION,
        random_state=42,
        shuffle=True,
        stratify=df_index["label_id"],
    )
    print(f"[INFO] Train size: {len(train_df)}")
    print(f"[INFO] Val size:   {len(val_df)}")

    # Load datasets
    train_ds = EnvironmentDataset(train_df, num_classes=num_classes)
    val_ds = EnvironmentDataset(val_df, num_classes=num_classes)

    # Class weights for CrossEntropy
    class_weights = compute_class_weights(train_df, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Weighted sampler (optional – helps balance)
    counts = np.bincount(train_df["label_id"].values, minlength=num_classes)
    sample_weights = 1.0 / (counts[train_df["label_id"].values] + 1e-6)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches:   {len(val_loader)}")

    # Model, optimizer, scheduler, scaler
    model = EnvironmentCNN(num_classes=num_classes).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.1)

    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    # Training loop
    best_val_macro_f1 = -1.0
    best_model_path = OUTPUT_ENV_DIR / "best_environment_model.pt"

    history = []
    print("[INFO] Starting training...")
    print("------------------------------------------------------------")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            scaler=scaler,
            epoch=epoch,
        )
        val_loss, val_acc, val_macro_f1 = evaluate(
            model, val_loader, device, criterion
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"macro_f1={val_macro_f1:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_macro_f1": float(val_macro_f1),
            }
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                },
                best_model_path,
            )
            print(
                f"  -> New best model saved (epoch {epoch+1}, macro_f1={val_macro_f1:.4f})"
            )

    print("------------------------------------------------------------")
    print(f"[FINAL] best_macro_f1: {best_val_macro_f1:.4f}")

    # Save history & config
    hist_df = pd.DataFrame(history)
    hist_path = OUTPUT_ENV_DIR / "environment_history.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"[INFO] Saved history to: {hist_path}")

    cfg = {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "val_fraction": VAL_FRACTION,
        "num_classes": num_classes,
    }
    cfg_path = OUTPUT_ENV_DIR / "environment_training_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Saved training config to: {cfg_path}")

    # Final evaluation with best model
    if best_model_path.is_file():
        print("[INFO] Reloading best model for final evaluation.")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_loss, val_acc, val_macro_f1 = evaluate(
            model, val_loader, device, criterion
        )
        print(
            f"[FINAL EVAL] val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, macro_f1={val_macro_f1:.4f}"
        )

        # classification report
        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())

        logits_full = torch.cat(all_logits, dim=0)
        targets_full = torch.cat(all_targets, dim=0)

        preds = logits_full.argmax(dim=1).numpy()
        targets_np = targets_full.numpy()

        # Build mapping id->label name list ordered by id
        label_names = [id_to_label[i] for i in range(len(id_to_label))]
        report = classification_report(
            targets_np,
            preds,
            target_names=label_names,
            digits=4,
        )

        report_path = OUTPUT_ENV_DIR / "environment_classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] Saved classification report to: {report_path}")
        print("[CLASSIFICATION REPORT (truncated)]:")
        print("\n".join(report.splitlines()[:30]))
    else:
        print("[WARN] Best model not found; skipping final evaluation.")


if __name__ == "__main__":
    main()
