"""
Unified evaluation script for Surround Sound models.

- Environment model: single-label scene classification
- Events model: multi-label sound event tagging

Outputs go to:
    src/results/environment/<split>/
    src/results/events/<split>/

Usage examples (from project root):
    python -m src.eval --task env --split val
    python -m src.eval --task events --split val
    python -m src.eval --task both --split val
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

# -------------------------------------------------------------------------
# Project helpers from models_live.py
# -------------------------------------------------------------------------
from live_demo.models_live import (
    DATA_ENV_DIR,
    DATA_EVENTS_DIR,
    OUTPUT_ENV_DIR,
    OUTPUT_EVENTS_DIR,
    load_environment_model,
    load_events_model,
    load_env_label_map,
    load_event_label_map,
    DEVICE,
)

# Where to write outputs (images & CSVs)
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# -------------------------------------------------------------------------
# CONFIG: matches your parquet schemas
# -------------------------------------------------------------------------

# Environment: data/environment/data_index.parquet
ENV_INDEX_PATH = DATA_ENV_DIR / "data_index.parquet"
ENV_SPEC_COL = "feature_path"
ENV_LABEL_COL = "y"
ENV_SPLIT_COL = None  # no split column in env parquet; --split is ignored

# Events: data/events/data_index.parquet
EVENTS_INDEX_PATH = DATA_EVENTS_DIR / "data_index.parquet"

EVENTS_SPEC_COL = "feature_path"
EVENTS_LABEL_IDS_COL = "label_ids"
EVENTS_SPLIT_COL = None  # no split column in events parquet; --split is ignored

# Threshold for multi-label predictions
EVENTS_THRESHOLD = 0.5


# -------------------------------------------------------------------------
# Utility: spec loading
# -------------------------------------------------------------------------

def _load_spec(path: Path) -> torch.Tensor:
    """
    Load a spectrogram saved as either .npy or torch tensor.
    Ensures shape is (1, F, T).

    feature_path points into ...processed/features/... as .npy (or .pt).
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        x = torch.from_numpy(arr).float()
    elif path.suffix in (".pt", ".pth"):
        x = torch.load(path).float()
    else:
        raise ValueError(f"Unsupported spec file extension: {path.suffix} ({path})")

    # shape fixups: want (1, F, T)
    if x.ndim == 2:
        x = x.unsqueeze(0)       # (F, T) -> (1, F, T)
    elif x.ndim == 3 and x.shape[0] != 1:
        x = x.permute(2, 0, 1)
    return x


# -------------------------------------------------------------------------
# Environment dataset (single-label)
# -------------------------------------------------------------------------

class EnvironmentEvalDataset(Dataset):
    def __init__(self, split: str = "val"):
        if not ENV_INDEX_PATH.exists():
            raise FileNotFoundError(f"Environment index not found: {ENV_INDEX_PATH}")

        df = pd.read_parquet(ENV_INDEX_PATH)

        if ENV_SPEC_COL not in df.columns or ENV_LABEL_COL not in df.columns:
            raise KeyError(
                f"Expected columns '{ENV_SPEC_COL}' and '{ENV_LABEL_COL}' "
                f"in {ENV_INDEX_PATH}. Got columns: {list(df.columns)}"
            )

        env_feat_dir = DATA_ENV_DIR / "processed" / "features"

        def remap_env_path(p):
            p = Path(p)
            return env_feat_dir / p.name

        df[ENV_SPEC_COL] = df[ENV_SPEC_COL].apply(remap_env_path)

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_path = Path(row[ENV_SPEC_COL])
        y = int(row[ENV_LABEL_COL])
        x = _load_spec(spec_path)
        return x, y


# -------------------------------------------------------------------------
# Events dataset (multi-label)
# -------------------------------------------------------------------------

class EventsEvalDataset(Dataset):
    def __init__(self, split: str, num_classes: int):
        if not EVENTS_INDEX_PATH.exists():
            raise FileNotFoundError(f"Events index not found: {EVENTS_INDEX_PATH}")

        df = pd.read_parquet(EVENTS_INDEX_PATH)

        if EVENTS_SPEC_COL not in df.columns or EVENTS_LABEL_IDS_COL not in df.columns:
            raise KeyError(
                f"Expected columns '{EVENTS_SPEC_COL}' and '{EVENTS_LABEL_IDS_COL}' "
                f"in {EVENTS_INDEX_PATH}. Got columns: {list(df.columns)}"
            )

        events_feat_dir = DATA_EVENTS_DIR / "processed" / "features"

        def remap_evt_path(p):
            p = Path(p)
            return events_feat_dir / p.name

        df[EVENTS_SPEC_COL] = df[EVENTS_SPEC_COL].apply(remap_evt_path)

        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def _parse_label_ids(self, v) -> np.ndarray:
        """
        Parse label_ids into a multi-hot numpy vector of length num_classes.

        In parquet, label_ids is already a list-like: [3, 7, 8, 9, 37, 38, 59]
        Supports list, np.ndarray, and string (fallback).
        """
        vec = np.zeros(self.num_classes, dtype=np.float32)

        # list or array of ints
        if isinstance(v, (list, tuple, np.ndarray)):
            for idx in v:
                idx = int(idx)
                if 0 <= idx < self.num_classes:
                    vec[idx] = 1.0
            return vec

        # missing
        if isinstance(v, float) and np.isnan(v):
            return vec

        # string fallback e.g. "[3, 7, 8]" or "3,7,8" or "3 7 8"
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return vec
            s = s.replace("[", "").replace("]", "")
            for token in s.replace(",", " ").split():
                idx = int(token)
                if 0 <= idx < self.num_classes:
                    vec[idx] = 1.0
            return vec

        # anything else -> empty
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_path = Path(row[EVENTS_SPEC_COL])
        label_ids_val = row[EVENTS_LABEL_IDS_COL]

        x = _load_spec(spec_path)
        y_vec = self._parse_label_ids(label_ids_val)
        y = torch.from_numpy(y_vec).float()  # (C,)

        return x, y


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, normalize, title, out_path: Path):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(values, labels, ylabel, title, out_path: Path, ylim=(0, 1.0)):
    x = np.arange(len(labels))
    plt.figure(figsize=(max(8, len(labels) * 0.5), 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------------------------------------------------------
# Environment evaluation
# -------------------------------------------------------------------------

def evaluate_environment(split: str, batch_size: int):
    print(f"\n=== Evaluating ENVIRONMENT model on split='{split}' ===")

    id2label_env = load_env_label_map()
    class_names = [id2label_env[i] for i in sorted(id2label_env.keys())]

    dataset = EnvironmentEvalDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_environment_model(num_classes=len(class_names))

    all_logits = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0).astype(int)
    y_pred = logits.argmax(axis=1)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    top2 = top_k_accuracy_score(y_true, logits, k=2)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Top-2 acc: {top2:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1e-9)

    print("\nPer-class accuracy:")
    for name, a in zip(class_names, per_class_acc):
        print(f"  {name:15s}: {a:.3f}")

    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=3
    )
    print("\nClassification report:\n")
    print(report_str)

    # CSV outputs
    env_results_dir = RESULTS_ROOT / "environment" / split
    env_results_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics CSV
    overall_df = pd.DataFrame(
        {
            "metric": ["accuracy", "top2_accuracy"],
            "value": [acc, top2],
        }
    )
    overall_df.to_csv(env_results_dir / "overall_metrics.csv", index=False)

    # Per-class metrics CSV
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "class_id": np.arange(len(class_names)),
            "class_name": class_names,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "accuracy": per_class_acc,
        }
    )
    per_class_df.to_csv(env_results_dir / "per_class_metrics.csv", index=False)

    # dump raw confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(env_results_dir / "confusion_matrix_counts.csv")

    # Plots
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=False,
        title=f"Environment Confusion Matrix ({split}, counts)",
        out_path=env_results_dir / "confusion_matrix_counts.png",
    )
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=True,
        title=f"Environment Confusion Matrix ({split}, normalized)",
        out_path=env_results_dir / "confusion_matrix_normalized.png",
    )

    plot_bar(
        values=f1,
        labels=class_names,
        ylabel="F1-score",
        title=f"Environment per-class F1 ({split})",
        out_path=env_results_dir / "per_class_f1.png",
    )


# -------------------------------------------------------------------------
# Events evaluation (multi-label)
# -------------------------------------------------------------------------

def evaluate_events(split: str, batch_size: int):
    print(f"\n=== Evaluating EVENTS model on split='{split}' ===")

    id2label_events = load_event_label_map()
    num_classes = len(id2label_events)
    class_names = [id2label_events[i] for i in sorted(id2label_events.keys())]

    dataset = EventsEvalDataset(split=split, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_events_model(num_classes=num_classes)

    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)         # (N, C)
    y_true = np.concatenate(all_targets, axis=0)      # (N, C)
    y_pred = (probs >= EVENTS_THRESHOLD).astype(int)  # (N, C)

    # Overall metrics
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_rec = recall_score(y_true, y_pred, average="micro", zero_division=0)

    # Average precision (per class and macro mAP)
    ap_per_class = average_precision_score(y_true, probs, average=None)
    mAP = float(np.mean(ap_per_class))

    print(f"Micro-F1:        {micro_f1:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Micro-precision: {micro_prec:.4f}")
    print(f"Micro-recall:    {micro_rec:.4f}")
    print(f"mAP (macro):     {mAP:.4f}")

    # Per-class metrics
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    events_results_dir = RESULTS_ROOT / "events" / split
    events_results_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics CSV
    overall_df = pd.DataFrame(
        {
            "metric": [
                "micro_f1",
                "macro_f1",
                "micro_precision",
                "micro_recall",
                "mAP",
            ],
            "value": [micro_f1, macro_f1, micro_prec, micro_rec, mAP],
        }
    )
    overall_df.to_csv(events_results_dir / "overall_metrics.csv", index=False)

    # Per-class metrics CSV
    per_class_df = pd.DataFrame(
        {
            "class_id": np.arange(num_classes),
            "class_name": class_names,
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "support": support_c,
            "average_precision": ap_per_class,
        }
    )
    per_class_df.to_csv(events_results_dir / "per_class_metrics.csv", index=False)

    # Plots: F1 per class & AP per class
    plot_bar(
        values=f1_c,
        labels=class_names,
        ylabel="F1-score",
        title=f"Events per-class F1 ({split})",
        out_path=events_results_dir / "per_class_f1.png",
    )
    plot_bar(
        values=ap_per_class,
        labels=class_names,
        ylabel="Average Precision",
        title=f"Events per-class AP ({split})",
        out_path=events_results_dir / "per_class_AP.png",
        ylim=(0, 1.0),
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["env", "events", "both"],
        default="both",
        help="Which model(s) to evaluate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split name (ignored for now since parquet has no split columns).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    if args.task in ("env", "both"):
        evaluate_environment(split=args.split, batch_size=args.batch_size)

    if args.task in ("events", "both"):
        evaluate_events(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
