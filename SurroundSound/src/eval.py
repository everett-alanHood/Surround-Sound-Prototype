"""
Unified evaluation script for Surround Sound V2.

- Environment model: 1D CNN over VGGish embeddings (10x128)
- Events model:      2D CNN over log-mel spectrograms, multi-label
- Speech model:      2D CNN over log-mel, 3-class detection

Outputs:
    src/results/environment/<split>/
    src/results/events/<split>/
    src/results/speech/<split>/

Usage:
    python src/eval.py --task env
    python src/eval.py --task events
    python src/eval.py --task speech
    python src/eval.py --task both
    python src/eval.py --task all
"""

import argparse
import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
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

from live_demo.models_live import (
    DATA_ENV_DIR,
    DATA_EVENTS_DIR,
    DATA_SPEECH_DIR,
    OUTPUT_ENV_DIR,
    OUTPUT_EVENTS_DIR,
    OUTPUT_SPEECH_DIR,
    load_environment_model,
    load_events_model,
    load_speech_model,
    load_env_label_map,
    load_event_label_map,
    load_speech_label_map,
    DEVICE,
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

# ── Index paths ───────────────────────────────────────────────────────────────

ENV_INDEX_PARQ    = DATA_ENV_DIR    / "data_index.parquet"
ENV_INDEX_CSV     = DATA_ENV_DIR    / "data_index.csv"
EVENTS_INDEX_PARQ = DATA_EVENTS_DIR / "data_index.parquet"
EVENTS_INDEX_CSV  = DATA_EVENTS_DIR / "data_index.csv"
SPEECH_INDEX_PARQ = DATA_SPEECH_DIR / "data_index.parquet"
SPEECH_INDEX_CSV  = DATA_SPEECH_DIR / "data_index.csv"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_index(parq: Path, csv: Path) -> pd.DataFrame:
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No index found at {parq} or {csv}")


def parse_label_ids(v) -> list:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [int(x) for x in v]
    if isinstance(v, str):
        s = v.strip().replace("[", "").replace("]", "")
        if not s:
            return []
        return [int(t) for t in s.replace(",", " ").split()]
    return []


# ── Environment dataset ───────────────────────────────────────────────────────

class EnvironmentEvalDataset(Dataset):
    N_FRAMES  = 10
    EMBED_DIM = 128

    def __init__(self):
        df = load_index(ENV_INDEX_PARQ, ENV_INDEX_CSV)
        feat_root = DATA_ENV_DIR / "processed" / "features"
        df["feature_path"] = df["feature_path"].apply(
            lambda p: str(feat_root / Path(str(p)).name)
        )
        if "y" in df.columns:
            df["label_id"] = df["y"].astype(int)
        elif "primary_label" in df.columns:
            with open(DATA_ENV_DIR / "label_to_id.json") as f:
                l2i = json.load(f)
            df["label_id"] = df["primary_label"].map(l2i)
        df = df[df["label_id"].notna()].copy()
        df["label_id"] = df["label_id"].astype(int)
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = np.load(row["feature_path"]).astype(np.float32)
        if emb.shape[0] < self.N_FRAMES:
            emb = np.pad(emb, ((0, self.N_FRAMES - emb.shape[0]), (0, 0)))
        else:
            emb = emb[:self.N_FRAMES]
        return torch.from_numpy(emb.T), int(row["label_id"])


# ── Events dataset ────────────────────────────────────────────────────────────

class EventsEvalDataset(Dataset):
    DB_MIN = -80.0
    DB_MAX  = 0.0
    TARGET_FRAMES = 313

    def __init__(self, num_classes: int):
        df = load_index(EVENTS_INDEX_PARQ, EVENTS_INDEX_CSV)
        feat_root = DATA_EVENTS_DIR / "processed" / "features"
        df["feature_path"] = df["feature_path"].apply(
            lambda p: str(feat_root / Path(str(p)).name)
        )
        df["label_ids"] = df["label_ids"].apply(parse_label_ids)
        df = df[df["label_ids"].map(len) > 0].reset_index(drop=True)
        self.df = df
        self.num_classes = num_classes

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        spec = np.load(row["feature_path"]).astype(np.float32)
        spec = np.clip((spec - self.DB_MIN) / (self.DB_MAX - self.DB_MIN), 0.0, 1.0)
        T = spec.shape[1]
        if T < self.TARGET_FRAMES:
            spec = np.pad(spec, ((0, 0), (0, self.TARGET_FRAMES - T)))
        else:
            spec = spec[:, :self.TARGET_FRAMES]
        x = torch.from_numpy(spec).unsqueeze(0)
        y = np.zeros(self.num_classes, dtype=np.float32)
        for cid in row["label_ids"]:
            if 0 <= cid < self.num_classes:
                y[cid] = 1.0
        return x, torch.from_numpy(y)


# ── Speech dataset ────────────────────────────────────────────────────────────

class SpeechEvalDataset(Dataset):
    TARGET_FRAMES = 313

    def __init__(self, label_to_id: dict):
        df = load_index(SPEECH_INDEX_PARQ, SPEECH_INDEX_CSV)
        feat_root = DATA_SPEECH_DIR / "processed" / "features"
        df["feature_path"] = df["feature_path"].apply(
            lambda p: str(feat_root / Path(str(p)).name)
        )

        # Remap index labels to match checkpoint label map
        # index uses 'conversation', checkpoint uses 'multi_speaker'
        remap = {"conversation": "multi_speaker"}
        if "label" in df.columns:
            df["label"] = df["label"].replace(remap)
            df["y"] = df["label"].map(label_to_id)
        elif "y" not in df.columns:
            raise KeyError("Speech index missing 'label' or 'y' column.")

        df = df[df["y"].notna()].copy()
        df["y"] = df["y"].astype(int)
        df = df[df["feature_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
        self.df = df
        print(f"[INFO] Speech eval dataset: {len(df)} clips")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        feat = np.load(row["feature_path"]).astype(np.float32)  # (128, T)
        T = feat.shape[1]
        if T < self.TARGET_FRAMES:
            feat = np.pad(feat, ((0, 0), (0, self.TARGET_FRAMES - T)))
        else:
            feat = feat[:, :self.TARGET_FRAMES]
        x = torch.from_numpy(feat).unsqueeze(0)  # (1, 128, T)
        return x, int(row["y"])


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, normalize, title, out_path: Path):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.7)))
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
    plt.figure(figsize=(max(8, len(labels) * 0.45), 4))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── Environment evaluation ────────────────────────────────────────────────────

def evaluate_environment(split: str, batch_size: int):
    print(f"\n=== Evaluating ENVIRONMENT model (split='{split}') ===")

    id2label    = load_env_label_map()
    class_names = [id2label[i] for i in sorted(id2label)]
    num_classes = len(class_names)

    dataset = EnvironmentEvalDataset()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model   = load_environment_model(num_classes=num_classes)

    all_logits, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(DEVICE)).cpu().numpy())
            all_targets.append(y.numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets).astype(int)
    y_pred = logits.argmax(axis=1)

    acc  = accuracy_score(y_true, y_pred)
    top2 = top_k_accuracy_score(y_true, logits, k=min(2, num_classes))

    print(f"Accuracy:  {acc:.4f}")
    print(f"Top-2 acc: {top2:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1e-9)

    print("\nPer-class accuracy:")
    for name, a in zip(class_names, per_class_acc):
        print(f"  {name:20s}: {a:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    out_dir = RESULTS_ROOT / "environment" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"metric": ["accuracy", "top2_accuracy"], "value": [acc, top2]}).to_csv(
        out_dir / "overall_metrics.csv", index=False)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0)
    pd.DataFrame({
        "class_id": np.arange(num_classes), "class_name": class_names,
        "precision": prec, "recall": rec, "f1": f1,
        "support": support, "accuracy": per_class_acc,
    }).to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / "confusion_matrix_counts.csv")

    plot_confusion_matrix(cm, class_names, normalize=False,
        title=f"Environment Confusion Matrix ({split}, counts)",
        out_path=out_dir / "confusion_matrix_counts.png")
    plot_confusion_matrix(cm, class_names, normalize=True,
        title=f"Environment Confusion Matrix ({split}, normalized)",
        out_path=out_dir / "confusion_matrix_normalized.png")
    plot_bar(f1, class_names, "F1-score",
        f"Environment per-class F1 ({split})", out_path=out_dir / "per_class_f1.png")

    print(f"[INFO] Results -> {out_dir}")


# ── Events evaluation ─────────────────────────────────────────────────────────

def evaluate_events(split: str, batch_size: int):
    print(f"\n=== Evaluating EVENTS model (split='{split}') ===")

    id2label    = load_event_label_map()
    num_classes = len(id2label)
    class_names = [id2label[i] for i in sorted(id2label)]

    config_path = OUTPUT_EVENTS_DIR / "events_training_config.json"
    threshold   = 0.5
    if config_path.exists():
        with config_path.open() as f:
            threshold = float(json.load(f).get("best_threshold", 0.5))
        print(f"[INFO] Using tuned threshold: {threshold:.3f}")

    dataset = EventsEvalDataset(num_classes=num_classes)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model   = load_events_model(num_classes=num_classes)

    all_probs, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
            all_targets.append(y.numpy())

    probs  = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    y_pred = (probs >= threshold).astype(int)

    micro_f1   = f1_score(y_true, y_pred, average="micro",   zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    micro_prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_rec  = recall_score(y_true, y_pred,    average="micro", zero_division=0)
    ap_per_cls = average_precision_score(y_true, probs, average=None)
    mAP        = float(np.mean(ap_per_cls))

    print(f"Threshold:       {threshold:.3f}")
    print(f"Micro-F1:        {micro_f1:.4f}")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Micro-precision: {micro_prec:.4f}")
    print(f"Micro-recall:    {micro_rec:.4f}")
    print(f"mAP (macro):     {mAP:.4f}")

    out_dir = RESULTS_ROOT / "events" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "metric": ["micro_f1","macro_f1","micro_precision","micro_recall","mAP","threshold"],
        "value":  [micro_f1, macro_f1, micro_prec, micro_rec, mAP, threshold],
    }).to_csv(out_dir / "overall_metrics.csv", index=False)

    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    pd.DataFrame({
        "class_id": np.arange(num_classes), "class_name": class_names,
        "precision": prec_c, "recall": rec_c, "f1": f1_c,
        "support": sup_c, "average_precision": ap_per_cls,
    }).to_csv(out_dir / "per_class_metrics.csv", index=False)

    plot_bar(f1_c, class_names, "F1-score",
        f"Events per-class F1 ({split})", out_path=out_dir / "per_class_f1.png")
    plot_bar(ap_per_cls, class_names, "Average Precision",
        f"Events per-class AP ({split})", out_path=out_dir / "per_class_AP.png")

    print(f"[INFO] Results -> {out_dir}")


# ── Speech evaluation ─────────────────────────────────────────────────────────

def evaluate_speech(split: str, batch_size: int):
    print(f"\n=== Evaluating SPEECH model (split='{split}') ===")

    # Load label maps from checkpoint to guarantee alignment
    from live_demo.models_live import OUTPUT_SPEECH_DIR, SPEECH_CKPT
    ckpt = torch.load(SPEECH_CKPT, map_location="cpu", weights_only=False)
    label_to_id = ckpt.get("label_to_id", {})
    id2label    = {int(v): k for k, v in label_to_id.items()}
    class_names = [id2label[i] for i in sorted(id2label)]
    num_classes = len(class_names)
    print(f"[INFO] Using checkpoint label map: {label_to_id}")

    dataset = SpeechEvalDataset(label_to_id=label_to_id)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model   = load_speech_model(num_classes=num_classes)

    all_logits, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(DEVICE)).cpu().numpy())
            all_targets.append(np.asarray(y, dtype=np.int64))

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_targets).astype(int)
    y_pred = logits.argmax(axis=1)

    acc  = accuracy_score(y_true, y_pred)
    top2 = top_k_accuracy_score(y_true, logits, k=min(2, num_classes), labels=np.arange(num_classes))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Accuracy:   {acc:.4f}")
    print(f"Top-2 acc:  {top2:.4f}")
    print(f"Macro-F1:   {macro_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1e-9)

    print("\nPer-class accuracy:")
    for name, a in zip(class_names, per_class_acc):
        print(f"  {name:20s}: {a:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    out_dir = RESULTS_ROOT / "speech" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "metric": ["accuracy", "top2_accuracy", "macro_f1"],
        "value":  [acc, top2, macro_f1],
    }).to_csv(out_dir / "overall_metrics.csv", index=False)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(num_classes), zero_division=0)
    pd.DataFrame({
        "class_id": np.arange(num_classes), "class_name": class_names,
        "precision": prec, "recall": rec, "f1": f1,
        "support": support, "accuracy": per_class_acc,
    }).to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(out_dir / "confusion_matrix_counts.csv")

    plot_confusion_matrix(cm, class_names, normalize=False,
        title=f"Speech Confusion Matrix ({split}, counts)",
        out_path=out_dir / "confusion_matrix_counts.png")
    plot_confusion_matrix(cm, class_names, normalize=True,
        title=f"Speech Confusion Matrix ({split}, normalized)",
        out_path=out_dir / "confusion_matrix_normalized.png")
    plot_bar(f1, class_names, "F1-score",
        f"Speech per-class F1 ({split})", out_path=out_dir / "per_class_f1.png")

    print(f"[INFO] Results -> {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["env", "events", "speech", "both", "all"],
                    default="all",
                    help="'both' = env+events, 'all' = env+events+speech")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if args.task in ("env", "both", "all"):
        evaluate_environment(split=args.split, batch_size=args.batch_size)
    if args.task in ("events", "both", "all"):
        evaluate_events(split=args.split, batch_size=args.batch_size)
    if args.task in ("speech", "all"):
        evaluate_speech(split=args.split, batch_size=args.batch_size)


if __name__ == "__main__":
    main()