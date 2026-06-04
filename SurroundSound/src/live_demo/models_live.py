"""
models_live.py

Model architectures and loading utilities for Surround Sound V2.

Environment model: 1D CNN over VGGish embeddings (10x128)
Events model:      2D CNN over log-mel spectrograms with SE + residual blocks
Speech model:      2D CNN over log-mel for speech detection (no_speech/single_speaker/conversation)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ENV_DIR    = PROJECT_ROOT / "data" / "environment"
DATA_EVENTS_DIR = PROJECT_ROOT / "data" / "events"
DATA_SPEECH_DIR = PROJECT_ROOT / "data" / "speech"

OUTPUT_ENV_DIR    = PROJECT_ROOT / "output" / "environment"
OUTPUT_EVENTS_DIR = PROJECT_ROOT / "output" / "events"
OUTPUT_SPEECH_DIR = PROJECT_ROOT / "output" / "speech"

ENV_CKPT    = OUTPUT_ENV_DIR    / "best_environment_model.pt"
EVENT_CKPT  = OUTPUT_EVENTS_DIR / "best_events_model.pt"
SPEECH_CKPT = OUTPUT_SPEECH_DIR / "best_speech_model.pt"

ENV_LABELS_PATH    = DATA_ENV_DIR    / "id_to_label.json"
EVENT_LABELS_PATH  = DATA_EVENTS_DIR / "id_to_label.json"
SPEECH_LABELS_PATH = DATA_SPEECH_DIR / "id_to_label.json"

WHISPER_CACHE = PROJECT_ROOT / "models" / "whisper"


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.zeros(1).cuda()
        return "cuda"
    except RuntimeError:
        return "cpu"

DEVICE = _get_device()


# ── Shared building blocks ────────────────────────────────────────────────────

class SE1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x.mean(dim=2)).unsqueeze(2)


class ResConv1dBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)
        self.se    = SE1d(channels, reduction=se_reduction)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.se(self.bn2(self.conv2(out))) + x)


class SE2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x.mean(dim=[2, 3]))[:, :, None, None]


class ResConv2dBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SE2d(channels, reduction=se_reduction)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.se(self.bn2(self.conv2(out))) + x)


# ── Environment CNN (1D CNN over VGGish embeddings) ───────────────────────────

class EnvironmentCNN(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128,
                 base_channels: int = 256, dropout: float = 0.4, se_reduction: int = 8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(embed_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels), nn.ReLU(),
        )
        self.block1 = ResConv1dBlock(base_channels, se_reduction)
        self.block2 = ResConv1dBlock(base_channels, se_reduction)
        self.down   = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm1d(base_channels // 2), nn.ReLU(),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels // 2, num_classes)

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.down(x)
        return self.fc(self.drop(self.gap(x).squeeze(-1)))


# ── Events CNN (2D CNN with SE + residual) ────────────────────────────────────

class EventsCNN(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 128,
                 dropout: float = 0.4, se_reduction: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block1 = ResConv2dBlock(base_channels, se_reduction)
        self.down1  = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = ResConv2dBlock(base_channels * 2, se_reduction)
        self.down2  = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block3 = ResConv2dBlock(base_channels * 4, se_reduction)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))


# ── Speech CNN (2D CNN for speech detection) ──────────────────────────────────

class SpeechCNN(nn.Module):
    """
    2D CNN over log-mel for speech scene detection.
    Input:  (B, 1, 128, T)
    Output: (B, num_classes)  — no_speech / single_speaker / conversation
    """
    def __init__(self, num_classes: int = 3, base_channels: int = 64,
                 dropout: float = 0.4, se_reduction: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block1 = ResConv2dBlock(base_channels, se_reduction)
        self.down1  = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = ResConv2dBlock(base_channels * 2, se_reduction)
        self.down2  = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))


# ── Label maps ────────────────────────────────────────────────────────────────

def _load_id_to_label(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"id_to_label.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_env_label_map()    -> Dict[int, str]: return _load_id_to_label(ENV_LABELS_PATH)
def load_event_label_map()  -> Dict[int, str]: return _load_id_to_label(EVENT_LABELS_PATH)
def load_speech_label_map() -> Dict[int, str]: return _load_id_to_label(SPEECH_LABELS_PATH)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_environment_model(num_classes: Optional[int] = None) -> nn.Module:
    if not ENV_CKPT.exists():
        raise FileNotFoundError(f"Environment checkpoint not found: {ENV_CKPT}")
    ckpt = torch.load(ENV_CKPT, map_location=DEVICE, weights_only=False)
    hp   = ckpt.get("hparams", {})
    if num_classes is None:
        l2i = ckpt.get("label_to_id", {})
        num_classes = len(l2i) if l2i else len(load_env_label_map())
    model = EnvironmentCNN(
        num_classes=num_classes,
        base_channels=int(hp.get("base_channels", 256)),
        dropout=float(hp.get("dropout", 0.4)),
        se_reduction=int(hp.get("se_reduction", 8)),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(DEVICE).eval()
    print(f"[INFO] Environment model loaded ({num_classes} classes)")
    return model


def load_events_model(num_classes: Optional[int] = None) -> nn.Module:
    if not EVENT_CKPT.exists():
        raise FileNotFoundError(f"Events checkpoint not found: {EVENT_CKPT}")
    ckpt = torch.load(EVENT_CKPT, map_location=DEVICE, weights_only=False)
    hp   = ckpt.get("hparams", {})
    if num_classes is None:
        num_classes = ckpt.get("num_classes", len(load_event_label_map()))
    model = EventsCNN(
        num_classes=num_classes,
        base_channels=int(hp.get("base_channels", 128)),
        dropout=float(hp.get("dropout", 0.4)),
        se_reduction=int(hp.get("se_reduction", 8)),
    )
    model.load_state_dict(ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
    model.to(DEVICE).eval()
    print(f"[INFO] Events model loaded ({num_classes} classes)")
    return model


def load_speech_model(num_classes: Optional[int] = None) -> nn.Module:
    if not SPEECH_CKPT.exists():
        raise FileNotFoundError(f"Speech checkpoint not found: {SPEECH_CKPT}")
    ckpt = torch.load(SPEECH_CKPT, map_location=DEVICE, weights_only=False)
    hp   = ckpt.get("hparams", {})
    if num_classes is None:
        l2i = ckpt.get("label_to_id", {})
        num_classes = len(l2i) if l2i else len(load_speech_label_map())
    model = SpeechCNN(
        num_classes=num_classes,
        base_channels=int(hp.get("base_channels", 64)),
        dropout=float(hp.get("dropout", 0.4)),
        se_reduction=int(hp.get("se_reduction", 8)),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(DEVICE).eval()
    print(f"[INFO] Speech model loaded ({num_classes} classes)")
    return model


def load_models_and_labels() -> Tuple[
    nn.Module, nn.Module, nn.Module,
    Dict[int, str], Dict[int, str], Dict[int, str]
]:
    env_id2label    = load_env_label_map()
    event_id2label  = load_event_label_map()
    speech_id2label = load_speech_label_map()
    env_model       = load_environment_model(num_classes=len(env_id2label))
    event_model     = load_events_model(num_classes=len(event_id2label))
    speech_model    = load_speech_model(num_classes=len(speech_id2label))
    return env_model, event_model, speech_model, env_id2label, event_id2label, speech_id2label