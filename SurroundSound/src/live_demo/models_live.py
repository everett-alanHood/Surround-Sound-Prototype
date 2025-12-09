from pathlib import Path
from typing import Dict, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project root:  /SurroundSound/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ENV_DIR = PROJECT_ROOT / "data" / "environment"
DATA_EVENTS_DIR = PROJECT_ROOT / "data" / "events"

OUTPUT_ENV_DIR = PROJECT_ROOT / "output" / "environment"
OUTPUT_EVENTS_DIR = PROJECT_ROOT / "output" / "events"

ENV_CKPT = OUTPUT_ENV_DIR / "best_environment_model.pt"
EVENT_CKPT = OUTPUT_EVENTS_DIR / "best_events_model.pt"

ENV_LABELS_PATH = DATA_ENV_DIR / "id_to_label.json"
EVENT_LABELS_PATH = DATA_EVENTS_DIR / "id_to_label.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model architectures (copied from training scripts)
# ============================================================

class EnvironmentCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = 32

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


class EventsCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.4)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        logits = self.fc(x)
        return logits


# ============================================================
# Label maps
# ============================================================

def _load_id_to_label(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"id_to_label.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def load_env_label_map() -> Dict[int, str]:
    return _load_id_to_label(ENV_LABELS_PATH)


def load_event_label_map() -> Dict[int, str]:
    return _load_id_to_label(EVENT_LABELS_PATH)


# ============================================================
# Model loading
# ============================================================

def load_environment_model(num_classes: int) -> nn.Module:
    if not ENV_CKPT.exists():
        raise FileNotFoundError(f"Environment checkpoint not found: {ENV_CKPT}")
    ckpt = torch.load(ENV_CKPT, map_location=DEVICE)

    model = EnvironmentCNN(num_classes=num_classes)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_events_model(num_classes: int) -> nn.Module:
    if not EVENT_CKPT.exists():
        raise FileNotFoundError(f"Events checkpoint not found: {EVENT_CKPT}")
    ckpt = torch.load(EVENT_CKPT, map_location=DEVICE)

    model = EventsCNN(num_classes=num_classes)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def load_models_and_labels() -> Tuple[nn.Module, nn.Module, Dict[int, str], Dict[int, str]]:
    env_id2label = load_env_label_map()
    event_id2label = load_event_label_map()

    env_model = load_environment_model(num_classes=len(env_id2label))
    event_model = load_events_model(num_classes=len(event_id2label))

    return env_model, event_model, env_id2label, event_id2label
