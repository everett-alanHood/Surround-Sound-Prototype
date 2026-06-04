"""
transcription.py

Speech detection + transcription pipeline for Surround Sound V2.

Uses:
  - SpeechCNN detector: no_speech / single_speaker / conversation
  - faster-whisper large-v3: transcription (only when speech detected)

Designed for post-recording transcription in the Streamlit demo,
with hooks for future real-time streaming.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
OUTPUT_SPEECH = PROJECT_ROOT / "output" / "speech"
WHISPER_CACHE = PROJECT_ROOT / "models" / "whisper"

SPEECH_CKPT       = OUTPUT_SPEECH / "best_speech_model.pt"
SPEECH_CONFIG     = OUTPUT_SPEECH / "speech_training_config.json"

SAMPLE_RATE   = 16000
TARGET_FRAMES = 313
N_MELS        = 128


# ── Speech detector model (mirrors 02_speech_training.py) ────────────────────

class SE2d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False), nn.ReLU(),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        s = x.mean(dim=[2, 3])
        return x * self.fc(s)[:, :, None, None]


class ResConv2dBlock(nn.Module):
    def __init__(self, channels, se_reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SE2d(channels, se_reduction)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.se(self.bn2(self.conv2(out))) + x)


class SpeechCNN(nn.Module):
    def __init__(self, num_classes=3, base_channels=64, dropout=0.4, se_reduction=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block1 = ResConv2dBlock(base_channels, se_reduction)
        self.down1  = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = ResConv2dBlock(base_channels*2, se_reduction)
        self.down2  = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        return self.fc(self.drop(self.gap(x).flatten(1)))


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TranscriptionSegment:
    start:      float
    end:        float
    text:       str
    avg_logprob: float = 0.0


@dataclass
class TranscriptionResult:
    speech_label:    str           # no_speech / single_speaker / conversation
    speech_conf:     float
    speech_probs:    dict          # all class probabilities
    transcript:      str           # full joined transcript
    segments:        List[TranscriptionSegment] = field(default_factory=list)
    language:        str  = "en"
    duration_s:      float = 0.0
    elapsed_s:       float = 0.0
    whisper_used:    bool  = False
    error:           Optional[str] = None


# ── Main pipeline class ───────────────────────────────────────────────────────

class SpeechPipeline:
    """
    Combined speech detector + Whisper transcriber.

    Usage:
        pipeline = SpeechPipeline()
        result   = pipeline.process(audio_array)
        print(result.speech_label, result.transcript)
    """

    LABEL_NAMES = ["no_speech", "single_speaker", "conversation"]

    def __init__(
        self,
        detector_ckpt: Path = SPEECH_CKPT,
        whisper_model_size: str = "large-v3",
        whisper_cache: Path = WHISPER_CACHE,
        device: Optional[str] = None,
        speech_threshold: float = 0.5,
    ):
        self.device           = self._get_device(device)
        self.speech_threshold = speech_threshold
        self.whisper_model_size = whisper_model_size
        self.whisper_cache    = whisper_cache

        self._detector    = None
        self._whisper     = None
        self._label_to_id = {n: i for i, n in enumerate(self.LABEL_NAMES)}
        self._id_to_label = {i: n for n, i in self._label_to_id.items()}

        # Load detector if checkpoint exists
        if detector_ckpt.exists():
            self._load_detector(detector_ckpt)
        else:
            print(f"[WARN] Speech detector checkpoint not found: {detector_ckpt}")
            print("       Run 02_speech_training.py first.")

    # ── Device ────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.zeros(1).cuda()
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")

    # ── Detector loading ──────────────────────────────────────────────────────

    def _load_detector(self, ckpt_path: Path) -> None:
        ckpt   = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        hp     = ckpt.get("hparams", {})
        model  = SpeechCNN(
            num_classes   = len(self.LABEL_NAMES),
            base_channels = int(hp.get("base_channels", 64)),
            dropout       = float(hp.get("dropout", 0.4)),
            se_reduction  = int(hp.get("se_reduction", 8)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self._detector = model
        print(f"[INFO] Speech detector loaded from {ckpt_path.name}")

        # Override label maps from checkpoint if available
        if "label_to_id" in ckpt:
            self._label_to_id = ckpt["label_to_id"]
            self._id_to_label = {int(v): k for k, v in self._label_to_id.items()}

    # ── Whisper loading (lazy) ────────────────────────────────────────────────

    def _load_whisper(self) -> bool:
        if self._whisper is not None:
            return True
        try:
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device.type == "cuda" else "int8"
            print(f"[INFO] Loading Whisper {self.whisper_model_size} ({compute_type})...")
            self._whisper = WhisperModel(
                self.whisper_model_size,
                device=self.device.type,
                compute_type=compute_type,
                download_root=str(self.whisper_cache),
            )
            print("[INFO] Whisper ready.")
            return True
        except ImportError:
            print("[WARN] faster-whisper not installed: pip install faster-whisper")
            return False
        except Exception as e:
            print(f"[WARN] Could not load Whisper: {e}")
            return False

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_logmel(self, audio: np.ndarray) -> np.ndarray:
        """Compute normalized log-mel for the detector."""
        import librosa
        y = audio.astype(np.float32)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        mel    = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=512, fmax=8000,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)

        # Pad/trim to TARGET_FRAMES
        T = logmel.shape[1]
        if T < TARGET_FRAMES:
            logmel = np.pad(logmel, ((0, 0), (0, TARGET_FRAMES - T)))
        else:
            logmel = logmel[:, :TARGET_FRAMES]

        return logmel.astype(np.float32)

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, audio: np.ndarray) -> Tuple[str, float, dict]:
        """
        Run speech detector on audio array.
        Returns (label, confidence, {label: prob}).
        Falls back to 'single_speaker' if no detector loaded.
        """
        if self._detector is None:
            return "single_speaker", 1.0, {n: 0.0 for n in self.LABEL_NAMES}

        logmel = self._extract_logmel(audio)
        x      = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self._detector(x), dim=1).cpu().numpy()[0]

        class_id = int(probs.argmax())
        label    = self._id_to_label.get(class_id, "single_speaker")
        conf     = float(probs[class_id])
        prob_map = {self._id_to_label.get(i, str(i)): float(p)
                    for i, p in enumerate(probs)}

        return label, conf, prob_map

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.
        Audio must be float32 at SAMPLE_RATE Hz.
        """
        if not self._load_whisper():
            return TranscriptionResult(
                speech_label="unknown", speech_conf=0.0, speech_probs={},
                transcript="", error="Whisper not available",
            )

        import tempfile, soundfile as sf

        t0 = time.perf_counter()
        try:
            # Write to temp WAV for faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio, SAMPLE_RATE)

            segments_gen, info = self._whisper.transcribe(
                tmp_path,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            segments = []
            text_parts = []
            for seg in segments_gen:
                segments.append(TranscriptionSegment(
                    start=seg.start, end=seg.end,
                    text=seg.text.strip(),
                    avg_logprob=getattr(seg, "avg_logprob", 0.0),
                ))
                text_parts.append(seg.text.strip())

            Path(tmp_path).unlink(missing_ok=True)

            return TranscriptionResult(
                speech_label="",
                speech_conf=0.0,
                speech_probs={},
                transcript=" ".join(text_parts),
                segments=segments,
                language=info.language,
                duration_s=round(len(audio) / SAMPLE_RATE, 2),
                elapsed_s=round(time.perf_counter() - t0, 3),
                whisper_used=True,
            )

        except Exception as e:
            return TranscriptionResult(
                speech_label="", speech_conf=0.0, speech_probs={},
                transcript="", error=str(e),
            )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Full pipeline: detect speech, then transcribe if speech found.
        audio: float32 numpy array at 16kHz mono.
        """
        label, conf, prob_map = self.detect(audio)

        result = TranscriptionResult(
            speech_label=label,
            speech_conf=conf,
            speech_probs=prob_map,
            transcript="",
            duration_s=round(len(audio) / SAMPLE_RATE, 2),
        )

        if label == "no_speech":
            result.transcript = ""
            return result

        # Transcribe
        t_result = self.transcribe(audio)
        result.transcript   = t_result.transcript
        result.segments     = t_result.segments
        result.language     = t_result.language
        result.elapsed_s    = t_result.elapsed_s
        result.whisper_used = t_result.whisper_used
        result.error        = t_result.error

        return result


# ── Conversation recorder ─────────────────────────────────────────────────────

@dataclass
class ConversationRecord:
    """Accumulates transcription results across multiple recordings."""
    entries: List[dict] = field(default_factory=list)

    def add(self, result: TranscriptionResult, timestamp: Optional[float] = None) -> None:
        self.entries.append({
            "timestamp":   timestamp or time.time(),
            "label":       result.speech_label,
            "confidence":  round(result.speech_conf, 3),
            "transcript":  result.transcript,
            "duration_s":  result.duration_s,
            "elapsed_s":   result.elapsed_s,
            "segments":    [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.segments
            ],
        })

    def full_transcript(self) -> str:
        """Join all non-empty transcripts in order."""
        return " ".join(
            e["transcript"] for e in self.entries
            if e["transcript"] and e["label"] != "no_speech"
        )

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "full_transcript": self.full_transcript(),
                "entries": self.entries,
            }, f, indent=2)
        print(f"[INFO] Conversation saved -> {path}")

    def clear(self) -> None:
        self.entries.clear()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import librosa

    print("Loading pipeline...")
    pipeline = SpeechPipeline()

    # Generate a test tone (silence) as a smoke test
    audio = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    print("\nProcessing 5s silence...")
    result = pipeline.process(audio)
    print(f"  Label:      {result.speech_label} ({result.speech_conf:.2f})")
    print(f"  Transcript: '{result.transcript}'")
    print(f"  Whisper:    {result.whisper_used}")