import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import torch

# importable /SurroundSound/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.live_demo.models_live import (
    DEVICE,
    load_models_and_labels,
)
from src.live_demo.feature_extraction import AudioFeatureExtractor
from src.live_demo.audio_utils import record_audio


SAMPLE_RATE = 16000
DEFAULT_RECORD_SECONDS = 3.0
TOP_K_EVENTS = 5


# -----------------------
# Cached resources
# -----------------------
@st.cache_resource
def get_models_labels_and_extractor():
    env_model, event_model, env_id2label, event_id2label = load_models_and_labels()
    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)
    return env_model, event_model, env_id2label, event_id2label, extractor


# -----------------------
# Inference
# -----------------------
def run_models_on_audio(
    env_model,
    event_model,
    extractor: AudioFeatureExtractor,
    audio: np.ndarray,
    env_id2label,
    event_id2label,
    device: str = DEVICE,
) -> Tuple[Tuple[str, float], List[Tuple[str, float]]]:

    # Separate env/events feature flows
    env_logmel = extractor.env_features(audio, sr=SAMPLE_RATE)   # (F, T)
    event_logmel = extractor.event_features(audio, sr=SAMPLE_RATE)  # (F, T)

    env_x = torch.from_numpy(env_logmel).unsqueeze(0).unsqueeze(0).to(device)
    event_x = torch.from_numpy(event_logmel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        env_logits = env_model(env_x)         # (1, C_env)
        event_logits = event_model(event_x)   # (1, C_event)

        env_probs = torch.softmax(env_logits, dim=1).cpu().numpy()[0]
        event_probs = torch.sigmoid(event_logits).cpu().numpy()[0]

    # Environment: single top prediction
    env_class_id = int(env_probs.argmax())
    env_conf = float(env_probs[env_class_id])
    env_label = env_id2label.get(env_class_id, f"class_{env_class_id}")

    # Events: top-k multi-label predictions
    top_indices = np.argsort(event_probs)[::-1][:TOP_K_EVENTS]
    event_results = []
    for idx in top_indices:
        conf = float(event_probs[idx])
        label = event_id2label.get(int(idx), f"event_{idx}")
        event_results.append((label, conf))

    return (env_label, env_conf), event_results, env_logmel


# -----------------------
# Plotting helpers
# -----------------------
def plot_waveform(audio: np.ndarray, sr: int = SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(audio) / sr, num=len(audio))
    ax.plot(t, audio)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    return fig


def plot_spectrogram(logmel: np.ndarray, sr: int = SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        logmel,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        ax=ax,
    )
    ax.set_title("Log-Mel Spectrogram (Environment features)")
    fig.colorbar(img, ax=ax, format="%+2.0f")
    fig.tight_layout()
    return fig


def plot_event_bars(events: List[Tuple[str, float]]):
    labels = [e[0] for e in events]
    confs = [e[1] for e in events]

    fig, ax = plt.subplots(figsize=(6, 3))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, confs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_title("Top Event Predictions")
    fig.tight_layout()
    return fig


# -----------------------
# Scene summarization
# -----------------------
def rule_based_summary(env_label: str, env_conf: float,
                       events: List[Tuple[str, float]]) -> str:
    """
    Simple confidence-aware summary with no fancy label cleanup.
    Uses only higher-confidence events.
    """
    strong = [(l, c) for (l, c) in events if c >= 0.4]

    env_name = env_label.replace("_", " ")

    if not strong:
        return (
            f"It sounds like you are in a {env_name} environment, "
            f"but no specific sound events stand out clearly."
        )

    strong_sorted = sorted(strong, key=lambda x: x[1], reverse=True)[:3]
    event_names = [l.replace("_", " ") for (l, _) in strong_sorted]

    if len(event_names) == 1:
        events_text = event_names[0]
    elif len(event_names) == 2:
        events_text = " and ".join(event_names)
    else:
        events_text = ", ".join(event_names[:-1]) + f", and {event_names[-1]}"

    return (
        f"It sounds like you are in a {env_name} environment with "
        f"{events_text} in the background."
    )


def llm_summary(env_label: str, env_conf: float,
                events: List[Tuple[str, float]]) -> str:
    """
    Uses a real LLM if OPENAI_API_KEY is set; otherwise falls back
    to the rule-based summary.

    Sends raw labels + confidences and explicitly tells the LLM to
    infer a plausible real-world scene based on higher-confidence
    events.
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rule_based_summary(env_label, env_conf, events)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    events_for_llm = [
        {
            "label": label,
            "confidence": round(float(conf), 3),
        }
        for label, conf in events
    ]

    env_payload = {
        "label": env_label,
        "confidence": round(float(env_conf), 3),
    }

    system_msg = (
        "You are an acoustic scene summarizer. "
        "You receive the output of an environment classifier and a multi-label "
        "event classifier, with confidences between 0 and 1. "
        "Your job is to infer what real-world scene the listener is in and "
        "describe it in one clear, natural sentence."
    )

    user_payload = {
        "environment": env_payload,
        "events": events_for_llm,
        "instructions": [
            "Focus mainly on events with confidence >= 0.4.",
            "Use your world knowledge to guess a plausible scene "
            "(for example: 'a rainforest during a heavy rainstorm', "
            "'a busy city street at night').",
            "Do not mention confidence numbers.",
            "Write exactly one sentence.",
        ],
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Here is the model output as JSON. "
                        "Respond with one natural-language sentence summarizing "
                        "the likely scene.\n\n"
                        + json.dumps(user_payload, indent=2)
                    ),
                },
            ],
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        if not text:
            return rule_based_summary(env_label, env_conf, events)
        return text
    except Exception:
        return rule_based_summary(env_label, env_conf, events)



# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="Surround Sound Live Demo", layout="wide")
    st.title("Surround Sound – Live Demo")

    st.markdown(
        "Record a short clip with your microphone and classify the "
        "environment and sound events. Features are computed with the "
        "same log-mel pipeline used for training."
    )

    env_model, event_model, env_id2label, event_id2label, extractor = (
        get_models_labels_and_extractor()
    )

    col_left, col_right = st.columns([1, 1])

    with st.sidebar:
        st.header("Settings")
        rec_sec = st.slider(
            "Recording length (seconds)",
            min_value=1.0,
            max_value=15.0,
            value=DEFAULT_RECORD_SECONDS,
            step=0.5,
        )
        st.write(f"Sample rate: {SAMPLE_RATE} Hz")
        st.write("Channels: mono")
        st.caption("Start with ~3 seconds for stable predictions.")

    if st.button("🎙️ Record from microphone"):
        audio = record_audio(rec_sec, fs=SAMPLE_RATE)

        # Run inference
        (env_label, env_conf), event_results, env_logmel = run_models_on_audio(
            env_model,
            event_model,
            extractor,
            audio,
            env_id2label,
            event_id2label,
            device=DEVICE,
        )

        # Left column: visuals
        with col_left:
            st.subheader("Waveform")
            st.pyplot(plot_waveform(audio, sr=SAMPLE_RATE))

            st.subheader("Spectrogram (env log-mel)")
            st.pyplot(plot_spectrogram(env_logmel, sr=SAMPLE_RATE))

        # Right column: predictions & summary
        with col_right:
            st.subheader("Environment Prediction")
            st.write(f"**{env_label}** ({env_conf:.2f} confidence)")
            st.progress(env_conf)

            st.subheader(f"Top {TOP_K_EVENTS} Event Predictions")
            st.pyplot(plot_event_bars(event_results))

            st.subheader("Scene Summary (LLM-backed)")
            summary_text = llm_summary(env_label, env_conf, event_results)
            st.write(summary_text)

    st.markdown("---")
    st.caption(
        "v1 demo"
    )


if __name__ == "__main__":
    main()
