import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.live_demo.models_live import (
    DEVICE,
    OUTPUT_EVENTS_DIR,
    OUTPUT_SPEECH_DIR,
    WHISPER_CACHE,
    load_models_and_labels,
)
from src.live_demo.feature_extraction import AudioFeatureExtractor
from src.live_demo.audio_utils import record_audio

SAMPLE_RATE            = 16000
DEFAULT_RECORD_SECONDS = 10.0
TOP_K_EVENTS           = 15


# ── Config loading ────────────────────────────────────────────────────────────

def _load_threshold(config_path: Path, key: str = "best_threshold", default: float = 0.5) -> float:
    if config_path.exists():
        with config_path.open() as f:
            return float(json.load(f).get(key, default))
    return default

EVENTS_THRESHOLD = _load_threshold(OUTPUT_EVENTS_DIR / "events_training_config.json")


# ── Cached model + pipeline loading ──────────────────────────────────────────

@st.cache_resource
def get_models_and_extractor():
    env_model, event_model, speech_model, env_id2label, event_id2label, speech_id2label = (
        load_models_and_labels()
    )
    extractor = AudioFeatureExtractor(sample_rate=SAMPLE_RATE)
    return (env_model, event_model, speech_model,
            env_id2label, event_id2label, speech_id2label, extractor)


@st.cache_resource
def get_whisper():
    """Lazy-load Whisper large-v3. Returns model or None if unavailable."""
    try:
        from faster_whisper import WhisperModel
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        print(f"[INFO] Loading Whisper large-v3 ({compute_type})...")
        model = WhisperModel(
            "large-v3",
            device=DEVICE,
            compute_type=compute_type,
            download_root=str(WHISPER_CACHE),
        )
        return model
    except Exception as e:
        st.warning(f"Whisper unavailable: {e}")
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    env_model, event_model, speech_model, extractor,
    audio, env_id2label, event_id2label, speech_id2label,
):
    # Environment embeddings (128, 10)
    env_emb = extractor.env_features(audio, sr=SAMPLE_RATE)
    env_x   = torch.from_numpy(env_emb).unsqueeze(0).to(DEVICE)

    # Events log-mel (1, 128, T)
    event_spec = extractor.event_features(audio, sr=SAMPLE_RATE)
    event_x    = torch.from_numpy(event_spec).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Speech log-mel (1, 128, T) — same as events but raw dB
    speech_logmel = extractor.env_logmel(audio, sr=SAMPLE_RATE)
    T = speech_logmel.shape[1]
    TARGET_FRAMES = 313
    if T < TARGET_FRAMES:
        speech_logmel = np.pad(speech_logmel, ((0, 0), (0, TARGET_FRAMES - T)))
    else:
        speech_logmel = speech_logmel[:, :TARGET_FRAMES]
    speech_x = torch.from_numpy(speech_logmel).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Display logmel
    display_logmel = extractor.env_logmel(audio, sr=SAMPLE_RATE)

    with torch.no_grad():
        env_probs    = torch.softmax(env_model(env_x), dim=1).cpu().numpy()[0]
        event_probs  = torch.sigmoid(event_model(event_x)).cpu().numpy()[0]
        speech_probs = torch.softmax(speech_model(speech_x), dim=1).cpu().numpy()[0]

    # Environment
    env_id    = int(env_probs.argmax())
    env_label = env_id2label.get(env_id, f"class_{env_id}")
    env_conf  = float(env_probs[env_id])

    # Events top-K
    top_idx      = np.argsort(event_probs)[::-1][:TOP_K_EVENTS]
    event_results = [(event_id2label.get(int(i), f"event_{i}"), float(event_probs[i]))
                     for i in top_idx]

    # Speech
    speech_id    = int(speech_probs.argmax())
    speech_label = speech_id2label.get(speech_id, "unknown")
    speech_conf  = float(speech_probs[speech_id])
    speech_prob_map = {speech_id2label.get(i, str(i)): float(p)
                       for i, p in enumerate(speech_probs)}

    return (
        (env_label, env_conf, env_probs),
        event_results,
        (speech_label, speech_conf, speech_prob_map),
        display_logmel,
    )


def run_transcription(audio: np.ndarray, speech_label: str, whisper_model) -> Tuple[str, float, bool]:
    """
    Transcribe audio if speech detected. Returns (transcript, elapsed_s, whisper_used).
    """
    if speech_label == "no_speech" or whisper_model is None:
        return "", 0.0, False

    import tempfile, soundfile as sf

    t0 = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

        segments_gen, _ = whisper_model.transcribe(
            tmp_path, language="en", beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        transcript = " ".join(seg.text.strip() for seg in segments_gen)
        Path(tmp_path).unlink(missing_ok=True)
        return transcript.strip(), round(time.perf_counter() - t0, 2), True
    except Exception as e:
        return f"[Transcription error: {e}]", 0.0, False


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_waveform(audio, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(np.linspace(0, len(audio)/sr, len(audio)), audio, linewidth=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title("Waveform")
    fig.tight_layout(); return fig


def plot_spectrogram(logmel, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(logmel, x_axis="time", y_axis="mel", sr=sr, ax=ax)
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB"); fig.tight_layout(); return fig


def plot_env_bars(env_probs, id2label):
    labels = [id2label.get(i, str(i)) for i in range(len(env_probs))]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(np.arange(len(labels)), env_probs, color="lightgray")
    bars[int(env_probs.argmax())].set_color("steelblue")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Probability")
    ax.set_title("Environment Probabilities"); fig.tight_layout(); return fig


def plot_event_bars(events):
    labels = [e[0] for e in events]; confs = [e[1] for e in events]
    fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.35)))
    ax.barh(np.arange(len(labels)), confs,
            color=["steelblue" if c >= EVENTS_THRESHOLD else "lightgray" for c in confs])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(EVENTS_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"threshold={EVENTS_THRESHOLD:.2f}")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("Confidence")
    ax.set_title(f"Top {TOP_K_EVENTS} Events"); fig.tight_layout(); return fig


def plot_speech_bars(prob_map: dict):
    labels = list(prob_map.keys()); confs = list(prob_map.values())
    colors = {"no_speech": "#d9534f", "single_speaker": "#5bc0de", "conversation": "#5cb85c"}
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.barh(np.arange(len(labels)), confs,
            color=[colors.get(l, "lightgray") for l in labels])
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Probability")
    ax.set_title("Speech Detection"); fig.tight_layout(); return fig


# ── Scene summary ─────────────────────────────────────────────────────────────

def rule_based_summary(env_label, env_conf, events, speech_label, transcript=""):
    strong = sorted([(l, c) for l, c in events if c >= EVENTS_THRESHOLD],
                    key=lambda x: x[1], reverse=True)[:3]
    env_name = env_label.replace("_", " ")
    speech_note = {
        "conversation":    "with an active conversation taking place",
        "single_speaker":  "with a single person speaking",
        "no_speech":       "with no speech detected",
    }.get(speech_label, "")

    if not strong:
        return (f"The recording suggests a {env_name} environment "
                f"(confidence: {env_conf:.0%}) {speech_note}.")

    event_names = [l.replace("_", " ") for l, _ in strong]
    events_text = (event_names[0] if len(event_names) == 1
                   else " and ".join(event_names) if len(event_names) == 2
                   else ", ".join(event_names[:-1]) + f", and {event_names[-1]}")

    summary = (f"The recording suggests a {env_name} environment "
               f"(confidence: {env_conf:.0%}) with {events_text} audible{', ' + speech_note if speech_note else ''}.")

    if transcript:
        summary += f'\n\n**Transcript:** "{transcript}"'
    return summary


def llm_summary(env_label, env_conf, events, speech_label, transcript=""):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return rule_based_summary(env_label, env_conf, events, speech_label, transcript), False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        payload = {
            "environment":      {"label": env_label, "confidence": round(float(env_conf), 3)},
            "detected_events":  [{"label": l, "confidence": round(float(c), 3)}
                                  for l, c in events if c >= EVENTS_THRESHOLD],
            "speech_detection": {"label": speech_label},
            "transcript":       transcript or None,
        }

        system_msg = (
            "You are an expert acoustic scene analyst. "
            "Given classifier outputs from an audio scene understanding system "
            "(environment class, detected sound events, speech detection, and optional transcript), "
            "describe what is likely happening in 2-3 natural sentences. "
            "If a transcript is provided, incorporate it naturally. "
            "Be specific and vivid. Do not mention confidence scores."
        )

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": json.dumps(payload, indent=2)},
            ],
            max_tokens=150, temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        if transcript:
            text += f'\n\n**Transcript:** "{transcript}"'
        return text or rule_based_summary(env_label, env_conf, events, speech_label, transcript), True

    except Exception as e:
        return (rule_based_summary(env_label, env_conf, events, speech_label, transcript)
                + f"\n\n*(LLM error: {e})*"), False


# ── Conversation log (session state) ─────────────────────────────────────────

def init_conversation_log():
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = []

def add_to_log(speech_label, transcript, timestamp):
    if transcript and speech_label != "no_speech":
        st.session_state.conversation_log.append({
            "time": time.strftime("%H:%M:%S", time.localtime(timestamp)),
            "label": speech_label,
            "text": transcript,
        })

def save_conversation_log():
    if not st.session_state.conversation_log:
        return None
    out = {
        "full_transcript": " ".join(e["text"] for e in st.session_state.conversation_log),
        "entries": st.session_state.conversation_log,
    }
    return json.dumps(out, indent=2).encode("utf-8")


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Surround Sound V2", layout="wide")
    st.title("Surround Sound V2 – Live Demo")
    st.markdown("Record from your microphone to classify the acoustic environment, "
                "detect sound events, and transcribe speech.")

    init_conversation_log()

    (env_model, event_model, speech_model,
     env_id2label, event_id2label, speech_id2label, extractor) = get_models_and_extractor()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        rec_sec  = st.slider("Recording length (s)", 3.0, 15.0, DEFAULT_RECORD_SECONDS, 0.5)
        use_llm  = st.toggle("LLM scene summary", value=True)
        use_whisper = st.toggle("Transcribe speech (Whisper)", value=True)

        st.caption(f"Events threshold: {EVENTS_THRESHOLD:.2f}")
        st.caption(f"Device: {DEVICE}")

        if use_llm and not os.getenv("OPENAI_API_KEY", "").strip():
            st.warning("OPENAI_API_KEY not set — rule-based summary will be used.")



    # ── Record button ─────────────────────────────────────────────────────────
    if st.button("Record", type="primary"):
        audio = record_audio(rec_sec, fs=SAMPLE_RATE)
        recorded_at = time.time()

        with st.spinner("Running inference..."):
            (env_label, env_conf, env_probs), event_results, \
            (speech_label, speech_conf, speech_prob_map), display_logmel = run_inference(
                env_model, event_model, speech_model, extractor,
                audio, env_id2label, event_id2label, speech_id2label,
            )

        # Transcription
        transcript = ""
        whisper_elapsed = 0.0
        whisper_used_flag = False
        if use_whisper:
            whisper_model = get_whisper()
            with st.spinner("Transcribing..."):
                transcript, whisper_elapsed, whisper_used_flag = run_transcription(
                    audio, speech_label, whisper_model
                )
            add_to_log(speech_label, transcript, recorded_at)

        # ── Row 1: Waveform | Spectrogram ─────────────────────────────────────
        st.markdown("---")
        col_wave, col_spec = st.columns([1, 1])
        with col_wave:
            st.subheader("Waveform")
            st.pyplot(plot_waveform(audio))
        with col_spec:
            st.subheader("Log-Mel Spectrogram")
            st.pyplot(plot_spectrogram(display_logmel))

        # ── Row 2: Environment | Speech Detection ─────────────────────────────
        st.markdown("---")
        col_env, col_speech = st.columns([1, 1])
        with col_env:
            st.subheader("Environment")
            st.metric(label=env_label.replace("_", " ").title(), value=f"{env_conf:.1%}")
            st.pyplot(plot_env_bars(env_probs, env_id2label))

        with col_speech:
            st.subheader("Speech Detection")
            speech_emoji = ""
            st.metric(label=speech_label.replace("_", " ").title(),
                      value=f"{speech_conf:.1%}")
            st.pyplot(plot_speech_bars(speech_prob_map))

        # ── Row 3: Events | Transcript + Summary ──────────────────────────────
        st.markdown("---")
        col_events, col_summary = st.columns([1, 1])
        with col_events:
            st.subheader(f"Top {TOP_K_EVENTS} Sound Events")
            st.pyplot(plot_event_bars(event_results))
            detected = [(l, c) for l, c in event_results if c >= EVENTS_THRESHOLD]
            if detected:
                st.write("**Detected above threshold:**")
                # Show as a clean table
                import pandas as pd
                det_df = pd.DataFrame(detected, columns=["Event", "Confidence"])
                det_df["Confidence"] = det_df["Confidence"].map("{:.2f}".format)
                st.dataframe(det_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No events above threshold ({EVENTS_THRESHOLD:.2f})")

        with col_summary:
            if use_whisper:
                st.subheader("Transcript")
                if transcript:
                    st.text_area("", transcript, height=120, label_visibility="collapsed")
                    st.caption(f"Whisper large-v3 · {whisper_elapsed:.1f}s")
                elif speech_label == "no_speech":
                    st.info("No speech detected — transcription skipped.")
                else:
                    st.info("No speech found in audio.")

            st.subheader("Scene Summary")
            if use_llm:
                summary, used_llm_flag = llm_summary(
                    env_label, env_conf, event_results, speech_label, transcript)
                st.write(summary)
                st.caption("LLM-generated" if used_llm_flag else "Rule-based")
            else:
                st.write(rule_based_summary(
                    env_label, env_conf, event_results, speech_label, transcript))

        # ── Conversation log ──────────────────────────────────────────────────
        if st.session_state.conversation_log:
            st.markdown("---")
            with st.expander(f"Conversation Log ({len(st.session_state.conversation_log)} entries)", expanded=False):
                col_log, col_actions = st.columns([3, 1])
                with col_log:
                    for entry in st.session_state.conversation_log[-15:]:
                        st.markdown(f"**[{entry['time']} · {entry['label']}]** {entry['text']}")
                with col_actions:
                    log_bytes = save_conversation_log()
                    if log_bytes:
                        st.download_button("Save", log_bytes,
                                           file_name="conversation.json",
                                           mime="application/json")
                    if st.button("Clear"):
                        st.session_state.conversation_log = []
                        st.rerun()

    st.markdown("---")
    st.caption("Surround Sound V2 · CSS 586 Deep Learning · CSS 590 Human-Computer Interaction · University of Washington Bothell")


if __name__ == "__main__":
    main()