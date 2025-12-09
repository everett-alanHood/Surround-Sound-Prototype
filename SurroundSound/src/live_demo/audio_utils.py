from typing import Union

import numpy as np
import sounddevice as sd
import streamlit as st


def record_audio(duration_sec: Union[int, float], fs: int = 16000) -> np.ndarray:
    """
    Record mono audio from the default input device.

    Returns:
        audio: np.ndarray of shape (num_samples,)
    """
    st.info(f"Recording {duration_sec:.1f} seconds from microphone...")
    sd.default.samplerate = fs
    sd.default.channels = 1

    audio = sd.rec(int(duration_sec * fs), dtype="float32")
    sd.wait()
    st.success("Recording complete.")

    if audio.ndim > 1:
        audio = audio.squeeze(-1)

    return audio
