# Surround Sound
Real-Time Environment & Sound Event Classification

Surround Sound is a machine learning system that performs **real-time audio scene understanding** using a single microphone. It uses two neural networks:
- **Environment Classifier (8 classes, softmax)**
- **Events Classifier (62 labels, multi-label sigmoid)**

A Streamlit demo records audio, displays waveform & spectrograms, runs both models, and optionally summarizes the detected scene using a lightweight LLM.

---

## Project Structure
```text
SurroundSound/
│
├── notebooks/
│   ├── 01_environment_setup.py      # Build AudioSet-based environment manifests
│   ├── 01_events_setup.py           # Build FSD50K-based events manifests
│   ├── 02_environment_training.py   # Train environment model
│   ├── 02_events_training.py        # Train events CNN
│
├── scripts/
│   ├── environment_filter.py        # Filter AudioSet CSVs into environment manifests
│   ├── environment_label.py         # Map AudioSet labels → 8 environment classes
│   ├── environment_preprocess.py    # Parse AudioSet TFRecords → (10, 128) embeddings
│   ├── events_download.py           # Download & extract FSD50K audio/metadata
│   ├── events_filter.py             # Build events metadata.jsonl from FSD50K GT
│   ├── events_manifest.py           # Build FSD50K event manifest (paths + labels)
│   └── events_preprocess.py         # Preprocess event WAVs → log-mel features
│
├── src/
│   ├── eval.py                      # Full evaluation pipeline (env + events)
│   │
│   ├── live_demo/
│   │   ├── audio_utils.py           # Microphone recording (sounddevice)
│   │   ├── feature_extraction.py    # Shared online feature extractor
│   │   ├── models_live.py           # Load trained model weights + inference
│   │   └── streamlit_app.py         # Real-time demo UI
│   │
│   └── results/                     # Confusion matrices, F1/AP plots, CSV summaries
│
├── output/
│   ├── environment/                 # best_model.pt + training logs
│   └── events/                      # best_model.pt + training logs
│
├── requirements.txt
└── .gitignore
```

---

## Model Overview

### Environment Classifier
- Small MLP or 1D CNN over **10 × 128** VGGish embeddings (1 embedding per second)
- Softmax over **8 classes**
- Input: pre-extracted AudioSet VGGish features (no raw audio required for training)
- Trained for ~20–25 epochs with class-balanced sampling

### Events CNN
- 4-layer Conv2D (64→128→256→256)
- Sigmoid over **62 event labels**
- Multi-label BCEWithLogits + positive reweighting
- Input: log-mel spectrogram (128 mels × T)
- Trained for ~40 epochs, with threshold tuning

---

## Installation

```bash
conda create -n surround-env python=3.10
conda activate surround-env
pip install -r requirements.txt
conda install -c conda-forge ffmpeg  # required for events pipeline only
```

Datasets are not included due to size. Download and preprocess them using the provided scripts.

---

## 1. Environment Dataset (AudioSet)

AudioSet provides pre-extracted VGGish embeddings (128-dim at 1Hz), avoiding the need to download raw YouTube audio. Each 10-second clip is represented as a **10 × 128** matrix.

**Step 1: Download AudioSet features (~2.4 GB)**
```bash
# Pick a regional mirror: us, eu, or asia
curl -O https://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
tar -xzf features.tar.gz -C SurroundSound/data/environment/tfrecords
```

**Step 2: Download AudioSet CSVs**

Download `balanced_train_segments.csv` and `eval_segments.csv` from the
[AudioSet downloads page](https://research.google.com/audioset/download.html) and place them in `SurroundSound/data/csv/`.

**Step 3: Filter CSVs into a manifest**
```bash
python SurroundSound/scripts/environment_filter.py \
  --csv_dir SurroundSound/data/csv \
  --ontology SurroundSound/ontology.json \
  --out SurroundSound/data/manifests/environment_segments.csv
```

**Step 4: Parse TFRecords → embeddings**
```bash
python SurroundSound/scripts/environment_preprocess.py
```

---

## 2. Events Dataset (FSD50K)

The events pipeline uses raw audio and produces log-mel spectrograms.

**Step 1: Download FSD50K**
```bash
python SurroundSound/scripts/events_download.py --root SurroundSound/data/events/FSD50K
```

**Step 2: Build manifest**
```bash
python SurroundSound/scripts/events_manifest.py
```

**Step 3: Preprocess → log-mel**
```bash
python SurroundSound/scripts/events_preprocess.py
```

---

## Training

```bash
# Environment model
python SurroundSound/notebooks/02_environment_training.py

# Events model
python SurroundSound/notebooks/02_events_training.py
```

---

## Evaluation

```bash
python SurroundSound/src/eval.py --task env
python SurroundSound/src/eval.py --task events
```

Outputs go to:
```
SurroundSound/src/results/environment/
SurroundSound/src/results/events/
```

---

## Live Demo

```bash
streamlit run SurroundSound/src/live_demo/streamlit_app.py
```

Demo includes:
- Microphone recording
- Waveform + log-mel spectrogram
- Environment prediction
- Top-K event predictions
- Optional LLM scene summary

---

## Acknowledgments

- [DCASE Challenge Community](https://dcase.community/) — inspiration for environment/event classification
- [Google AudioSet](https://research.google.com/audioset/) — environment features and labels
- [FSD50K](https://zenodo.org/record/4060432) — sound event dataset
- [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) — pre-extracted audio embeddings
- [Librosa](https://librosa.org/) — audio feature extraction
- [PyTorch](https://pytorch.org/) — deep learning framework
- [Streamlit](https://streamlit.io/) — live demo UI
- [OpenAI](https://openai.com/) — optional scene summarization
