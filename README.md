# Surround Sound
Real-Time Environment & Sound Event Classification

Surround Sound is a machine learning system that performs **real-time audio scene understanding** using a single microphone. It uses two convolutional neural networks:

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
│   ├── 02_environment_training.py   # Train environment CNN
│   ├── 02_events_training.py        # Train events CNN
│
├── scripts/
│   ├── environment_download.py      # AudioSet clip downloader (yt-dlp + ffmpeg)
│   ├── environment_filter.py        # Filter AudioSet CSVs into environment manifests
│   ├── environment_label.py         # Map AudioSet labels → 8 environment classes
│   ├── environment_preprocess.py    # Preprocess environment WAVs → log-mel features
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
│   │   ├── models_live.py           # Load trained CNN weights + inference
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

# Model Overview

### **Environment CNN**
- 4-layer Conv2D (Conv → BN → ReLU → MaxPool)
- Softmax over **8 classes**
- Input: log-mel spectrogram (128 mels × T)
- Trained for ~20–25 epochs with class-balanced sampling

### **Events CNN**
- 4-layer deeper Conv2D (64→128→256→256)
- Sigmoid over **62 event labels**
- Multi-label BCEWithLogits + positive reweighting
- Trained for ~40 epochs, with threshold tuning

---

# Installation

```bash
conda create -n surround-env python=3.10
conda activate surround-env
pip install -r requirements.txt


Datasets are not included due to size.
Download & preprocess them using the provided scripts.

1. Environment Dataset (AudioSet)
Step 1: Filter AudioSet CSVs
python scripts/environment_filter.py --csv_dir data/csv --ontology ontology.json --out data/manifests/environment_segments.csv

Step 2: Download Audio Segments
python scripts/environment_download.py --manifest data/manifests/environment_segments.csv

Step 3: Preprocess Audio → Log-Mel
python scripts/environment_preprocess.py

2. Events Dataset (FSD50K)
Step 1: Download FSD50K
python scripts/events_download.py --root data/events/FSD50K

Step 2: Build Manifest
python scripts/events_manifest.py

Step 3: Preprocess → Log-Mel
python scripts/events_preprocess.py

Training
Environment model:
python notebooks/02_environment_training.py

Events model:
python notebooks/02_events_training.py

Evaluation
python src/eval.py --task env
python src/eval.py --task events


Outputs go to:

src/results/environment/
src/results/events/

Live Demo
streamlit run src/live_demo/streamlit_app.py


Demo includes:

Microphone recording

Waveform + log-mel spectrogram

Environment prediction

Top-K event predictions

Optional GPT mini LLM scene summary
```
Acknowledgments

DCASE Challenge Community – inspiration for environment/event classification

Google AudioSet – source for environment labels

FSD50K – sound event dataset

Librosa – audio feature extraction

PyTorch – deep learning framework

Streamlit – live demo UI

OpenAI – optional scene summarization
