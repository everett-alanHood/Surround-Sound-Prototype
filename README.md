# Surround Sound V2
Real-Time Environment, Sound Event, and Speech Classification
 
Surround Sound V2 is a machine learning system that performs **real-time auditory
scene understanding** using a single microphone. It simultaneously classifies the
acoustic environment, detects sound events across 208 labels, identifies speech
conditions, transcribes speech via Whisper large-v3, and optionally generates a
natural-language scene summary using GPT-4.1 Mini.
 
> **Note:** These instructions are written for Windows users.
 
---
 
## Project Structure
 
```text
SurroundSound/
│
├── notebooks/
│   ├── 01_environment_setup.py      # Build AudioSet environment manifests
│   ├── 01_events_setup.py           # Build FSD50K events manifests
│   ├── 01_speech_setup.py           # Build CHiME-6 + Common Voice speech manifests
│   ├── 02_environment_training.py   # Train environment model
│   ├── 02_events_training.py        # Train events CNN
│   └── 02_speech_training.py        # Train speech classifier
│
├── scripts/
│   ├── cleanup_datasets.py          # Free disk space after training (see below)
│   ├── environment_filter.py        # Filter AudioSet CSVs into environment manifests
│   ├── environment_label.py         # Map AudioSet labels → 8 environment classes
│   ├── environment_preprocess.py    # Parse AudioSet TFRecords → embeddings
│   ├── events_download.py           # Download & extract FSD50K audio/metadata
│   ├── events_filter.py             # Build events metadata from FSD50K ground truth
│   ├── events_manifest.py           # Build FSD50K event manifest
│   ├── events_preprocess.py         # Preprocess event WAVs → log-mel features
│   ├── events_taxonomy.py           # Generate taxonomy_events.json from vocabulary
│   ├── speech_download.py           # Download CHiME-6 and Common Voice audio
│   ├── speech_manifest.py           # Build speech manifest
│   └── speech_preprocess.py         # Preprocess speech WAVs → log-mel features
│
├── src/
│   ├── eval.py                      # Evaluation pipeline (env, events, speech)
│   │
│   ├── live_demo/
│   │   ├── audio_utils.py           # Microphone recording (sounddevice)
│   │   ├── feature_extraction.py    # Shared online feature extractor
│   │   ├── models_live.py           # Load trained model weights + inference
│   │   ├── transcription.py         # Whisper large-v3 ASR integration
│   │   └── streamlit_app.py         # Real-time demo UI
│   │
│   └── results/                     # Confusion matrices, F1/AP plots, CSV summaries
│
├── output/
│   ├── environment/                 # best_environment_model.pt + training logs
│   ├── events/                      # best_events_model.pt + training logs
│   └── speech/                      # best_speech_model.pt + training logs
│
├── ontology.json
├── requirements.txt
└── .gitignore
```
 
---
 
## Requirements
 
- [Anaconda](https://www.anaconda.com/download) or Miniconda
- Python 3.10
- An OpenAI API key (optional — required only for GPT-4.1 Mini scene
  summarization; transcription runs locally via Whisper)
---
 
## Setup
 
**1. Create and activate the conda environment**
 
```bash
conda create -n surround-env python=3.10
conda activate surround-env
```
 
**2. Install dependencies**
 
```bash
pip install -r requirements.txt
```
 
**3. Install ffmpeg** (required for audio pipelines)
 
```bash
conda install -c conda-forge ffmpeg
```
 
---
 
## 1. Environment Dataset (AudioSet)
 
AudioSet provides pre-extracted VGGish embeddings (128-dim at 1Hz). Each
10-second clip is represented as a **10 × 128** matrix, avoiding the need to
download raw YouTube audio.
 
**Step 1: Download AudioSet features (~2.4 GB)**
 
```bash
curl -O https://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
tar -xzf features.tar.gz -C SurroundSound/data/environment/tfrecords
```
 
**Step 2: Download AudioSet CSVs**
 
Download `balanced_train_segments.csv` and `eval_segments.csv` from the
[AudioSet downloads page](https://research.google.com/audioset/download.html)
and place them in `SurroundSound/data/csv/`.
 
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
 
**Step 5: Build manifest**
 
```bash
python SurroundSound/notebooks/01_environment_setup.py
```
 
---
 
## 2. Events Dataset (FSD50K)
 
**Step 1: Download FSD50K**
 
```bash
python SurroundSound/scripts/events_download.py --root SurroundSound/data/events/FSD50K
```
 
**Step 2: Build event taxonomy**
 
```bash
python SurroundSound/scripts/events_taxonomy.py
```
 
**Step 3: Build manifest**
 
```bash
python SurroundSound/scripts/events_manifest.py
```
 
**Step 4: Preprocess → log-mel**
 
```bash
python SurroundSound/scripts/events_preprocess.py
```
 
**Step 5: Build index**
 
```bash
python SurroundSound/notebooks/01_events_setup.py
```
 
---
 
## 3. Speech Dataset (CHiME-6 + Common Voice)
 
**Step 1: Download datasets**
 
```bash
python SurroundSound/scripts/speech_download.py
```
 
**Step 2: Build manifest**
 
```bash
python SurroundSound/scripts/speech_manifest.py
```
 
**Step 3: Preprocess → log-mel**
 
```bash
python SurroundSound/scripts/speech_preprocess.py
```
 
**Step 4: Build index**
 
```bash
python SurroundSound/notebooks/01_speech_setup.py
```
 
---
 
## Training
 
Pre-trained model weights are included in `output/`. You only need to run
training if you want to retrain from scratch.
 
```bash
# Navigate to project folder first
cd SurroundSound
 
# Environment model
python notebooks/02_environment_training.py
 
# Events model
python notebooks/02_events_training.py
 
# Speech model
python notebooks/02_speech_training.py
```
 
---
 
## Evaluation
 
Run from inside the `SurroundSound/src/` directory:
 
```bash
cd SurroundSound/src
 
# Evaluate a single task
python eval.py --task env
python eval.py --task events
python eval.py --task speech
 
# Evaluate environment + events
python eval.py --task both
 
# Evaluate all three
python eval.py --task all
```
 
Optional flags:
 
```bash
python eval.py --task all --split val --batch_size 64
```
 
Results are saved to:
 
```
src/results/environment/val/
src/results/events/val/
src/results/speech/val/
```
 
Each directory contains overall metrics, per-class metrics, confusion matrices,
and F1/AP plots as CSV and PNG files.
 
---
 
## Cleaning Up Datasets
 
After training, raw and processed dataset files can take significant disk space.
Use `cleanup_datasets.py` to free space while preserving model checkpoints,
label maps, and manifests.
 
**Dry run** (lists what would be deleted without removing anything):
 
```bash
python SurroundSound/scripts/cleanup_datasets.py
```
 
**Actually delete:**
 
```bash
python SurroundSound/scripts/cleanup_datasets.py --delete
```
 
---
 
## Live Demo
 
**1. Navigate to the project folder**
 
```bash
cd SurroundSound
```
 
**2. (Optional) Set your OpenAI API key for scene summarization**
 
In PowerShell:
 
```powershell
$env:OPENAI_API_KEY="your-key-here"
```
 
If skipped, the demo runs normally — scene summarization will be disabled.
 
**3. Launch the demo**
 
```bash
streamlit run src/live_demo/streamlit_app.py
```
 
The demo includes:
 
- Microphone recording (1–15 seconds)
- Waveform + log-mel spectrogram visualization
- Environment prediction (8 classes)
- Top event predictions (208 labels)
- Speech detection (no speech / single speaker / multi-speaker)
- Whisper large-v3 transcription (when speech is detected)
- Optional GPT-4.1 Mini scene summary
---
 
## Acknowledgments
 
- [DCASE Challenge Community](https://dcase.community/)
- [Google AudioSet](https://research.google.com/audioset/)
- [FSD50K](https://zenodo.org/record/4060432)
- [CHiME-6](https://openslr.org/150/)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
- [OpenAI Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Librosa](https://librosa.org/) · [PyTorch](https://pytorch.org/) ·
  [Streamlit](https://streamlit.io/) · [OpenAI](https://openai.com/)
