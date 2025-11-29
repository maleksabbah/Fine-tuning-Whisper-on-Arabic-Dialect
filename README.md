AI Dubbing & Whisper Fine-Tuning Pipeline

A fully integrated end-to-end system for Arabic ASR, Whisper fine-tuning, segment cleaning, and episode-based data processing.
This repository powers large-scale dubbing, subtitling, and translation workflows for shows such as Taj and Al-Atawla, with robust repetition filtering, safe decoding, and multi-episode dataset handling.

🚀 Features
🎧 1. Episode-Based Audio Chunking

Converts full episodes into consistent WAV chunks

Uses FFmpeg + PyDub

Auto-creates folders like:

Al_Atawla_Eps01_chunks/
Al_Atawla_Eps02_chunks/
...

🧹 2. Transcript Cleaning & Repetition Removal

Detects hallucinated loops (2-word, 3-word, 4-word patterns)

Sliding-window duplicate detection

Normalisation, trimming, and timestamp preservation

Works with Whisper outputs or external subtitle files

🤖 3. Whisper Fine-Tuning

Supports Tiny, Small, Medium, Large-v3

Progressive unfreezing

Per-epoch:

Learning rate schedule

Gradient clipping

Mixed precision

Clean, single-bar tqdm progress display

🔒 4. Safe Decoding Engine

Temperature fallback

Probability-mass scanning

Max-repetition constraints
→ Prevents long, rambly hallucinations and language-drift.

📊 5. Evaluation Suite

WER / CER (mean, median, std)

Loss curves

Gradient stats (mean, max, clipping rate)

Per-segment error reports

📂 Project Structure
├── chunker/
│   ├── audio_chunker.py
│   └── utils_ffmpeg.py
├── cleaning/
│   ├── repetition_filter.py
│   ├── transcript_cleaner.py
│   └── alignment_tools.py
├── training/
│   ├── train_whisper.py
│   ├── dataset_loader.py
│   ├── safe_decode.py
│   └── hyperparams.py
├── evaluation/
│   ├── eval_pipeline.py
│   └── wer_cer.py
├── data/
│   ├── raw/
│   ├── chunks/
│   └── cleaned_segments/
└── README.md

📦 Installation
git clone https://github.com/maleksabbahh/Fine-tuning-whisper.git
cd ai-dubbing-whisper-pipeline
pip install -r requirements.txt


You may also install dependencies manually:

pip install transformers datasets torchaudio soundfile tqdm rapidfuzz jiwer bert-score torchcodec

🔧 Usage
1️⃣ Chunk an Episode
python chunker/audio_chunker.py \
  --input "Al Atawla Eps 01.mp4" \
  --output "./data/chunks/Al_Atawla_Eps01_chunks"

2️⃣ Clean Transcripts
python cleaning/transcript_cleaner.py \
  --segments taj_ep15_segments.json \
  --output taj_ep15_segments_clean.json

3️⃣ Train Whisper
python training/train_whisper.py \
  --model small \
  --dataset-dir ./data/cleaned_segments \
  --epochs 6

4️⃣ Evaluate
python evaluation/eval_pipeline.py \
  --predictions out.json \
  --targets gt.json

🧠 Model Notes

Encoder freeze in epoch 1 stabilizes training

Gradient clipping prevents exploding gradients

Temperature scanning and probability monitoring stop hallucinations

Adding multiple shows improves robustness across dialects

Clean transcripts drastically reduce WER spikes

📈 Example Results
Metric	Value
WER (mean)	0.25–0.50
CER (mean)	0.10–0.25
Hallucination loops	Eliminated
Stability across episodes	High
