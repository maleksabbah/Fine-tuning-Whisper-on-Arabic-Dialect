"""
Arabic ASR Benchmark Suite — matches old Benchmark_A100.py methodology.
Runs FLEURS (MSA), MGB-3 (Egyptian), Casablanca (8 dialects).
Compares baseline vs LoRA checkpoint side by side.

Requires: datasets==2.21.0, peft==0.11.1, transformers==4.40.0

Usage:
  python Training/Evaluate.py --checkpoint /workspace/checkpoints/lora_epoch_3 --samples 200
  python Training/Evaluate.py --checkpoint /workspace/checkpoints/lora_epoch_3 --samples 200 --baseline-only
"""
import argparse
import time
import re
import numpy as np
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset
from jiwer import wer as compute_wer

ARABIC_DIACRITICS = set("\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0655")

def normalize_arabic(t):
    if not t: return ""
    t = "".join(c for c in t if c not in ARABIC_DIACRITICS)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ـ', '')
    return " ".join(t.split()).strip()


def load_model(checkpoint_path=None):
    print(f"[MODEL] Loading whisper-large-v3...", flush=True)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16
    )
    if checkpoint_path:
        print(f"[MODEL] Loading LoRA from {checkpoint_path}...", flush=True)
        model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to("cuda")
    model.eval()
    return model, processor


def transcribe(model, processor, audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
    with torch.no_grad():
        ids = model.generate(input_features=inputs, language="ar", task="transcribe", max_new_tokens=440)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def get_audio(sample):
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def benchmark_fleurs(model, processor, max_samples=200):
    """FLEURS Arabic — MSA clean read speech."""
    print(f"  [FLEURS] Loading...", end=" ", flush=True)
    ds = load_dataset("google/fleurs", "ar_eg", split="test")
    refs, hyps = [], []
    for i, s in enumerate(ds):
        if i >= max_samples: break
        audio = get_audio(s)
        hyp = transcribe(model, processor, audio)
        refs.append(normalize_arabic(s["transcription"]))
        hyps.append(normalize_arabic(hyp))
        if (i+1) % 50 == 0:
            print(f"{i+1}", end=" ", flush=True)
    w = compute_wer(refs, hyps)
    print(f"→ {w:.4f} ({len(refs)} samples)")
    return w


def benchmark_mgb3(model, processor, max_samples=200):
    """MGB-3 — Egyptian dialect."""
    print(f"  [MGB-3] Loading...", end=" ", flush=True)
    ds = load_dataset("MightyStudent/Egyptian-ASR-MGB-3", split="train")
    refs, hyps = [], []
    for i, s in enumerate(ds):
        if i >= max_samples: break
        audio = get_audio(s)
        hyp = transcribe(model, processor, audio)
        refs.append(normalize_arabic(s["sentence"]))
        hyps.append(normalize_arabic(hyp))
        if (i+1) % 50 == 0:
            print(f"{i+1}", end=" ", flush=True)
    w = compute_wer(refs, hyps)
    print(f"→ {w:.4f} ({len(refs)} samples)")
    return w


CASABLANCA_DIALECTS = [
    "Algeria", "Egypt", "Jordan", "Morocco",
    "Palestine", "Saudi_Arabia", "UAE", "Yemen"
]

def benchmark_casablanca(model, processor, max_samples_per_dialect=25):
    """Casablanca — 8 Arabic dialects. 25 samples per dialect = 200 total."""
    print(f"  [CASABLANCA] Loading 8 dialects...", flush=True)
    all_refs, all_hyps = [], []
    dialect_results = {}

    for dialect in CASABLANCA_DIALECTS:
        try:
            ds = load_dataset("UBC-NLP/Casablanca", dialect, split="test")
            refs, hyps = [], []
            for i, s in enumerate(ds):
                if i >= max_samples_per_dialect: break
                audio = get_audio(s)
                hyp = transcribe(model, processor, audio)
                refs.append(normalize_arabic(s["transcription"]))
                hyps.append(normalize_arabic(hyp))
            w = compute_wer(refs, hyps) if refs else 1.0
            dialect_results[dialect] = w
            all_refs.extend(refs)
            all_hyps.extend(hyps)
            print(f"    {dialect}: {w:.4f} ({len(refs)} samples)", flush=True)
        except Exception as e:
            print(f"    {dialect}: FAILED — {e}", flush=True)
            dialect_results[dialect] = None

    overall = compute_wer(all_refs, all_hyps) if all_refs else 1.0
    print(f"  [CASABLANCA] Overall: {overall:.4f} ({len(all_refs)} samples)")
    return overall, dialect_results


def run_all(model, processor, max_samples):
    results = {}
    t0 = time.time()

    try:
        results["fleurs"] = benchmark_fleurs(model, processor, max_samples)
    except Exception as e:
        print(f"  [FLEURS] FAILED: {e}")
        results["fleurs"] = None

    try:
        results["mgb3"] = benchmark_mgb3(model, processor, max_samples)
    except Exception as e:
        print(f"  [MGB-3] FAILED: {e}")
        results["mgb3"] = None

    try:
        samples_per_dialect = max(max_samples // 8, 10)
        overall, dialects = benchmark_casablanca(model, processor, samples_per_dialect)
        results["casablanca"] = overall
        results["casablanca_dialects"] = dialects
    except Exception as e:
        print(f"  [CASABLANCA] FAILED: {e}")
        results["casablanca"] = None

    results["time"] = time.time() - t0
    return results


def print_comparison(baseline, lora, lora_name):
    print(f"\n{'='*70}")
    print(f"{'Benchmark':<25} {'Baseline':>12} {'LoRA':>12} {'Change':>12}")
    print(f"{'='*70}")

    for key in ["fleurs", "mgb3", "casablanca"]:
        b = baseline.get(key)
        l = lora.get(key)
        b_str = f"{b:.4f}" if b is not None else "FAILED"
        l_str = f"{l:.4f}" if l is not None else "FAILED"
        if b is not None and l is not None:
            diff = l - b
            pct = (diff / b) * 100
            c_str = f"{diff:+.4f} ({pct:+.1f}%)"
        else:
            c_str = "N/A"

        names = {"fleurs": "FLEURS (MSA)", "mgb3": "MGB-3 (Egyptian)", "casablanca": "Casablanca (8 dial.)"}
        print(f"{names.get(key, key):<25} {b_str:>12} {l_str:>12} {c_str:>12}")

    print(f"{'='*70}")

    # Casablanca per-dialect
    if "casablanca_dialects" in lora and lora["casablanca_dialects"]:
        print(f"\nCasablanca per-dialect:")
        bd = baseline.get("casablanca_dialects", {})
        ld = lora.get("casablanca_dialects", {})
        for dialect in CASABLANCA_DIALECTS:
            b = bd.get(dialect)
            l = ld.get(dialect)
            b_str = f"{b:.4f}" if b is not None else "N/A"
            l_str = f"{l:.4f}" if l is not None else "N/A"
            print(f"  {dialect:<20} {b_str:>10} {l_str:>10}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="/workspace/checkpoints/lora_epoch_3")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--lora-only", action="store_true")
    a = p.parse_args()

    print("=" * 70)
    print("ARABIC ASR BENCHMARK SUITE")
    print("=" * 70)
    print(f"Samples: {a.samples}")
    print(f"Checkpoint: {a.checkpoint}")
    print(f"Benchmarks: FLEURS (MSA), MGB-3 (Egyptian), Casablanca (8 dialects)")
    print()

    baseline_results = {}
    lora_results = {}

    if not a.lora_only:
        print("=" * 70)
        print("BASELINE: whisper-large-v3 (no LoRA)")
        print("=" * 70)
        model, processor = load_model(None)
        baseline_results = run_all(model, processor, a.samples)
        del model
        torch.cuda.empty_cache()
        print(f"Baseline done in {baseline_results['time']/60:.1f}m\n")

    if not a.baseline_only:
        print("=" * 70)
        print(f"LORA: {a.checkpoint}")
        print("=" * 70)
        model, processor = load_model(a.checkpoint)
        lora_results = run_all(model, processor, a.samples)
        del model
        torch.cuda.empty_cache()
        print(f"LoRA done in {lora_results['time']/60:.1f}m\n")

    if baseline_results and lora_results:
        print_comparison(baseline_results, lora_results, a.checkpoint)


if __name__ == "__main__":
    main()