from google.colab import drive
drive.mount('/content/drive')

# =========================================================
# 1️⃣ INSTALL DEPENDENCIES
# =========================================================
!pip install -q transformers datasets torchaudio soundfile tqdm torchcodec jiwer rapidfuzz bert-score evaluate --no-deps

import os, gc, json, zipfile, torch, torchaudio, shutil
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Dataset, concatenate_datasets
from jiwer import wer, cer
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
import evaluate

# =========================================================
# 🎯 USER CONFIGURATION - CUSTOMIZE HERE!
# =========================================================

# 📂 EPISODE SELECTION
# List the episode numbers you want to include in training
# Example: [5, 8, 11, 12, 13, 14, 15, 16] or range: list(range(5, 22))
EPISODES_TO_INCLUDE = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]  # ← EDIT THIS!

# 📁 FILE PATHS
# Update these paths to match your file locations
EPISODE_JSON_PATTERN = "/content/Al_Atawla_Eps_{ep:02d}_chunks_whisper_segments_cleaned.json"  # ← Pattern for JSON files
EPISODE_ZIP_PATTERN = "/content/Al_Atawla_Eps_{ep:02d}_chunks.zip"            # ← Pattern for audio ZIP files
EXTRACT_BASE_DIR = "/content/Al_Atawla_Eps_{ep:02d}_chunks"                   # ← Base extraction directory

# 💾 OUTPUT DIRECTORY
CHECKPOINT_DIR = "/content/drive/MyDrive/whisper_cedars"

# 🎛️ TRAINING HYPERPARAMETERS
EPOCHS = 3
LEARNING_RATES = [5e-6, 3e-6, 2e-6, 1e-6, 5e-7, 5e-7]  # One per epoch
GRADIENT_CLIPS = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]        # One per epoch
FREEZE_ENCODER = [True, False, False, False, False, False]  # Freeze in epoch 1 only

# 📊 DATA SPLIT
VALIDATION_SPLIT = 0.15  # 15% for validation (use 0.15-0.20)
RANDOM_SEED = 42

# 📈 GRADIENT TRACKING
GRADIENT_LOG_FREQUENCY = 100  # Log gradients every N steps
GRADIENT_DETAILED_LOG = 500   # Detailed gradient flow every N steps

# 🎯 EVALUATION
SAVE_DETAILED_RESULTS = True  # Save per-sample evaluation results
SHOW_EXAMPLES = 5             # Number of best/worst examples to show

print("="*80)
print("🎯 MULTI-EPISODE WHISPER TRAINING CONFIGURATION")
print("="*80)
print(f"📂 Episodes to include: {EPISODES_TO_INCLUDE}")
print(f"📊 Total episodes: {len(EPISODES_TO_INCLUDE)}")
print(f"🎛️ Training epochs: {EPOCHS}")
print(f"📈 Learning rates: {LEARNING_RATES}")
print(f"✂️ Validation split: {VALIDATION_SPLIT * 100}%")
print("="*80)

# =========================================================
# 2️⃣ DEVICE SETUP
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Using device: {device}")

# =========================================================
# 3️⃣ LOAD WHISPER MODEL
# =========================================================

if os.path.exists(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "model.safetensors")):
    print("📂 Loading from existing checkpoint...")
    processor = WhisperProcessor.from_pretrained(CHECKPOINT_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_DIR).to(device)
    print("   ✅ Resumed from checkpoint")
else:
    print("🔥 Loading fresh model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    print("   ✅ Fresh model loaded")

print("   🟢 Fresh encoder")
print("   🟢 Fresh decoder")

# =========================================================
# 4️⃣ LOAD BERTSCORE METRIC
# =========================================================
print("\n📊 Loading BERTScore metric...")
try:
    bertscore = evaluate.load("bertscore")
    print("   ✅ BERTScore loaded")
except Exception as e:
    print(f"   ⚠️ BERTScore loading failed: {e}")
    bertscore = None

# =========================================================
# 5️⃣ CREATE OUTPUT DIRECTORY
# =========================================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"\n📁 Training directory: {CHECKPOINT_DIR}")

# =========================================================
# 6️⃣ LOAD AND COMBINE EPISODES
# =========================================================
print("\n📂 Loading and combining episodes...")

all_data = []
episode_stats = {}
audio_file_mapping = {}  # Map file to episode extract directory

for ep_num in EPISODES_TO_INCLUDE:
    print(f"\n   Processing Episode {ep_num}...")

    # Load JSON
    json_path = EPISODE_JSON_PATTERN.format(ep=ep_num)
    if not os.path.exists(json_path):
        print(f"   ⚠️ JSON file not found: {json_path}")
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        ep_data = json.load(f)

    # Extract audio if needed
    zip_path = EPISODE_ZIP_PATTERN.format(ep=ep_num)
    extract_dir = EXTRACT_BASE_DIR.format(ep=ep_num)

    if not os.path.exists(extract_dir):
        if os.path.exists(zip_path):
            print(f"   📦 Extracting audio from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"   ✅ Extracted to {extract_dir}")
        else:
            print(f"   ⚠️ Audio ZIP not found: {zip_path}")
            continue
    else:
        print(f"   ✅ Audio already extracted: {extract_dir}")

    # Map files to extract directory
    for item in ep_data:
        audio_file_mapping[item['file']] = extract_dir

    # Add episode number to each item
    for item in ep_data:
        item['episode'] = ep_num

    all_data.extend(ep_data)
    episode_stats[ep_num] = len(ep_data)
    print(f"   ✅ Loaded {len(ep_data)} samples from Episode {ep_num}")

print("\n" + "="*80)
print("📊 DATASET STATISTICS")
print("="*80)
print(f"Total samples: {len(all_data)}")
print(f"\nPer-episode breakdown:")
for ep_num, count in sorted(episode_stats.items()):
    percentage = (count / len(all_data)) * 100
    print(f"   Episode {ep_num:2d}: {count:4d} samples ({percentage:5.2f}%)")
print("="*80)

# Save combined dataset
combined_json_path = os.path.join(CHECKPOINT_DIR, "combined_dataset.json")
with open(combined_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
print(f"\n💾 Combined dataset saved: {combined_json_path}")

# Create dataset
dataset = Dataset.from_list(all_data)

# Split into train/validation
dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=RANDOM_SEED)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

print(f"\n📊 Dataset split:")
print(f"   Training:   {len(train_dataset)} samples ({(1-VALIDATION_SPLIT)*100:.1f}%)")
print(f"   Validation: {len(val_dataset)} samples ({VALIDATION_SPLIT*100:.1f}%)")

# =========================================================
# 7️⃣ GRADIENT TRACKING UTILITIES
# =========================================================
def compute_gradient_stats(model):
    """Compute gradient statistics for the model"""
    total_norm = 0.0
    encoder_norm = 0.0
    decoder_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')

    grad_norms = {'encoder': [], 'decoder': []}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2

            if 'encoder' in name:
                encoder_norm += param_norm ** 2
                grad_norms['encoder'].append(param_norm)
            elif 'decoder' in name:
                decoder_norm += param_norm ** 2
                grad_norms['decoder'].append(param_norm)

            max_grad = max(max_grad, param.grad.abs().max().item())
            min_grad = min(min_grad, param.grad.abs().min().item())

    total_norm = total_norm ** 0.5
    encoder_norm = encoder_norm ** 0.5 if grad_norms['encoder'] else 0.0
    decoder_norm = decoder_norm ** 0.5 if grad_norms['decoder'] else 0.0

    return {
        'total_norm': total_norm,
        'encoder_norm': encoder_norm,
        'decoder_norm': decoder_norm,
        'max_grad': max_grad,
        'min_grad': min_grad if min_grad != float('inf') else 0.0,
        'mean_encoder': np.mean(grad_norms['encoder']) if grad_norms['encoder'] else 0.0,
        'mean_decoder': np.mean(grad_norms['decoder']) if grad_norms['decoder'] else 0.0
    }

def log_gradient_flow(model, step, log_file):
    """Log detailed gradient flow information"""
    grad_info = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info.append({
                'name': name,
                'grad_norm': param.grad.data.norm(2).item(),
                'grad_mean': param.grad.data.mean().item(),
                'grad_std': param.grad.data.std().item(),
            })

    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\nStep: {step}\n{'='*80}\n")
        for info in grad_info[:10]:
            f.write(f"Layer: {info['name']}\n")
            f.write(f"  Grad Norm: {info['grad_norm']:.6f}\n")
            f.write(f"  Grad Mean: {info['grad_mean']:.6f}\n")
            f.write(f"  Grad Std: {info['grad_std']:.6f}\n")

    return grad_info

# =========================================================
# 8️⃣ EVALUATION UTILITIES
# =========================================================
def compute_bertscore(references, predictions):
    """Compute BERTScore if available"""
    if bertscore is None:
        return None

    try:
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="ar",
            model_type="bert-base-multilingual-cased"
        )
        return {
            'precision': float(np.mean(results['precision'])),
            'recall': float(np.mean(results['recall'])),
            'f1': float(np.mean(results['f1']))
        }
    except Exception as e:
        print(f"   ⚠️ BERTScore computation failed: {e}")
        return None

def evaluate_comprehensive(model, processor, eval_dataset, audio_file_mapping, device, save_details=False):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()

    results = []
    wer_scores = []
    cer_scores = []
    losses = []
    all_references = []
    all_predictions = []

    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE EVALUATION")
    print("="*80)

    with torch.no_grad():
        for idx, row in enumerate(tqdm(eval_dataset, desc="Evaluating", leave=False)):
            if not isinstance(row, dict) or "file" not in row or not row.get("text", "").strip():
                continue

            # Get correct audio directory for this file
            extract_dir = audio_file_mapping.get(row["file"])
            if not extract_dir:
                continue

            path = os.path.join(extract_dir, row["file"])
            if not os.path.exists(path):
                continue

            wav, sr = torchaudio.load(path)
            wav = wav.squeeze()

            inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
            reference = row["text"].strip()

            # Compute loss
            try:
                labels = processor.tokenizer(reference, return_tensors="pt").input_ids.to(device)
                max_len = model.config.max_target_positions

                if labels.shape[1] > max_len:
                    # Skip long samples in evaluation
                    continue

                with torch.cuda.amp.autocast():
                    out = model(input_features=inputs.input_features, labels=labels)
                    loss = out.loss.item()
                losses.append(loss)
            except Exception as e:
                loss = None

            # Generate prediction - FORCE ARABIC
            pred_ids = model.generate(
                        inputs.input_features,
                        language="ar",
                        task="transcribe",
                        num_beams=3,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.4,
                        temperature=0.8,
                        top_p=0.9,
)

            prediction = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

            # Compute WER and CER
            try:
                wer_score = wer(reference, prediction)
                cer_score = cer(reference, prediction)
                wer_scores.append(wer_score)
                cer_scores.append(cer_score)
            except Exception as e:
                wer_score = None
                cer_score = None

            all_references.append(reference)
            all_predictions.append(prediction)

            result_entry = {
                'file': row['file'],
                'episode': row.get('episode', 'unknown'),
                'reference': reference,
                'prediction': prediction,
                'wer': float(wer_score) if wer_score is not None else None,
                'cer': float(cer_score) if cer_score is not None else None,
                'loss': float(loss) if loss is not None else None,
            }
            results.append(result_entry)

    # Compute BERTScore
    print("\n📊 Computing BERTScore...")
    bert_scores = compute_bertscore(all_references, all_predictions)

    # Safe statistics functions
    def safe_mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def safe_median(vals):
        return float(np.median(vals)) if vals else 0.0

    def safe_std(vals):
        return float(np.std(vals)) if vals else 0.0

    def safe_min(vals):
        return float(np.min(vals)) if vals else 0.0

    def safe_max(vals):
        return float(np.max(vals)) if vals else 0.0

    # Compute statistics
    metrics_summary = {
        'wer': {
            'mean': safe_mean(wer_scores),
            'median': safe_median(wer_scores),
            'std': safe_std(wer_scores),
            'min': safe_min(wer_scores),
            'max': safe_max(wer_scores),
        },
        'cer': {
            'mean': safe_mean(cer_scores),
            'median': safe_median(cer_scores),
            'std': safe_std(cer_scores),
            'min': safe_min(cer_scores),
            'max': safe_max(cer_scores),
        },
        'loss': {
            'mean': safe_mean(losses),
            'median': safe_median(losses),
            'std': safe_std(losses),
            'min': safe_min(losses),
            'max': safe_max(losses),
        },
        'bertscore': bert_scores,
        'total_samples': len(results)
    }

    # Print summary
    print("\n" + "="*80)
    print("📈 EVALUATION SUMMARY")
    print("="*80)

    print(f"\n🎯 Word Error Rate (WER):")
    print(f"   Mean:   {metrics_summary['wer']['mean']:.4f}")
    print(f"   Median: {metrics_summary['wer']['median']:.4f}")
    print(f"   Std:    {metrics_summary['wer']['std']:.4f}")
    print(f"   Min:    {metrics_summary['wer']['min']:.4f}")
    print(f"   Max:    {metrics_summary['wer']['max']:.4f}")

    print(f"\n🎯 Character Error Rate (CER):")
    print(f"   Mean:   {metrics_summary['cer']['mean']:.4f}")
    print(f"   Median: {metrics_summary['cer']['median']:.4f}")
    print(f"   Std:    {metrics_summary['cer']['std']:.4f}")
    print(f"   Min:    {metrics_summary['cer']['min']:.4f}")
    print(f"   Max:    {metrics_summary['cer']['max']:.4f}")

    print(f"\n🎯 Loss:")
    print(f"   Mean:   {metrics_summary['loss']['mean']:.4f}")
    print(f"   Median: {metrics_summary['loss']['median']:.4f}")
    print(f"   Std:    {metrics_summary['loss']['std']:.4f}")

    if bert_scores:
        print(f"\n🎯 BERTScore:")
        print(f"   Precision: {bert_scores['precision']:.4f}")
        print(f"   Recall:    {bert_scores['recall']:.4f}")
        print(f"   F1:        {bert_scores['f1']:.4f}")

    print(f"\n📊 Evaluated {metrics_summary['total_samples']} samples")
    print("="*80)

    # Show examples
    if save_details and results:
        sorted_by_wer = sorted([r for r in results if r['wer'] is not None],
                               key=lambda x: x['wer'], reverse=True)

        print("\n" + "="*80)
        print(f"📉 WORST {min(SHOW_EXAMPLES, len(sorted_by_wer))} PREDICTIONS:")
        print("="*80)
        for i, r in enumerate(sorted_by_wer[:SHOW_EXAMPLES], 1):
            print(f"\n{i}. Episode {r['episode']} - {r['file']}")
            print(f"   WER: {r['wer']:.4f} | CER: {r['cer']:.4f}")
            print(f"   REF: {r['reference'][:80]}...")
            print(f"   PRE: {r['prediction'][:80]}...")

        print("\n" + "="*80)
        print(f"📈 BEST {min(SHOW_EXAMPLES, len(sorted_by_wer))} PREDICTIONS:")
        print("="*80)
        for i, r in enumerate(sorted_by_wer[-SHOW_EXAMPLES:], 1):
            print(f"\n{i}. Episode {r['episode']} - {r['file']}")
            print(f"   WER: {r['wer']:.4f} | CER: {r['cer']:.4f}")
            print(f"   REF: {r['reference'][:80]}...")
            print(f"   PRE: {r['prediction'][:80]}...")

    model.train()
    return metrics_summary, results

# =========================================================
# 9️⃣ TRAINING LOOP
# =========================================================
print("\n" + "="*80)
print("🎯 TRAINING CONFIGURATION")
print("="*80)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}: LR={LEARNING_RATES[epoch]}, Clip={GRADIENT_CLIPS[epoch]}, Freeze={FREEZE_ENCODER[epoch]}")
print("="*80)

metrics_log = {"config": {
    "episodes": EPISODES_TO_INCLUDE,
    "total_samples": len(all_data),
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "epochs": EPOCHS,
    "learning_rates": LEARNING_RATES,
    "gradient_clips": GRADIENT_CLIPS,
}, "epochs": [], "gradient_stats": []}

best_wer = float("inf")
gradient_log_file = os.path.join(CHECKPOINT_DIR, "gradient_flow.log")

with open(gradient_log_file, 'w') as f:
    f.write(f"Training started at {datetime.now()}\n")

global_step = 0
training_losses = []

for epoch in range(EPOCHS):
    print("\n" + "="*80)
    print(f"📍 EPOCH {epoch+1}/{EPOCHS}")
    print("="*80)

    lr = LEARNING_RATES[epoch]
    clip = GRADIENT_CLIPS[epoch]
    freeze = FREEZE_ENCODER[epoch]

    print(f"   LR = {lr}")
    print(f"   Clip = {clip}")
    print(f"   Encoder frozen = {freeze}")

    # Skip tracking counters
    skipped_long = 0
    skipped_no_text = 0
    skipped_no_file = 0
    processed_samples = 0

    # Freeze/unfreeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = not freeze
    for p in model.model.decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    scaler = torch.cuda.amp.GradScaler()

    running_loss = 0
    epoch_losses = []
    epoch_gradient_stats = []
    model.train()

    bar = tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=130)

    for step in bar:
        row = train_dataset[step]

        # Check for text
        text = row.get("text", "").strip()
        if not text:
            skipped_no_text += 1
            continue

        # Check for file mapping
        extract_dir = audio_file_mapping.get(row.get("file"))
        if not extract_dir:
            skipped_no_file += 1
            continue

        # Check if file exists
        path = os.path.join(extract_dir, row["file"])
        if not os.path.exists(path):
            skipped_no_file += 1
            continue

        # Load audio
        wav, sr = torchaudio.load(path)
        wav = wav.squeeze()

        # Process inputs
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
        labels = processor.tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Check for long labels
        max_len = model.config.max_target_positions
        if labels.shape[1] > max_len:
            skipped_long += 1
            continue

        # Forward pass
        with torch.cuda.amp.autocast():
            out = model(input_features=inputs.input_features, labels=labels)
            loss = out.loss

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_stats_before = compute_gradient_stats(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        grad_stats_after = compute_gradient_stats(model)

        scaler.step(optimizer)
        scaler.update()

        loss_value = loss.item()
        running_loss += loss_value
        epoch_losses.append(loss_value)
        processed_samples += 1
        training_losses.append(loss_value)
        global_step += 1

        if global_step % GRADIENT_LOG_FREQUENCY == 0:
            grad_log_entry = {
                'epoch': epoch + 1,
                'step': global_step,
                'loss': loss_value,
                'before_clip': grad_stats_before,
                'after_clip': grad_stats_after,
                'clipped': grad_stats_before['total_norm'] > clip
            }
            epoch_gradient_stats.append(grad_log_entry)

            bar.set_postfix({
                'loss': f"{loss_value:.3f}",
                'avg': f"{np.mean(epoch_losses):.3f}",
                'grad': f"{grad_stats_after['total_norm']:.3f}",
            })
        else:
            bar.set_postfix({'loss': f"{loss_value:.3f}", 'avg': f"{np.mean(epoch_losses):.3f}"})

        if global_step % GRADIENT_DETAILED_LOG == 0:
            log_gradient_flow(model, global_step, gradient_log_file)

    metrics_log["gradient_stats"].extend(epoch_gradient_stats)

    if epoch_gradient_stats:
        avg_grad = np.mean([g['after_clip']['total_norm'] for g in epoch_gradient_stats])
        clipped_count = sum([1 for g in epoch_gradient_stats if g['clipped']])
        print(f"\n📈 Gradient: Avg={avg_grad:.4f}, Clipped={clipped_count}/{len(epoch_gradient_stats)}")

    print(f"\n📊 Loss: Mean={np.mean(epoch_losses):.4f}, Median={np.median(epoch_losses):.4f}")

    # Print skip statistics
    total_skipped = skipped_long + skipped_no_text + skipped_no_file
    print(f"\n📊 Epoch {epoch+1} Sample Statistics:")
    print(f"   Total samples:     {len(train_dataset)}")
    print(f"   Processed:         {processed_samples}")
    print(f"   Skipped (long):    {skipped_long}")
    print(f"   Skipped (no text): {skipped_no_text}")
    print(f"   Skipped (no file): {skipped_no_file}")
    print(f"   Skip rate:         {total_skipped/len(train_dataset)*100:.2f}%")

    # Validation
    metrics_summary, detailed_results = evaluate_comprehensive(
        model, processor, val_dataset, audio_file_mapping, device, save_details=SAVE_DETAILED_RESULTS
    )

    if metrics_summary['wer']['mean'] < best_wer:
        best_wer = metrics_summary['wer']['mean']
        model.save_pretrained(CHECKPOINT_DIR)
        processor.save_pretrained(CHECKPOINT_DIR)
        print("   💾 Saved BEST model")

    metrics_log["epochs"].append({
        "epoch": epoch+1,
        "loss": {"mean": float(np.mean(epoch_losses)), "median": float(np.median(epoch_losses))},
        "validation_metrics": metrics_summary,
        "samples": {
            "total": len(train_dataset),
            "processed": processed_samples,
            "skipped_long": skipped_long,
            "skipped_no_text": skipped_no_text,
            "skipped_no_file": skipped_no_file,
            "skip_rate": total_skipped / len(train_dataset)
        }
    })

# Save all metrics
print("\n💾 Saving metrics...")
with open(os.path.join(CHECKPOINT_DIR, "training_log.json"), "w") as f:
    json.dump(metrics_log, f, indent=2)

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print(f"📌 Best WER: {best_wer:.4f}")
print(f"📁 Saved at: {CHECKPOINT_DIR}")
print("="*80)
