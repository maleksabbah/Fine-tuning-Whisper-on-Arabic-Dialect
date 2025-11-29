# =========================================================
# 1️⃣ MOUNT GOOGLE DRIVE
# =========================================================
from google.colab import drive
drive.mount('/content/drive')


DRIVE_FOLDER = "/content/drive/MyDrive"


# =========================================================
# 2️⃣ SETUP PATHS (EDIT THESE)
# =========================================================
ZIP_FILES = [

    "Al_Atawla_Eps_14_chunks.zip",
    "Al_Atawla_Eps_15_chunks.zip",
    "Al_Atawla_Eps_16_chunks.zip",
    "Al_Atawla_Eps_17_chunks.zip",
    "Al_Atawla_Eps_18_chunks.zip",
    "Al_Atawla_Eps_19_chunks.zip",
    "Al_Atawla_Eps_20_chunks.zip",
    "Al_Atawla_Eps_21_chunks.zip",
    "Al_Atawla_Eps_22_chunks.zip",
    "Al_Atawla_Eps_23_chunks.zip",
    "Al_Atawla_Eps_24_chunks.zip",
    "Al_Atawla_Eps_25_chunks.zip",
    "Al_Atawla_Eps_26_chunks.zip",
    "Al_Atawla_Eps_27_chunks.zip",
    "Al_Atawla_Eps_28_chunks.zip",
    "Al_Atawla_Eps_29_chunks.zip",
    "Al_Atawla_Eps_30_chunks.zip"

]

# ⬆️ Add as many as you want

DRIVE_FOLDER = "/content"

# =========================================================
# 3️⃣ INSTALL DEPENDENCIES
# =========================================================
!pip install -q git+https://github.com/openai/whisper.git tqdm torch soundfile

import os, json, zipfile, torch, whisper, contextlib
from tqdm import tqdm

# =========================================================
# 4️⃣ DEVICE SETUP
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", device)

import whisper.utils
whisper.utils.TQDM_DISABLE = True

# =========================================================
# LOAD WHISPER LARGE MODEL ONCE
# =========================================================
model = whisper.load_model("large", device=device)
print("✅ Whisper Large model loaded")

# =========================================================
# 5️⃣ PROCESS EACH ZIP FILE
# =========================================================
for ZIP_NAME in ZIP_FILES:

    print("\n" + "="*80)
    print(f"📦 PROCESSING ZIP: {ZIP_NAME}")
    print("="*80)

    zip_path = f"{DRIVE_FOLDER}/{ZIP_NAME}"
    extract_dir = f"/content/{ZIP_NAME.replace('.zip','')}"
    output_json = f"/content/{ZIP_NAME.replace('.zip','')}_whisper_segments.json"

    # -----------------------------
    # UNZIP AUDIO CHUNKS
    # -----------------------------
    if not os.path.exists(extract_dir):
        print("DEBUG ZIP PATH: ", zip_path)
        print("EXISTS?",os.path.exists(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Extracted audio to: {extract_dir}")
    else:
        print("ℹ️ Audio folder already extracted.")

    # -----------------------------
    # LIST AUDIO FILES
    # -----------------------------
    audio_files = sorted([
        f for f in os.listdir(extract_dir)
        if f.lower().endswith(('.wav', '.mp3', '.flac'))
    ])

    print(f"🎧 Found {len(audio_files)} chunks...\n")

    # -----------------------------
    # TRANSCRIBE WITH ONE TQDM
    # -----------------------------
    results = []
    progress = tqdm(audio_files, desc=f"🪶 Transcribing {ZIP_NAME}", ncols=100)

    for filename in progress:
        file_path = os.path.join(extract_dir, filename)

        # silence Whisper internals
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = model.transcribe(file_path, verbose=None, fp16=False)

        results.append({
            "file": filename,
            "segments": [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                for seg in result["segments"]
            ],
            "text": result["text"].strip()
        })

    # -----------------------------
    # SAVE JSON
    # -----------------------------
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved JSON: {output_json}")

    # Copy to Drive
    drive_output_path = f"{DRIVE_FOLDER}/{ZIP_NAME.replace('.zip','')}_whisper_segments.json"
    !cp "{output_json}" "{drive_output_path}"
    print(f"✅ Copied to Drive: {drive_output_path}")

print("\n🎉 ALL ZIP FILES PROCESSED SUCCESSFULLY 🎉")
