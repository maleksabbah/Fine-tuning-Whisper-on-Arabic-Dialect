import os
import zipfile
from pydub import AudioSegment
from pydub.silence import split_on_silence


class MultiEpisodeChunker:
    def __init__(
        self,
        input_files: list,      # list of mp4 + wav files, any show
        output_root: str,       # e.g. "all_chunks"
        max_chunk_ms: int = 25000,
        min_silence_len: int = 500,
        silence_thresh_db: int = 20,
        keep_silence_ms: int = 300,
    ):
        self.input_files = input_files
        self.output_root = output_root
        self.max_chunk_ms = max_chunk_ms
        self.min_silence_len = min_silence_len
        self.silence_thresh_db = silence_thresh_db
        self.keep_silence_ms = keep_silence_ms

        os.makedirs(self.output_root, exist_ok=True)

    def mp4_to_wav(self, mp4_path):
        """Convert MP4 → WAV and return the wav path."""
        wav_path = mp4_path.rsplit(".", 1)[0] + ".wav"
        print(f"🎧 Converting: {mp4_path} → {wav_path}")
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio.export(wav_path, format="wav")
        return wav_path

    def split_long_chunk(self, chunk):
        if len(chunk) <= self.max_chunk_ms:
            return [chunk]
        return [
            chunk[i:i+self.max_chunk_ms]
            for i in range(0, len(chunk), self.max_chunk_ms)
        ]

    def zip_episode(self,episode_folder):
        zip_path = episode_folder + ".zip"
        print(f"/n Zipping folder: {episode_folder}")


        with zipfile.ZipFile(zip_path, "w",zipfile.ZIP_DEFLATED) as zip as z:

    def process_episode(self, wav_path):
        # Episode folder name = filename without extension
        ep_name = os.path.splitext(os.path.basename(wav_path))[0]
        ep_outdir = os.path.join(self.output_root, f"{ep_name}_chunks")
        os.makedirs(ep_outdir, exist_ok=True)

        print(f"\n🔊 Loading episode: {ep_name}")
        audio = AudioSegment.from_wav(wav_path)

        print("✂️ Splitting on silence...")
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=audio.dBFS - self.silence_thresh_db,
            keep_silence=self.keep_silence_ms,
        )

        print(f"📦 Found {len(chunks)} initial segments")

        counter = 0
        for c in chunks:
            for sub in self.split_long_chunk(c):
                out_path = os.path.join(ep_outdir, f"chunk_{counter:03d}.wav")
                sub.export(out_path, format="wav")
                print(f"  → Saved {out_path} ({len(sub)/1000:.1f}s)")
                counter += 1

        print(f"✅ Finished {ep_name}: {counter} chunks.")



    def process_all(self):
        for file_path in self.input_files:

            # Convert if MP4
            if file_path.lower().endswith(".mp4"):
                wav_path = self.mp4_to_wav(file_path)
            else:
                wav_path = file_path

            # Process that episode
            self.process_episode(wav_path)

        print("\n🎉 All episodes (all shows) processed successfully!")

input_files = [
]


chunker = MultiEpisodeChunker(
    input_files = input_files,
    output_root = r"C:\Users\Ali\Downloads\Chunked_Audio"
    )
chunker.process_all()
