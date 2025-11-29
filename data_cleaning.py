#!/usr/bin/env python3
"""
Whisper JSON Cleaner - Vector Version
Simply edit the INPUT_FILES and OUTPUT_FILES lists below
"""

# ============================================================================
# DEFINE YOUR INPUT AND OUTPUT FILES HERE (AS VECTORS/LISTS)
# ============================================================================

# INPUT FILES: List of paths to your JSON files
INPUT_FILES = [
]

# OUTPUT FILES: Corresponding output paths (must be same length as INPUT_FILES)
OUTPUT_DIRECTORY = r"C:\Users\Ali\Downloads\Al_Atawla_Scripts"
# ============================================================================
# DON'T EDIT BELOW THIS LINE
# ============================================================================

# !/usr/bin/env python3
"""
Whisper JSON Cleaner - Simple Version with Variables
Edit the variables below to set your input/output paths
"""


import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


class WhisperJSONCleaner:
    def __init__(self):
        """Initialize the Whisper JSON cleaner with Arabic-specific filters"""

        # Arabic Unicode pattern
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')

        # Common Whisper hallucinations (case-insensitive)
        self.hallucinations = [
            r'^thank you\.?$',
            r'^i don\'?t know\.?$',
            r'^продолжение следует\.{3}$',
            r'^а я умер\.?$',
            r'^salam\.?$',
            r'^oughta accept that\.?$',
            r'^get off!?$',
            r'^yol\.?$',
            r'^ben alayim\.?$',
            r'^\.\.\.[a-z\s]+\.?$',
            r'^ishtariko? fil[- ]?canat?\.?$',
            r'^ishtar[ie]ko? fil[- ]?qanat?\.?$',
            r'^subscribe\.?$',
            r'^like and subscribe\.?$',
            r'^اشترك في القناة\.?$',
            r'^اشتركوا في القناة\.?$',

        ]

        # Music/sound placeholders (Arabic)
        self.music_placeholders = [
            r'^موسيقى$',
            r'^موسيقا$',
            r'^مو+سيقى$',
        ]

        # Translator credits to remove
        self.translator_patterns = [
            r'ترجمة\s+\w+',
            r'نانا\s+محمد',
            r'ترجمة:?\s*',
        ]

        # Very short Arabic interjections when they appear alone
        self.short_interjections = [
            r'^اه+$',
            r'^ها+$',
            r'^ايه$',
            r'^اي+$',
            r'^اوه+$',
            r'^يا+$',
        ]

    def is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        return bool(self.arabic_pattern.search(text))

    def is_hallucination(self, text: str) -> bool:
        """Check if text is a common Whisper hallucination"""
        text_lower = text.lower().strip()
        for pattern in self.hallucinations:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def is_music_placeholder(self, text: str) -> bool:
        """Check if text is a music placeholder"""
        text_clean = text.strip()
        for pattern in self.music_placeholders:
            if re.match(pattern, text_clean):
                return True
        return False

    def has_translator_credit(self, text: str) -> bool:
        """Check if text contains translator credits"""
        for pattern in self.translator_patterns:
            if re.search(pattern, text):
                return True
        return False

    def is_repetitive(self, text: str, threshold: int = 2) -> bool:
        """
        Check if text is overly repetitive

        Args:
            text: The text to check
            threshold: How many times a phrase can repeat before being flagged (default 2 = flags 3+)

        Returns:
            True if text is too repetitive
        """
        words = text.split()

        # Check even very short texts (2+ words)
        if len(words) < 2:
            return False

        # Special check for 2-3 word texts that are just the same word(s) repeated
        if len(words) <= 3:
            unique_words = set(words)
            if len(unique_words) == 1:  # All words are identical: "word word word"
                return True

        # Special check for exactly 2 words that are identical
        if len(words) == 2:
            if words[0] == words[1]:  # "word word"
                return True

        # For 4+ words, do full repetition checks
        if len(words) < 4:
            return False

        # Check for exact word repetition
        # Example: "لا لا لا لا لا لا" should be flagged
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # If any single word repeats more than threshold times
        for word, count in word_counts.items():
            if count > threshold and len(word) > 1:  # Don't count single letters
                return True

        # Check for phrase repetition
        # Example: "شكرا جزيلا شكرا جزيلا شكرا جزيلا" should be flagged
        for phrase_len in [2, 3, 4]:  # Check 2-word, 3-word, 4-word phrases
            if len(words) < phrase_len * 2:
                continue

            phrases = {}
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                phrases[phrase] = phrases.get(phrase, 0) + 1

            # If any phrase repeats more than threshold/2 times
            for phrase, count in phrases.items():
                if count > max(2, threshold // 2):
                    return True

        # Check for alternating pattern (A B A B A B)
        # Example: "نعم لا نعم لا نعم لا" should be flagged
        if len(words) >= 6:
            for i in range(len(words) - 5):
                if (words[i] == words[i + 2] == words[i + 4] and
                        words[i + 1] == words[i + 3] == words[i + 5] and
                        words[i] != words[i + 1]):
                    return True

        return False

    def is_short_interjection(self, text: str) -> bool:
        """Check if text is just a short interjection"""
        text_clean = text.strip()
        if len(text_clean) <= 3:
            for pattern in self.short_interjections:
                if re.match(pattern, text_clean):
                    return True
        return False

    def is_valid_arabic_segment(self, text: str) -> bool:
        """
        Determine if a segment should be kept
        STRICT MODE: Only allows Arabic characters + basic punctuation/spaces
        """
        text_clean = text.strip()

        if not text_clean:
            return False

        if self.is_hallucination(text_clean):
            return False

        if self.is_music_placeholder(text_clean):
            return False

        if self.has_translator_credit(text_clean):
            return False

        if self.is_short_interjection(text_clean):
            return False

        # NEW: Check for repetition
        if self.is_repetitive(text_clean):
            return False

        if not self.is_arabic_text(text_clean):
            return False

        # STRICT FILTER: Only Arabic + spaces + Arabic punctuation
        allowed_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s،؛؟!.\-]'

        allowed_chars = len([c for c in text_clean if re.match(allowed_pattern, c)])
        allowed_percentage = allowed_chars / len(text_clean)

        if allowed_percentage < 0.95:
            return False

        # Must have at least 3 Arabic letters
        arabic_letters = len([c for c in text_clean if
                              re.match(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', c)])

        if arabic_letters < 3:
            return False

        return True

    def clean_segment(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single segment"""
        text = segment.get('text', '').strip()

        if not self.is_valid_arabic_segment(text):
            return None

        for pattern in self.translator_patterns:
            text = re.sub(pattern, '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return None

        return {
            'start': segment['start'],
            'end': segment['end'],
            'text': text
        }

    def clean_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single chunk with its segments"""
        cleaned_segments = []

        for segment in chunk.get('segments', []):
            cleaned_seg = self.clean_segment(segment)
            if cleaned_seg:
                cleaned_segments.append(cleaned_seg)

        cleaned_text = ' '.join([seg['text'] for seg in cleaned_segments])

        return {
            'file': chunk['file'],
            'segments': cleaned_segments,
            'text': cleaned_text
        }

    def clean_json_file(self, input_path: str, output_path: str) -> dict:
        """Clean a Whisper JSON transcription file"""
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_file = Path(output_path)

        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clean all chunks
        original_segments = 0
        cleaned_segments = 0

        cleaned_data = []
        for chunk in data:
            original_segments += len(chunk.get('segments', []))
            cleaned_chunk = self.clean_chunk(chunk)
            cleaned_segments += len(cleaned_chunk['segments'])

            if cleaned_chunk['segments']:
                cleaned_data.append(cleaned_chunk)

        # Write output JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

        # Calculate statistics
        stats = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'original_chunks': len(data),
            'cleaned_chunks': len(cleaned_data),
            'original_segments': original_segments,
            'cleaned_segments': cleaned_segments,
            'segments_removed': original_segments - cleaned_segments,
            'removal_percentage': round((1 - cleaned_segments / original_segments) * 100,
                                        2) if original_segments > 0 else 0
        }

        return stats


def main():
    print("=" * 60)
    print("WHISPER JSON CLEANER - SIMPLE VERSION")
    print("=" * 60)
    print()

    # Check if variables are set
    if INPUT_FILES == ["path/to/your/file1.json", "path/to/your/file2.json", "path/to/your/file3.json"]:
        print("❌ ERROR: Please edit the INPUT_FILES variable at the top of this script!")
        print()
        print("Open this file and change:")
        print('  INPUT_FILES = ["path/to/your/file1.json", ...]')
        print()
        print("To your actual file paths, like:")
        print('  INPUT_FILES = ["/home/user/data/episode1.json", ...]')
        return

    if OUTPUT_DIRECTORY == "path/to/your/output/folder":
        print("❌ ERROR: Please edit the OUTPUT_DIRECTORY variable at the top of this script!")
        print()
        print("Open this file and change:")
        print('  OUTPUT_DIRECTORY = "path/to/your/output/folder"')
        print()
        print("To your actual output path, like:")
        print('  OUTPUT_DIRECTORY = "/home/user/cleaned_data"')
        return

    # Create output directory
    output_dir = Path(OUTPUT_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create cleaner
    cleaner = WhisperJSONCleaner()

    print(f"Input files: {len(INPUT_FILES)}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print()

    # Clean all files
    all_stats = []

    for input_file in INPUT_FILES:
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"✗ File not found: {input_file}")
            continue

        output_path = output_dir / f"{input_path.stem}_cleaned{input_path.suffix}"

        try:
            stats = cleaner.clean_json_file(str(input_path), str(output_path))
            all_stats.append(stats)
            print(f"✓ Cleaned: {input_path.name}")
            print(f"  → {output_path}")
            print(
                f"  Segments: {stats['original_segments']} → {stats['cleaned_segments']} ({stats['removal_percentage']}% removed)")
            print()
        except Exception as e:
            print(f"✗ Error: {input_file}: {str(e)}")
            print()

    # Summary
    if all_stats:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_original = sum(s['original_segments'] for s in all_stats)
        total_cleaned = sum(s['cleaned_segments'] for s in all_stats)
        total_removed = sum(s['segments_removed'] for s in all_stats)

        print(f"Files processed: {len(all_stats)}/{len(INPUT_FILES)}")
        print(f"Total segments: {total_original:,}")
        print(f"Kept: {total_cleaned:,} ({round(total_cleaned / total_original * 100, 1)}%)")
        print(f"Removed: {total_removed:,} ({round(total_removed / total_original * 100, 1)}%)")
        print()
        print(f"✓ Cleaned files saved to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
