"""Convert labelled audio folders into model-ready Mel spectrogram images.

Expected input layout:
    dataset/
      english/real/*.wav
      english/fake/*.wav
      lang/<language>/real/*.wav
      lang/<language>/fake/*.wav
"""

from argparse import ArgumentParser
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 16_000
TARGET_SAMPLES = SAMPLE_RATE * 3
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg"}


def process_audio(file_path: Path, save_path: Path) -> bool:
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        audio = librosa.util.fix_length(audio, size=TARGET_SAMPLES)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(4, 4), dpi=128)
        librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=512, cmap="magma", ax=axis)
        axis.axis("off")
        figure.subplots_adjust(0, 0, 1, 1)
        figure.savefig(save_path, format="png", dpi=128, bbox_inches=None, pad_inches=0)
        plt.close(figure)
        return True
    except Exception as exc:
        print(f"Could not process {file_path}: {exc}")
        return False


def convert_tree(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    processed = failed = 0
    for source in input_dir.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative = source.relative_to(input_dir).with_suffix(".png")
        if process_audio(source, output_dir / relative):
            processed += 1
        else:
            failed += 1
    return processed, failed


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Root folder containing labelled audio folders.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_spectrograms"))
    args = parser.parse_args()
    processed, failed = convert_tree(args.input_dir, args.output_dir)
    print(f"Processed: {processed}; failed: {failed}; output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
