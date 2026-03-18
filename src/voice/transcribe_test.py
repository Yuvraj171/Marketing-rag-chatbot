from __future__ import annotations

import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(str(path), beam_size=5)

    segment_list = list(segments)

    transcript_parts = []
    for segment in segment_list:
        text = segment.text.strip()
        if text:
            transcript_parts.append(text)

    transcript = " ".join(transcript_parts).strip()

    print("\n" + "=" * 80)
    print("TRANSCRIPTION INFO")
    print("=" * 80)
    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.4f}")
    print(f"Model size: {model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")

    print("\n" + "=" * 80)
    print("TRANSCRIPT")
    print("=" * 80)
    print(transcript if transcript else "[No transcript produced]")

    return transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="Test faster-whisper transcription locally.")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--model_size", default="base", help="Whisper model size: tiny, base, small, medium, large-v3, etc.")
    parser.add_argument("--device", default="cpu", help="Device to run on: cpu or cuda")
    parser.add_argument("--compute_type", default="int8", help="Compute type, e.g. int8, float16, int8_float16")

    args = parser.parse_args()

    transcribe_audio(
        audio_path=args.audio_path,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )


if __name__ == "__main__":
    main()