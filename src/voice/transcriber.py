from __future__ import annotations

from dataclasses import dataclass

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    transcript: str
    language: str
    language_probability: float
    model_size: str
    device: str
    compute_type: str


class AudioTranscriptionError(Exception):
    pass


class AudioTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        try:
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            segment_list = list(segments)

            transcript_parts = []
            for segment in segment_list:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)

            transcript = " ".join(transcript_parts).strip()

            return TranscriptionResult(
                transcript=transcript,
                language=info.language,
                language_probability=info.language_probability,
                model_size=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        except Exception as exc:
            raise AudioTranscriptionError(f"Audio transcription failed: {exc}") from exc