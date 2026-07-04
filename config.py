"""
Central configuration — reads from environment / .env file.
All pipeline modules import from here rather than reading os.environ directly.

No API keys are required.  Every stage runs on the local machine using
open-source models downloaded from HuggingFace on first run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


# ─── Type aliases ────────────────────────────────────────────────────────────

TranslationBackend = Literal["nllb"]
HindiTone = Literal["formal", "casual", "auto"]
InferenceDevice = Literal["cpu", "cuda", "auto"]


@dataclass
class AudioConfig:
    sample_rate: int = field(default_factory=lambda: _env_int("SAMPLE_RATE", 16000))
    channels: int = 1
    frame_duration_ms: int = 30      # WebRTC VAD requires 10, 20, or 30 ms
    input_device: int = field(default_factory=lambda: _env_int("INPUT_DEVICE", -1))

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_duration_ms / 1000)


@dataclass
class VADConfig:
    aggressiveness: int = field(
        default_factory=lambda: _env_int("VAD_AGGRESSIVENESS", 2)
    )
    silence_duration_ms: int = field(
        default_factory=lambda: _env_int("SILENCE_DURATION_MS", 800)
    )
    min_speech_duration_ms: int = field(
        default_factory=lambda: _env_int("MIN_SPEECH_DURATION_MS", 300)
    )
    pre_speech_pad_ms: int = 200
    post_speech_pad_ms: int = 300


@dataclass
class ASRConfig:
    model: str = field(
        default_factory=lambda: _env("WHISPER_MODEL", "base.en")
    )
    device: InferenceDevice = field(  # type: ignore[assignment]
        default_factory=lambda: _env("WHISPER_DEVICE", "auto")
    )
    # Comma-separated technical hotwords fed into Whisper's initial_prompt
    hotwords: str = field(
        default_factory=lambda: _env(
            "ASR_HOTWORDS",
            "Asterisk,Twilio,WebRTC,ETA,SIP,VoIP,Google Meet,Zoom,GitHub,Jira",
        )
    )
    beam_size: int = 5
    language: str = "en"
    # Segment-level silence probability above which a transcription chunk is
    # discarded as background noise / Whisper hallucination.
    no_speech_threshold: float = field(
        default_factory=lambda: _env_float("ASR_NO_SPEECH_THRESHOLD", 0.6)
    )
    # Segments whose average token log-probability is below this value are
    # treated as low-confidence and dropped (typically repetitive hallucinations).
    low_confidence_logprob: float = field(
        default_factory=lambda: _env_float("ASR_LOW_CONFIDENCE_LOGPROB", -1.0)
    )


@dataclass
class TranslationConfig:
    # NLLB-200 — facebook/nllb-200-distilled-600M (~1.2 GB)
    # Best quality; handles slang, technical text, and mixed register.
    backend: TranslationBackend = field(  # type: ignore[assignment]
        default_factory=lambda: _env("TRANSLATION_BACKEND", "nllb")
    )
    device: InferenceDevice = field(  # type: ignore[assignment]
        default_factory=lambda: _env("TRANSLATION_DEVICE", "auto")
    )
    tone: HindiTone = field(  # type: ignore[assignment]
        default_factory=lambda: _env("HINDI_TONE", "auto")
    )

    NLLB_MODEL_ID: str = "facebook/nllb-200-distilled-600M"

    @property
    def model_id(self) -> str:
        return self.NLLB_MODEL_ID


@dataclass
class TTSConfig:
    # Piper voice name — files downloaded automatically on first run.
    # Available Hindi voices: hi_IN-rohan-medium | hi_IN-pratham-medium | hi_IN-priyamvada-medium
    voice: str = field(
        default_factory=lambda: _env("PIPER_VOICE", "hi_IN-rohan-medium")
    )
    # Directory where *.onnx and *.onnx.json voice files are stored
    voices_dir: Path = field(
        default_factory=lambda: (
            Path(_env("PIPER_VOICES_DIR", "")) if _env("PIPER_VOICES_DIR", "")
            else Path.home() / ".local" / "share" / "gnani" / "voices"
        )
    )
    # Speaking rate multiplier (1.0 = normal speed)
    speed: float = field(default_factory=lambda: _env_float("PIPER_SPEED", 1.0))

    def __post_init__(self) -> None:
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    @property
    def onnx_path(self) -> Path:
        return self.voices_dir / f"{self.voice}.onnx"

    @property
    def config_path(self) -> Path:
        return self.voices_dir / f"{self.voice}.onnx.json"


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    show_latency: bool = True
    text_only: bool = False


config = PipelineConfig()
