"""
Real-time English → Hindi voice translation pipeline.

Stage order:
  AudioCapture → ASREngine → SmartBuffer → Translator → TTSEngine → AudioPlayer
"""
from .audio_capture import AudioCapture
from .asr_engine import ASREngine
from .smart_buffer import SmartBuffer
from .translator import Translator
from .tts_engine import TTSEngine
from .audio_player import AudioPlayer
from .pipeline import VoiceTranslationPipeline

__all__ = [
    "AudioCapture",
    "ASREngine",
    "SmartBuffer",
    "Translator",
    "TTSEngine",
    "AudioPlayer",
    "VoiceTranslationPipeline",
]
