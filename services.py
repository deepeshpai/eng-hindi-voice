"""
Application-level service layer.

Two classes live here:

ModelRegistry
    Owns every ML model handle.  Loaded exactly once at server startup
    (via FastAPI lifespan), then injected into route handlers through
    FastAPI's Depends() mechanism — no module-level mutable globals.

TranslationService
    Orchestrates the full text→audio pipeline:
      SmartBuffer clean → Translator → TTSEngine → WAV encode
    Route handlers call this service; they know nothing about ML internals.
"""
from __future__ import annotations

import asyncio
import base64
import io
import math
import time
import wave
from typing import Optional

import numpy as np

from config import PipelineConfig, TranslationConfig
from models import TranslateResponse
from utils import get_logger

log = get_logger(__name__)


# ── Audio helper ──────────────────────────────────────────────────────────────

def numpy_to_wav_b64(audio: np.ndarray, sample_rate: int) -> str:
    """Encode a float32 numpy audio array as a base-64 WAV string."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def decode_audio_to_pcm(path: str, target_sr: int = 16000) -> bytes:
    """
    Decode any audio container (webm/opus, mp3, ogg, wav, m4a…) to
    mono int16 PCM at *target_sr* Hz.

    Uses PyAV as the primary decoder (no system ffmpeg binary required).
    Falls back to stdlib `wave` for plain WAV files only.
    """
    try:
        import av  # type: ignore

        container = av.open(path)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
        chunks: list[bytes] = []
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                chunks.append(bytes(rf.planes[0]))
        for rf in resampler.resample(None):   # flush resampler
            chunks.append(bytes(rf.planes[0]))
        container.close()
        pcm = b"".join(chunks)
        if pcm:
            return pcm
    except Exception:
        pass

    # Fallback: stdlib wave (plain WAV only)
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        src_sr = wf.getframerate()
        src_ch = wf.getnchannels()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if src_ch > 1:
        samples = samples.reshape(-1, src_ch)[:, 0]
    if src_sr != target_sr:
        ratio = target_sr / src_sr
        n_out = math.ceil(len(samples) * ratio)
        xs = np.linspace(0, len(samples) - 1, n_out)
        samples = np.interp(xs, np.arange(len(samples)), samples)
    return samples.astype(np.int16).tobytes()


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Holds every loaded ML model handle.

    Usage (FastAPI lifespan)
    ------------------------
    registry = ModelRegistry()
    await asyncio.get_event_loop().run_in_executor(None, registry.load)
    # store on app.state; inject via Depends
    """

    def __init__(self) -> None:
        self.ready: bool = False
        self._asr: Optional[object] = None
        self._tts: Optional[object] = None
        self._translators: dict[str, object] = {}

    # ── Model accessors (raises if not loaded) ────────────────────────────────

    @property
    def asr(self):
        if self._asr is None:
            raise RuntimeError("ASR model not loaded")
        return self._asr

    @property
    def tts(self):
        if self._tts is None:
            raise RuntimeError("TTS model not loaded")
        return self._tts

    def translator(self, backend: str) -> object:
        t = self._translators.get(backend) or next(iter(self._translators.values()), None)
        if t is None:
            raise RuntimeError("No translation model loaded")
        return t

    @property
    def available_backends(self) -> list[str]:
        return list(self._translators.keys())

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Blocking model load — run in a thread-pool executor at startup."""
        from pipeline.asr_engine import ASREngine
        from pipeline.tts_engine import TTSEngine
        from pipeline.translator import Translator

        cfg = PipelineConfig()
        self._load_asr(cfg, ASREngine)
        self._load_tts(cfg, TTSEngine)
        self._load_translators(cfg, Translator)
        self.ready = True
        log.info("ModelRegistry: all models ready.")

    def _load_asr(self, cfg: PipelineConfig, ASREngine) -> None:
        self._asr = ASREngine(cfg.asr, cfg.audio)
        self._asr.initialize()  # type: ignore[union-attr]
        log.info("ASR loaded: faster-whisper %s", cfg.asr.model)

        # Register hotwords as custom protected entities so domain-specific
        # terms (Asterisk, Twilio, SIP, Google Meet…) are always shielded from
        # translation regardless of whether spaCy detects them as proper nouns.
        from pipeline.entity_guard import register_custom_entities
        hotwords = [w.strip() for w in cfg.asr.hotwords.split(",") if w.strip()]
        if hotwords:
            register_custom_entities(hotwords)
            log.info("Entity guard: registered %d custom entities from hotwords.", len(hotwords))

    def _load_tts(self, cfg: PipelineConfig, TTSEngine) -> None:
        self._tts = TTSEngine(cfg.tts)
        self._tts.initialize()  # type: ignore[union-attr]
        log.info("TTS loaded: piper %s", cfg.tts.voice)

    def _load_translators(self, cfg: PipelineConfig, Translator) -> None:
        for backend in ("opus", "nllb"):
            try:
                t_cfg = TranslationConfig()
                t_cfg.backend = backend  # type: ignore[assignment]
                t = Translator(t_cfg)
                t.initialize()
                self._translators[backend] = t
                log.info("Translator loaded: %s", backend)
            except Exception as exc:
                log.warning("Skipping backend '%s': %s", backend, exc)


# ── TranslationService ────────────────────────────────────────────────────────

class TranslationService:
    """
    Orchestrates the text→audio pipeline for one translation request.

    Intentionally thin: it knows *what* to call in which order, but the
    heavy lifting (ML inference) is delegated to the individual pipeline
    classes that the ModelRegistry holds.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    async def translate(
        self,
        english: str,
        backend: str = "nllb",
        tone: str = "auto",
    ) -> TranslateResponse:
        from pipeline.smart_buffer import detect_tone, remove_fillers

        t_wall = time.perf_counter()

        # ── SmartBuffer ───────────────────────────────────────────────────────
        cleaned = remove_fillers(english)
        if not cleaned.strip():
            raise ValueError("Text is empty after filler removal")

        effective_tone = detect_tone(cleaned) if tone == "auto" else tone

        # ── Translation ───────────────────────────────────────────────────────
        translator = self._registry.translator(backend)
        t0 = time.perf_counter()
        hindi: str = await translator.translate(cleaned, effective_tone)  # type: ignore[union-attr]
        trans_ms = round((time.perf_counter() - t0) * 1000)

        # ── TTS ───────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        audio_np, sr = await self._registry.tts.synthesise(hindi)  # type: ignore[union-attr]
        tts_ms = round((time.perf_counter() - t0) * 1000)

        return TranslateResponse(
            english=english,
            cleaned=cleaned,
            hindi=hindi,
            tone=effective_tone,
            audio_b64=numpy_to_wav_b64(audio_np, sr),
            sample_rate=sr,
            backend=backend,
            trans_ms=trans_ms,
            tts_ms=tts_ms,
            total_ms=round((time.perf_counter() - t_wall) * 1000),
        )

    async def transcribe(self, audio_path: str) -> str:
        """Decode an audio file and run ASR. Returns the English transcript."""
        loop = asyncio.get_event_loop()
        try:
            pcm = await loop.run_in_executor(None, decode_audio_to_pcm, audio_path)
        except Exception as exc:
            raise ValueError(f"Cannot decode audio: {exc}") from exc
        if not pcm:
            raise ValueError("Audio file is empty")
        return await self._registry.asr.transcribe(pcm)  # type: ignore[union-attr]
