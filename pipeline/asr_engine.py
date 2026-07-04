"""
Stage 2 — Automatic Speech Recognition (ASR).

Transcribes PCM speech segments to English text using faster-whisper — a
CTranslate2-optimised implementation of OpenAI Whisper that runs entirely
locally with no API key required.

Model sizes (downloaded from HuggingFace on first run)
------------------------------------------------------
tiny.en   ~39 MB  — fastest
base.en   ~74 MB  — recommended default
small.en  ~244 MB — better for strong accents
medium.en ~769 MB — near-SOTA quality on CPU
large-v3  ~1.5 GB — state-of-the-art, GPU recommended

Named-entity / hotword biasing
-------------------------------
Whisper accepts an `initial_prompt` that nudges the model toward specific
spellings.  We build a short prompt from `ASRConfig.hotwords` so technical
terms like "Asterisk", "Twilio", or "WebRTC" are spelled correctly even when
spoken with unusual pronunciation.

Latency
-------
faster-whisper with beam_size=5 on CPU takes ~0.5–1.5× real-time on modern
hardware for base.en.  On GPU it runs well below real-time.
The executor pattern keeps the asyncio event loop unblocked.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from config import ASRConfig, AudioConfig
from utils import get_logger

log = get_logger(__name__)


class ASREngine:
    """
    Wraps either faster-whisper (local) or the OpenAI Whisper API.

    Parameters
    ----------
    asr_cfg   : ASRConfig
    audio_cfg : AudioConfig  — needed for sample-rate metadata
    """

    def __init__(self, asr_cfg: ASRConfig, audio_cfg: AudioConfig) -> None:
        self._cfg = asr_cfg
        self._audio = audio_cfg
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._model: Optional[object] = None

        # Build a one-sentence hotword prompt so Whisper biases toward them
        self._initial_prompt = self._build_initial_prompt()

    def _build_initial_prompt(self) -> str:
        """
        A short English sentence containing the hotwords is the most reliable
        way to bias Whisper's beam search without requiring fine-tuning.
        """
        hotwords = [w.strip() for w in self._cfg.hotwords.split(",") if w.strip()]
        if not hotwords:
            return ""
        return (
            "The following technical terms may appear in the audio: "
            + ", ".join(hotwords)
            + "."
        )

    # ── Initialization (lazy) ─────────────────────────────────────────────────

    def _load_local_model(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        device = self._cfg.device
        if device == "auto":
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "int8" if device == "cpu" else "float16"
        log.info(
            "Loading faster-whisper model '%s' on %s (compute_type=%s)…",
            self._cfg.model,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self._cfg.model,
            device=device,
            compute_type=compute_type,
        )
        log.info("Whisper model ready.")

    def initialize(self) -> None:
        """
        Load faster-whisper synchronously.  Call this once before the pipeline
        starts so the first transcription has no cold-start delay.
        """
        self._load_local_model()

    # ── Transcription helpers (run in thread pool) ───────────────────────────

    def _transcribe_local(self, pcm_bytes: bytes) -> str:
        """
        Run faster-whisper on raw int16 PCM bytes.

        Quality guards applied per segment:
        1. no_speech_prob threshold  — drops segments that are likely silence or
           background noise; prevents "Thank you." / "You." hallucinations.
        2. avg_logprob threshold     — drops segments where the model is very
           uncertain (usually repetitive hallucinations like "...the the the...").
        3. Temperature fallback      — if greedy (temp=0) decoding trips a
           compression-ratio threshold, faster-whisper automatically re-tries
           with higher temperatures (0.2, 0.4) before returning results.
        """
        assert self._model is not None

        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _info = self._model.transcribe(  # type: ignore
            audio_np,
            language=self._cfg.language,
            beam_size=self._cfg.beam_size,
            initial_prompt=self._initial_prompt or None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 100},
            word_timestamps=False,
            condition_on_previous_text=False,
            # Try increasing temperatures when compression ratio is suspicious
            temperature=(0.0, 0.2, 0.4),
        )

        parts: list[str] = []
        for seg in segments:
            # Guard 1: probably silence / noise
            if seg.no_speech_prob > self._cfg.no_speech_threshold:
                log.debug(
                    "Dropped silent segment (no_speech_prob=%.2f): %r",
                    seg.no_speech_prob, seg.text[:40],
                )
                continue
            # Guard 2: very low confidence — often repetitive hallucination
            if seg.avg_logprob < self._cfg.low_confidence_logprob:
                log.debug(
                    "Dropped low-confidence segment (avg_logprob=%.2f): %r",
                    seg.avg_logprob, seg.text[:40],
                )
                continue
            parts.append(seg.text.strip())

        return " ".join(parts).strip()

    # ── Public API ────────────────────────────────────────────────────────────

    async def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Asynchronously transcribe a speech segment.

        Parameters
        ----------
        pcm_bytes : raw int16 mono PCM at `audio_cfg.sample_rate`

        Returns
        -------
        str — English transcription (empty if VAD filtered silence)
        """
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        text = await loop.run_in_executor(
            self._executor, self._transcribe_local, pcm_bytes
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_s = len(pcm_bytes) / 2 / self._audio.sample_rate
        rtf = elapsed_ms / (duration_s * 1000)
        log.debug(
            "ASR  [%.0f ms, RTF=%.2fx]: %s",
            elapsed_ms,
            rtf,
            repr(text[:80]),
        )
        return text

    def close(self) -> None:
        self._executor.shutdown(wait=False)
