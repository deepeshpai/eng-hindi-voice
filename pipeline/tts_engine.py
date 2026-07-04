"""
Stage 5 — Hindi Text-to-Speech (piper-tts, fully local).

piper-tts
---------
A fast, lightweight, local neural TTS engine by Rhasspy.
  • ONNX-based inference — runs on CPU without GPU
  • Hindi voices (from rhasspy/piper-voices on HuggingFace):
      hi_IN-rohan-medium     (male, ~63 MB)   ← default
      hi_IN-pratham-medium   (male, ~63 MB)
      hi_IN-priyamvada-medium (female, ~63 MB)
  • Voice files downloaded automatically on first run via huggingface_hub
  • Output: float32 audio chunks via a generator (piper >= 2.x API)

Output format
-------------
`synthesise()` returns a tuple  (audio_np: np.ndarray[float32], sample_rate: int)
so the AudioPlayer / server can feed it directly to sounddevice or WAV encoding
without any intermediate MP3/WAV decoding step.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import TTSConfig
from utils import get_logger

log = get_logger(__name__)

AudioArray = Tuple[np.ndarray, int]  # (float32 samples, sample_rate)


# ── Voice model download via HuggingFace Hub ─────────────────────────────────

def _voice_hf_paths(voice: str) -> tuple[str, str]:
    """
    Convert a piper voice name to its file paths inside rhasspy/piper-voices.

    "hi_IN-rohan-medium"
      → "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"
         "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"

    Convention: {lang_code}-{speaker}-{quality}
    """
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot parse piper voice name: {voice!r}\n"
            "Expected format: {{lang_code}}-{{speaker}}-{{quality}}, "
            "e.g. hi_IN-rohan-medium"
        )
    lang_code   = parts[0]                      # e.g. "hi_IN"
    speaker     = parts[1]                      # e.g. "rohan"
    quality     = parts[2]                      # e.g. "medium"
    lang_prefix = lang_code.split("_")[0]       # "hi"
    base = f"{lang_prefix}/{lang_code}/{speaker}/{quality}/{voice}"
    return f"{base}.onnx", f"{base}.onnx.json"


def _download_voice(cfg: TTSConfig) -> None:
    """Download the piper voice ONNX model and config via huggingface_hub."""
    from huggingface_hub import hf_hub_download  # type: ignore

    onnx_hf_path, json_hf_path = _voice_hf_paths(cfg.voice)

    for hf_path, local_dest in [
        (onnx_hf_path, cfg.onnx_path),
        (json_hf_path, cfg.config_path),
    ]:
        if local_dest.exists():
            log.debug("Voice file already cached: %s", local_dest.name)
            continue

        log.info("Downloading piper voice file: %s …", hf_path)
        cached = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=hf_path,
        )
        # Copy from HuggingFace cache to our voices directory
        import shutil
        shutil.copy2(cached, local_dest)
        log.info("Saved → %s  (%.1f MB)", local_dest, local_dest.stat().st_size / 1e6)


# ── TTS engine ────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Synthesises Hindi text to a float32 NumPy audio array using piper-tts.

    Parameters
    ----------
    cfg : TTSConfig
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self._cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._voice: Optional[object] = None

    def initialize(self) -> None:
        """Download voice model if needed, then load it into memory."""
        _download_voice(self._cfg)
        self._load_voice()

    def _load_voice(self) -> None:
        from piper import PiperVoice  # type: ignore

        log.info(
            "Loading piper voice '%s' from %s …",
            self._cfg.voice,
            self._cfg.voices_dir,
        )
        self._voice = PiperVoice.load(
            str(self._cfg.onnx_path),
            config_path=str(self._cfg.config_path),
        )
        log.info("piper voice loaded.")

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesise_sync(self, hindi_text: str) -> AudioArray:
        """
        Run piper inference and return (float32_samples, sample_rate).

        piper >= 2.x synthesize() is a generator that yields AudioChunk
        objects, each carrying audio_float_array (float32, [-1, 1]) and
        sample_rate.  We concatenate all chunks into one contiguous array.
        """
        assert self._voice is not None

        from piper import SynthesisConfig  # type: ignore

        syn_cfg = SynthesisConfig(
            length_scale=1.0 / self._cfg.speed,  # speed > 1 → shorter (faster)
        )

        chunks = list(self._voice.synthesize(hindi_text, syn_config=syn_cfg))  # type: ignore

        if not chunks:
            log.warning("piper returned no audio chunks for text: %r", hindi_text[:60])
            return np.zeros(0, dtype=np.float32), 22050

        sample_rate: int = chunks[0].sample_rate
        audio_float32 = np.concatenate(
            [chunk.audio_float_array for chunk in chunks], axis=0
        ).astype(np.float32)

        return audio_float32, sample_rate

    # ── Public API ────────────────────────────────────────────────────────────

    async def synthesise(self, hindi_text: str) -> AudioArray:
        """
        Asynchronously synthesise *hindi_text* to a float32 audio array.

        Returns
        -------
        (audio_np, sample_rate) — feed directly to sounddevice.play()
        """
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        audio, sr = await loop.run_in_executor(
            self._executor, self._synthesise_sync, hindi_text
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        duration_s = len(audio) / sr
        log.debug(
            "TTS [%.0f ms]: synthesised %.2f s of audio for %s",
            elapsed_ms,
            duration_s,
            repr(hindi_text[:40]),
        )
        return audio, sr

    def close(self) -> None:
        self._executor.shutdown(wait=False)
