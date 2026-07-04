"""
Stage 1 — Audio Capture with Voice Activity Detection (VAD).

Reads PCM audio from the microphone using sounddevice, segments it into
speech/non-speech regions using WebRTC VAD, and enqueues complete speech
segments (as raw bytes) for the ASR stage.

Design notes
------------
- WebRTC VAD expects 16-bit mono PCM at 8/16/32 kHz with 10/20/30 ms frames.
- We use 30 ms frames at 16 kHz → 480 samples per frame.
- A "speech segment" is a contiguous run of speech frames padded with a small
  amount of audio before and after to avoid clipping the first/last phoneme.
- Silence longer than `VADConfig.silence_duration_ms` ends the current segment.
- Segments shorter than `VADConfig.min_speech_duration_ms` are discarded
  (button clicks, background pops, etc.).

Interruption contract
---------------------
When a NEW speech segment starts while the TTS stage is still playing,
`speech_started_event` is set.  The AudioPlayer watches this event and stops
the current utterance immediately.
"""
from __future__ import annotations

import asyncio
import collections
import threading
from typing import AsyncIterator, Deque

import numpy as np
import sounddevice as sd
import webrtcvad

from config import AudioConfig, VADConfig
from utils import get_logger

log = get_logger(__name__)

# Number of silent frames required before closing a speech segment
_SILENCE_FRAMES_REQUIRED = lambda vad_cfg, audio_cfg: max(
    1,
    int(vad_cfg.silence_duration_ms / audio_cfg.frame_duration_ms),
)

# Pre-roll frames to keep before speech detection (prevents clipping leading phoneme)
_PRE_ROLL_FRAMES = lambda vad_cfg, audio_cfg: max(
    1,
    int(vad_cfg.pre_speech_pad_ms / audio_cfg.frame_duration_ms),
)

# Post-roll frames appended after speech ends
_POST_ROLL_FRAMES = lambda vad_cfg, audio_cfg: max(
    1,
    int(vad_cfg.post_speech_pad_ms / audio_cfg.frame_duration_ms),
)


class AudioCapture:
    """
    Streams microphone audio, applies WebRTC VAD, and yields speech segments.

    Parameters
    ----------
    audio_cfg : AudioConfig
    vad_cfg   : VADConfig
    loop      : asyncio event loop running in the main thread
    """

    def __init__(
        self,
        audio_cfg: AudioConfig,
        vad_cfg: VADConfig,
        loop: asyncio.AbstractEventLoop,
        speech_started_event: asyncio.Event,
    ) -> None:
        self._audio = audio_cfg
        self._vad_cfg = vad_cfg
        self._loop = loop
        self._speech_started_event = speech_started_event
        self._vad = webrtcvad.Vad(vad_cfg.aggressiveness)

        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stop_flag = threading.Event()

        # Pre-roll ring buffer (keeps the last N frames before speech detected)
        self._pre_roll: Deque[bytes] = collections.deque(
            maxlen=_PRE_ROLL_FRAMES(vad_cfg, audio_cfg)
        )
        self._speech_buffer: list[bytes] = []
        self._silent_frame_count: int = 0
        self._in_speech: bool = False

        # Minimum frames to qualify as real speech
        self._min_speech_frames = max(
            1,
            int(vad_cfg.min_speech_duration_ms / audio_cfg.frame_duration_ms),
        )
        self._silence_threshold = _SILENCE_FRAMES_REQUIRED(vad_cfg, audio_cfg)
        self._post_roll_needed = _POST_ROLL_FRAMES(vad_cfg, audio_cfg)

        # Collects post-roll frames after speech ends
        self._post_roll_remaining: int = 0
        self._pending_segment: list[bytes] = []

    # ── Internal callback (runs in sounddevice's thread) ─────────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.debug("sounddevice status: %s", status)

        # Convert float32 → int16 PCM bytes expected by WebRTC VAD
        pcm_int16 = (indata[:, 0] * 32767).astype(np.int16)
        frame_bytes = pcm_int16.tobytes()

        # WebRTC VAD may raise on bad input; guard against it
        try:
            is_speech = self._vad.is_speech(frame_bytes, self._audio.sample_rate)
        except Exception:
            is_speech = False

        self._process_frame(frame_bytes, is_speech)

    def _process_frame(self, frame: bytes, is_speech: bool) -> None:
        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                # Transition into speech — flush pre-roll into the buffer
                self._in_speech = True
                self._speech_buffer = list(self._pre_roll)
                self._silent_frame_count = 0
                # Signal any ongoing TTS playback to stop
                self._loop.call_soon_threadsafe(self._speech_started_event.set)
        else:
            if is_speech:
                self._speech_buffer.append(frame)
                self._silent_frame_count = 0
            else:
                self._silent_frame_count += 1
                self._speech_buffer.append(frame)  # keep trailing silence for post-roll
                if self._silent_frame_count >= self._silence_threshold:
                    self._flush_segment()

    def _flush_segment(self) -> None:
        self._in_speech = False
        total_frames = len(self._speech_buffer)
        if total_frames < self._min_speech_frames:
            log.debug("Discarding short segment (%d frames)", total_frames)
            self._speech_buffer.clear()
            return

        audio_bytes = b"".join(self._speech_buffer)
        self._speech_buffer.clear()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, audio_bytes)
        log.debug(
            "Speech segment enqueued — %.2f s",
            len(audio_bytes) / 2 / self._audio.sample_rate,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone input stream (non-blocking)."""
        device = None if self._audio.input_device == -1 else self._audio.input_device
        self._stream = sd.InputStream(
            samplerate=self._audio.sample_rate,
            channels=self._audio.channels,
            dtype="float32",
            blocksize=self._audio.frame_size,
            device=device,
            callback=self._audio_callback,
        )
        self._stream.start()
        log.info(
            "Microphone open — %d Hz, frame=%d ms, VAD aggressiveness=%d",
            self._audio.sample_rate,
            self._audio.frame_duration_ms,
            self._vad_cfg.aggressiveness,
        )

    def stop(self) -> None:
        """Close the microphone stream."""
        self._stream.stop()
        self._stream.close()
        log.info("Microphone closed.")

    async def speech_segments(self) -> AsyncIterator[bytes]:
        """Async generator yielding raw PCM speech segments (int16 bytes)."""
        while True:
            segment = await self._queue.get()
            yield segment
