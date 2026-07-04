"""
Stage 6 — Audio Playback with Interruption Support (sounddevice, no pygame).

Plays float32 NumPy audio arrays through the system speakers using sounddevice,
which is already a dependency for microphone capture.  This replaces the
previous pygame-based player and avoids an additional binary dependency.

Interruption contract
---------------------
`speech_started_event` is set by AudioCapture as soon as a new speech segment
begins.  The playback loop polls this event every 20 ms and calls
`sounddevice.stop()` when it fires, then drains any queued audio that is now
stale.  Outdated translations are silently discarded.

Thread model
------------
sounddevice.play() / sounddevice.wait() are blocking calls.  We run them on a
dedicated background thread so the asyncio event loop remains unblocked.
The async side enqueues (audio_np, sample_rate) tuples via a thread-safe queue;
the playback thread dequeues and plays them sequentially.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd

from utils import get_logger

log = get_logger(__name__)

AudioItem = Tuple[np.ndarray, int]   # (float32 samples, sample_rate)
_STOP = object()                      # sentinel — shuts down the playback thread


class AudioPlayer:
    """
    Plays audio arrays sequentially, interrupts on new speech.

    Parameters
    ----------
    speech_started_event : asyncio.Event
        Set by AudioCapture when the speaker starts a new utterance.
    loop : asyncio.AbstractEventLoop
        The running asyncio loop (used to clear the event from a thread).
    """

    def __init__(
        self,
        speech_started_event: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
        on_interrupt: Optional[Callable[[], None]] = None,
    ) -> None:
        self._speech_started = speech_started_event
        self._loop = loop
        # Called (on the asyncio loop) when playback is cut short by new speech.
        # Used by the pipeline to reset the SmartBuffer and discard in-flight work.
        self._on_interrupt = on_interrupt
        self._audio_queue: queue.Queue[object] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def initialize(self) -> None:
        """Verify sounddevice can open an output stream."""
        try:
            devices = sd.query_devices()
            default_out = sd.default.device[1]
            if default_out < 0:
                log.warning("No default audio output device found.")
            else:
                name = devices[default_out]["name"] if devices else "unknown"
                log.info("Audio output: %s", name)
        except Exception as exc:
            log.warning("sounddevice output check failed: %s", exc)

    # ── Playback thread ───────────────────────────────────────────────────────

    def _playback_loop(self) -> None:
        while self._running:
            # Block until an item is available (short timeout for shutdown check)
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            audio_np, sample_rate = item  # type: ignore

            # Ensure audio is 1-D float32
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            audio_np = audio_np.astype(np.float32)

            try:
                sd.play(audio_np, samplerate=sample_rate)

                # Poll: wait for playback to finish OR interruption
                while True:
                    # sd.get_status() is non-blocking; remaining frames check
                    if not sd.get_stream().active:
                        break
                    if self._speech_started.is_set():
                        sd.stop()
                        self._drain_queue()
                        self._loop.call_soon_threadsafe(self._speech_started.clear)
                        if self._on_interrupt is not None:
                            # Schedule the interrupt handler on the asyncio loop
                            # so it can safely touch asyncio primitives.
                            self._loop.call_soon_threadsafe(self._on_interrupt)
                        log.info("Playback interrupted — new speech detected.")
                        break
                    time.sleep(0.02)

            except Exception as exc:
                log.error("AudioPlayer playback error: %s", exc, exc_info=True)
                sd.stop()

    def _drain_queue(self) -> None:
        """Discard all stale audio items (called after an interruption)."""
        n = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        if n:
            log.debug("Drained %d stale audio item(s).", n)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._playback_loop, name="audio-player", daemon=True
        )
        self._thread.start()

    def enqueue(self, audio: AudioItem) -> None:
        """Queue (audio_np, sample_rate) for playback.  Non-blocking."""
        self._audio_queue.put(audio)

    def stop(self) -> None:
        """Shut down the playback thread gracefully."""
        self._running = False
        self._audio_queue.put(_STOP)
        try:
            sd.stop()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("AudioPlayer stopped.")
