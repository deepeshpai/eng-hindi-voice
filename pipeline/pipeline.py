"""
Async orchestrator — wires all six stages into a real-time pipeline.

Stage dataflow
--------------

  AudioCapture ─┐
                ├─→ ASREngine ─→ SmartBuffer ─→ Translator ─→ TTSEngine ─→ AudioPlayer
  speech_started┘              (sentence queue)

Each stage runs as an independent asyncio Task.  Stages communicate via the
async generators / queues defined inside each module.

Latency
-------
The critical path is:
  microphone → VAD silence → ASR → SmartBuffer flush → Translation → TTS → speaker

Typical numbers on a 2023 MacBook with base.en Whisper:
  VAD end-of-segment  ~800 ms silence threshold (configurable)
  ASR (CPU, base.en)  ~300–600 ms
  Translation (GPT)   ~400–800 ms
  TTS (gTTS)          ~200–400 ms
  ──────────────────────────────
  End-to-end          ~1.7–2.6 s  (well within conversational tolerance)

Using GPU for Whisper and streaming TTS would bring this below 1 second.
"""
from __future__ import annotations

import asyncio
import signal
import time
from typing import Optional

from config import PipelineConfig
from pipeline.audio_capture import AudioCapture
from pipeline.asr_engine import ASREngine
from pipeline.smart_buffer import SmartBuffer
from pipeline.translator import Translator
from pipeline.tts_engine import TTSEngine
from pipeline.audio_player import AudioPlayer
from utils import get_logger, LatencyTracker

log = get_logger(__name__)


class VoiceTranslationPipeline:
    """
    Real-time English → Hindi voice translation pipeline.

    Usage
    -----
    async with VoiceTranslationPipeline(config) as pipeline:
        await pipeline.run()
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._speech_started = asyncio.Event()
        # Set when TTS playback is actually interrupted (new speech while playing).
        # Checked by the translation stage to skip stale in-flight work.
        self._playback_interrupted = asyncio.Event()

        # Stage instances
        self._capture: Optional[AudioCapture] = None
        self._asr: Optional[ASREngine] = None
        self._buffer: Optional[SmartBuffer] = None
        self._translator: Optional[Translator] = None
        self._tts: Optional[TTSEngine] = None
        self._player: Optional[AudioPlayer] = None

        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "VoiceTranslationPipeline":
        await self._initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._shutdown()

    async def _initialize(self) -> None:
        self._loop = asyncio.get_event_loop()
        log.info("[bold green]Initialising pipeline…[/bold green]")

        # Build stages
        self._capture = AudioCapture(
            audio_cfg=self._cfg.audio,
            vad_cfg=self._cfg.vad,
            loop=self._loop,
            speech_started_event=self._speech_started,
        )
        self._asr = ASREngine(self._cfg.asr, self._cfg.audio)
        self._buffer = SmartBuffer()
        self._translator = Translator(self._cfg.translation)
        self._tts = TTSEngine(self._cfg.tts)
        self._player = AudioPlayer(
            speech_started_event=self._speech_started,
            loop=self._loop,
            on_interrupt=self._handle_playback_interrupt,
        )

        # Initialize blocking resources in order
        log.info(
            "Loading models — ASR: %s  |  Translation: %s  |  TTS voice: %s",
            self._cfg.asr.model,
            self._cfg.translation.backend,
            self._cfg.tts.voice,
        )
        log.info("(First run downloads models from HuggingFace — this may take a few minutes.)")
        self._asr.initialize()
        self._translator.initialize()
        self._tts.initialize()
        self._player.initialize()
        self._player.start()

        self._running = True
        log.info("[bold green]Pipeline ready. Speak into your microphone.[/bold green]")

    async def _shutdown(self) -> None:
        self._running = False
        if self._capture:
            self._capture.stop()
        if self._asr:
            self._asr.close()
        if self._translator:
            self._translator.close()
        if self._tts:
            self._tts.close()
        if self._player:
            self._player.stop()
        log.info("Pipeline shut down.")

    # ── Interrupt handler (called on asyncio loop from AudioPlayer thread) ──────

    def _handle_playback_interrupt(self) -> None:
        """
        Invoked when TTS playback is cut short by new user speech.

        Sets the `_playback_interrupted` event so the translation stage can
        detect and discard any in-flight (or queued) work that is now stale.
        Also resets the SmartBuffer to discard partial accumulated sentences
        from the interrupted utterance.
        """
        self._playback_interrupted.set()
        if self._buffer is not None:
            self._buffer.reset()
        log.info("Interrupt: SmartBuffer cleared, stale translations will be dropped.")

    # ── Stage coroutines ──────────────────────────────────────────────────────

    async def _asr_stage(self) -> None:
        """Reads speech segments from AudioCapture, transcribes, feeds SmartBuffer."""
        assert self._capture and self._asr and self._buffer
        async for pcm_bytes in self._capture.speech_segments():
            if not self._running:
                break
            english = await self._asr.transcribe(pcm_bytes)
            if english:
                log.info("[ASR] %s", english)
                self._buffer.feed(english)

    async def _translation_stage(self) -> None:
        """Reads complete sentences from SmartBuffer, translates, enqueues TTS."""
        assert self._buffer and self._translator and self._tts and self._player
        async for sentence in self._buffer.sentences():
            if not self._running:
                break

            # If playback was interrupted after this sentence was enqueued, skip it.
            # The buffer was already reset by _handle_playback_interrupt, so the
            # sentence is stale — translating it would produce audio that the user
            # has already moved past.
            if self._playback_interrupted.is_set():
                self._playback_interrupted.clear()
                log.info("Skipping stale sentence after interruption: %r", sentence.text[:40])
                continue

            tracker = LatencyTracker()
            tracker.mark("sentence_ready")

            log.info(
                "[EN] [bold]%s[/bold]  [dim](tone=%s)[/dim]",
                sentence.text,
                sentence.tone,
            )

            try:
                hindi = await self._translator.translate(sentence.text, sentence.tone)
                tracker.mark("translated")

                # Check again: user may have interrupted DURING the (potentially
                # multi-second) translation call.
                if self._playback_interrupted.is_set():
                    self._playback_interrupted.clear()
                    log.info("Discarding translation result — interrupted during MT.")
                    continue

                log.info("[HI] [bold cyan]%s[/bold cyan]", hindi)

                if not self._cfg.text_only:
                    audio_tuple = await self._tts.synthesise(hindi)
                    tracker.mark("synthesised")
                    self._player.enqueue(audio_tuple)
                    tracker.mark("enqueued")

                if self._cfg.show_latency:
                    log.info("Latency breakdown:\n%s", tracker.report())

            except Exception as exc:
                log.error("Pipeline error for %r: %s", sentence.text[:40], exc, exc_info=True)

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start the pipeline.  Runs until Ctrl-C or the process receives SIGINT.
        """
        assert self._capture

        self._capture.start()

        # Register graceful shutdown on Ctrl-C
        stop_event = asyncio.Event()

        def _handle_sigint() -> None:
            log.info("\nStopping… (Ctrl-C received)")
            stop_event.set()

        self._loop.add_signal_handler(signal.SIGINT, _handle_sigint)  # type: ignore

        asr_task = asyncio.create_task(self._asr_stage(), name="asr-stage")
        translate_task = asyncio.create_task(self._translation_stage(), name="translation-stage")

        try:
            await stop_event.wait()
        finally:
            asr_task.cancel()
            translate_task.cancel()
            await asyncio.gather(asr_task, translate_task, return_exceptions=True)
