"""
Stage 3 — Smart Buffer.

Sits between ASR output and the translator.  Responsibilities:

1. Filler word removal  (conservative — only clearly-discourse words)
   Strip "uh", "um", "you know" etc. before translation.  We split fillers
   into two tiers:
     • ANYWHERE — purely phonetic hesitations (uh/um/er/hmm): always removed.
     • LEADING   — discourse markers (well/anyway/i mean): only removed when
                   they appear at the START of the utterance so we don't
                   accidentally strip "he did well" or "call me anyway you want".
   Additionally, comma-bracketed occurrences (", you know,") are collapsed.

2. Sentence boundary detection
   Accumulates chunks and flushes when a sentence-ending boundary is detected
   OR a timeout expires, keeping the pipeline moving even when a speaker trails
   off mid-sentence.

3. Tone detection + session memory
   A lightweight heuristic classifies each sentence as 'formal', 'casual', or
   'auto'.  The SmartBuffer tracks session-level tone confidence so tone doesn't
   flip after a single ambiguous sentence: once a tone is established it takes
   multiple counter-signals to change it.

4. Backpressure
   If the downstream queue depth exceeds `max_pending`, the oldest pending
   sentence is discarded so the pipeline doesn't fall further behind a fast
   speaker.

5. Deduplication
   Avoids re-translating the same sentence if Whisper emits it twice (can
   happen on long silences when VAD doesn't trigger).

6. Reset on interruption
   `reset()` clears the accumulation buffer WITHOUT emitting to the queue.
   Called when TTS playback is interrupted by new speech so stale partial
   sentences are not eventually sent to the translator.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from utils import get_logger

log = get_logger(__name__)

# ── Filler word patterns ──────────────────────────────────────────────────────
#
# Tier 1 — ANYWHERE: purely phonetic hesitations that carry no semantic load.
# Safe to remove regardless of position in the utterance.
_FILLERS_ANYWHERE: frozenset[str] = frozenset({
    "uh", "uhh", "uhhh",
    "um", "umm", "ummm",
    "er", "err",
    "ah", "ahh",
    "hmm", "hm",
    "uh-huh", "mm-hmm", "mhm",
})

# Tier 2 — LEADING: discourse markers that are fillers only at the START of an
# utterance.  Mid-sentence "He did well" or "call me anyway you want" must be
# preserved, so these are only stripped when they are the first word(s).
_FILLERS_LEADING: frozenset[str] = frozenset({
    "you know",
    "i mean",
    "okay so",
    "so yeah",
    "well",
    "anyway",
})

_FILLER_ANYWHERE_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(_FILLERS_ANYWHERE, key=len, reverse=True)) + r")\b,?\s*",
    flags=re.IGNORECASE,
)

# Matches a leading-filler that optionally follows optional trailing punctuation
_FILLER_LEADING_RE = re.compile(
    r"^(" + "|".join(re.escape(f) for f in sorted(_FILLERS_LEADING, key=len, reverse=True)) + r")[,.]?\s*",
    flags=re.IGNORECASE,
)

# Comma-bracketed mid-sentence discourse markers:  ", you know," → ","
_FILLER_COMMA_RE = re.compile(
    r",\s*(?:you know|i mean|you see)\s*,",
    flags=re.IGNORECASE,
)

# ── Tone detection heuristics ─────────────────────────────────────────────────

_FORMAL_SIGNALS: frozenset[str] = frozenset(
    {"please", "could you", "would you", "kindly", "may i", "i request", "respectfully"}
)
_CASUAL_SIGNALS: frozenset[str] = frozenset(
    {"wanna", "gonna", "gotta", "hey", "dude", "bro", "guys", "ain't", "y'all", "lemme"}
)

# ── Sentence boundary ─────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"[.!?]+\s*$")

# How long (seconds) to wait before flushing an incomplete sentence
_FLUSH_TIMEOUT_S = 2.5

# How many consecutive counter-tone signals are needed to flip the session tone.
_TONE_FLIP_THRESHOLD = 2


@dataclass
class BufferedSentence:
    text: str
    tone: str  # "formal" | "casual" | "auto"
    timestamp: float = field(default_factory=time.time)


def remove_fillers(text: str) -> str:
    """
    Remove filler words / phrases from an English transcription.

    Strategy (in order):
    1. Collapse comma-bracketed discourse markers (", you know,").
    2. Remove Tier-1 (anywhere) phonetic fillers.
    3. Collapse whitespace, then strip Tier-2 (leading-only) discourse markers.
    """
    # Step 1: mid-sentence comma-bracketed forms
    text = _FILLER_COMMA_RE.sub(",", text)
    # Step 2: phonetic hesitations — safe anywhere
    text = _FILLER_ANYWHERE_RE.sub(" ", text)
    # Step 3: collapse spaces, then remove leading discourse markers
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = _FILLER_LEADING_RE.sub("", text)
    # Step 4: final cleanup
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text


def detect_tone(text: str) -> str:
    """Lightweight tone classifier → 'formal', 'casual', or 'auto'."""
    lower = text.lower()
    formal_hits = sum(1 for s in _FORMAL_SIGNALS if s in lower)
    casual_hits = sum(1 for s in _CASUAL_SIGNALS if s in lower)
    if formal_hits > casual_hits:
        return "formal"
    if casual_hits > formal_hits:
        return "casual"
    return "auto"


def is_sentence_complete(text: str) -> bool:
    """True if text ends with a sentence-ending punctuation mark."""
    return bool(_SENTENCE_END_RE.search(text.rstrip()))


class SmartBuffer:
    """
    Accumulates ASR chunks and emits complete, clean sentences.

    Parameters
    ----------
    flush_timeout_s : float
        Maximum wait (in seconds) before flushing a partial sentence.
    max_pending : int
        Maximum number of sentences allowed to queue up downstream before
        the oldest one is silently dropped (backpressure).
    """

    def __init__(
        self,
        flush_timeout_s: float = _FLUSH_TIMEOUT_S,
        max_pending: int = 3,
    ) -> None:
        self._flush_timeout = flush_timeout_s
        self._max_pending = max_pending
        self._buffer: list[str] = []
        self._last_update: float = time.monotonic()
        self._seen: set[str] = set()     # dedup cache (last 5 sentences)
        self._seen_list: list[str] = []  # ordered insertion for LRU eviction
        self._queue: asyncio.Queue[BufferedSentence] = asyncio.Queue()
        self._flush_task: Optional[asyncio.Task[None]] = None

        # Session tone memory — avoids flipping tone after a single ambiguous
        # sentence.  Flip only after _TONE_FLIP_THRESHOLD counter-signals.
        self._session_tone: str = "auto"
        self._tone_confidence: int = 0

    # ── Tone memory ───────────────────────────────────────────────────────────

    def _resolve_tone(self, detected: str) -> str:
        """
        Update session tone and return the effective tone for this sentence.

        Rules:
        • 'auto' detected  → return current session tone (or 'auto' if no
          session tone established yet).
        • Same as session  → reinforce (cap at 3 votes).
        • Different        → decrement confidence; flip only when it hits 0.
        """
        if detected == "auto":
            return self._session_tone

        if detected == self._session_tone:
            self._tone_confidence = min(3, self._tone_confidence + 1)
        else:
            self._tone_confidence -= 1
            if self._tone_confidence <= 0:
                self._session_tone = detected
                self._tone_confidence = 1

        return self._session_tone if self._session_tone != "auto" else detected

    # ── Internals ─────────────────────────────────────────────────────────────

    def _current_text(self) -> str:
        return " ".join(self._buffer).strip()

    def _should_flush(self) -> bool:
        text = self._current_text()
        if not text:
            return False
        return is_sentence_complete(text)

    def _flush(self) -> None:
        text = self._current_text()
        if not text:
            return

        cleaned = remove_fillers(text)
        cleaned = cleaned.strip(" ,;")

        if not cleaned:
            log.debug("Buffer flushed empty after filler removal.")
            self._buffer.clear()
            return

        # Dedup: skip if same sentence was emitted recently
        norm = re.sub(r"\W+", " ", cleaned.lower()).strip()
        if norm in self._seen:
            log.debug("Duplicate sentence skipped: %s", repr(cleaned[:60]))
            self._buffer.clear()
            return

        tone_detected = detect_tone(cleaned)
        effective_tone = self._resolve_tone(tone_detected)
        sentence = BufferedSentence(text=cleaned, tone=effective_tone)

        # Backpressure: drop oldest if queue is too deep
        if self._queue.qsize() >= self._max_pending:
            try:
                dropped = self._queue.get_nowait()
                log.warning(
                    "SmartBuffer backpressure: dropped stale sentence %r",
                    dropped.text[:40],
                )
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(sentence)
        log.info(
            "[SmartBuffer] → [bold]%s[/bold]  (tone=%s)",
            cleaned[:80],
            effective_tone,
        )

        # Update dedup cache (keep last 5)
        self._seen.add(norm)
        self._seen_list.append(norm)
        if len(self._seen_list) > 5:
            oldest = self._seen_list.pop(0)
            self._seen.discard(oldest)

        self._buffer.clear()

    async def _timeout_watcher(self) -> None:
        """Flush the buffer if it hasn't been updated for `_flush_timeout` seconds."""
        await asyncio.sleep(self._flush_timeout)
        if self._buffer and (time.monotonic() - self._last_update) >= self._flush_timeout:
            log.debug("Flush timeout — emitting partial sentence.")
            self._flush()

    def _schedule_timeout(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.ensure_future(self._timeout_watcher())

    # ── Public API ────────────────────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """
        Accept a new ASR transcript chunk.

        If the accumulated text forms a complete sentence it is flushed
        immediately.  Otherwise a timeout watcher is (re)scheduled.
        """
        stripped = text.strip()
        if not stripped:
            return

        self._buffer.append(stripped)
        self._last_update = time.monotonic()

        if self._should_flush():
            self._flush()
        else:
            self._schedule_timeout()

    def flush_now(self) -> None:
        """Force-flush whatever is in the buffer (e.g. on pipeline shutdown)."""
        self._flush()

    def reset(self) -> None:
        """
        Discard the accumulation buffer WITHOUT emitting to the downstream queue.

        Called when TTS playback is interrupted by new user speech so that
        partially accumulated sentences from the interrupted utterance are not
        eventually translated.
        """
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._buffer.clear()
        log.debug("SmartBuffer reset due to interruption.")

    async def sentences(self) -> AsyncIterator[BufferedSentence]:
        """Async generator yielding cleaned, complete English sentences."""
        while True:
            sentence = await self._queue.get()
            yield sentence
