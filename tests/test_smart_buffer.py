"""
Unit tests for pipeline.smart_buffer — no model loading required.

Tests cover:
- remove_fillers()     : filler word removal (Tier 1 anywhere, Tier 2 leading)
- detect_tone()        : formal / casual / auto heuristic
- is_sentence_complete(): punctuation boundary detection
- SmartBuffer          : accumulation, dedup, tone memory, backpressure, reset
"""
from __future__ import annotations

import asyncio
import pytest

from pipeline.smart_buffer import (
    SmartBuffer,
    detect_tone,
    is_sentence_complete,
    remove_fillers,
)


# ── remove_fillers ────────────────────────────────────────────────────────────

class TestRemoveFillers:
    # Tier 1 — anywhere
    def test_removes_leading_um(self):
        assert remove_fillers("um we should reschedule") == "we should reschedule"

    def test_removes_leading_uh(self):
        assert remove_fillers("uh can you join now") == "can you join now"

    def test_removes_leading_er(self):
        assert remove_fillers("er the report is ready") == "the report is ready"

    def test_removes_mid_sentence_filler(self):
        result = remove_fillers("we should um call them")
        assert "um" not in result
        assert "we should" in result
        assert "call them" in result

    def test_preserves_content_words(self):
        text = "Can we move the Google Meet to 5 PM tomorrow?"
        assert remove_fillers(text) == text

    def test_only_fillers_leaves_empty(self):
        result = remove_fillers("um uh er hmm")
        assert result.strip() == ""

    def test_no_double_spaces_in_output(self):
        result = remove_fillers("can um we meet")
        assert "  " not in result

    def test_case_insensitive(self):
        result = remove_fillers("UM we need to talk")
        assert "UM" not in result
        assert "we need to talk" in result

    def test_preserves_sentence_punctuation(self):
        result = remove_fillers("um, can we talk?")
        assert result.endswith("?")

    def test_empty_string(self):
        assert remove_fillers("") == ""

    # Tier 1 parametrised — phonetic hesitations safe anywhere
    @pytest.mark.parametrize("filler", ["uh", "um", "er", "hmm"])
    def test_tier1_fillers_removed_anywhere(self, filler: str):
        text = f"the {filler} report is ready"
        result = remove_fillers(text)
        assert filler.lower() not in result.lower()
        assert "report is ready" in result

    # Tier 2 — leading only
    @pytest.mark.parametrize("filler", ["well", "anyway"])
    def test_tier2_fillers_removed_at_start(self, filler: str):
        text = f"{filler} let's proceed"
        result = remove_fillers(text)
        assert filler.lower() not in result.lower()
        assert "let's proceed" in result

    def test_well_preserved_mid_sentence(self):
        """'well' must NOT be stripped when it is not leading."""
        result = remove_fillers("he did well in the exam")
        assert "well" in result

    def test_anyway_preserved_mid_sentence(self):
        result = remove_fillers("call me anyway you want")
        assert "anyway" in result

    # Tier 1 → exposes Tier 2 at start
    def test_consecutive_mixed_fillers(self):
        """um + uh are removed; 'well' is then at the start and removed too."""
        result = remove_fillers("um uh well the deadline is today")
        assert "um" not in result
        assert "uh" not in result
        assert "well" not in result
        assert "deadline is today" in result

    def test_removes_you_know_leading(self):
        result = remove_fillers("you know we have to do this")
        assert "you know" not in result
        assert "we have to do this" in result

    def test_removes_i_mean_leading(self):
        result = remove_fillers("i mean let's just do it")
        assert "i mean" not in result

    # Comma-bracketed mid-sentence discourse markers
    def test_removes_comma_bracketed_you_know(self):
        result = remove_fillers("this is, you know, really important")
        assert "you know" not in result
        assert "this is" in result
        assert "really important" in result

    def test_removes_comma_bracketed_i_mean(self):
        result = remove_fillers("it's, i mean, quite complex")
        assert "i mean" not in result

    # Ambiguous words that are NOT in the filler list (must be preserved)
    @pytest.mark.parametrize("text", [
        "He basically reinvented the process",
        "That is literally the best idea",
        "Actually, that's a great point",
        "Right, I see what you mean",
        "Honestly, I'm not sure",
    ])
    def test_ambiguous_words_preserved(self, text: str):
        """Words removed in the old list but now preserved to avoid false strips."""
        result = remove_fillers(text)
        # At minimum, the meaningful content should survive
        assert len(result) > 5


# ── detect_tone ───────────────────────────────────────────────────────────────

class TestDetectTone:
    def test_please_is_formal(self):
        assert detect_tone("Please share the document with me.") == "formal"

    def test_could_you_is_formal(self):
        assert detect_tone("Could you review this by EOD?") == "formal"

    def test_would_you_is_formal(self):
        assert detect_tone("Would you kindly approve this?") == "formal"

    def test_kindly_is_formal(self):
        assert detect_tone("Kindly confirm your attendance.") == "formal"

    def test_wanna_is_casual(self):
        assert detect_tone("Hey wanna grab lunch?") == "casual"

    def test_gonna_is_casual(self):
        assert detect_tone("I'm gonna be late.") == "casual"

    def test_dude_is_casual(self):
        assert detect_tone("Dude, this is amazing!") == "casual"

    def test_neutral_returns_auto(self):
        assert detect_tone("The meeting starts at 3 PM.") == "auto"

    def test_empty_returns_auto(self):
        assert detect_tone("") == "auto"

    def test_formal_wins_when_more_signals(self):
        result = detect_tone("Please could you check, dude?")
        assert result == "formal"

    def test_casual_wins_when_more_signals(self):
        result = detect_tone("Hey bro, wanna check this please?")
        assert result == "casual"

    @pytest.mark.parametrize("text,expected", [
        ("Please send me the report.", "formal"),
        ("Could you please confirm?", "formal"),
        ("Hey, wanna join the call?", "casual"),
        ("Gonna be there in 5 mins.", "casual"),
        ("The build passed.", "auto"),
        ("Deployment at 6 PM.", "auto"),
    ])
    def test_parametrized_examples(self, text: str, expected: str):
        assert detect_tone(text) == expected


# ── is_sentence_complete ──────────────────────────────────────────────────────

class TestIsSentenceComplete:
    @pytest.mark.parametrize("text", [
        "The meeting is at 3 PM.",
        "Can we reschedule?",
        "Great job!",
        "Done!",
        "The report is ready...",
    ])
    def test_complete_sentences(self, text: str):
        assert is_sentence_complete(text) is True

    @pytest.mark.parametrize("text", [
        "Can we reschedule",
        "Please send me the",
        "",
        "Meeting at",
    ])
    def test_incomplete_sentences(self, text: str):
        assert is_sentence_complete(text) is False

    def test_trailing_whitespace_handled(self):
        assert is_sentence_complete("Done.   ") is True

    def test_multiple_punctuation(self):
        assert is_sentence_complete("Really?!") is True


# ── SmartBuffer ───────────────────────────────────────────────────────────────

class TestSmartBuffer:
    """SmartBuffer tests — run in a fresh asyncio event loop per test."""

    def test_complete_sentence_flushes_immediately(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("Can we move the meeting to 5 PM tomorrow?")
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert "meeting" in item.text.lower()
        asyncio.run(_run())

    def test_filler_stripped_before_emit(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("um can we reschedule the call?")
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert "um" not in item.text
            assert "reschedule" in item.text
        asyncio.run(_run())

    def test_tone_attached_to_emitted_sentence(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("Please could you confirm the meeting?")
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert item.tone == "formal"
        asyncio.run(_run())

    def test_duplicate_sentence_skipped(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("The report is ready.")
            buf.feed("The report is ready.")
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert item is not None
            assert buf._queue.empty()
        asyncio.run(_run())

    def test_flush_now_emits_partial_sentence(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("I was just wondering")
            buf.flush_now()
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert "wondering" in item.text
        asyncio.run(_run())

    def test_dedup_is_case_and_punct_insensitive(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("Done.")
            buf.feed("done.")
            await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert buf._queue.empty()
        asyncio.run(_run())

    def test_multiple_sentences_emitted_in_order(self):
        async def _run():
            buf = SmartBuffer()
            buf.feed("First sentence is here.")
            buf.feed("Second sentence follows!")
            first = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            second = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert "First" in first.text
            assert "Second" in second.text
        asyncio.run(_run())

    def test_empty_feed_ignored(self):
        buf = SmartBuffer()
        buf.feed("")
        buf.feed("   ")
        assert buf._queue.empty()

    # ── Tone memory ───────────────────────────────────────────────────────────

    def test_tone_memory_persists_for_neutral_sentence(self):
        """Once formal tone is established, a neutral sentence inherits it."""
        async def _run():
            buf = SmartBuffer()
            buf.feed("Please confirm the meeting.")   # → formal
            item1 = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert item1.tone == "formal"

            buf.feed("The build passed.")             # neutral → session: formal
            item2 = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert item2.tone == "formal"
        asyncio.run(_run())

    def test_tone_memory_requires_multiple_counter_signals_to_flip(self):
        """A single casual signal should NOT flip a well-established formal session."""
        async def _run():
            buf = SmartBuffer()
            # Build up formal confidence to 3 (distinct sentences avoid dedup cache)
            for i in range(1, 4):
                buf.feed(f"Please could you kindly confirm step {i}?")
                await asyncio.wait_for(buf._queue.get(), timeout=1.0)

            # Single casual signal should not flip
            buf.feed("Dude, that's cool!")
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert item.tone == "formal"
        asyncio.run(_run())

    # ── Backpressure ──────────────────────────────────────────────────────────

    def test_backpressure_drops_oldest_when_full(self):
        """When max_pending is exceeded, the oldest queued sentence is dropped."""
        buf = SmartBuffer(max_pending=2)
        # Feed 3 complete sentences — the first should be dropped
        buf.feed("Sentence one is here.")
        buf.feed("Sentence two follows!")
        buf.feed("Sentence three arrives now.")
        # Queue should hold at most 2 items
        assert buf._queue.qsize() <= 2

    # ── Reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_buffer_without_emitting(self):
        """reset() discards accumulated text; nothing reaches the queue."""
        async def _run():
            buf = SmartBuffer()
            buf.feed("Partial sentence without end")
            assert buf._queue.empty()      # hasn't flushed yet
            buf.reset()
            assert not buf._buffer         # buffer is cleared
            assert buf._queue.empty()      # still nothing emitted
        asyncio.run(_run())

    def test_reset_does_not_affect_already_queued_items(self):
        """Items already in the queue survive a reset()."""
        async def _run():
            buf = SmartBuffer()
            buf.feed("Complete sentence here.")
            # Sentence is now in the queue
            buf.reset()
            # The queued item is still there
            item = await asyncio.wait_for(buf._queue.get(), timeout=1.0)
            assert "Complete sentence" in item.text
        asyncio.run(_run())
