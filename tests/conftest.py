"""
Shared fixtures and pytest configuration for the Gnani test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable from any test
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Async mode ────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (requires real model loading — skipped by default).",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test against real API endpoints.",
    )


# ── Reusable audio fixtures ───────────────────────────────────────────────────

@pytest.fixture
def silence_audio() -> np.ndarray:
    """1-second float32 silence array at 22050 Hz."""
    return np.zeros(22050, dtype=np.float32)


@pytest.fixture
def tone_audio() -> np.ndarray:
    """0.5-second 440 Hz sine wave (float32, 22050 Hz) for WAV round-trip tests."""
    t = np.linspace(0, 0.5, int(22050 * 0.5), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def sample_rate() -> int:
    return 22050


# ── Mock translator ───────────────────────────────────────────────────────────

class _FakeTranslator:
    """Minimal stand-in that returns a predictable Hindi string without loading models."""

    def initialize(self) -> None:
        pass

    async def translate(self, english: str, tone: str = "auto") -> str:
        # Echo-style fake: prepend a Hindi marker so tests can verify it ran
        return f"[हिन्दी] {english}"

    def close(self) -> None:
        pass


@pytest.fixture
def fake_translator() -> _FakeTranslator:
    return _FakeTranslator()


# ── Mock TTS engine ───────────────────────────────────────────────────────────

class _FakeTTS:
    """Returns a short sine-wave array instead of running piper."""

    def initialize(self) -> None:
        pass

    async def synthesise(self, hindi_text: str):
        t = np.linspace(0, 0.2, int(22050 * 0.2), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return audio, 22050

    def close(self) -> None:
        pass


@pytest.fixture
def fake_tts() -> _FakeTTS:
    return _FakeTTS()


# ── Mock ASR engine ───────────────────────────────────────────────────────────

class _FakeASR:
    def initialize(self) -> None:
        pass

    async def transcribe(self, pcm_bytes: bytes) -> str:
        return "Can we move the meeting to 5 PM tomorrow?"

    def close(self) -> None:
        pass


@pytest.fixture
def fake_asr() -> _FakeASR:
    return _FakeASR()
