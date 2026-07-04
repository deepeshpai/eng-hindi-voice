"""
Integration tests for the FastAPI server endpoints.

Fast tests (no real model loading):
  - Inject a fake ModelRegistry via FastAPI dependency_overrides.
  - Verify HTTP status codes, response shapes, and error handling.

Slow tests (real model loading, ~30 s startup):
  - Marked with @pytest.mark.slow — excluded from the default run.
  - Run with:  pytest -m slow tests/test_api.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav_bytes(duration_s: float = 0.1, sr: int = 16000) -> bytes:
    n = int(sr * duration_s)
    pcm = np.zeros(n, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sine_wav_bytes(duration_s: float = 0.2, sr: int = 16000) -> bytes:
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _fake_audio_np() -> np.ndarray:
    t = np.linspace(0, 0.2, int(22050 * 0.2), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ── Fake ModelRegistry ────────────────────────────────────────────────────────

class _FakeRegistry:
    """Minimal ModelRegistry stand-in for fast tests (no ML weights)."""

    def __init__(
        self,
        ready: bool = True,
        fake_translator=None,
        fake_tts=None,
        fake_asr=None,
    ) -> None:
        self.ready = ready
        self._asr_obj = fake_asr
        self._tts_obj = fake_tts
        self._translators: dict[str, object] = {}
        if fake_translator:
            self._translators = {"nllb": fake_translator}

    @property
    def asr(self):
        if self._asr_obj is None:
            raise RuntimeError("ASR not loaded")
        return self._asr_obj

    @property
    def tts(self):
        if self._tts_obj is None:
            raise RuntimeError("TTS not loaded")
        return self._tts_obj

    def translator(self, backend: str) -> object:
        obj = self._translators.get(backend) or next(iter(self._translators.values()), None)
        if obj is None:
            raise RuntimeError("No translation model loaded")
        return obj

    @property
    def available_backends(self) -> list[str]:
        return list(self._translators.keys())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def patched_app():
    """
    Override `get_registry` with a fake registry so no real models are loaded.
    Uses FastAPI's dependency_overrides — clean, no monkeypatching internals.
    """
    from server import app, get_registry

    fake_translator = MagicMock()
    fake_translator.translate = AsyncMock(return_value="नमस्ते दुनिया।")

    fake_tts = MagicMock()
    fake_tts.synthesise = AsyncMock(return_value=(_fake_audio_np(), 22050))

    fake_asr = MagicMock()
    fake_asr.transcribe = AsyncMock(
        return_value="Can we move the meeting to 5 PM tomorrow?"
    )

    registry = _FakeRegistry(
        ready=True,
        fake_translator=fake_translator,
        fake_tts=fake_tts,
        fake_asr=fake_asr,
    )
    app.dependency_overrides[get_registry] = lambda: registry

    client = TestClient(app, raise_server_exceptions=True)
    yield client, fake_translator, fake_tts, fake_asr

    app.dependency_overrides.clear()


@pytest.fixture
def unready_app():
    """Registry with ready=False — simulates models still loading."""
    from server import app, get_registry

    registry = _FakeRegistry(ready=False)
    app.dependency_overrides[get_registry] = lambda: registry

    yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


# ── /api/status ───────────────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_status_ready(self, patched_app):
        client, *_ = patched_app
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is True
        assert "nllb" in data["backends"]

    def test_status_not_ready(self, unready_app):
        r = unready_app.get("/api/status")
        assert r.status_code == 200
        data = r.json()
        assert data["ready"] is False
        assert data["backends"] == []

    def test_status_includes_model_names(self, patched_app):
        client, *_ = patched_app
        data = client.get("/api/status").json()
        assert "faster-whisper" in data["asr"]
        assert "piper" in data["tts"]


# ── /api/translate ─────────────────────────────────────────────────────────────

class TestTranslateEndpoint:
    def test_basic_translation_success(self, patched_app):
        client, *_ = patched_app
        r = client.post("/api/translate", json={
            "text": "Can we move the meeting to 5 PM tomorrow?",
            "backend": "nllb",
            "tone": "auto",
        })
        assert r.status_code == 200
        data = r.json()
        assert "hindi" in data
        assert "audio_b64" in data

    def test_response_contains_all_fields(self, patched_app):
        client, *_ = patched_app
        r = client.post("/api/translate", json={
            "text": "Hello world.",
            "backend": "nllb",
            "tone": "auto",
        })
        data = r.json()
        required = {"english", "cleaned", "hindi", "tone", "audio_b64",
                    "sample_rate", "trans_ms", "tts_ms", "total_ms", "backend"}
        assert required.issubset(data.keys())

    def test_audio_b64_is_valid_wav(self, patched_app):
        client, *_ = patched_app
        r = client.post("/api/translate", json={
            "text": "Good morning.",
            "backend": "nllb",
            "tone": "formal",
        })
        raw = base64.b64decode(r.json()["audio_b64"])
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    def test_503_when_models_not_ready(self, unready_app):
        r = unready_app.post("/api/translate", json={
            "text": "Hello.",
            "backend": "nllb",
            "tone": "auto",
        })
        assert r.status_code == 503

    def test_400_on_empty_text(self, patched_app):
        client, *_ = patched_app
        # "um uh er" → all fillers → empty after removal
        r = client.post("/api/translate", json={
            "text": "um uh er",
            "backend": "nllb",
            "tone": "auto",
        })
        assert r.status_code == 400

    def test_timing_fields_are_non_negative(self, patched_app):
        client, *_ = patched_app
        data = client.post("/api/translate", json={
            "text": "This is a test.",
            "backend": "nllb",
            "tone": "auto",
        }).json()
        assert data["trans_ms"] >= 0
        assert data["tts_ms"] >= 0
        assert data["total_ms"] >= 0

    def test_filler_removal_reflected_in_cleaned_field(self, patched_app):
        client, *_ = patched_app
        r = client.post("/api/translate", json={
            "text": "um can we reschedule the call?",
            "backend": "nllb",
            "tone": "auto",
        })
        data = r.json()
        assert "um" not in data["cleaned"]
        assert "reschedule" in data["cleaned"]

    @pytest.mark.parametrize("tone", ["auto", "formal", "casual"])
    def test_all_tone_values_accepted(self, patched_app, tone):
        client, *_ = patched_app
        r = client.post("/api/translate", json={
            "text": "Let's talk tomorrow.",
            "backend": "nllb",
            "tone": tone,
        })
        assert r.status_code == 200


# ── /api/asr ──────────────────────────────────────────────────────────────────

class TestASREndpoint:
    def test_wav_upload_returns_transcript(self, patched_app):
        client, *_ = patched_app
        r = client.post(
            "/api/asr",
            files={"file": ("audio.wav", _sine_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "english" in data
        assert isinstance(data["english"], str)
        assert len(data["english"]) > 0

    def test_asr_transcript_content(self, patched_app):
        client, *_ = patched_app
        r = client.post(
            "/api/asr",
            files={"file": ("audio.wav", _sine_wav_bytes(), "audio/wav")},
        )
        assert "meeting" in r.json()["english"].lower()

    def test_503_when_models_not_ready(self, unready_app):
        r = unready_app.post(
            "/api/asr",
            files={"file": ("audio.wav", _sine_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 503

    def test_invalid_audio_payload_returns_400(self, patched_app):
        """Garbage bytes with .webm extension should return 400, not 500."""
        client, *_ = patched_app
        r = client.post(
            "/api/asr",
            files={"file": ("recording.webm", b"\x00" * 100, "audio/webm")},
        )
        assert r.status_code == 400

    def test_valid_wav_accepted(self, patched_app):
        client, *_ = patched_app
        r = client.post(
            "/api/asr",
            files={"file": ("audio.wav", _sine_wav_bytes(), "audio/wav")},
        )
        assert r.status_code == 200


# ── /  (static frontend) ──────────────────────────────────────────────────────

class TestStaticFrontend:
    def test_index_html_served(self, patched_app):
        client, *_ = patched_app
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_index_contains_voicebot_title(self, patched_app):
        client, *_ = patched_app
        assert "VoiceBot" in client.get("/").text

    def test_404_for_missing_static_file(self, patched_app):
        client, *_ = patched_app
        r = client.get("/does-not-exist.js")
        assert r.status_code == 404


# ── Slow integration tests (real models) ─────────────────────────────────────

@pytest.mark.slow
class TestRealModelIntegration:
    """
    End-to-end tests that load actual model weights.
    Run with:  pytest -m slow tests/test_api.py
    """

    @pytest.fixture(scope="class")
    def live_client(self):
        from server import app
        with TestClient(app) as client:
            yield client

    def test_status_ready_after_startup(self, live_client):
        r = live_client.get("/api/status")
        assert r.status_code == 200
        assert r.json()["ready"] is True

    def test_nllb_translation_end_to_end(self, live_client):
        r = live_client.post("/api/translate", json={
            "text": "Can we move the Google Meet to 5 PM tomorrow?",
            "backend": "nllb",
            "tone": "auto",
        })
        assert r.status_code == 200
        hindi = r.json()["hindi"]
        assert any("\u0900" <= c <= "\u097F" for c in hindi), f"No Devanagari: {hindi!r}"
        assert "Google" in hindi or "Meet" in hindi

    def test_audio_output_is_non_silent(self, live_client):
        r = live_client.post("/api/translate", json={
            "text": "Hello, how are you?",
            "backend": "nllb",
            "tone": "auto",
        })
        assert r.status_code == 200
        raw = base64.b64decode(r.json()["audio_b64"])
        with wave.open(io.BytesIO(raw), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        assert rms > 10.0, f"Audio appears silent (RMS={rms:.2f})"

    def test_latency_under_10_seconds(self, live_client):
        import time
        start = time.perf_counter()
        r = live_client.post("/api/translate", json={
            "text": "Quick test.",
            "backend": "nllb",
            "tone": "auto",
        })
        elapsed = time.perf_counter() - start
        assert r.status_code == 200
        assert elapsed < 10.0, f"Translation took {elapsed:.1f}s — too slow"
