"""
Unit tests for audio helpers — no model loading required.

Tests cover:
- services.numpy_to_wav_b64() : float32 numpy → valid base64 WAV
- TTSEngine._synthesise_sync() : piper integration (mocked voice)
"""
from __future__ import annotations

import base64
import io
import wave

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services import numpy_to_wav_b64


# ── numpy_to_wav_b64 ──────────────────────────────────────────────────────────

class TestNumpyToWavB64:
    def test_returns_valid_base64(self, tone_audio, sample_rate):
        b64 = numpy_to_wav_b64(tone_audio, sample_rate)
        raw = base64.b64decode(b64)
        assert len(raw) > 44  # WAV header minimum

    def test_decoded_wav_has_correct_params(self, tone_audio, sample_rate):
        b64 = numpy_to_wav_b64(tone_audio, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == sample_rate

    def test_decoded_frame_count_matches_input(self, tone_audio, sample_rate):
        b64 = numpy_to_wav_b64(tone_audio, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getnframes() == len(tone_audio)

    def test_silence_encodes_correctly(self, silence_audio, sample_rate):
        b64 = numpy_to_wav_b64(silence_audio, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples == 0)

    def test_clipping_applied_to_out_of_range_values(self, sample_rate):
        """Values outside [-1, 1] must be clipped, not wrapped/overflowed."""
        loud = np.array([2.0, -3.0, 1.5], dtype=np.float32)
        b64 = numpy_to_wav_b64(loud, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples <= 32767)
        assert np.all(samples >= -32768)

    def test_stereo_input_converted_to_mono(self, sample_rate):
        stereo = np.zeros((1000, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        b64 = numpy_to_wav_b64(stereo, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() == 1000

    def test_empty_array_produces_valid_wav(self, sample_rate):
        empty = np.zeros(0, dtype=np.float32)
        b64 = numpy_to_wav_b64(empty, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getnframes() == 0

    def test_different_sample_rates(self, tone_audio):
        for sr in [8000, 16000, 22050, 44100]:
            b64 = numpy_to_wav_b64(tone_audio, sr)
            raw = base64.b64decode(b64)
            with wave.open(io.BytesIO(raw), "rb") as wf:
                assert wf.getframerate() == sr

    def test_pcm_values_scale_correctly(self, sample_rate):
        """A +1.0 float32 sample should encode as 32767 int16."""
        ones = np.ones(10, dtype=np.float32)
        b64 = numpy_to_wav_b64(ones, sample_rate)
        raw = base64.b64decode(b64)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples == 32767)


# ── TTSEngine (mocked piper) ──────────────────────────────────────────────────

class TestTTSEngineWithMock:
    """
    Test TTSEngine._synthesise_sync() by injecting a fake PiperVoice that
    yields a known AudioChunk, so the test never loads real ONNX weights.
    """

    def _make_engine(self):
        from unittest.mock import MagicMock
        from pipeline.tts_engine import TTSEngine
        from config import TTSConfig

        engine = TTSEngine(TTSConfig())

        fake_chunk = MagicMock()
        fake_chunk.sample_rate = 22050
        t = np.linspace(0, 0.1, int(22050 * 0.1), endpoint=False)
        fake_chunk.audio_float_array = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        fake_voice = MagicMock()
        fake_voice.synthesize.return_value = [fake_chunk]
        engine._voice = fake_voice
        return engine

    def test_synthesise_sync_returns_float32(self):
        audio, sr = self._make_engine()._synthesise_sync("नमस्ते दुनिया")
        assert audio.dtype == np.float32

    def test_synthesise_sync_returns_correct_sample_rate(self):
        _, sr = self._make_engine()._synthesise_sync("नमस्ते")
        assert sr == 22050

    def test_synthesise_sync_audio_in_range(self):
        audio, _ = self._make_engine()._synthesise_sync("हिन्दी में आज़माइश")
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

    def test_synthesise_sync_concatenates_multiple_chunks(self):
        from unittest.mock import MagicMock
        from pipeline.tts_engine import TTSEngine
        from config import TTSConfig

        engine = TTSEngine(TTSConfig())

        def make_chunk(n_samples):
            chunk = MagicMock()
            chunk.sample_rate = 22050
            chunk.audio_float_array = np.zeros(n_samples, dtype=np.float32)
            return chunk

        fake_voice = MagicMock()
        fake_voice.synthesize.return_value = [make_chunk(100), make_chunk(200)]
        engine._voice = fake_voice

        audio, sr = engine._synthesise_sync("Two sentence text.")
        assert len(audio) == 300

    def test_synthesise_sync_empty_chunks_returns_silence(self):
        from unittest.mock import MagicMock
        from pipeline.tts_engine import TTSEngine
        from config import TTSConfig

        engine = TTSEngine(TTSConfig())
        fake_voice = MagicMock()
        fake_voice.synthesize.return_value = []
        engine._voice = fake_voice

        audio, sr = engine._synthesise_sync("")
        assert len(audio) == 0
        assert sr == 22050
