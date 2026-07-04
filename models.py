"""
Pydantic request/response schemas for the Gnani HTTP API.

Keeping schemas separate from route handlers means:
  - routes stay thin (one job: HTTP ↔ service)
  - schemas can be imported by tests without importing FastAPI
  - OpenAPI docs are auto-generated with accurate types
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="English text to translate")
    backend: str = Field("nllb", description="Translation backend (only 'nllb' supported)")
    tone: str = Field(
        "auto",
        description="Hindi register: 'formal' (आप), 'casual' (तुम), or 'auto'",
    )


class TranslateResponse(BaseModel):
    english: str
    cleaned: str = Field(description="English text after filler removal")
    hindi: str = Field(description="Translated Hindi text")
    tone: str = Field(description="Effective tone used ('formal' or 'casual')")
    audio_b64: str = Field(description="Base-64 encoded WAV (mono 16-bit)")
    sample_rate: int
    backend: str
    trans_ms: int = Field(description="Translation wall-clock time (ms)")
    tts_ms: int = Field(description="TTS synthesis wall-clock time (ms)")
    total_ms: int = Field(description="Total request wall-clock time (ms)")


class ASRResponse(BaseModel):
    english: str = Field(description="Transcribed English text")


class StatusResponse(BaseModel):
    ready: bool
    backends: list[str]
    asr: str
    tts: str
