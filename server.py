"""
FastAPI HTTP shell for the VoiceBot translation pipeline.

This file owns one concern: HTTP.
All business logic lives in `services.TranslationService`.
All ML model state lives in `services.ModelRegistry`.

Endpoints
---------
GET  /api/status    — readiness probe + available backends
POST /api/translate — text → Hindi + base-64 WAV audio
POST /api/asr       — upload audio file → English transcript

Static frontend is served from ./static/ (index.html at /).

Startup
-------
    uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from models import ASRResponse, StatusResponse, TranslateRequest, TranslateResponse
from services import ModelRegistry, TranslationService


# ── Lifespan: load models once, store on app.state ───────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, registry.load)
    app.state.registry = registry
    yield
    # Nothing to clean up; OS reclaims memory when the process exits.


app = FastAPI(
    title="VoiceBot Translation API",
    version="2.0.0",
    description="Real-time English → Hindi translation pipeline (local, open-source).",
    lifespan=lifespan,
)


# ── Dependency injectors ──────────────────────────────────────────────────────

def get_registry() -> ModelRegistry:
    return app.state.registry


def get_service(registry: ModelRegistry = Depends(get_registry)) -> TranslationService:
    return TranslationService(registry)


def require_ready(registry: ModelRegistry = Depends(get_registry)) -> ModelRegistry:
    if not registry.ready:
        raise HTTPException(503, "Models still loading — try again in a moment")
    return registry


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status", response_model=StatusResponse)
def status(registry: ModelRegistry = Depends(get_registry)) -> StatusResponse:
    return StatusResponse(
        ready=registry.ready,
        backends=registry.available_backends,
        asr="faster-whisper base.en",
        tts="piper hi_IN-rohan-medium",
    )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(
    req: TranslateRequest,
    svc: TranslationService = Depends(get_service),
    _ready: ModelRegistry = Depends(require_ready),
) -> TranslateResponse:
    try:
        return await svc.translate(req.text, req.backend, req.tone)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.post("/api/asr", response_model=ASRResponse)
async def transcribe(
    file: UploadFile = File(...),
    svc: TranslationService = Depends(get_service),
    _ready: ModelRegistry = Depends(require_ready),
) -> ASRResponse:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.mktemp(suffix=suffix)
    try:
        content = await file.read()
        with open(tmp, "wb") as fh:
            fh.write(content)
        transcript = await svc.transcribe(tmp)
        return ASRResponse(english=transcript)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"ASR error: {exc}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── Static frontend ───────────────────────────────────────────────────────────

_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
