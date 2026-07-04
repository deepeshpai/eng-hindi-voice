# VoiceBot — English → Hindi Voice Translation

Real-time English → Hindi translation in the browser.  
Speak or type English → get Hindi text + synthesised speech back.  
Everything runs **100% locally** — no API keys, nothing leaves your machine.

**Stack:** faster-whisper (ASR) → NLLB-200 (translation) → Piper TTS

---

## Quickstart (Docker — recommended)

```bash
git clone <repo-url>
cd eng-hindi-voice
./docker_run.sh
```

Open [http://localhost:8000](http://localhost:8000).

First start builds the image and downloads ~1.5 GB of model weights to a Docker volume (`voicebot_models`).  
Every subsequent start reuses the cache and is ready in under a minute.

### `docker_run.sh` commands

| Command | What it does |
|---|---|
| `./docker_run.sh` | Build image (if needed) + start server on port 8000 |
| `./docker_run.sh --port 9000` | Start on a custom port |
| `./docker_run.sh --rebuild` | Force a fresh image rebuild (e.g. after code changes) |
| `./docker_run.sh --logs` | Tail live server logs (shows model loading progress) |
| `./docker_run.sh --stop` | Stop and remove the running container |

The script polls the health endpoint and prints dots until the server is ready, then shows the URL.  
Model weights are cached in a named Docker volume — they survive container restarts and rebuilds.

---

## Local dev (without Docker)

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Add `--skip-models` if weights are already cached:

```bash
./setup_and_run.sh --skip-models
```

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Whisper model size (`tiny.en`, `small.en`, `medium.en`) |
| `TRANSLATION_BACKEND` | `nllb` | Translation model (only `nllb` supported) |
| `TRANSLATION_DEVICE` | `auto` | `cpu` or `cuda` |
| `PIPER_VOICE` | `hi_IN-rohan-medium` | Hindi TTS voice |
| `HINDI_TONE` | `auto` | `formal`, `casual`, or `auto` |
| `ASR_HOTWORDS` | *(see .env.example)* | Comma-separated terms protected from mistranslation |



## Project layout

```
server.py          FastAPI routes
services.py        ModelRegistry (ASR / Translator / TTS lifecycle)
config.py          All configuration via env vars
models.py          Pydantic request/response schemas
pipeline/
  asr_engine.py    faster-whisper ASR
  translator.py    NLLB-200 translation + tone adjustment
  tts_engine.py    Piper TTS synthesis
  smart_buffer.py  Filler removal, tone detection, backpressure
  preprocessing.py Abbreviation / date / currency expansion
  entity_guard.py  Named-entity placeholder protection
static/
  index.html       Single-page frontend (vanilla JS, Web Audio API)
tests/             pytest suite (fast, no real models)
```
