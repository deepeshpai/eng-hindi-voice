# Gnani — English → Hindi Voice Translation

Real-time English → Hindi translation in the browser.  
Speak or type English → get Hindi text + synthesised speech back.  
Everything runs **100% locally** — no API keys, nothing leaves your machine.

**Stack:** faster-whisper (ASR) → NLLB-200 (translation) → Piper TTS

---

## Quickstart (Docker — recommended)

```bash
git clone <repo-url>
cd eng-hindi-voice
docker compose up
```

Open [http://localhost:8000](http://localhost:8000).

First start downloads ~1.5 GB of model weights to a Docker volume (`gnani_models`).  
Every subsequent start loads from cache and is ready in under a minute.

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

---

## Edge cases handled

| Problem | Solution |
|---|---|
| Strong accents / fast speech | Confidence filtering + temperature fallback in Whisper |
| Named entity mistranslation | spaCy NER + custom hotword registry → placeholder round-trip |
| Partial inputs / fillers | Two-tier filler removal; sentence-boundary detection |
| Latency buildup | Queue backpressure; interruption cancellation |
| Tone drift | Session-level tone memory in SmartBuffer |
| Numbers / dates / currency | Preprocessing: ordinals, date normalization, currency expansion |
| Abbreviations | Expansion table (ETA, PM, SIP, VoIP, …) |

---

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
