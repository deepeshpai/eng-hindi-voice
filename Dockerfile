# ─────────────────────────────────────────────────────────────────────────────
# Gnani — English → Hindi Voice Translation
#
# Build:  docker compose up --build
# Run:    docker compose up
#
# First cold-start downloads ML models (~1.5 GB) to a named Docker volume.
# Every subsequent start loads weights from cache and is ready in seconds.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="Gnani — English → Hindi Voice Translation"

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential  — compile native wheels (sentencepiece, webrtcvad, etc.)
# libgomp1         — OpenMP used by PyTorch / ctranslate2
# libsndfile1      — audio file I/O (piper-tts)
# git              — some pip packages fetch via git at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy the manifest first so this layer is cached until pyproject.toml changes.
COPY pyproject.toml ./
# Stub out the source packages so pip can read the project metadata without
# needing the full source tree during the dependency install step.
RUN mkdir -p pipeline utils && touch pipeline/__init__.py utils/__init__.py

# Install CPU-only PyTorch first (saves ~700 MB vs the default CUDA wheel).
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

# Install all remaining project dependencies.
RUN pip install --no-cache-dir --no-build-isolation ".[dev]"

# Download the spaCy English NER model into the image layer so it's available
# without network access at runtime.
RUN python -m spacy download en_core_web_sm

# ── Application source ────────────────────────────────────────────────────────
# Copy after deps so source changes don't bust the pip cache layer.
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
# ML model weights live on a mounted volume — never baked into the image.
# docker-compose.yml mounts gnani_models → /models.
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface/hub \
    PIPER_VOICES_DIR=/models/piper \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Security ──────────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 gnani && chown -R gnani /app
USER gnani

EXPOSE 8000

# Health check — polls /api/status until models finish loading (~30–60 s).
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=8 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/status')" \
        || exit 1

CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
