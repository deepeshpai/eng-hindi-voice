#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_and_run.sh — Gnani English → Hindi voice translation
#
# What this script does (in order):
#   1. Installs uv (if not already present)
#   2. Creates a Python 3.11 virtual environment via uv
#   3. Installs all project dependencies from pyproject.toml
#   4. Downloads the spaCy English NER model (en_core_web_sm)
#   5. Copies .env.example → .env if no .env exists
#   6. Pre-downloads all ML models to local cache:
#        • faster-whisper base.en         (~74 MB)
#        • facebook/nllb-200-distilled-600M (~1.2 GB, primary translator)
#        • Helsinki-NLP/opus-mt-en-hi     (~300 MB, fast translator)
#        • piper hi_IN-rohan-medium voice (~63 MB, TTS)
#   7. Starts the FastAPI web server at http://localhost:8000
#
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh                 # default: port 8000
#   ./setup_and_run.sh --port 9000
#   ./setup_and_run.sh --skip-models   # skip model downloads (already cached)
#   ./setup_and_run.sh --host 0.0.0.0  # expose to LAN
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Terminal colours ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN='\033[0;32m'; CYAN='\033[0;36m'
    YELLOW='\033[1;33m'; RED='\033[0;31m'
    BOLD='\033[1m'; NC='\033[0m'
else
    GREEN=''; CYAN=''; YELLOW=''; RED=''; BOLD=''; NC=''
fi

step()  { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }
info()  { echo -e "  ${GREEN}✓${NC} $*"; }
warn()  { echo -e "  ${YELLOW}⚠${NC}  $*"; }
die()   { echo -e "\n${RED}✗ Error:${NC} $*" >&2; exit 1; }

# ── Parse arguments ───────────────────────────────────────────────────────────
PORT=8000
HOST=127.0.0.1
SKIP_MODELS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";       shift 2 ;;
        --port=*)  PORT="${1#*=}";  shift   ;;
        --host)    HOST="$2";       shift 2 ;;
        --host=*)  HOST="${1#*=}";  shift   ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# ──/p' "$0" | grep -E '^\s*#' | sed 's/^#\s\?//'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Resolve project root (directory containing this script) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "server.py" ]] || die "server.py not found — run this script from the project root."

echo -e "\n${BOLD}Gnani  ·  English → Hindi Voice Translation${NC}"
echo -e "${BOLD}$(printf '─%.0s' {1..48})${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install uv
# ─────────────────────────────────────────────────────────────────────────────
step "Checking uv package manager"

if ! command -v uv &>/dev/null; then
    warn "uv not found — installing…"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # astral's installer puts uv in ~/.local/bin on macOS/Linux
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    command -v uv &>/dev/null || die "uv install succeeded but binary not found in PATH.
  Try running:  export PATH=\"\$HOME/.local/bin:\$PATH\"
  Then re-run this script."
fi

UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
info "uv $UV_VERSION at $(command -v uv)"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Virtual environment
# ─────────────────────────────────────────────────────────────────────────────
step "Setting up virtual environment"

if [[ ! -d ".venv" ]]; then
    info "Creating .venv with Python 3.11…"
    uv venv .venv --python 3.11 2>/dev/null \
        || uv venv .venv --python 3.10 2>/dev/null \
        || uv venv .venv
else
    info ".venv already exists — reusing."
fi

# Activate so subsequent python / pip calls use it
source .venv/bin/activate
PY=".venv/bin/python"

PYTHON_VERSION=$($PY --version 2>&1)
info "Using $PYTHON_VERSION"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "Installing Python dependencies"
info "This downloads PyTorch, transformers, faster-whisper, piper-tts…"
info "(First run: ~5–15 min depending on bandwidth; subsequent runs: seconds)"

uv pip install -e ".[dev]"
info "All Python packages installed."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — spaCy English NER model
# ─────────────────────────────────────────────────────────────────────────────
step "spaCy language model (en_core_web_sm)"

if $PY -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null 2>&1; then
    info "en_core_web_sm already installed."
else
    info "Downloading en_core_web_sm (~12 MB)…"
    $PY -m spacy download en_core_web_sm
    info "en_core_web_sm ready."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — .env file
# ─────────────────────────────────────────────────────────────────────────────
step "Environment configuration"

if [[ ! -f ".env" ]]; then
    cp .env.example .env
    info "Created .env from .env.example"
    warn "You can edit .env to change Whisper model size, Hindi voice, etc."
else
    info ".env already exists — keeping existing settings."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Pre-download ML models
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_MODELS" -eq 1 ]]; then
    warn "Skipping model downloads (--skip-models flag set)."
else
    step "Pre-downloading ML models (skips files already cached)"
    info "Models are cached to ~/.cache/huggingface/hub and ~/.local/share/gnani/"
    info "This runs once — subsequent server starts load from cache instantly."
    echo ""

    $PY - <<'PYTHON'
import sys, os
from pathlib import Path

# Honour .env before importing config
from dotenv import load_dotenv
load_dotenv()

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
NC     = "\033[0m"
ok   = lambda msg: print(f"  {GREEN}✓{NC} {msg}")
warn = lambda msg: print(f"  {YELLOW}⚠{NC}  {msg}")

# ── faster-whisper (ASR) ────────────────────────────────────────────────────
whisper_model = os.getenv("WHISPER_MODEL", "base.en")
print(f"  Downloading faster-whisper '{whisper_model}' (~74 MB for base.en)…")
try:
    from faster_whisper import WhisperModel
    # Loading the model downloads its CTranslate2 weights to HF cache.
    # We immediately delete the Python object so RAM is freed.
    _m = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    del _m
    ok(f"faster-whisper '{whisper_model}' ready.")
except Exception as e:
    warn(f"faster-whisper download failed: {e}  (server will retry at startup)")

# ── NLLB-200 translation model ───────────────────────────────────────────────
print("  Downloading facebook/nllb-200-distilled-600M (~1.2 GB)…")
print("  (This is the primary translation model — worth the wait!)")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    ok("NLLB-200 ready.")
except Exception as e:
    warn(f"NLLB-200 download failed: {e}  (server will retry at startup)")

# ── Opus-MT translation model ────────────────────────────────────────────────
print("  Downloading Helsinki-NLP/opus-mt-en-hi (~300 MB)…")
try:
    snapshot_download(
        repo_id="Helsinki-NLP/opus-mt-en-hi",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    ok("Opus-MT ready.")
except Exception as e:
    warn(f"Opus-MT download failed: {e}  (server will retry at startup)")

# ── Piper TTS voice ─────────────────────────────────────────────────────────
piper_voice = os.getenv("PIPER_VOICE", "hi_IN-rohan-medium")
print(f"  Downloading piper voice '{piper_voice}' (~63 MB)…")
try:
    from config import TTSConfig
    from pipeline.tts_engine import _download_voice
    tts_cfg = TTSConfig()
    _download_voice(tts_cfg)
    ok(f"Piper voice '{piper_voice}' ready at {tts_cfg.voices_dir}")
except Exception as e:
    warn(f"Piper voice download failed: {e}  (server will retry at startup)")

print()
print(f"  {GREEN}All models cached — server cold-start will be fast.{NC}")
PYTHON

fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Launch the web server
# ─────────────────────────────────────────────────────────────────────────────
step "Starting Gnani web server"
echo ""
echo -e "  ${BOLD}URL:${NC}  http://${HOST}:${PORT}"
echo -e "  ${BOLD}API:${NC}  http://${HOST}:${PORT}/api/status"
echo -e "  ${BOLD}Docs:${NC} http://${HOST}:${PORT}/docs"
echo ""
warn "Models load in the background — the UI shows a spinner until ready."
warn "Press Ctrl-C to stop."
echo ""

exec $PY -m uvicorn server:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info \
    --reload
