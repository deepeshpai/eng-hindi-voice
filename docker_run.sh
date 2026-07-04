#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker_run.sh — Build and run Gnani with Docker
#
# Usage:
#   ./docker_run.sh                 # build (if needed) + run on port 8000
#   ./docker_run.sh --port 9000     # custom port
#   ./docker_run.sh --rebuild       # force image rebuild before running
#   ./docker_run.sh --stop          # stop and remove the running container
#   ./docker_run.sh --logs          # tail logs of the running container
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

step() { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }
info() { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
die()  { echo -e "\n${RED}✗ Error:${NC} $*" >&2; exit 1; }

# ── Constants ─────────────────────────────────────────────────────────────────
COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"
PROJECT_NAME="gnani"

# ── Parse args ────────────────────────────────────────────────────────────────
PORT=8000
REBUILD=0
CMD=run   # run | stop | logs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";   shift 2 ;;
        --port=*)  PORT="${1#*=}"; shift ;;
        --rebuild) REBUILD=1;   shift ;;
        --stop)    CMD=stop;    shift ;;
        --logs)    CMD=logs;    shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# ──/p' "$0" | grep -E '^\s*#' | sed 's/^#\s\?//'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
command -v docker &>/dev/null || die "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/"
docker info &>/dev/null       || die "Docker daemon is not running. Start Docker Desktop and try again."

cd "$(dirname "$COMPOSE_FILE")"

# ── --stop ────────────────────────────────────────────────────────────────────
if [[ "$CMD" == "stop" ]]; then
    step "Stopping Gnani containers"
    docker compose -p "$PROJECT_NAME" down
    info "Containers stopped."
    exit 0
fi

# ── --logs ────────────────────────────────────────────────────────────────────
if [[ "$CMD" == "logs" ]]; then
    exec docker compose -p "$PROJECT_NAME" logs --follow
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}Gnani  ·  English → Hindi Voice Translation  [Docker]${NC}"
echo -e "${BOLD}$(printf '─%.0s' {1..52})${NC}"

step "Checking Docker image"
IMAGE_EXISTS=$(docker images -q gnani:latest 2>/dev/null)

if [[ -z "$IMAGE_EXISTS" || "$REBUILD" -eq 1 ]]; then
    if [[ "$REBUILD" -eq 1 ]]; then
        warn "Forcing image rebuild (--rebuild flag)."
    else
        info "Image not found — building for the first time."
        warn "This installs PyTorch, transformers, piper-tts (~5–15 min on first run)."
    fi
    echo ""
    docker compose -p "$PROJECT_NAME" build
    info "Image built: gnani:latest"
else
    info "Image gnani:latest already exists. Use --rebuild to force a fresh build."
fi

# ── Run ───────────────────────────────────────────────────────────────────────
step "Starting Gnani server"

# Inject the port override into compose via environment variable
export GNANI_PORT="$PORT"

# Bring up (detached) — compose will reuse an existing running container
# gracefully, so re-running the script is idempotent.
docker compose -p "$PROJECT_NAME" up --detach

echo ""
echo -e "  ${BOLD}URL:${NC}  http://localhost:${PORT}"
echo -e "  ${BOLD}API:${NC}  http://localhost:${PORT}/api/status"
echo -e "  ${BOLD}Docs:${NC} http://localhost:${PORT}/docs"
echo ""
warn "Models load in the background on first start (~1.5 GB download)."
warn "The UI shows a spinner until ready — usually 30–60 s after weights are cached."
echo ""
info "Tail logs:    ./docker_run.sh --logs"
info "Stop server:  ./docker_run.sh --stop"
echo ""

# ── Wait for healthy ──────────────────────────────────────────────────────────
step "Waiting for server to become ready"
CONTAINER=$(docker compose -p "$PROJECT_NAME" ps -q gnani 2>/dev/null | head -1)

if [[ -z "$CONTAINER" ]]; then
    warn "Could not find container — check 'docker compose ps' manually."
    exit 0
fi

echo -n "  "
for i in $(seq 1 40); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "unknown")
    if [[ "$STATUS" == "healthy" ]]; then
        echo ""
        info "Server is healthy at http://localhost:${PORT}"
        break
    fi
    echo -n "."
    sleep 3
done

if [[ "$STATUS" != "healthy" ]]; then
    echo ""
    warn "Server is still starting (health: ${STATUS})."
    warn "Run './docker_run.sh --logs' to watch model loading progress."
fi
