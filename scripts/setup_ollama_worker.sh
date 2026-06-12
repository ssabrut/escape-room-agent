#!/usr/bin/env bash
# Make this Mac's Ollama instance discoverable for distributed LLM inference.
#
# Run this on each ADDITIONAL Mac on the LAN that has Ollama running with the
# same model(s) as the main machine (see BUILDER_MODEL/STORYBOARD_MODEL in
# .env). The repo must already be present on this machine — run this script
# from its root.
#
# Usage:
#   ./scripts/setup_ollama_worker.sh          # set up env, then start advertising
#   ./scripts/setup_ollama_worker.sh --setup  # only set up env + deps, don't start advertising
#
# This advertises the local Ollama instance via Bonjour/mDNS as
# "_ollama-worker._tcp.local." so the main Mac can auto-discover it with
# ./scripts/setup_main.sh, which writes OLLAMA_WORKERS in .env. Ollama itself
# keeps serving on its normal port (default 11434) — this script only
# registers the Bonjour record.
#
# Make sure Ollama is reachable from the LAN (OLLAMA_HOST=0.0.0.0 if needed)
# and is already running with the required model(s) pulled.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="escape-rooms"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install Miniconda/Anaconda first: https://docs.conda.io/" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Creating conda env '${ENV_NAME}' (python 3.11)..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<this-mac-ip>")"
OLLAMA_PORT="$(echo "${OLLAMA_HOST:-11434}" | sed 's/.*://')"

echo ""
echo "Setup complete."
echo "Run ./scripts/setup_main.sh on the main Mac to auto-discover this worker,"
echo "or set manually: OLLAMA_WORKERS=http://${LAN_IP}:${OLLAMA_PORT}"
echo ""

if [[ "${1:-}" == "--setup" ]]; then
    exit 0
fi

if ! curl -fsS --max-time 5 "http://localhost:${OLLAMA_PORT}/" >/dev/null 2>&1; then
    echo "Warning: Ollama doesn't seem to be running on port ${OLLAMA_PORT}." >&2
    echo "Start it first (e.g. \`ollama serve\`), then re-run this script." >&2
fi

echo "Advertising Ollama on port ${OLLAMA_PORT} via Bonjour..."
exec python scripts/advertise_ollama.py "${OLLAMA_PORT}"
