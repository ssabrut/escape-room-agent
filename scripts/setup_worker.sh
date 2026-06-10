#!/usr/bin/env bash
# Set up and run the sprite generation worker on this Mac.
#
# Run this on the SECOND MacBook Pro (the one that will help generate
# pixel-art sprites over LAN). The repo must already be present on this
# machine (e.g. `git clone` or copied over) — run this script from its root.
#
# Usage:
#   ./scripts/setup_worker.sh          # set up env, install deps, then start the worker
#   ./scripts/setup_worker.sh --setup  # only set up env + deps, don't start the server
#
# The server listens on 0.0.0.0:8001 and advertises itself via Bonjour/mDNS
# as "_sprite-worker._tcp.local." so the main Mac can auto-discover it
# (see scripts/setup_main.sh). It's also reachable directly at
# http://<this-mac>.local:8001

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="escape-rooms"
export SPRITE_WORKER_PORT="${SPRITE_WORKER_PORT:-8001}"
PORT="${SPRITE_WORKER_PORT}"

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
echo ""
echo "Setup complete."
echo "Run ./scripts/setup_main.sh on the main Mac to auto-discover this worker,"
echo "or set manually: SPRITE_WORKERS=http://${LAN_IP}:${PORT}"
echo ""

if [[ "${1:-}" == "--setup" ]]; then
    exit 0
fi

echo "Starting sprite worker on 0.0.0.0:${PORT} ..."
exec uvicorn sprite_worker:app --host 0.0.0.0 --port "${PORT}"
