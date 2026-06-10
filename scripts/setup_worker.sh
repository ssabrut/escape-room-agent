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
# The server listens on 0.0.0.0:8001 so the main Mac can reach it at
# http://<this-mac>.local:8001

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="escape-rooms"
PORT="${SPRITE_WORKER_PORT:-8001}"

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

HOSTNAME_LOCAL="$(scutil --get LocalHostName 2>/dev/null || hostname).local"
echo ""
echo "Setup complete."
echo "Main Mac should set: SPRITE_WORKERS=http://${HOSTNAME_LOCAL}:${PORT}"
echo ""

if [[ "${1:-}" == "--setup" ]]; then
    exit 0
fi

echo "Starting sprite worker on 0.0.0.0:${PORT} ..."
exec uvicorn sprite_worker:app --host 0.0.0.0 --port "${PORT}"
