#!/usr/bin/env bash
# Configure this Mac (the main machine) to offload sprite generation to a
# worker Mac on the same LAN.
#
# Usage:
#   ./scripts/setup_main.sh                       # auto-discover worker via Bonjour/mDNS
#   ./scripts/setup_main.sh <worker-hostname-or-ip> [port]   # specify it manually
#
# Example:
#   ./scripts/setup_main.sh
#   ./scripts/setup_main.sh my-second-mac.local
#   ./scripts/setup_main.sh my-second-mac.local 8001
#
# This writes/updates SPRITE_WORKERS in .env and checks that the worker's
# /health endpoint is reachable. Start the worker first with
# ./scripts/setup_worker.sh on the other machine.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="escape-rooms"

if [[ $# -ge 1 ]]; then
    WORKER_HOST="$1"
    PORT="${2:-8001}"
    WORKER_URL="http://${WORKER_HOST}:${PORT}"
else
    echo "No worker specified — searching LAN for a sprite worker via Bonjour/mDNS..."

    PYTHON_BIN="python3"
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
            PYTHON_BIN="$(conda run -n ${ENV_NAME} which python)"
        fi
    fi

    DISCOVERED="$("${PYTHON_BIN}" scripts/discover_worker.py 3 | head -n1)"

    if [[ -z "${DISCOVERED}" ]]; then
        echo "No sprite worker found on the LAN." >&2
        echo "Make sure ./scripts/setup_worker.sh is running on the worker Mac," >&2
        echo "or specify it manually: $0 <worker-hostname-or-ip> [port]" >&2
        exit 1
    fi

    WORKER_URL="http://${DISCOVERED}"
    echo "Found sprite worker at ${WORKER_URL}"
fi

ENV_FILE=".env"
[[ -f "${ENV_FILE}" ]] || touch "${ENV_FILE}"

if grep -q "^SPRITE_WORKERS=" "${ENV_FILE}"; then
    sed -i.bak "s|^SPRITE_WORKERS=.*|SPRITE_WORKERS=${WORKER_URL}|" "${ENV_FILE}"
    rm -f "${ENV_FILE}.bak"
else
    {
        echo ""
        echo "# Pixel-art sprite generation worker (LAN)"
        echo "SPRITE_WORKERS=${WORKER_URL}"
    } >> "${ENV_FILE}"
fi

echo "Set SPRITE_WORKERS=${WORKER_URL} in ${ENV_FILE}"
echo ""
echo "Checking worker health at ${WORKER_URL}/health ..."

if curl -fsS --max-time 5 "${WORKER_URL}/health" >/dev/null 2>&1; then
    echo "Worker is reachable. Distributed sprite generation is ready."
else
    echo "Could not reach the worker at ${WORKER_URL}." >&2
    echo "Make sure ./scripts/setup_worker.sh is running on the worker Mac." >&2
    exit 1
fi
