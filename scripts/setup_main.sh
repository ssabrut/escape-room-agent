#!/usr/bin/env bash
# Configure this Mac (the main machine) to offload sprite generation to a
# worker Mac on the same LAN.
#
# Usage:
#   ./scripts/setup_main.sh <worker-hostname-or-ip> [port]
#
# Example:
#   ./scripts/setup_main.sh my-second-mac.local
#   ./scripts/setup_main.sh my-second-mac.local 8001
#
# This writes/updates SPRITE_WORKERS in .env and checks that the worker's
# /health endpoint is reachable. Start the worker first with
# ./scripts/setup_worker.sh on the other machine.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <worker-hostname-or-ip> [port]" >&2
    echo "Example: $0 my-second-mac.local" >&2
    exit 1
fi

WORKER_HOST="$1"
PORT="${2:-8001}"
WORKER_URL="http://${WORKER_HOST}:${PORT}"

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
    echo "Could not reach the worker." >&2
    echo "Make sure ./scripts/setup_worker.sh is running on ${WORKER_HOST}." >&2
    exit 1
fi
