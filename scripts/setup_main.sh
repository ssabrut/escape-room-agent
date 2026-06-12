#!/usr/bin/env bash
# Configure this Mac (the main machine) to offload sprite generation to one or
# more worker Macs on the same LAN.
#
# Usage:
#   ./scripts/setup_main.sh                       # auto-discover all workers via Bonjour/mDNS
#   ./scripts/setup_main.sh <worker-hostname-or-ip> [port]   # add one worker manually
#
# Example:
#   ./scripts/setup_main.sh
#   ./scripts/setup_main.sh my-second-mac.local
#   ./scripts/setup_main.sh my-third-mac.local 8001
#
# Run this once per additional worker (manual mode appends to the existing
# list) or just once in auto-discover mode once all workers are up — it picks
# up every "_sprite-worker._tcp.local." instance currently advertising on the
# LAN. Either way, newly found URLs are merged (deduped) with any already in
# SPRITE_WORKERS in .env, then each worker's /health endpoint is checked.
#
# Start each worker first with ./scripts/setup_worker.sh on that machine.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="escape-rooms"
ENV_FILE=".env"
[[ -f "${ENV_FILE}" ]] || touch "${ENV_FILE}"

# Existing SPRITE_WORKERS entries (if any), one per line.
EXISTING="$(grep "^SPRITE_WORKERS=" "${ENV_FILE}" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
declare -a URLS=()
if [[ -n "${EXISTING}" ]]; then
    IFS=',' read -ra URLS <<< "${EXISTING}"
fi

NEW_URLS=()

if [[ $# -ge 1 ]]; then
    WORKER_HOST="$1"
    PORT="${2:-8001}"
    NEW_URLS+=("http://${WORKER_HOST}:${PORT}")
else
    echo "No worker specified — searching LAN for sprite workers via Bonjour/mDNS..."

    PYTHON_BIN="python3"
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
            PYTHON_BIN="$(conda run -n ${ENV_NAME} which python)"
        fi
    fi

    DISCOVERED="$("${PYTHON_BIN}" scripts/discover_worker.py 3)"

    if [[ -z "${DISCOVERED}" ]]; then
        echo "No sprite workers found on the LAN." >&2
        echo "Make sure ./scripts/setup_worker.sh is running on each worker Mac," >&2
        echo "or add one manually: $0 <worker-hostname-or-ip> [port]" >&2
        exit 1
    fi

    while IFS= read -r addr; do
        [[ -n "${addr}" ]] && NEW_URLS+=("http://${addr}")
    done <<< "${DISCOVERED}"

    echo "Found ${#NEW_URLS[@]} sprite worker(s): ${NEW_URLS[*]}"
fi

# Merge (dedup) existing + new URLs, preserving order.
declare -a MERGED=()
for u in "${URLS[@]}" "${NEW_URLS[@]}"; do
    [[ -z "${u}" ]] && continue
    skip=0
    for existing in "${MERGED[@]}"; do
        [[ "${existing}" == "${u}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 0 ]] && MERGED+=("${u}")
done

MERGED_CSV="$(IFS=,; echo "${MERGED[*]}")"

if grep -q "^SPRITE_WORKERS=" "${ENV_FILE}"; then
    sed -i.bak "s|^SPRITE_WORKERS=.*|SPRITE_WORKERS=${MERGED_CSV}|" "${ENV_FILE}"
    rm -f "${ENV_FILE}.bak"
else
    {
        echo ""
        echo "# Pixel-art sprite generation worker(s) (LAN)"
        echo "SPRITE_WORKERS=${MERGED_CSV}"
    } >> "${ENV_FILE}"
fi

echo "Set SPRITE_WORKERS=${MERGED_CSV} in ${ENV_FILE}"
echo ""
echo "Checking worker health..."

FAILED=0
for url in "${MERGED[@]}"; do
    if curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1; then
        echo "  OK    ${url}"
    else
        echo "  FAIL  ${url}"
        FAILED=1
    fi
done

echo ""
if [[ "${FAILED}" -eq 0 ]]; then
    echo "All ${#MERGED[@]} worker(s) reachable. Distributed sprite generation is ready."
else
    echo "One or more workers are unreachable." >&2
    echo "Make sure ./scripts/setup_worker.sh is running on each worker Mac." >&2
    exit 1
fi
