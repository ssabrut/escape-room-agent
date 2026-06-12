#!/usr/bin/env bash
# Configure this Mac (the main machine) to offload sprite generation AND LLM
# inference to one or more worker Macs on the same LAN.
#
# Usage:
#   ./scripts/setup_main.sh                       # auto-discover all workers via Bonjour/mDNS
#   ./scripts/setup_main.sh <worker-hostname-or-ip> [port]   # add one sprite worker manually
#
# Example:
#   ./scripts/setup_main.sh
#   ./scripts/setup_main.sh my-second-mac.local
#   ./scripts/setup_main.sh my-third-mac.local 8001
#
# Run this once per additional worker (manual mode appends to the existing
# sprite-worker list) or just once in auto-discover mode once all workers are
# up — auto-discover mode picks up every "_sprite-worker._tcp.local." AND
# "_ollama-worker._tcp.local." instance currently advertising on the LAN.
# Found URLs are merged (deduped) with any already configured, then each
# worker's health endpoint is checked. Only workers that pass the health
# check are written to .env (SPRITE_WORKERS / OLLAMA_WORKERS) — unreachable
# ones are dropped (and reported) so the main machine continues distributing
# work across the remaining healthy workers.
#
# Start each sprite worker with ./scripts/setup_worker.sh and each Ollama
# worker with ./scripts/advertise_ollama.py on that machine.

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

PYTHON_BIN="python3"
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
        PYTHON_BIN="$(conda run -n ${ENV_NAME} which python)"
    fi
fi

NEW_URLS=()
AUTO_DISCOVER=1

if [[ $# -ge 1 ]]; then
    AUTO_DISCOVER=0
    WORKER_HOST="$1"
    PORT="${2:-8001}"
    NEW_URLS+=("http://${WORKER_HOST}:${PORT}")
else
    echo "No worker specified — searching LAN for sprite workers via Bonjour/mDNS..."

    DISCOVERED="$("${PYTHON_BIN}" scripts/discover_worker.py 3)"

    if [[ -z "${DISCOVERED}" ]]; then
        echo "No sprite workers found on the LAN." >&2
    else
        while IFS= read -r addr; do
            [[ -n "${addr}" ]] && NEW_URLS+=("http://${addr}")
        done <<< "${DISCOVERED}"

        echo "Found ${#NEW_URLS[@]} sprite worker(s): ${NEW_URLS[*]}"
    fi
fi

# Merge (dedup) existing + new URLs, preserving order.
# The ${arr[@]+"${arr[@]}"} form avoids "unbound variable" under `set -u` on
# bash 3.2 (macOS default) when an array is empty.
declare -a MERGED=()
for u in "${URLS[@]+"${URLS[@]}"}" "${NEW_URLS[@]+"${NEW_URLS[@]}"}"; do
    [[ -z "${u}" ]] && continue
    skip=0
    for existing in "${MERGED[@]+"${MERGED[@]}"}"; do
        [[ "${existing}" == "${u}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 0 ]] && MERGED+=("${u}")
done

if [[ "${#MERGED[@]}" -eq 0 ]]; then
    echo "No sprite workers configured or discovered — skipping SPRITE_WORKERS."
else
    echo ""
    echo "Checking sprite worker health..."

    declare -a HEALTHY=()
    declare -a UNHEALTHY=()
    for url in "${MERGED[@]+"${MERGED[@]}"}"; do
        if curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1; then
            echo "  OK    ${url}"
            HEALTHY+=("${url}")
        else
            echo "  FAIL  ${url}"
            UNHEALTHY+=("${url}")
        fi
    done

    HEALTHY_CSV="$(IFS=,; echo "${HEALTHY[*]+"${HEALTHY[*]}"}")"

    if grep -q "^SPRITE_WORKERS=" "${ENV_FILE}"; then
        sed -i.bak "s|^SPRITE_WORKERS=.*|SPRITE_WORKERS=${HEALTHY_CSV}|" "${ENV_FILE}"
        rm -f "${ENV_FILE}.bak"
    else
        {
            echo ""
            echo "# Pixel-art sprite generation worker(s) (LAN)"
            echo "SPRITE_WORKERS=${HEALTHY_CSV}"
        } >> "${ENV_FILE}"
    fi

    echo "Set SPRITE_WORKERS=${HEALTHY_CSV} in ${ENV_FILE}"
    echo ""

    if [[ "${#UNHEALTHY[@]}" -gt 0 ]]; then
        echo "Skipping ${#UNHEALTHY[@]} unreachable sprite worker(s):" >&2
        for url in "${UNHEALTHY[@]}"; do
            echo "  - ${url}" >&2
        done
        echo "Make sure ./scripts/setup_worker.sh is running on each worker Mac to re-add them." >&2
        echo "" >&2
    fi

    if [[ "${#HEALTHY[@]}" -gt 0 ]]; then
        echo "${#HEALTHY[@]} sprite worker(s) reachable. Distributed sprite generation will use these workers."
    else
        echo "No sprite workers reachable — sprite generation will run locally only."
    fi
fi

# ---------------------------------------------------------------------------
# Ollama workers — distributed LLM inference (per-room theming, storyboard
# beats/flavor passes; see Settings.ollama_workers / get_worker_llms)
# ---------------------------------------------------------------------------

echo ""

EXISTING_OLLAMA="$(grep "^OLLAMA_WORKERS=" "${ENV_FILE}" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
declare -a OLLAMA_URLS=()
if [[ -n "${EXISTING_OLLAMA}" ]]; then
    IFS=',' read -ra OLLAMA_URLS <<< "${EXISTING_OLLAMA}"
fi

declare -a NEW_OLLAMA_URLS=()

if [[ "${AUTO_DISCOVER}" -eq 1 ]]; then
    echo "Searching LAN for Ollama workers via Bonjour/mDNS..."

    DISCOVERED_OLLAMA="$("${PYTHON_BIN}" scripts/discover_ollama.py 3)"

    if [[ -z "${DISCOVERED_OLLAMA}" ]]; then
        echo "No Ollama workers found on the LAN." >&2
        echo "Make sure ./scripts/advertise_ollama.py is running on each worker Mac." >&2
    else
        while IFS= read -r addr; do
            [[ -n "${addr}" ]] && NEW_OLLAMA_URLS+=("http://${addr}")
        done <<< "${DISCOVERED_OLLAMA}"

        echo "Found ${#NEW_OLLAMA_URLS[@]} Ollama worker(s): ${NEW_OLLAMA_URLS[*]}"
    fi
else
    echo "Skipping Ollama worker discovery (manual mode — pass no arguments to auto-discover)."
fi

declare -a MERGED_OLLAMA=()
for u in "${OLLAMA_URLS[@]+"${OLLAMA_URLS[@]}"}" "${NEW_OLLAMA_URLS[@]+"${NEW_OLLAMA_URLS[@]}"}"; do
    [[ -z "${u}" ]] && continue
    skip=0
    for existing in "${MERGED_OLLAMA[@]+"${MERGED_OLLAMA[@]}"}"; do
        [[ "${existing}" == "${u}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 0 ]] && MERGED_OLLAMA+=("${u}")
done

if [[ "${#MERGED_OLLAMA[@]}" -eq 0 ]]; then
    echo "No Ollama workers configured or discovered — skipping OLLAMA_WORKERS."
else
    echo ""
    echo "Checking Ollama worker health..."

    declare -a HEALTHY_OLLAMA=()
    declare -a UNHEALTHY_OLLAMA=()
    for url in "${MERGED_OLLAMA[@]+"${MERGED_OLLAMA[@]}"}"; do
        if curl -fsS --max-time 5 "${url}/" >/dev/null 2>&1; then
            echo "  OK    ${url}"
            HEALTHY_OLLAMA+=("${url}")
        else
            echo "  FAIL  ${url}"
            UNHEALTHY_OLLAMA+=("${url}")
        fi
    done

    HEALTHY_OLLAMA_CSV="$(IFS=,; echo "${HEALTHY_OLLAMA[*]+"${HEALTHY_OLLAMA[*]}"}")"

    if grep -q "^OLLAMA_WORKERS=" "${ENV_FILE}"; then
        sed -i.bak "s|^OLLAMA_WORKERS=.*|OLLAMA_WORKERS=${HEALTHY_OLLAMA_CSV}|" "${ENV_FILE}"
        rm -f "${ENV_FILE}.bak"
    else
        {
            echo ""
            echo "# Distributed Ollama inference worker(s) (LAN)"
            echo "OLLAMA_WORKERS=${HEALTHY_OLLAMA_CSV}"
        } >> "${ENV_FILE}"
    fi

    echo "Set OLLAMA_WORKERS=${HEALTHY_OLLAMA_CSV} in ${ENV_FILE}"
    echo ""

    if [[ "${#UNHEALTHY_OLLAMA[@]}" -gt 0 ]]; then
        echo "Skipping ${#UNHEALTHY_OLLAMA[@]} unreachable Ollama worker(s):" >&2
        for url in "${UNHEALTHY_OLLAMA[@]}"; do
            echo "  - ${url}" >&2
        done
        echo "Make sure ./scripts/advertise_ollama.py is running on each worker Mac to re-add them." >&2
        echo "" >&2
    fi

    if [[ "${#HEALTHY_OLLAMA[@]}" -gt 0 ]]; then
        echo "${#HEALTHY_OLLAMA[@]} Ollama worker(s) reachable. Distributed inference will use these workers."
    else
        echo "No Ollama workers reachable — inference will run locally only."
    fi
fi
