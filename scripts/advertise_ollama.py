"""Advertise a locally-running Ollama instance via Bonjour/mDNS.

Ollama itself doesn't do mDNS, so run this alongside it (on each additional
Mac on the LAN) to make it auto-discoverable as an
"_ollama-worker._tcp.local." service — same pattern as sprite_worker.py's
"_sprite-worker._tcp.local." advertisement, but Ollama keeps serving on its
normal port; this script only registers the Bonjour record.

The main machine discovers these with scripts/discover_ollama.py and fans
out independent LLM calls (per-room theming, storyboard passes, ...) across
the local Ollama instance and each discovered worker — see
Settings.ollama_workers / get_worker_llms in
src/escape_rooms/utils/settings.py.

Usage:
    python scripts/advertise_ollama.py [port]   # default port 11434

Leave running for as long as you want this machine discoverable.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys

from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

SERVICE_TYPE = "_ollama-worker._tcp.local."


def _local_ip() -> str:
    """Best-effort LAN IP — opens a UDP socket to a public address (no packets sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


async def main(port: int) -> None:
    azc = AsyncZeroconf()
    hostname = socket.gethostname().split(".")[0]
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(_local_ip())],
        port=port,
        server=f"{hostname}.local.",
    )
    await azc.async_register_service(info)
    print(f"Advertised Ollama worker via Bonjour as {info.name} on port {port}")
    print("Press Ctrl+C to stop advertising.")
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await azc.async_unregister_service(info)
        await azc.async_close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        # Match Ollama's own default port-resolution: OLLAMA_HOST may be
        # "host:port" or just "port".
        host = os.getenv("OLLAMA_HOST", "11434")
        port = int(host.rsplit(":", 1)[-1]) if host else 11434

    try:
        asyncio.run(main(port))
    except KeyboardInterrupt:
        pass
