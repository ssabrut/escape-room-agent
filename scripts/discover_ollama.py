"""Discover advertise_ollama.py instances via Bonjour/mDNS on the LAN.

Browses for "_ollama-worker._tcp.local." services for a few seconds and
prints "<ip>:<port>" for each one found (one per line). IP addresses are
used (rather than .local hostnames) since curl/requests don't reliably
resolve mDNS names. Used by setup_main.sh to auto-configure OLLAMA_WORKERS.

Usage:
    python scripts/discover_ollama.py [timeout_seconds]
"""

from __future__ import annotations

import sys
import time

from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf

SERVICE_TYPE = "_ollama-worker._tcp.local."


def main() -> None:
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    found: list[str] = []

    def on_change(zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
        if state_change is not ServiceStateChange.Added:
            return
        info = zeroconf.get_service_info(service_type, name)
        if info is None:
            return
        addresses = info.parsed_addresses()
        if not addresses:
            return
        found.append(f"{addresses[0]}:{info.port}")

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, handlers=[on_change])
        time.sleep(timeout)
    finally:
        zc.close()

    for entry in found:
        print(entry)


if __name__ == "__main__":
    main()
