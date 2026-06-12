"""Standalone sprite-generation worker.

Run this on one or more additional machines (e.g. other Apple Silicon Macs) to
offload part of the SDXL pixel-art sprite generation workload. The main process
(`src.escape_rooms.utils.pixel_art`) dispatches sprite jobs to any worker
URLs listed in `SPRITE_WORKERS`, splitting work across local + remote
generation in parallel.

Usage:
    uvicorn sprite_worker:app --host 0.0.0.0 --port 8001

On startup this advertises itself via Bonjour/mDNS as a
"_sprite-worker._tcp.local." service so the main machine can auto-discover
it (see scripts/setup_main.sh). It can still be addressed manually:
    SPRITE_WORKERS=http://<worker-lan-ip>:8001
"""

from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from src.escape_rooms.utils.logging import get_node_logger
from src.escape_rooms.utils.pixel_art import _generate_via_sd

log = get_node_logger("sprite_worker")

SERVICE_TYPE = "_sprite-worker._tcp.local."
# Must match the --port passed to uvicorn (see scripts/setup_worker.sh).
SERVICE_PORT = int(os.getenv("SPRITE_WORKER_PORT", "8001"))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    azc = AsyncZeroconf()
    hostname = socket.gethostname().split(".")[0]
    info = ServiceInfo(
        SERVICE_TYPE,
        f"{hostname}.{SERVICE_TYPE}",
        addresses=[socket.inet_aton(_local_ip())],
        port=SERVICE_PORT,
        server=f"{hostname}.local.",
    )
    await azc.async_register_service(info)
    log.success("Advertised sprite worker via Bonjour as {}", info.name)
    try:
        yield
    finally:
        await azc.async_unregister_service(info)
        await azc.async_close()


app = FastAPI(title="Sprite Generation Worker", version="1.0.0", lifespan=lifespan)


class SpriteRequest(BaseModel):
    id: str
    description: str


class SpriteResponse(BaseModel):
    id: str
    sprite: str  # base64 PNG


class _ObjLike:
    """Minimal stand-in for WorldObject — _generate_via_sd only needs id/description."""

    def __init__(self, id: str, description: str) -> None:
        self.id = id
        self.description = description


@app.post("/sprite", response_model=SpriteResponse)
def generate_sprite(req: SpriteRequest) -> SpriteResponse:
    log.info("Generating sprite for {!r}", req.id)
    b64 = _generate_via_sd(_ObjLike(req.id, req.description))
    return SpriteResponse(id=req.id, sprite=b64)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
