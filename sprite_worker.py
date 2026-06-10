"""Standalone sprite-generation worker.

Run this on a second machine (e.g. another Apple Silicon Mac) to offload
part of the SDXL pixel-art sprite generation workload. The main process
(`src.escape_rooms.utils.pixel_art`) dispatches sprite jobs to any worker
URLs listed in `SPRITE_WORKERS`, splitting work across local + remote
generation in parallel.

Usage:
    uvicorn sprite_worker:app --host 0.0.0.0 --port 8001

Then on the main machine, point at it:
    SPRITE_WORKERS=http://<worker-ip>:8001
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from src.escape_rooms.utils.logging import get_node_logger
from src.escape_rooms.utils.pixel_art import _generate_via_sd

log = get_node_logger("sprite_worker")

app = FastAPI(title="Sprite Generation Worker", version="1.0.0")


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
