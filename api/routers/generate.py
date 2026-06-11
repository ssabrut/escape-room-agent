from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.escape_rooms.nodes.world_builder import world_builder_node
from src.escape_rooms.nodes.puzzle_builder import puzzle_builder_node
from src.escape_rooms.nodes.solver import solver_node
from src.escape_rooms.state import GameState
from src.escape_rooms.utils.pixel_art import generate_world_sprites
from src.escape_rooms.utils.renderer import render_world

router = APIRouter(prefix="/generate", tags=["generate"])

OUTPUT_DIR = Path("api_runs")

THEMES = [
    "Haunted House",
    "Murder Mystery",
    "Prison Break",
    "Pirate Adventure",
    "Bank Robbery",
    "Cosmic Crisis",
    "Treasure Hunt",
    "Zombie Apocalypse",
    "Secret Agents and Spies",
    "Horror",
]


class GenerateRequest(BaseModel):
    theme: str = Field(..., description=f"Escape room theme. One of: {THEMES}")
    hard_mode: bool = Field(False, description="Multi-room world with deep puzzle chains")
    num_rooms: int = Field(4, ge=2, le=10, description="Number of rooms (hard mode only)")
    solve: bool = Field(False, description="Run the LLM solver after generation")


class SolverLog(BaseModel):
    won: bool
    ticks: int
    optimal: int
    reward: float
    efficiency: float
    wasted: int
    history: list[str]


class GenerateResponse(BaseModel):
    world: dict
    render: dict
    solution_path: list[str]
    num_rooms: int
    num_objects: int
    solver: SolverLog | None = None
    sprites: dict[str, str] = {}  # object_id → base64 PNG


def _run_pipeline(req: GenerateRequest, emit: "queue.Queue[dict]") -> dict:
    """Run the generation pipeline, pushing progress events onto *emit*.

    Returns the final GenerateResponse payload (as a dict) on success, or
    raises an exception on failure — the caller is responsible for turning
    that into an "error" event.
    """
    # Settings reads from env vars — set them before constructing anything.
    if req.hard_mode:
        os.environ["HARD_MODE"] = "true"
        os.environ["NUM_ROOMS"] = str(req.num_rooms)
    else:
        os.environ.pop("HARD_MODE", None)
        os.environ.pop("NUM_ROOMS", None)

    state = GameState(theme=req.theme)

    emit.put({"type": "progress", "stage": "world", "message": "Designing the world and rooms..."})
    wb_update = world_builder_node(state)
    if not wb_update.get("world"):
        raise RuntimeError("world_builder produced no world")

    state = state.model_copy(update={"world": wb_update["world"]})

    emit.put({"type": "progress", "stage": "puzzles", "message": "Placing puzzles, locks, and clues..."})
    pb_update = puzzle_builder_node(state)
    world = pb_update.get("world") or wb_update.get("world")
    if world is None:
        raise RuntimeError("puzzle_builder produced no world")

    solver_result = None
    if req.solve:
        emit.put({"type": "progress", "stage": "solving", "message": "Solving the room to verify it's winnable..."})
        state = state.model_copy(update={"world": world})
        solver_update = solver_node(state)
        solver_result = solver_update.get("solver_result")

    solver_log = None
    if solver_result is not None:
        solver_log = SolverLog(
            won=solver_result.won,
            ticks=solver_result.ticks,
            optimal=solver_result.optimal,
            reward=solver_result.reward,
            efficiency=solver_result.efficiency,
            wasted=solver_result.wasted,
            history=solver_result.history,
        )

    total_objects = len(world.objects)
    emit.put({
        "type": "progress",
        "stage": "sprites",
        "message": f"Generating sprites (0/{total_objects})...",
        "current": 0,
        "total": total_objects,
    })

    def on_sprite_progress(done: int, total: int, object_id: str) -> None:
        emit.put({
            "type": "progress",
            "stage": "sprites",
            "message": f"Generating sprites ({done}/{total}): {object_id}",
            "current": done,
            "total": total,
        })

    sprites = generate_world_sprites(world.objects, on_progress=on_sprite_progress)

    response = GenerateResponse(
        world=world.model_dump(mode="json", exclude_none=True),
        render=render_world(
            rooms=world.rooms,
            objects=world.objects,
            current_room=world.rooms[0].id if world.rooms else "",
        ),
        solution_path=world.solution_path or [],
        num_rooms=len(world.rooms),
        num_objects=len(world.objects),
        solver=solver_log,
        sprites=sprites,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = req.theme.lower().replace(" ", "_")
    out_path = OUTPUT_DIR / f"{timestamp}_{slug}.json"
    payload = response.model_dump()
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return payload


def _stream_generate(req: GenerateRequest) -> Iterator[bytes]:
    events: "queue.Queue[dict]" = queue.Queue()
    result: dict = {}
    error: dict = {}

    def worker() -> None:
        try:
            result["payload"] = _run_pipeline(req, events)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            error["detail"] = str(exc)
        finally:
            events.put({"type": "_done"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        event = events.get()
        if event["type"] == "_done":
            break
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    thread.join()

    if error:
        yield (json.dumps({"type": "error", "detail": error["detail"]}, ensure_ascii=False) + "\n").encode("utf-8")
        return

    final = {"type": "done", **result["payload"]}
    yield (json.dumps(final, ensure_ascii=False) + "\n").encode("utf-8")


@router.post("")
def generate(req: GenerateRequest) -> StreamingResponse:
    if req.theme not in THEMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown theme {req.theme!r}. Valid themes: {THEMES}",
        )

    return StreamingResponse(_stream_generate(req), media_type="application/x-ndjson")
