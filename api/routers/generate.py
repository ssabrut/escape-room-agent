from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.escape_rooms.nodes.world_builder import world_builder_node
from src.escape_rooms.nodes.puzzle_builder import puzzle_builder_node
from src.escape_rooms.nodes.solver import solver_node
from src.escape_rooms.state import GameState
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


@router.post("", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if req.theme not in THEMES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown theme {req.theme!r}. Valid themes: {THEMES}",
        )

    # Settings reads from env vars — set them before constructing anything.
    if req.hard_mode:
        os.environ["HARD_MODE"] = "true"
        os.environ["NUM_ROOMS"] = str(req.num_rooms)
    else:
        os.environ.pop("HARD_MODE", None)
        os.environ.pop("NUM_ROOMS", None)

    state = GameState(theme=req.theme)

    wb_update = world_builder_node(state)
    if not wb_update.get("world"):
        raise HTTPException(status_code=500, detail="world_builder produced no world")

    state = state.model_copy(update={"world": wb_update["world"]})

    pb_update = puzzle_builder_node(state)
    world = pb_update.get("world") or wb_update.get("world")
    if world is None:
        raise HTTPException(status_code=500, detail="puzzle_builder produced no world")

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
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = req.theme.lower().replace(" ", "_")
    out_path = OUTPUT_DIR / f"{timestamp}_{slug}.json"
    out_path.write_text(
        json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return response
