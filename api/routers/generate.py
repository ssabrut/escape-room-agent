from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.escape_rooms.nodes.world_builder import world_builder_node
from src.escape_rooms.nodes.puzzle_builder import puzzle_builder_node
from src.escape_rooms.state import GameState

router = APIRouter(prefix="/generate", tags=["generate"])

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


class GenerateResponse(BaseModel):
    world: dict
    solution_path: list[str]
    num_rooms: int
    num_objects: int


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

    return GenerateResponse(
        world=world.model_dump(mode="json", exclude_none=True),
        solution_path=world.solution_path or [],
        num_rooms=len(world.rooms),
        num_objects=len(world.objects),
    )
