from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from src.escape_rooms.nodes.world_builder import world_builder_node
from src.escape_rooms.nodes.puzzle_builder import puzzle_builder_node
from src.escape_rooms.nodes.storyboard_builder import storyboard_builder_node
from src.escape_rooms.nodes.solver import solver_node
from src.escape_rooms.state import GameState, GameWorld, PartyState, Storyboard
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
    num_agents: int = Field(1, ge=1, le=4, description="Number of cooperating solver agents")


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
    storyboard: dict | None = None


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

    state = state.model_copy(update={"world": world})

    emit.put({"type": "progress", "stage": "story", "message": "Writing the story and characters..."})
    sb_update = storyboard_builder_node(state)
    storyboard: Storyboard | None = sb_update.get("storyboard")

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
    emit.put({"type": "sprites", "sprites": sprites})

    solver_result = None
    last_render: dict | None = None
    if req.solve:
        emit.put({"type": "progress", "stage": "solving", "message": "Solving the room to verify it's winnable..."})
        state = state.model_copy(update={"world": world})

        def on_tick(record: dict, ps: PartyState, tick_world: GameWorld, agent_id: str) -> None:
            nonlocal last_render
            last_render = render_world(
                rooms=tick_world.rooms,
                objects=tick_world.objects,
                current_room=ps.agent_rooms.get(agent_id, ps.current_room),
                inventory=ps.agent_inventories.get(agent_id, ps.inventory),
                object_states=ps.object_states,
                tick=ps.tick,
                agent_rooms=dict(ps.agent_rooms),
                agent_inventories={k: list(v) for k, v in ps.agent_inventories.items()},
            )
            emit.put({"type": "tick", "render": last_render, **record})

        solver_update = solver_node(state, on_tick=on_tick, num_agents=req.num_agents)
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
        render=last_render or render_world(
            rooms=world.rooms,
            objects=world.objects,
            current_room=world.rooms[0].id if world.rooms else "",
        ),
        solution_path=world.solution_path or [],
        num_rooms=len(world.rooms),
        num_objects=len(world.objects),
        solver=solver_log,
        sprites=sprites,
        storyboard=storyboard.model_dump(mode="json", exclude_none=True) if storyboard else None,
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


def _stream_pipeline(run: "Callable[[queue.Queue[dict]], dict]") -> Iterator[bytes]:
    events: "queue.Queue[dict]" = queue.Queue()
    result: dict = {}
    error: dict = {}

    def worker() -> None:
        try:
            result["payload"] = run(events)
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

    return StreamingResponse(_stream_pipeline(lambda emit: _run_pipeline(req, emit)), media_type="application/x-ndjson")


class SavedRunSummary(BaseModel):
    filename: str
    theme: str
    created_at: datetime
    num_rooms: int
    num_objects: int
    solver: SolverLog | None = None


@router.get("/runs")
def list_runs() -> list[SavedRunSummary]:
    """List previously generated worlds saved under api_runs/."""
    if not OUTPUT_DIR.exists():
        return []

    summaries: list[SavedRunSummary] = []
    for path in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        timestamp_part = path.stem.split("_")[:2]
        try:
            created_at = datetime.strptime("_".join(timestamp_part), "%Y%m%d_%H%M%S")
        except ValueError:
            created_at = datetime.fromtimestamp(path.stat().st_mtime)

        theme_slug = path.stem[len("_".join(timestamp_part)) + 1:] or path.stem
        theme = theme_slug.replace("_", " ").title()

        solver_data = data.get("solver")
        summaries.append(SavedRunSummary(
            filename=path.name,
            theme=theme,
            created_at=created_at,
            num_rooms=data.get("num_rooms", 0),
            num_objects=data.get("num_objects", 0),
            solver=SolverLog(**solver_data) if solver_data else None,
        ))

    return summaries


@router.get("/runs/{filename}")
def get_run(filename: str) -> GenerateResponse:
    """Fetch a previously generated world's full API-shaped JSON."""
    if "/" in filename or "\\" in filename or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read run: {exc}") from exc

    return GenerateResponse(**data)


class SolveRequest(BaseModel):
    world: dict = Field(..., description="A previously generated world (the 'world' field of a /generate response)")
    num_agents: int = Field(1, ge=1, le=4, description="Number of cooperating solver agents")


class SolveResponse(BaseModel):
    render: dict
    solution_path: list[str]
    solver: SolverLog | None = None


def _run_solve_pipeline(req: SolveRequest, emit: "queue.Queue[dict]") -> dict:
    """Run only the solver against an already-built world, streaming ticks."""
    try:
        world = GameWorld.model_validate(req.world)
    except ValidationError as exc:
        raise ValueError(f"Invalid world: {exc}") from exc

    if not world.rooms:
        raise ValueError("World has no rooms")
    if not world.win_condition.object_id:
        raise ValueError("World has no win condition — cannot solve")

    state = GameState(theme=world.scenario or "mystery", world=world)

    emit.put({"type": "progress", "stage": "solving", "message": "Solving the room to verify it's winnable..."})

    last_render: dict | None = None

    def on_tick(record: dict, ps: PartyState, tick_world: GameWorld, agent_id: str) -> None:
        nonlocal last_render
        last_render = render_world(
            rooms=tick_world.rooms,
            objects=tick_world.objects,
            current_room=ps.agent_rooms.get(agent_id, ps.current_room),
            inventory=ps.agent_inventories.get(agent_id, ps.inventory),
            object_states=ps.object_states,
            tick=ps.tick,
            agent_rooms=dict(ps.agent_rooms),
            agent_inventories={k: list(v) for k, v in ps.agent_inventories.items()},
        )
        emit.put({"type": "tick", "render": last_render, **record})

    solver_update = solver_node(state, on_tick=on_tick, num_agents=req.num_agents)
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

    response = SolveResponse(
        render=last_render or render_world(
            rooms=world.rooms,
            objects=world.objects,
            current_room=world.rooms[0].id if world.rooms else "",
        ),
        solution_path=world.solution_path or [],
        solver=solver_log,
    )
    return response.model_dump()


@router.post("/solve")
def solve(req: SolveRequest) -> StreamingResponse:
    return StreamingResponse(_stream_pipeline(lambda emit: _run_solve_pipeline(req, emit)), media_type="application/x-ndjson")
