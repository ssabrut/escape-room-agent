"""World Builder agent — generates the escape room setting: scenario, objective, and rooms.

Produces a GameWorld with rooms and goals only. Objects and the solution path are
built by puzzle_builder_node in the next graph step.
"""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import Settings, get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Prerequisite, Room

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")
BANK_GENERATION_PROMPT = load_prompt("game_master", "generation_bank")

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}

_VALID_PREREQ_TYPES = {"object_state", "known_info", "has_item", "power_active"}


def _parse_json(text: str) -> dict | None:
    fence_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else text.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            return None
    return None


def _build_prerequisite(raw) -> Prerequisite | None:
    if not isinstance(raw, dict):
        return None
    ptype = raw.get("type")
    if ptype not in _VALID_PREREQ_TYPES:
        return None
    return Prerequisite(
        type=ptype,
        object_id=raw.get("object_id"),
        state=raw.get("state"),
        info=raw.get("info"),
        id=raw.get("id"),
    )


def _build_rooms(raw_rooms: list) -> list[Room]:
    rooms: list[Room] = []
    for raw_room in raw_rooms:
        if isinstance(raw_room, str):
            rooms.append(Room(id=raw_room, description="", adjacency={}))
            continue
        if not isinstance(raw_room, dict):
            continue
        room_id = raw_room.get("id") or raw_room.get("name")
        if not room_id:
            continue
        raw_adj = raw_room.get("adjacency", {})
        adjacency = (
            {k: v for k, v in raw_adj.items() if isinstance(v, str) and v}
            if isinstance(raw_adj, dict)
            else {}
        )
        key_objects_raw = raw_room.get("key_objects", [])
        key_objects = (
            [k for k in key_objects_raw if isinstance(k, str)]
            if isinstance(key_objects_raw, list)
            else []
        )
        rooms.append(
            Room(
                id=room_id,
                description=raw_room.get("description", ""),
                adjacency=adjacency,
                goal=raw_room.get("goal", ""),
                goal_completion=_build_prerequisite(raw_room.get("goal_completion")),
                key_objects=key_objects,
            )
        )
    return rooms


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    known = {r.id for r in rooms}

    cleaned: dict[str, dict[str, str]] = {}
    for room in rooms:
        cleaned[room.id] = {
            d: n for d, n in room.adjacency.items() if n in known and d in OPPOSITES
        }

    for room_id, adj in list(cleaned.items()):
        for direction, neighbor_id in adj.items():
            reverse = OPPOSITES[direction]
            neighbor_adj = cleaned.get(neighbor_id, {})
            if reverse not in neighbor_adj:
                neighbor_adj[reverse] = room_id
                cleaned[neighbor_id] = neighbor_adj

    return [
        Room(
            id=r.id,
            description=r.description,
            adjacency=cleaned.get(r.id, {}),
            goal=r.goal,
            goal_completion=r.goal_completion,
            key_objects=r.key_objects,
        )
        for r in rooms
    ]


MAX_ROOMS = 2


def _select_rooms(rooms: list[Room], limit: int) -> list[Room]:
    """Truncate to `limit` rooms, keeping the start and last rooms."""
    if len(rooms) <= limit:
        return rooms
    anchors = {rooms[0].id, rooms[-1].id}
    anchored = [r for r in rooms if r.id in anchors]
    rest = [r for r in rooms if r.id not in anchors]
    return (anchored + rest)[:limit]


def _build_world(data: dict, max_rooms: int) -> GameWorld:
    """Parse and validate room/scenario data into a rooms-only GameWorld skeleton."""
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    if max_rooms > 0 and len(rooms) > max_rooms:
        rooms = _select_rooms(rooms, max_rooms)
        kept_ids = {r.id for r in rooms}
        for r in rooms:
            r.adjacency = {d: n for d, n in r.adjacency.items() if n in kept_ids}
        rooms = _repair_adjacency(rooms)

    return GameWorld(
        scenario=data.get("scenario", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
        objects=[],
        rules=[],
        solution_path=[],
    )


def _generation_prompt(theme: str) -> str:
    s = Settings()
    if s.hard_mode:
        return BANK_GENERATION_PROMPT.format(
            theme=theme,
            num_rooms=s.num_rooms,
            chain_depth=s.chain_depth,
        )
    return GENERATION_PROMPT.format(theme=theme)


def _generate_world(llm, theme: str, max_rooms: int) -> tuple[GameWorld, str]:
    prompt = _generation_prompt(theme)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    return _build_world(data, max_rooms), response.content


# ---------------------------------------------------------------------------
# Eval A — deterministic structural check on the rooms-only world skeleton
# ---------------------------------------------------------------------------

_OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _eval_world_structure(world: GameWorld, expected_rooms: int) -> list[str]:
    """Return structural violation strings for the rooms-only world skeleton.

    Only checks what world_builder owns: room count, metadata, and adjacency
    topology. Object-level checks belong to eval B in puzzle_builder_node.
    """
    issues: list[str] = []

    if not world.scenario:
        issues.append("missing scenario")
    if not world.objective:
        issues.append("missing objective")

    if len(world.rooms) != expected_rooms:
        issues.append(f"expected {expected_rooms} room(s), got {len(world.rooms)}")

    seen_ids: set[str] = set()
    for r in world.rooms:
        if r.id in seen_ids:
            issues.append(f"duplicate room id: '{r.id}'")
        seen_ids.add(r.id)

    room_ids = {r.id for r in world.rooms}
    for r in world.rooms:
        if not r.description:
            issues.append(f"room '{r.id}': missing description")
        if not r.goal:
            issues.append(f"room '{r.id}': missing goal")
        if r.goal_completion is None:
            issues.append(f"room '{r.id}': missing goal_completion")
        for direction, neighbor_id in r.adjacency.items():
            if neighbor_id not in room_ids:
                issues.append(
                    f"room '{r.id}': adjacency '{direction}' → '{neighbor_id}' does not exist"
                )
            else:
                reverse = _OPPOSITES.get(direction)
                neighbor = next((x for x in world.rooms if x.id == neighbor_id), None)
                if reverse and neighbor and neighbor.adjacency.get(reverse) != r.id:
                    issues.append(
                        f"room '{r.id}': adjacency not mirrored — "
                        f"'{neighbor_id}' missing '{reverse}' → '{r.id}'"
                    )

    return issues


def world_builder_node(state: GameState) -> dict:
    s = Settings()
    llm = get_llm("game_master")
    max_rooms = s.num_rooms if s.hard_mode else MAX_ROOMS
    mode = "HARD" if s.hard_mode else "standard"
    max_attempts = s.gen_max_attempts if s.gen_max_attempts > 0 else 1

    attempt_log: list[dict] = []

    start = time.perf_counter()
    world, raw = _generate_world(llm, state.theme, max_rooms)

    attempts = 1
    issues = _eval_world_structure(world, max_rooms)

    while issues and attempts < max_attempts:
        attempt_log.append({"attempt": attempts, "issues": issues, "raw": raw})
        print(
            f"[world_builder] attempt {attempts} rejected — "
            f"{len(issues)} structural issue(s):",
            flush=True,
        )
        for issue in issues:
            print(f"  • {issue}", flush=True)
        print(
            f"[world_builder] regenerating (attempt {attempts + 1}/{max_attempts})...",
            flush=True,
        )
        world, raw = _generate_world(llm, state.theme, max_rooms)
        attempts += 1
        issues = _eval_world_structure(world, max_rooms)

    elapsed = time.perf_counter() - start

    if issues:
        print(
            f"[world_builder] WARNING: {len(issues)} issue(s) remain after "
            f"{attempts} attempt(s):",
            flush=True,
        )
        for issue in issues:
            print(f"  • {issue}", flush=True)
    else:
        print(
            f"[world_builder] generated {mode} world ({len(world.rooms)} room(s)) "
            f"in {attempts} attempt(s) in {elapsed:.2f}s",
            flush=True,
        )

    messages: list[AIMessage] = []
    for entry in attempt_log:
        header = (
            f"=== ATTEMPT {entry['attempt']} (rejected) ===\n"
            + "\n".join(f"  • {i}" for i in entry["issues"])
        )
        messages.append(AIMessage(content=f"{header}\n\n{entry['raw']}"))
    messages.append(AIMessage(content=f"=== ATTEMPT {attempts} (final) ===\n\n{raw}"))

    return {
        "messages": messages,
        "world": world,
    }