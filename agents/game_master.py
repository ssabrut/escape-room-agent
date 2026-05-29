"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Prerequisite, Room, WinCondition, WorldObject

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}

_OPTIONAL_OBJECT_FIELDS = (
    "requires_code",
    "code_digits",
    "requires_tool",
    "requires_liquid",
    "requires_power",
    "fuses",
    "contains_info",
    "slot_description",
    "note",
)


def _parse_json(text: str) -> dict | None:
    """Three-tier JSON extraction: fence → raw → first {...} block."""
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


_VALID_PREREQ_TYPES = {"object_state", "known_info", "has_item", "power_active"}


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


def _build_prerequisites(raw_prereqs) -> list[Prerequisite]:
    if not isinstance(raw_prereqs, list):
        return []
    return [p for p in (_build_prerequisite(r) for r in raw_prereqs) if p is not None]


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
                prerequisites=_build_prerequisites(raw_room.get("prerequisites", [])),
                key_objects=key_objects,
            )
        )
    return rooms


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    """Drop adjacency entries pointing at unknown rooms and mirror missing reverse edges."""
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
            prerequisites=r.prerequisites,
            key_objects=r.key_objects,
        )
        for r in rooms
    ]


def _build_objects(raw_objects: list[dict], room_ids: set[str]) -> list[WorldObject]:
    """Build objects, validating that locations and tool references resolve."""
    candidates: list[dict] = [o for o in raw_objects if isinstance(o, dict) and o.get("id")]
    known_object_ids = {o["id"] for o in candidates}
    valid_locations = room_ids | known_object_ids

    objects: list[WorldObject] = []
    for raw in candidates:
        location = raw.get("location", "")
        if location not in valid_locations:
            continue  # drop objects placed nowhere real

        requires_tool = raw.get("requires_tool") or None
        if requires_tool and requires_tool not in known_object_ids:
            requires_tool = None  # null out dangling tool references

        kwargs = {
            "id": raw["id"],
            "location": location,
            "description": raw.get("description", ""),
            "state": raw.get("state", "visible"),
            "interactable": bool(raw.get("interactable", False)),
            "takeable": bool(raw.get("takeable", False)),
            "requires_tool": requires_tool,
        }
        for field in _OPTIONAL_OBJECT_FIELDS:
            if field == "requires_tool":
                continue
            if field in raw and raw[field] not in (None, ""):
                kwargs[field] = raw[field]

        objects.append(WorldObject(**kwargs))
    return objects


def _build_win_condition(raw: dict, object_ids: set[str]) -> WinCondition:
    if not isinstance(raw, dict):
        return WinCondition()
    object_id = raw.get("object_id", "")
    if object_id not in object_ids:
        object_id = next(iter(object_ids), "")
    return WinCondition(object_id=object_id, state=raw.get("state", ""))


def _scrub_room_refs(rooms: list[Room], object_ids: set[str]) -> list[Room]:
    """Drop key_object and prerequisite entries that reference unknown object ids."""
    def _valid(p: Prerequisite) -> bool:
        if p.type in {"object_state", "has_item"}:
            return p.object_id in object_ids
        return True

    for room in rooms:
        room.key_objects = [k for k in room.key_objects if k in object_ids]
        room.prerequisites = [p for p in room.prerequisites if _valid(p)]
        if room.goal_completion is not None and not _valid(room.goal_completion):
            room.goal_completion = None
    return rooms


SINGLE_ROOM_MODE = True


def _build_world(data: dict) -> GameWorld:
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    if SINGLE_ROOM_MODE and rooms:
        kept = rooms[0]
        kept.adjacency = {}
        kept.prerequisites = []
        rooms = [kept]
    room_ids = {r.id for r in rooms}
    objects = _build_objects(data.get("objects", []), room_ids)
    object_ids = {o.id for o in objects}
    rooms = _scrub_room_refs(rooms, object_ids)

    rules = [r for r in data.get("rules", []) if isinstance(r, str)]
    solution_path = [s for s in data.get("solution_path", []) if isinstance(s, str)]

    return GameWorld(
        scenario=data.get("scenario", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
        objects=objects,
        rules=rules,
        win_condition=_build_win_condition(data.get("win_condition", {}), object_ids),
        solution_path=solution_path,
    )


def game_master_node(state: GameState) -> dict:
    llm = get_llm("game_master")
    prompt = GENERATION_PROMPT.format(theme=state.theme)

    storyline_start = time.perf_counter()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    storyline_elapsed = time.perf_counter() - storyline_start

    map_start = time.perf_counter()
    data = _parse_json(response.content) or {}
    world = _build_world(data)
    map_elapsed = time.perf_counter() - map_start

    print(
        f"[game_master] storyline (LLM): {storyline_elapsed:.2f}s | "
        f"map (parse+build): {map_elapsed:.2f}s",
        flush=True,
    )

    return {
        "messages": [AIMessage(content=response.content)],
        "world": world,
    }
