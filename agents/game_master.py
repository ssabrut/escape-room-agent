"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Room, WinCondition, WorldObject

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
        rooms.append(
            Room(
                id=room_id,
                description=raw_room.get("description", ""),
                adjacency=adjacency,
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
        Room(id=r.id, description=r.description, adjacency=cleaned.get(r.id, {}))
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


def _build_world(data: dict) -> GameWorld:
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    room_ids = {r.id for r in rooms}
    objects = _build_objects(data.get("objects", []), room_ids)
    object_ids = {o.id for o in objects}

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
    llm = get_llm()
    prompt = GENERATION_PROMPT.format(theme=state.theme)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    data = _parse_json(response.content) or {}
    world = _build_world(data)

    return {
        "messages": [AIMessage(content=response.content)],
        "world": world,
    }
