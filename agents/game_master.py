"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, GameWorld, PlayerState, Room, RoomItem

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}
MAX_ITEMS = 2


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


def _build_rooms(raw_rooms: list[dict]) -> list[Room]:
    rooms: list[Room] = []
    for raw_room in raw_rooms:
        room_items = [
            RoomItem(
                name=i.get("name", "Unknown"),
                description=i.get("description", ""),
            )
            for i in raw_room.get("items", [])
            if isinstance(i, dict)
        ]
        raw_adj = raw_room.get("adjacency", {})
        adjacency = raw_adj if isinstance(raw_adj, dict) else {}
        rooms.append(
            Room(
                name=raw_room.get("name", "Unnamed Room"),
                description=raw_room.get("description", ""),
                adjacency=adjacency,
                items=room_items,
            )
        )
    return rooms


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    """
    Three-pass repair on the LLM-generated room list:
      1. Strip adjacency references to rooms that don't exist.
      2. Mirror missing reverse edges (A→east→B but B missing west→A).
      3. Enforce MAX_ITEMS per room.
    """
    known = {r.name for r in rooms}

    cleaned: dict[str, dict[str, str]] = {}
    for room in rooms:
        cleaned[room.name] = {
            d: n for d, n in room.adjacency.items()
            if n in known and d in OPPOSITES
        }

    for room_name, adj in list(cleaned.items()):
        for direction, neighbor_name in adj.items():
            reverse = OPPOSITES[direction]
            neighbor_adj = cleaned.get(neighbor_name, {})
            if reverse not in neighbor_adj:
                neighbor_adj[reverse] = room_name
                cleaned[neighbor_name] = neighbor_adj

    return [
        Room(
            name=r.name,
            description=r.description,
            adjacency=cleaned.get(r.name, {}),
            items=r.items[:MAX_ITEMS],
        )
        for r in rooms
    ]


def _build_world(data: dict) -> GameWorld:
    narrative = data.get("narrative", {}) or {}
    rooms = _repair_adjacency(_build_rooms(data.get("room_layout", [])))
    return GameWorld(
        title=narrative.get("title", ""),
        setup=narrative.get("setup", ""),
        atmosphere=narrative.get("atmosphere", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
    )


def _initial_player_state(world: GameWorld) -> PlayerState:
    if not world.rooms:
        return PlayerState()
    starting_room = world.rooms[0].name
    return PlayerState(
        current_room=starting_room,
        visited={starting_room},
        items_remaining={r.name: list(r.items) for r in world.rooms},
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
    player = _initial_player_state(world)

    return {
        "messages": [AIMessage(content=response.content)],
        "world": world,
        "player": player,
    }
