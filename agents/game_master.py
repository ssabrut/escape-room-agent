"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, Room, RoomItem

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")


def _extract_room_layout(text: str) -> list[Room]:
    # Tier 1: extract from ```json ... ``` fence
    fence_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else text.strip()

    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Tier 2: try the full raw text directly
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            pass

    if data is None:
        # Tier 3: grab the first {...} block as last resort
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    rooms: list[Room] = []
    for raw_room in data.get("room_layout", []):
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


OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}
MAX_ITEMS = 2


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    """
    Three-pass repair on the LLM-generated room list:
      1. Strip adjacency references to rooms that don't exist.
      2. Mirror missing reverse edges (A→east→B but B missing west→A).
      3. Enforce MAX_ITEMS per room.
    Returns new Room objects; does not mutate in place.
    """
    known = {r.name for r in rooms}

    # Pass 1 — drop references to unknown rooms
    cleaned: dict[str, dict[str, str]] = {}
    for room in rooms:
        cleaned[room.name] = {
            d: n for d, n in room.adjacency.items()
            if n in known and d in OPPOSITES
        }

    # Pass 2 — mirror missing reverse edges
    for room_name, adj in list(cleaned.items()):
        for direction, neighbor_name in adj.items():
            reverse = OPPOSITES[direction]
            neighbor_adj = cleaned.get(neighbor_name, {})
            if reverse not in neighbor_adj:
                neighbor_adj[reverse] = room_name
                cleaned[neighbor_name] = neighbor_adj

    # Pass 3 — rebuild Room objects with repaired adjacency and capped items
    return [
        Room(
            name=r.name,
            description=r.description,
            adjacency=cleaned.get(r.name, {}),
            items=r.items[:MAX_ITEMS],
        )
        for r in rooms
    ]


def game_master_node(state: GameState) -> dict:
    llm = get_llm()

    prompt = GENERATION_PROMPT.format(theme=state.theme)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    rooms = _repair_adjacency(_extract_room_layout(response.content))

    return {
        "messages": [AIMessage(content=response.content)],
        "room_layout": rooms,
    }
