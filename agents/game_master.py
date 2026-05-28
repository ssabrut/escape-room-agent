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


def game_master_node(state: GameState) -> dict:
    llm = get_llm()

    prompt = GENERATION_PROMPT.format(theme=state.theme)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    rooms = _extract_room_layout(response.content)

    return {
        "messages": [AIMessage(content=response.content)],
        "room_layout": rooms,
    }
