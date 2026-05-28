"""Puzzle Master agent — generates one riddle per gate in the game flow."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, Puzzle

SYSTEM_PROMPT = load_prompt("puzzle_master", "system")
GENERATION_PROMPT = load_prompt("puzzle_master", "generation")


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


def _build_puzzles(data: dict, known_rooms: set[str], items_by_room: dict[str, set[str]]) -> list[Puzzle]:
    """Parse and repair the puzzles array from raw LLM data."""
    puzzles: list[Puzzle] = []
    for raw in data.get("puzzles", []):
        if not isinstance(raw, dict):
            continue

        room = raw.get("room", "")
        if room not in known_rooms:
            continue  # drop puzzles referencing non-existent rooms

        unlocks_item = raw.get("unlocks_item", "") or ""
        # Repair: if unlocks_item references an item not in that room, clear it
        if unlocks_item and unlocks_item not in items_by_room.get(room, set()):
            unlocks_item = ""

        puzzles.append(
            Puzzle(
                room=room,
                gate_index=int(raw.get("gate_index", 0)),
                riddle=raw.get("riddle", ""),
                answer=raw.get("answer", ""),
                clue_on_solve=raw.get("clue_on_solve", ""),
                unlocks_item=unlocks_item,
            )
        )

    # Sort by gate_index so they're always in flow order
    puzzles.sort(key=lambda p: p.gate_index)
    return puzzles


def _format_rooms_and_items(world) -> str:
    lines = []
    for room in world.rooms:
        item_names = ", ".join(i.name for i in room.items) if room.items else "(no items)"
        lines.append(f"  - {room.name}: {item_names}")
    return "\n".join(lines)


def _format_gates(world) -> str:
    lines = []
    for i, gate in enumerate(world.game_flow.gates):
        if gate.unlocks == "VICTORY":
            continue  # no puzzle needed for the final victory gate
        req = gate.requires or "none"
        lines.append(f"  Gate {i}: room={gate.room}, requires={req}, unlocks={gate.unlocks}")
    return "\n".join(lines) if lines else "  (no gates)"


def puzzle_master_node(state: GameState) -> dict:
    llm = get_llm()
    world = state.world

    if not world or not world.game_flow.gates:
        return {"puzzles": []}

    known_rooms = {r.name for r in world.rooms}
    items_by_room = {r.name: {i.name for i in r.items} for r in world.rooms}

    prompt = GENERATION_PROMPT.format(
        title=world.title,
        theme=state.theme,
        objective=world.objective,
        rooms_and_items=_format_rooms_and_items(world),
        gates=_format_gates(world),
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    data = _parse_json(response.content) or {}
    puzzles = _build_puzzles(data, known_rooms, items_by_room)

    return {
        "messages": [AIMessage(content=response.content)],
        "puzzles": puzzles,
    }
