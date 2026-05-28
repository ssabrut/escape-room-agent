"""Mission Master agent — generates one interactive mission per gate in the game flow."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, Mission

SYSTEM_PROMPT = load_prompt("mission_master", "system")
GENERATION_PROMPT = load_prompt("mission_master", "generation")


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


def _build_missions(data: dict, known_rooms: set[str], items_by_room: dict[str, set[str]], known_room_list: list[str]) -> list[Mission]:
    missions: list[Mission] = []
    for raw in data.get("missions", []):
        if not isinstance(raw, dict):
            continue

        room = raw.get("room", "")
        if room not in known_rooms:
            continue

        reward_item = raw.get("reward_item", "") or ""
        if reward_item and reward_item not in items_by_room.get(room, set()):
            reward_item = next(iter(items_by_room.get(room, set())), "")

        unlocks_exit_to = raw.get("unlocks_exit_to", "") or ""
        if unlocks_exit_to != "VICTORY" and unlocks_exit_to not in known_rooms:
            unlocks_exit_to = "VICTORY"

        required_actions = raw.get("required_actions", [])
        if not isinstance(required_actions, list):
            required_actions = []
        required_actions = [str(a) for a in required_actions if a]

        missions.append(
            Mission(
                room=room,
                gate_index=int(raw.get("gate_index", 0)),
                description=raw.get("description", ""),
                required_actions=required_actions,
                reward_item=reward_item,
                unlocks_exit_to=unlocks_exit_to,
            )
        )

    missions.sort(key=lambda m: m.gate_index)
    return missions


def _format_rooms_and_items(world) -> str:
    lines = []
    for room in world.rooms:
        item_names = ", ".join(i.name for i in room.items) if room.items else "(no items)"
        lines.append(f"  - {room.name}: {item_names}")
    return "\n".join(lines)


def _format_gates(world) -> str:
    lines = []
    for i, gate in enumerate(world.game_flow.gates):
        req = gate.requires or "none"
        lines.append(f"  Gate {i}: room={gate.room}, requires={req}, unlocks={gate.unlocks}")
    return "\n".join(lines) if lines else "  (no gates)"


def mission_master_node(state: GameState) -> dict:
    llm = get_llm()
    world = state.world

    if not world or not world.game_flow.gates:
        return {"missions": []}

    known_rooms = {r.name for r in world.rooms}
    known_room_list = [r.name for r in world.rooms]
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
    missions = _build_missions(data, known_rooms, items_by_room, known_room_list)

    return {
        "messages": [AIMessage(content=response.content)],
        "missions": missions,
    }
