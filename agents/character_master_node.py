"""Character Master agent — generates a roster of playable characters."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import Character, GameState

SYSTEM_PROMPT = load_prompt("character_master", "system")
GENERATION_PROMPT = load_prompt("character_master", "generation")


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


def _build_characters(data: dict) -> list[Character]:
    characters = []
    for raw in data.get("characters", []):
        if not isinstance(raw, dict):
            continue
        characters.append(
            Character(
                name=raw.get("name", "Unknown"),
                role=raw.get("role", ""),
                backstory=raw.get("backstory", ""),
            )
        )
    return characters


def character_master_node(state: GameState) -> dict:
    llm = get_llm("game_master")
    world = state.world

    room_names = ", ".join(r.id for r in world.rooms) if world else ""

    prompt = GENERATION_PROMPT.format(
        scenario=world.scenario if world else "",
        theme=state.theme,
        objective=world.objective if world else "",
        room_names=room_names,
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    data = _parse_json(response.content) or {}
    characters = _build_characters(data)

    return {
        "messages": [AIMessage(content=response.content)],
        "characters": characters,
    }
