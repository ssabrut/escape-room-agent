"""Character Master agent — generates a roster of playable characters."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import Character, GameState
from state.game_state import ABILITY_EFFECTS, ABILITY_TRIGGERS, Ability

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


def _build_ability(raw: dict | None, character_name: str) -> Ability:
    raw = raw if isinstance(raw, dict) else {}

    effect = raw.get("effect", "")
    if effect not in ABILITY_EFFECTS:
        effect = "spot_clue"

    trigger = raw.get("trigger", "")
    if trigger not in ABILITY_TRIGGERS:
        trigger = "passive" if effect in {"negate_hazard", "spot_clue"} else "on_action"

    max_uses_raw = raw.get("max_uses", 1)
    try:
        max_uses = int(max_uses_raw)
    except (TypeError, ValueError):
        max_uses = 1
    if max_uses == 0:
        max_uses = 1

    return Ability(
        name=raw.get("name") or f"{character_name}'s Talent",
        description=raw.get("description", ""),
        trigger=trigger,
        effect=effect,
        target=raw.get("target"),
        max_uses=max_uses,
        uses_remaining=max_uses,
    )


def _build_characters(data: dict) -> list[Character]:
    characters = []
    for raw in data.get("characters", []):
        if not isinstance(raw, dict):
            continue
        name = raw.get("name", "Unknown")
        characters.append(
            Character(
                name=name,
                role=raw.get("role", ""),
                backstory=raw.get("backstory", ""),
                ability=_build_ability(raw.get("ability"), name),
            )
        )
    return characters


def character_master_node(state: GameState) -> dict:
    llm = get_llm()
    world = state.world

    room_names = ", ".join(r.name for r in world.rooms) if world else ""

    prompt = GENERATION_PROMPT.format(
        title=world.title if world else "",
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
