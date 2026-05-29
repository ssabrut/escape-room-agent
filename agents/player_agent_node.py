"""Player Agent — picks one character from the roster.

Two instances run sequentially so agent 2 cannot pick the same character as agent 1.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import Character, GameState, PartyMember

SYSTEM_PROMPT = load_prompt("player_agent", "system")
SELECTION_PROMPT = load_prompt("player_agent", "selection")


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


def _format_characters(characters: list[Character]) -> str:
    lines = []
    for i, c in enumerate(characters, 1):
        lines.append(f"  [{i}] {c.name} — {c.role}")
        lines.append(f"       Backstory: {c.backstory}")
        ab = c.ability
        uses = "passive" if ab.max_uses < 0 else f"{ab.max_uses} use(s)"
        lines.append(
            f"       Ability: {ab.name} [{ab.effect}, {uses}] — {ab.description}"
        )
    return "\n".join(lines)


def _format_teammate_context(party: list[PartyMember]) -> str:
    if not party:
        return "You are picking first. No teammate has chosen yet."
    lines = ["Your teammate has already chosen:"]
    for member in party:
        lines.append(
            f"  - {member.agent_id}: {member.character.name} ({member.character.role})"
        )
    lines.append("Pick a character whose strengths complement your teammate's choice.")
    return "\n".join(lines)


def _select_character(agent_id: str, state: GameState) -> PartyMember | None:
    world = state.world

    taken_names = {member.character.name for member in state.party}
    available = [c for c in state.characters if c.name not in taken_names]
    if not available:
        return None

    llm = get_llm()
    prompt = SELECTION_PROMPT.format(
        agent_id=agent_id,
        scenario=world.scenario if world else "",
        objective=world.objective if world else "",
        available_characters=_format_characters(available),
        teammate_context=_format_teammate_context(state.party),
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    data = _parse_json(response.content) or {}
    chosen_name = data.get("chosen_character_name", "")
    reasoning = data.get("reasoning", "")

    chosen = next((c for c in available if c.name == chosen_name), None)
    if chosen is None:
        chosen = available[0]
        reasoning = reasoning or "(fallback) selected first available character"

    return PartyMember(
        agent_id=agent_id,
        character=chosen,
        reasoning=reasoning,
    )


def _make_player_node(agent_id: str):
    def node(state: GameState) -> dict:
        if not state.characters:
            return {"party": state.party}

        member = _select_character(agent_id, state)
        if member is None:
            return {"party": state.party}

        new_party = state.party + [member]
        return {
            "messages": [
                AIMessage(
                    content=f"[{agent_id}] chose {member.character.name}: {member.reasoning}"
                )
            ],
            "party": new_party,
        }

    node.__name__ = f"player_{agent_id}_node"
    return node


player_agent_1_node = _make_player_node("agent_1")
player_agent_2_node = _make_player_node("agent_2")
