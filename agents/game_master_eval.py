"""Game Master in-loop adjudication — decides whether the party may leave a room.

The world-generation Game Master (`game_master.py`) runs once up front. This module
is the GM's *runtime* role: every time a player picks `go <room>`, the gameplay loop
asks the GM to evaluate the current room's `goal_completion`. The verdict is decided
deterministically; the GM LLM only writes the narration. If the LLM or its JSON fails,
a templated fallback keeps the game moving.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config.settings import get_llm
from prompts import load_prompt
from state import GameWorld, PartyState, Room

SYSTEM_PROMPT = load_prompt("game_master", "system")
EVALUATION_PROMPT = load_prompt("game_master", "evaluation")


class GMVerdict(BaseModel):
    """The Game Master's ruling on a single exit attempt."""

    allow: bool
    narration: str
    missing: str = ""


def _parse_json(text: str) -> dict | None:
    fence_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else text.strip()
    for candidate in (json_str, text.strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            return None
    return None


def _narrate(
    room: Room,
    dest_room: str,
    completion_str: str,
    satisfied: bool,
    missing: str,
) -> str:
    """Ask the GM LLM for in-world narration of the (already-decided) verdict."""
    verdict = "ALLOW" if satisfied else "BLOCK"
    prompt = EVALUATION_PROMPT.format(
        room_id=room.id,
        room_goal=room.goal or "(no specific goal)",
        dest_room=dest_room,
        goal_completion=completion_str,
        satisfied="yes" if satisfied else "no",
        missing=missing or "(nothing)",
        verdict=verdict,
    )
    try:
        llm = get_llm("game_master")
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        data = _parse_json(response.content) or {}
        narration = str(data.get("narration", "")).strip()
        if narration:
            return narration
    except Exception:
        pass

    # Templated fallback — keeps the loop deterministic if the LLM is unavailable.
    if satisfied:
        return f"The way to {dest_room} opens. The party steps through."
    return f"The way to {dest_room} stays sealed — {missing or 'the room goal is not yet done'}."


def evaluate_room_exit(
    world: GameWorld,
    ps: PartyState,
    room: Room,
    dest_room: str,
    satisfied: bool,
    completion_str: str,
) -> GMVerdict:
    """Adjudicate an attempt to leave `room` for `dest_room`.

    Decision is the caller-supplied `satisfied` (deterministic). The GM narrates it.
    A room with no completion condition is always allowed.
    """
    if room.goal_completion is None:
        return GMVerdict(
            allow=True,
            narration=f"Nothing holds the party here. They head toward {dest_room}.",
        )

    missing = "" if satisfied else f"still need: {completion_str}"
    narration = _narrate(room, dest_room, completion_str, satisfied, missing)
    return GMVerdict(allow=satisfied, narration=narration, missing=missing)
