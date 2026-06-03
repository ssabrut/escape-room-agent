"""Game Master in-loop adjudication — decides whether the party may leave a room.

The world-generation Game Master (`game_master.py`) runs once up front. This module
is the GM's *runtime* role: after each gameplay tick, ``game_master_eval_node`` checks
whether the current room's local goal is met, narrates the verdict, and decides whether
the graph should loop back to gameplay or terminate.  The verdict is always decided
deterministically; the LLM only writes the narration.  If the LLM or its JSON fails,
a templated fallback keeps the game moving.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, GameWorld, PartyState, Room, TickAction

SYSTEM_PROMPT = load_prompt("game_master", "system")
EVALUATION_PROMPT = load_prompt("game_master", "evaluation")
DIRECTIVE_PROMPT = load_prompt("game_master", "directive")


class GMVerdict(BaseModel):
    """The Game Master's ruling on a single exit attempt."""

    allow: bool
    narration: str
    missing: str = ""


class GMDirective(BaseModel):
    """The Game Master's proactive prompt to move on once a room's goal is done."""

    dest_room: str
    narration: str


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


def announce_room_cleared(
    world: GameWorld,
    ps: PartyState,
    room: Room,
    dest_room: str,
    completion_str: str,
) -> GMDirective:
    """The GM speaks up once `room`'s goal is done, directing the party onward.

    Called proactively at the start of a tick (not in response to a `go` action).
    The destination is decided deterministically by the caller; the GM narrates
    the order to move to `dest_room`. A templated fallback keeps the loop running
    if the LLM is unavailable.
    """
    prompt = DIRECTIVE_PROMPT.format(
        room_id=room.id,
        room_goal=room.goal or "(no specific goal)",
        dest_room=dest_room,
        goal_completion=completion_str or "(no condition)",
    )
    try:
        llm = get_llm("game_master")
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        data = _parse_json(response.content) or {}
        narration = str(data.get("narration", "")).strip()
        if narration:
            return GMDirective(dest_room=dest_room, narration=narration)
    except Exception:
        pass

    return GMDirective(
        dest_room=dest_room,
        narration=(
            f"The goal of {room.id} is done. Move on to {dest_room} — "
            "there is nothing more for you here."
        ),
    )


# ---------------------------------------------------------------------------
# Rendering helpers (self-contained to avoid circular imports with gameplay_node)
# ---------------------------------------------------------------------------

_PANEL_WIDTH = 94


def _gm_stream(line: str = "") -> None:
    print(line, flush=True)


def _gm_banner(title: str, char: str = "=") -> None:
    line = char * _PANEL_WIDTH
    pad = max(0, _PANEL_WIDTH - len(title) - 4)
    _gm_stream("\n" + line)
    _gm_stream(f"{char}{char} {title}{' ' * pad}{char}{char}")
    _gm_stream(line)


def _gm_panel(label: str, body_lines: list[str]) -> None:
    inner = _PANEL_WIDTH - 2
    top = f"+- {label} " + "-" * max(0, inner - len(label) - 3) + "+"
    bottom = "+" + "-" * inner + "+"
    _gm_stream(top)
    for line in body_lines:
        _gm_stream(f"| {line[:inner - 2].ljust(inner - 2)} |")
    _gm_stream(bottom)


def _render_eval_directive(room_id: str, directive: GMDirective) -> None:
    _gm_panel(
        "GAME MASTER",
        [
            f"ADVANCE -> goal of {room_id} complete; head to {directive.dest_room}",
            f'"{directive.narration}"',
        ],
    )


def _render_eval_final(ps: PartyState, world: GameWorld, max_ticks: int) -> None:
    _gm_banner("FINAL RESULT")
    outcome = "VICTORY" if ps.victory else f"ENDED (final room: {ps.current_room})"
    inv = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    known = ", ".join(ps.known_info) if ps.known_info else "(none)"
    win_obj_state = (
        ps.object_states.get(world.win_condition.object_id, "?")
        if world.win_condition.object_id
        else "?"
    )
    _gm_panel(
        "SUMMARY",
        [
            f"Outcome      : {outcome}",
            f"Ticks used   : {ps.tick} / {max_ticks}",
            f"Inventory    : {inv}",
            f"Known clues  : {known}",
            f"Visited      : {', '.join(sorted(ps.visited))}",
            f"Win object   : {world.win_condition.object_id} "
            f"(state: {win_obj_state}, target: {world.win_condition.state})",
        ],
    )
    _gm_stream()


# ---------------------------------------------------------------------------
# Helpers shared with gameplay_node (imported there via this module)
# ---------------------------------------------------------------------------

_OPENED_STATES = {"unlocked", "open", "opened", "unsealed", "dissolved", "deactivated"}


def _state_satisfies(actual: str | None, target: str | None) -> bool:
    if actual == target:
        return True
    if actual is None or target is None:
        return False
    return actual in _OPENED_STATES and target in _OPENED_STATES


def _check_victory(world: GameWorld, ps: PartyState) -> bool:
    win = world.win_condition
    if not win.object_id:
        return False
    return _state_satisfies(ps.object_states.get(win.object_id), win.state)


def _goal_completion_satisfied_eval(completion, ps: PartyState) -> bool:
    """Mirror of gameplay_node._goal_completion_satisfied without the import cycle."""
    if completion is None:
        return True
    t = completion.type
    if t == "object_state":
        return _state_satisfies(
            ps.object_states.get(completion.object_id or ""),
            completion.state,
        )
    if t == "has_item":
        return (completion.object_id or "") in ps.inventory
    if t == "known_info":
        return (completion.info or "") in ps.known_info
    if t == "power_active":
        return (completion.id or "") in ps.power_active
    return False


def _next_room_from_world(world: GameWorld, current_room_id: str) -> str | None:
    """Return the next room in world.rooms order after current_room_id, or None."""
    ids = [r.id for r in world.rooms]
    try:
        idx = ids.index(current_room_id)
        return ids[idx + 1] if idx + 1 < len(ids) else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

MAX_TICKS = 40  # must match gameplay_node.MAX_TICKS


def game_master_eval_node(state: GameState) -> dict:
    """Observe the result of one gameplay tick and decide what happens next.

    - Victory: set game_over + victory, render final summary, signal END.
    - Time-up: set game_over, render final summary, signal END.
    - Room goal met: narrate via LLM, inject GM directive into the log so the
      next gameplay tick shows it, loop back to gameplay.
    - Otherwise: loop back to gameplay silently.

    The routing decision is encoded in ``party_state.game_over`` so the
    conditional edge in graph.py can branch without extra state fields.
    """
    world = state.world
    if not world or not state.party_state:
        return {}

    ps = state.party_state
    new_messages: list[AIMessage] = []

    # --- Victory check ---
    if _check_victory(world, ps):
        ps.game_over = True
        ps.victory = True
        _gm_banner("VICTORY", char="*")
        _gm_stream(f"  Party achieved the win condition at tick {ps.tick}.")
        new_messages.append(AIMessage(content=f"[game_master_eval] VICTORY at tick {ps.tick}"))
        _render_eval_final(ps, world, MAX_TICKS)
        return {"messages": new_messages, "party_state": ps}

    # --- Time-up check ---
    if ps.tick >= MAX_TICKS:
        ps.game_over = True
        _gm_banner("TIME UP", char="*")
        _gm_stream(f"  Reached MAX_TICKS={MAX_TICKS} without satisfying the win condition.")
        new_messages.append(AIMessage(content=f"[game_master_eval] Stopped at MAX_TICKS={MAX_TICKS}"))
        _render_eval_final(ps, world, MAX_TICKS)
        return {"messages": new_messages, "party_state": ps}

    # --- Local room goal check ---
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    if room is not None and room.goal_completion is not None:
        if _goal_completion_satisfied_eval(room.goal_completion, ps):
            # If the goal was already done at this tick's START, gameplay_node
            # already computed + rendered the directive (steering the agents
            # in-tick) and stashed it here, using the graph-correct destination.
            # Reuse it — no second LLM call, no duplicate panel. Only when the goal
            # became done DURING this tick (no pending directive) do we narrate it
            # fresh, falling back to world-order for the destination.
            pending = ps.pending_directive
            if pending is not None:
                directive = GMDirective(dest_room=pending[0], narration=pending[1])
                dest = directive.dest_room
                rendered_in_tick = True
            else:
                dest = _next_room_from_world(world, ps.current_room)
                directive = None
                rendered_in_tick = False

            if dest is not None:
                if directive is None:
                    completion_str = str(
                        room.goal_completion.model_dump(exclude_none=True)
                    )
                    directive = announce_room_cleared(
                        world, ps, room, dest, completion_str
                    )

                if not rendered_in_tick:
                    _render_eval_directive(ps.current_room, directive)
                ps.log.append(
                    TickAction(
                        tick=ps.tick,
                        agent_id="game_master",
                        say=directive.narration,
                        action="gm_directive",
                        note=f"advance to {directive.dest_room}",
                    )
                )
                new_messages.append(
                    AIMessage(
                        content=f"[game_master_eval] room {ps.current_room} cleared — advance to {dest}"
                    )
                )

    # Consume the in-tick directive so a later tick can't reuse a stale one.
    ps.pending_directive = None

    return {"messages": new_messages, "party_state": ps}


def route_after_eval(state: GameState) -> str:
    """Conditional edge: END when game is over, otherwise loop back to gameplay."""
    if state.party_state and state.party_state.game_over:
        return "end"
    return "gameplay"
