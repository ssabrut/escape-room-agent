"""Gameplay node — co-op loop driven by the object-graph world model.

Each tick every party member picks one action from a discrete, mechanically
generated action space. The engine resolves the action against the target
object's preconditions and mutates runtime state in party_state. Victory fires
when the win_condition object reaches its target state.
"""

from __future__ import annotations

import json
import re
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.escape_rooms.utils.settings import get_llm
from src.escape_rooms.prompts import load_prompt
from src.escape_rooms.state import (
    GameState,
    GameWorld,
    ObjectObservation,
    PartyMember,
    PartyState,
    TickAction,
    WorldObject,
)
from src.escape_rooms.utils.renderer import render_room_layout

SYSTEM_PROMPT = load_prompt("gameplay_agent", "system")
ACTION_PROMPT = load_prompt("gameplay_agent", "action")
OBSERVE_PROMPT = load_prompt("gameplay_agent", "observe")
PLAN_PROPOSE_PROMPT = load_prompt("gameplay_agent", "plan_propose")
PLAN_CRITIQUE_PROMPT = load_prompt("gameplay_agent", "plan_critique")
PLAN_SYNTHESIZE_PROMPT = load_prompt("gameplay_agent", "plan_synthesize")

MAX_TICKS = 40
IDLE_ACTION = "wait"

# Number of critique/revise rounds in the planning debate (0 = propose+synthesize
# only). Each round costs one LLM call per agent, so keep this small.
DEBATE_ROUNDS = 1

# Log entries that record observation/planning passes rather than a real move.
# Excluded from teammate "last action" context and stall detection.
NON_GAMEPLAY_ACTIONS = {"observe", "plan", "reobserve", "replan", "gm_directive"}

HIDDEN_STATES = {"locked", "locked_bolt", "locked_room", "hidden"}
UNLOCKED_STATES = {"unlocked", "open", "visible"}

# "Puzzle solved" terminal states that a goal/win target may name. The resolvers
# always write "unlocked" when a lock is cleared, but the generator names the
# matching goal target with any synonym (open, opened, dissolved, ...). Treat
# them as equivalent so a correctly-cleared lock actually satisfies the goal.
# (Excludes "visible" — a merely-visible object is not a solved lock.)
OPENED_STATES = {"unlocked", "open", "opened", "unsealed", "dissolved", "deactivated"}


def _state_satisfies(actual: str | None, target: str | None) -> bool:
    """True if object state `actual` meets goal/win target `target`.

    Exact match, or both sides are in the "opened" synonym family — so a chest the
    resolver marked "unlocked" satisfies a goal/win that asked for "open".
    """
    if actual == target:
        return True
    if actual is None or target is None:
        return False
    return actual in OPENED_STATES and target in OPENED_STATES


# ---------- world graph ----------


class WorldGraph:
    def __init__(self, world: GameWorld) -> None:
        self._adj: dict[str, list[str]] = {}
        ids = {r.id for r in world.rooms}
        for room in world.rooms:
            self._adj[room.id] = [n for n in room.adjacency.values() if n in ids]

    def neighbors(self, room: str) -> list[str]:
        return list(self._adj.get(room, []))

    def path(self, src: str, dst: str) -> list[str]:
        if src == dst:
            return [src]
        if src not in self._adj or dst not in self._adj:
            return []
        parents: dict[str, str] = {src: src}
        queue: deque[str] = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                break
            for nxt in self._adj[cur]:
                if nxt in parents:
                    continue
                parents[nxt] = cur
                queue.append(nxt)
        if dst not in parents:
            return []
        out = [dst]
        while out[-1] != src:
            out.append(parents[out[-1]])
        out.reverse()
        return out


# ---------- helpers ----------

PANEL_WIDTH = 94


def _stream(line: str = "") -> None:
    print(line, flush=True)


def _wrap(text: str, width: int) -> list[str]:
    """Naive wrap on spaces, preserving short lines verbatim."""
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= width:
            current = f"{current} {w}"
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]


def _banner(title: str, char: str = "=") -> None:
    line = char * PANEL_WIDTH
    pad = max(0, PANEL_WIDTH - len(title) - 4)
    _stream("\n" + line)
    _stream(f"{char}{char} {title}{' ' * pad}{char}{char}")
    _stream(line)


def _panel(label: str, body_lines: list[str]) -> None:
    """Render a labeled box panel containing one or more body lines."""
    inner = PANEL_WIDTH - 2
    top = f"+- {label} " + "-" * max(0, inner - len(label) - 3) + "+"
    bottom = "+" + "-" * inner + "+"
    _stream(top)
    for line in body_lines:
        for wrapped in _wrap(line, inner - 2):
            _stream(f"| {wrapped.ljust(inner - 2)} |")
    _stream(bottom)


def _rule(label: str = "") -> None:
    if not label:
        _stream("-" * PANEL_WIDTH)
        return
    pad = PANEL_WIDTH - len(label) - 4
    _stream("-- " + label + " " + "-" * max(0, pad))


# Outcome status codes used to badge each action.
STATUS_OK = "OK"  # state changed (unlocked, took item, moved, info learned)
STATUS_INFO = "..."  # action ran but nothing changed (examine without new info, idle)
STATUS_FAIL = "XX"  # action invalid / missing precondition

_OK_KEYWORDS = (
    "unlocked",
    "took",
    "moved",
    "learned",
    "flipped",
    "inserted",
    "entered",
    "used",
)
_FAIL_KEYWORDS = (
    "missing",
    "no matching",
    "unknown",
    "cannot",
    "not accessible",
    "no direct",
    "no target",
    "no fuse",
    "dead end",
    "nothing new",
    "blocked",
)


def _classify_outcome(note: str) -> str:
    n = note.lower()
    if any(k in n for k in _FAIL_KEYWORDS):
        return STATUS_FAIL
    if any(k in n for k in _OK_KEYWORDS):
        return STATUS_OK
    return STATUS_INFO


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


def _as_bullets(value) -> list[str]:
    """Coerce an LLM JSON field into a clean list of bullet strings.

    Accepts a JSON array, or falls back to splitting a string on newlines /
    sentence boundaries. Strips any leading bullet/numbering markers.
    """
    items: list[str]
    if isinstance(value, list):
        items = [str(v) for v in value]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        items = re.split(r"\n+|(?<=[.;])\s+", text)
    else:
        return []
    out: list[str] = []
    for item in items:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_ ]+", " ", text.lower()).strip()


# ---------- object visibility ----------


def _object_visible(
    obj: WorldObject, ps: PartyState, parent_lookup: dict[str, WorldObject]
) -> bool:
    """An object is visible if it (or its container chain) resolves up to the current room."""
    state = ps.object_states.get(obj.id, obj.state)
    if state == "hidden":
        return False

    cursor = obj
    while True:
        if cursor.location == ps.current_room:
            return True
        parent = parent_lookup.get(cursor.location)
        if parent is None:
            return False
        parent_state = ps.object_states.get(parent.id, parent.state)
        if parent_state in HIDDEN_STATES:
            return False
        cursor = parent


def _objects_in_room(world: GameWorld, ps: PartyState) -> list[WorldObject]:
    parent_lookup = {o.id: o for o in world.objects}
    return [o for o in world.objects if _object_visible(o, ps, parent_lookup)]


# ---------- action space ----------


def _already_examined(obj: WorldObject, ps: PartyState) -> bool:
    """True once `obj` has been examined — re-examining can never reveal anything.

    `_resolve_examine` only yields new info on the first examine (it captures
    `contains_info` into `known_info` then); every later examine is a dead end.
    So once examined, the verb is dropped from the action space entirely rather
    than left in the menu for an agent to waste a tick on.
    """
    return any(
        e.action == f"examine {obj.id}" and e.note.startswith("examined")
        for e in ps.log
    )


def _code_satisfied_by(required: str, info: str) -> bool:
    """True if a known clue `info` supplies the code `required`.

    The generator names the door's `requires_code` with a bare stem (e.g.
    ``captain_combination``) but carries the clue as a decorated token that
    embeds the answer (e.g. ``captain_combination_8429``). Matching only on exact
    equality or the clue's last segment ("8429") misses that prefix case and
    leaves a solvable door permanently locked. So we accept a match when:
      - the tokens are equal, or
      - one token's stem is a prefix/substring of the other (covers the
        stem-vs-decorated case both directions), or
      - their digit sequences are equal and non-empty (covers code-as-number).
    """
    req, inf = required.lower(), info.lower()
    if req == inf:
        return True
    # Clue follows the "<required>_<answer>" convention: required is a prefix of
    # the clue token, on a token (underscore) boundary. Check both directions.
    if req and (inf.startswith(req + "_") or inf == req or req.startswith(inf + "_")):
        return True
    if required in info.split("_") or info in required.split("_"):
        return True
    req_digits = re.sub(r"\D", "", req)
    inf_digits = re.sub(r"\D", "", inf)
    return bool(req_digits) and req_digits == inf_digits


def _code_known(required: str, ps: PartyState) -> bool:
    """Mirror `_resolve_enter_code`'s success test: a known clue supplies the code."""
    return any(_code_satisfied_by(required, info) for info in ps.known_info)


def _liquid_available(
    required: str, ps: PartyState, by_id: dict[str, WorldObject]
) -> bool:
    """Mirror `_resolve_insert_liquid`: a held item matches the required liquid."""
    for held_id in ps.inventory:
        held = by_id.get(held_id)
        haystack = " ".join([held_id] + ([held.description] if held else []))
        if _liquid_token_matches(required, haystack):
            return True
    return False


def _verbs_for(
    obj: WorldObject, ps: PartyState, by_id: dict[str, WorldObject] | None = None
) -> list[str]:
    state = ps.object_states.get(obj.id, obj.state)
    by_id = by_id if by_id is not None else {}
    verbs: list[str] = []
    # Take before examine: pick up the object first, then examine from inventory.
    # Examine is suppressed on takeable objects that haven't been picked up yet,
    # so agents don't waste a tick examining in-world then taking next tick.
    # Non-takeable objects (and items already in inventory) are examined normally.
    can_take = obj.takeable and state in UNLOCKED_STATES and obj.id not in ps.inventory
    if can_take:
        verbs.append(f"take {obj.id}")
    if not can_take and not _already_examined(obj, ps):
        verbs.append(f"examine {obj.id}")
    # Only offer an unlock verb when its precondition is CURRENTLY satisfiable —
    # otherwise it's a guaranteed dead end ("missing tool", "code unknown", ...)
    # that agents would burn ticks retrying. Once they gather the tool/code/liquid
    # or bring power online, the verb reappears.
    if (
        obj.requires_code
        and state in HIDDEN_STATES
        and _code_known(obj.requires_code, ps)
    ):
        verbs.append(f"enter_code {obj.id}")
    if (
        obj.requires_tool
        and state in HIDDEN_STATES
        and obj.requires_tool in ps.inventory
    ):
        verbs.append(f"use_tool {obj.id}")
    if (
        obj.requires_liquid
        and state in HIDDEN_STATES
        and _liquid_available(obj.requires_liquid, ps, by_id)
    ):
        verbs.append(f"insert_liquid {obj.id}")
    if obj.fuses is not None:
        for label in obj.fuses:
            verbs.append(f"flip_fuse {obj.id} {label}")
    if (
        obj.requires_power
        and state in HIDDEN_STATES
        and obj.requires_power in ps.power_active
    ):
        verbs.append(f"open {obj.id}")
    return verbs


def _build_action_space(
    world: GameWorld, ps: PartyState, visible: list[WorldObject]
) -> list[str]:
    seen: set[str] = set()
    space: list[str] = []
    by_id = {o.id: o for o in world.objects}
    for obj in visible:
        for v in _verbs_for(obj, ps, by_id):
            if v not in seen:
                seen.add(v)
                space.append(v)
    current_room = next((r for r in world.rooms if r.id == ps.current_room), None)
    # Exits are always offered so the locked door is visible as an option; whether
    # the move actually succeeds is enforced in _resolve_action (the exit gate:
    # advancing to a new room needs this room's goal satisfied).
    if current_room:
        for direction, neighbor in current_room.adjacency.items():
            move = f"go {neighbor}"
            if move not in seen:
                seen.add(move)
                space.append(move)
    # Only offer 'wait' as a true last resort — when no productive action exists.
    # Otherwise agents idle instead of examining, applying clues, or moving.
    if not space:
        space.append(IDLE_ACTION)
    return space


# ---------- action resolution ----------


def _resolve_examine(obj: WorldObject, ps: PartyState) -> str:
    if obj.contains_info and obj.contains_info not in ps.known_info:
        ps.known_info.append(obj.contains_info)
        return f"examined {obj.id}; learned {obj.contains_info}"
    already = any(
        e.action == f"examine {obj.id}" and e.note.startswith("examined")
        for e in ps.log
    )
    if already:
        return f"examined {obj.id} again — nothing new (dead end)"
    if obj.contains_info:
        return f"examined {obj.id}; already knew {obj.contains_info}"
    return f"examined {obj.id} — no hidden info"


def _resolve_take(obj: WorldObject, ps: PartyState) -> str:
    if not obj.takeable:
        return f"cannot take {obj.id}"
    state = ps.object_states.get(obj.id, obj.state)
    if state in HIDDEN_STATES:
        return f"{obj.id} not accessible"
    if obj.id in ps.inventory:
        return f"{obj.id} already carried"
    ps.inventory.append(obj.id)
    return f"took {obj.id}"


def _resolve_enter_code(obj: WorldObject, ps: PartyState) -> str:
    if not obj.requires_code:
        return f"{obj.id} has no keypad"
    state = ps.object_states.get(obj.id, obj.state)
    if state not in HIDDEN_STATES:
        return f"{obj.id} already open"
    if _code_known(obj.requires_code, ps):
        ps.object_states[obj.id] = "unlocked"
        return f"entered code {obj.requires_code} → {obj.id} unlocked"
    return f"code unknown — examine clues first"


def _resolve_use_tool(obj: WorldObject, ps: PartyState) -> str:
    if not obj.requires_tool:
        return f"{obj.id} needs no tool"
    state = ps.object_states.get(obj.id, obj.state)
    if state not in HIDDEN_STATES:
        return f"{obj.id} already open"
    if obj.requires_tool not in ps.inventory:
        return f"missing tool {obj.requires_tool}"
    ps.object_states[obj.id] = "unlocked"
    return f"used {obj.requires_tool} on {obj.id} → unlocked"


def _liquid_token_matches(required: str, text: str) -> bool:
    """Match e.g. 'pH_7' against 'pH 7' or 'ph7' inside a description/id."""
    norm_req = re.sub(r"[^a-z0-9]+", "", required.lower())
    norm_text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return norm_req in norm_text


def _resolve_insert_liquid(
    obj: WorldObject, ps: PartyState, by_id: dict[str, WorldObject]
) -> str:
    if not obj.requires_liquid:
        return f"{obj.id} has no liquid slot"
    state = ps.object_states.get(obj.id, obj.state)
    if state not in HIDDEN_STATES:
        return f"{obj.id} already open"
    required = obj.requires_liquid
    for held_id in ps.inventory:
        held = by_id.get(held_id)
        haystack = " ".join([held_id] + ([held.description] if held else []))
        if _liquid_token_matches(required, haystack):
            ps.object_states[obj.id] = "unlocked"
            return f"inserted matching liquid ({required}) into {obj.id} → unlocked"
    return f"no matching liquid for {required}"


def _resolve_flip_fuse(obj: WorldObject, label: str, ps: PartyState) -> str:
    state = ps.object_states.get(obj.id, obj.state)
    if state in HIDDEN_STATES:
        return f"{obj.id} still locked"
    if obj.fuses is None or label not in obj.fuses:
        return f"no fuse {label} on {obj.id}"
    current = ps.fuse_states.setdefault(obj.id, dict(obj.fuses))
    new_state = "OFF" if current.get(label, "OFF") == "ON" else "ON"
    current[label] = new_state
    power_token = f"sekring_{label}_{new_state}"
    if new_state == "ON":
        ps.power_active.add(power_token)
        opposite = f"sekring_{label}_OFF"
        ps.power_active.discard(opposite)
    else:
        ps.power_active.discard(power_token)
    return f"flipped fuse {label} on {obj.id} → {new_state}"


def _resolve_open(obj: WorldObject, ps: PartyState) -> str:
    if not obj.requires_power:
        return f"{obj.id} needs no power"
    state = ps.object_states.get(obj.id, obj.state)
    if state not in HIDDEN_STATES:
        return f"{obj.id} already open"
    if obj.requires_power in ps.power_active:
        ps.object_states[obj.id] = "unlocked"
        return f"power {obj.requires_power} satisfied → {obj.id} unlocked"
    return f"missing power {obj.requires_power}"


def _goal_completion_satisfied(completion, ps: PartyState) -> bool:
    """Check if a goal_completion prerequisite is satisfied."""
    if completion is None:
        return False
    if completion.type == "object_state":
        return bool(completion.object_id) and _state_satisfies(
            ps.object_states.get(completion.object_id), completion.state
        )
    if completion.type == "known_info":
        return bool(completion.info) and completion.info in ps.known_info
    if completion.type == "has_item":
        return bool(completion.object_id) and completion.object_id in ps.inventory
    if completion.type == "power_active":
        return bool(completion.id) and completion.id in ps.power_active
    return False


def _format_goal_completion(completion) -> str:
    """Format a goal_completion for display."""
    if completion is None:
        return "(no condition)"
    if completion.type == "object_state":
        return f"{completion.object_id} → {completion.state}"
    if completion.type == "known_info":
        return f"learn '{completion.info}'"
    if completion.type == "has_item":
        return f"carry {completion.object_id}"
    if completion.type == "power_active":
        return f"power {completion.id} on"
    return completion.type


def _resolve_action(
    action: str, world: GameWorld, ps: PartyState
) -> tuple[str, str | None]:
    """Return (outcome note, target_object_id or None)."""
    parts = action.split()
    if not parts:
        return ("idle", None)
    verb = parts[0]

    by_id = {o.id: o for o in world.objects}
    rooms_by_id = {r.id: r for r in world.rooms}

    if verb == IDLE_ACTION:
        return ("idle", None)
    if verb == "go" and len(parts) >= 2:
        dest = parts[1]
        if dest in rooms_by_id:
            current = rooms_by_id.get(ps.current_room)
            if current and dest in current.adjacency.values():
                # Exit gate: advancing to a NOT-yet-visited room requires the
                # current room's goal to be satisfied — its locked door must be
                # opened first. Backtracking to an already-visited room is always
                # allowed, so the party can never softlock itself.
                if (
                    dest not in ps.visited
                    and current.goal_completion is not None
                    and not _goal_completion_satisfied(current.goal_completion, ps)
                ):
                    needs = _format_goal_completion(current.goal_completion)
                    return (
                        f"the way to {dest} is locked — {current.goal} ({needs})",
                        None,
                    )
                ps.current_room = dest
                ps.visited.add(dest)
                return (f"moved to {dest}", None)
            return (f"no direct route to {dest}", None)
        return (f"unknown room {dest}", None)

    if len(parts) < 2:
        return ("no target", None)
    target_id = parts[1]
    obj = by_id.get(target_id)
    if obj is None:
        return (f"unknown object {target_id}", None)

    if verb == "examine":
        return (_resolve_examine(obj, ps), obj.id)
    if verb == "take":
        return (_resolve_take(obj, ps), obj.id)
    if verb == "enter_code":
        return (_resolve_enter_code(obj, ps), obj.id)
    if verb == "use_tool":
        return (_resolve_use_tool(obj, ps), obj.id)
    if verb == "insert_liquid":
        return (_resolve_insert_liquid(obj, ps, by_id), obj.id)
    if verb == "flip_fuse" and len(parts) >= 3:
        return (_resolve_flip_fuse(obj, parts[2], ps), obj.id)
    if verb == "open":
        return (_resolve_open(obj, ps), obj.id)
    return (f"unknown verb {verb}", obj.id)


# ---------- llm action selection ----------


def _resolve_choice(response: str, space: list[str]) -> str:
    text = response.strip()
    idx_match = re.match(r"^\s*(\d+)\s*$", text)
    if idx_match:
        i = int(idx_match.group(1)) - 1
        if 0 <= i < len(space):
            return space[i]
    norm = _normalize(text)
    for option in space:
        if _normalize(option) == norm:
            return option
    # Unparseable reply: prefer idling if offered, else the first productive option.
    if IDLE_ACTION in space:
        return IDLE_ACTION
    return space[0] if space else IDLE_ACTION


def _requirement_hint(obj: WorldObject) -> str:
    """A type-only hint of what an object needs to open — never names the answer.

    Tells the agent an object needs *a* tool / code / liquid / power so it stops
    fumbling blindly, while preserving discovery (it must still find the right
    tool/code itself). Empty string when the object gates on nothing.
    """
    needs: list[str] = []
    if obj.requires_code:
        if obj.code_digits:
            needs.append(f"a {obj.code_digits}-digit code")
        else:
            needs.append("a code")
    if obj.requires_tool:
        needs.append("a tool")
    if obj.requires_liquid:
        needs.append("a liquid")
    if obj.requires_power:
        needs.append("power")
    if not needs:
        return ""
    return " (needs " + " + ".join(needs) + ")"


def _format_objects(visible: list[WorldObject], ps: PartyState) -> str:
    if not visible:
        return "  (none visible)"
    lines = []
    for o in visible:
        state = ps.object_states.get(o.id, o.state)
        # Only hint requirements while the object is still locked/hidden — once
        # open there is nothing left to gate on.
        hint = _requirement_hint(o) if state in HIDDEN_STATES else ""
        lines.append(f"  - {o.id} [{state}]: {o.description}{hint}")
    return "\n".join(lines)


HISTORY_WINDOW = 6
ACTION_LOG_WINDOW = 12


def _format_agent_history(
    ps: PartyState, agent_id: str, limit: int = HISTORY_WINDOW
) -> str:
    entries = [
        e
        for e in ps.log
        if e.agent_id == agent_id and e.action not in NON_GAMEPLAY_ACTIONS
    ][-limit:]
    if not entries:
        return "  (none yet)"
    return "\n".join(
        f"  tick {e.tick} [{e.action}] -> {e.note or '(no change)'}" for e in entries
    )


STALL_TICKS = 2


def _party_stalled(ps: PartyState, party_size: int) -> bool:
    """True if every action across the last STALL_TICKS ticks was idle.

    A mutual-idle cascade: nobody is making progress, so the next agent should
    be nudged to break the loop (move rooms or apply a clue).
    """
    if party_size <= 0 or ps.tick <= STALL_TICKS:
        return False
    window = [
        e
        for e in ps.log
        if e.tick > ps.tick - 1 - STALL_TICKS and e.action not in NON_GAMEPLAY_ACTIONS
    ]
    if len(window) < party_size * STALL_TICKS:
        return False
    return all(e.action == IDLE_ACTION for e in window)


def _agent_act(
    agent_id: str,
    member: PartyMember,
    world: GameWorld,
    ps: PartyState,
    action_space: list[str],
    visible: list[WorldObject],
    teammate_last: TickAction | None,
    stalled: bool = False,
    escape_plan: str = "",
    gm_directive: str = "",
) -> dict:
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    inventory_str = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    known_str = ", ".join(ps.known_info) if ps.known_info else "(none)"
    space_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_space))

    win = world.win_condition
    win_str = (
        f"object {win.object_id} reaches state '{win.state}'"
        if win.object_id
        else "unknown"
    )

    room_goal = room.goal if room and room.goal else "(no specific goal — explore)"
    if room and room.goal_completion is not None:
        if _goal_completion_satisfied(room.goal_completion, ps):
            room_goal_status = f"DONE ✓"
        else:
            room_goal_status = (
                f"IN PROGRESS — need: {_format_goal_completion(room.goal_completion)}"
            )
    else:
        room_goal_status = "(no completion condition)"
    key_objs = (
        ", ".join(room.key_objects) if room and room.key_objects else "(none flagged)"
    )

    # Show adjacent rooms.
    exit_lines: list[str] = []
    if room:
        for direction, neighbor_id in room.adjacency.items():
            exit_lines.append(f"{direction} → {neighbor_id}")
    room_exit_status = "; ".join(exit_lines) if exit_lines else "(no exits)"

    history_str = _format_agent_history(ps, agent_id)
    if stalled:
        history_str = (
            "  !! THE PARTY IS STALLED — everyone has been idling. Do NOT 'wait'. "
            "Examine an un-inspected object, apply a known clue/tool/code, or "
            "'go <room_id>' to a new room THIS tick.\n" + history_str
        )

    gm_directive_str = (
        gm_directive if gm_directive else "(none — keep working this room's goal)"
    )

    prompt = ACTION_PROMPT.format(
        gm_directive=gm_directive_str,
        agent_id=agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        tick=ps.tick + 1,
        current_room=ps.current_room,
        room_description=room.description if room else "",
        room_goal=room_goal,
        room_goal_status=room_goal_status,
        room_key_objects=key_objs,
        room_exit_status=room_exit_status,
        objective=world.objective,
        win_condition=win_str,
        objects_in_room=_format_objects(visible, ps),
        escape_plan=escape_plan or "(none)",
        inventory=inventory_str,
        known_info=known_str,
        action_space=space_str,
        teammate_last_say=teammate_last.say if teammate_last else "(none)",
        teammate_last_action=teammate_last.action if teammate_last else "(none)",
        agent_recent_history=history_str,
    )

    llm = get_llm("player")
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    raw_choice = str(data.get("action", "")).strip()
    return {
        "say": str(data.get("say", "")).strip(),
        "action": _resolve_choice(raw_choice, action_space),
    }


# ---------- observation phase ----------


def _objects_for_observation(world: GameWorld, ps: PartyState) -> list[WorldObject]:
    """Every object whose container chain roots in the current room.

    Unlike `_objects_in_room`, this does NOT hide objects nested in still-closed
    containers — the observation should survey the FULL state of the room (what
    has been handled and what is still pending), not only what is currently
    reachable.
    """
    by_id = {o.id: o for o in world.objects}

    def _roots_in_room(obj: WorldObject) -> bool:
        cursor = obj
        seen: set[str] = set()
        while cursor.id not in seen:
            seen.add(cursor.id)
            if cursor.location == ps.current_room:
                return True
            parent = by_id.get(cursor.location)
            if parent is None:
                return False
            cursor = parent
        return False

    return [o for o in world.objects if _roots_in_room(o)]


def _format_object_states(
    objects: list[WorldObject], world: GameWorld, ps: PartyState
) -> str:
    """'object - state [done|pending]' survey of the whole room for observation.

    Includes every object (interacted or not, reachable or still nested) with its
    current state and whether its puzzle has been resolved, so planning accounts
    for the complete picture.
    """
    if not objects:
        return "  (no objects in this room)"
    resolved = _resolved_object_ids(world, ps)
    by_id = {o.id: o for o in world.objects}
    lines = []
    for o in objects:
        state = ps.object_states.get(o.id, o.state)
        status = "done" if o.id in resolved else "pending"
        # surface nesting so the agent knows a clue/tool sits inside a container
        if o.location != ps.current_room and o.location in by_id:
            where = f" (inside {o.location})"
        else:
            where = ""
        lines.append(f"  - {o.id} - {state} [{status}]{where}")
    return "\n".join(lines)


def _update_global_observations(
    world: GameWorld,
    ps: PartyState,
    new_notes: dict[str, list[str]] | None = None,
) -> None:
    """Upsert an ObjectObservation entry for every object the party has encountered.

    Covers all objects in visited rooms plus anything in the party's inventory.
    ``new_notes`` maps object_id -> fresh bullet strings produced by an observe/
    re-observe call this tick; when provided they replace the stored notes for
    that object. Objects without new notes keep their existing notes but get their
    state and last_seen_tick refreshed.
    """
    seen_locations: set[str] = ps.visited | {ps.current_room}
    for obj in world.objects:
        obj_room = obj.location
        # Resolve nested objects up to the room level.
        visited = False
        loc = obj_room
        for _ in range(10):
            if loc in seen_locations:
                visited = True
                break
            parent = next((o for o in world.objects if o.id == loc), None)
            if parent is None:
                break
            loc = parent.location
        if not visited and obj.id not in ps.inventory:
            continue

        current_state = ps.object_states.get(obj.id, obj.state)
        existing = ps.global_object_observations.get(obj.id)
        notes = (new_notes or {}).get(obj.id)
        ps.global_object_observations[obj.id] = ObjectObservation(
            object_id=obj.id,
            state=current_state,
            location=loc if loc in seen_locations else obj_room,
            notes=notes if notes is not None else (existing.notes if existing else []),
            last_seen_tick=ps.tick,
        )


def _format_global_observations(ps: PartyState) -> str:
    """Render global_object_observations as a compact table for prompt injection."""
    if not ps.global_object_observations:
        return "(none yet)"
    lines: list[str] = []
    for obj_id, obs in sorted(ps.global_object_observations.items()):
        notes_str = "; ".join(obs.notes) if obs.notes else "-"
        lines.append(
            f"  {obj_id} | room: {obs.location} | state: {obs.state} | notes: {notes_str}"
        )
    return "\n".join(lines)


def _global_observation_panel_lines(ps: PartyState) -> list[str]:
    """Global observations as panel body lines, grouped by room (current first).

    Unlike the per-room OBSERVED RESULT snapshot (frozen at room entry), this is
    rebuilt every tick from ``global_object_observations`` so states stay live and
    objects from every visited room — plus carried inventory — remain in view.
    """
    obs = ps.global_object_observations
    if not obs:
        return ["(nothing observed yet)"]

    by_room: dict[str, list[ObjectObservation]] = {}
    for o in obs.values():
        by_room.setdefault(o.location, []).append(o)

    # Current room first, then the rest alphabetically, so the acting view leads
    # with where the party is.
    room_order = sorted(by_room, key=lambda r: (r != ps.current_room, r))
    lines: list[str] = []
    for room in room_order:
        marker = " (here)" if room == ps.current_room else ""
        lines.append(f"### {room}{marker}")
        for o in sorted(by_room[room], key=lambda x: x.object_id):
            notes_str = "; ".join(o.notes) if o.notes else "-"
            lines.append(f"- {o.object_id} [{o.state}] — {notes_str}")
    return lines


def _agent_observe(
    agent_id: str,
    member: PartyMember,
    world: GameWorld,
    ps: PartyState,
) -> tuple[list[str], dict[str, list[str]]]:
    """One agent surveys the room and lists the objects present — no action taken.

    Returns (observation_bullets, object_notes) where object_notes maps object_id
    to a list containing the agent's note for that object. Both are appended to
    global_object_observations by the caller.
    """
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    win = world.win_condition
    win_str = (
        f"object {win.object_id} reaches state '{win.state}'"
        if win.object_id
        else "unknown"
    )

    # Survey the FULL room state (handled + pending, reachable + nested), not just
    # the currently-reachable `visible` set.
    all_objects = _objects_for_observation(world, ps)

    prompt = OBSERVE_PROMPT.format(
        agent_id=agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        current_room=ps.current_room,
        room_description=room.description if room else "",
        room_goal=room.goal if room and room.goal else "(no specific goal — explore)",
        win_condition=win_str,
        object_state_list=_format_object_states(all_objects, world, ps),
        inventory=", ".join(ps.inventory) if ps.inventory else "(empty)",
        known_info=", ".join(ps.known_info) if ps.known_info else "(none)",
        global_observations=_format_global_observations(ps),
    )
    llm = get_llm("player")
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    bullets = _as_bullets(data.get("observation"))
    raw_notes = data.get("object_notes") or {}
    object_notes: dict[str, list[str]] = {
        k: [str(v)] for k, v in raw_notes.items() if isinstance(v, str) and v
    }
    return bullets, object_notes


# ---------- planning debate ----------


def _plan_context(world: GameWorld, ps: PartyState, observation: list[str]) -> dict:
    """Shared template fields used by every prompt in the planning debate."""
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    win = world.win_condition
    win_str = (
        f"object {win.object_id} reaches state '{win.state}'"
        if win.object_id
        else "unknown"
    )
    all_objects = _objects_for_observation(world, ps)
    return {
        "current_room": ps.current_room,
        "room_description": room.description if room else "",
        "room_goal": (
            room.goal if room and room.goal else "(no specific goal — explore)"
        ),
        "win_condition": win_str,
        "observation": (
            "\n".join(f"- {b}" for b in observation) if observation else "(none)"
        ),
        "object_state_list": _format_object_states(all_objects, world, ps),
        "inventory": ", ".join(ps.inventory) if ps.inventory else "(empty)",
        "known_info": ", ".join(ps.known_info) if ps.known_info else "(none)",
    }


def _invoke_plan(prompt: str) -> tuple[list[str], str]:
    """Return (plan bullets, one-line reasoning) from a planning prompt."""
    llm = get_llm("player")
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    return _as_bullets(data.get("plan")), str(data.get("reasoning", "")).strip()


def _format_proposals(proposals: dict[str, tuple[str, list[str]]]) -> str:
    """Render each member's current plan for the critique/synthesis prompts."""
    blocks: list[str] = []
    for agent_id, (name, plan) in proposals.items():
        bullets = "\n".join(f"    - {b}" for b in plan) if plan else "    - (no plan)"
        blocks.append(f"  {agent_id} ({name}):\n{bullets}")
    return "\n".join(blocks)


def _agent_propose(
    member: PartyMember, world: GameWorld, ps: PartyState, observation: list[str]
) -> tuple[list[str], str]:
    """One agent's opening plan proposal, colored by their role."""
    prompt = PLAN_PROPOSE_PROMPT.format(
        agent_id=member.agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        **_plan_context(world, ps, observation),
    )
    return _invoke_plan(prompt)


def _agent_critique(
    member: PartyMember,
    world: GameWorld,
    ps: PartyState,
    observation: list[str],
    proposals: dict[str, tuple[str, list[str]]],
) -> tuple[list[str], str]:
    """One agent revises their plan after reading every teammate's current plan."""
    prompt = PLAN_CRITIQUE_PROMPT.format(
        agent_id=member.agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        proposals=_format_proposals(proposals),
        **_plan_context(world, ps, observation),
    )
    return _invoke_plan(prompt)


def _synthesize_plan(
    world: GameWorld,
    ps: PartyState,
    observation: list[str],
    proposals: dict[str, tuple[str, list[str]]],
) -> tuple[list[str], str]:
    """Merge the debated plans into the single unified plan the party acts on."""
    prompt = PLAN_SYNTHESIZE_PROMPT.format(
        proposals=_format_proposals(proposals),
        **_plan_context(world, ps, observation),
    )
    return _invoke_plan(prompt)


def _debate_plan(
    party: list[PartyMember],
    world: GameWorld,
    ps: PartyState,
    observation: list[str],
    render: bool = True,
) -> list[str]:
    """Run a multi-agent debate and return the unified escape plan.

    Propose (each agent) -> DEBATE_ROUNDS x critique/revise -> synthesize.
    With one agent the debate collapses to a single proposal (no synth call).
    """
    # Opening proposals.
    proposals: dict[str, tuple[str, list[str]]] = {}
    for member in party:
        plan, reasoning = _agent_propose(member, world, ps, observation)
        proposals[member.agent_id] = (member.character.name, plan)
        if render:
            _render_agent_plan(member, plan, label="PROPOSES", reasoning=reasoning)

    # A lone agent has nothing to debate — its proposal is the plan.
    if len(party) == 1:
        return proposals[party[0].agent_id][1]

    # Critique / revise rounds.
    for round_no in range(DEBATE_ROUNDS):
        revised: dict[str, tuple[str, list[str]]] = {}
        for member in party:
            plan, reasoning = _agent_critique(member, world, ps, observation, proposals)
            revised[member.agent_id] = (member.character.name, plan)
            if render:
                _render_agent_plan(
                    member,
                    plan,
                    label=f"REVISES (round {round_no + 1})",
                    reasoning=reasoning,
                )
        proposals = revised

    # Final synthesis into one unified plan.
    unified, reasoning = _synthesize_plan(world, ps, observation, proposals)
    if render:
        _render_unified_plan(ps.current_room, unified, reasoning=reasoning)
    return unified


def _render_observation(
    room_id: str, visible: list[WorldObject], ps: PartyState
) -> None:
    body = [f"Entered {room_id} — objects observed (object - state):"]
    for o in visible:
        state = ps.object_states.get(o.id, o.state)
        body.append(f"  {o.id} - {state}")
    if not visible:
        body.append("  (no objects visible)")
    _panel("OBSERVATION", body)


def _bullet_lines(items: list[str], empty: str) -> list[str]:
    return [f"- {b}" for b in items] if items else [empty]


def _render_agent_observation(
    member: PartyMember, observation: list[str], label: str = "OBSERVES"
) -> None:
    _panel(
        f"{member.agent_id} -- {member.character.name} ({member.character.role}) {label}",
        _bullet_lines(observation, "(no observation)"),
    )


def _room_fingerprint(world: GameWorld, ps: PartyState) -> str:
    """A stable string snapshot of everything that could change the room picture.

    Used to decide whether the lead agent should re-observe + re-plan: if the
    states of objects rooted in the current room (plus shared inventory, known
    info, and power) are unchanged since last tick, nothing new was revealed and
    the standing observation/plan still hold — so we skip the extra LLM calls.
    Returned as a str so it can be stored on PartyState across graph node calls.
    """
    room_objs = _objects_for_observation(world, ps)
    obj_states = sorted((o.id, ps.object_states.get(o.id, o.state)) for o in room_objs)
    parts = [
        ps.current_room,
        repr(obj_states),
        repr(sorted(ps.inventory)),
        repr(sorted(ps.known_info)),
        repr(sorted(ps.power_active)),
    ]
    return "|".join(parts)


def _agent_reobserve(
    party: list[PartyMember], world: GameWorld, ps: PartyState
) -> None:
    """Re-survey the room (lead) and rebuild the unified plan via debate.

    Called only on ticks where the room state actually changed (new object
    revealed, item taken, lock opened, power on), so the standing observation and
    unified plan stay consistent with what the party can now see.
    """
    lead = party[0]
    observation, object_notes = _agent_observe(lead.agent_id, lead, world, ps)
    ps.room_observations[ps.current_room] = observation
    _update_global_observations(world, ps, object_notes)
    ps.log.append(
        TickAction(
            tick=ps.tick,
            agent_id=lead.agent_id,
            say="; ".join(observation),
            action="reobserve",
            note="re-observed room",
        )
    )
    _render_agent_observation(lead, observation, label="RE-OBSERVES")

    unified = _debate_plan(party, world, ps, observation)
    ps.room_plans[ps.current_room] = unified
    ps.log.append(
        TickAction(
            tick=ps.tick,
            agent_id=lead.agent_id,
            say="; ".join(unified),
            action="replan",
            note="revised unified plan",
        )
    )


def _render_agent_plan(
    member: PartyMember, plan: list[str], label: str = "PLANS", reasoning: str = ""
) -> None:
    body = _bullet_lines(plan, "(no plan)")
    if reasoning:
        body = body + [f"why: {reasoning}"]
    _panel(
        f"{member.agent_id} -- {member.character.name} ({member.character.role}) {label}",
        body,
    )


def _render_unified_plan(room_id: str, plan: list[str], reasoning: str = "") -> None:
    body = _bullet_lines(plan, "(no plan)")
    if reasoning:
        body = body + [f"why: {reasoning}"]
    _panel(
        f"UNIFIED PLAN -- {room_id} (party agreed)",
        body,
    )


# ---------- initial state ----------


def _build_initial_party_state(world: GameWorld) -> PartyState:
    starting_room = world.rooms[0].id if world.rooms else ""
    object_states = {o.id: o.state for o in world.objects}
    fuse_states = {o.id: dict(o.fuses) for o in world.objects if o.fuses is not None}
    power_active: set[str] = set()
    for panel_id, fuses in fuse_states.items():
        for label, state in fuses.items():
            if state == "ON":
                power_active.add(f"sekring_{label}_ON")
    return PartyState(
        current_room=starting_room,
        visited={starting_room} if starting_room else set(),
        object_states=object_states,
        fuse_states=fuse_states,
        power_active=power_active,
    )


def _check_victory(world: GameWorld, ps: PartyState) -> bool:
    win = world.win_condition
    if not win.object_id:
        return False
    return _state_satisfies(ps.object_states.get(win.object_id), win.state)


def _render_party_map(world: GameWorld, ps: PartyState) -> None:
    _stream()
    _rule("PARTY LOCATION")
    render_room_layout(
        world.rooms, world.objects, party_room=ps.current_room, party_label="* PARTY"
    )
    _stream()


def _render_intro(world: GameWorld, party: list[PartyMember]) -> None:
    _banner("LIVE GAMEPLAY")

    win = world.win_condition
    if win.object_id:
        _panel("OBJECTIVE", [f"WIN WHEN: {win.object_id} -> {win.state}"])
        _stream()


def _resolved_object_ids(world: GameWorld, ps: PartyState) -> set[str]:
    """Objects the party has interacted with — state changed, taken, or examined.

    The checkmark means "this object has been handled": its state changed from
    initial, it was picked up, or it was examined at least once. Examining marks
    an object as interacted so the party doesn't re-examine it needlessly.
    """
    resolved: set[str] = set()
    initial = {o.id: o.state for o in world.objects}
    examined = {
        e.action.split(" ", 1)[1]
        for e in ps.log
        if e.action.startswith("examine ") and e.note.startswith("examined")
    }
    for obj in world.objects:
        if obj.id in ps.inventory:
            resolved.add(obj.id)
            continue
        if obj.id in examined:
            resolved.add(obj.id)
            continue
        current = ps.object_states.get(obj.id, obj.state)
        if current != initial.get(obj.id, obj.state):
            resolved.add(obj.id)
    return resolved


def _compute_map_flow(world: GameWorld, ps: PartyState) -> list[str]:
    """BFS path from the party's current room to the room containing the win object."""
    win = world.win_condition
    if not win.object_id:
        return []
    win_obj = next((o for o in world.objects if o.id == win.object_id), None)
    if win_obj is None:
        return []
    target_room = win_obj.location
    if target_room not in {r.id for r in world.rooms}:
        return []
    graph = WorldGraph(world)
    return graph.path(ps.current_room, target_room)


def _render_tick_header(
    world: GameWorld,
    ps: PartyState,
    party: list[PartyMember],
) -> None:
    _stream()
    _rule(f"TICK {ps.tick} / {MAX_TICKS}")

    resolved = _resolved_object_ids(world, ps)

    flow = _compute_map_flow(world, ps)
    flow_str = " -> ".join(flow) if flow else "(no route)"
    _panel("MAP FLOW", [f"start -> end: {flow_str}"])

    _stream()
    _rule("MAP LAYOUT")
    render_room_layout(
        world.rooms,
        world.objects,
        party_room=ps.current_room,
        party_label="* PARTY",
        interacted_ids=resolved,
        object_states=ps.object_states,
    )
    _stream()

    map_lines: list[str] = ["### Map"]
    for room in world.rooms:
        marker = " (party here)" if room.id == ps.current_room else ""
        map_lines.append(f"* {room.id}{marker}")
        room_objects = [o for o in world.objects if o.location == room.id]
        if room_objects:
            map_lines.append("    * objects:")
            for o in room_objects:
                tick_mark = "[x]" if o.id in resolved else "[ ]"
                st = ps.object_states.get(o.id, o.state)
                map_lines.append(f"        * {tick_mark} {o.id} [{st}]")
        else:
            map_lines.append("    * objects: (none)")
    _panel("MAP", map_lines)

    state_lines: list[str] = ["### Current State", "* agent locations:"]
    for member in party:
        state_lines.append(
            f"    * {member.agent_id} ({member.character.name}) -> {ps.current_room}"
        )
    inv = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    state_lines.append(f"* shared inventory: {inv}")
    known = ", ".join(ps.known_info) if ps.known_info else "(none)"
    state_lines.append(f"* known info: {known}")
    _panel("CURRENT STATE", state_lines)

    # OBSERVED RESULT is the party's GLOBAL view — every object seen across all
    # visited rooms (plus inventory), rebuilt each tick so states never go stale.
    # Refresh first so a lock opened last tick shows as unlocked here, then render.
    _update_global_observations(world, ps)
    _panel("OBSERVED RESULT", _global_observation_panel_lines(ps))
    plan = ps.room_plans.get(ps.current_room, [])
    plan_lines = [f"### {ps.current_room}"] + _bullet_lines(plan, "(not yet planned)")
    _panel("ESCAPE PLAN", plan_lines)

    # Cumulative log of actions already performed by the party (most recent last).
    # Observation/planning passes are shown in their own panels, not here.
    action_log = [e for e in ps.log if e.action not in NON_GAMEPLAY_ACTIONS]
    log_lines: list[str] = ["### Actions performed so far"]
    if action_log:
        for e in action_log[-ACTION_LOG_WINDOW:]:
            log_lines.append(
                f"* tick {e.tick} {e.agent_id}: {e.action} -> {e.note or '(no change)'}"
            )
        if len(action_log) > ACTION_LOG_WINDOW:
            log_lines.append(f"  (+{len(action_log) - ACTION_LOG_WINDOW} earlier)")
    else:
        log_lines.append("* (none yet)")
    _panel("ACTION LOG", log_lines)


def _render_action_space(member: PartyMember, space: list[str]) -> None:
    lines = [f"  {i + 1}. {a}" for i, a in enumerate(space)]
    _panel(
        f"{member.agent_id} -- {member.character.name} ACTION SPACE",
        lines or ["(none)"],
    )


def _render_agent_action(
    member: PartyMember, decided: dict, note: str, teammate: TickAction | None = None
) -> None:
    status = _classify_outcome(note)
    if teammate is not None:
        saw = (
            f"{teammate.agent_id} -> {teammate.action} ({teammate.note or 'no change'})"
        )
    else:
        saw = "(nothing yet)"
    body = [
        f"SAW : teammate {saw}",
        f"DO : {decided['action']}",
        f"{status} : {note}",
    ]
    _panel(
        f"{member.agent_id} -- {member.character.name} ({member.character.role})", body
    )


def _render_final(ps: PartyState, world: GameWorld) -> None:
    _banner("FINAL RESULT")
    outcome = "VICTORY" if ps.victory else f"ENDED (final room: {ps.current_room})"
    inv = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    known = ", ".join(ps.known_info) if ps.known_info else "(none)"
    win_obj_state = (
        ps.object_states.get(world.win_condition.object_id, "?")
        if world.win_condition.object_id
        else "?"
    )
    body = [
        f"Outcome      : {outcome}",
        f"Ticks used   : {ps.tick} / {MAX_TICKS}",
        f"Inventory    : {inv}",
        f"Known clues  : {known}",
        f"Visited      : {', '.join(sorted(ps.visited))}",
        f"Win object   : {world.win_condition.object_id} (state: {win_obj_state}, target: {world.win_condition.state})",
    ]
    _panel("SUMMARY", body)
    _stream()


# ---------- main node ----------


def route_after_gameplay(state: GameState) -> str:
    """Conditional edge: END when game is over, otherwise loop back to gameplay."""
    if state.party_state and state.party_state.game_over:
        return "end"
    return "gameplay"


def gameplay_node(state: GameState) -> dict:
    """Run exactly one tick of gameplay, check for end conditions, and return.

    Sets ps.game_over (and ps.victory) in place so route_after_gameplay can
    branch to END without a separate eval node.
    """
    import time

    world = state.world
    if not world or not state.party:
        return {}

    _tick_start = time.perf_counter()
    is_first_tick = state.party_state is None
    ps = state.party_state or _build_initial_party_state(world)
    new_messages: list[AIMessage] = []

    if is_first_tick:
        _render_intro(world, state.party)
        _render_party_map(world, ps)

    # Resume the fingerprint stored from the end of the previous tick.
    last_fingerprint: str | None = ps.last_fingerprint

    ps.tick += 1
    visible = _objects_in_room(world, ps)
    action_space = _build_action_space(world, ps, visible)

    _render_tick_header(world, ps, state.party)

    # Entry observation+planning: the first tick in a newly-entered room is
    # spent (1) OBSERVING — agents enumerate the objects present — then
    # (2) PLANNING — forming an ordered escape plan from that observation.
    # No action is taken on this tick.
    if ps.current_room not in ps.observed_rooms:
        _render_observation(ps.current_room, visible, ps)
        lead = state.party[0]
        merged_notes: dict[str, list[str]] = {}
        for member in state.party:
            observation, object_notes = _agent_observe(
                member.agent_id, member, world, ps
            )
            for obj_id, notes in object_notes.items():
                merged_notes.setdefault(obj_id, []).extend(notes)
            ps.log.append(
                TickAction(
                    tick=ps.tick,
                    agent_id=member.agent_id,
                    say="; ".join(observation),
                    action="observe",
                    note="observed room",
                )
            )
            _render_agent_observation(member, observation, label="OBSERVES")
            if member.agent_id == lead.agent_id:
                ps.room_observations[ps.current_room] = observation
        _update_global_observations(world, ps, merged_notes)

        # Planning debate: agents propose, critique/revise, then a single
        # unified plan is synthesized as the party's shared action reference.
        unified = _debate_plan(
            state.party,
            world,
            ps,
            ps.room_observations.get(ps.current_room, []),
        )
        ps.room_plans[ps.current_room] = unified
        ps.log.append(
            TickAction(
                tick=ps.tick,
                agent_id=lead.agent_id,
                say="; ".join(unified),
                action="plan",
                note="unified escape plan",
            )
        )

        ps.observed_rooms.add(ps.current_room)
        ps.last_fingerprint = _room_fingerprint(world, ps)
        elapsed = time.perf_counter() - _tick_start
        print(
            f"[gameplay] tick {ps.tick} (observe+plan) elapsed {elapsed:.2f}s",
            flush=True,
        )
        return {"messages": new_messages, "party_state": ps}

    # If last tick changed the room picture (revealed an object, took an item,
    # opened a lock, brought power online), the entry observation/plan is stale.
    # The lead agent re-observes + re-plans once to resync before anyone acts.
    fingerprint = _room_fingerprint(world, ps)
    if last_fingerprint is not None and fingerprint != last_fingerprint:
        _agent_reobserve(state.party, world, ps)

    # Most recent action per agent, seeded from prior ticks. Updated in-place
    # as each player acts this tick so later players see what earlier players
    # just did and avoid duplicating it.
    last_action_by_agent: dict[str, TickAction | None] = {
        m.agent_id: None for m in state.party
    }
    for entry in reversed(ps.log):
        # Skip observation/planning entries — teammates care about each other's
        # last *gameplay* action, not the lead's re-observe/re-plan bookkeeping.
        if entry.action in NON_GAMEPLAY_ACTIONS:
            continue
        if last_action_by_agent.get(entry.agent_id) is None:
            last_action_by_agent[entry.agent_id] = entry
        if all(v is not None for v in last_action_by_agent.values()):
            break

    prev_room = ps.current_room
    prev_inv = list(ps.inventory)
    prev_known = list(ps.known_info)
    prev_power = set(ps.power_active)

    stalled = _party_stalled(ps, len(state.party))
    if stalled:
        _stream("  !! Party stalled (mutual idle) — nudging agents to act.")

    tick_actions: list[TickAction] = []
    # Pending go actions are deferred until all agents have chosen so that a
    # single agent cannot split the party mid-tick. Only the first go vote
    # executes; later votes are overridden to match it so every agent moves
    # together or not at all.
    pending_go: tuple[str, str] | None = None  # (agent_id, dest)
    deferred_renders: list[tuple] = []  # (member, decided, teammate) for go actions

    for member in state.party:
        teammate = next(
            (
                last_action_by_agent[m.agent_id]
                for m in state.party
                if m.agent_id != member.agent_id
            ),
            None,
        )
        plan = "\n".join(f"- {b}" for b in ps.room_plans.get(ps.current_room, []))
        gm_directive_text = ""
        _render_action_space(member, action_space)
        decided = _agent_act(
            member.agent_id,
            member,
            world,
            ps,
            action_space,
            visible,
            teammate,
            stalled,
            plan,
            gm_directive_text,
        )

        chosen_action = decided["action"]

        # Defer go actions — they move ps.current_room which would split the
        # party if a later agent picks a different destination.
        if chosen_action.startswith("go "):
            if pending_go is None:
                # First go vote this tick: record it, resolve at end-of-tick.
                pending_go = (member.agent_id, chosen_action.split()[1])
            else:
                # Subsequent agent: override to match the already-pending move
                # so every agent travels to the same destination.
                chosen_action = f"go {pending_go[1]}"
            # Defer render until after the loop when the real note is known.
            note, target = "", None
            deferred_renders.append((member, decided, teammate))
        else:
            note, target = _resolve_action(chosen_action, world, ps)

        this_action = TickAction(
            tick=ps.tick,
            agent_id=member.agent_id,
            say=decided["say"],
            action=chosen_action,
            target_object=target,
            note=note,
        )
        tick_actions.append(this_action)
        # Publish this action immediately so the next player this tick sees it.
        last_action_by_agent[member.agent_id] = this_action
        if not chosen_action.startswith("go "):
            _render_agent_action(member, decided, note, teammate)

        visible = _objects_in_room(world, ps)
        action_space = _build_action_space(world, ps, visible)

    # Resolve the single deferred go action now that all agents have chosen.
    # Pydantic models are frozen — rebuild any go entries with the real note,
    # then render them in agent order with the correct outcome.
    if pending_go is not None:
        go_action = f"go {pending_go[1]}"
        real_note, _ = _resolve_action(go_action, world, ps)
        tick_actions = [
            TickAction(
                tick=ta.tick,
                agent_id=ta.agent_id,
                say=ta.say,
                action=ta.action,
                target_object=ta.target_object,
                note=real_note if ta.action.startswith("go ") else ta.note,
            )
            for ta in tick_actions
        ]
        for member, decided, teammate in deferred_renders:
            _render_agent_action(member, decided, real_note, teammate)

    # Highlight state changes that happened this tick.
    callouts: list[str] = []
    if ps.current_room != prev_room:
        callouts.append(f">> Party moved: {prev_room} -> {ps.current_room}")
    gained = [i for i in ps.inventory if i not in prev_inv]
    for g in gained:
        callouts.append(f">> Picked up: {g}")
    learned = [k for k in ps.known_info if k not in prev_known]
    for l in learned:
        callouts.append(f">> Learned clue: {l}")
    new_power = ps.power_active - prev_power
    for p in sorted(new_power):
        callouts.append(f">> Power online: {p}")
    if callouts:
        _stream()
        for c in callouts:
            _stream(f"  {c}")

    ps.log.extend(tick_actions)
    ps.last_fingerprint = _room_fingerprint(world, ps)
    # Refresh global observations after every action tick so state changes
    # (unlocked, taken, power on) are reflected without needing an observe pass.
    _update_global_observations(world, ps)

    # --- end-of-tick: victory / time-up check ---
    if _check_victory(world, ps):
        ps.victory = True
        ps.game_over = True
        _banner("VICTORY", char="*")
        _stream(f"  Party achieved the win condition at tick {ps.tick}.")
        _render_final(ps, world)
        new_messages.append(AIMessage(content=f"[gameplay] VICTORY at tick {ps.tick}"))
    elif ps.tick >= MAX_TICKS:
        ps.game_over = True
        _banner("TIME UP", char="*")
        _stream(
            f"  Reached MAX_TICKS={MAX_TICKS} without satisfying the win condition."
        )
        _render_final(ps, world)
        new_messages.append(
            AIMessage(content=f"[gameplay] stopped at MAX_TICKS={MAX_TICKS}")
        )

    elapsed = time.perf_counter() - _tick_start
    print(f"[gameplay] tick {ps.tick} elapsed {elapsed:.2f}s", flush=True)

    return {
        "messages": new_messages,
        "party_state": ps,
    }
