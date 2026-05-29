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

from config.settings import get_llm
from prompts import load_prompt
from state import (
    GameState,
    GameWorld,
    PartyMember,
    PartyState,
    Prerequisite,
    Room,
    TickAction,
    WorldObject,
)
from state.game_state import Ability
from visualization import render_room_layout

SYSTEM_PROMPT = load_prompt("gameplay_agent", "system")
ACTION_PROMPT = load_prompt("gameplay_agent", "action")

MAX_TICKS = 40
IDLE_ACTION = "wait"

HIDDEN_STATES = {"locked", "locked_bolt", "locked_room", "hidden"}
UNLOCKED_STATES = {"unlocked", "open", "visible"}


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
STATUS_OK = "OK"        # state changed (unlocked, took item, moved, info learned)
STATUS_INFO = "..."     # action ran but nothing changed (examine without new info, idle)
STATUS_FAIL = "XX"      # action invalid / missing precondition

_OK_KEYWORDS = ("unlocked", "took", "moved", "learned", "flipped", "inserted", "entered", "used")
_FAIL_KEYWORDS = ("missing", "no matching", "unknown", "cannot", "not accessible", "no direct", "no target", "no fuse")


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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_ ]+", " ", text.lower()).strip()


def _format_ability(ability: Ability) -> str:
    uses = "passive" if ability.max_uses < 0 else f"{ability.uses_remaining} use(s) left"
    return f"{ability.name} [{ability.effect}, {uses}] — {ability.description}"


def _consume_use(ability: Ability) -> bool:
    if ability.max_uses < 0:
        return True
    if ability.uses_remaining <= 0:
        return False
    ability.uses_remaining -= 1
    return True


# ---------- object visibility ----------

def _object_visible(obj: WorldObject, ps: PartyState, parent_lookup: dict[str, WorldObject]) -> bool:
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


def _objects_in_room(
    world: GameWorld, ps: PartyState
) -> list[WorldObject]:
    parent_lookup = {o.id: o for o in world.objects}
    return [o for o in world.objects if _object_visible(o, ps, parent_lookup)]


# ---------- action space ----------

def _verbs_for(obj: WorldObject, ps: PartyState) -> list[str]:
    state = ps.object_states.get(obj.id, obj.state)
    verbs: list[str] = [f"examine {obj.id}"]
    if obj.takeable and state in UNLOCKED_STATES and obj.id not in ps.inventory:
        verbs.append(f"take {obj.id}")
    if obj.requires_code and state in HIDDEN_STATES:
        verbs.append(f"enter_code {obj.id}")
    if obj.requires_tool and state in HIDDEN_STATES:
        verbs.append(f"use_tool {obj.id}")
    if obj.requires_liquid and state in HIDDEN_STATES:
        verbs.append(f"insert_liquid {obj.id}")
    if obj.fuses is not None:
        for label in obj.fuses:
            verbs.append(f"flip_fuse {obj.id} {label}")
    if obj.requires_power and state in HIDDEN_STATES:
        verbs.append(f"open {obj.id}")
    return verbs


def _build_action_space(
    world: GameWorld, ps: PartyState, visible: list[WorldObject]
) -> list[str]:
    seen: set[str] = set()
    space: list[str] = []
    for obj in visible:
        for v in _verbs_for(obj, ps):
            if v not in seen:
                seen.add(v)
                space.append(v)
    current_room = next((r for r in world.rooms if r.id == ps.current_room), None)
    if current_room:
        for direction, neighbor in current_room.adjacency.items():
            move = f"go {neighbor}"
            if move not in seen:
                seen.add(move)
                space.append(move)
    if IDLE_ACTION not in seen:
        space.append(IDLE_ACTION)
    return space


# ---------- action resolution ----------

def _resolve_examine(obj: WorldObject, ps: PartyState) -> str:
    if obj.contains_info and obj.contains_info not in ps.known_info:
        ps.known_info.append(obj.contains_info)
        return f"examined {obj.id}; learned {obj.contains_info}"
    return f"examined {obj.id}"


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
    if obj.requires_code in ps.known_info or obj.requires_code in (
        info.split("_")[-1] for info in ps.known_info
    ):
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
        haystack = " ".join(
            [held_id]
            + ([held.description] if held else [])
        )
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


def _prerequisite_satisfied(p: Prerequisite, ps: PartyState) -> bool:
    if p.type == "object_state":
        return bool(p.object_id) and ps.object_states.get(p.object_id) == p.state
    if p.type == "known_info":
        return bool(p.info) and p.info in ps.known_info
    if p.type == "has_item":
        return bool(p.object_id) and p.object_id in ps.inventory
    if p.type == "power_active":
        return bool(p.id) and p.id in ps.power_active
    return True


def _unmet_prerequisites(room: Room, ps: PartyState) -> list[Prerequisite]:
    return [p for p in room.prerequisites if not _prerequisite_satisfied(p, ps)]


def _format_prereq(p: Prerequisite) -> str:
    if p.type == "object_state":
        return f"{p.object_id} must be {p.state}"
    if p.type == "known_info":
        return f"must know {p.info}"
    if p.type == "has_item":
        return f"must carry {p.object_id}"
    if p.type == "power_active":
        return f"power {p.id} must be on"
    return p.type


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
                dest_room = rooms_by_id[dest]
                unmet = _unmet_prerequisites(dest_room, ps)
                if unmet:
                    reasons = "; ".join(_format_prereq(p) for p in unmet)
                    return (f"cannot enter {dest} — {reasons}", None)
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
    return IDLE_ACTION


def _format_objects(visible: list[WorldObject], ps: PartyState) -> str:
    if not visible:
        return "  (none visible)"
    lines = []
    for o in visible:
        state = ps.object_states.get(o.id, o.state)
        lines.append(f"  - {o.id} [{state}]: {o.description}")
    return "\n".join(lines)


HISTORY_WINDOW = 6


def _format_agent_history(ps: PartyState, agent_id: str, limit: int = HISTORY_WINDOW) -> str:
    entries = [e for e in ps.log if e.agent_id == agent_id][-limit:]
    if not entries:
        return "  (none yet)"
    return "\n".join(
        f"  tick {e.tick} [{e.action}] -> {e.note or '(no change)'}" for e in entries
    )


def _agent_act(
    agent_id: str,
    member: PartyMember,
    world: GameWorld,
    ps: PartyState,
    action_space: list[str],
    visible: list[WorldObject],
    teammate_last: TickAction | None,
) -> dict:
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    inventory_str = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    known_str = ", ".join(ps.known_info) if ps.known_info else "(none)"
    space_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_space))

    win = world.win_condition
    win_str = f"object {win.object_id} reaches state '{win.state}'" if win.object_id else "unknown"

    room_goal = room.goal if room and room.goal else "(no specific goal — explore)"
    if room and room.goal_completion is not None:
        if _prerequisite_satisfied(room.goal_completion, ps):
            room_goal_status = f"DONE ({_format_prereq(room.goal_completion)})"
        else:
            room_goal_status = f"IN PROGRESS — need: {_format_prereq(room.goal_completion)}"
    else:
        room_goal_status = "(no completion condition)"
    key_objs = (
        ", ".join(room.key_objects) if room and room.key_objects else "(none flagged)"
    )

    # Show what's still missing to unlock any adjacent rooms.
    exit_lines: list[str] = []
    if room:
        rooms_by_id = {r.id: r for r in world.rooms}
        for direction, neighbor_id in room.adjacency.items():
            neighbor = rooms_by_id.get(neighbor_id)
            if neighbor is None:
                continue
            unmet = _unmet_prerequisites(neighbor, ps)
            if unmet:
                reasons = "; ".join(_format_prereq(p) for p in unmet)
                exit_lines.append(f"{direction} -> {neighbor_id}: BLOCKED ({reasons})")
            else:
                exit_lines.append(f"{direction} -> {neighbor_id}: OPEN")
    room_exit_status = "; ".join(exit_lines) if exit_lines else "(no exits)"

    prompt = ACTION_PROMPT.format(
        agent_id=agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        character_ability=_format_ability(member.character.ability),
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
        inventory=inventory_str,
        known_info=known_str,
        action_space=space_str,
        teammate_last_say=teammate_last.say if teammate_last else "(none)",
        teammate_last_action=teammate_last.action if teammate_last else "(none)",
        agent_recent_history=_format_agent_history(ps, agent_id),
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
    return ps.object_states.get(win.object_id) == win.state


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


def _interacted_object_ids(ps: PartyState) -> set[str]:
    return {entry.target_object for entry in ps.log if entry.target_object}


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

    interacted = _interacted_object_ids(ps)

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
        interacted_ids=interacted,
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
                tick_mark = "[x]" if o.id in interacted else "[ ]"
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
    _panel("CURRENT STATE", state_lines)


def _render_agent_action(
    member: PartyMember, decided: dict, note: str
) -> None:
    ab = member.character.ability
    uses = "passive" if ab.max_uses < 0 else f"{ab.uses_remaining} use(s) left"
    status = _classify_outcome(note)
    body = [
        f"Ability: {ab.name} [{ab.effect}, {uses}]",
        f"DO : {decided['action']}",
        f"{status} : {note}",
    ]
    _panel(f"{member.agent_id} -- {member.character.name} ({member.character.role})", body)


def _render_final(ps: PartyState, world: GameWorld) -> None:
    _banner("FINAL RESULT")
    outcome = "VICTORY" if ps.victory else f"ENDED (final room: {ps.current_room})"
    inv = ", ".join(ps.inventory) if ps.inventory else "(empty)"
    known = ", ".join(ps.known_info) if ps.known_info else "(none)"
    win_obj_state = ps.object_states.get(world.win_condition.object_id, "?") if world.win_condition.object_id else "?"
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

def gameplay_node(state: GameState) -> dict:
    world = state.world
    if not world or not state.party:
        return {}

    ps = state.party_state or _build_initial_party_state(world)
    new_messages: list[AIMessage] = []

    _render_intro(world, state.party)
    _render_party_map(world, ps)

    while not ps.game_over and ps.tick < MAX_TICKS:
        if _check_victory(world, ps):
            ps.game_over = True
            ps.victory = True
            _banner("VICTORY", char="*")
            _stream(f"  Party achieved the win condition at tick {ps.tick}.")
            new_messages.append(AIMessage(content=f"[gameplay] VICTORY at tick {ps.tick}"))
            break

        ps.tick += 1
        visible = _objects_in_room(world, ps)
        action_space = _build_action_space(world, ps, visible)

        _render_tick_header(world, ps, state.party)

        teammate_last_per_agent = {m.agent_id: None for m in state.party}
        for entry in reversed(ps.log):
            if teammate_last_per_agent.get(entry.agent_id) is None:
                teammate_last_per_agent[entry.agent_id] = entry
            if all(v is not None for v in teammate_last_per_agent.values()):
                break

        prev_room = ps.current_room
        prev_inv = list(ps.inventory)
        prev_known = list(ps.known_info)
        prev_power = set(ps.power_active)

        tick_actions: list[TickAction] = []
        for member in state.party:
            teammate = next(
                (
                    teammate_last_per_agent[m.agent_id]
                    for m in state.party
                    if m.agent_id != member.agent_id
                ),
                None,
            )
            decided = _agent_act(
                member.agent_id, member, world, ps, action_space, visible, teammate
            )
            note, target = _resolve_action(decided["action"], world, ps)
            tick_actions.append(
                TickAction(
                    tick=ps.tick,
                    agent_id=member.agent_id,
                    say=decided["say"],
                    action=decided["action"],
                    target_object=target,
                    note=note,
                )
            )
            _render_agent_action(member, decided, note)

            if _check_victory(world, ps):
                break

            visible = _objects_in_room(world, ps)
            action_space = _build_action_space(world, ps, visible)

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

    if not ps.game_over:
        ps.game_over = True
        _banner("TIME UP", char="*")
        _stream(f"  Reached MAX_TICKS={MAX_TICKS} without satisfying the win condition.")
        new_messages.append(AIMessage(content=f"[gameplay] Stopped at MAX_TICKS={MAX_TICKS}"))

    _render_final(ps, world)

    return {
        "messages": new_messages,
        "party_state": ps,
    }
