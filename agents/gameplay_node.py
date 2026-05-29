"""Gameplay node — runs the shared-state co-op loop where both party agents act each tick."""

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
    Mission,
    PartyMember,
    PartyState,
    RoomItem,
    TickAction,
)
from visualization import render_room_layout

SYSTEM_PROMPT = load_prompt("gameplay_agent", "system")
ACTION_PROMPT = load_prompt("gameplay_agent", "action")

MAX_TICKS = 30


class WorldGraph:
    """Strict directed adjacency graph built from world.rooms.

    Edges are taken literally from each room's adjacency dict — if A lists B
    as a neighbor but B doesn't list A, you cannot walk back from B to A.
    """

    def __init__(self, world: GameWorld) -> None:
        self._adj: dict[str, list[str]] = {}
        names = {r.name for r in world.rooms}
        for room in world.rooms:
            self._adj[room.name] = [n for n in room.adjacency.values() if n in names]

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


def _stream(line: str = "") -> None:
    """Print immediately so live gameplay shows up in real time."""
    print(line, flush=True)


def _render_party_map(world: GameWorld, current_room: str) -> None:
    _stream()
    _stream("  ┌── party location ──┐")
    render_room_layout(world.rooms, party_room=current_room, party_label="★ PARTY")
    _stream()


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
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


IDLE_ACTION = "wait"


def _build_action_space(remaining: list[str]) -> list[str]:
    """Deterministic discrete action set the agent must pick from this tick.

    Stable order: remaining required actions first (in mission order), then idle.
    Duplicates removed while preserving order.
    """
    seen: set[str] = set()
    space: list[str] = []
    for a in remaining:
        if a not in seen:
            seen.add(a)
            space.append(a)
    if IDLE_ACTION not in seen:
        space.append(IDLE_ACTION)
    return space


def _resolve_choice(response: str, space: list[str]) -> str:
    """Map an LLM response to an entry in the action space.

    Accepts either an index (1-based) or a verbatim/normalized match. Falls back to
    the idle action when nothing resolves — keeping selection fully deterministic.
    """
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


def _current_mission(
    missions: list[Mission], party_state: PartyState
) -> Mission | None:
    for m in sorted(missions, key=lambda x: x.gate_index):
        if m.gate_index not in party_state.completed_gates:
            return m
    return None


def _build_initial_party_state(world: GameWorld) -> PartyState:
    starting_room = world.game_flow.starting_room or (
        world.rooms[0].name if world.rooms else ""
    )
    return PartyState(
        current_room=starting_room,
        visited={starting_room} if starting_room else set(),
    )


def _agent_act(
    agent_id: str,
    member: PartyMember,
    world: GameWorld,
    ps: PartyState,
    mission: Mission,
    action_space: list[str],
    teammate_last: TickAction | None,
) -> dict:
    room = next((r for r in world.rooms if r.name == ps.current_room), None)
    room_description = room.description if room else ""

    completed = ps.completed_actions_by_gate.get(mission.gate_index, [])
    inventory_str = (
        ", ".join(i.name for i in ps.inventory) if ps.inventory else "(empty)"
    )
    required_str = (
        ", ".join(mission.required_actions) if mission.required_actions else "(none)"
    )
    completed_str = ", ".join(completed) if completed else "(none yet)"
    action_space_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_space))

    prompt = ACTION_PROMPT.format(
        agent_id=agent_id,
        character_name=member.character.name,
        character_role=member.character.role,
        character_trait=member.character.special_trait,
        tick=ps.tick + 1,
        current_room=ps.current_room,
        room_description=room_description,
        inventory=inventory_str,
        mission_description=mission.description,
        required_actions=required_str,
        completed_actions=completed_str,
        reward_item=mission.reward_item,
        action_space=action_space_str,
        teammate_last_say=teammate_last.say if teammate_last else "(none)",
        teammate_last_action=teammate_last.action if teammate_last else "(none)",
    )

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    data = _parse_json(response.content) or {}
    raw_choice = str(data.get("action", "")).strip()
    resolved = _resolve_choice(raw_choice, action_space)
    return {
        "say": str(data.get("say", "")).strip(),
        "action": resolved,
    }


def _next_room_for_completed_mission(world: GameWorld, mission: Mission) -> str | None:
    if mission.unlocks_exit_to == "VICTORY":
        return None
    if any(r.name == mission.unlocks_exit_to for r in world.rooms):
        return mission.unlocks_exit_to
    return None


def gameplay_node(state: GameState) -> dict:
    world = state.world
    if not world or not state.party or not state.missions:
        # Fan-in barrier: this node has two incoming edges (player_agent_2 and
        # mission_master). LangGraph schedules it once per completed predecessor,
        # so the first firing may see only one branch's writes. Return an empty
        # delta so we don't poison party_state — the second firing will have
        # everything and run for real.
        return {}

    ps = state.party_state or _build_initial_party_state(world)
    graph = WorldGraph(world)
    new_messages: list[AIMessage] = []

    _stream("\n" + "=" * 94)
    _stream(" LIVE GAMEPLAY")
    _stream("=" * 94)
    _render_party_map(world, ps.current_room)

    while not ps.game_over and ps.tick < MAX_TICKS:
        mission = _current_mission(state.missions, ps)
        if mission is None:
            ps.game_over = True
            ps.victory = True
            _stream(f"  >>> VICTORY — all missions complete at tick {ps.tick}")
            new_messages.append(
                AIMessage(content=f"[gameplay] VICTORY at tick {ps.tick}")
            )
            break

        # Walk one step along the shortest path to the mission's room
        if mission.room != ps.current_room:
            route = graph.path(ps.current_room, mission.room)
            if len(route) < 2:
                ps.game_over = True
                _stream(
                    f"  >>> No route from {ps.current_room} to {mission.room} — aborting"
                )
                new_messages.append(
                    AIMessage(
                        content=f"[gameplay] No route to {mission.room} from {ps.current_room}"
                    )
                )
                break
            next_step = route[1]
            ps.tick += 1
            ps.current_room = next_step
            ps.visited.add(next_step)
            _stream(
                f"\n  ── Tick {ps.tick}  Party moves to {next_step}  "
                f"(en route to {mission.room}; path: {' -> '.join(route)})"
            )
            _render_party_map(world, ps.current_room)
            continue

        # If a mission has no required actions, auto-complete it without spending a tick
        if not mission.required_actions:
            ps.completed_gates.append(mission.gate_index)
            next_room = _next_room_for_completed_mission(world, mission)
            if next_room is None:
                ps.game_over = True
                ps.victory = True
                _stream(f"  >>> VICTORY (auto, no actions) at tick {ps.tick}")
                new_messages.append(
                    AIMessage(content=f"[gameplay] VICTORY (auto) at tick {ps.tick}")
                )
                break
            ps.current_room = next_room
            ps.visited.add(next_room)
            _stream(
                f"  >>> Auto-completed gate {mission.gate_index}; moved to {next_room}"
            )
            _render_party_map(world, ps.current_room)
            continue

        ps.tick += 1
        _stream(
            f"\n  ── Tick {ps.tick}  [room: {ps.current_room}]  mission: {mission.description[:70]}"
        )
        _stream(
            f"     remaining required actions: "
            f"{[a for a in mission.required_actions if a not in ps.completed_actions_by_gate.get(mission.gate_index, [])]}"
        )

        teammate_last_per_agent = {m.agent_id: None for m in state.party}
        for entry in reversed(ps.log):
            if teammate_last_per_agent.get(entry.agent_id) is None:
                teammate_last_per_agent[entry.agent_id] = entry
            if all(v is not None for v in teammate_last_per_agent.values()):
                break

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
            remaining = [
                a
                for a in mission.required_actions
                if a not in ps.completed_actions_by_gate.get(mission.gate_index, [])
            ]
            action_space = _build_action_space(remaining)
            decided = _agent_act(
                member.agent_id, member, world, ps, mission, action_space, teammate
            )

            chosen = decided["action"]
            matched = chosen if chosen in remaining else None

            if matched:
                completed_list = ps.completed_actions_by_gate.setdefault(
                    mission.gate_index, []
                )
                if matched not in completed_list:
                    completed_list.append(matched)
                    note = f"completed required action '{matched}'"
                else:
                    note = f"redundant: '{matched}' already done"
            elif chosen == IDLE_ACTION:
                note = "idle"
            else:
                note = "no effect"

            tick_actions.append(
                TickAction(
                    tick=ps.tick,
                    agent_id=member.agent_id,
                    say=decided["say"],
                    action=decided["action"],
                    matched_required_action=matched,
                    note=note,
                )
            )

            marker = "✓" if matched else "·"
            label = f"{member.agent_id} ({member.character.name})"
            _stream(f"     {marker} {label}")
            _stream(f"         say   : \"{decided['say']}\"")
            _stream(f"         action: {decided['action']}  ({note})")

        ps.log.extend(tick_actions)

        # Check mission completion after both agents acted
        done = ps.completed_actions_by_gate.get(mission.gate_index, [])
        if mission.required_actions and set(done) >= set(mission.required_actions):
            ps.completed_gates.append(mission.gate_index)
            _stream(f"  >>> Mission complete: gate {mission.gate_index}")

            if mission.reward_item:
                room = next((r for r in world.rooms if r.name == mission.room), None)
                reward = next(
                    (
                        i
                        for i in (room.items if room else [])
                        if i.name == mission.reward_item
                    ),
                    None,
                )
                if reward and not any(i.name == reward.name for i in ps.inventory):
                    ps.inventory.append(reward)
                    _stream(f"  >>> Party picked up: {reward.name}")
                    new_messages.append(
                        AIMessage(content=f"[gameplay] Party picked up {reward.name}")
                    )

            next_room = _next_room_for_completed_mission(world, mission)
            if next_room is None:
                ps.game_over = True
                ps.victory = True
                _stream(f"  >>> VICTORY at tick {ps.tick}")
                new_messages.append(
                    AIMessage(content=f"[gameplay] VICTORY at tick {ps.tick}")
                )
            else:
                ps.current_room = next_room
                ps.visited.add(next_room)
                _stream(f"  >>> Party moved to {next_room}")
                new_messages.append(
                    AIMessage(
                        content=f"[gameplay] Party moved to {next_room} at tick {ps.tick}"
                    )
                )
                _render_party_map(world, ps.current_room)

    if not ps.game_over:
        ps.game_over = True
        _stream(f"  >>> Stopped: hit MAX_TICKS={MAX_TICKS} without victory")
        new_messages.append(
            AIMessage(content=f"[gameplay] Stopped at MAX_TICKS={MAX_TICKS}")
        )

    return {
        "messages": new_messages,
        "party_state": ps,
    }
