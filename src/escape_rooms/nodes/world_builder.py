"""World Builder agent — generates the escape room setting: scenario, objective, and rooms.

Produces a GameWorld with rooms and goals only. Objects and the solution path are
built by puzzle_builder_node in the next graph step.
"""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.escape_rooms.utils.settings import Settings, get_llm
from src.escape_rooms.utils.logging import get_node_logger
from src.escape_rooms.prompts import load_prompt
from src.escape_rooms.state import GameState, GameWorld, Prerequisite, Room

log = get_node_logger("world_builder")

SYSTEM_PROMPT = load_prompt("world_builder", "system")
GENERATION_PROMPT = load_prompt("world_builder", "generation")
BANK_GENERATION_PROMPT = load_prompt("world_builder", "generation_bank")
REPAIR_PROMPT = load_prompt("world_builder", "repair")

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}

_VALID_PREREQ_TYPES = {"object_state", "known_info", "has_item", "power_active"}


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
            result = json.loads(brace_match.group())
            log.trace("_parse_json: extracted JSON via brace-match fallback")
            return result
        except json.JSONDecodeError:
            log.warning("_parse_json: all JSON parse strategies failed (len={})", len(text))
            return None
    log.warning("_parse_json: no JSON structure found in LLM response (len={})", len(text))
    return None


def _build_prerequisite(raw) -> Prerequisite | None:
    if not isinstance(raw, dict):
        return None
    ptype = raw.get("type")
    if ptype not in _VALID_PREREQ_TYPES:
        log.trace("_build_prerequisite: unknown type {!r}, skipping", ptype)
        return None
    return Prerequisite(
        type=ptype,
        object_id=raw.get("object_id"),
        state=raw.get("state"),
        info=raw.get("info"),
        id=raw.get("id"),
    )


def _build_rooms(raw_rooms: list) -> list[Room]:
    rooms: list[Room] = []
    for raw_room in raw_rooms:
        if isinstance(raw_room, str):
            log.trace("_build_rooms: bare string room id={!r}", raw_room)
            rooms.append(Room(id=raw_room, description="", adjacency={}))
            continue
        if not isinstance(raw_room, dict):
            log.trace("_build_rooms: skipping non-dict entry: {!r}", raw_room)
            continue
        room_id = raw_room.get("id") or raw_room.get("name")
        if not room_id:
            log.trace("_build_rooms: skipping room with no id: {!r}", raw_room)
            continue
        raw_adj = raw_room.get("adjacency", {})
        adjacency = (
            {k: v for k, v in raw_adj.items() if isinstance(v, str) and v}
            if isinstance(raw_adj, dict)
            else {}
        )
        key_objects_raw = raw_room.get("key_objects", [])
        key_objects = (
            [k for k in key_objects_raw if isinstance(k, str)]
            if isinstance(key_objects_raw, list)
            else []
        )
        room = Room(
            id=room_id,
            description=raw_room.get("description", ""),
            adjacency=adjacency,
            goal=raw_room.get("goal", ""),
            goal_completion=_build_prerequisite(raw_room.get("goal_completion")),
            key_objects=key_objects,
        )
        log.trace(
            "_build_rooms: parsed room id={!r} adj={} goal_completion={}",
            room.id,
            list(room.adjacency.keys()),
            room.goal_completion.type if room.goal_completion else "none",
        )
        rooms.append(room)
    log.debug("_build_rooms: built {} room(s)", len(rooms))
    return rooms


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    log.debug("_repair_adjacency: checking {} room(s)", len(rooms))
    known = {r.id for r in rooms}

    cleaned: dict[str, dict[str, str]] = {}
    for room in rooms:
        cleaned[room.id] = {
            d: n for d, n in room.adjacency.items() if n in known and d in OPPOSITES
        }

    for room_id, adj in list(cleaned.items()):
        for direction, neighbor_id in adj.items():
            reverse = OPPOSITES[direction]
            neighbor_adj = cleaned.get(neighbor_id, {})
            if reverse not in neighbor_adj:
                log.debug(
                    "_repair_adjacency: adding missing reverse edge {} ←{} {}",
                    neighbor_id, reverse, room_id,
                )
                neighbor_adj[reverse] = room_id
                cleaned[neighbor_id] = neighbor_adj

    return [
        Room(
            id=r.id,
            description=r.description,
            adjacency=cleaned.get(r.id, {}),
            goal=r.goal,
            goal_completion=r.goal_completion,
            key_objects=r.key_objects,
        )
        for r in rooms
    ]


MAX_ROOMS = 2


def _select_rooms(rooms: list[Room], limit: int) -> list[Room]:
    """Truncate to `limit` rooms, keeping the start and last rooms."""
    if len(rooms) <= limit:
        return rooms
    anchors = {rooms[0].id, rooms[-1].id}
    anchored = [r for r in rooms if r.id in anchors]
    rest = [r for r in rooms if r.id not in anchors]
    return (anchored + rest)[:limit]


def _build_world(data: dict, max_rooms: int) -> GameWorld:
    """Parse and validate room/scenario data into a rooms-only GameWorld skeleton."""
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    if max_rooms > 0 and len(rooms) > max_rooms:
        rooms = _select_rooms(rooms, max_rooms)
        kept_ids = {r.id for r in rooms}
        for r in rooms:
            r.adjacency = {d: n for d, n in r.adjacency.items() if n in kept_ids}
        rooms = _repair_adjacency(rooms)

    return GameWorld(
        scenario=data.get("scenario", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
        objects=[],
        rules=[],
        solution_path=[],
    )


def _generation_prompt(theme: str) -> str:
    s = Settings()
    if s.hard_mode:
        return BANK_GENERATION_PROMPT.format(
            theme=theme,
            num_rooms=s.num_rooms,
            chain_depth=s.chain_depth,
        )
    return GENERATION_PROMPT.format(theme=theme)


def _generate_world(llm, theme: str, max_rooms: int) -> tuple[GameWorld, str]:
    log.info("Calling LLM to generate world — theme={!r} max_rooms={}", theme, max_rooms)
    t0 = time.perf_counter()
    prompt = _generation_prompt(theme)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    elapsed = time.perf_counter() - t0
    log.debug("LLM response received in {:.2f}s — {} chars", elapsed, len(response.content))
    log.trace("LLM raw response:\n{}", response.content[:2000])
    data = _parse_json(response.content) or {}
    world = _build_world(data, max_rooms)
    log.debug(
        "World parsed — {} room(s), scenario={!r}",
        len(world.rooms),
        world.scenario[:60] if world.scenario else "",
    )
    return world, response.content


def _room_ids_from_issues(issues: list[str]) -> set[str]:
    """Extract room ids mentioned in issue strings (format: "room '<id>': ...")."""
    ids: set[str] = set()
    for issue in issues:
        m = re.search(r"room '([^']+)'", issue)
        if m:
            ids.add(m.group(1))
    return ids


def _repair_world(llm, world: GameWorld, issues: list[str]) -> tuple[GameWorld, str]:
    """Ask the LLM to fix only the rooms mentioned in `issues`, then merge back."""
    log.info("Surgical repair — {} issue(s): {}", len(issues), issues)
    broken_ids = _room_ids_from_issues(issues)

    # Issues with no room reference (e.g. "missing scenario") can't be targeted —
    # fall back to a full regeneration for those; here we repair what we can.
    rooms_to_fix = [r for r in world.rooms if r.id in broken_ids]
    if not rooms_to_fix:
        log.debug("_repair_world: no room-level targets found — skipping surgical repair")
        return world, ""
    log.debug("_repair_world: targeting rooms {}", [r.id for r in rooms_to_fix])

    repairs_payload = [
        {"id": r.id, "issues": [i for i in issues if f"room '{r.id}'" in i]}
        for r in rooms_to_fix
    ]

    world_json = json.dumps(
        {
            "scenario": world.scenario,
            "objective": world.objective,
            "rooms": [
                {
                    "id": r.id,
                    "description": r.description,
                    "adjacency": r.adjacency,
                    "goal": r.goal,
                    "goal_completion": (
                        r.goal_completion.model_dump(exclude_none=True)
                        if r.goal_completion
                        else None
                    ),
                    "key_objects": r.key_objects,
                }
                for r in world.rooms
            ],
        },
        indent=2,
    )

    prompt = REPAIR_PROMPT.format(
        world_json=world_json,
        repairs_json=json.dumps(repairs_payload, indent=2),
    )
    log.info("Calling LLM for surgical room repair — {} room(s)", len(rooms_to_fix))
    t0 = time.perf_counter()
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    log.debug("Repair LLM response in {:.2f}s — {} chars", time.perf_counter() - t0, len(response.content))
    log.trace("Repair LLM raw response:\n{}", response.content[:2000])
    data = _parse_json(response.content) or {}
    fixed_rooms = _build_rooms(data.get("rooms", []))
    log.debug("_repair_world: LLM returned {} fixed room(s)", len(fixed_rooms))

    # Merge: replace only the rooms that were sent for repair.
    fixed_by_id = {r.id: r for r in fixed_rooms}
    merged_rooms = [fixed_by_id.get(r.id, r) for r in world.rooms]
    merged_rooms = _repair_adjacency(merged_rooms)

    # win_condition is a computed property derived from the final room's
    # goal_completion — it is never set explicitly; it follows from merged_rooms.
    repaired = GameWorld(
        scenario=world.scenario,
        objective=world.objective,
        rooms=merged_rooms,
        objects=world.objects,
        rules=world.rules,
        solution_path=world.solution_path,
    )
    return repaired, response.content


def repair_world(world: GameWorld, issues: list[str], llm=None) -> tuple[GameWorld, str]:
    """Public entry to surgically repair the rooms named in `issues`.

    Lets other nodes (e.g. puzzle_builder, when a declared key object cannot be
    materialised) revise just the offending room skeletons without importing the
    private helper or wiring up an LLM themselves.
    """
    if llm is None:
        llm = get_llm("game_master")
    return _repair_world(llm, world, issues)


# ---------------------------------------------------------------------------
# Eval A — deterministic structural check on the rooms-only world skeleton
# ---------------------------------------------------------------------------

_OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _eval_world_structure(world: GameWorld, expected_rooms: int) -> list[str]:
    """Return structural violation strings for the rooms-only world skeleton.

    Only checks what world_builder owns: room count, metadata, and adjacency
    topology. Object-level checks belong to eval B in puzzle_builder_node.
    """
    issues: list[str] = []

    if not world.scenario:
        issues.append("missing scenario")
    if not world.objective:
        issues.append("missing objective")

    if len(world.rooms) != expected_rooms:
        issues.append(f"expected {expected_rooms} room(s), got {len(world.rooms)}")

    seen_ids: set[str] = set()
    for r in world.rooms:
        if r.id in seen_ids:
            issues.append(f"duplicate room id: '{r.id}'")
        seen_ids.add(r.id)

    room_ids = {r.id for r in world.rooms}
    for r in world.rooms:
        if not r.description:
            issues.append(f"room '{r.id}': missing description")
        if not r.goal:
            issues.append(f"room '{r.id}': missing goal")
        if r.goal_completion is None:
            issues.append(f"room '{r.id}': missing goal_completion")
        elif r.goal_completion.object_id:
            # The goal must revolve around a key object: the object the
            # goal_completion checks should be one of the room's key_objects.
            if r.goal_completion.object_id not in r.key_objects:
                issues.append(
                    f"room '{r.id}': goal_completion object_id "
                    f"'{r.goal_completion.object_id}' is not in key_objects"
                )
        for direction, neighbor_id in r.adjacency.items():
            if neighbor_id not in room_ids:
                issues.append(
                    f"room '{r.id}': adjacency '{direction}' → '{neighbor_id}' does not exist"
                )
            else:
                reverse = _OPPOSITES.get(direction)
                neighbor = next((x for x in world.rooms if x.id == neighbor_id), None)
                if reverse and neighbor and neighbor.adjacency.get(reverse) != r.id:
                    issues.append(
                        f"room '{r.id}': adjacency not mirrored — "
                        f"'{neighbor_id}' missing '{reverse}' → '{r.id}'"
                    )

    return issues


def world_builder_node(state: GameState) -> dict:
    s = Settings()
    llm = get_llm("game_master")
    max_rooms = s.num_rooms if s.hard_mode else MAX_ROOMS
    mode = "HARD" if s.hard_mode else "standard"
    max_attempts = s.gen_max_attempts if s.gen_max_attempts > 0 else 1

    log.info("Starting — theme={!r} mode={} max_rooms={} max_attempts={}", state.theme, mode, max_rooms, max_attempts)

    attempt_log: list[dict] = []

    start = time.perf_counter()
    world, raw = _generate_world(llm, state.theme, max_rooms)

    attempts = 1
    issues = _eval_world_structure(world, max_rooms)
    log.debug("Structural eval after attempt {}: {} issue(s)", attempts, len(issues))

    while issues and attempts < max_attempts:
        attempt_log.append({"attempt": attempts, "issues": issues})
        log.warning("Attempt {} rejected — {} structural issue(s):", attempts, len(issues))
        for issue in issues:
            log.warning("  • {}", issue)

        broken_ids = _room_ids_from_issues(issues)
        if broken_ids:
            log.info(
                "Repairing room(s) {} (attempt {}/{})...",
                sorted(broken_ids), attempts + 1, max_attempts,
            )
            world, raw = _repair_world(llm, world, issues)
        else:
            log.info("Regenerating world (attempt {}/{})...", attempts + 1, max_attempts)
            world, raw = _generate_world(llm, state.theme, max_rooms)

        attempts += 1
        issues = _eval_world_structure(world, max_rooms)
        log.debug("Structural eval after attempt {}: {} issue(s)", attempts, len(issues))

    elapsed = time.perf_counter() - start

    if issues:
        log.warning(
            "{} issue(s) remain after {} attempt(s):", len(issues), attempts,
        )
        for issue in issues:
            log.warning("  • {}", issue)
    else:
        log.success(
            "Generated {} world — {} room(s), {} attempt(s), {:.2f}s",
            mode, len(world.rooms), attempts, elapsed,
        )
        for room in world.rooms:
            log.debug(
                "  Room {!r} — adj={} goal={!r} key_objects={}",
                room.id, list(room.adjacency.keys()), room.goal[:60] if room.goal else "", room.key_objects,
            )

    messages: list[AIMessage] = []
    for entry in attempt_log:
        header = f"=== ATTEMPT {entry['attempt']} (rejected) ===\n" + "\n".join(
            f"  • {i}" for i in entry["issues"]
        )
        messages.append(AIMessage(content=header))
    messages.append(AIMessage(content=f"=== ATTEMPT {attempts} (final) ===\n\n{raw}"))

    loggable_attempts = attempt_log

    return {
        "messages": messages,
        "world": world,
        "_attempt_log": loggable_attempts,
    }
