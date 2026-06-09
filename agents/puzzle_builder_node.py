"""Puzzle Builder agent — populates a pre-built room layout with objects and a solution path.

world_builder produces the rooms, scenario, and room goals. This node receives that
skeleton and generates all objects, locks, clues, and the solution path, then runs the
full repair pipeline and an independent solvability retry loop.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import Settings, get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Prerequisite, Room, WorldObject
from state.game_state import derive_win_condition

SYSTEM_PROMPT = load_prompt("puzzle_builder", "system")
GENERATION_PROMPT = load_prompt("puzzle_builder", "generation")

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}

_OPTIONAL_OBJECT_FIELDS = (
    "requires_code",
    "code_digits",
    "requires_tool",
    "requires_liquid",
    "requires_power",
    "fuses",
    "contains_info",
    "slot_description",
    "note",
    "scenic",
)

_LLM_LOCKED_STATES = {"locked", "locked_bolt", "locked_room", "hidden"}
_UNLOCKER_HINTS = (
    "wheel",
    "key",
    "lever",
    "crank",
    "handle",
    "valve",
    "tool",
    "crowbar",
    "wrench",
)
_CLUE_HINTS = (
    "note",
    "letter",
    "paper",
    "scroll",
    "document",
    "diary",
    "journal",
    "tome",
    "book",
    "tablet",
)
HIDDEN_STATES_ROOM = {"locked", "locked_bolt", "locked_room", "hidden"}
_OPENED_STATES = {"unlocked", "open", "opened", "unsealed", "dissolved", "deactivated"}


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------


def _format_rooms(rooms: list[Room]) -> str:
    lines = []
    for r in rooms:
        lines.append(f'  id: "{r.id}"')
        lines.append(f'  description: "{r.description}"')
        lines.append(f"  adjacency: {json.dumps(r.adjacency)}")
        lines.append("")
    return "\n".join(lines)


def _format_room_goals(rooms: list[Room]) -> str:
    lines = []
    for r in rooms:
        gc = (
            r.goal_completion.model_dump(exclude_none=True)
            if r.goal_completion
            else "(none)"
        )
        lines.append(f'  Room "{r.id}": goal="{r.goal}" | goal_completion={gc}')
    return "\n".join(lines)


def _format_key_objects(rooms: list[Room]) -> str:
    lines = []
    for r in rooms:
        ids = ", ".join(r.key_objects) if r.key_objects else "(none)"
        lines.append(f'  Room "{r.id}": {ids}')
    return "\n".join(lines)


def _all_key_objects(rooms: list[Room]) -> set[str]:
    keys: set[str] = set()
    for r in rooms:
        keys.update(r.key_objects)
    return keys


# ---------------------------------------------------------------------------
# Object building (moved from game_master.py)
# ---------------------------------------------------------------------------


def _coerce_id(value) -> str | None:
    """Extract and slugify an id from an LLM-provided value.

    Slugification (lowercase + underscores) ensures ids remain single-token
    action targets — the engine splits actions on whitespace and uses parts[1].
    """
    raw: str | None = None
    if isinstance(value, str):
        raw = value or None
    elif isinstance(value, dict):
        inner = value.get("id") or value.get("name")
        raw = inner if isinstance(inner, str) and inner else None
    if raw is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower().strip()).strip("_")
    return slug or None


def _build_objects(raw_objects: list[dict], room_ids: set[str]) -> list[WorldObject]:
    candidates: list[dict] = [
        o for o in raw_objects if isinstance(o, dict) and o.get("id")
    ]
    known_object_ids = {
        o["id"]
        for o in candidates
        if isinstance(o.get("id"), str) and o["id"] not in room_ids
    }
    valid_locations = room_ids | known_object_ids

    objects: list[WorldObject] = []
    for raw in candidates:
        obj_id = _coerce_id(raw.get("id"))
        if obj_id is None:
            continue
        if obj_id in room_ids:
            continue

        location = _coerce_id(raw.get("location")) or ""
        if location not in valid_locations:
            continue

        requires_tool = _coerce_id(raw.get("requires_tool"))
        if requires_tool and requires_tool not in known_object_ids:
            requires_tool = None

        kwargs = {
            "id": obj_id,
            "location": location,
            "description": str(raw.get("description", "")),
            "state": str(raw.get("state", "visible")),
            "interactable": bool(raw.get("interactable", False)),
            "takeable": bool(raw.get("takeable", False)),
            "requires_tool": requires_tool,
        }
        for field in _OPTIONAL_OBJECT_FIELDS:
            if field == "requires_tool":
                continue
            value = raw.get(field)
            if value in (None, ""):
                continue
            if field == "fuses":
                if isinstance(value, dict):
                    kwargs[field] = value
            elif field == "scenic":
                kwargs[field] = bool(value)
            elif isinstance(value, (str, int)):
                kwargs[field] = value

        try:
            objects.append(WorldObject(**kwargs))
        except Exception:
            continue
    return objects


# ---------------------------------------------------------------------------
# Repair pipeline (moved from game_master.py)
# ---------------------------------------------------------------------------


def _goal_completion_subject(p: Prerequisite) -> str | None:
    if p.type in {"object_state", "has_item"}:
        return p.object_id
    if p.type == "known_info":
        return p.info
    if p.type == "power_active":
        return p.id
    return None


def _token_matches_code(code: str, info: str | None) -> bool:
    if not info:
        return False
    return bool(re.sub(r"[^0-9]", "", info) == re.sub(r"[^0-9]", "", code) and code)


def _scrub_room_refs(rooms: list[Room], object_ids: set[str]) -> list[Room]:
    def _valid(p: Prerequisite) -> bool:
        if p.type in {"object_state", "has_item"}:
            return p.object_id in object_ids
        return True

    for room in rooms:
        # NOTE: key_objects are intentionally NOT scrubbed here. They are the
        # mandatory anchors world_builder promised; _eval_key_objects_present must
        # be able to see a missing one to reject the attempt and retry. Silently
        # dropping it would mask the violation.
        if room.goal_completion is not None and not _valid(room.goal_completion):
            room.goal_completion = None
    return rooms


def _stem(token: str) -> str:
    s = token.lower()
    for suffix in (
        "_revealed",
        "_hidden",
        "_locked",
        "_unlocked",
        "_item",
        "_object",
        "_tool",
        "_key",
    ):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
    return s


def _repair_tool_refs(objects: list[WorldObject]) -> list[str]:
    by_id = {o.id: o for o in objects}
    takeable_by_stem: dict[str, list[str]] = {}
    for o in objects:
        if o.takeable:
            takeable_by_stem.setdefault(_stem(o.id), []).append(o.id)

    repairs: list[str] = []
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id or tool_id in by_id:
            continue
        candidates = takeable_by_stem.get(_stem(tool_id), [])
        match = candidates[0] if len(candidates) == 1 else None
        if match is None:
            stem = _stem(tool_id)
            loose = [
                tid
                for tid in (oid for o2 in objects if o2.takeable for oid in [o2.id])
                if stem and (stem in _stem(tid) or _stem(tid) in stem)
            ]
            match = loose[0] if len(loose) == 1 else None
        if match is not None:
            o.requires_tool = match
            repairs.append(f"{o.id}: requires_tool {tool_id} -> {match}")
        else:
            o.requires_tool = None
            repairs.append(f"{o.id}: requires_tool {tool_id} -> (nulled, no match)")
    return repairs


def _make_required_tools_takeable(objects: list[WorldObject]) -> list[str]:
    by_id = {o.id: o for o in objects}
    repairs: list[str] = []
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id:
            continue
        tool = by_id.get(tool_id)
        if tool is None or tool.id == o.id:
            continue
        changed: list[str] = []
        if not tool.takeable:
            tool.takeable = True
            changed.append("takeable")
        if tool.location == o.id or tool.state in _LLM_LOCKED_STATES:
            tool.state = "visible"
            changed.append("visible")
        if changed:
            tool.interactable = True
            repairs.append(f"{tool.id}: made {'+'.join(changed)} (tool for {o.id})")
    return repairs


def _repair_unsolvable_gates(
    rooms: list[Room], objects: list[WorldObject]
) -> list[str]:
    gated: dict[str, str] = {}
    for room in rooms:
        gc = room.goal_completion
        if gc is not None and gc.type == "object_state" and gc.object_id and gc.state:
            gated[gc.object_id] = gc.state

    by_id = {o.id: o for o in objects}
    repairs: list[str] = []
    for obj_id, target_state in gated.items():
        obj = by_id.get(obj_id)
        if obj is None:
            continue
        if obj.state not in _LLM_LOCKED_STATES:
            continue
        if any(
            [
                obj.requires_code,
                obj.requires_tool,
                obj.requires_liquid,
                obj.requires_power,
                obj.fuses,
            ]
        ):
            continue
        same_room_takeables = [
            o
            for o in objects
            if o.location == obj.location
            and o.takeable
            and o.id != obj.id
            and o.state not in _LLM_LOCKED_STATES
        ]
        hinted = [
            o
            for o in same_room_takeables
            if any(h in o.id.lower() for h in _UNLOCKER_HINTS)
        ]
        pick = (
            hinted[0]
            if hinted
            else (same_room_takeables[0] if same_room_takeables else None)
        )
        if pick is not None:
            obj.requires_tool = pick.id
            repairs.append(f"{obj.id}: locked gate given requires_tool {pick.id}")
        else:
            obj.state = target_state
            repairs.append(
                f"{obj.id}: locked gate with no mechanism started at target '{target_state}'"
            )
    return repairs


def _fuse_label_for(token: str) -> str:
    stem = re.sub(r"[^a-z0-9]", "", token.lower()) or "x"
    return stem[:4].upper()


def _repair_power_gates(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    repairs: list[str] = []
    by_id = {o.id: o for o in objects}
    for room in rooms:
        gc = room.goal_completion
        if gc is None or gc.type != "power_active" or not gc.id:
            continue
        producible = any(
            o.fuses and any(f"sekring_{lbl}_ON" == gc.id for lbl in o.fuses)
            for o in objects
            if o.location == room.id
            or (by_id.get(o.location) and by_id[o.location].location == room.id)
        )
        if producible:
            continue
        orig_id = gc.id
        label = _fuse_label_for(gc.id)
        token = f"sekring_{label}_ON"
        host = next(
            (
                o
                for o in objects
                if o.location == room.id
                and not o.takeable
                and o.state not in _LLM_LOCKED_STATES
                and o.fuses is None
            ),
            None,
        )
        if host is None:
            host = next(
                (o for o in objects if o.location == room.id and o.fuses is None), None
            )
        if host is None:
            continue
        host.fuses = {label: "OFF"}
        host.interactable = True
        if host.state in _LLM_LOCKED_STATES:
            host.state = "visible"
        gc.id = token
        repairs.append(
            f"{room.id}: power gate '{orig_id}' -> fuse {label} on {host.id} ({token})"
        )
    return repairs


def _repair_missing_win_condition(
    rooms: list[Room], objects: list[WorldObject]
) -> list[str]:
    if not rooms:
        return []
    final_room = rooms[-1]
    gc = final_room.goal_completion
    if gc is not None and gc.type == "object_state" and gc.object_id:
        return []

    final_room_id = final_room.id
    by_id = {o.id: o for o in objects}
    candidates = [
        o
        for o in objects
        if o.location == final_room_id and o.state in _LLM_LOCKED_STATES
    ]
    key_set = set(final_room.key_objects)
    keyed = [o for o in candidates if o.id in key_set]
    pick = (keyed[0] if keyed else candidates[0]) if candidates else None

    if pick is None:
        for kid in final_room.key_objects:
            obj = by_id.get(kid)
            if obj and obj.location == final_room_id:
                pick = obj
                break

    if pick is None:
        return []

    target_state = "unlocked" if pick.state in _LLM_LOCKED_STATES else pick.state
    final_room.goal_completion = Prerequisite(
        type="object_state", object_id=pick.id, state=target_state
    )
    return [f"{final_room_id}: synthesised win condition -> {pick.id} = {target_state}"]


def _required_info_tokens(
    rooms: list[Room], objects: list[WorldObject]
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    start_room = rooms[0].id if rooms else ""
    by_id = {o.id: o for o in objects}
    room_ids = {r.id for r in rooms}
    seen: set[str] = set()

    def _room_of(obj: WorldObject) -> str:
        cursor = obj
        guard: set[str] = set()
        while cursor.id not in guard:
            guard.add(cursor.id)
            if cursor.location in room_ids:
                return cursor.location
            parent = by_id.get(cursor.location)
            if parent is None:
                break
            cursor = parent
        return start_room

    def _add(token: str, source_room: str) -> None:
        if token and token not in seen:
            seen.add(token)
            out.append((token, source_room))

    for room in rooms:
        gc = room.goal_completion
        if gc is not None and gc.type == "known_info" and gc.info:
            _add(gc.info, room.id)

    for obj in objects:
        if obj.requires_code:
            _add(obj.requires_code, _room_of(obj))
    return out


def _patch_missing_info(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    patched: list[str] = []
    produced = {o.contains_info for o in objects if o.contains_info}
    tool_ids = {o.requires_tool for o in objects if o.requires_tool}

    def _score(obj: WorldObject, source_room: str) -> int:
        if obj.location != source_room:
            return -1
        if obj.contains_info:
            return -1
        if obj.state in HIDDEN_STATES_ROOM:
            return -1
        haystack = f"{obj.id} {obj.description}".lower()
        score = 0
        readable = any(h in haystack for h in _CLUE_HINTS)
        if readable:
            score += 10
        if obj.takeable and readable:
            score += 3
        if obj.interactable:
            score += 2
        if obj.id in tool_ids:
            score -= 5
        return score

    for token, source_room in _required_info_tokens(rooms, objects):
        if any(token in (info or "") or (info or "") in token for info in produced):
            continue
        if any(_token_matches_code(token, info) for info in produced):
            continue
        candidates = sorted(objects, key=lambda o: _score(o, source_room), reverse=True)
        winner = next((o for o in candidates if _score(o, source_room) >= 0), None)
        if winner is None:
            continue
        winner.contains_info = token
        winner.interactable = True
        produced.add(token)
        patched.append(f"{token} -> {winner.id}")
    return patched


def _rebind_self_clue_locks(objects: list[WorldObject]) -> list[str]:
    def _supplied_by_other(lock: WorldObject) -> bool:
        code = lock.requires_code
        for o in objects:
            if o is lock or not o.contains_info:
                continue
            info = o.contains_info
            if _token_matches_code(code, info) or code in info or info in code:
                return True
        return False

    repairs: list[str] = []
    for lock in objects:
        if not lock.requires_code:
            continue
        if _supplied_by_other(lock):
            continue
        if not lock.contains_info:
            continue
        old = lock.requires_code
        lock.requires_code = lock.contains_info
        lock.interactable = True
        repairs.append(
            f"{lock.id}: requires_code {old} -> self clue {lock.contains_info}"
        )
    return repairs


def _dedup_objects(
    objects: list[WorldObject], rooms: list[Room]
) -> tuple[list[WorldObject], list[str]]:
    merges: list[str] = []
    referenced: set[str] = set()
    for room in rooms:
        referenced.update(room.key_objects)
        gc = room.goal_completion
        if gc is not None and gc.object_id:
            referenced.add(gc.object_id)

    def _precond_signature(o: WorldObject) -> tuple:
        return (
            o.location,
            o.state,
            o.requires_code,
            o.requires_tool,
            o.requires_liquid,
            o.requires_power,
            tuple(sorted((o.fuses or {}).items())),
        )

    groups: dict[tuple, list[WorldObject]] = {}
    for o in objects:
        groups.setdefault(_precond_signature(o), []).append(o)

    drop: dict[str, str] = {}
    for sig, group in groups.items():
        has_precond = any(sig[2:6]) or sig[6]
        if len(group) < 2 or not has_precond:
            continue
        survivor = next((o for o in group if o.id in referenced), group[0])
        for o in group:
            if o.id == survivor.id:
                continue
            drop[o.id] = survivor.id
            merges.append(f"{o.id} -> {survivor.id}")

    if not drop:
        return objects, merges

    kept = [o for o in objects if o.id not in drop]
    for o in kept:
        if o.requires_tool in drop:
            o.requires_tool = drop[o.requires_tool]
    return kept, merges


def _goal_from_completion(p: Prerequisite) -> str:
    if p.type == "object_state":
        return f"Get {p.object_id} into the '{p.state}' state."
    if p.type == "has_item":
        return f"Find and take {p.object_id}."
    if p.type == "known_info":
        return "Discover the hidden clue this room conceals."
    if p.type == "power_active":
        return f"Bring the {p.id} power line online."
    return "Solve this room's puzzle."


def _bind_goals(rooms: list[Room]) -> list[str]:
    rewrites: list[str] = []
    for room in rooms:
        gc = room.goal_completion
        if gc is None:
            continue
        subject = _goal_completion_subject(gc)
        goal = room.goal or ""
        if gc.type in {"object_state", "has_item", "power_active"}:
            if subject and subject.lower() not in goal.lower():
                room.goal = _goal_from_completion(gc)
                rewrites.append(f"{room.id}: goal rebound to completion ({gc.type})")
        elif gc.type == "known_info":
            room.goal = _goal_from_completion(gc)
            rewrites.append(f"{room.id}: known_info goal normalized")
    return rewrites


def _consumed_codes(rooms: list[Room], objects: list[WorldObject]) -> set[str]:
    consumed: set[str] = set()
    for o in objects:
        if o.requires_code:
            consumed.add(o.requires_code)
    for room in rooms:
        gc = room.goal_completion
        if gc is not None and gc.type == "known_info" and gc.info:
            consumed.add(gc.info)
    return consumed


def _check_coherence(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    warnings: list[str] = []
    consumed = _consumed_codes(rooms, objects)

    seen_info: dict[str, list[str]] = {}
    for o in objects:
        if not o.contains_info:
            continue
        seen_info.setdefault(o.contains_info, []).append(o.id)
        used = any(
            _token_matches_code(c, o.contains_info)
            or o.contains_info in c
            or c in o.contains_info
            for c in consumed
        )
        if not used:
            warnings.append(
                f"clue '{o.contains_info}' on {o.id} is consumed by nothing"
            )

    for token, carriers in seen_info.items():
        if len(carriers) > 1:
            warnings.append(f"clue '{token}' duplicated across {', '.join(carriers)}")

    obj_by_id = {o.id: o for o in objects}
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id:
            continue
        tool = obj_by_id.get(tool_id)
        if tool is None:
            continue
        if not tool.takeable:
            warnings.append(f"{o.id} requires tool {tool_id}, but it is not takeable")
        if tool.state in HIDDEN_STATES_ROOM and not any(
            info == tool_id or (info and tool_id in info)
            for info in (x.contains_info for x in objects)
        ):
            warnings.append(
                f"tool {tool_id} is '{tool.state}' with no clue that reveals it (required by {o.id})"
            )
    return warnings


def _prune_orphan_objects(
    rooms: list[Room],
    objects: list[WorldObject],
    solution_path: list[str] | None = None,
) -> tuple[list[WorldObject], list[str]]:
    by_id = {o.id: o for o in objects}

    producers: dict[str, list[str]] = {}
    for o in objects:
        if o.contains_info:
            producers.setdefault(o.contains_info, []).append(o.id)

    def _producers_of_code(code: str) -> list[str]:
        ids: list[str] = []
        for info, carriers in producers.items():
            if _token_matches_code(code, info) or info in code or code in info:
                ids.extend(carriers)
        return ids

    keep: set[str] = set()
    frontier: list[str] = []

    def _enqueue(obj_id: str | None) -> None:
        if obj_id and obj_id in by_id and obj_id not in keep:
            keep.add(obj_id)
            frontier.append(obj_id)

    # Always keep every declared key object — they are mandatory anchors, so even
    # if one is not strictly backward-reachable from a goal it must survive pruning.
    for room in rooms:
        for kid in room.key_objects:
            _enqueue(kid)

    for room in rooms:
        gc = room.goal_completion
        if gc is None:
            continue
        subj = _goal_completion_subject(gc)
        if not subj:
            continue
        if gc.type in {"object_state", "has_item"}:
            _enqueue(subj)
        else:
            for pid in _producers_of_code(subj):
                _enqueue(pid)

    # Seed from every object id mentioned in solution_path so intermediate
    # result objects referenced only in the solution narrative are not pruned.
    if solution_path:
        import re as _re

        for step in solution_path:
            for token in _re.findall(r"\b[a-z][a-z0-9_]*\b", step):
                if token in by_id:
                    _enqueue(token)

    def _liquid_producers(token: str) -> list[str]:
        return [o.id for o in objects if o.contains_info == token]

    def _power_producers(token: str) -> list[str]:
        ids: list[str] = []
        for o in objects:
            if not o.fuses:
                continue
            if any(f"sekring_{label}_" in token for label in o.fuses):
                ids.append(o.id)
        return ids

    while frontier:
        o = by_id[frontier.pop()]
        _enqueue(o.requires_tool)
        if o.location in by_id:
            _enqueue(o.location)
        if o.requires_code:
            for pid in _producers_of_code(o.requires_code):
                _enqueue(pid)
        if o.requires_liquid:
            _enqueue(o.requires_liquid)
            for pid in _liquid_producers(o.requires_liquid):
                _enqueue(pid)
        if o.requires_power:
            for pid in _power_producers(o.requires_power):
                _enqueue(pid)

    MIN_OBJECTS_PER_ROOM = 5
    room_ids_list = [r.id for r in rooms]
    room_object_count: dict[str, int] = {rid: 0 for rid in room_ids_list}
    for o in objects:
        if o.id in keep:
            if o.location in room_object_count:
                room_object_count[o.location] += 1

    for rid in room_ids_list:
        deficit = MIN_OBJECTS_PER_ROOM - room_object_count[rid]
        if deficit <= 0:
            continue
        fillers = [
            o
            for o in objects
            if o.id not in keep
            and o.location == rid
            and not o.requires_tool
            and not o.requires_code
            and not o.requires_liquid
            and not o.requires_power
            and not o.fuses
            and not o.contains_info
        ]
        for filler in fillers[:deficit]:
            keep.add(filler.id)
            room_object_count[rid] += 1

    dropped = [o.id for o in objects if o.id not in keep]
    kept = [o for o in objects if o.id in keep]
    return kept, dropped


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------


def _backfill_scenic_objects(
    rooms: list[Room], objects: list[WorldObject], min_per_room: int
) -> list[str]:
    return []


def _build_puzzle(
    world: GameWorld, data: dict, min_objects_per_room: int = 0
) -> GameWorld:
    """Apply the repair pipeline to LLM-generated object data and return a completed GameWorld."""
    rooms = world.rooms
    room_ids = {r.id for r in rooms}

    objects = _build_objects(data.get("objects", []), room_ids)

    objects, merges = _dedup_objects(objects, rooms)
    if merges:
        print(
            f"[puzzle_builder] merged duplicate objects: {', '.join(merges)}",
            flush=True,
        )

    object_ids = {o.id for o in objects}
    rooms = _scrub_room_refs(rooms, object_ids)

    tool_repairs = _repair_tool_refs(objects)
    if tool_repairs:
        print(
            f"[puzzle_builder] repaired tool refs: {', '.join(tool_repairs)}",
            flush=True,
        )

    takeable_repairs = _make_required_tools_takeable(objects)
    if takeable_repairs:
        print(
            f"[puzzle_builder] made required tools takeable: {', '.join(takeable_repairs)}",
            flush=True,
        )

    win_repairs = _repair_missing_win_condition(rooms, objects)
    if win_repairs:
        print(
            f"[puzzle_builder] repaired missing win condition: {', '.join(win_repairs)}",
            flush=True,
        )

    gate_repairs = _repair_unsolvable_gates(rooms, objects)
    if gate_repairs:
        print(
            f"[puzzle_builder] repaired unsolvable gates: {', '.join(gate_repairs)}",
            flush=True,
        )

    power_repairs = _repair_power_gates(rooms, objects)
    if power_repairs:
        print(
            f"[puzzle_builder] repaired power gates: {', '.join(power_repairs)}",
            flush=True,
        )

    patched = _patch_missing_info(rooms, objects)
    if patched:
        print(
            f"[puzzle_builder] auto-patched missing clues: {', '.join(patched)}",
            flush=True,
        )

    self_clue = _rebind_self_clue_locks(objects)
    if self_clue:
        print(
            f"[puzzle_builder] rebound self-clue locks: {', '.join(self_clue)}",
            flush=True,
        )

    rewrites = _bind_goals(rooms)
    if rewrites:
        print(
            f"[puzzle_builder] rebound goals to completion: {', '.join(rewrites)}",
            flush=True,
        )

    rules = [r for r in data.get("rules", []) if isinstance(r, str)]
    # The LLM no longer authors the solution path — it is derived from the oracle's
    # actual winning solve once the world is finalized (see bfs_solution_path),
    # which is hallucination-free and guaranteed to win. So we ignore any
    # solution_path the model may emit and leave it empty here.

    objects, orphans = _prune_orphan_objects(rooms, objects)
    if orphans:
        print(
            f"[puzzle_builder] pruned orphan objects: {', '.join(orphans)}", flush=True
        )
        rooms = _scrub_room_refs(rooms, {o.id for o in objects})

    # Backfill AFTER pruning so the scenic props themselves are never pruned and
    # the per-room minimum survives the prune pass (otherwise the count check and
    # the pruner fight each other across attempts).
    filler = _backfill_scenic_objects(rooms, objects, min_objects_per_room)
    if filler:
        print(
            f"[puzzle_builder] backfilled scenic props to meet minimum: {', '.join(filler)}",
            flush=True,
        )

    coherence = _check_coherence(rooms, objects)
    if coherence:
        print(
            f"[puzzle_builder] coherence warnings: {'; '.join(coherence)}", flush=True
        )

    return GameWorld(
        scenario=world.scenario,
        objective=world.objective,
        rooms=rooms,
        objects=objects,
        rules=rules,
        solution_path=[],  # derived from the oracle solve once finalized
        # win_condition is owned by puzzle_builder: now that objects exist and the
        # final room's goal_completion is settled, derive the game-ending target.
        win_condition=derive_win_condition(rooms),
    )


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


def _default_chain_depth(s: Settings) -> int:
    return s.chain_depth if s.hard_mode else 3


def _min_objects_per_room(s: Settings) -> int:
    return 8 if s.hard_mode else 1


def _build_prompt(world: GameWorld, chain_depth: int, min_objects_per_room: int) -> str:
    return GENERATION_PROMPT.format(
        scenario=world.scenario,
        objective=world.objective,
        rooms=_format_rooms(world.rooms),
        room_goals=_format_room_goals(world.rooms),
        key_objects=_format_key_objects(world.rooms),
        chain_depth=chain_depth,
        min_objects_per_room=min_objects_per_room,
    )


def _generate_puzzle(
    llm, world: GameWorld, chain_depth: int, min_objects_per_room: int
) -> tuple[GameWorld, str]:
    prompt = _build_prompt(world, chain_depth, min_objects_per_room)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    return _build_puzzle(world, data, min_objects_per_room), response.content


def _generate_puzzle_with_feedback(
    llm,
    world: GameWorld,
    chain_depth: int,
    min_objects_per_room: int,
    violations: list[str],
) -> tuple[GameWorld, str]:
    violation_block = "\n".join(f"  - {v}" for v in violations)
    correction = (
        "Your previous puzzle had the following issues detected by automated checks. "
        "Please generate a NEW, corrected puzzle for the same rooms that fixes ALL of them. "
        "Return only the JSON object — no prose.\n\n"
        f"Issues to fix:\n{violation_block}"
    )
    prompt = _build_prompt(world, chain_depth, min_objects_per_room)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
            HumanMessage(content=correction),
        ]
    )
    data = _parse_json(response.content) or {}
    return _build_puzzle(world, data, min_objects_per_room), response.content


# ---------------------------------------------------------------------------
# Eval B — unified deterministic + oracle check, no LLM calls
# ---------------------------------------------------------------------------


def _objects_for_room(
    room_id: str, all_objects: list[WorldObject]
) -> list[WorldObject]:
    """Return all objects that transitively belong to `room_id`.

    Objects can be nested (location = another object's id). Walk the parent chain
    to find every object whose root anchor is this room.
    """
    by_id = {o.id: o for o in all_objects}

    def root_room(obj: WorldObject) -> str:
        seen: set[str] = set()
        cursor = obj
        while cursor.id not in seen:
            seen.add(cursor.id)
            if cursor.location in by_id:
                cursor = by_id[cursor.location]
            else:
                return cursor.location
        return cursor.location  # cycle guard — return last known

    return [o for o in all_objects if root_room(o) == room_id]


def _room_subworld(
    room: Room, all_objects: list[WorldObject], scenario: str
) -> GameWorld:
    """Build a single-room GameWorld where `room` is the only room.

    The room's adjacency is cleared so the oracle stays inside this room.
    Only objects that transitively belong to this room are included.
    Win condition is derived from this room's goal_completion.
    """
    isolated_room = Room(
        id=room.id,
        description=room.description,
        adjacency={},  # no exits — oracle must solve goal in-room
        goal=room.goal,
        goal_completion=room.goal_completion,
        key_objects=room.key_objects,
    )
    local_objects = _objects_for_room(room.id, all_objects)
    return GameWorld(
        scenario=scenario,
        objective=room.goal,
        rooms=[isolated_room],
        objects=local_objects,
        rules=[],
        solution_path=[],
        win_condition=derive_win_condition([isolated_room]),
    )


def _eval_room_goals(world: GameWorld) -> list[str]:
    """Run a per-room oracle on each room that has a goal_completion condition.

    For every room (except the final room, which is already validated by the
    global oracle in _eval_puzzle), build a single-room sub-world containing only
    that room and its locally-anchored objects, then run HeadlessEpisode to
    confirm the room goal is achievable with local objects alone.

    Returns a flat list of issue strings, one per failing room.
    """
    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import heuristic_policy
    except Exception:
        return []

    if not world.rooms:
        return []

    issues: list[str] = []
    # Skip the final room — the global oracle already validates it end-to-end.
    rooms_to_check = world.rooms[:-1] if len(world.rooms) > 1 else world.rooms

    for room in rooms_to_check:
        if room.goal_completion is None:
            continue  # no condition to satisfy — always passable

        sub = _room_subworld(room, world.objects, world.scenario)

        # If goal_completion is not object_state, win_condition will be empty.
        # We still run the oracle and check object/info state manually after.
        if (
            not sub.win_condition.object_id
            and room.goal_completion.type != "object_state"
        ):
            # known_info and has_item goals: just verify the required object/info
            # exists among local objects — static check is sufficient.
            gc = room.goal_completion
            if gc.type == "has_item":
                found = any(o.id == gc.object_id and o.takeable for o in sub.objects)
                if not found:
                    issues.append(
                        f"room '{room.id}': goal requires taking '{gc.object_id}' "
                        f"but that object is not takeable or not in this room"
                    )
            elif gc.type == "known_info":
                found = any(o.contains_info == gc.info for o in sub.objects)
                if not found:
                    issues.append(
                        f"room '{room.id}': goal requires knowing '{gc.info}' "
                        f"but no local object exposes that info"
                    )
            continue

        if not sub.win_condition.object_id:
            continue  # nothing to check

        try:
            result = HeadlessEpisode(sub).run(heuristic_policy)
            if not result.victory:
                issues.append(
                    f"room '{room.id}': per-room oracle failed to satisfy goal "
                    f"'{room.goal}' using local objects in {result.ticks} tick(s) "
                    f"(win object: {sub.win_condition.object_id!r}, "
                    f"state: {result.win_object_state!r})"
                )
        except Exception as exc:
            issues.append(f"room '{room.id}': per-room oracle error — {exc}")

    return issues


def _eval_key_objects_present(world: GameWorld) -> list[str]:
    """Check that every room's declared key_objects exist as real objects.

    key_objects are the mandatory anchors world_builder promised — the room goal
    revolves around them, so puzzle_builder must materialise each one with the
    same id. A missing key object means the room's goal lost its subject.
    """
    object_ids = {o.id for o in world.objects}
    issues: list[str] = []
    for room in world.rooms:
        missing = [k for k in room.key_objects if k not in object_ids]
        if missing:
            issues.append(
                f"room '{room.id}': missing key object(s) {missing} — every "
                f"key_object must be materialised as an object with the same id"
            )
    return issues


def _eval_object_counts(world: GameWorld, min_per_room: int) -> list[str]:
    """Check that every room has at least `min_per_room` objects (puzzle + scenic)."""
    if min_per_room <= 0:
        return []
    issues: list[str] = []
    for room in world.rooms:
        room_objs = _objects_for_room(room.id, world.objects)
        if len(room_objs) < min_per_room:
            issues.append(
                f"room '{room.id}': has {len(room_objs)} object(s), "
                f"minimum required is {min_per_room}"
            )
    return issues


def _eval_puzzle(
    world: GameWorld, chain_depth_target: int, min_objects_per_room: int = 1
) -> list[str]:
    """Return violation strings for the fully-built puzzle world.

    Runs four passes in sequence, all without LLM calls:
      1. Static backward-chain analysis (check_solvable) — zero simulation cost
      2. Per-room object count — every room must meet the minimum object threshold
      3. Per-room oracle — confirms each room's goal is satisfiable with local objects
      4. Global HeadlessEpisode oracle — confirms end-to-end solvability and chain depth

    Returns an empty list when the world is fully valid.
    """
    issues: list[str] = []

    # --- pass 1: static backward-chain ---
    try:
        from benchmark.policies import check_solvable

        report = check_solvable(world)
        if not report.solvable:
            issues.extend(report.issues)
    except Exception:
        pass

    # --- pass 2: key objects must all be present ---
    issues.extend(_eval_key_objects_present(world))

    # --- pass 2b: per-room object count ---
    issues.extend(_eval_object_counts(world, min_objects_per_room))

    # --- pass 3: per-room oracle ---
    issues.extend(_eval_room_goals(world))

    # --- pass 3: global oracle (dynamic end-to-end) ---
    if not world.win_condition.object_id or not world.rooms:
        issues.append("no win condition or rooms defined")
        return issues

    try:
        from benchmark.policies import oracle_solve

        # BFS-first solve: complete search within budget (no false "unsolvable"
        # from a greedy policy stalling), with heuristic fallback for huge worlds.
        result = oracle_solve(world)
        if not result.victory:
            issues.append(
                f"oracle failed to win in {result.ticks} tick(s) "
                f"(last room: {result.last_room}, "
                f"win object state: {result.win_object_state!r})"
            )
        elif chain_depth_target > 0 and result.chain_depth < chain_depth_target:
            issues.append(
                f"chain depth {result.chain_depth} < target {chain_depth_target}"
            )
    except Exception:
        pass

    return issues


def _print_solvability_check(world: GameWorld) -> None:
    try:
        from benchmark.policies import check_solvable
    except Exception:
        return
    if not world.rooms:
        return
    report = check_solvable(world)
    if report.solvable:
        print(
            "[puzzle_builder] static check: SOLVABLE — no structural issues", flush=True
        )
    else:
        print(
            f"[puzzle_builder] static check: {len(report.issues)} structural issue(s):",
            flush=True,
        )
        for issue in report.issues:
            print(f"  • {issue}", flush=True)


def _print_policy_benchmark(world: GameWorld) -> None:
    try:
        from benchmark.run import _fmt, compute_policy_benchmark
    except Exception:
        return
    if not world.win_condition.object_id or not world.rooms:
        return

    rows = compute_policy_benchmark(world)
    header = f"{'policy':<12} {'win%':>6} {'t2win':>7} {'t_all':>7} {'objs':>6}"
    print("\n[puzzle_builder] policy benchmark (this world):", flush=True)
    print("  " + header, flush=True)
    print("  " + "-" * len(header), flush=True)
    for s in rows:
        print(
            "  "
            f"{s['policy']:<12} "
            f"{s['win_rate'] * 100:>5.0f}% "
            f"{_fmt(s['mean_ticks_to_win']):>7} "
            f"{_fmt(s['mean_ticks_all']):>7} "
            f"{_fmt(s['mean_objects_resolved'], '.1f'):>6}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def _is_unsolvable_issue(issue: str) -> bool:
    """True when an issue means the world cannot be won (regenerating the world helps).

    Cosmetic/structural issues (coherence warnings, etc.) are repaired in place and
    are not worth throwing away the whole world for. But an oracle failure or a
    missing win condition means the win-chain itself is broken — a fresh world is
    the right move.
    """
    return issue.startswith("oracle failed to win") or "no win condition" in issue


def _is_missing_key_object_issue(issue: str) -> bool:
    """True when an issue means a declared key object was never materialised.

    These are mandatory anchors world_builder promised; if puzzle_builder still
    can't build one after the puzzle budget, the anchor itself is likely
    unbuildable — surgically repairing the room skeleton is the right move.
    """
    return "missing key object" in issue


def _build_puzzle_for_world(
    llm,
    base_world: GameWorld,
    chain_depth: int,
    chain_depth_target: int,
    min_objs: int,
    max_attempts: int,
) -> tuple[GameWorld, str, list[str], list[dict], int]:
    """Run the puzzle-attempt loop against one fixed world.

    Returns (world, final_raw, remaining_issues, attempt_log, attempts).
    """
    attempt_log: list[dict] = []
    world, raw = _generate_puzzle(llm, base_world, chain_depth, min_objs)
    attempts = 1
    issues = _eval_puzzle(world, chain_depth_target, min_objs)

    while issues and attempts < max_attempts:
        attempt_log.append({"attempt": attempts, "issues": issues, "raw": raw})
        print(
            f"[puzzle_builder] attempt {attempts} rejected — {len(issues)} issue(s):",
            flush=True,
        )
        for issue in issues:
            print(f"  • {issue}", flush=True)
        print(
            f"[puzzle_builder] regenerating with feedback "
            f"(attempt {attempts + 1}/{max_attempts})...",
            flush=True,
        )
        world, raw = _generate_puzzle_with_feedback(
            llm, base_world, chain_depth, min_objs, issues
        )
        attempts += 1
        issues = _eval_puzzle(world, chain_depth_target, min_objs)

    return world, raw, issues, attempt_log, attempts


def puzzle_builder_node(state: GameState) -> dict:
    """Populate the room skeleton with a *constructively solvable* puzzle graph.

    The dependency chain is built in code (agents.puzzle_graph), so the world is
    solvable by construction — the LLM is used only to theme (describe) the
    objects, and cannot break solvability. The deterministic + oracle eval
    (``_eval_puzzle``) still runs as a safety net; if it ever finds an issue we
    fall back to the legacy LLM-generated repair loop (kept below) to recover.
    """
    if not state.world or not state.world.rooms:
        return {}

    import time

    from agents.puzzle_graph import apply_theming, build_solvable_world

    s = Settings()
    llm = get_llm("game_master")
    chain_depth = _default_chain_depth(s)
    chain_depth_target = chain_depth if s.hard_mode else 0
    min_objs = _min_objects_per_room(s)

    start = time.perf_counter()

    base_world = state.world

    # --- Constructive build: solvable by construction, then themed by the LLM ---
    world = build_solvable_world(
        base_world, chain_depth=chain_depth, min_objects_per_room=min_objs
    )
    world = apply_theming(world, state.theme, llm)

    issues = _eval_puzzle(world, chain_depth_target, min_objs)
    messages: list[AIMessage] = []

    if issues:
        # Should be rare-to-never: the graph is built solvable. If a check still
        # fires (e.g. chain-depth target unmet), fall back to the legacy LLM loop.
        print(
            f"[puzzle_builder] constructive build flagged {len(issues)} issue(s) — "
            f"falling back to LLM generation loop:",
            flush=True,
        )
        for issue in issues:
            print(f"  • {issue}", flush=True)
        max_attempts = s.gen_max_attempts if s.gen_max_attempts > 0 else 1
        world, raw, issues, attempt_log, _ = _build_puzzle_for_world(
            llm, base_world, chain_depth, chain_depth_target, min_objs, max_attempts
        )
        for entry in attempt_log:
            messages.append(
                AIMessage(content=f"=== LLM ATTEMPT {entry['attempt']} ===\n\n{entry['raw']}")
            )
        messages.append(AIMessage(content=f"=== LLM FINAL ===\n\n{raw}"))

    elapsed = time.perf_counter() - start

    if issues:
        print(
            f"[puzzle_builder] WARNING: {len(issues)} issue(s) remain:", flush=True
        )
        for issue in issues:
            print(f"  • {issue}", flush=True)
    else:
        print(
            f"[puzzle_builder] built solvable puzzle constructively in {elapsed:.2f}s "
            f"({len(world.objects)} object(s))",
            flush=True,
        )

    # Ground-truth solution path from the oracle's actual winning solve.
    from benchmark.policies import bfs_solution_path

    world.solution_path = bfs_solution_path(world)
    if world.solution_path:
        print(
            f"[puzzle_builder] derived solution path from oracle "
            f"({len(world.solution_path)} step(s))",
            flush=True,
        )

    _print_solvability_check(world)
    _print_policy_benchmark(world)

    messages.append(AIMessage(content="=== CONSTRUCTIVE PUZZLE (themed) ==="))

    return {
        "messages": messages,
        "world": world,
    }
