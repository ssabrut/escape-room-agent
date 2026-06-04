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
    "wheel", "key", "lever", "crank", "handle", "valve", "tool", "crowbar", "wrench",
)
_CLUE_HINTS = (
    "note", "letter", "paper", "scroll", "document", "diary",
    "journal", "tome", "book", "tablet",
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
        lines.append(f'  adjacency: {json.dumps(r.adjacency)}')
        lines.append("")
    return "\n".join(lines)


def _format_room_goals(rooms: list[Room]) -> str:
    lines = []
    for r in rooms:
        gc = r.goal_completion.model_dump(exclude_none=True) if r.goal_completion else "(none)"
        lines.append(f'  Room "{r.id}": goal="{r.goal}" | goal_completion={gc}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Object building (moved from game_master.py)
# ---------------------------------------------------------------------------

def _coerce_id(value) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        inner = value.get("id") or value.get("name")
        return inner if isinstance(inner, str) and inner else None
    return None


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
        room.key_objects = [k for k in room.key_objects if k in object_ids]
        if room.goal_completion is not None and not _valid(room.goal_completion):
            room.goal_completion = None
    return rooms


def _stem(token: str) -> str:
    s = token.lower()
    for suffix in (
        "_revealed", "_hidden", "_locked", "_unlocked",
        "_item", "_object", "_tool", "_key",
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


def _repair_unsolvable_gates(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
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
        if any([obj.requires_code, obj.requires_tool, obj.requires_liquid,
                obj.requires_power, obj.fuses]):
            continue
        same_room_takeables = [
            o for o in objects
            if o.location == obj.location and o.takeable
            and o.id != obj.id and o.state not in _LLM_LOCKED_STATES
        ]
        hinted = [o for o in same_room_takeables
                  if any(h in o.id.lower() for h in _UNLOCKER_HINTS)]
        pick = hinted[0] if hinted else (same_room_takeables[0] if same_room_takeables else None)
        if pick is not None:
            obj.requires_tool = pick.id
            repairs.append(f"{obj.id}: locked gate given requires_tool {pick.id}")
        else:
            obj.state = target_state
            repairs.append(f"{obj.id}: locked gate with no mechanism started at target '{target_state}'")
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
            (o for o in objects
             if o.location == room.id and not o.takeable
             and o.state not in _LLM_LOCKED_STATES and o.fuses is None),
            None,
        )
        if host is None:
            host = next((o for o in objects if o.location == room.id and o.fuses is None), None)
        if host is None:
            continue
        host.fuses = {label: "OFF"}
        host.interactable = True
        if host.state in _LLM_LOCKED_STATES:
            host.state = "visible"
        gc.id = token
        repairs.append(f"{room.id}: power gate '{orig_id}' -> fuse {label} on {host.id} ({token})")
    return repairs


def _repair_missing_win_condition(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    if not rooms:
        return []
    final_room = rooms[-1]
    gc = final_room.goal_completion
    if gc is not None and gc.type == "object_state" and gc.object_id:
        return []

    final_room_id = final_room.id
    by_id = {o.id: o for o in objects}
    candidates = [o for o in objects if o.location == final_room_id and o.state in _LLM_LOCKED_STATES]
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


def _required_info_tokens(rooms: list[Room], objects: list[WorldObject]) -> list[tuple[str, str]]:
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
        repairs.append(f"{lock.id}: requires_code {old} -> self clue {lock.contains_info}")
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
            o.location, o.state, o.requires_code, o.requires_tool,
            o.requires_liquid, o.requires_power,
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
            _token_matches_code(c, o.contains_info) or o.contains_info in c or c in o.contains_info
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
    rooms: list[Room], objects: list[WorldObject]
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
        if o.id in keep or o.scenic:
            if o.location in room_object_count:
                room_object_count[o.location] += 1

    for rid in room_ids_list:
        deficit = MIN_OBJECTS_PER_ROOM - room_object_count[rid]
        if deficit <= 0:
            continue
        fillers = [
            o for o in objects
            if o.id not in keep and not o.scenic and o.location == rid
            and not o.requires_tool and not o.requires_code
            and not o.requires_liquid and not o.requires_power
            and not o.fuses and not o.contains_info
        ]
        for filler in fillers[:deficit]:
            keep.add(filler.id)
            room_object_count[rid] += 1

    dropped = [o.id for o in objects if o.id not in keep and not o.scenic]
    kept = [o for o in objects if o.id in keep or o.scenic]
    return kept, dropped


def _secret_tokens(objects: list[WorldObject]) -> set[str]:
    tokens: set[str] = set()
    for o in objects:
        code = o.requires_code
        if code:
            tokens.add(code)
            digits = re.sub(r"[^0-9]", "", code)
            if digits:
                tokens.add(digits)
    return tokens


def _scrub_spoilers(
    solution_path: list[str], secrets: set[str]
) -> tuple[list[str], list[str]]:
    redactions: list[str] = []
    ordered = sorted(secrets, key=len, reverse=True)
    secret_digits: set[str] = {
        d for tok in secrets if (d := re.sub(r"[^0-9]", "", tok)) and len(d) >= 3
    }

    def _redact(text: str, where: str) -> str:
        out = text
        for tok in ordered:
            if tok and tok in out:
                out = out.replace(tok, "the hidden code")
                redactions.append(f"{where}: '{tok}'")

        def _digit_sub(m: re.Match) -> str:
            digits = m.group(0)
            if digits in secret_digits:
                redactions.append(f"{where}: bare digits '{digits}'")
                return "the hidden code"
            return digits
        out = re.sub(r"\b\d{3,}\b", _digit_sub, out)
        return out

    cleaned_path = [_redact(step, "solution_path") for step in solution_path]
    return cleaned_path, redactions


def _scrub_ghost_ids(
    solution_path: list[str], valid_ids: set[str]
) -> tuple[list[str], list[str]]:
    replacements: list[str] = []
    _SAFE = {"the_hidden", "hidden_code", "solution_path", "the_object"}

    def _replace_ghosts(text: str) -> str:
        def _sub(m: re.Match) -> str:
            tok = m.group(0)
            if tok in valid_ids or tok in _SAFE:
                return tok
            if len(tok) <= 4 or "_" not in tok:
                return tok
            replacements.append(tok)
            return "the object"
        return re.sub(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", _sub, text)

    cleaned = [_replace_ghosts(step) for step in solution_path]
    return cleaned, replacements


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def _build_puzzle(world: GameWorld, data: dict) -> GameWorld:
    """Apply the repair pipeline to LLM-generated object data and return a completed GameWorld."""
    rooms = world.rooms
    room_ids = {r.id for r in rooms}

    objects = _build_objects(data.get("objects", []), room_ids)

    objects, merges = _dedup_objects(objects, rooms)
    if merges:
        print(f"[puzzle_builder] merged duplicate objects: {', '.join(merges)}", flush=True)

    object_ids = {o.id for o in objects}
    rooms = _scrub_room_refs(rooms, object_ids)

    tool_repairs = _repair_tool_refs(objects)
    if tool_repairs:
        print(f"[puzzle_builder] repaired tool refs: {', '.join(tool_repairs)}", flush=True)

    takeable_repairs = _make_required_tools_takeable(objects)
    if takeable_repairs:
        print(f"[puzzle_builder] made required tools takeable: {', '.join(takeable_repairs)}", flush=True)

    win_repairs = _repair_missing_win_condition(rooms, objects)
    if win_repairs:
        print(f"[puzzle_builder] repaired missing win condition: {', '.join(win_repairs)}", flush=True)

    gate_repairs = _repair_unsolvable_gates(rooms, objects)
    if gate_repairs:
        print(f"[puzzle_builder] repaired unsolvable gates: {', '.join(gate_repairs)}", flush=True)

    power_repairs = _repair_power_gates(rooms, objects)
    if power_repairs:
        print(f"[puzzle_builder] repaired power gates: {', '.join(power_repairs)}", flush=True)

    patched = _patch_missing_info(rooms, objects)
    if patched:
        print(f"[puzzle_builder] auto-patched missing clues: {', '.join(patched)}", flush=True)

    self_clue = _rebind_self_clue_locks(objects)
    if self_clue:
        print(f"[puzzle_builder] rebound self-clue locks: {', '.join(self_clue)}", flush=True)

    rewrites = _bind_goals(rooms)
    if rewrites:
        print(f"[puzzle_builder] rebound goals to completion: {', '.join(rewrites)}", flush=True)

    rules = [r for r in data.get("rules", []) if isinstance(r, str)]
    solution_path = [s for s in data.get("solution_path", []) if isinstance(s, str)]

    objects, orphans = _prune_orphan_objects(rooms, objects)
    if orphans:
        print(f"[puzzle_builder] pruned orphan objects: {', '.join(orphans)}", flush=True)
        rooms = _scrub_room_refs(rooms, {o.id for o in objects})

    valid_ids = {o.id for o in objects} | {r.id for r in rooms}
    solution_path, ghost_replacements = _scrub_ghost_ids(solution_path, valid_ids)
    if ghost_replacements:
        print(
            f"[puzzle_builder] scrubbed ghost ids from solution_path: {', '.join(set(ghost_replacements))}",
            flush=True,
        )

    secrets = _secret_tokens(objects)
    solution_path, redactions = _scrub_spoilers(solution_path, secrets)
    if redactions:
        print(f"[puzzle_builder] scrubbed spoilers: {', '.join(redactions)}", flush=True)

    coherence = _check_coherence(rooms, objects)
    if coherence:
        print(f"[puzzle_builder] coherence warnings: {'; '.join(coherence)}", flush=True)

    return GameWorld(
        scenario=world.scenario,
        objective=world.objective,
        rooms=rooms,
        objects=objects,
        rules=rules,
        solution_path=solution_path,
    )


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _default_chain_depth(s: Settings) -> int:
    return s.chain_depth if s.hard_mode else 3


def _build_prompt(world: GameWorld, chain_depth: int) -> str:
    return GENERATION_PROMPT.format(
        scenario=world.scenario,
        objective=world.objective,
        rooms=_format_rooms(world.rooms),
        room_goals=_format_room_goals(world.rooms),
        chain_depth=chain_depth,
    )


def _generate_puzzle(llm, world: GameWorld, chain_depth: int) -> tuple[GameWorld, str]:
    prompt = _build_prompt(world, chain_depth)
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}
    return _build_puzzle(world, data), response.content


def _generate_puzzle_with_feedback(
    llm, world: GameWorld, chain_depth: int, violations: list[str]
) -> tuple[GameWorld, str]:
    violation_block = "\n".join(f"  - {v}" for v in violations)
    correction = (
        "Your previous puzzle had the following issues detected by an automated judge. "
        "Please generate a NEW, corrected puzzle for the same rooms that fixes ALL of them. "
        "Return only the JSON object — no prose.\n\n"
        f"Issues to fix:\n{violation_block}"
    )
    prompt = _build_prompt(world, chain_depth)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
            HumanMessage(content=correction),
        ]
    )
    data = _parse_json(response.content) or {}
    return _build_puzzle(world, data), response.content


# ---------------------------------------------------------------------------
# Solvability checks (same as game_master — duplicated to avoid circular import)
# ---------------------------------------------------------------------------

def _world_is_solvable(world: GameWorld) -> bool:
    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import heuristic_policy
    except Exception:
        return True
    if not world.win_condition.object_id or not world.rooms:
        return False
    return HeadlessEpisode(world).run(heuristic_policy).victory


def _world_meets_chain_depth(world: GameWorld, target: int) -> bool:
    if target <= 0:
        return True
    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import heuristic_policy
    except Exception:
        return True
    if not world.win_condition.object_id or not world.rooms:
        return False
    return HeadlessEpisode(world).run(heuristic_policy).chain_depth >= target


def _print_solvability_check(world: GameWorld) -> None:
    try:
        from benchmark.policies import check_solvable
    except Exception:
        return
    if not world.rooms:
        return
    report = check_solvable(world)
    if report.solvable:
        print("[puzzle_builder] static check: SOLVABLE — no structural issues", flush=True)
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

def puzzle_builder_node(state: GameState) -> dict:
    """Generate objects and solution path for the rooms built by world_builder.

    Retries independently when the puzzle is unsolvable or too shallow —
    the room layout is preserved across retries.
    """
    if not state.world or not state.world.rooms:
        return {}

    s = Settings()
    llm = get_llm("game_master")
    chain_depth = _default_chain_depth(s)

    world, raw = _generate_puzzle(llm, state.world, chain_depth)

    def _world_ok(w: GameWorld) -> tuple[bool, str]:
        if not _world_is_solvable(w):
            return False, "unsolvable"
        if not _world_meets_chain_depth(w, chain_depth if s.hard_mode else 0):
            return False, f"chain depth < {chain_depth}"
        return True, ""

    attempts = 1
    if s.gen_max_attempts > 0:
        ok, why = _world_ok(world)
        while not ok and attempts < s.gen_max_attempts:
            try:
                from benchmark.narrative_eval import quick_eval_for_feedback
                qr = quick_eval_for_feedback(world)
                violations = qr.violations
            except Exception:
                violations = []

            if violations:
                print(
                    f"[puzzle_builder] puzzle rejected ({why}), "
                    f"{len(violations)} violation(s) — regenerating with feedback "
                    f"(attempt {attempts + 1}/{s.gen_max_attempts})",
                    flush=True,
                )
                for v in violations:
                    print(f"  {v}", flush=True)
                world, raw = _generate_puzzle_with_feedback(llm, state.world, chain_depth, violations)
            else:
                print(
                    f"[puzzle_builder] puzzle rejected ({why}) — regenerating "
                    f"(attempt {attempts + 1}/{s.gen_max_attempts})",
                    flush=True,
                )
                world, raw = _generate_puzzle(llm, state.world, chain_depth)

            attempts += 1
            ok, why = _world_ok(world)

    print(f"[puzzle_builder] built puzzle in {attempts} attempt(s)", flush=True)

    _print_solvability_check(world)
    _print_policy_benchmark(world)

    return {
        "messages": [AIMessage(content=raw)],
        "world": world,
    }
