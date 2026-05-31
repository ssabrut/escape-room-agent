"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Prerequisite, Room, WinCondition, WorldObject

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")

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
)


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


_VALID_PREREQ_TYPES = {"object_state", "known_info", "has_item", "power_active"}


def _build_prerequisite(raw) -> Prerequisite | None:
    if not isinstance(raw, dict):
        return None
    ptype = raw.get("type")
    if ptype not in _VALID_PREREQ_TYPES:
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
            rooms.append(Room(id=raw_room, description="", adjacency={}))
            continue
        if not isinstance(raw_room, dict):
            continue
        room_id = raw_room.get("id") or raw_room.get("name")
        if not room_id:
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
        rooms.append(
            Room(
                id=room_id,
                description=raw_room.get("description", ""),
                adjacency=adjacency,
                goal=raw_room.get("goal", ""),
                goal_completion=_build_prerequisite(raw_room.get("goal_completion")),
                next_step=raw_room.get("next_step", ""),
                key_objects=key_objects,
            )
        )
    return rooms


def _repair_adjacency(rooms: list[Room]) -> list[Room]:
    """Drop adjacency entries pointing at unknown rooms and mirror missing reverse edges."""
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
                neighbor_adj[reverse] = room_id
                cleaned[neighbor_id] = neighbor_adj

    return [
        Room(
            id=r.id,
            description=r.description,
            adjacency=cleaned.get(r.id, {}),
            goal=r.goal,
            goal_completion=r.goal_completion,
            next_step=r.next_step,
            key_objects=r.key_objects,
        )
        for r in rooms
    ]


def _coerce_id(value) -> str | None:
    """Normalize an id-like field to a string.

    The LLM sometimes emits an object reference as a dict (e.g.
    {"id": "crowbar"}) or a list instead of a plain id string. Pull out a
    sensible id; otherwise drop it so set membership and Pydantic don't choke.
    """
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        inner = value.get("id") or value.get("name")
        return inner if isinstance(inner, str) and inner else None
    return None


def _build_objects(raw_objects: list[dict], room_ids: set[str]) -> list[WorldObject]:
    """Build objects, validating that locations and tool references resolve."""
    candidates: list[dict] = [o for o in raw_objects if isinstance(o, dict) and o.get("id")]
    known_object_ids = {o["id"] for o in candidates if isinstance(o.get("id"), str)}
    valid_locations = room_ids | known_object_ids

    objects: list[WorldObject] = []
    for raw in candidates:
        obj_id = _coerce_id(raw.get("id"))
        if obj_id is None:
            continue  # malformed id

        location = _coerce_id(raw.get("location")) or ""
        if location not in valid_locations:
            continue  # drop objects placed nowhere real

        requires_tool = _coerce_id(raw.get("requires_tool"))
        if requires_tool and requires_tool not in known_object_ids:
            requires_tool = None  # null out dangling tool references

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
            # fuses is a dict; everything else must be a scalar (str/int). Drop
            # malformed nested structures the LLM occasionally emits.
            if field == "fuses":
                if isinstance(value, dict):
                    kwargs[field] = value
            elif isinstance(value, (str, int)):
                kwargs[field] = value

        try:
            objects.append(WorldObject(**kwargs))
        except Exception:
            continue  # skip objects that still fail validation rather than crash
    return objects


def _build_win_condition(raw: dict, object_ids: set[str]) -> WinCondition:
    if not isinstance(raw, dict):
        return WinCondition()
    object_id = raw.get("object_id", "")
    if object_id not in object_ids:
        object_id = next(iter(object_ids), "")
    return WinCondition(object_id=object_id, state=raw.get("state", ""))


def _scrub_room_refs(rooms: list[Room], object_ids: set[str]) -> list[Room]:
    """Drop key_object and goal_completion entries that reference unknown object ids."""
    def _valid(p: Prerequisite) -> bool:
        if p.type in {"object_state", "has_item"}:
            return p.object_id in object_ids
        return True

    for room in rooms:
        room.key_objects = [k for k in room.key_objects if k in object_ids]
        if room.goal_completion is not None and not _valid(room.goal_completion):
            room.goal_completion = None
    return rooms


def _secret_tokens(objects: list[WorldObject]) -> set[str]:
    """Codes the player must DISCOVER — these must never be spelled out in hints.

    Includes both the raw token (e.g. 'safe_code_1942') and the bare digits the
    engine actually matches on (e.g. '1942'), so either form gets redacted.
    """
    tokens: set[str] = set()
    for o in objects:
        code = o.requires_code
        if code:
            tokens.add(code)
            digits = re.sub(r"[^0-9]", "", code)
            if digits:
                tokens.add(digits)
    return tokens


def _scrub_spoilers(rooms: list[Room], solution_path: list[str], secrets: set[str]) -> tuple[list[str], list[str]]:
    """Redact literal codes from player-facing hints (room.next_step, solution_path).

    next_step is shown live to players, so a leaked code hands them the answer.
    Returns (redactions, cleaned_solution_path) for logging.
    """
    redactions: list[str] = []
    # longest-first so '1942' inside 'safe_code_1942' is handled by the token pass
    ordered = sorted(secrets, key=len, reverse=True)

    def _redact(text: str, where: str) -> str:
        out = text
        for tok in ordered:
            if tok and tok in out:
                out = out.replace(tok, "the hidden code")
                redactions.append(f"{where}: '{tok}'")
        return out

    for room in rooms:
        if room.next_step:
            room.next_step = _redact(room.next_step, f"next_step[{room.id}]")

    cleaned_path = [_redact(step, "solution_path") for step in solution_path]
    return redactions, cleaned_path


def _goal_completion_subject(p: Prerequisite) -> str | None:
    """The id/info token a goal_completion hinges on — used to bind goal prose to it."""
    if p.type in {"object_state", "has_item"}:
        return p.object_id
    if p.type == "known_info":
        return p.info
    if p.type == "power_active":
        return p.id
    return None


def _goal_from_completion(p: Prerequisite) -> str:
    """A deterministic, spoiler-free goal sentence derived from the condition."""
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
    """Ensure each room's `goal` prose actually refers to its goal_completion subject.

    When the LLM's narrative goal diverges from the machine condition (e.g. goal
    says 'open the safe' but completion only needs a known_info), replace the goal
    with a deterministic sentence derived from the condition. Returns rewrites for
    logging. known_info goals are always normalized (the subject is a hidden token
    that must not appear in the prose).
    """
    rewrites: list[str] = []
    for room in rooms:
        gc = room.goal_completion
        if gc is None:
            continue
        subject = _goal_completion_subject(gc)
        goal = room.goal or ""
        # has_item/object_state/power: prose should name the subject id.
        if gc.type in {"object_state", "has_item", "power_active"}:
            if subject and subject.lower() not in goal.lower():
                room.goal = _goal_from_completion(gc)
                rewrites.append(f"{room.id}: goal rebound to completion ({gc.type})")
        # known_info: the subject is a secret token — force a non-leaking goal.
        elif gc.type == "known_info":
            room.goal = _goal_from_completion(gc)
            rewrites.append(f"{room.id}: known_info goal normalized")
    return rewrites


MAX_ROOMS = 2

_CLUE_HINTS = ("note", "letter", "paper", "scroll", "document", "diary", "journal", "tome", "book", "tablet")


def _required_info_tokens(rooms: list[Room], objects: list[WorldObject]) -> list[tuple[str, str]]:
    """Collect (info_token, source_room) pairs that the world must produce.

    Source room is where the info needs to be available (i.e., a room the party
    can reach before needing the token). For now we use the first room as the
    source since maps are at most 2 rooms.
    """
    out: list[tuple[str, str]] = []
    start_room = rooms[0].id if rooms else ""
    seen: set[str] = set()

    def _add(token: str) -> None:
        if token and token not in seen:
            seen.add(token)
            out.append((token, start_room))

    for room in rooms:
        gc = room.goal_completion
        if gc is not None and gc.type == "known_info" and gc.info:
            _add(gc.info)

    for obj in objects:
        if obj.requires_code:
            _add(obj.requires_code)
    return out


def _patch_missing_info(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    """Attach missing `contains_info` to a plausible carrier object so the world is solvable."""
    patched: list[str] = []
    produced = {o.contains_info for o in objects if o.contains_info}

    def _score(obj: WorldObject, source_room: str) -> int:
        if obj.location != source_room:
            return -1
        if obj.contains_info:
            return -1  # already carries info; don't overwrite
        if obj.state in HIDDEN_STATES_ROOM:
            return -1
        haystack = f"{obj.id} {obj.description}".lower()
        score = 0
        if any(h in haystack for h in _CLUE_HINTS):
            score += 10
        if obj.takeable:
            score += 3
        if obj.interactable:
            score += 2
        return score

    for token, source_room in _required_info_tokens(rooms, objects):
        if any(token in (info or "") or (info or "") in token for info in produced):
            continue
        if any(_token_matches_code(token, info) for info in produced):
            continue
        candidates = sorted(
            objects, key=lambda o: _score(o, source_room), reverse=True
        )
        winner = next((o for o in candidates if _score(o, source_room) >= 0), None)
        if winner is None:
            continue
        winner.contains_info = token
        winner.interactable = True
        produced.add(token)
        patched.append(f"{token} -> {winner.id}")
    return patched


HIDDEN_STATES_ROOM = {"locked", "locked_bolt", "locked_room", "hidden"}


def _token_matches_code(code: str, info: str | None) -> bool:
    if not info:
        return False
    import re as _re
    return bool(_re.sub(r"[^0-9]", "", info) == _re.sub(r"[^0-9]", "", code) and code)


def _build_world(data: dict) -> GameWorld:
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    if MAX_ROOMS > 0 and len(rooms) > MAX_ROOMS:
        rooms = rooms[:MAX_ROOMS]
        kept_ids = {r.id for r in rooms}
        for r in rooms:
            r.adjacency = {d: n for d, n in r.adjacency.items() if n in kept_ids}
        rooms = _repair_adjacency(rooms)
    room_ids = {r.id for r in rooms}
    objects = _build_objects(data.get("objects", []), room_ids)
    object_ids = {o.id for o in objects}
    rooms = _scrub_room_refs(rooms, object_ids)

    patched = _patch_missing_info(rooms, objects)
    if patched:
        print(f"[game_master] auto-patched missing clues: {', '.join(patched)}", flush=True)

    rewrites = _bind_goals(rooms)
    if rewrites:
        print(f"[game_master] rebound goals to completion: {', '.join(rewrites)}", flush=True)

    rules = [r for r in data.get("rules", []) if isinstance(r, str)]
    solution_path = [s for s in data.get("solution_path", []) if isinstance(s, str)]

    secrets = _secret_tokens(objects)
    redactions, solution_path = _scrub_spoilers(rooms, solution_path, secrets)
    if redactions:
        print(f"[game_master] scrubbed spoilers: {', '.join(redactions)}", flush=True)

    return GameWorld(
        scenario=data.get("scenario", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
        objects=objects,
        rules=rules,
        win_condition=_build_win_condition(data.get("win_condition", {}), object_ids),
        solution_path=solution_path,
    )


def game_master_node(state: GameState) -> dict:
    llm = get_llm("game_master")
    prompt = GENERATION_PROMPT.format(theme=state.theme)

    storyline_start = time.perf_counter()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    storyline_elapsed = time.perf_counter() - storyline_start

    map_start = time.perf_counter()
    data = _parse_json(response.content) or {}
    world = _build_world(data)
    map_elapsed = time.perf_counter() - map_start

    print(
        f"[game_master] storyline (LLM): {storyline_elapsed:.2f}s | "
        f"map (parse+build): {map_elapsed:.2f}s",
        flush=True,
    )

    return {
        "messages": [AIMessage(content=response.content)],
        "world": world,
    }
