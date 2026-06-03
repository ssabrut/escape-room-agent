"""Game Master agent — sole node; generates all narrative dynamically."""

from __future__ import annotations

import json
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import Settings, get_llm
from prompts import load_prompt
from state import GameState, GameWorld, Prerequisite, Room, WorldObject

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")
# Hard-mode prompt: N-room worlds with deep chains + decoys (see settings.hard_mode).
BANK_GENERATION_PROMPT = load_prompt("game_master", "generation_bank")

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
    candidates: list[dict] = [
        o for o in raw_objects if isinstance(o, dict) and o.get("id")
    ]
    # Object ids that collide with a room id are rejected, so don't treat them as
    # valid locations or tool targets either.
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
            continue  # malformed id
        if obj_id in room_ids:
            continue  # an object's id must not collide with a room id

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


def _scrub_spoilers(
    solution_path: list[str], secrets: set[str]
) -> tuple[list[str], list[str]]:
    """Redact literal codes from the player-facing solution_path.

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


def _required_info_tokens(
    rooms: list[Room], objects: list[WorldObject]
) -> list[tuple[str, str]]:
    """Collect (info_token, source_room) pairs that the world must produce.

    Source room is where the info's clue should live: the room of the object that
    actually consumes the token, so the clue is discoverable in the same room as
    (or before) the lock it opens. A known_info goal sources to its own room. A
    requires_code sources to the room holding that locked object — NOT blindly the
    start room, which would strand a second-room safe's clue back in room one.
    """
    out: list[tuple[str, str]] = []
    start_room = rooms[0].id if rooms else ""
    by_id = {o.id: o for o in objects}
    room_ids = {r.id for r in rooms}
    seen: set[str] = set()

    def _room_of(obj: WorldObject) -> str:
        # Walk the container chain up to the room the object ultimately sits in.
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
    """Attach missing `contains_info` to a plausible carrier object so the world is solvable."""
    patched: list[str] = []
    produced = {o.contains_info for o in objects if o.contains_info}
    # Objects used as a tool by something else read as "use me", not "examine me",
    # so they're poor places to bury a code clue the party must READ.
    tool_ids = {o.requires_tool for o in objects if o.requires_tool}

    def _score(obj: WorldObject, source_room: str) -> int:
        if obj.location != source_room:
            return -1
        if obj.contains_info:
            return -1  # already carries info; don't overwrite
        if obj.state in HIDDEN_STATES_ROOM:
            return -1
        haystack = f"{obj.id} {obj.description}".lower()
        score = 0
        readable = any(h in haystack for h in _CLUE_HINTS)
        if readable:
            score += 10
        # A takeable object only makes a good clue carrier if it also reads like
        # something you examine (a note/journal). A bare takeable tool does not.
        if obj.takeable and readable:
            score += 3
        if obj.interactable:
            score += 2
        if obj.id in tool_ids:
            score -= 5  # it's somebody's tool; prefer a readable object instead
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
    """Fix locks whose required code is supplied by NOTHING but themselves.

    The LLM sometimes writes the answer onto the lock's own ``contains_info`` while
    its ``requires_code`` names a different token — e.g. a door with
    ``requires_code: 'door_seq'`` but ``contains_info: 'final_seq_5555'``. Examining
    the door teaches '5555', but the engine only accepts 'door_seq', so the lock can
    never open and the room (or whole game) is unwinnable.

    When a ``requires_code`` token is produced by no object's ``contains_info`` (by
    digit-equality or substring), but the locked object itself carries a
    ``contains_info``, we repoint ``requires_code`` to that self-carried token so the
    engine's equality test succeeds. This yields a one-object puzzle (examine the
    lock to learn its code, then enter it) — degenerate but solvable, which is the
    point: a benchmark target must be winnable. Returns a repair log.
    """
    def _supplied_by_other(lock: WorldObject) -> bool:
        code = lock.requires_code
        for o in objects:
            if o is lock or not o.contains_info:
                continue
            info = o.contains_info
            if (
                _token_matches_code(code, info)
                or code in info
                or info in code
            ):
                return True
        return False

    repairs: list[str] = []
    for lock in objects:
        if not lock.requires_code:
            continue
        if _supplied_by_other(lock):
            continue  # some upstream clue already opens it
        if not lock.contains_info:
            continue  # nothing to rebind to; left for _check_coherence to warn
        old = lock.requires_code
        lock.requires_code = lock.contains_info
        lock.interactable = True
        repairs.append(f"{lock.id}: requires_code {old} -> self clue {lock.contains_info}")
    return repairs


def _dedup_objects(
    objects: list[WorldObject], rooms: list[Room]
) -> tuple[list[WorldObject], list[str]]:
    """Collapse near-duplicate objects in the same room into one.

    The LLM sometimes emits a container AND a redundant "lock" object for it —
    e.g. ``golden_chest`` plus ``gold_chest_lock`` in the same room, both locked,
    both requiring the same tool, with nothing distinguishing them functionally.
    Two objects are treated as duplicates when they share a location, state, and
    identical precondition fields (same requires_code/tool/liquid/power/fuses).

    The survivor is the one referenced by a room goal_completion / win target /
    key_objects (the "real" object the game checks); the other is dropped, and any
    requires_tool references to the dropped id are repointed at the survivor.
    """
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

    # group by precondition signature
    groups: dict[tuple, list[WorldObject]] = {}
    for o in objects:
        groups.setdefault(_precond_signature(o), []).append(o)

    drop: dict[str, str] = {}  # dropped_id -> survivor_id
    for sig, group in groups.items():
        # only collapse when the signature carries a real precondition (locked +
        # a requirement); identical plain scenery is left alone.
        has_precond = any(sig[2:6]) or sig[6]
        if len(group) < 2 or not has_precond:
            continue
        survivor = next((o for o in group if o.id in referenced), group[0])
        for o in group:
            if o.id is survivor.id or o.id == survivor.id:
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


def _stem(token: str) -> str:
    """Reduce an id to its matchable core: lowercased, qualifier suffixes dropped.

    The generator often refers to one object by a bare name in one place and a
    decorated name in another — e.g. a door wants ``brg_key`` while the takeable
    object is ``brg_key_revealed``. Stripping common state/qualifier suffixes lets
    those resolve to the same stem so a dangling tool ref can be repaired instead
    of nulled (which would make the lock open with no tool at all).
    """
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
    """Repoint dangling requires_tool refs at a fuzzy-matching takeable object.

    Runs after dedup/scrub, once final object ids are settled. A door whose
    requires_tool names no real object is unwinnable; rather than null it (a free
    open), we look for a takeable object whose id stem matches and repoint to it.
    Refs with no plausible match are nulled as before. Returns repair log lines.
    """
    by_id = {o.id: o for o in objects}
    takeable_by_stem: dict[str, list[str]] = {}
    for o in objects:
        if o.takeable:
            takeable_by_stem.setdefault(_stem(o.id), []).append(o.id)

    repairs: list[str] = []
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id or tool_id in by_id:
            continue  # no ref, or already resolves to a real object
        candidates = takeable_by_stem.get(_stem(tool_id), [])
        # prefer a candidate sharing the stem; if exactly one, it's unambiguous.
        match = candidates[0] if len(candidates) == 1 else None
        if match is None:
            # fall back to a takeable id that contains (or is contained by) the stem
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
    """Force every `requires_tool` target to be pickup-able, else the lock is dead.

    A door whose requires_tool names a real object that is NOT takeable (e.g. a
    fixed terminal) can never be used — the party cannot carry it to satisfy
    `use_tool`, so the lock (and any room it gates) is unwinnable. The generator
    only WARNS about this; here we repair it: the referenced tool is made takeable
    and, if it started locked/hidden behind its own gate, relaxed to a reachable
    state so it can actually be grabbed.

    Guard: never make the GATE object itself its own tool, and never relocate a
    tool out of a container — we only flip takeable/state on the existing tool.
    Runs after _repair_tool_refs, once requires_tool ids are final.
    """
    by_id = {o.id: o for o in objects}
    repairs: list[str] = []
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id:
            continue
        tool = by_id.get(tool_id)
        if tool is None or tool.id == o.id:
            continue  # dangling refs are handled upstream; skip self-reference
        changed: list[str] = []
        if not tool.takeable:
            tool.takeable = True
            changed.append("takeable")
        # A tool locked/hidden behind the very lock it opens can't be reached; if
        # it is nested inside the gate object, or starts in a locked/hidden state,
        # relax it to visible so it is grabbable before the lock is cleared.
        if tool.location == o.id or tool.state in _LLM_LOCKED_STATES:
            tool.state = "visible"
            changed.append("visible")
        if changed:
            tool.interactable = True
            repairs.append(f"{tool.id}: made {'+'.join(changed)} (tool for {o.id})")
    return repairs


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


def _has_unlock_mechanism(o: WorldObject) -> bool:
    return bool(
        o.requires_code
        or o.requires_tool
        or o.requires_liquid
        or o.requires_power
        or o.fuses
    )


def _repair_unsolvable_gates(
    rooms: list[Room], objects: list[WorldObject]
) -> list[str]:
    """Make goal-gating objects winnable when the LLM locked them with no mechanism.

    A room's goal_completion (and the final room's, which is the win condition)
    often targets ``object_state -> unlocked/open`` on a locked object. If that
    object starts in a locked/hidden state but carries NO requires_* mechanism,
    nothing can ever change its state — the room (or whole game) is unwinnable and
    the party spins on a GM-blocked exit. We repair it by pointing requires_tool at
    a same-room takeable that reads like an unlocker; failing that, we relax its
    initial state to ``visible`` so the goal is at least reachable. Returns a log.
    """
    # Collect ids that a goal_completion gates to a non-open state via object_state.
    gated: dict[str, str] = {}  # object_id -> required target state
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
            continue  # already openable / not locked
        if _has_unlock_mechanism(obj):
            continue  # has a real path to the target state
        # Prefer a same-room takeable that reads like the intended unlocker; else
        # any same-room takeable (use_tool drives the gate to 'unlocked', matching
        # the typical goal target). With no takeable at all, fall back to relaxing
        # the gate's initial state to its required target so the goal is reachable.
        same_room_takeables = [
            o
            for o in objects
            if o.location == obj.location
            and o.takeable
            and o.id != obj.id
            and o.state not in _LLM_LOCKED_STATES  # the unlocker must be grabbable
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
    """A single-letter-ish fuse label derived from a power token.

    The engine only ever produces power tokens of the form ``sekring_<label>_ON``
    (see gameplay_node._resolve_flip_fuse). A ``power_active`` goal whose id is any
    other string can never be satisfied. We pick a stable label from the token so
    the rewritten goal id and the panel's fuse agree.
    """
    stem = re.sub(r"[^a-z0-9]", "", token.lower()) or "x"
    return stem[:4].upper()


def _repair_power_gates(rooms: list[Room], objects: list[WorldObject]) -> list[str]:
    """Make ``power_active`` goal gates satisfiable.

    The engine can only turn power ON via a fuse flip, which yields the token
    ``sekring_<label>_ON``. A goal_completion of type ``power_active`` whose id is
    not such a token (or whose token no object's fuses produce) is unwinnable: the
    party reaches the gated exit and spins forever (exactly the
    ``emergency_relay_power`` case the benchmark surfaced). We repair by giving a
    same-room object a fuse panel and rewriting the goal id to the
    ``sekring_<label>_ON`` token that panel produces. Returns a repair log.
    """
    repairs: list[str] = []
    by_id = {o.id: o for o in objects}
    for room in rooms:
        gc = room.goal_completion
        if gc is None or gc.type != "power_active" or not gc.id:
            continue
        # Already producible? Some object's fuses must yield exactly gc.id.
        producible = any(
            o.fuses and any(f"sekring_{lbl}_ON" == gc.id for lbl in o.fuses)
            for o in objects
            if o.location == room.id or by_id.get(o.location) and by_id[o.location].location == room.id
        )
        if producible:
            continue
        orig_id = gc.id
        label = _fuse_label_for(gc.id)
        token = f"sekring_{label}_ON"
        # Prefer an existing reachable, non-takeable object in the room to host the
        # panel (a panel is fixed scenery); else the gate object itself; else skip.
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
            host = next((o for o in objects if o.location == room.id and o.fuses is None), None)
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


def _consumed_codes(rooms: list[Room], objects: list[WorldObject]) -> set[str]:
    """Tokens that something actually USES: requires_code values and known_info goals."""
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
    """Flag dangling puzzle pieces without mutating the world.

    Conservative, warn-only complement to ``_patch_missing_info`` (which fixes the
    opposite, unambiguous direction). We cannot safely guess which lock a stray
    clue belongs to, so we surface it for inspection instead of rewiring blindly:
      - a `contains_info` clue that no requires_code / known_info goal consumes;
      - a tool that exists but is never reachable (hidden with no reveal path);
      - duplicated clues (same token carried by multiple objects).
    """
    warnings: list[str] = []
    consumed = _consumed_codes(rooms, objects)

    # clues nobody consumes
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
                f"clue '{o.contains_info}' on {o.id} is consumed by nothing "
                f"(no requires_code / known_info goal uses it)"
            )

    # duplicated clue tokens
    for token, carriers in seen_info.items():
        if len(carriers) > 1:
            warnings.append(f"clue '{token}' duplicated across {', '.join(carriers)}")

    # tools required but unreachable
    obj_by_id = {o.id: o for o in objects}
    for o in objects:
        tool_id = o.requires_tool
        if not tool_id:
            continue
        tool = obj_by_id.get(tool_id)
        if tool is None:
            continue  # already nulled by _build_objects if dangling
        if not tool.takeable:
            warnings.append(f"{o.id} requires tool {tool_id}, but it is not takeable")
        if tool.state in HIDDEN_STATES_ROOM and not any(
            info == tool_id or (info and tool_id in info)
            for info in (x.contains_info for x in objects)
        ):
            warnings.append(
                f"tool {tool_id} is '{tool.state}' with no clue that reveals it "
                f"(required by {o.id})"
            )
    return warnings


HIDDEN_STATES_ROOM = {"locked", "locked_bolt", "locked_room", "hidden"}


def _token_matches_code(code: str, info: str | None) -> bool:
    if not info:
        return False
    import re as _re

    return bool(_re.sub(r"[^0-9]", "", info) == _re.sub(r"[^0-9]", "", code) and code)


def _select_rooms(rooms: list[Room], data: dict, limit: int) -> list[Room]:
    """Truncate to `limit` rooms, keeping the start room and the win-object room.

    The LLM is asked for two rooms, but if it over-produces, blindly keeping the
    first `limit` rooms can drop the room containing the win_condition object
    (making the world unsolvable). Anchor on both the starting room (rooms[0])
    and the win object's room so the playable start->win chain survives.
    """
    win = data.get("win_condition")
    win_object_id = win.get("object_id") if isinstance(win, dict) else None
    win_room: str | None = None
    if win_object_id:
        for obj in data.get("objects", []):
            if isinstance(obj, dict) and obj.get("id") == win_object_id:
                loc = obj.get("location")
                win_room = loc if isinstance(loc, str) else None
                break

    ordered = list(rooms)
    start_id = ordered[0].id if ordered else None
    anchors = [rid for rid in (start_id, win_room) if rid]
    if anchors:
        anchored = [r for r in ordered if r.id in anchors]
        rest = [r for r in ordered if r.id not in anchors]
        ordered = anchored + rest
    return ordered[:limit]


def _build_world(data: dict) -> GameWorld:
    rooms = _repair_adjacency(_build_rooms(data.get("rooms", [])))
    if MAX_ROOMS > 0 and len(rooms) > MAX_ROOMS:
        rooms = _select_rooms(rooms, data, MAX_ROOMS)
        kept_ids = {r.id for r in rooms}
        for r in rooms:
            r.adjacency = {d: n for d, n in r.adjacency.items() if n in kept_ids}
        rooms = _repair_adjacency(rooms)
    room_ids = {r.id for r in rooms}
    objects = _build_objects(data.get("objects", []), room_ids)

    # collapse redundant container/lock pairs before scrubbing, while room refs
    # still name the survivor we want to keep.
    objects, merges = _dedup_objects(objects, rooms)
    if merges:
        print(
            f"[game_master] merged duplicate objects: {', '.join(merges)}", flush=True
        )

    object_ids = {o.id for o in objects}
    rooms = _scrub_room_refs(rooms, object_ids)

    # Repair (or null) requires_tool refs that dedup/scrub left dangling, so a
    # near-miss name like brg_key vs brg_key_revealed doesn't strand a door.
    tool_repairs = _repair_tool_refs(objects)
    if tool_repairs:
        print(
            f"[game_master] repaired tool refs: {', '.join(tool_repairs)}", flush=True
        )

    # A requires_tool that names a real-but-non-takeable object (or one locked
    # behind its own gate) is unwinnable; force such tools grabbable.
    takeable_repairs = _make_required_tools_takeable(objects)
    if takeable_repairs:
        print(
            f"[game_master] made required tools takeable: {', '.join(takeable_repairs)}",
            flush=True,
        )

    # Make goal-gating objects that the LLM locked with no unlock path winnable,
    # so the party can't get stranded on a GM-blocked exit it can never clear.
    gate_repairs = _repair_unsolvable_gates(rooms, objects)
    if gate_repairs:
        print(
            f"[game_master] repaired unsolvable gates: {', '.join(gate_repairs)}",
            flush=True,
        )

    # power_active gates need a fuse panel to be satisfiable (the engine only
    # produces sekring_<label>_ON tokens); repair any that lack one.
    power_repairs = _repair_power_gates(rooms, objects)
    if power_repairs:
        print(
            f"[game_master] repaired power gates: {', '.join(power_repairs)}",
            flush=True,
        )

    patched = _patch_missing_info(rooms, objects)
    if patched:
        print(
            f"[game_master] auto-patched missing clues: {', '.join(patched)}",
            flush=True,
        )

    # A lock whose required code is supplied by nothing but its own contains_info
    # (the answer written onto the lock) can't open — bind requires_code to that
    # self clue so it is at least solvable. Runs after the patch pass so a genuine
    # upstream clue is preferred over the degenerate self-clue.
    self_clue = _rebind_self_clue_locks(objects)
    if self_clue:
        print(
            f"[game_master] rebound self-clue locks: {', '.join(self_clue)}",
            flush=True,
        )

    rewrites = _bind_goals(rooms)
    if rewrites:
        print(
            f"[game_master] rebound goals to completion: {', '.join(rewrites)}",
            flush=True,
        )

    rules = [r for r in data.get("rules", []) if isinstance(r, str)]
    solution_path = [s for s in data.get("solution_path", []) if isinstance(s, str)]

    secrets = _secret_tokens(objects)
    redactions, solution_path = _scrub_spoilers(solution_path, secrets)
    if redactions:
        print(f"[game_master] scrubbed spoilers: {', '.join(redactions)}", flush=True)

    coherence = _check_coherence(rooms, objects)
    if coherence:
        print(f"[game_master] coherence warnings: {'; '.join(coherence)}", flush=True)

    # win_condition is a computed property of GameWorld, derived from the final
    # room's goal_completion — no longer stored separately.
    return GameWorld(
        scenario=data.get("scenario", ""),
        objective=data.get("objective", ""),
        rooms=rooms,
        objects=objects,
        rules=rules,
        solution_path=solution_path,
    )


def _world_is_solvable(world: GameWorld) -> bool:
    """True if the heuristic oracle can win the world (i.e. it is winnable).

    Lazy-imports the benchmark harness so the agents package doesn't take an
    import-time dependency on benchmark/ (which itself imports gameplay_node).
    Used only in hard mode to reject the occasional unwinnable generation.
    """
    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import heuristic_policy
    except Exception:
        return True  # harness unavailable — don't block generation
    if not world.win_condition.object_id or not world.rooms:
        return False
    return HeadlessEpisode(world).run(heuristic_policy).victory


def _generate_world(llm, theme: str) -> tuple[GameWorld, str]:
    """One LLM generation -> (validated GameWorld, raw response).

    In hard mode, uses the N-room deep-chain prompt and raises MAX_ROOMS for the
    build so all rooms survive; otherwise the original 2-room prompt/behavior.
    """
    global MAX_ROOMS
    s = Settings()
    if s.hard_mode:
        prompt = BANK_GENERATION_PROMPT.format(
            theme=theme, num_rooms=s.num_rooms,
            chain_depth=s.chain_depth, decoys=s.decoys,
        )
    else:
        prompt = GENERATION_PROMPT.format(theme=theme)

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = _parse_json(response.content) or {}

    orig_cap = MAX_ROOMS
    if s.hard_mode:
        MAX_ROOMS = s.num_rooms
    try:
        world = _build_world(data)
    finally:
        MAX_ROOMS = orig_cap
    return world, response.content


def game_master_node(state: GameState) -> dict:
    s = Settings()
    llm = get_llm("game_master")

    storyline_start = time.perf_counter()
    world, raw = _generate_world(llm, state.theme)
    elapsed = time.perf_counter() - storyline_start

    # Hard mode: regenerate until the oracle confirms the world is winnable, so
    # the live game never starts an unsolvable world.
    attempts = 1
    if s.hard_mode and s.gen_max_attempts > 0:
        while not _world_is_solvable(world) and attempts < s.gen_max_attempts:
            print(
                f"[game_master] world unsolvable — regenerating "
                f"(attempt {attempts + 1}/{s.gen_max_attempts})",
                flush=True,
            )
            world, raw = _generate_world(llm, state.theme)
            attempts += 1

    mode = (
        f"HARD ({len(world.rooms)} rooms, {attempts} attempt(s))"
        if s.hard_mode
        else "standard"
    )
    print(
        f"[game_master] generated {mode} world in {elapsed:.2f}s",
        flush=True,
    )

    return {
        "messages": [AIMessage(content=raw)],
        "world": world,
    }
