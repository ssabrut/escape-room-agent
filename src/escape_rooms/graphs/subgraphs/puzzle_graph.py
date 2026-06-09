"""Constructive puzzle-graph generator — builds a solvable object graph in code.

Instead of asking the LLM for a finished, hopefully-solvable world and then
validating/repairing it, this module *constructs* each room's dependency chain
backward from its goal object using only the four mechanics the engine actually
executes (code, tool, power, liquid). Because every clue/tool is placed on an
object that is reachable strictly before its consumer, the resulting world is
solvable by construction — ``check_solvable`` and the BFS oracle cannot fail on
structure. The LLM is used afterwards only to theme (describe) the nodes.

Engine contract this generator targets (see agents/gameplay_node.py):
  - HIDDEN_STATES = {"locked", "locked_bolt", "locked_room", "hidden"}
  - a locked object is opened by exactly one mechanism:
      requires_code   -> a known contains_info token containing the code digits
      requires_tool   -> a takeable, visible object held in inventory
      requires_power  -> "sekring_<LABEL>_ON" produced by flipping a fuse panel
      requires_liquid -> a held item whose id/description contains the token
  - examine learns contains_info; take grabs a visible takeable object.
  - win = final room's goal object reaching its (non-trivial) target state.
"""

from __future__ import annotations

import random

import json
import re

from src.escape_rooms.state import GameWorld, Prerequisite, Room, WorldObject

# States the engine treats as "needs unlocking". The goal/win target is always
# "unlocked" — never a trivial default ("visible"/"fixed") that check_solvable
# rejects, and never the object's own start state (which would win at tick 0).
LOCKED_STATE = "locked"
TARGET_STATE = "unlocked"

# The mechanics the constructive generator uses. Liquid is intentionally excluded:
# the engine matches it by fuzzy id/description substring while check_solvable
# requires an exact contains_info/id token, so the two disagree on supply — code,
# tool, and power give clean, statically-verifiable chains that fully cover play.
MECHANICS = ("code", "tool", "power")
_MECHANIC_WEIGHTS = (0.5, 0.3, 0.2)

MIN_OBJECTS_PER_ROOM_DEFAULT = 1


class _Builder:
    """Accumulates objects for one world with collision-free ids."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.objects: list[WorldObject] = []
        self._counts: dict[str, int] = {}

    def _new_id(self, stem: str, room_id: str) -> str:
        key = f"{stem}_{room_id}"
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n
        return f"{stem}_{room_id}_{n}"

    def add(self, obj: WorldObject) -> WorldObject:
        self.objects.append(obj)
        return obj

    def _code(self) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.choice((3, 4))))

    # -- one unlocking mechanism applied to `target`, returns helper objects --

    def _gate_with_code(self, target: WorldObject, room_id: str) -> list[WorldObject]:
        code = self._code()
        target.requires_code = code
        target.code_digits = len(code)
        clue = WorldObject(
            id=self._new_id("clue", room_id),
            location=room_id,
            description="",
            state="visible",
            interactable=True,
            takeable=False,
            contains_info=code,
        )
        return [clue]

    def _gate_with_tool(self, target: WorldObject, room_id: str) -> list[WorldObject]:
        tool = WorldObject(
            id=self._new_id("tool", room_id),
            location=room_id,
            description="",
            state="visible",  # reachable before the thing it opens — no cycle
            interactable=True,
            takeable=True,
        )
        target.requires_tool = tool.id
        return [tool]

    def _gate_with_power(self, target: WorldObject, room_id: str) -> list[WorldObject]:
        label = self.rng.choice("ABCDEFGH")
        token = f"sekring_{label}_ON"
        panel = WorldObject(
            id=self._new_id("panel", room_id),
            location=room_id,
            description="",
            state="visible",  # must be reachable to flip the fuse
            interactable=True,
            takeable=False,
            fuses={label: "OFF"},
        )
        target.requires_power = token
        return [panel]

    def gate(self, target: WorldObject, room_id: str, mechanic: str) -> list[WorldObject]:
        return {
            "code": self._gate_with_code,
            "tool": self._gate_with_tool,
            "power": self._gate_with_power,
        }[mechanic](target, room_id)

    def pick_mechanic(self) -> str:
        return self.rng.choices(MECHANICS, weights=_MECHANIC_WEIGHTS, k=1)[0]


def _build_room_chain(
    builder: _Builder, room: Room, chain_depth: int
) -> tuple[WorldObject, list[WorldObject]]:
    """Construct a backward dependency chain ending at this room's goal object.

    Returns (goal_object, all_room_objects). The goal object starts LOCKED and is
    opened by a mechanic; that mechanic's helper may itself be a locked object
    opened by a deeper mechanic, up to ``chain_depth`` links. Every helper is
    reachable strictly before its consumer, so the chain is solvable in order.

    All intermediate links use the tool mechanic so every object in the chain is
    locked behind the previous one — forming a true linked-list dependency. Only
    the final (root) link uses a terminating mechanic (code or power) so its
    helper is directly reachable with no further prerequisite.
    """
    room_objs: list[WorldObject] = []

    goal = builder.add(
        WorldObject(
            id=builder._new_id("goal", room.id),
            location=room.id,
            description="",
            state=LOCKED_STATE,
            interactable=True,
            takeable=False,
        )
    )
    room_objs.append(goal)

    depth = max(1, chain_depth)
    target = goal
    for link in range(depth):
        is_last_link = link == depth - 1
        # Intermediate links always use the tool mechanic so the helper becomes a
        # locked takeable item gated by the next link — every object in the room
        # is reachable only after the one before it. The final (root) link must
        # use a terminating mechanic (code or power) whose helper needs nothing
        # further to be accessed; a terminal tool would be un-takeable (deadlock).
        if is_last_link:
            mechanic = builder.rng.choice(("code", "power"))
        else:
            mechanic = "tool"

        helpers = builder.gate(target, room.id, mechanic)
        for h in helpers:
            builder.add(h)
            room_objs.append(h)

        if is_last_link:
            break
        # Lock the tool helper so it requires the next link to unlock it first.
        nxt = next((h for h in helpers if h.takeable), None)
        if nxt is None:
            break  # safety: terminating mechanic chosen — end chain here
        target = nxt
        target.state = LOCKED_STATE

    return goal, room_objs


def _backfill_scenic(
    builder: _Builder, room_id: str, have: int, min_per_room: int
) -> None:
    for _ in range(max(0, min_per_room - have)):
        builder.add(
            WorldObject(
                id=builder._new_id("scenic", room_id),
                location=room_id,
                description="",
                state="visible",
                interactable=False,
                takeable=False,
                scenic=True,
            )
        )


def build_solvable_world(
    skeleton: GameWorld,
    chain_depth: int,
    min_objects_per_room: int = MIN_OBJECTS_PER_ROOM_DEFAULT,
    seed: int | None = None,
) -> GameWorld:
    """Build a fully-formed, solvable GameWorld from a rooms-only skeleton.

    ``skeleton`` supplies scenario/objective/rooms/adjacency (from world_builder).
    This function discards any goal_completion the skeleton carried and installs a
    constructed, guaranteed-solvable goal chain per room. Object descriptions are
    left blank for the theming pass to fill in.

    Every object in a room is part of the dependency chain — forming a linked list
    where each object requires the previous one. The effective chain depth is
    max(chain_depth, min_objects_per_room - 1) so that the per-room object minimum
    is met entirely by real chained objects, with no inert scenic fillers.
    """
    rng = random.Random(seed)
    builder = _Builder(rng)

    # Each room gets exactly (effective_depth + 1) objects: the goal plus one helper
    # per link. min_objects_per_room - 1 links are needed to reach the minimum count.
    effective_depth = max(chain_depth, max(min_objects_per_room - 1, 1))

    rooms: list[Room] = []
    for room in skeleton.rooms:
        goal, room_objs = _build_room_chain(builder, room, effective_depth)
        # No scenic backfill: all slots are filled by real chained objects above.
        rooms.append(
            Room(
                id=room.id,
                description=room.description,
                adjacency=room.adjacency,
                goal=f"Get {goal.id} into the '{TARGET_STATE}' state.",
                goal_completion=Prerequisite(
                    type="object_state", object_id=goal.id, state=TARGET_STATE
                ),
                key_objects=[goal.id],
            )
        )

    from src.escape_rooms.state import derive_win_condition

    return GameWorld(
        scenario=skeleton.scenario,
        objective=skeleton.objective,
        rooms=rooms,
        objects=builder.objects,
        rules=list(skeleton.rules),
        solution_path=[],
        win_condition=derive_win_condition(rooms),
    )


# ---------------------------------------------------------------------------
# Theming pass — the only LLM call. It cannot alter structure: it returns a
# {id: description} map that we merge onto the already-solvable objects. Any id
# the LLM drops or renames falls back to a generated description, so the world
# ships solvable regardless of theming quality or an LLM failure.
# ---------------------------------------------------------------------------


def classify_role(obj: WorldObject, world: GameWorld) -> str:
    """Human-readable role label for an object, for the theming prompt and fallbacks."""
    if obj.scenic:
        return "scenic"
    if obj.fuses:
        return "panel"
    if obj.contains_info:
        return "clue"
    win_ids = {r.goal_completion.object_id for r in world.rooms if r.goal_completion}
    if obj.id in win_ids:
        return "locked-goal"
    if obj.takeable:
        return "locked-tool" if obj.state in ("locked", "hidden") else "tool"
    return "locked-goal"


_FALLBACK = {
    "scenic": "Nondescript clutter that fills the space — of no use to anyone.",
    "panel": "A fuse panel with a switch that controls power to the room.",
    "clue": "A scrap of writing — study it and a hidden code reveals itself.",
    "tool": "A handy implement, just the thing for prying something open.",
    "locked-tool": "A sealed cache; force it open and a useful tool is inside.",
    "locked-goal": "A stubbornly locked fixture — the heart of this room's puzzle.",
}


def _fallback_description(obj: WorldObject, world: GameWorld) -> str:
    return _FALLBACK.get(classify_role(obj, world), _FALLBACK["locked-goal"])


def _graph_spec(world: GameWorld) -> str:
    lines: list[str] = []
    for room in world.rooms:
        lines.append(f'Room "{room.id}" — goal: {room.goal}')
        for obj in world.objects:
            if obj.location != room.id:
                continue  # constructor anchors every object directly in a room
            role = classify_role(obj, world)
            parts = [f"  - {obj.id} [{role}]"]
            if obj.requires_tool:
                parts.append(f"unlocked-by={obj.requires_tool}")
            if obj.requires_code:
                parts.append("unlocked-by=code")
            if obj.requires_power:
                parts.append(f"unlocked-by=power({obj.requires_power})")
            if obj.contains_info:
                parts.append("reveals=code")
            if obj.fuses:
                labels = ",".join(obj.fuses.keys())
                parts.append(f"controls-power({labels})")
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def _parse_descriptions(text: str) -> dict[str, str]:
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    for candidate in (raw, text):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return {}
    descs = data.get("descriptions", data)
    if not isinstance(descs, dict):
        return {}
    return {k: v for k, v in descs.items() if isinstance(k, str) and isinstance(v, str)}


def _parse_theming_response(text: str) -> dict[str, dict[str, str]]:
    """Parse LLM theming response with both 'names' and 'descriptions' keys.
    
    Returns a dict with 'names' and 'descriptions' keys, each containing a mapping
    of object_id to creative name/description. Handles both new format (with both
    keys) and legacy format (descriptions only).
    """
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    data = None
    for candidate in (raw, text):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            pass
    
    if data is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = None
    
    if not isinstance(data, dict):
        return {"names": {}, "descriptions": {}}
    
    # Extract names and descriptions with fallback for legacy format
    names_dict = data.get("names", {})
    descs_dict = data.get("descriptions", data)  # fallback to top-level for legacy
    
    if not isinstance(names_dict, dict):
        names_dict = {}
    if not isinstance(descs_dict, dict):
        descs_dict = {}
    
    return {
        "names": {k: v for k, v in names_dict.items() if isinstance(k, str) and isinstance(v, str)},
        "descriptions": {k: v for k, v in descs_dict.items() if isinstance(k, str) and isinstance(v, str)},
    }


def apply_theming(world: GameWorld, theme: str, llm=None) -> GameWorld:
    """Fill object names and descriptions from an LLM theming pass, with code fallbacks.

    The structure and puzzle mechanics of ``world`` are never touched — only ``id``
    and ``description`` strings are updated. Objects the LLM omits keep generated,
    role-appropriate values. LLM-suggested names are applied to objects and used
    throughout the world (rooms and object references are updated accordingly).
    """
    names: dict[str, str] = {}  # original_id -> creative_name
    descs: dict[str, str] = {}
    if llm is not None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from src.escape_rooms.prompts import load_prompt

            system = load_prompt("puzzle_builder", "system")
            prompt = load_prompt("puzzle_builder", "theming").format(
                theme=theme,
                scenario=world.scenario,
                objective=world.objective,
                graph=_graph_spec(world),
            )
            resp = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=prompt)]
            )
            parsed = _parse_theming_response(resp.content)
            names = parsed.get("names", {})
            descs = parsed.get("descriptions", {})
        except Exception:
            names = {}
            descs = {}

    # Build old_id -> new_id mapping for updating references.
    # Slugify LLM names to snake_case so they remain valid single-token action
    # targets (the engine splits actions on whitespace and uses parts[1] as the id).
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower().strip()).strip("_")
        return slug or "object"

    id_mapping: dict[str, str] = {}
    used_slugs: set[str] = set()
    for obj in world.objects:
        if obj.id in names:
            slug = _slugify(names[obj.id])
            # Ensure uniqueness: append a counter if slug already taken
            if slug in used_slugs:
                n = 2
                while f"{slug}_{n}" in used_slugs:
                    n += 1
                slug = f"{slug}_{n}"
            used_slugs.add(slug)
            id_mapping[obj.id] = slug
            obj.id = slug
        else:
            used_slugs.add(obj.id)

    # Update references throughout the world
    for obj in world.objects:
        if obj.location in id_mapping:
            obj.location = id_mapping[obj.location]
        if obj.requires_tool and obj.requires_tool in id_mapping:
            obj.requires_tool = id_mapping[obj.requires_tool]

    for room in world.rooms:
        room.key_objects = [id_mapping.get(oid, oid) for oid in room.key_objects]
        if room.goal_completion and room.goal_completion.object_id:
            if room.goal_completion.object_id in id_mapping:
                new_oid = id_mapping[room.goal_completion.object_id]
                room.goal_completion.object_id = new_oid
                room.goal = f"Get {new_oid} into the '{room.goal_completion.state}' state."

    # Update win condition if present
    if world.win_condition and world.win_condition.object_id in id_mapping:
        world.win_condition.object_id = id_mapping[world.win_condition.object_id]

    # Remap descriptions from original ids to new slugs so lookups work after renaming.
    slug_descs: dict[str, str] = {}
    for orig_id, desc in descs.items():
        new_id = id_mapping.get(orig_id, orig_id)
        slug_descs[new_id] = desc

    # Apply descriptions with fallback
    for obj in world.objects:
        themed = slug_descs.get(obj.id)
        obj.description = themed if themed else _fallback_description(obj, world)
    return world

