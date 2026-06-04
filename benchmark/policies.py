"""LLM-free policies for the headless benchmark.

A policy is ``(world, ps, action_space) -> action_str`` returning one of the
strings in ``action_space``. These are the baselines a future RL policy is
measured against:

- ``random_policy``      — uniform over the legal space (lower bound).
- ``first_policy``       — always the first listed action (cheap deterministic).
- ``heuristic_policy``   — win-directed greedy: prefer unlocks/clues, then step
                           toward the win room via the engine's own BFS flow.

Static solvability check (not a policy):

- ``check_solvable``     — backward-chain analysis over the world graph. Walks
                           backward from win_condition, resolving every
                           prerequisite to its supplier (contains_info,
                           takeable tool, power fuse). Returns a
                           ``SolvabilityReport`` with a boolean result and a
                           list of blocking issues. Runs in O(objects²) — no
                           simulation, no tick budget. Use it before running
                           ``HeadlessEpisode`` to distinguish structural
                           impossibilities from runtime policy failures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agents import gameplay_node as gp
from state import GameWorld, WorldObject

# Verb priority for the heuristic: progress-making actions first, movement last,
# wait never (unless it's all that's offered).
_VERB_PRIORITY = {
    "enter_code": 0,
    "use_tool": 0,
    "insert_liquid": 0,
    "open": 0,
    "flip_fuse": 1,
    "examine": 2,
    "take": 2,
    "go": 3,
    "wait": 9,
}


def _verb_of(action: str) -> str:
    return action.split(" ", 1)[0]


# Note fragments that mean an action did NOT advance state — a retry is pointless
# until something else changes. Mirrors gameplay_node._FAIL_KEYWORDS plus the GM
# block phrasing.
_DEAD_NOTE_FRAGMENTS = (
    "blocked",
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
    "code unknown",
    "still locked",
    "already",
    "needs no",
    "no hidden info",
)


# Note fragments meaning an action DID change world state — clears the "dead"
# mark on previously-failed actions, since a precondition may now be met.
_PROGRESS_FRAGMENTS = ("unlocked", "took ", "learned", "→ on", "satisfied", "moved to")


def _fuse_is_on(action: str, ps) -> bool:
    """True if the fuse named by a 'flip_fuse <obj> <label>' action is already ON.

    Re-flipping an ON fuse toggles it back OFF, so the oracle must avoid re-picking
    it once power is up. Reads the live fuse_states the engine mutates.
    """
    parts = action.split()
    if len(parts) < 3:
        return False
    _, obj_id, label = parts[0], parts[1], parts[2]
    return ps.fuse_states.get(obj_id, {}).get(label) == "ON"


def _recently_dead(action: str, ps) -> bool:
    """True if `action` failed and NOTHING has changed world state since.

    The oracle is otherwise stateless per tick, so without this it re-picks the
    same GM-blocked 'go' or 'missing tool' action every tick until timeout. We
    scan the log newest-first:
      - if we hit a state-changing action first, the world moved since any earlier
        failure of `action`, so it might now succeed -> NOT dead.
      - if we hit a prior failure of `action` first, it's still dead (no progress
        in between) -> skip it this tick.
    """
    for entry in reversed(ps.log):
        note = (entry.note or "").lower()
        if any(frag in note for frag in _PROGRESS_FRAGMENTS):
            return False  # progress since the last failure — give it another shot
        if entry.action == action and any(
            frag in note for frag in _DEAD_NOTE_FRAGMENTS
        ):
            return True
    return False


def random_policy(rng: random.Random):
    def _policy(world, ps, action_space):
        return rng.choice(action_space)

    return _policy


def first_policy(world, ps, action_space):
    return action_space[0]


def heuristic_policy(world, ps, action_space):
    """Greedy, win-directed action selection over the legal space.

    1. Take any directly-unlocking action (code/tool/liquid/power) — these are
       only offered when satisfiable, so each clears a lock.
    2. Else grab/examine/flip to gather clues, tools, power.
    3. Else move — preferring the BFS next-hop toward the win room when that
       exit is offered, otherwise any exit.
    4. Else wait.

    Actions that failed since the last state change are deprioritized so the
    oracle stops head-banging a GM-blocked exit or a 'missing tool' it can't yet
    satisfy — but they remain available as a last resort if nothing else is live.
    """
    if not action_space:
        return gp.IDLE_ACTION

    next_hop = gp._next_room_toward_win(world, ps)

    def _rank(action: str):
        verb = _verb_of(action)
        base = _VERB_PRIORITY.get(verb, 5)
        # Among 'go' moves, the win-directed hop sorts ahead of the rest.
        toward = 0 if verb == "go" and next_hop and action == f"go {next_hop}" else 1
        # A fuse already ON should not be re-flipped (that toggles it back OFF and
        # oscillates forever); sink it so the oracle moves on once power is up.
        if verb == "flip_fuse" and _fuse_is_on(action, ps):
            base = 8
        # Sink known-dead actions to the bottom so live work is tried first; they
        # still sort among themselves and can be chosen if they're all that's left.
        dead = 1 if _recently_dead(action, ps) else 0
        return (dead, base, toward, action)

    return min(action_space, key=_rank)


# ---------------------------------------------------------------------------
# Static backward-chain solvability checker
# ---------------------------------------------------------------------------

@dataclass
class SolvabilityReport:
    solvable: bool
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.solvable:
            return "SOLVABLE — no structural issues found"
        lines = [f"UNSOLVABLE — {len(self.issues)} issue(s):"]
        for i, issue in enumerate(self.issues, 1):
            lines.append(f"  {i}. {issue}")
        return "\n".join(lines)


def check_solvable(world: "GameWorld") -> SolvabilityReport:
    """Backward-chain static solvability analysis — no simulation, no tick budget.

    Walks backward from the win_condition through every prerequisite, confirming
    that each has a reachable supplier in the object graph. Reports every blocking
    issue found rather than stopping at the first.

    Catches:
    - Missing contains_info supplier for a requires_code or known_info goal
    - Bare digit codes with no matching contains_info token
    - requires_tool referencing a non-existent or non-takeable object
    - requires_tool circular dependencies (A needs B, B needs A)
    - requires_power with no fuse panel that can activate it
    - known_info goal tokens with no upstream contains_info
    - Room goal_completion targets that reference non-existent objects
    - Win condition object that does not exist
    - Rooms with no path to the win room (disconnected graph)
    """
    issues: list[str] = []

    obj_by_id: dict[str, WorldObject] = {o.id: o for o in world.objects}
    room_ids: set[str] = {r.id for r in world.rooms}

    # --- Room connectivity: every room must be reachable from the start room ---
    if world.rooms:
        start = world.rooms[0].id
        reachable: set[str] = set()
        queue = [start]
        while queue:
            cur = queue.pop()
            if cur in reachable:
                continue
            reachable.add(cur)
            room = next((r for r in world.rooms if r.id == cur), None)
            if room:
                for neighbor in room.adjacency.values():
                    if neighbor in room_ids and neighbor not in reachable:
                        queue.append(neighbor)
        for room in world.rooms:
            if room.id not in reachable:
                issues.append(f"room '{room.id}' is disconnected — unreachable from start '{start}'")

    # --- Win condition object must exist ---
    win = world.win_condition
    if not win.object_id:
        issues.append("win_condition has no object_id — final room has no object_state goal_completion")
    elif win.object_id not in obj_by_id:
        issues.append(f"win_condition references non-existent object '{win.object_id}'")

    # --- Build lookup tables for the backward walk ---

    # All info tokens produced by the world (contains_info values)
    info_producers: dict[str, str] = {}  # token -> object_id that produces it
    for o in world.objects:
        if o.contains_info:
            if o.contains_info in info_producers:
                issues.append(
                    f"duplicate contains_info token '{o.contains_info}' on "
                    f"'{o.id}' and '{info_producers[o.contains_info]}'"
                )
            else:
                info_producers[o.contains_info] = o.id

    # All power tokens activatable by fuse panels
    power_producers: set[str] = set()
    for o in world.objects:
        if o.fuses and o.requires_power is None:
            # A panel with fuses but no requires_power itself is a power source.
            # The power token is the requires_power value on objects that need it,
            # and each panel's fuses dict keys are the fuse labels. The power token
            # that the panel activates is stored in fuses values OR derived from the
            # object id. We look for any object whose requires_power matches a panel
            # id pattern by checking all fuse panels.
            power_producers.add(o.id)

    # Map requires_power token -> panel object ids that can satisfy it.
    # A panel satisfies a requires_power token when the panel's id appears as a
    # prefix of the token OR the token matches a fuse label on the panel. We use a
    # broad match: any panel whose id is a substring of the token, consistent with
    # the game_master convention (e.g. requires_power="sekring_C_ON", panel id
    # "sekring_C").
    def _panel_satisfies_power(panel: "WorldObject", token: str) -> bool:
        if panel.id in token:
            return True
        if panel.fuses:
            return any(label in token for label in panel.fuses)
        return False

    panels = [o for o in world.objects if o.fuses]

    # --- Tool circular dependency detection via DFS ---
    def _has_tool_cycle(start_id: str) -> list[str] | None:
        visited: set[str] = set()
        path: list[str] = []

        def _dfs(oid: str) -> list[str] | None:
            if oid in visited:
                return None
            if oid in path:
                cycle_start = path.index(oid)
                return path[cycle_start:] + [oid]
            path.append(oid)
            obj = obj_by_id.get(oid)
            if obj and obj.requires_tool and obj.requires_tool in obj_by_id:
                result = _dfs(obj.requires_tool)
                if result:
                    return result
            path.pop()
            visited.add(oid)
            return None

        return _dfs(start_id)

    seen_cycles: set[frozenset] = set()

    # --- Walk every object that has a prerequisite ---
    for obj in world.objects:
        if obj.scenic:
            continue  # scenic props are exempt — they carry no prerequisites

        # requires_code: needs a contains_info token containing the code digits
        if obj.requires_code:
            code = obj.requires_code
            matching = [
                tok for tok in info_producers if code in tok
            ]
            if not matching:
                issues.append(
                    f"object '{obj.id}' requires_code '{code}' but no contains_info "
                    f"token in the world contains '{code}'"
                )

        # requires_tool: the tool must exist and be takeable, with no circular dep
        if obj.requires_tool:
            tool_id = obj.requires_tool
            if tool_id not in obj_by_id:
                issues.append(
                    f"object '{obj.id}' requires_tool '{tool_id}' which does not exist"
                )
            else:
                tool = obj_by_id[tool_id]
                if not tool.takeable:
                    issues.append(
                        f"object '{obj.id}' requires_tool '{tool_id}' but that object "
                        f"is not takeable"
                    )
                # Circular dependency check
                cycle = _has_tool_cycle(obj.id)
                if cycle:
                    key = frozenset(cycle)
                    if key not in seen_cycles:
                        seen_cycles.add(key)
                        issues.append(
                            f"circular tool dependency: {' -> '.join(cycle)}"
                        )

        # requires_power: needs a fuse panel that can activate the token
        if obj.requires_power:
            token = obj.requires_power
            satisfying = [p for p in panels if _panel_satisfies_power(p, token)]
            if not satisfying:
                issues.append(
                    f"object '{obj.id}' requires_power '{token}' but no fuse panel "
                    f"in the world can activate it"
                )

        # requires_liquid: we can only flag if no object contains matching info —
        # liquid supply is free-form so just warn if no object references the token
        if obj.requires_liquid:
            token = obj.requires_liquid
            producers = [
                o for o in world.objects
                if (o.contains_info and token in o.contains_info) or o.id == token
            ]
            if not producers:
                issues.append(
                    f"object '{obj.id}' requires_liquid '{token}' but no object "
                    f"in the world supplies it"
                )

    # --- Room goal_completion prerequisites ---
    for room in world.rooms:
        gc = room.goal_completion
        if gc is None:
            continue
        if gc.type == "object_state":
            if gc.object_id and gc.object_id not in obj_by_id:
                issues.append(
                    f"room '{room.id}' goal_completion references non-existent "
                    f"object '{gc.object_id}'"
                )
        elif gc.type == "known_info":
            if gc.info and gc.info not in info_producers:
                issues.append(
                    f"room '{room.id}' goal_completion requires known_info '{gc.info}' "
                    f"but no object has contains_info matching that token"
                )
        elif gc.type == "has_item":
            if gc.object_id and gc.object_id not in obj_by_id:
                issues.append(
                    f"room '{room.id}' goal_completion requires has_item '{gc.object_id}' "
                    f"which does not exist"
                )
            elif gc.object_id and not obj_by_id[gc.object_id].takeable:
                issues.append(
                    f"room '{room.id}' goal_completion requires has_item '{gc.object_id}' "
                    f"but that object is not takeable"
                )
        elif gc.type == "power_active":
            if gc.id:
                satisfying = [p for p in panels if _panel_satisfies_power(p, gc.id)]
                if not satisfying:
                    issues.append(
                        f"room '{room.id}' goal_completion requires power_active '{gc.id}' "
                        f"but no fuse panel can activate it"
                    )

    return SolvabilityReport(solvable=len(issues) == 0, issues=issues)
