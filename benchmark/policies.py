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

from src.escape_rooms.nodes import gameplay as gp
from src.escape_rooms.state import GameWorld, WorldObject

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


def _win_room(world: "GameWorld") -> str | None:
    """Room id that (transitively) holds the win-condition object, or None."""
    win = world.win_condition
    if not win.object_id:
        return None
    by_id = {o.id: o for o in world.objects}
    obj = by_id.get(win.object_id)
    seen: set[str] = set()
    while obj is not None and obj.id not in seen:
        seen.add(obj.id)
        if obj.location in by_id:
            obj = by_id[obj.location]  # object nested inside another — walk up
        else:
            return obj.location  # location is a room id
    return None


def _next_room_toward_win(world: "GameWorld", ps) -> str | None:
    """First room-id hop on the shortest adjacency path from the party's current
    room to the win room. None when already there or no path exists.

    Movement is free in the current model, so routing is a plain BFS over the
    room adjacency graph (no goal gates to consider).
    """
    from collections import deque

    target = _win_room(world)
    if target is None or target == ps.current_room:
        return None
    rooms = {r.id: r for r in world.rooms}
    prev: dict[str, str | None] = {ps.current_room: None}
    queue = deque([ps.current_room])
    while queue:
        cur = queue.popleft()
        if cur == target:
            break
        for neighbor in rooms.get(cur).adjacency.values() if cur in rooms else []:
            if neighbor not in prev:
                prev[neighbor] = cur
                queue.append(neighbor)
    if target not in prev:
        return None  # unreachable
    # Walk back from target to the first hop out of the current room.
    hop = target
    while prev[hop] is not None and prev[hop] != ps.current_room:
        hop = prev[hop]
    return hop


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

    next_hop = _next_room_toward_win(world, ps)

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
# BFS policy
# ---------------------------------------------------------------------------


def bfs_policy(world: "GameWorld", max_states: int = 50_000):
    """Return a policy that replays the shortest winning action sequence found by BFS.

    BFS explores the full reachable state space from the initial party state,
    expanding one action per node. When the win condition is reached, the path
    is traced back and returned as a callable policy that emits the pre-computed
    actions in order, then idles.

    Returns ``None`` if no winning state is found within ``max_states`` nodes
    (world is provably unsolvable under the engine's rules, or the state space
    is too large). In that case, fall back to ``heuristic_policy``.

    The returned policy is a plain function ``(world, ps, action_space) -> str``
    compatible with ``HeadlessEpisode.run()``.

    State representation: a frozenset-based snapshot of the fields that gate
    progress — room, inventory, known_info, object_states, power_active. Log
    and tick are excluded (they don't affect what actions are possible).
    """
    from collections import deque

    from src.escape_rooms.nodes import gameplay as _gp
    from src.escape_rooms.state import PartyState as _PartyState

    def _snapshot(ps: _PartyState) -> tuple:
        return (
            ps.current_room,
            tuple(sorted(ps.inventory)),
            tuple(sorted(ps.known_info)),
            tuple(sorted(ps.object_states.items())),
            tuple(sorted(ps.power_active)),
            # fuse_states affects power_active, but power_active already captures
            # what matters for precondition checks — skip the redundant nested dict
        )

    def _copy_ps(ps: _PartyState) -> _PartyState:
        # Shallow-copy the mutable containers; log stays empty (not needed for BFS)
        return ps.model_copy(
            update={
                "inventory": list(ps.inventory),
                "known_info": list(ps.known_info),
                "object_states": dict(ps.object_states),
                "power_active": set(ps.power_active),
                "fuse_states": {k: dict(v) for k, v in ps.fuse_states.items()},
                "visited": set(ps.visited),
                "log": [],
                "spotted_clues": [],
                "observed_rooms": set(ps.observed_rooms),
                "room_observations": dict(ps.room_observations),
                "room_plans": dict(ps.room_plans),
                "global_object_observations": dict(ps.global_object_observations),
                "last_fingerprint": None,
            }
        )

    # Movement is free in the current model (no GM exit gate), so there is
    # nothing to patch — just silence engine stdout for the search.
    orig_stream = _gp._stream
    _gp._stream = lambda line="": None

    plan: list[str] = []

    try:
        initial_ps = _gp._build_initial_party_state(world)
        start_snap = _snapshot(initial_ps)

        # BFS: each node is (snapshot, ps_copy, actions_so_far)
        # We store only the snapshot->parent_action map to reconstruct the path.
        parent: dict[tuple, tuple[tuple, str] | None] = {start_snap: None}
        queue: deque[tuple] = deque([(start_snap, initial_ps)])
        found: tuple | None = None

        while queue and len(parent) < max_states:
            snap, ps = queue.popleft()

            if _gp._check_victory(world, ps):
                found = snap
                break

            visible = _gp._objects_in_room(world, ps)
            action_space = _gp._build_action_space(world, ps, visible)

            for action in action_space:
                if action == _gp.IDLE_ACTION:
                    continue  # idle never advances state — prune to avoid explosion
                child_ps = _copy_ps(ps)
                child_ps.tick += 1
                _gp._resolve_action(action, world, child_ps)
                child_snap = _snapshot(child_ps)
                if child_snap in parent:
                    continue
                parent[child_snap] = (snap, action)
                queue.append((child_snap, child_ps))

        # Reconstruct action sequence by tracing back through parent pointers.
        if found is not None:
            path: list[str] = []
            cur = found
            while parent[cur] is not None:
                prev_snap, action = parent[cur]
                path.append(action)
                cur = prev_snap
            path.reverse()
            plan = path

    finally:
        _gp._stream = orig_stream

    if not plan:
        # BFS found no solution — fall back to heuristic so the episode still runs
        return heuristic_policy

    pending = list(plan)

    def _bfs_policy(w, ps, action_space):
        while pending:
            nxt = pending[0]
            if nxt in action_space:
                pending.pop(0)
                return nxt
            # Action not yet legal (gate not open) — idle and wait
            return _gp.IDLE_ACTION
        return _gp.IDLE_ACTION

    return _bfs_policy


def _world_tick_budget(world: "GameWorld") -> int:
    """Compute a tick budget scaled to the world's complexity.

    Each non-scenic object needs at most ~3 ticks to resolve (examine, take/flip,
    unlock). Room transitions add one tick each. We use 4× the object count plus
    the number of rooms as a generous upper bound, floored at MAX_TICKS.
    """
    non_scenic = sum(1 for o in world.objects if not o.scenic)
    num_rooms = len(world.rooms)
    return max(gp.MAX_TICKS, 4 * non_scenic + num_rooms)


def oracle_solve(world: "GameWorld"):
    """Deterministic best-effort solve (no LLM): exhaustive BFS first, greedy fallback.

    `bfs_policy` does a complete breadth-first search of the reachable state space
    and replays the shortest winning path — so if a win exists within its state
    budget, the run WILL escape (no false "unsolvable" verdict from a greedy policy
    getting stuck). When BFS finds no win in budget (state space too large to
    enumerate), fall back to `heuristic_policy` for a best-effort second opinion.

    Returns the EpisodeResult; `.victory` is the authoritative solvability verdict
    and `.chain_depth` is the depth of the (shortest, when BFS succeeds) solution.
    """
    from benchmark.engine import HeadlessEpisode

    policy = bfs_policy(world) or heuristic_policy
    return HeadlessEpisode(world, max_ticks=_world_tick_budget(world)).run(
        policy, record_history=True
    )


# ---------------------------------------------------------------------------
# Ground-truth solution path — derived from the oracle's winning trace
# ---------------------------------------------------------------------------


def _action_of(trace_line: str) -> str:
    """Extract the raw action ('enter_code safe_3') from a 'tN <action> -> note' line."""
    body = (
        trace_line.split(" ", 1)[1]
        if trace_line[:1] == "t" and " " in trace_line
        else trace_line
    )
    return body.split(" -> ", 1)[0].strip()


def _replay_wins(world: "GameWorld", actions: list[str]) -> bool:
    """Replay a fixed action sequence through the engine; True if it reaches victory.

    A scripted policy emits the given actions in order (skipping any not currently
    legal), so we can test whether a *subset* of the trace still solves the world.
    """
    from benchmark.engine import HeadlessEpisode

    pending = list(actions)

    def _scripted(_world, _ps, action_space):
        while pending:
            nxt = pending.pop(0)
            if nxt in action_space:
                return nxt
        return gp.IDLE_ACTION

    return HeadlessEpisode(world, max_ticks=_world_tick_budget(world)).run(_scripted).victory


def _minimal_actions(world: "GameWorld", actions: list[str]) -> list[str]:
    """Greedy leave-one-out minimization of a winning action sequence.

    A step is necessary iff dropping it makes the remaining sequence fail to win.
    One in-order pass removes every individually-droppable step (dead-end opens,
    redundant examines) while preserving a still-winning path.
    """
    kept = list(actions)
    i = 0
    while i < len(kept):
        trial = kept[:i] + kept[i + 1 :]
        if _replay_wins(world, trial):
            kept = trial
        else:
            i += 1
    return kept


def _annotate_path(world: "GameWorld", actions: list[str]) -> list[str]:
    """Replay the minimal winning actions, annotating each with the room it happens
    in and the engine's outcome note — i.e. action + context.

    e.g. "3. [cargo_hold] use_tool door_1 — used brass_key on door_1 → unlocked"
    The room shown is where the party stands when the action is taken (for a 'go'
    that is the room being left).
    """
    orig_stream = gp._stream
    gp._stream = lambda line="": None
    steps: list[str] = []
    try:
        ps = gp._build_initial_party_state(world)
        for action in actions:
            room = ps.current_room
            note, _ = gp._resolve_action(action, world, ps)
            steps.append(f"{len(steps) + 1}. [{room}] {action} — {note}")
    finally:
        gp._stream = orig_stream
    return steps


def bfs_solution_path(world: "GameWorld") -> list[str]:
    """Ground-truth solution path for `world`, derived from a deterministic solve.

    Solves with `bfs_policy` (the shortest winning path, or the greedy heuristic
    fallback when the state space exceeds the BFS budget), minimises the winning
    trace leave-one-out, then annotates each step with its room and outcome note.
    The result is a numbered, hallucination-free path over the world's ACTUAL
    object graph — the canonical answer key. Returns [] if the world is unsolvable.
    """
    from benchmark.engine import HeadlessEpisode

    policy = bfs_policy(world)
    result = HeadlessEpisode(world, max_ticks=_world_tick_budget(world)).run(
        policy, record_history=True
    )
    if not result.victory:
        return []
    actions = [_action_of(line) for line in result.history if line.strip()]
    minimal = _minimal_actions(world, actions)
    return _annotate_path(world, minimal)


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
    - Goal objects that are not present/reachable in the room they gate
      (object_state goal object must live in that room; has_item goal item
      must live in a room reachable on the way to it)
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
                issues.append(
                    f"room '{room.id}' is disconnected — unreachable from start '{start}'"
                )

    # --- Win condition object must exist and require real effort ---
    win = world.win_condition
    if not win.object_id:
        issues.append(
            "win_condition has no object_id — final room has no object_state goal_completion"
        )
    elif win.object_id not in obj_by_id:
        issues.append(f"win_condition references non-existent object '{win.object_id}'")
    else:
        win_obj = obj_by_id[win.object_id]
        # A win object that already starts in its target state means the world is
        # won at tick 0 — every policy "wins" instantly with 0 ticks. Reject so
        # puzzle_builder regenerates a goal that demands actual play.
        if win.state and win_obj.state == win.state:
            issues.append(
                f"win_condition is trivially satisfied — '{win.object_id}' already "
                f"starts in target state '{win.state}', so the world is won at tick 0"
            )
        # 'visible'/'fixed' are inert default states, never a meaningful WIN target
        # (the goal should be an unlock/transform, e.g. 'unlocked').
        if win.state in {"visible", "fixed"}:
            issues.append(
                f"win_condition target state '{win.state}' is a trivial/default state "
                f"for '{win.object_id}' — a win goal must require a real change "
                f"(e.g. 'unlocked')"
            )

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
            matching = [tok for tok in info_producers if code in tok]
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
                        issues.append(f"circular tool dependency: {' -> '.join(cycle)}")

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
                o
                for o in world.objects
                if (o.contains_info and token in o.contains_info) or o.id == token
            ]
            if not producers:
                issues.append(
                    f"object '{obj.id}' requires_liquid '{token}' but no object "
                    f"in the world supplies it"
                )

    # --- Resolve an object's home room by walking its location chain ---
    # An object's `location` is a room id, or another object's id when nested.
    # Walk up parent objects until we land on a room id. Returns None on a
    # dangling/cyclic chain.
    def _room_of(object_id: str) -> str | None:
        seen: set[str] = set()
        cur = object_id
        while cur in obj_by_id:
            if cur in seen:
                return None  # cycle in nesting chain
            seen.add(cur)
            loc = obj_by_id[cur].location
            if loc in room_ids:
                return loc
            cur = loc
        return None

    # --- Reachable-room set: every room on a path from start up to a target ---
    # For a has_item goal the party may carry the item in from an earlier room,
    # so the supplier may live in any room reachable on the way to the goal room.
    def _rooms_up_to(target_room: str) -> set[str]:
        if not world.rooms:
            return set()
        start_room = world.rooms[0].id
        # BFS from start; any room from which target_room is still reachable counts.
        adj: dict[str, set[str]] = {r.id: set() for r in world.rooms}
        for r in world.rooms:
            for n in r.adjacency.values():
                if n in room_ids:
                    adj[r.id].add(n)
        # rooms reachable from start
        from_start: set[str] = set()
        q = [start_room]
        while q:
            c = q.pop()
            if c in from_start:
                continue
            from_start.add(c)
            q.extend(adj.get(c, ()))
        return from_start if target_room in from_start else set()

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
            elif gc.object_id:
                # Goal achievability: an object_state goal is satisfied by acting on
                # the object where it sits, so it MUST live in this room.
                home = _room_of(gc.object_id)
                if home != room.id:
                    where = f"room '{home}'" if home else "no resolvable room"
                    issues.append(
                        f"room '{room.id}' goal unachievable — goal object "
                        f"'{gc.object_id}' is in {where}, not in this room"
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
            elif gc.object_id:
                # Goal achievability: the party can carry the item in, so its home
                # room only has to be reachable on a path leading to this room.
                home = _room_of(gc.object_id)
                if home is None:
                    issues.append(
                        f"room '{room.id}' goal unachievable — goal item "
                        f"'{gc.object_id}' has no resolvable room location"
                    )
                elif home not in _rooms_up_to(room.id):
                    issues.append(
                        f"room '{room.id}' goal unachievable — goal item "
                        f"'{gc.object_id}' lives in room '{home}', which is not "
                        f"reachable on the way to this room"
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
