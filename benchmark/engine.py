"""Thin, LLM-free wrapper around the gameplay engine for benchmarking.

Reuses the deterministic pieces of :mod:`agents.gameplay_node` unchanged:
the action-space builder, the action resolvers, victory check, and initial
state. A policy is any callable ``(world, ps, action_space) -> action_str``.

The wrapper deliberately mirrors the SINGLE-agent slice of the real loop: one
actor picks one action per tick from the same pruned action space the live
agents see. Multi-agent coordination is intentionally excluded here so a policy
comparison isn't confounded by turn-order/claim effects — it can be layered on
later once single-agent baselines are established.

Note on the GM exit gate: ``_resolve_action`` calls ``evaluate_room_exit`` for
'go' moves, which is an LLM call in the live game. For headless runs we patch
that to a deterministic verdict (allow iff the room goal is satisfied) so the
benchmark stays model-free and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.escape_rooms.nodes import gameplay as gp
from src.escape_rooms.state import GameWorld, PartyState, TickAction


@dataclass
class EpisodeResult:
    victory: bool
    ticks: int
    rooms_visited: int
    objects_resolved: int
    known_info: int
    last_room: str
    win_object_state: str | None
    # Number of ordered dependency links the oracle had to clear: each action that
    # actually unlocked a lock, brought power online, or moved to a new room. This
    # is the real "how deep is the puzzle" measure (unlike raw ticks, which inflate
    # with shuffling) — used to reject shallow worlds from the bank.
    chain_depth: int = 0
    history: list[str] = field(default_factory=list)


# Verbs whose successful resolution clears a dependency link (advances the chain).
_PROGRESS_VERBS = {"enter_code", "use_tool", "insert_liquid", "open", "flip_fuse"}
# Note substrings that mean the action actually changed gate/lock/power state.
_PROGRESS_NOTES = ("unlocked", "→ on", "satisfied", "moved to")


class HeadlessEpisode:
    """Run one world to victory/timeout under a given policy, no LLM calls.

    ``policy(world, ps, action_space) -> str`` must return one of the strings in
    ``action_space``. The engine resolves it and mutates ``ps`` in place,
    exactly as the live loop does.
    """

    def __init__(self, world: GameWorld, max_ticks: int = gp.MAX_TICKS) -> None:
        self.world = world
        self.max_ticks = max_ticks

    def run(self, policy, record_history: bool = False) -> EpisodeResult:
        world = self.world
        ps: PartyState = gp._build_initial_party_state(world)
        history: list[str] = []
        chain_depth = 0

        # The engine renders panels inside _resolve_action; silence all engine
        # stdout for the headless run, then restore in the finally below.
        # Movement is free in the current model (the party may leave any room
        # without a GM exit gate), so there is no gate to patch — the policy
        # drives action selection and _resolve_action mutates state directly.
        orig_stream = gp._stream
        gp._stream = lambda line="": None

        try:
            while not ps.game_over and ps.tick < self.max_ticks:
                if gp._check_victory(world, ps):
                    ps.victory = True
                    break
                ps.tick += 1
                visible = gp._objects_in_room(world, ps)
                action_space = gp._build_action_space(world, ps, visible)
                action = policy(world, ps, action_space)
                if action not in action_space:
                    # A policy must return a legal action; fall back to waiting
                    # rather than corrupting state on a bad return.
                    action = gp.IDLE_ACTION
                note, target = gp._resolve_action(action, world, ps)
                verb = action.split(" ", 1)[0]
                lnote = note.lower()
                if (verb in _PROGRESS_VERBS or verb == "go") and any(
                    p in lnote for p in _PROGRESS_NOTES
                ):
                    chain_depth += 1
                # The live loop appends every action to ps.log; several engine
                # helpers (e.g. _already_examined dedup) read that log, so the
                # headless run must populate it too or it re-picks dead actions.
                ps.log.append(
                    TickAction(
                        tick=ps.tick,
                        agent_id="bench",
                        say="",
                        action=action,
                        target_object=target,
                        note=note,
                    )
                )
                if record_history:
                    history.append(f"t{ps.tick} {action} -> {note}")
        finally:
            gp._stream = orig_stream

        resolved = gp._resolved_object_ids(world, ps)
        win = world.win_condition
        return EpisodeResult(
            victory=bool(ps.victory),
            ticks=ps.tick,
            rooms_visited=len(ps.visited),
            objects_resolved=len(resolved),
            known_info=len(ps.known_info),
            last_room=ps.current_room,
            win_object_state=(
                ps.object_states.get(win.object_id) if win.object_id else None
            ),
            chain_depth=chain_depth,
            history=history,
        )


@dataclass
class AgentEpisodeResult:
    agent_id: str
    final_room: str
    inventory: list[str]
    history: list[str] = field(default_factory=list)


@dataclass
class CooperativeEpisodeResult:
    victory: bool
    ticks: int
    rooms_visited: int
    objects_resolved: int
    known_info: int
    win_object_state: str | None
    chain_depth: int = 0
    agents: list[AgentEpisodeResult] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


class MultiAgentEpisode:
    """Run one world to victory/timeout with N cooperating agents.

    ``policy(agent_id, world, ps, action_space) -> str`` — same contract as
    :class:`HeadlessEpisode`'s policy plus a leading ``agent_id``.

    Per tick: ``ps.tick`` advances once, then each agent in ``agent_ids`` order
    takes one turn (load its perspective, build action space, call policy,
    resolve, save perspective, log). Victory is checked before the tick AND
    after each agent's action so a winning move ends the episode immediately.
    """

    def __init__(
        self, world: GameWorld, agent_ids: list[str], max_ticks: int = gp.MAX_TICKS
    ) -> None:
        self.world = world
        self.agent_ids = agent_ids
        self.max_ticks = max_ticks

    def run(self, policy, record_history: bool = False) -> CooperativeEpisodeResult:
        world = self.world
        ps: PartyState = gp._build_initial_party_state(world, agent_ids=self.agent_ids)
        history: list[str] = []
        per_agent_history: dict[str, list[str]] = {a: [] for a in self.agent_ids}
        chain_depth = 0

        orig_stream = gp._stream
        gp._stream = lambda line="": None

        try:
            while not ps.game_over and ps.tick < self.max_ticks:
                if gp._check_victory(world, ps):
                    ps.victory = True
                    break
                ps.tick += 1

                for agent_id in self.agent_ids:
                    if gp._check_victory(world, ps):
                        ps.victory = True
                        break

                    gp._load_agent_view(ps, agent_id)
                    visible = gp._objects_in_room(world, ps)
                    action_space = gp._build_action_space(world, ps, visible)
                    action = policy(agent_id, world, ps, action_space)
                    if action not in action_space:
                        action = gp.IDLE_ACTION
                    note, target = gp._resolve_action(action, world, ps)
                    gp._save_agent_view(ps, agent_id)

                    verb = action.split(" ", 1)[0]
                    lnote = note.lower()
                    if (verb in _PROGRESS_VERBS or verb == "go") and any(
                        p in lnote for p in _PROGRESS_NOTES
                    ):
                        chain_depth += 1

                    ps.log.append(
                        TickAction(
                            tick=ps.tick,
                            agent_id=agent_id,
                            say="",
                            action=action,
                            target_object=target,
                            note=note,
                        )
                    )
                    if record_history:
                        history.append(f"t{ps.tick} [{agent_id}] {action} -> {note}")
                        per_agent_history[agent_id].append(f"t{ps.tick} {action} -> {note}")

                if ps.victory:
                    break
        finally:
            gp._stream = orig_stream

        resolved: set[str] = set()
        for a in self.agent_ids:
            gp._load_agent_view(ps, a)
            resolved |= gp._resolved_object_ids(world, ps)

        win = world.win_condition
        return CooperativeEpisodeResult(
            victory=bool(ps.victory),
            ticks=ps.tick,
            rooms_visited=len(ps.visited),
            objects_resolved=len(resolved),
            known_info=len(ps.known_info),
            win_object_state=(
                ps.object_states.get(win.object_id) if win.object_id else None
            ),
            chain_depth=chain_depth,
            agents=[
                AgentEpisodeResult(
                    agent_id=a,
                    final_room=ps.agent_rooms.get(a, ""),
                    inventory=list(ps.agent_inventories.get(a, [])),
                    history=per_agent_history[a],
                )
                for a in self.agent_ids
            ],
            history=history,
        )
