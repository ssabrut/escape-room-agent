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

from agents import gameplay_node as gp
from agents.game_master_eval import GMVerdict
from state import GameWorld, PartyState, TickAction


def _deterministic_exit_gate(
    world, ps, room, dest, satisfied, completion_str
) -> GMVerdict:
    """Model-free stand-in for ``evaluate_room_exit`` used during benchmarking.

    Allows the move exactly when the room's goal_completion is satisfied (the
    same fact the live GM is told); skips the narration LLM call entirely.
    """
    if satisfied:
        return GMVerdict(
            allow=True, narration="(headless) goal met — proceed", missing=""
        )
    return GMVerdict(
        allow=False,
        narration="(headless) goal not met",
        missing=f"needs: {completion_str}",
    )


@dataclass
class EpisodeResult:
    victory: bool
    ticks: int
    rooms_visited: int
    objects_resolved: int
    known_info: int
    last_room: str
    win_object_state: str | None
    history: list[str] = field(default_factory=list)


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

        # Patch the GM exit gate to the deterministic verdict for the duration of
        # this episode, then restore — keeps the run model-free without editing
        # the engine module.
        orig_gate = gp._gm_gate_exit

        def _patched_gate(w, p, room, dest):
            completion = room.goal_completion
            satisfied = completion is None or gp._goal_completion_satisfied(
                completion, p
            )
            completion_str = gp._format_goal_completion(completion)
            return _deterministic_exit_gate(w, p, room, dest, satisfied, completion_str)

        # The engine renders GM verdict panels inside _resolve_action; silence
        # all engine stdout for the headless run.
        orig_stream = gp._stream
        gp._stream = lambda line="": None

        gp._gm_gate_exit = _patched_gate
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
            gp._gm_gate_exit = orig_gate
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
            history=history,
        )
