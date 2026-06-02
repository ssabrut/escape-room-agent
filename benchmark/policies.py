"""LLM-free policies for the headless benchmark.

A policy is ``(world, ps, action_space) -> action_str`` returning one of the
strings in ``action_space``. These are the baselines a future RL policy is
measured against:

- ``random_policy``      — uniform over the legal space (lower bound).
- ``first_policy``       — always the first listed action (cheap deterministic).
- ``heuristic_policy``   — win-directed greedy: prefer unlocks/clues, then step
                           toward the win room via the engine's own BFS flow.
"""

from __future__ import annotations

import random

from agents import gameplay_node as gp

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
    "blocked", "missing", "no matching", "unknown", "cannot", "not accessible",
    "no direct", "no target", "no fuse", "dead end", "nothing new", "code unknown",
    "still locked", "already", "needs no", "no hidden info",
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
