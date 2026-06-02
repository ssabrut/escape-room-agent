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
    """
    if not action_space:
        return gp.IDLE_ACTION

    next_hop = gp._next_room_toward_win(world, ps)

    def _rank(action: str):
        verb = _verb_of(action)
        base = _VERB_PRIORITY.get(verb, 5)
        # Among 'go' moves, the win-directed hop sorts ahead of the rest.
        toward = 0 if verb == "go" and next_hop and action == f"go {next_hop}" else 1
        return (base, toward, action)

    return min(action_space, key=_rank)
