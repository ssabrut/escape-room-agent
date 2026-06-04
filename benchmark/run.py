"""Headless benchmark runner — compare policies over saved worlds, no LLM.

Loads every ``game_master/output.json`` under a smoke-run directory as a
``GameWorld``, runs each policy on each world, and prints an aggregate table of
win-rate and mean ticks-to-win.

Usage:
    python -m benchmark.run                      # all worlds under smoke_runs/
    python -m benchmark.run --glob 'smoke_runs/20260602_133313/**/game_master/output.json'
    python -m benchmark.run --episodes 5         # repeat stochastic policies N times
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics

# Ensure project root is on sys.path so this file can be run directly
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.engine import EpisodeResult, HeadlessEpisode
from benchmark.policies import bfs_policy, first_policy, heuristic_policy, random_policy
from state import GameWorld

DEFAULT_GLOB = "smoke_runs/**/game_master/output.json"


def load_worlds(pattern: str) -> list[tuple[str, GameWorld]]:
    """Load each saved game_master output as (label, GameWorld).

    Skips files with no usable win condition (nothing to solve toward).
    """
    out: list[tuple[str, GameWorld]] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        world_data = raw.get("world", raw)
        try:
            world = GameWorld.model_validate(world_data)
        except Exception:
            continue
        if not world.win_condition.object_id or not world.rooms:
            continue
        # Label by the two parent dirs so runs stay distinguishable.
        p = Path(path)
        label = f"{p.parent.parent.parent.name}/{p.parent.parent.name}"
        out.append((label, world))
    return out


def _summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    wins = [r for r in results if r.victory]
    win_ticks = [r.ticks for r in wins]
    return {
        "episodes": n,
        "win_rate": (len(wins) / n) if n else 0.0,
        "mean_ticks_to_win": statistics.mean(win_ticks) if win_ticks else None,
        "mean_ticks_all": statistics.mean([r.ticks for r in results]) if n else None,
        "mean_objects_resolved": (
            statistics.mean([r.objects_resolved for r in results]) if n else None
        ),
    }


# Episodes per policy for a single-world benchmark: stochastic policies are
# averaged so win% is meaningful; deterministic ones take one exact path.
# BFS is deterministic and stateful (pending list drains), so it always runs
# once — the factory regenerates a fresh closure per world/episode.
_SINGLE_WORLD_EPISODES = {"random": 20, "first": 20, "heuristic": 20, "bfs": 1}


def compute_policy_benchmark(world) -> list[dict]:
    """Run every baseline policy on one world and return their summary rows.

    Shared by the live game_master print path and the smoke runner's JSON dump
    so both report identical numbers. Seeded for reproducibility.
    """
    rng = random.Random(0)
    policies = [
        ("random", random_policy(rng)),
        ("first", first_policy),
        ("heuristic", heuristic_policy),
        ("bfs", bfs_policy),  # factory: called per world to get a fresh closure
    ]
    rows: list[dict] = []
    for name, policy in policies:
        eps = _SINGLE_WORLD_EPISODES.get(name, 1)
        rows.append(run_policy(name, policy, [("live", world)], episodes=eps))
    return rows


def run_policy(name, policy, worlds, episodes: int) -> dict:
    """Run `policy` on each world for `episodes` episodes each.

    `policy` may be either a plain policy callable ``(world, ps, action_space)
    -> str`` or a *factory* ``(world) -> policy_callable`` (used for BFS, whose
    returned closure is stateful and must be regenerated per episode). A factory
    is detected by checking whether calling it with just the world returns a
    callable rather than a string.
    """
    results: list[EpisodeResult] = []
    for _label, world in worlds:
        ep = HeadlessEpisode(world)
        for _ in range(episodes):
            # Detect factory: bfs_policy(world) returns a callable, not a string.
            resolved = policy(world) if callable(policy) and _is_factory(policy, world) else policy
            results.append(ep.run(resolved))
    summary = _summarize(results)
    summary["policy"] = name
    return summary


def _is_factory(policy, world) -> bool:
    """True if `policy` is a world factory (takes one arg and returns a callable).

    Checks the function signature rather than calling it — avoids running BFS
    twice just to detect the type. A factory has exactly one required parameter.
    """
    import inspect
    try:
        sig = inspect.signature(policy)
        params = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        return len(params) == 1
    except (ValueError, TypeError):
        return False


def _fmt(v, spec=".2f"):
    return "  -  " if v is None else format(v, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless gameplay benchmark")
    parser.add_argument(
        "--glob", default=DEFAULT_GLOB, help="glob for saved world JSON"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="episodes per world (averages stochastic policies)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    worlds = load_worlds(args.glob)
    if not worlds:
        print(f"No worlds matched: {args.glob}")
        return

    rng = random.Random(args.seed)
    policies = [
        ("random", random_policy(rng)),
        ("first", first_policy),
        ("heuristic", heuristic_policy),
        ("bfs", bfs_policy),
    ]

    print(f"Loaded {len(worlds)} world(s); {args.episodes} episode(s) each\n")
    header = f"{'policy':<12} {'win%':>6} {'t2win':>7} {'t_all':>7} {'objs':>6}"
    print(header)
    print("-" * len(header))
    for name, policy in policies:
        s = run_policy(name, policy, worlds, args.episodes)
        print(
            f"{s['policy']:<12} "
            f"{s['win_rate'] * 100:>5.0f}% "
            f"{_fmt(s['mean_ticks_to_win']):>7} "
            f"{_fmt(s['mean_ticks_all']):>7} "
            f"{_fmt(s['mean_objects_resolved'], '.1f'):>6}"
        )


if __name__ == "__main__":
    main()
