"""LLM solver agent — drives a world to escape via the policy seam.

A *policy* is ``(world, ps, action_space) -> action_str`` (the same contract as the
deterministic ``heuristic_policy`` / ``bfs_policy`` in ``benchmark.policies``), so
this LLM solver runs under ``benchmark.engine.HeadlessEpisode`` exactly like them
and is directly comparable to the BFS optimum.

Partial observability: each tick the agent sees only the current room's visible
objects, what it carries, and the clues it has discovered — the same information the
``action_space`` already encodes. Movement is gated (a room's door must be unlocked
before its exit works), so the agent must solve a room before progressing.

CLI:
    python -m agents.solver_agent --world smoke_runs/<ts>/run_001.world.json
    python -m agents.solver_agent --world benchmark/worlds/world_001.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from agents.gameplay_node import (
    IDLE_ACTION,
    _format_goal_completion,
    _format_objects,
    _goal_completion_satisfied,
    _objects_in_room,
    _parse_json,
    _resolve_choice,
)
from config.settings import get_llm
from state import GameWorld, PartyState

SOLVER_SYSTEM = (
    "You are an expert escape-room solver. Each turn you see the current room, the "
    "objects you can see, what you carry, the clues you have learned, and a list of "
    "legal actions. Choose the SINGLE best action that makes progress toward "
    "escaping. Prefer examining un-inspected objects, applying a known clue/code, "
    "and using tools to unlock things over waiting. To move to another room you must "
    "FIRST unlock that room's locked door (its goal). "
    'Respond ONLY with JSON: {"action": "<one action copied EXACTLY from the list>"}.'
)


def _build_prompt(
    world: GameWorld, ps: PartyState, action_space: list[str], visible
) -> str:
    room = next((r for r in world.rooms if r.id == ps.current_room), None)
    win = world.win_condition
    win_str = (
        f"object '{win.object_id}' reaches state '{win.state}'"
        if win.object_id
        else "(unknown)"
    )
    if room and room.goal_completion is not None:
        room_status = (
            "DONE"
            if _goal_completion_satisfied(room.goal_completion, ps)
            else f"need {_format_goal_completion(room.goal_completion)}"
        )
    else:
        room_status = "(no completion condition)"
    exits = (
        "; ".join(f"{d} -> {n}" for d, n in room.adjacency.items())
        if room and room.adjacency
        else "(none)"
    )
    space_str = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_space))
    return (
        f"SCENARIO: {world.scenario}\n"
        f"OBJECTIVE: {world.objective}\n"
        f"WIN WHEN: {win_str}\n\n"
        f"TICK: {ps.tick + 1}\n"
        f"CURRENT ROOM: {ps.current_room}\n"
        f"{room.description if room else ''}\n"
        f"ROOM GOAL: {(room.goal if room else '') or '(explore)'}  [{room_status}]\n"
        f"EXITS: {exits}  (a locked door blocks the exit until this room's goal is met)\n\n"
        f"OBJECTS YOU CAN SEE:\n{_format_objects(visible, ps)}\n\n"
        f"INVENTORY: {', '.join(ps.inventory) if ps.inventory else '(empty)'}\n"
        f"CLUES KNOWN: {', '.join(ps.known_info) if ps.known_info else '(none)'}\n\n"
        f"LEGAL ACTIONS (choose exactly one, copy it verbatim):\n{space_str}\n"
    )


def llm_solver_policy(role: str = "solver"):
    """Return a policy ``(world, ps, action_space) -> action_str`` driven by an LLM.

    On any parse/LLM failure it falls back to the first legal action so the episode
    never crashes (it just makes a non-ideal move that tick).
    """
    llm = get_llm(role)

    def _policy(world: GameWorld, ps: PartyState, action_space: list[str]) -> str:
        if not action_space:
            return IDLE_ACTION
        visible = _objects_in_room(world, ps)
        prompt = _build_prompt(world, ps, action_space, visible)
        try:
            response = llm.invoke(
                [SystemMessage(content=SOLVER_SYSTEM), HumanMessage(content=prompt)]
            )
            data = _parse_json(response.content) or {}
            return _resolve_choice(str(data.get("action", "")).strip(), action_space)
        except Exception:
            return action_space[0]

    return _policy


REACT_SYSTEM = (
    "You are an expert escape-room solver using the ReAct method: you REASON about "
    "the situation, then ACT. Each turn you see the current room, visible objects, "
    "your inventory, clues learned, the legal actions, and a SCRATCHPAD of your "
    "previous thoughts, actions, and their observations. Use the scratchpad to avoid "
    "repeating dead-end moves and to chain steps: examine to find clues, apply "
    "clues/codes/tools to unlock things, and unlock a room's door before moving on. "
    'Respond ONLY with JSON: {"thought": "<brief reasoning>", "action": '
    '"<one action copied EXACTLY from the list>"}.'
)


def react_solver_policy(
    role: str = "solver", scratchpad_limit: int = 16, trace: list | None = None
):
    """ReAct policy: Thought -> Action each tick, carrying a running scratchpad of
    (Thought -> Action -> Observation) across ticks.

    The observation of the previous action is read back from ``ps.log`` (the engine
    appends each resolved action's note there), so the agent reasons over its own
    history. If ``trace`` is given, each tick's thought is appended to it for display.
    """
    llm = get_llm(role)
    scratchpad: list[str] = []
    state = {"seen_log": 0}

    def _policy(world: GameWorld, ps: PartyState, action_space: list[str]) -> str:
        if not action_space:
            return IDLE_ACTION
        # Ingest the observation(s) of prior action(s) from the engine log.
        while state["seen_log"] < len(ps.log):
            note = ps.log[state["seen_log"]].note
            if note:
                scratchpad.append(f"Observation: {note}")
            state["seen_log"] += 1

        visible = _objects_in_room(world, ps)
        pad = "\n".join(scratchpad[-scratchpad_limit:]) or "(nothing yet)"
        prompt = (
            _build_prompt(world, ps, action_space, visible)
            + f"\nSCRATCHPAD (your recent thought -> action -> observation):\n{pad}\n"
        )
        try:
            response = llm.invoke(
                [SystemMessage(content=REACT_SYSTEM), HumanMessage(content=prompt)]
            )
            data = _parse_json(response.content) or {}
            thought = str(data.get("thought", "")).strip()
            action = _resolve_choice(str(data.get("action", "")).strip(), action_space)
        except Exception:
            thought, action = "(parse error)", action_space[0]

        if thought:
            scratchpad.append(f"Thought: {thought}")
            if trace is not None:
                trace.append(f"t{ps.tick + 1} THINK: {thought}")
        scratchpad.append(f"Action: {action}")
        return action

    return _policy


# ---------------------------------------------------------------------------
# Objective function — score an episode against the BFS optimal
# ---------------------------------------------------------------------------

# Notes the engine emits when an action made no progress (no state change).
_NO_PROGRESS = ("no hidden info", "is locked", "unknown", "idle", "no direct route")


def objective(world: GameWorld, result, optimal_len: int | None = None) -> dict:
    """Score a solve episode against the BFS optimum.

    optimal = len(bfs_solution_path) — the minimum actions to escape (ground truth).
      reward = -1.0                  if the agent failed to escape
      reward = optimal / ticks       if it escaped  (1.0 = optimal, lower = wasteful)

    Returns the scalar reward plus its components so it can also feed RL / ranking.
    """
    from benchmark.policies import bfs_solution_path

    optimal = optimal_len if optimal_len is not None else len(bfs_solution_path(world))
    won = bool(result.victory)
    ticks = result.ticks
    wasted = sum(
        1 for line in result.history if any(k in line for k in _NO_PROGRESS)
    )
    if not won:
        reward = -1.0
        efficiency = 0.0
    else:
        efficiency = (optimal / ticks) if ticks else 0.0
        reward = efficiency
    return {
        "reward": round(reward, 3),
        "won": won,
        "ticks": ticks,
        "optimal": optimal,
        "efficiency": round(efficiency, 3),
        "wasted": wasted,
    }


def _load_world(path: Path) -> GameWorld:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return GameWorld.model_validate(raw.get("world", raw))


def solve_world(
    world: GameWorld, role: str = "solver", use_react: bool = False, trace: list | None = None
):
    """Run the LLM solver once. Returns (EpisodeResult, optimal_path_steps)."""
    from benchmark.engine import HeadlessEpisode
    from benchmark.policies import bfs_solution_path

    policy = (
        react_solver_policy(role, trace=trace) if use_react else llm_solver_policy(role)
    )
    result = HeadlessEpisode(world).run(policy, record_history=True)
    optimal = bfs_solution_path(world)
    return result, optimal


def benchmark(paths: list[Path], role: str = "solver", use_react: bool = False) -> None:
    """Run the LLM solver over many worlds; report solve-rate, ticks-vs-optimal, and
    the objective reward (optimal/ticks on a win, -1 on failure).

    Only worlds the BFS oracle can solve count toward the solve-rate denominator
    (an unsolvable world is not the agent's fault).
    """
    mode = "ReAct" if use_react else "direct"
    print(f"LLM solver benchmark — role={role}, mode={mode}, {len(paths)} world(s)\n")
    header = (
        f"{'world':<40} {'result':<8} {'ticks':>5} {'optimal':>7} "
        f"{'wasted':>6} {'reward':>7}"
    )
    print(header)
    print("-" * len(header))

    solved = solvable = 0
    rewards: list[float] = []
    for p in paths:
        world = _load_world(p)
        result, optimal = solve_world(world, role, use_react=use_react)
        score = objective(world, result, optimal_len=len(optimal))
        solvable += int(score["optimal"] > 0)
        solved += int(score["won"])
        if score["won"]:
            rewards.append(score["reward"])
        verdict = "ESCAPED" if score["won"] else "FAILED"
        label = str(Path(*p.parts[-2:]))  # filenames collide across run dirs
        print(
            f"{label:<40} {verdict:<8} {score['ticks']:>5} {score['optimal']:>7} "
            f"{score['wasted']:>6} {score['reward']:>7.2f}"
        )

    print("-" * len(header))
    rate = (solved / solvable * 100) if solvable else 0.0
    mean_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
    print(
        f"\nsolve-rate: {solved}/{solvable} solvable worlds = {rate:.0f}%"
        + (f"   |   mean reward (on solves): {mean_reward:.2f}" if rewards else "")
    )


def main() -> None:
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Run the LLM solver agent.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--world", help="solve a single saved world JSON (verbose trace)")
    src.add_argument(
        "--bench",
        help="glob of world JSONs to benchmark, e.g. 'smoke_runs/*/run_*.world.json'",
    )
    parser.add_argument(
        "--role",
        default="solver",
        help="LLM role from settings: solver (default), game_master, or player",
    )
    parser.add_argument(
        "--react",
        action="store_true",
        help="use the ReAct policy (Thought->Action with a running scratchpad)",
    )
    args = parser.parse_args()

    if args.bench:
        paths = [Path(p) for p in sorted(glob.glob(args.bench))]
        if not paths:
            parser.error(f"no worlds matched: {args.bench}")
        benchmark(paths, args.role, use_react=args.react)
        return

    world = _load_world(Path(args.world))
    print(f"Solving: {args.world}  (mode={'ReAct' if args.react else 'direct'})")
    print(f"  scenario : {world.scenario[:80]}...")
    print(f"  win when : {world.win_condition.object_id} -> {world.win_condition.state}")

    trace: list[str] = []
    result, optimal = solve_world(world, args.role, use_react=args.react, trace=trace)
    print(
        f"\n=== LLM solver: {'ESCAPED' if result.victory else 'FAILED'} "
        f"in {result.ticks} tick(s) ==="
    )
    # Interleave ReAct thoughts (t{N} THINK) with the action trace (t{N} ...).
    thought_by_tick = {ln.split(" ", 1)[0]: ln for ln in trace}
    for line in result.history:
        tk = line.split(" ", 1)[0]
        if tk in thought_by_tick:
            print(f"  {thought_by_tick[tk]}")
        print(f"  {line}")

    score = objective(world, result, optimal_len=len(optimal))
    print(f"\n=== BFS optimal ({len(optimal)} step(s)) — for comparison ===")
    for step in optimal:
        print(f"  {step}")
    print(
        f"\n=== objective ===\n"
        f"  reward={score['reward']}  won={score['won']}  ticks={score['ticks']}  "
        f"optimal={score['optimal']}  efficiency={score['efficiency']}  "
        f"wasted={score['wasted']}"
    )


if __name__ == "__main__":
    main()
