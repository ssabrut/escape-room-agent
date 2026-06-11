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
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.escape_rooms.nodes.gameplay import (
    IDLE_ACTION,
    _format_goal_completion,
    _format_objects,
    _goal_completion_satisfied,
    _objects_in_room,
    _parse_json,
    _resolve_choice,
)
from src.escape_rooms.utils.settings import get_llm
from src.escape_rooms.state import GameWorld, PartyState

REACT_SYSTEM = (
    "You are an expert escape-room solver using the ReAct method: you REASON about "
    "the situation, then ACT. Each turn you see the current room, visible objects, "
    "your inventory, clues learned, the legal actions, and a SCRATCHPAD of your "
    "previous thoughts, actions, and their observations. "
    "RULES: "
    "(1) Never examine an object listed in DEAD ENDS — it has no hidden info. "
    "(2) Check ROOM PROGRESS before deciding what to do — a DONE room needs no more work. "
    "(3) If a clue is listed in CLUES KNOWN with a '→ needed by' annotation, apply it "
    "to that object immediately rather than searching for where to use it. "
    "(4) Keep your thought to 1-2 sentences — state what you will do and why, then act. "
    "(5) Write a 'plan' field: one sentence summarising your current multi-step intent "
    "(e.g. 'take obsidian_lens then unlock recovery_door'). Update it whenever your "
    "intent changes. This plan persists across ticks so you stay on track. "
    "(6) Never attempt to go to a room listed in BLOCKED EXITS — the door is locked and "
    "you must solve this room's goal first before that exit will open. "
    'Respond ONLY with JSON: {"thought": "<1-2 sentence reasoning>", '
    '"plan": "<current intent>", "action": "<one action copied EXACTLY from the list>"}.'
)


def _room_progress(world: GameWorld, ps: PartyState) -> str:
    """One line per room: DONE or what remains, so the agent knows what's solved."""
    lines = []
    for r in world.rooms:
        if r.goal_completion is None:
            status = "DONE (no condition)"
        elif _goal_completion_satisfied(r.goal_completion, ps):
            status = "DONE"
        else:
            status = f"PENDING — need {_format_goal_completion(r.goal_completion)}"
        marker = " ← YOU ARE HERE" if r.id == ps.current_room else ""
        lines.append(f"  {r.id}: {status}{marker}")
    return "\n".join(lines)


def _annotated_clues(world: GameWorld, ps: PartyState) -> str:
    """List known clues annotated with which object needs each one."""
    if not ps.known_info:
        return "(none)"
    # Build a map: clue token -> list of object ids that consume it
    consumers: dict[str, list[str]] = {}
    for obj in world.objects:
        if obj.requires_code:
            consumers.setdefault(obj.requires_code, []).append(obj.id)
    for room in world.rooms:
        gc = room.goal_completion
        if gc and gc.type == "known_info" and gc.info:
            consumers.setdefault(gc.info, []).append(f"room:{room.id}")

    parts = []
    for clue in sorted(ps.known_info):
        needed_by = consumers.get(clue, [])
        if needed_by:
            parts.append(f"{clue}  (→ needed by: {', '.join(needed_by)})")
        else:
            parts.append(clue)
    return ", ".join(parts)


def _build_prompt(
    world: GameWorld, ps: PartyState, action_space: list[str], visible,
    dead_ends: set[str] | None = None,
    blocked_exits: set[str] | None = None,
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
    dead_str = ", ".join(sorted(dead_ends)) if dead_ends else "(none)"
    blocked_str = ", ".join(sorted(blocked_exits)) if blocked_exits else "(none)"
    return (
        f"SCENARIO: {world.scenario}\n"
        f"OBJECTIVE: {world.objective}\n"
        f"WIN WHEN: {win_str}\n\n"
        f"ROOM PROGRESS (all rooms):\n{_room_progress(world, ps)}\n\n"
        f"TICK: {ps.tick + 1}\n"
        f"CURRENT ROOM: {ps.current_room}\n"
        f"{room.description if room else ''}\n"
        f"ROOM GOAL: {(room.goal if room else '') or '(explore)'}  [{room_status}]\n"
        f"EXITS: {exits}  (a locked door blocks the exit until this room's goal is met)\n\n"
        f"OBJECTS YOU CAN SEE:\n{_format_objects(visible, ps)}\n\n"
        f"INVENTORY: {', '.join(ps.inventory) if ps.inventory else '(empty)'}\n"
        f"CLUES KNOWN: {_annotated_clues(world, ps)}\n"
        f"DEAD ENDS (no info here — do NOT examine these): {dead_str}\n"
        f"BLOCKED EXITS (door locked — do NOT go here until room goal is done): {blocked_str}\n\n"
        f"LEGAL ACTIONS (choose exactly one, copy it verbatim):\n{space_str}\n"
    )


def react_solver_policy(
    role: str = "solver", scratchpad_limit: int = 30, trace: list | None = None
):
    """ReAct policy: Thought -> Action each tick, carrying a running scratchpad of
    (Thought -> Action -> Observation) across ticks.

    Improvements over naive ReAct:
    - Dead-ends set: objects confirmed to hold no info never age out of context.
    - Sticky plan: the agent writes a short current plan each tick; it is prepended
      to the scratchpad so the model re-reads its own intent before reasoning.
    - scratchpad_limit raised to 30 so ~10 full ticks of history stay in context.
    - Annotated clues and room progress are injected by _build_prompt.
    """
    llm = get_llm(role)
    scratchpad: list[str] = []
    dead_ends: set[str] = set()
    blocked_exits: set[str] = set()
    current_plan: list[str] = ["(none yet)"]
    # Sliding window of recent actions for cycle detection (period 1–4, 3 full cycles).
    action_history: list[str] = []
    _CYCLE_REPS = 3   # how many full repetitions before we intervene
    _MAX_PERIOD = 4   # longest oscillation pattern to detect
    state = {"seen_log": 0}

    def _is_cycling(history: list[str]) -> bool:
        """Return True if the tail of history is a repeated pattern of length 1.._MAX_PERIOD."""
        for period in range(1, _MAX_PERIOD + 1):
            needed = period * _CYCLE_REPS
            if len(history) < needed:
                continue
            tail = history[-needed:]
            pattern = tail[:period]
            if tail == pattern * _CYCLE_REPS:
                return True
        return False

    def _policy(world: GameWorld, ps: PartyState, action_space: list[str]) -> str:
        if not action_space:
            return IDLE_ACTION

        # Ingest observations from the engine log; record dead-ends and blocked exits.
        while state["seen_log"] < len(ps.log):
            entry = ps.log[state["seen_log"]]
            note = entry.note
            if note:
                scratchpad.append(f"Observation: {note}")
                parts = entry.action.split()
                if "no hidden info" in note and len(parts) >= 2:
                    dead_ends.add(parts[1])
                # "is locked" means a go <room> was blocked by an unsolved door.
                if "is locked" in note and len(parts) >= 2 and parts[0] == "go":
                    blocked_exits.add(parts[1])
            state["seen_log"] += 1

        visible = _objects_in_room(world, ps)
        # Prepend the sticky plan so it is always in context regardless of scratchpad age.
        pad_lines = [f"Current plan: {current_plan[0]}"] + scratchpad[-scratchpad_limit:]
        pad = "\n".join(pad_lines)
        prompt = (
            _build_prompt(world, ps, action_space, visible, dead_ends, blocked_exits)
            + f"\nSCRATCHPAD (plan + recent thought → action → observation):\n{pad}\n"
        )
        try:
            response = llm.invoke(
                [SystemMessage(content=REACT_SYSTEM), HumanMessage(content=prompt)]
            )
            data = _parse_json(response.content) or {}
            thought = str(data.get("thought", "")).strip()
            plan = str(data.get("plan", "")).strip()
            action = _resolve_choice(str(data.get("action", "")).strip(), action_space)
        except Exception:
            thought, plan, action = "(parse error)", "", action_space[0]

        # Cycle guard: detect any repeating pattern of length 1-_MAX_PERIOD in the
        # recent action history and force a different action to break out of the loop.
        action_history.append(action)
        if _is_cycling(action_history):
            cycle_set = set(action_history[-(2 * _MAX_PERIOD):])
            alternatives = [a for a in action_space if a not in cycle_set]
            if not alternatives:
                alternatives = [a for a in action_space if a != action]
            if alternatives:
                forced = alternatives[0]
                scratchpad.append(
                    f"[override] cycle detected {action_history[-(2*_MAX_PERIOD):]} — forcing '{forced}'"
                )
                action = forced
                action_history[-1] = forced

        if plan:
            current_plan[0] = plan
        if thought:
            scratchpad.append(f"Thought: {thought}")
            if trace is not None:
                trace.append(f"t{ps.tick} THINK: {thought}")
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
    world: GameWorld, role: str = "solver", trace: list | None = None,
    strategy: str = "cognitive", debug_log: list[dict] | None = None,
    on_tick: Callable[[dict], None] | None = None,
):
    """Run the LLM solver once. Returns (EpisodeResult, optimal_path_steps).

    ``strategy="cognitive"`` (default) uses the TeamCognition + ActionPlanner
    policy (see ``agents.multi_solver``). ``strategy="react"`` uses the
    single-pass ReAct policy.

    ``debug_log``, if given, is only populated by the ``"cognitive"`` strategy:
    one dict per tick with the LLM's thought/plan, the planner's ranked
    candidates, any gate overrides, and the final action.

    ``on_tick``, if given, is also only used by the ``"cognitive"`` strategy: it's
    called with that same per-tick dict as soon as it's produced, e.g. to stream
    live progress to a client.
    """
    from benchmark.engine import HeadlessEpisode
    from benchmark.policies import bfs_solution_path

    if strategy == "cognitive":
        from src.escape_rooms.agents.multi_solver import cognitive_solver_policy
        policy = cognitive_solver_policy(role, trace=trace, debug_log=debug_log, on_tick=on_tick)
    elif strategy == "react":
        policy = react_solver_policy(role, trace=trace)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Expected 'react' or 'cognitive'.")

    result = HeadlessEpisode(world).run(policy, record_history=True)
    optimal = bfs_solution_path(world)
    return result, optimal


def benchmark(paths: list[Path], role: str = "solver", strategy: str = "cognitive") -> None:
    """Run the LLM solver over many worlds; report solve-rate, ticks-vs-optimal, and
    the objective reward (optimal/ticks on a win, -1 on failure).

    Only worlds the BFS oracle can solve count toward the solve-rate denominator
    (an unsolvable world is not the agent's fault).
    """
    print(f"LLM solver benchmark — role={role}, strategy={strategy}, {len(paths)} world(s)\n")
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
        result, optimal = solve_world(world, role, strategy=strategy)
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
        "--strategy",
        default="cognitive",
        choices=["react", "cognitive"],
        help="solver strategy: cognitive (default, TeamCognition + ActionPlanner) "
             "or react (single-pass ReAct)",
    )
    args = parser.parse_args()

    if args.bench:
        paths = [Path(p) for p in sorted(glob.glob(args.bench))]
        if not paths:
            parser.error(f"no worlds matched: {args.bench}")
        benchmark(paths, args.role, args.strategy)
        return

    world = _load_world(Path(args.world))
    print(f"Solving: {args.world}  (mode={args.strategy})")
    print(f"  scenario : {world.scenario[:80]}...")
    print(f"  win when : {world.win_condition.object_id} -> {world.win_condition.state}")

    trace: list[str] = []
    result, optimal = solve_world(world, args.role, trace=trace, strategy=args.strategy)
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
