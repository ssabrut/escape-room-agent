"""Main entrypoint — runs the escape room pipeline.

Modes (--mode):
  generate  Run the game master node only — generate and display the world, skip
            characters and gameplay. Fast for testing world generation in isolation.
  full      Run the complete pipeline: world generation, character creation,
            player agent selection, and gameplay (default).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from graph import build_graph

# Cache compiled graphs by player count so repeated runs (e.g. --smoke N) don't
# rebuild the same graph each iteration.
_GRAPH_CACHE: dict[int, object] = {}


def _get_graph(num_players: int = 1):
    if num_players not in _GRAPH_CACHE:
        _GRAPH_CACHE[num_players] = build_graph(num_players)
    return _GRAPH_CACHE[num_players]
from state import GameState, GameWorld
from visualization import render_room_layout

MODE_GENERATE = "generate"
MODE_FULL = "full"

THEMES = [
    "Haunted House",
    "Murder Mystery",
    "Prison Break",
    "Pirate Adventure",
    "Bank Robbery",
    "Cosmic Crisis",
    "Treasure Hunt",
    "Zombie Apocalypse",
    "Secret Agents and Spies",
    "Horror",
]


def _pick_theme() -> str:
    """Prompt the user to choose a theme with arrow-key navigation."""
    import questionary

    choice = questionary.select(
        "Choose your escape room theme:",
        choices=THEMES,
        use_shortcuts=False,
    ).ask()
    if choice is None:
        print("No theme selected — exiting.")
        sys.exit(0)
    return choice


SMOKE_DIR = Path("smoke_runs")
LOG_DIR = Path("logs")
NODE_NAMES = (
    "world_builder",
    "puzzle_builder",
    "character_master",
    "player_agent_1",
    "gameplay",
)


def _jsonable(value):
    if isinstance(value, BaseModel):
        # exclude_none drops always-null sibling fields (e.g. Prerequisite's
        # unused type-fields, WorldObject's irrelevant precondition slots) so the
        # logged JSON shows only the keys that carry meaning for each record.
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, BaseMessage):
        return {"type": value.type, "content": value.content}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_world_json(world, path: Path) -> str:
    """Serialize the final assembled GameWorld to JSON. Returns a status note."""
    if world is None:
        return ""
    path.write_text(
        json.dumps({"world": _jsonable(world)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return f" + {path.name}"


def _write_node_log(node: str, update: dict, root: Path = LOG_DIR) -> Path:
    node_dir = root / node
    node_dir.mkdir(parents=True, exist_ok=True)
    messages = update.get("messages") or []
    raw = "\n\n---\n\n".join(
        m.content for m in messages if isinstance(m, BaseMessage) and m.content
    )
    (node_dir / "raw.txt").write_text(raw, encoding="utf-8")

    attempt_log = update.get("_attempt_log")
    if attempt_log:
        (node_dir / "attempts.json").write_text(
            json.dumps(attempt_log, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    parsed = {
        k: _jsonable(v)
        for k, v in update.items()
        if k not in ("messages", "_attempt_log")
    }
    (node_dir / "output.json").write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return node_dir


def _render_characters(characters: list) -> None:
    if not characters:
        print("  [No characters could be generated]\n")
        return

    print("\n" + "=" * 94)
    print(" CHOOSE YOUR CHARACTER")
    print("=" * 94 + "\n")

    for i, char in enumerate(characters, 1):
        print(f"  [{i}] {char.name}  —  {char.role}")
        print(f"       {char.backstory}")
        print()


def _render(result: dict) -> None:
    world = result.get("world")
    rooms = world.rooms if world else []
    objects = world.objects if world else []

    print("\n" + "=" * 94)
    print(" ESCAPE ROOM MAP")
    print("=" * 94 + "\n")

    if rooms:
        render_room_layout(rooms, objects)
    else:
        print("  [No room layout could be parsed from the LLM response]\n")

    if world:
        print("\n" + "=" * 94)
        print(" SCENARIO")
        print("=" * 94 + "\n")
        print(f"  {world.scenario}\n")
        print(f"  Objective: {world.objective}")
        win = world.win_condition
        print(f"  Win when : {win.object_id} → {win.state}\n")
        if world.solution_path:
            print("  Solution path:")
            for step in world.solution_path:
                print(f"    {step}")
            print()

    characters = result.get("characters", [])
    _render_characters(characters)

    party = result.get("party", [])
    if party:
        print("\n" + "=" * 94)
        print(" PARTY SELECTIONS")
        print("=" * 94 + "\n")
        for member in party:
            print(f"  {member.agent_id}")
            print(f"    Chose    : {member.character.name} — {member.character.role}")
            print(f"    Reasoning: {member.reasoning}")
            print()

    party_state = result.get("party_state")
    if party_state:
        print("\n" + "=" * 94)
        print(" FINAL RESULT")
        print("=" * 94 + "\n")
        outcome = (
            "VICTORY"
            if party_state.victory
            else f"ENDED (final room: {party_state.current_room})"
        )
        inv = ", ".join(party_state.inventory) if party_state.inventory else "(empty)"
        known = (
            ", ".join(party_state.known_info) if party_state.known_info else "(none)"
        )
        print(f"  Result    : {outcome}")
        print(f"  Ticks used: {party_state.tick}")
        print(f"  Inventory : {inv}")
        print(f"  Known     : {known}")
        print(f"  Visited   : {', '.join(party_state.visited)}")
        print()


def _format_timing_table(node_times: dict[str, float]) -> list[str]:
    """Return lines for a per-node elapsed-time table."""
    if not node_times:
        return []
    lines = ["", "-- Node timing " + "-" * 79]
    total = sum(node_times.values())
    for node, elapsed in node_times.items():
        pct = elapsed / total * 100 if total else 0
        bar = "#" * int(pct / 5)
        lines.append(f"  {node:<22} {elapsed:6.2f}s  {pct:5.1f}%  {bar}")
    lines.append(f"  {'TOTAL':<22} {total:6.2f}s")
    return lines


def _print_timing_table(node_times: dict[str, float]) -> None:
    """Print the per-node timing table to stdout."""
    for line in _format_timing_table(node_times):
        print(line)
    print()


def _write_run_summary(
    result: dict, path: Path, node_times: dict[str, float] | None = None
) -> None:
    """Write a clean, human-readable summary of the final output from each node."""
    lines: list[str] = []
    W = 94

    def _h1(title: str) -> None:
        lines.append("=" * W)
        lines.append(f"  {title}")
        lines.append("=" * W)

    def _h2(title: str) -> None:
        lines.append("")
        lines.append(f"-- {title} " + "-" * max(0, W - len(title) - 4))

    # --- world_builder ---
    world = result.get("world")
    _h1("WORLD BUILDER")
    if world:
        lines.append(f"  Scenario  : {world.scenario}")
        lines.append(f"  Objective : {world.objective}")
        win = world.win_condition
        lines.append(f"  Win when  : {win.object_id} → {win.state}")
        _h2("Rooms")
        for r in world.rooms:
            lines.append(f"  [{r.id}]  {r.description}")
            lines.append(f"    Goal       : {r.goal}")
            gc = (
                r.goal_completion.model_dump(exclude_none=True)
                if r.goal_completion
                else "(none)"
            )
            lines.append(f"    Completion : {gc}")
            lines.append(f"    Adjacency  : {r.adjacency}")
    else:
        lines.append("  (no world generated)")

    # --- puzzle_builder ---
    lines.append("")
    _h1("PUZZLE BUILDER")
    if world and world.objects:
        _h2("Objects")
        for o in world.objects:
            parts = [f"  [{o.id}] @ {o.location} | state={o.state}"]
            if o.requires_tool:
                parts.append(f"requires_tool={o.requires_tool}")
            if o.requires_code:
                parts.append(f"requires_code={o.requires_code!r}")
            if o.requires_liquid:
                parts.append(f"requires_liquid={o.requires_liquid}")
            if o.requires_power:
                parts.append(f"requires_power={o.requires_power}")
            if o.fuses:
                parts.append(f"fuses={list(o.fuses.keys())}")
            if o.contains_info:
                parts.append(f"contains_info={o.contains_info!r}")
            if o.scenic:
                parts.append("scenic")
            lines.append(" | ".join(parts))
        if world.solution_path:
            _h2("Solution path")
            for step in world.solution_path:
                lines.append(f"  {step}")
    else:
        lines.append("  (no objects generated)")

    # --- character_master ---
    characters = result.get("characters", [])
    lines.append("")
    _h1("CHARACTER MASTER")
    if characters:
        for c in characters:
            lines.append(f"  {c.name} — {c.role}")
            lines.append(f"    {c.backstory}")
    else:
        lines.append("  (no characters generated)")

    # --- player agents ---
    party = result.get("party", [])
    lines.append("")
    _h1("PLAYER AGENTS")
    if party:
        for m in party:
            lines.append(f"  {m.agent_id} → {m.character.name} ({m.character.role})")
            lines.append(f"    Reasoning: {m.reasoning}")
    else:
        lines.append("  (no party formed)")

    # --- gameplay / final result ---
    party_state = result.get("party_state")
    lines.append("")
    _h1("FINAL RESULT")
    if party_state:
        outcome = (
            "VICTORY"
            if party_state.victory
            else f"ENDED (final room: {party_state.current_room})"
        )
        lines.append(f"  Result    : {outcome}")
        lines.append(f"  Ticks used: {party_state.tick}")
        lines.append(f"  Inventory : {', '.join(party_state.inventory) or '(empty)'}")
        lines.append(f"  Known     : {', '.join(party_state.known_info) or '(none)'}")
        lines.append(f"  Visited   : {', '.join(sorted(party_state.visited))}")
    else:
        lines.append("  (gameplay did not run)")

    if node_times:
        lines.extend(_format_timing_table(node_times))

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _merge_update(result: dict, update: dict) -> None:
    for key, value in update.items():
        if key == "messages":
            result.setdefault("messages", []).extend(value or [])
        elif key != "_attempt_log":
            result[key] = value


def _report_eval_failures(eval_failures: list[str]) -> None:
    """Print a final summary line for per-node eval tracing."""
    if not eval_failures:
        return
    print(
        f"\n  [eval] {len(eval_failures)} node(s) FAILED their eval: "
        f"{', '.join(eval_failures)}"
    )


def run(
    mode: str = MODE_FULL,
    log_nodes: list[str] | None = None,
    theme: str = "",
    trace_eval: bool = False,
    num_players: int = 1,
) -> None:
    log_nodes = log_nodes or []
    if not theme:
        theme = _pick_theme()

    node_times: dict[str, float] = {}
    log_set = set(log_nodes)
    eval_failures: list[str] = []
    eval_root = LOG_DIR if trace_eval else None

    if mode == MODE_GENERATE:
        from agents.game_master import world_builder_node
        from agents.puzzle_builder_node import puzzle_builder_node
        from state import GameState as _GS

        state = _GS(theme=theme)
        t0 = time.perf_counter()
        wb_update = world_builder_node(state)
        node_times["world_builder"] = time.perf_counter() - t0
        if "world_builder" in log_set:
            node_dir = _write_node_log("world_builder", wb_update)
            print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")
        if trace_eval and not _eval_node("world_builder", wb_update, eval_root):
            eval_failures.append("world_builder")

        state = state.model_copy(update={"world": wb_update.get("world")})
        t0 = time.perf_counter()
        pb_update = puzzle_builder_node(state)
        node_times["puzzle_builder"] = time.perf_counter() - t0
        if "puzzle_builder" in log_set:
            node_dir = _write_node_log("puzzle_builder", pb_update)
            print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

        result = {**wb_update, **pb_update}
        if trace_eval and not _eval_node("puzzle_builder", result, eval_root):
            eval_failures.append("puzzle_builder")

        _render(result)
        _print_timing_table(node_times)
        _report_eval_failures(eval_failures)
        return

    result: dict = {}
    game_graph = _get_graph(num_players)
    _step_start = time.perf_counter()
    for step in game_graph.stream(GameState(theme=theme), stream_mode="updates"):
        _step_end = time.perf_counter()
        for node, update in step.items():
            node_times[node] = node_times.get(node, 0.0) + (_step_end - _step_start)
            _merge_update(result, update)
            if node in log_set:
                node_dir = _write_node_log(node, update)
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")
            # Trace per node against the merged result (so puzzle_builder sees
            # the assembled world, not just its own partial update).
            if trace_eval and not _eval_node(node, result, eval_root):
                eval_failures.append(node)
        _step_start = time.perf_counter()
    _render(result)
    _print_timing_table(node_times)
    _report_eval_failures(eval_failures)


def _run_once_captured(
    log_nodes: list[str] | None,
    log_root: Path | None,
    mode: str = MODE_FULL,
    theme: str = "pirate",
    num_players: int = 1,
) -> tuple[str, dict, dict[str, float]]:
    """Run the pipeline once with stdout captured.

    If log_nodes is set, write per-node logs under log_root. Returns the captured
    text, the merged result dict, and a per-node elapsed-time mapping.
    """
    log_set = set(log_nodes or [])
    node_times: dict[str, float] = {}
    buf = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = buf
    try:
        if mode == MODE_GENERATE:
            # Time each sub-node manually for generate-only mode.
            from agents.game_master import world_builder_node
            from agents.puzzle_builder_node import puzzle_builder_node
            from state import GameState as _GS

            state = _GS(theme=theme)

            t0 = time.perf_counter()
            wb_update = world_builder_node(state)
            node_times["world_builder"] = time.perf_counter() - t0
            if log_set and "world_builder" in log_set:
                node_dir = _write_node_log(
                    "world_builder", wb_update, root=log_root or LOG_DIR
                )
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

            state = state.model_copy(update={"world": wb_update.get("world")})

            t0 = time.perf_counter()
            pb_update = puzzle_builder_node(state)
            node_times["puzzle_builder"] = time.perf_counter() - t0
            if log_set and "puzzle_builder" in log_set:
                node_dir = _write_node_log(
                    "puzzle_builder", pb_update, root=log_root or LOG_DIR
                )
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

            result = {**wb_update, **pb_update}
        else:
            result = {}
            game_graph = _get_graph(num_players)
            _step_start = time.perf_counter()
            for step in game_graph.stream(GameState(theme=theme), stream_mode="updates"):
                _step_end = time.perf_counter()
                for node, update in step.items():
                    node_times[node] = node_times.get(node, 0.0) + (
                        _step_end - _step_start
                    )
                    _merge_update(result, update)
                    if node in log_set and log_root is not None:
                        _write_node_log(node, update, root=log_root)
                _step_start = time.perf_counter()
        _render(result)
    finally:
        sys.stdout = orig_stdout
    return buf.getvalue(), result, node_times


def _write_policy_benchmark(world, out_path: Path) -> None:
    """Run the LLM-free policy benchmark for `world` and write it to `out_path`.

    Diagnostic only — used by the --struct-eval flow so the per-node trace also
    captures how the baseline policies fare on the assembled world. Failures are
    swallowed so a benchmark hiccup never aborts the eval trace.
    """
    if world is None or not getattr(world, "rooms", None):
        return
    try:
        from benchmark.run import compute_policy_benchmark

        rows = compute_policy_benchmark(world)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"      [eval] wrote {out_path}")
    except Exception as exc:
        print(f"      [eval] policy benchmark skipped: {exc}")


def _eval_node(
    node: str, result: dict, out_root: Path | None = None
) -> bool:
    """Run the structural eval that belongs to `node` and print a PASS/FAIL trace.

    Returns True if the node's eval passed (or had no eval to run). This is a
    fast, deterministic check per pipeline stage.
    """
    world = result.get("world")
    issues: list[str] = []

    if node == "world_builder":
        if world is None:
            issues.append("no world produced")
        else:
            from agents.game_master import _eval_world_structure, MAX_ROOMS
            from benchmark.policies import check_solvable

            issues += _eval_world_structure(world, len(world.rooms) or MAX_ROOMS)
            # Solvability is only meaningful once objects exist; at this stage
            # objects is usually empty, so skip the object-graph walk.
            if world.objects:
                issues += check_solvable(world).issues
    elif node == "puzzle_builder":
        if world is None:
            issues.append("no world to check (puzzle_builder produced nothing)")
        else:
            from benchmark.policies import check_solvable, oracle_solve

            issues += check_solvable(world).issues
            # Dynamic verdict: actually play start->escape (BFS-first, complete
            # within its budget) — catches solvability gaps the static walk misses.
            result = oracle_solve(world)
            if not result.victory:
                issues.append(
                    f"oracle failed to win in {result.ticks} tick(s) "
                    f"(last room: {result.last_room}, "
                    f"win object state: {result.win_object_state!r})"
                )
    else:
        # No structural eval defined for this node (e.g. character_master,
        # player agents, gameplay). Nothing to trace.
        return True

    passed = not issues
    status = "PASS" if passed else f"FAIL ({len(issues)} issue(s))"
    print(f"  [eval:{node}] {status}")
    for i, msg in enumerate(issues, 1):
        print(f"      {i}. {msg}")

    if out_root is not None:
        out_path = out_root / node / "eval.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "node": node,
                    "status": "PASS" if passed else "FAIL",
                    "passed": passed,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "issues": issues,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"      [eval] wrote {out_path}")

        # The policy benchmark needs the fully-assembled world (objects exist
        # only once puzzle_builder has run), so emit it at that stage.
        if node == "puzzle_builder" and world is not None:
            _write_policy_benchmark(world, out_root / node / "benchmark.json")

    return passed


def _load_game_state_from_json(path: Path, theme: str = "") -> "GameState | None":
    """Load a saved GameState from a node output.json (or a bare world.json).

    Accepts either a full GameState dump (keys like world/characters/party) or a
    bare world JSON (wrapped or unwrapped in a 'world' key). Returns a GameState
    seeded with whatever upstream fields the file carries, so a single node can
    run against it.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [node] could not read {path}: {exc}")
        return None
    if not isinstance(raw, dict):
        print(f"  [node] {path} is not a JSON object")
        return None

    # A GameState dump has top-level node fields; a bare world has rooms/scenario.
    is_state = any(k in raw for k in ("characters", "party", "party_state")) or (
        "world" in raw and "rooms" not in raw
    )
    fields: dict = {}
    if is_state:
        fields = {k: v for k, v in raw.items() if k != "messages"}
    else:
        fields = {"world": raw.get("world", raw)}
    if theme:
        fields["theme"] = theme

    try:
        return GameState.model_validate(fields)
    except Exception as exc:
        print(f"  [node] could not parse GameState from {path}: {exc}")
        return None


# Maps a node name to (entry function, required GameState fields it reads).
# world_builder needs nothing but a theme, so its requirements are empty.
def _node_registry() -> dict:
    from agents.character_master_node import character_master_node
    from agents.game_master import world_builder_node
    from agents.gameplay_node import gameplay_node
    from agents.player_agent_node import player_agent_1_node
    from agents.puzzle_builder_node import puzzle_builder_node

    return {
        "world_builder": (world_builder_node, ()),
        "puzzle_builder": (puzzle_builder_node, ("world",)),
        "character_master": (character_master_node, ("world",)),
        "player_agent_1": (player_agent_1_node, ("characters",)),
        "gameplay": (gameplay_node, ("world", "party")),
    }


def run_node(
    node: str,
    from_path: str | None = None,
    theme: str = "",
    trace_eval: bool = False,
) -> None:
    """Run a single pipeline node independently against saved upstream state.

    world_builder runs from a theme alone; every other node loads its inputs
    from --from PATH (a prior node's output.json or a world.json). The node's
    output is written to logs/<node>/output.json.
    """
    registry = _node_registry()
    if node not in registry:
        print(f"  [node] unknown node '{node}'. Choices: {', '.join(registry)}")
        sys.exit(1)
    fn, required = registry[node]

    # Build the seed state.
    if node == "world_builder":
        if not theme:
            theme = _pick_theme()
        state = GameState(theme=theme)
    else:
        if not from_path:
            print(
                f"  [node] '{node}' needs upstream state — pass --from PATH "
                f"(a saved output.json carrying: {', '.join(required)})"
            )
            sys.exit(1)
        loaded = _load_game_state_from_json(Path(from_path), theme=theme)
        if loaded is None:
            sys.exit(1)
        state = loaded
        # Verify the inputs this node depends on are actually present.
        missing = [
            f for f in required if not getattr(state, f, None)
        ]
        if missing:
            print(
                f"  [node] '{node}' is missing required input(s) {missing} in "
                f"{from_path}. Run the upstream node first."
            )
            sys.exit(1)

    print(f"  [node] running '{node}' independently...")
    t0 = time.perf_counter()
    update = fn(state)
    elapsed = time.perf_counter() - t0

    node_dir = _write_node_log(node, update)
    print(f"  [node] wrote {node_dir}/output.json + {node_dir}/raw.txt")
    _print_timing_table({node: elapsed})

    if trace_eval:
        # Eval against the merged state so e.g. puzzle_builder sees the world.
        merged = {**{k: getattr(state, k) for k in ("world",)}, **update}
        if not _eval_node(node, merged, LOG_DIR):
            _report_eval_failures([node])


def _write_benchmark(world, path: Path) -> str:
    """Write the LLM-free policy benchmark for `world` to `path` as JSON.

    Returns a short suffix for the per-run console line (e.g. " + benchmark.json")
    or "" if there was no usable world to benchmark. Diagnostic only, so failures
    are swallowed rather than aborting the smoke run.
    """
    if world is None or not getattr(world, "rooms", None):
        return ""
    try:
        from benchmark.run import compute_policy_benchmark

        rows = compute_policy_benchmark(world)
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        return ""
    return f" + {path.name}"


def _write_bfs_path(world, path: Path) -> str:
    """Solve `world` with the no-LLM oracle and write the winning action path.

    Uses `bfs_policy` — an exhaustive breadth-first search that returns the
    SHORTEST winning action sequence (or the greedy heuristic as a fallback when
    no win is found within the search budget). The path is replayed through the
    headless engine with history recording and written to `path`. Diagnostic
    only, so failures are swallowed rather than aborting the smoke run.

    Returns a short suffix for the per-run console line, or "" on no usable world.
    """
    if world is None or not getattr(world, "rooms", None):
        return ""
    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import bfs_policy, heuristic_policy

        policy = bfs_policy(world)
        is_bfs = policy is not heuristic_policy  # bfs falls back to heuristic on miss
        result = HeadlessEpisode(world).run(policy, record_history=True)

        kind = "BFS shortest" if is_bfs else "heuristic best-effort (BFS found no path in budget)"
        lines = [
            f"solvable (oracle won): {result.victory}",
            f"path source: {kind}",
            f"ticks: {result.ticks}    chain_depth: {result.chain_depth}",
            "",
            "winning action path:" if result.victory else "action trace (no win reached):",
            *(f"  {h}" for h in result.history),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        return ""
    return f" + {path.name}"


def smoke(
    n: int,
    log_nodes: list[str] | None = None,
    mode: str = MODE_FULL,
    trace_eval: bool = False,
    theme: str = "pirate",
    num_players: int = 1,
) -> None:
    SMOKE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_DIR / timestamp
    run_dir.mkdir()

    print(f"Smoke test ({mode}, theme: {theme}): {n} run(s) → {run_dir}/")
    if log_nodes:
        print(f"  [log] per-run node logs under {run_dir}/run_<NNN>_logs/")

    errors = 0
    eval_records: list[dict] = []  # per-run structural eval outcomes for the roll-up
    for i in range(1, n + 1):
        print(f"  [{i}/{n}] generating...", end=" ", flush=True)
        try:
            log_root = run_dir / f"run_{i:03d}_logs" if log_nodes else None
            out_file = run_dir / f"run_{i:03d}.txt"
            output, result, node_times = _run_once_captured(
                log_nodes, log_root, mode=mode, theme=theme, num_players=num_players
            )
            _write_run_summary(result, out_file, node_times=node_times)
            (run_dir / f"run_{i:03d}_stdout.txt").write_text(output, encoding="utf-8")
            world_note = _write_world_json(
                result.get("world"), run_dir / f"run_{i:03d}.world.json"
            )
            bench_note = _write_benchmark(
                result.get("world"), run_dir / f"run_{i:03d}.benchmark.json"
            )
            bfs_note = _write_bfs_path(
                result.get("world"), run_dir / f"run_{i:03d}.bfs_path.txt"
            )
            total_elapsed = sum(node_times.values())
            print(
                f"done in {total_elapsed:.1f}s → "
                f"{out_file.name}{world_note}{bench_note}{bfs_note}"
            )
            _print_timing_table(node_times)
            if trace_eval:
                # Structural per-node eval (PASS/FAIL + status/timestamp), written
                # to each node's eval.json under this run's log dir.
                eval_root = run_dir / f"run_{i:03d}_logs"
                nodes = ("world_builder", "puzzle_builder")
                per_node = {node: _eval_node(node, result, eval_root) for node in nodes}
                eval_failures = [node for node, ok in per_node.items() if not ok]
                _report_eval_failures(eval_failures)
                eval_records.append(
                    {
                        "run": i,
                        "passed": not eval_failures,
                        "nodes": {
                            node: ("PASS" if ok else "FAIL")
                            for node, ok in per_node.items()
                        },
                        "failures": eval_failures,
                    }
                )
        except Exception as e:
            errors += 1
            err_file = run_dir / f"run_{i:03d}.error.txt"
            err_file.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"ERROR → {err_file.name} ({e})")

    if eval_records:
        passed = sum(1 for r in eval_records if r["passed"])
        per_node_pass: dict[str, int] = {}
        for r in eval_records:
            for node, status in r["nodes"].items():
                per_node_pass[node] = per_node_pass.get(node, 0) + (status == "PASS")
        roll_up = {
            "runs_evaluated": len(eval_records),
            "passed": passed,
            "failed": len(eval_records) - passed,
            "per_node_pass": per_node_pass,
            "records": eval_records,
        }
        summary_path = run_dir / "trace_eval_summary.json"
        summary_path.write_text(
            json.dumps(roll_up, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"\n  [trace-eval] {passed}/{len(eval_records)} run(s) passed "
            f"→ {summary_path.name}"
        )

    summary = f"\nAll runs saved to {run_dir}/"
    if errors:
        summary += f" ({errors}/{n} failed)"
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Escape room pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --mode generate          # world generation only\n"
            "  python main.py --mode full              # full pipeline (default)\n"
            "  python main.py --mode generate --smoke 5\n"
            "  python main.py --mode generate --log world_builder --log puzzle_builder\n"
            "  python main.py --mode generate --struct-eval  # fast per-node structural eval\n"
            "  python main.py --node world_builder --theme pirate   # run one node\n"
            "  python main.py --node puzzle_builder --from logs/world_builder/output.json\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_GENERATE, MODE_FULL],
        default=MODE_FULL,
        help=(
            f"'{MODE_GENERATE}': run world_builder + puzzle_builder only; "
            f"'{MODE_FULL}': run the complete pipeline including characters and gameplay "
            f"(default: {MODE_FULL})"
        ),
    )
    parser.add_argument(
        "--smoke",
        metavar="N",
        type=int,
        help="Run N times and save each output to smoke_runs/<timestamp>/",
    )
    parser.add_argument(
        "--log",
        metavar="NODE",
        action="append",
        choices=(*NODE_NAMES, "all"),
        help=(
            "Log a node's parsed output and raw LLM response. Repeatable. "
            "Use 'all' to log every node. "
            "Choices: " + ", ".join((*NODE_NAMES, "all"))
        ),
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Hard mode: generate multi-room worlds with deep puzzle chains, "
        "validated solvable before play (default: 2-room mode).",
    )
    parser.add_argument(
        "--rooms",
        type=int,
        metavar="N",
        help="Number of rooms in hard mode (implies --hard). Default: 4.",
    )
    parser.add_argument(
        "--struct-eval",
        "--trace-eval",  # deprecated alias
        dest="trace_eval",
        action="store_true",
        help="STRUCTURAL evaluator (fast, deterministic, no LLM). Runs inline as "
        "each node finishes: world_builder is checked for room count/adjacency/goal "
        "coherence; puzzle_builder is checked for SOLVABILITY (object-graph walk). "
        "(alias: --trace-eval)",
    )
    parser.add_argument(
        "--theme",
        default="",
        metavar="THEME",
        help="Theme for generation (skips the interactive picker). "
        "Used by --node world_builder and the normal pipeline.",
    )
    parser.add_argument(
        "--node",
        choices=list(NODE_NAMES),
        metavar="NAME",
        help="Run a SINGLE node independently against saved upstream state "
        "(see --from). world_builder runs from --theme alone. Output is written "
        "to logs/<node>/output.json. Choices: " + ", ".join(NODE_NAMES),
    )
    parser.add_argument(
        "--player",
        type=int,
        default=1,
        metavar="N",
        help="Number of player agents in the party (default: 1). Each picks one "
        "distinct character.",
    )
    parser.add_argument(
        "--from",
        dest="from_path",
        metavar="PATH",
        help="With --node: path to a saved output.json (or world.json) carrying "
        "the upstream state the node needs as input.",
    )
    args = parser.parse_args()

    if args.player < 1:
        parser.error("--player requires a positive integer")

    # Translate hard-mode flags into env vars BEFORE the graph runs; Settings()
    # is constructed fresh inside world_builder_node and reads these.
    if args.hard or args.rooms is not None:
        os.environ["HARD_MODE"] = "true"
        if args.rooms is not None:
            os.environ["NUM_ROOMS"] = str(args.rooms)

    log_nodes = args.log
    if log_nodes and "all" in log_nodes:
        log_nodes = list(NODE_NAMES)

    # --node NAME: run a single node independently against saved state, then exit.
    # --smoke takes precedence — when both are given, --node is ignored and the
    # smoke runner uses --mode to control which pipeline runs.
    if args.node and args.smoke is None:
        run_node(
            args.node,
            from_path=args.from_path,
            theme=args.theme,
            trace_eval=args.trace_eval,
        )
        sys.exit(0)

    # For all generation paths, ask the user to pick a theme now (once) unless
    # one was supplied via --theme.
    theme = args.theme or _pick_theme()

    if args.smoke is not None:
        if args.smoke < 1:
            parser.error("--smoke requires a positive integer")
        smoke(
            args.smoke,
            log_nodes=log_nodes,
            mode=args.mode,
            trace_eval=args.trace_eval,
            theme=theme,
            num_players=args.player,
        )
    else:
        run(
            mode=args.mode,
            log_nodes=log_nodes,
            theme=theme,
            trace_eval=args.trace_eval,
            num_players=args.player,
        )
