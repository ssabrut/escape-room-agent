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

from graph import graph
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
    "player_agent_2",
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

    parsed = {k: _jsonable(v) for k, v in update.items() if k != "messages"}
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
            for i, step in enumerate(world.solution_path, 1):
                lines.append(f"  {i}. {step}")
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
        else:
            result[key] = value


def run(
    mode: str = MODE_FULL, log_nodes: list[str] | None = None, theme: str = ""
) -> None:
    log_nodes = log_nodes or []
    if not theme:
        theme = _pick_theme()

    node_times: dict[str, float] = {}
    log_set = set(log_nodes)

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

        state = state.model_copy(update={"world": wb_update.get("world")})
        t0 = time.perf_counter()
        pb_update = puzzle_builder_node(state)
        node_times["puzzle_builder"] = time.perf_counter() - t0
        if "puzzle_builder" in log_set:
            node_dir = _write_node_log("puzzle_builder", pb_update)
            print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

        result = {**wb_update, **pb_update}
        _render(result)
        _print_timing_table(node_times)
        return

    result: dict = {}
    _step_start = time.perf_counter()
    for step in graph.stream(GameState(theme=theme), stream_mode="updates"):
        _step_end = time.perf_counter()
        for node, update in step.items():
            node_times[node] = node_times.get(node, 0.0) + (_step_end - _step_start)
            _merge_update(result, update)
            if node in log_set:
                node_dir = _write_node_log(node, update)
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")
        _step_start = time.perf_counter()
    _render(result)
    _print_timing_table(node_times)


def _run_once_captured(
    log_nodes: list[str] | None,
    log_root: Path | None,
    mode: str = MODE_FULL,
    theme: str = "pirate",
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
            _step_start = time.perf_counter()
            for step in graph.stream(GameState(theme=theme), stream_mode="updates"):
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


def _run_narrative_eval(
    world, show_trace: bool = False, out_path: Path | None = None
) -> None:
    """Evaluate `world` with the LLM-as-judge + oracle, print the report, and optionally save it."""
    from benchmark.narrative_eval import evaluate_world, print_report, write_report

    report = evaluate_world(world)
    print_report(report, show_trace=show_trace)
    if out_path is not None:
        write_report(report, out_path)
        print(f"  [eval] wrote {out_path}")


def _load_world_from_json(path: Path) -> "GameWorld | None":
    """Load a GameWorld from a saved game_master output.json."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [eval] could not read {path}: {exc}")
        return None
    world_data = raw.get("world", raw)
    try:
        return GameWorld.model_validate(world_data)
    except Exception as exc:
        print(f"  [eval] could not parse GameWorld from {path}: {exc}")
        return None


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


def smoke(
    n: int,
    log_nodes: list[str] | None = None,
    mode: str = MODE_FULL,
    eval_narrative: bool = False,
    eval_trace: bool = False,
    theme: str = "pirate",
) -> None:
    SMOKE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_DIR / timestamp
    run_dir.mkdir()

    print(f"Smoke test ({mode}, theme: {theme}): {n} run(s) → {run_dir}/")
    if log_nodes:
        print(f"  [log] per-run node logs under {run_dir}/run_<NNN>_logs/")

    errors = 0
    for i in range(1, n + 1):
        print(f"  [{i}/{n}] generating...", end=" ", flush=True)
        try:
            log_root = run_dir / f"run_{i:03d}_logs" if log_nodes else None
            out_file = run_dir / f"run_{i:03d}.txt"
            output, result, node_times = _run_once_captured(
                log_nodes, log_root, mode=mode, theme=theme
            )
            _write_run_summary(result, out_file, node_times=node_times)
            (run_dir / f"run_{i:03d}_stdout.txt").write_text(output, encoding="utf-8")
            world_note = _write_world_json(
                result.get("world"), run_dir / f"run_{i:03d}.world.json"
            )
            bench_note = _write_benchmark(
                result.get("world"), run_dir / f"run_{i:03d}.benchmark.json"
            )
            total_elapsed = sum(node_times.values())
            print(
                f"done in {total_elapsed:.1f}s → {out_file.name}{world_note}{bench_note}"
            )
            _print_timing_table(node_times)
            if eval_narrative:
                world = result.get("world")
                if world:
                    eval_out = run_dir / f"run_{i:03d}.eval.json"
                    _run_narrative_eval(world, show_trace=eval_trace, out_path=eval_out)
        except Exception as e:
            errors += 1
            err_file = run_dir / f"run_{i:03d}.error.txt"
            err_file.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"ERROR → {err_file.name} ({e})")

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
            "  python main.py --eval path/to/output.json   # evaluate a saved world\n"
            "  python main.py --mode generate --eval        # generate then evaluate\n"
            "  python main.py --mode generate --smoke 3 --eval  # smoke + eval each run\n"
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
        "--eval",
        nargs="?",
        const=True,  # --eval with no arg: evaluate freshly generated world
        metavar="PATH",
        help=(
            "Evaluate world narrative quality (LLM-as-judge + oracle). "
            "Supply a path to an existing output.json to evaluate it directly, "
            "or omit the path to evaluate the world produced by the current run."
        ),
    )
    parser.add_argument(
        "--eval-trace",
        action="store_true",
        help="When --eval is active, also print the oracle's tick-by-tick solve trace.",
    )
    args = parser.parse_args()

    # Translate hard-mode flags into env vars BEFORE the graph runs; Settings()
    # is constructed fresh inside world_builder_node and reads these.
    if args.hard or args.rooms is not None:
        os.environ["HARD_MODE"] = "true"
        if args.rooms is not None:
            os.environ["NUM_ROOMS"] = str(args.rooms)

    log_nodes = args.log
    if log_nodes and "all" in log_nodes:
        log_nodes = list(NODE_NAMES)

    eval_arg = args.eval
    eval_trace = args.eval_trace

    # --eval PATH: load and evaluate a saved world — no theme needed, exit immediately.
    if isinstance(eval_arg, str):
        eval_path = Path(eval_arg)
        world = _load_world_from_json(eval_path)
        if world is None:
            sys.exit(1)
        eval_out = eval_path.parent / (eval_path.stem + ".eval.json")
        _run_narrative_eval(world, show_trace=eval_trace, out_path=eval_out)
        sys.exit(0)

    # For all generation paths, ask the user to pick a theme now (once).
    theme = _pick_theme()

    eval_inline = eval_arg is True  # --eval without a path: evaluate after generation

    if args.smoke is not None:
        if args.smoke < 1:
            parser.error("--smoke requires a positive integer")
        smoke(
            args.smoke,
            log_nodes=log_nodes,
            mode=args.mode,
            eval_narrative=eval_inline,
            eval_trace=eval_trace,
            theme=theme,
        )
    else:
        if eval_inline:
            # Capture the result dict so we can pass the world to the evaluator
            # without a second generation call.
            log_root = LOG_DIR if log_nodes else None
            _, result, node_times = _run_once_captured(
                log_nodes, log_root, mode=args.mode, theme=theme
            )
            _print_timing_table(node_times)
            world = result.get("world")
            if world:
                eval_out = (log_root or Path(".")) / "eval.json"
                _run_narrative_eval(world, show_trace=eval_trace, out_path=eval_out)
            else:
                print("[eval] no world available to evaluate")
        else:
            run(mode=args.mode, log_nodes=log_nodes, theme=theme)
