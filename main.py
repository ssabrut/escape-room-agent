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


def _merge_update(result: dict, update: dict) -> None:
    for key, value in update.items():
        if key == "messages":
            result.setdefault("messages", []).extend(value or [])
        else:
            result[key] = value


def _run_generate_only(
    theme: str = "pirate",
    log_nodes: list[str] | None = None,
    log_root: Path = LOG_DIR,
) -> dict:
    """Invoke world_builder then puzzle_builder and return the merged result dict."""
    from agents.game_master import world_builder_node
    from agents.puzzle_builder_node import puzzle_builder_node
    from state import GameState as _GS

    state = _GS(theme=theme)

    wb_update = world_builder_node(state)
    if log_nodes and "world_builder" in log_nodes:
        node_dir = _write_node_log("world_builder", wb_update, root=log_root)
        print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

    # Feed world_builder output into puzzle_builder
    state = state.model_copy(update={"world": wb_update.get("world")})
    pb_update = puzzle_builder_node(state)
    if log_nodes and "puzzle_builder" in log_nodes:
        node_dir = _write_node_log("puzzle_builder", pb_update, root=log_root)
        print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")

    return {**wb_update, **pb_update}


def run(mode: str = MODE_FULL, log_nodes: list[str] | None = None, theme: str = "") -> None:
    log_nodes = log_nodes or []
    if not theme:
        theme = _pick_theme()

    if mode == MODE_GENERATE:
        result = _run_generate_only(theme=theme, log_nodes=log_nodes)
        _render(result)
        return

    # Full pipeline
    if not log_nodes:
        result = graph.invoke(GameState(theme=theme))
        _render(result)
        return

    log_set = set(log_nodes)
    result: dict = {}
    for step in graph.stream(GameState(theme=theme), stream_mode="updates"):
        for node, update in step.items():
            _merge_update(result, update)
            if node in log_set:
                node_dir = _write_node_log(node, update)
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")
    _render(result)


def _run_once_captured(
    log_nodes: list[str] | None,
    log_root: Path | None,
    mode: str = MODE_FULL,
    theme: str = "pirate",
) -> tuple[str, dict]:
    """Run the pipeline once with stdout captured.

    If log_nodes is set, write per-node logs under log_root. Returns the captured
    text plus the merged result dict (so the caller can inspect the world, e.g.
    for the policy benchmark).
    """
    log_set = set(log_nodes or [])
    buf = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = buf
    try:
        if mode == MODE_GENERATE:
            result = _run_generate_only(
                theme=theme,
                log_nodes=list(log_set),
                log_root=log_root or LOG_DIR,
            )
        elif not log_set:
            result = graph.invoke(GameState(theme=theme))
        else:
            result = {}
            for step in graph.stream(GameState(theme=theme), stream_mode="updates"):
                for node, update in step.items():
                    _merge_update(result, update)
                    if node in log_set and log_root is not None:
                        _write_node_log(node, update, root=log_root)
        _render(result)
    finally:
        sys.stdout = orig_stdout
    return buf.getvalue(), result


def _run_narrative_eval(world, show_trace: bool = False, out_path: Path | None = None) -> None:
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
            output, result = _run_once_captured(log_nodes, log_root, mode=mode, theme=theme)
            out_file = run_dir / f"run_{i:03d}.txt"
            out_file.write_text(output, encoding="utf-8")
            bench_note = _write_benchmark(result.get("world"), run_dir / f"run_{i:03d}.benchmark.json")
            rooms_count = output.count("┌" + "─")
            print(f"done ({rooms_count} room(s)) → {out_file.name}{bench_note}")
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
        const=True,        # --eval with no arg: evaluate freshly generated world
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
        smoke(args.smoke, log_nodes=log_nodes, mode=args.mode,
              eval_narrative=eval_inline, eval_trace=eval_trace, theme=theme)
    else:
        if eval_inline:
            # Capture the result dict so we can pass the world to the evaluator
            # without a second generation call.
            log_root = LOG_DIR if log_nodes else None
            _, result = _run_once_captured(log_nodes, log_root, mode=args.mode, theme=theme)
            world = result.get("world")
            if world:
                eval_out = (log_root or Path(".")) / "eval.json"
                _run_narrative_eval(world, show_trace=eval_trace, out_path=eval_out)
            else:
                print("[eval] no world available to evaluate")
        else:
            run(mode=args.mode, log_nodes=log_nodes, theme=theme)
