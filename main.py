"""Main entrypoint — runs game_master once and prints output."""

from __future__ import annotations

import argparse
import io
import sys
import traceback
from datetime import datetime
from pathlib import Path

from graph import graph
from state import GameState
from visualization import render_room_layout

SMOKE_DIR = Path("smoke_runs")


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
        print(f"       ✦ {char.special_trait}")
        print()


def _render(result: dict) -> None:
    world = result.get("world")
    rooms = world.rooms if world else []

    print("\n" + "=" * 94)
    print(" ESCAPE ROOM MAP")
    print("=" * 94 + "\n")

    if rooms:
        render_room_layout(rooms)
    else:
        print("  [No room layout could be parsed from the LLM response]\n")

    characters = result.get("characters", [])
    _render_characters(characters)

    world = result.get("world")
    if world and world.game_flow.gates:
        flow = world.game_flow
        print("\n" + "=" * 94)
        print(" GAME FLOW")
        print("=" * 94 + "\n")
        print(f"  Start : {flow.starting_room}")
        print(f"  Goal  : {flow.win_condition}\n")
        for i, gate in enumerate(flow.gates, 1):
            req = gate.requires or "—"
            print(f"  Step {i}  [{gate.room}]")
            print(f"    Requires : {req}")
            print(f"    Unlocks  : {gate.unlocks}")
            print()


def run() -> None:
    result = graph.invoke(GameState(theme="pirate"))
    _render(result)


def _run_once_captured() -> str:
    """Run the graph once with stdout captured, returning the buffered output."""
    buf = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = buf
    try:
        result = graph.invoke(GameState(theme="pirate"))
        _render(result)
    finally:
        sys.stdout = orig_stdout
    return buf.getvalue()


def smoke(n: int) -> None:
    SMOKE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_DIR / timestamp
    run_dir.mkdir()

    print(f"Smoke test: {n} run(s) → {run_dir}/")

    errors = 0
    for i in range(1, n + 1):
        print(f"  [{i}/{n}] generating...", end=" ", flush=True)
        try:
            output = _run_once_captured()
            out_file = run_dir / f"run_{i:03d}.txt"
            out_file.write_text(output, encoding="utf-8")
            rooms_count = output.count("┌" + "─")
            print(f"done ({rooms_count} room(s)) → {out_file.name}")
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
    parser = argparse.ArgumentParser(description="Escape room game master")
    parser.add_argument(
        "--smoke",
        metavar="N",
        type=int,
        help="Run the generator N times and save each output to smoke_runs/<timestamp>/",
    )
    args = parser.parse_args()

    if args.smoke is not None:
        if args.smoke < 1:
            parser.error("--smoke requires a positive integer")
        smoke(args.smoke)
    else:
        run()
