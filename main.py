"""Main entrypoint — runs game_master once and prints output."""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from graph import graph
from state import GameState
from visualization import render_room_layout

SMOKE_DIR = Path("smoke_runs")
LOG_DIR = Path("logs")
NODE_NAMES = (
    "game_master",
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
        ab = char.ability
        uses = "passive" if ab.max_uses < 0 else f"{ab.max_uses} use(s)"
        print(f"       ✦ {ab.name} [{ab.effect}, {uses}] — {ab.description}")
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


def run(log_nodes: list[str] | None = None) -> None:
    log_nodes = log_nodes or []
    if not log_nodes:
        result = graph.invoke(GameState(theme="pirate"))
        _render(result)
        return

    log_set = set(log_nodes)
    result: dict = {}
    for step in graph.stream(GameState(theme="pirate"), stream_mode="updates"):
        for node, update in step.items():
            _merge_update(result, update)
            if node in log_set:
                node_dir = _write_node_log(node, update)
                print(f"  [log] wrote {node_dir}/output.json + {node_dir}/raw.txt")
    _render(result)


def _run_once_captured(log_nodes: list[str] | None, log_root: Path | None) -> str:
    """Run the graph once with stdout captured. If log_nodes is set, write per-node logs under log_root."""
    log_set = set(log_nodes or [])
    buf = io.StringIO()
    orig_stdout = sys.stdout
    sys.stdout = buf
    try:
        if not log_set:
            result = graph.invoke(GameState(theme="pirate"))
        else:
            result = {}
            for step in graph.stream(GameState(theme="pirate"), stream_mode="updates"):
                for node, update in step.items():
                    _merge_update(result, update)
                    if node in log_set and log_root is not None:
                        _write_node_log(node, update, root=log_root)
        _render(result)
    finally:
        sys.stdout = orig_stdout
    return buf.getvalue()


def smoke(n: int, log_nodes: list[str] | None = None) -> None:
    SMOKE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SMOKE_DIR / timestamp
    run_dir.mkdir()

    print(f"Smoke test: {n} run(s) → {run_dir}/")
    if log_nodes:
        print(f"  [log] per-run node logs under {run_dir}/run_<NNN>_logs/")

    errors = 0
    for i in range(1, n + 1):
        print(f"  [{i}/{n}] generating...", end=" ", flush=True)
        try:
            log_root = run_dir / f"run_{i:03d}_logs" if log_nodes else None
            output = _run_once_captured(log_nodes, log_root)
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
    parser.add_argument(
        "--log",
        metavar="NODE",
        action="append",
        choices=(*NODE_NAMES, "all"),
        help=(
            "Log a node's parsed output (logs/<node>.json) and raw LLM response "
            "(logs/<node>.raw.txt). Repeatable. Use 'all' to log every node. "
            "Choices: " + ", ".join((*NODE_NAMES, "all"))
        ),
    )
    args = parser.parse_args()

    log_nodes = args.log
    if log_nodes and "all" in log_nodes:
        log_nodes = list(NODE_NAMES)

    if args.smoke is not None:
        if args.smoke < 1:
            parser.error("--smoke requires a positive integer")
        smoke(args.smoke, log_nodes=log_nodes)
    else:
        run(log_nodes=log_nodes)
