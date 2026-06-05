"""Main entrypoint — runs world_builder node only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.game_master import world_builder_node
from state import GameState, GameWorld


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


LOG_DIR = Path("logs")


def _jsonable(value):
    if isinstance(value, BaseModel):
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


def run(theme: str = "") -> None:
    if not theme:
        theme = _pick_theme()

    state = GameState(theme=theme)
    t0 = time.perf_counter()
    update = world_builder_node(state)
    elapsed = time.perf_counter() - t0

    node_dir = _write_node_log("world_builder", update)
    print(f"  wrote {node_dir}/output.json + {node_dir}/raw.txt")
    print(f"  elapsed: {elapsed:.2f}s")

    world = update.get("world")
    if world:
        print(f"\n  Scenario: {world.scenario}")
        print(f"  Objective: {world.objective}")
        print(f"  Rooms: {len(world.rooms)}")
        for room in world.rooms:
            print(f"    - {room.id}: {room.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="World Builder — generates escape room worlds",
    )
    parser.add_argument(
        "--theme",
        default="",
        metavar="THEME",
        help="Theme for generation (skips the interactive picker).",
    )
    args = parser.parse_args()

    run(theme=args.theme)
