"""Generate a bank of HARD, solvable worlds for the benchmark — LLM-backed.

Uses the local Ollama model to author N-room worlds with deep puzzle chains and
decoys (prompt: ``game_master/generation_bank.txt``), runs each through the live
``_build_world`` repair/validation pipeline so it is structurally sound, then
keeps only worlds that the heuristic oracle can actually SOLVE (so the bank is
hard but winnable). Passing worlds are saved as JSON under ``benchmark/worlds/``.

This is a benchmark-only path: it does NOT touch the live game's generation
prompt or the production ``MAX_ROOMS`` default (it overrides the module global
just for the duration of a bank build).

Usage:
    python -m benchmark.generate_bank --count 20 --rooms 4 --model qwen3.5:9b-mlx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from agents import game_master as gm
from benchmark.engine import HeadlessEpisode
from benchmark.policies import heuristic_policy
from config.settings import Settings
from prompts import load_prompt
from state import GameWorld

WORLDS_DIR = ROOT / "benchmark" / "worlds"
BANK_PROMPT = load_prompt("game_master", "generation_bank")
SYSTEM_PROMPT = load_prompt("game_master", "system")

THEMES = [
    "pirate ship",
    "abandoned space station",
    "haunted manor",
    "ancient tomb",
    "underground laboratory",
    "sunken submarine",
    "clockwork tower",
    "frozen research base",
]


def _make_llm(model: str, temperature: float) -> ChatOllama:
    s = Settings()
    return ChatOllama(
        model=model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        reasoning=False,
    )


def _generate_one(llm: ChatOllama, theme: str, rooms: int, chain_depth: int,
                  decoys: int) -> tuple[GameWorld | None, list[str]]:
    """One LLM generation -> (validated GameWorld | None, build-log lines).

    Temporarily raises gm.MAX_ROOMS so _build_world keeps all N rooms instead of
    truncating to the production cap of 2. The repair/coherence warnings that
    _build_world prints are captured (not swallowed) and returned so the caller
    can show them inline per attempt under --debug.
    """
    prompt = BANK_PROMPT.format(
        theme=theme, num_rooms=rooms, chain_depth=chain_depth, decoys=decoys
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    data = gm._parse_json(response.content)
    if not data:
        return None, []
    orig_cap = gm.MAX_ROOMS
    gm.MAX_ROOMS = rooms
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            world = gm._build_world(data)
    except Exception:
        world = None
    finally:
        gm.MAX_ROOMS = orig_cap
    build_log = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return world, build_log


def _oracle_solve(world: GameWorld, trace: bool = False):
    """Run the heuristic oracle once; return (victory, measured_chain_depth, history).

    chain_depth is the count of ordered dependency links the oracle had to clear
    (unlocks / power / room moves) — the real difficulty measure used to reject
    worlds the repair pipeline flattened to a trivial grab-the-nearby-tool win.
    `history` is the tick-by-tick solve trace (empty unless trace=True).
    """
    if not world.win_condition.object_id or len(world.rooms) < 1:
        return False, 0, []
    result = HeadlessEpisode(world).run(heuristic_policy, record_history=trace)
    return result.victory, result.chain_depth, result.history


def _world_signature(world: GameWorld) -> tuple:
    """Cheap dedup key so near-identical regenerations don't fill the bank."""
    return (
        len(world.rooms),
        len(world.objects),
        world.win_condition.object_id,
        tuple(sorted(o.id for o in world.objects)),
    )


def _print_indented(lines: list[str], header: str | None = None) -> None:
    """Print captured lines indented under the current attempt for live debugging."""
    if header and lines:
        print(f"    {header}")
    for line in lines:
        print(f"      {line}")


def _print_trace(history: list[str]) -> None:
    """Indented tick-by-tick oracle solve trace for live debugging."""
    if not history:
        print("      (oracle made no moves)")
        return
    _print_indented(history, header="oracle trace:")


def generate_bank(count: int, rooms: int, chain_depth: int, decoys: int,
                  model: str, temperature: float, max_attempts: int,
                  min_chain_depth: int, debug: bool = False,
                  fresh: bool = False) -> None:
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    if fresh:
        # Clear leftover worlds so a new bank isn't contaminated by a prior run.
        # (A shell `rm *.json` silently no-ops under zsh when the dir is empty, so
        # do it here where emptiness is harmless.)
        stale = sorted(WORLDS_DIR.glob("world_*.json"))
        for p in stale:
            p.unlink()
        if stale:
            print(f"  (cleared {len(stale)} stale world file(s))")
    llm = _make_llm(model, temperature)

    print(
        f"Generating {count} world(s): {rooms} rooms, request chain>={chain_depth}, "
        f"ACCEPT measured depth>={min_chain_depth}, {decoys} decoys/room, "
        f"model={model}\n"
    )

    kept = 0
    seen: set[tuple] = set()
    attempts = 0
    while kept < count and attempts < max_attempts:
        attempts += 1
        theme = THEMES[attempts % len(THEMES)]
        print(f"  attempt {attempts:>3} [{theme}] ...", end="", flush=True)
        world, build_log = _generate_one(llm, theme, rooms, chain_depth, decoys)
        if world is None:
            print(" parse/build failed")
            if debug:
                _print_indented(build_log, header="build log:")
            continue
        sig = _world_signature(world)
        if sig in seen:
            print(" duplicate, skipped")
            continue
        victory, depth, history = _oracle_solve(world, trace=debug)
        size = f"{len(world.rooms)}r/{len(world.objects)}o"
        if not victory:
            print(f" unsolvable by oracle ({size})")
            if debug:
                _print_indented(build_log, header="build log:")
                _print_trace(history)
            continue
        if depth < min_chain_depth:
            # Solvable but the repair net flattened it — too shallow to be a useful
            # benchmark target. Reject rather than pad the bank with trivial worlds.
            print(f" too shallow: depth {depth} < {min_chain_depth} ({size})")
            if debug:
                _print_indented(build_log, header="build log:")
                _print_trace(history)
            continue
        seen.add(sig)
        kept += 1
        out = WORLDS_DIR / f"world_{kept:03d}.json"
        out.write_text(
            json.dumps({"world": world.model_dump(mode="json", exclude_none=True)},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f" KEPT -> {out.name} ({size}, depth {depth})")
        if debug:
            _print_indented(build_log, header="build log:")
            _print_trace(history)

    print(f"\nDone: {kept}/{count} kept in {attempts} attempt(s) -> {WORLDS_DIR}/")
    if kept < count:
        print("  (raise --max-attempts or lower difficulty if short)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hard benchmark world bank")
    parser.add_argument("--count", type=int, default=10, help="worlds to keep")
    parser.add_argument("--rooms", type=int, default=4, help="rooms per world")
    parser.add_argument("--chain-depth", type=int, default=4,
                        help="dependent solution steps to REQUEST in the prompt")
    parser.add_argument("--min-chain-depth", type=int, default=0,
                        help="reject worlds the oracle solves in fewer MEASURED "
                             "dependency steps (default: same as --chain-depth)")
    parser.add_argument("--decoys", type=int, default=3, help="decoy objects per room")
    parser.add_argument("--model", default="qwen3.5:9b-mlx", help="Ollama model")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="cap on generations (default: count * 4)")
    parser.add_argument("--debug", action="store_true",
                        help="print the oracle's tick-by-tick solve trace for "
                             "every attempt (kept or rejected)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete existing world_*.json before generating, so "
                             "the bank isn't contaminated by a prior run")
    args = parser.parse_args()

    max_attempts = args.max_attempts or args.count * 4
    min_chain_depth = args.min_chain_depth or args.chain_depth
    generate_bank(
        count=args.count,
        rooms=args.rooms,
        chain_depth=args.chain_depth,
        decoys=args.decoys,
        model=args.model,
        temperature=args.temperature,
        max_attempts=max_attempts,
        min_chain_depth=min_chain_depth,
        debug=args.debug,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
