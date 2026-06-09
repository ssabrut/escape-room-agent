"""Generate fine-tuning datasets for the two generation agents via the live pipeline.

The real game runs world generation in two stages, each owned by a different agent:

  world_builder  (agents.world_builder)
      Theme  ->  rooms-only skeleton (scenario, objective, rooms, room goals).
      Validated by `_eval_world_structure` (deterministic topology check).

  puzzle_builder (agents.puzzle_builder_node)
      Rooms skeleton  ->  objects, locks, clues, solution path.
      Validated by `_eval_puzzle` (static backward-chain + per-room + global oracle).

This script fine-tunes BOTH agents. For each accepted world it emits training
examples into two independent dataset trees:

  dataset/world/...    — train world_builder (theme -> rooms skeleton)
  dataset/puzzle/...    — train puzzle_builder (rooms skeleton -> objects + solution)

Each tree carries two formats:

  SFT  — instruction-tuning pairs (system + user prompt + accepted assistant JSON).
         Every accepted artifact yields one SFT example regardless of retries.

  DPO  — preference pairs (chosen = accepted artifact, rejected = a bad artifact
         corrected into it). Produced only for steps that had a real violation →
         fix transition, so the trainer sees a clean bad→good signal with the
         exact correction prompt the model received.

Output layout:
  dataset/world/sft/<theme>/<nnnn>.jsonl    — one SFT file per accepted world
  dataset/world/dpo/<theme>/<nnnn>.jsonl    — one DPO file per world rejection step
  dataset/puzzle/sft/<theme>/<nnnn>.jsonl   — one SFT file per accepted puzzle
  dataset/puzzle/dpo/<theme>/<nnnn>.jsonl   — one DPO file per puzzle rejection step
  dataset/world/sft_all.jsonl   dataset/world/dpo_all.jsonl
  dataset/puzzle/sft_all.jsonl  dataset/puzzle/dpo_all.jsonl
  dataset/manifest.json                     — counts and run metadata

Usage:
    python -m benchmark.generate_dataset
    python -m benchmark.generate_dataset --per-theme 120
    python -m benchmark.generate_dataset --themes "Horror" "Pirate Adventure"
    python -m benchmark.generate_dataset --target world      # only world_builder data
    python -m benchmark.generate_dataset --target puzzle     # only puzzle_builder data
    python -m benchmark.generate_dataset --resume            # continue partial themes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.escape_rooms.nodes.world_builder import MAX_ROOMS
from src.escape_rooms.nodes.world_builder import SYSTEM_PROMPT as WORLD_SYSTEM_PROMPT
from src.escape_rooms.nodes.world_builder import (
    _eval_world_structure,
    _generate_world,
    _generation_prompt,
)
from src.escape_rooms.nodes.puzzle_builder import SYSTEM_PROMPT as PUZZLE_SYSTEM_PROMPT
from src.escape_rooms.nodes.puzzle_builder import _build_prompt as _puzzle_prompt
from src.escape_rooms.nodes.puzzle_builder import (
    _default_chain_depth,
    _eval_puzzle,
    _generate_puzzle,
    _generate_puzzle_with_feedback,
    _min_objects_per_room,
)
from benchmark.policies import bfs_solution_path
from src.escape_rooms.utils.settings import Settings, get_llm
from src.escape_rooms.state import GameWorld

ALL_THEMES = [
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

DATASET_DIR = ROOT / "dataset"
WORLD_DIR = DATASET_DIR / "world"
PUZZLE_DIR = DATASET_DIR / "puzzle"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slug(theme: str) -> str:
    return theme.lower().replace(" ", "_").replace("/", "_")


def _world_signature(world: GameWorld) -> tuple:
    return (
        len(world.rooms),
        len(world.objects),
        world.win_condition.object_id,
        tuple(sorted(o.id for o in world.objects)),
    )


def _world_json(world: GameWorld) -> str:
    return json.dumps(
        {"world": world.model_dump(mode="json", exclude_none=True)},
        ensure_ascii=False,
    )


def _count_existing(theme_dir: Path) -> int:
    return sum(1 for _ in theme_dir.glob("*.jsonl")) if theme_dir.exists() else 0


def _silent(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with stdout suppressed; return result or raise."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _correction(kind: str, violations: list[str]) -> str:
    block = "\n".join(f"  - {v}" for v in violations)
    return (
        f"Your previous {kind} had the following issues detected by automated "
        f"checks. Please generate a NEW, corrected {kind} that fixes ALL of them. "
        "Return only the JSON object — no prose.\n\n"
        f"Issues to fix:\n{block}"
    )


def _puzzle_issues(world: GameWorld, chain_target: int, min_objs: int) -> list[str]:
    """Deterministic acceptance gate for a puzzle world.

    Runs `_eval_puzzle` (static solvability + key objects + object counts +
    oracle). An empty list is the single accept condition for the revision loop.
    """
    return _eval_puzzle(world, chain_target, min_objs)


# ---------------------------------------------------------------------------
# Contrastive corruptors — turn an ACCEPTED (solvable + deep) world into a
# deliberately WORSE one on a single, named axis. The accepted world is always
# `chosen`; the corrupted copy is `rejected`. Because both share the identical
# prompt, the only signal DPO can learn is the defect itself — never the story.
#
# Each corruptor returns (corrupted_world, violation_label) or None when it
# cannot apply to this particular world.
# ---------------------------------------------------------------------------


from src.escape_rooms.state import WorldObject


def _clone(world: GameWorld) -> GameWorld:
    return world.model_copy(deep=True)


def _corrupt_unsolvable(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Break the win chain: re-lock the win object and strip its unlock mechanism.

    Produces a world the oracle can no longer win — the clearest 'unsolvable'
    contrast. Returns None if no win object can be located.
    """
    win_id = world.win_condition.object_id
    if not win_id:
        return None
    bad = _clone(world)
    target = next((o for o in bad.objects if o.id == win_id), None)
    if target is None:
        return None
    target.state = "locked"
    target.requires_code = None
    target.requires_tool = None
    target.requires_liquid = None
    target.requires_power = None
    target.fuses = None
    return (
        bad,
        f"win object '{win_id}' is locked with no unlock mechanism — world is unsolvable",
    )


def _corrupt_shallow(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Flatten the difficulty: open every gate so the win object is reachable in
    one step. Stays solvable but collapses chain depth — the 'easy' contrast.

    Returns None if there is nothing gated to flatten (already trivially easy).
    """
    bad = _clone(world)
    opened = 0
    for o in bad.objects:
        had_gate = any(
            [
                o.state in {"locked", "locked_bolt", "locked_room", "hidden"},
                o.requires_code,
                o.requires_tool,
                o.requires_liquid,
                o.requires_power,
                o.fuses,
            ]
        )
        if had_gate:
            o.state = "visible"
            o.requires_code = None
            o.requires_tool = None
            o.requires_liquid = None
            o.requires_power = None
            o.fuses = None
            opened += 1
    if opened == 0:
        return None
    return (
        bad,
        f"all {opened} gate(s) removed — world is solvable but trivially shallow (no puzzle chain)",
    )


def _corrupt_orphans(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Inject uncorrelated objects that connect to nothing in the solution chain.

    Re-introduces exactly what `_prune_orphan_objects` removes: takeable props
    with a gated dependency that no clue, tool, or goal ever consumes. Returns
    None if the world has no rooms.
    """
    if not world.rooms:
        return None
    bad = _clone(world)
    existing = {o.id for o in bad.objects}
    n = rng.randint(2, 3)
    added: list[str] = []
    for i in range(n):
        room = rng.choice(bad.rooms)
        oid = f"orphan_relic_{i + 1}"
        while oid in existing:
            oid += "_x"
        existing.add(oid)
        bad.objects.append(
            WorldObject(
                id=oid,
                location=room.id,
                description="An ornate relic that seems important but connects to nothing.",
                state="locked",
                interactable=True,
                takeable=False,
                requires_code="9981",  # a code no object ever reveals
            )
        )
        added.append(oid)
    return bad, (
        f"objects {', '.join(added)} are uncorrelated — gated by a code/clue that "
        "nothing in the world produces and never used by the solution"
    )


def _corrupt_phantom_solution(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Inject solution_path steps that reference object ids which do not exist.

    Corrupts the oracle-derived ground-truth path by splicing in a narrated step
    that names a phantom object id. Returns None if the world has no solution_path.
    """
    if not world.solution_path:
        return None
    bad = _clone(world)
    phantom_id = "phantom_keystone_relic"
    insert_at = rng.randint(0, len(bad.solution_path))
    step = (
        f"Use the {phantom_id} on the final lock to finish — "
        "an object that appears nowhere in this world."
    )
    bad.solution_path.insert(insert_at, step)
    return bad, (
        f"solution_path references '{phantom_id}', an object id that does not exist "
        "in the world"
    )


def _corrupt_untakeable_tool(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Make a tool the solution depends on impossible to pick up.

    Re-introduces what `_make_required_tools_takeable` repairs: a gate whose
    `requires_tool` points at an object the player can never carry, breaking the
    win chain at an *intermediate* dependency rather than at the win object
    itself. A common, subtle LLM failure mode and distinct from `unsolvable`.

    Returns None if no object depends on a takeable tool.
    """
    bad = _clone(world)
    by_id = {o.id: o for o in bad.objects}
    tool_ids = [
        o.requires_tool
        for o in bad.objects
        if o.requires_tool
        and o.requires_tool in by_id
        and by_id[o.requires_tool].takeable
    ]
    if not tool_ids:
        return None
    tool_id = rng.choice(tool_ids)
    by_id[tool_id].takeable = False
    return bad, (
        f"required tool '{tool_id}' is not takeable — the player can never carry it "
        "to the lock it opens, breaking the solution chain"
    )


# Puzzle-stage corruptors keyed by axis name (CLI-selectable).
PUZZLE_CORRUPTORS = {
    "unsolvable": _corrupt_unsolvable,
    "shallow": _corrupt_shallow,
    "orphans": _corrupt_orphans,
    "phantom": _corrupt_phantom_solution,
    "untakeable_tool": _corrupt_untakeable_tool,
}


# ---------------------------------------------------------------------------
# World-stage corruptors — same idea as the puzzle ones, but degrade the
# rooms-only SKELETON on the structural axes `_eval_world_structure` checks
# (room count, metadata, adjacency topology). The accepted skeleton is `chosen`;
# the corrupted copy is `rejected`. These give world_builder real preference
# signal, which the live retry loop almost never produces (skeletons usually
# pass structural eval first-try, so 0 natural DPO pairs).
# ---------------------------------------------------------------------------


def _corrupt_broken_adjacency(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Point an exit at a room that doesn't exist (dangling adjacency)."""
    rooms_with_exits = [r for r in world.rooms if r.adjacency]
    if not rooms_with_exits:
        return None
    bad = _clone(world)
    room = next(r for r in bad.rooms if r.id == rng.choice(rooms_with_exits).id)
    direction = rng.choice(list(room.adjacency))
    room.adjacency[direction] = "nonexistent_void_room"
    return bad, (
        f"room '{room.id}': adjacency '{direction}' points to "
        "'nonexistent_void_room', which is not a real room"
    )


def _corrupt_unmirrored_adjacency(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Delete the reverse exit so adjacency is one-directional (not mirrored)."""
    opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
    candidates = [
        (r, d, n)
        for r in world.rooms
        for d, n in r.adjacency.items()
        if d in opp
    ]
    if not candidates:
        return None
    bad = _clone(world)
    r0, d0, n0 = rng.choice(candidates)
    neighbor = next((x for x in bad.rooms if x.id == n0), None)
    if neighbor is None or opp[d0] not in neighbor.adjacency:
        return None
    del neighbor.adjacency[opp[d0]]
    return bad, (
        f"room '{r0.id}': adjacency not mirrored — '{n0}' is missing the "
        f"'{opp[d0]}' exit back to '{r0.id}'"
    )


def _corrupt_missing_room_goal(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Strip a room's goal_completion so it has no win condition."""
    gated = [r for r in world.rooms if r.goal_completion is not None]
    if not gated:
        return None
    bad = _clone(world)
    target = next(r for r in bad.rooms if r.id == rng.choice(gated).id)
    target.goal_completion = None
    return bad, f"room '{target.id}': missing goal_completion"


def _corrupt_duplicate_room(
    world: GameWorld, rng: random.Random
) -> tuple[GameWorld, str] | None:
    """Duplicate a room id and drop another, keeping the count right but invalid."""
    if len(world.rooms) < 2:
        return None
    bad = _clone(world)
    a, b = rng.sample(range(len(bad.rooms)), 2)
    bad.rooms[b].id = bad.rooms[a].id  # now two rooms share an id
    return bad, f"duplicate room id: '{bad.rooms[a].id}'"


# World-stage corruptors keyed by axis name (CLI-selectable).
WORLD_CORRUPTORS = {
    "broken_adjacency": _corrupt_broken_adjacency,
    "unmirrored_adjacency": _corrupt_unmirrored_adjacency,
    "missing_room_goal": _corrupt_missing_room_goal,
    "duplicate_room": _corrupt_duplicate_room,
}


def _corruption_is_real(
    axis: str, bad: GameWorld, chain_target: int, min_objs: int
) -> bool:
    """Confirm a corrupted world is genuinely worse on its axis before keeping it.

    We do not trust the corruptor's claim blindly: the `rejected` example must be
    demonstrably degraded, or the DPO pair teaches nothing. Validation differs by
    axis because `_eval_puzzle` models solvability/depth but NOT solution-text or
    orphan defects (those are silently repaired at build time, never evaluated).
    """
    if axis == "unsolvable":
        # Oracle must now fail end-to-end.
        return any(
            _is_oracle_failure(i) for i in _eval_puzzle(bad, chain_target, min_objs)
        )
    if axis == "shallow":
        # Must stay winnable but drop below the chain-depth target. Only meaningful
        # when a positive target is set (hard mode); otherwise skip the axis.
        if chain_target <= 0:
            return False
        issues = _eval_puzzle(bad, chain_target, min_objs)
        if any(_is_oracle_failure(i) for i in issues):
            return False  # became unsolvable, not merely shallow — wrong axis
        return any("chain depth" in i for i in issues)
    if axis == "orphans":
        # At least one injected object must be unreachable from the win chain.
        return _has_orphan_objects(bad)
    if axis == "phantom":
        # solution_path must reference an id absent from the world.
        return _has_phantom_solution_ids(bad)
    if axis == "untakeable_tool":
        # Breaking a tool dependency must make the oracle fail end-to-end.
        return any(
            _is_oracle_failure(i) for i in _eval_puzzle(bad, chain_target, min_objs)
        )
    return False


def _is_oracle_failure(issue: str) -> bool:
    return issue.startswith("oracle failed to win") or "no win condition" in issue


def _has_phantom_solution_ids(world: GameWorld) -> bool:
    valid = {o.id for o in world.objects} | {r.id for r in world.rooms}
    for step in world.solution_path:
        for tok in re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", step):
            if tok not in valid and tok not in {
                "the_hidden",
                "hidden_code",
                "the_object",
                "solution_path",
            }:
                return True
    return False


def _has_orphan_objects(world: GameWorld) -> bool:
    """True if any non-scenic object is gated by a code/clue nothing produces."""
    produced = {o.contains_info for o in world.objects if o.contains_info}
    for o in world.objects:
        if o.scenic:
            continue
        code = o.requires_code
        if code and not any(
            (
                info
                and (
                    re.sub(r"[^0-9]", "", info) == re.sub(r"[^0-9]", "", code) and code
                )
            )
            or (info and (code in info or info in code))
            for info in produced
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def _world_sft(theme: str, world: GameWorld) -> dict:
    """world_builder SFT: theme prompt -> rooms-only skeleton."""
    return {
        "messages": [
            {"role": "system", "content": WORLD_SYSTEM_PROMPT},
            {"role": "user", "content": _generation_prompt(theme)},
            {"role": "assistant", "content": _world_json(world)},
        ]
    }


def _world_dpo(
    theme: str, violations: list[str], bad: GameWorld, good: GameWorld
) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": WORLD_SYSTEM_PROMPT},
            {"role": "user", "content": _generation_prompt(theme)},
            {"role": "user", "content": _correction("world", violations)},
        ],
        "chosen": [{"role": "assistant", "content": _world_json(good)}],
        "rejected": [{"role": "assistant", "content": _world_json(bad)}],
        "violations": violations,
    }


def _puzzle_sft(
    skeleton: GameWorld, full: GameWorld, chain_depth: int, min_objs: int
) -> dict:
    """puzzle_builder SFT: rooms skeleton prompt -> objects + solution path."""
    return {
        "messages": [
            {"role": "system", "content": PUZZLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _puzzle_prompt(skeleton, chain_depth, min_objs),
            },
            {"role": "assistant", "content": _world_json(full)},
        ]
    }


def _puzzle_dpo(
    skeleton: GameWorld,
    violations: list[str],
    bad: GameWorld,
    good: GameWorld,
    chain_depth: int,
    min_objs: int,
) -> dict:
    user = _puzzle_prompt(skeleton, chain_depth, min_objs)
    return {
        "prompt": [
            {"role": "system", "content": PUZZLE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "user", "content": _correction("puzzle", violations)},
        ],
        "chosen": [{"role": "assistant", "content": _world_json(good)}],
        "rejected": [{"role": "assistant", "content": _world_json(bad)}],
        "violations": violations,
    }


CURRENT_DIFFICULTY = "env"  # set in main(); stamped onto every written example


def _write(path: Path, payload: dict) -> None:
    payload = {**payload, "difficulty": CURRENT_DIFFICULTY}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-theme generation
# ---------------------------------------------------------------------------


def generate_theme(
    theme: str,
    per_theme: int,
    max_attempts: int,
    max_total_attempts: int,
    max_rooms: int,
    chain_depth: int,
    chain_target: int,
    min_objs: int,
    targets: set[str],
    corruptors: list[str],
    world_corruptors: list[str],
    max_dpo_per_world: int,
    rng: random.Random,
    resume_from_world: int,
    resume_from_puzzle: int,
    debug: bool,
) -> dict[str, int]:
    """Run the live two-stage pipeline for one theme.

    Returns a dict of counts: world_sft, world_dpo, puzzle_sft, puzzle_dpo.
    """
    slug = _slug(theme)
    dirs = {
        "world_sft": WORLD_DIR / "sft" / slug,
        "world_dpo": WORLD_DIR / "dpo" / slug,
        "puzzle_sft": PUZZLE_DIR / "sft" / slug,
        "puzzle_dpo": PUZZLE_DIR / "dpo" / slug,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    llm = get_llm("game_master")

    counts = {
        "world_sft": resume_from_world,
        "world_dpo": _count_existing(dirs["world_dpo"]),
        "puzzle_sft": resume_from_puzzle,
        "puzzle_dpo": _count_existing(dirs["puzzle_dpo"]),
    }
    seen: set[tuple] = set()
    total_attempts = 0

    # Progress is driven by whichever target(s) we are collecting. An accepted
    # world produces (when its target is enabled) one world SFT and one puzzle
    # SFT, so we track them against the same per_theme budget.
    def _need_more() -> bool:
        if "world" in targets and counts["world_sft"] < per_theme:
            return True
        if "puzzle" in targets and counts["puzzle_sft"] < per_theme:
            return True
        return False

    print(f"\n{'─' * 64}")
    print(
        f"  Theme : {theme}  (target {per_theme}; "
        f"world {counts['world_sft']}, puzzle {counts['puzzle_sft']})"
    )
    print(f"{'─' * 64}")

    while _need_more() and total_attempts < max_total_attempts:
        total_attempts += 1
        print(f"  [{theme}] attempt {total_attempts} ...", end="", flush=True)

        # --- Stage 1: world_builder, with structural-eval retry + DPO capture ---
        try:
            world, _ = _silent(_generate_world, llm, theme, max_rooms)
        except Exception as exc:
            print(f" world gen error: {exc}")
            continue

        world_issues = _eval_world_structure(world, max_rooms)
        retry = 0
        while world_issues and retry < max_attempts - 1:
            retry += 1
            bad_world = world
            if debug:
                for v in world_issues:
                    print(f"\n      world: {v}", end="")
            try:
                world, _ = _silent(_generate_world, llm, theme, max_rooms)
            except Exception as exc:
                print(f" world regen error: {exc}")
                break
            world_issues = _eval_world_structure(world, max_rooms)
            if "world" in targets and not world_issues:
                counts["world_dpo"] += 1
                out = dirs["world_dpo"] / f"{counts['world_dpo']:04d}.jsonl"
                _write(
                    out,
                    _world_dpo(
                        theme,
                        _eval_world_structure(bad_world, max_rooms),
                        bad_world,
                        world,
                    ),
                )

        if world_issues:
            print(
                f"\n    => world discarded after {retry + 1} attempt(s): "
                f"{len(world_issues)} issue(s)"
            )
            continue

        skeleton = world  # rooms-only; keep a clean copy for the puzzle prompt

        # --- Stage 2: puzzle_builder, with eval-B retry + DPO capture ---
        try:
            full, _ = _silent(_generate_puzzle, llm, skeleton, chain_depth, min_objs)
        except Exception as exc:
            print(f" puzzle gen error: {exc}")
            continue

        # --- Revision loop -------------------------------------------------
        # Drives the world until it satisfies the deterministic policy
        # (`_puzzle_issues` → `_eval_puzzle`: solvable + deep). Every attempt
        # feeds the full issue list back into the regen. The loop exits only
        # when issues == [] or the attempt budget is exhausted; in the latter
        # case the world is DISCARDED below (hard gate), so a written world is
        # always solvable and deep.
        puzzle_issues = _puzzle_issues(full, chain_target, min_objs)
        pretry = 0
        while puzzle_issues and pretry < max_attempts - 1:
            pretry += 1
            bad_full = full
            cleared_axis = "solvability"
            if debug:
                for v in puzzle_issues:
                    print(f"\n      puzzle: {v}", end="")
            try:
                full, _ = _silent(
                    _generate_puzzle_with_feedback,
                    llm,
                    skeleton,
                    chain_depth,
                    min_objs,
                    puzzle_issues,
                )
            except Exception as exc:
                print(f" puzzle regen error: {exc}")
                break
            prev_issues = puzzle_issues
            puzzle_issues = _puzzle_issues(full, chain_target, min_objs)
            # Capture the bad->good transition as a real DPO pair the moment the
            # regen clears ALL issues.
            if "puzzle" in targets and not puzzle_issues:
                counts["puzzle_dpo"] += 1
                out = dirs["puzzle_dpo"] / f"{counts['puzzle_dpo']:04d}.jsonl"
                pair = _puzzle_dpo(
                    skeleton, prev_issues, bad_full, full, chain_depth, min_objs
                )
                pair["axis"] = cleared_axis
                _write(out, pair)

        if puzzle_issues:
            print(
                f"\n    => puzzle discarded after {pretry + 1} attempt(s): "
                f"{len(puzzle_issues)} issue(s) (policy not satisfied)"
            )
            continue

        # Past this point `full` is guaranteed solvable and deep — the revised,
        # accepted world.

        sig = _world_signature(full)
        if sig in seen:
            print(" duplicate, skipped")
            continue
        seen.add(sig)

        # Ground-truth solution path: derive it from the oracle's actual winning
        # solve over the finalized object graph (never the LLM's own). This is the
        # canonical path baked into the SFT target and the basis the phantom-path
        # corruptor mutates for DPO, so it must be set before either runs.
        full.solution_path = bfs_solution_path(full)

        # --- Both stages accepted: write SFT examples for enabled targets ---
        notes = []
        if "world" in targets:
            counts["world_sft"] += 1
            out = dirs["world_sft"] / f"{counts['world_sft']:04d}.jsonl"
            _write(out, _world_sft(theme, skeleton))
            notes.append(f"world {counts['world_sft']}/{per_theme}")

            # Synthetic world DPO — same cap/shuffle/validate pattern as puzzle.
            # Skeletons almost always pass structural eval first-try, so without
            # these the world half would have ~0 preference signal. Each corruptor
            # must produce a real `_eval_world_structure` violation to be kept.
            w_shuffled = list(world_corruptors)
            rng.shuffle(w_shuffled)
            w_written = 0
            for axis in w_shuffled:
                if max_dpo_per_world > 0 and w_written >= max_dpo_per_world:
                    break
                made = WORLD_CORRUPTORS[axis](skeleton, rng)
                if made is None:
                    continue
                bad_skel, label = made
                if not _eval_world_structure(bad_skel, max_rooms):
                    continue  # corruption didn't actually break structure — skip
                counts["world_dpo"] += 1
                out = dirs["world_dpo"] / f"{counts['world_dpo']:04d}.jsonl"
                pair = _world_dpo(theme, [label], bad_skel, skeleton)
                pair["axis"] = axis
                _write(out, pair)
                notes.append(f"wdpo:{axis}")
                w_written += 1
        if "puzzle" in targets:
            counts["puzzle_sft"] += 1
            out = dirs["puzzle_sft"] / f"{counts['puzzle_sft']:04d}.jsonl"
            _write(out, _puzzle_sft(skeleton, full, chain_depth, min_objs))
            notes.append(f"puzzle {counts['puzzle_sft']}/{per_theme}")

            # --- Synthetic contrastive DPO: accepted world is `chosen`, a
            # deliberately degraded copy is `rejected`. Each corruptor targets a
            # single axis (solvability / difficulty / orphans / phantom / tool).
            # We only keep a pair if the corruption actually fails eval — that
            # guarantees the contrast is real, not merely asserted.
            #
            # We cap the pairs per world (max_dpo_per_world): visiting axes in a
            # seeded-shuffled order and stopping after `cap` validated pairs keeps
            # any single `chosen` world from being reinforced across all axes
            # (which would invite memorizing that story), while the rotating
            # subset keeps every axis globally balanced across the dataset.
            shuffled = list(corruptors)
            rng.shuffle(shuffled)
            written = 0
            for axis in shuffled:
                if max_dpo_per_world > 0 and written >= max_dpo_per_world:
                    break
                made = PUZZLE_CORRUPTORS[axis](full, rng)
                if made is None:
                    continue
                bad_world, label = made
                if not _corruption_is_real(axis, bad_world, chain_target, min_objs):
                    continue  # corruption didn't actually degrade — skip
                counts["puzzle_dpo"] += 1
                out = dirs["puzzle_dpo"] / f"{counts['puzzle_dpo']:04d}.jsonl"
                pair = _puzzle_dpo(
                    skeleton, [label], bad_world, full, chain_depth, min_objs
                )
                pair["axis"] = axis
                _write(out, pair)
                notes.append(f"dpo:{axis}")
                written += 1

        size = f"{len(full.rooms)}r/{len(full.objects)}o"
        via = []
        if retry:
            via.append(f"{retry} world-retry")
        if pretry:
            via.append(f"{pretry} puzzle-retry")
        via_s = f", {', '.join(via)}" if via else ""
        print(f" SFT [{', '.join(notes)}] ({size}{via_s})")

    for tgt, key in (("world", "world_sft"), ("puzzle", "puzzle_sft")):
        if tgt in targets and counts[key] < per_theme:
            print(
                f"  WARNING: {tgt} only {counts[key]}/{per_theme} accepted after "
                f"{total_attempts} attempt(s) — raise --max-total-attempts."
            )

    return counts


# ---------------------------------------------------------------------------
# Merge + manifest
# ---------------------------------------------------------------------------


def _merge(src_dirs: list[Path], out_path: Path) -> int:
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in src_dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jsonl")):
                line = p.read_text(encoding="utf-8").strip()
                if line:
                    f.write(line + "\n")
                    total += 1
    return total


def merge_all(themes: list[str], targets: set[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    plan = {
        "world": (WORLD_DIR, "world"),
        "puzzle": (PUZZLE_DIR, "puzzle"),
    }
    for tgt in ("world", "puzzle"):
        if tgt not in targets:
            continue
        base, _ = plan[tgt]
        sft_total = _merge(
            [base / "sft" / _slug(t) for t in themes], base / "sft_all.jsonl"
        )
        dpo_total = _merge(
            [base / "dpo" / _slug(t) for t in themes], base / "dpo_all.jsonl"
        )
        totals[f"{tgt}_sft"] = sft_total
        totals[f"{tgt}_dpo"] = dpo_total
        print(f"\nMerged {sft_total} {tgt} SFT examples -> {base}/sft_all.jsonl")
        print(f"Merged {dpo_total} {tgt} DPO pairs    -> {base}/dpo_all.jsonl")
    return totals


def write_manifest(
    themes: list[str],
    per_theme: int,
    model: str,
    chain_target: int,
    targets: set[str],
    totals: dict[str, int],
) -> None:
    manifest: dict = {
        "themes": themes,
        "per_theme_target": per_theme,
        "chain_depth_target": chain_target,
        "model": model,
        "targets": sorted(targets),
        "totals": totals,
    }

    print(f"\n  {'Theme':<28} ", end="")
    if "world" in targets:
        print(f"{'W-SFT':>6} {'W-DPO':>6} ", end="")
    if "puzzle" in targets:
        print(f"{'P-SFT':>6} {'P-DPO':>6}", end="")
    print()
    print(f"  {'─' * 56}")

    per_theme_counts: dict[str, dict] = {}
    for t in themes:
        row = {}
        print(f"  {t:<28} ", end="")
        if "world" in targets:
            ws = _count_existing(WORLD_DIR / "sft" / _slug(t))
            wd = _count_existing(WORLD_DIR / "dpo" / _slug(t))
            row["world_sft"], row["world_dpo"] = ws, wd
            print(f"{ws:>6} {wd:>6} ", end="")
        if "puzzle" in targets:
            ps = _count_existing(PUZZLE_DIR / "sft" / _slug(t))
            pd = _count_existing(PUZZLE_DIR / "dpo" / _slug(t))
            row["puzzle_sft"], row["puzzle_dpo"] = ps, pd
            print(f"{ps:>6} {pd:>6}", end="")
        print()
        per_theme_counts[t] = row

    manifest["per_theme_counts"] = per_theme_counts
    out = DATASET_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {'─' * 56}")
    print(f"\nManifest -> {out}")


# ---------------------------------------------------------------------------
# Stats — read-only report over an existing dataset (no generation/LLM)
# ---------------------------------------------------------------------------


def print_stats() -> None:
    """Summarise SFT/DPO counts and the per-axis breakdown of puzzle DPO pairs."""

    def _count_tree(base: Path, kind: str) -> int:
        d = base / kind
        return sum(1 for _ in d.glob("*/*.jsonl")) if d.exists() else 0

    print("Dataset statistics")
    print(f"  root: {DATASET_DIR}/\n")

    for tgt, base in (("world", WORLD_DIR), ("puzzle", PUZZLE_DIR)):
        sft = _count_tree(base, "sft")
        dpo = _count_tree(base, "dpo")
        if sft == 0 and dpo == 0:
            continue
        print(f"  {tgt:<8} SFT: {sft:>5}    DPO: {dpo:>5}")

    # Per-axis breakdown for puzzle DPO (each file carries an "axis" field;
    # live-retry pairs have none and are bucketed as 'live-retry').
    axis_counts: dict[str, int] = {}
    dpo_root = PUZZLE_DIR / "dpo"
    if dpo_root.exists():
        for p in dpo_root.glob("*/*.jsonl"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            axis_counts[rec.get("axis", "live-retry")] = (
                axis_counts.get(rec.get("axis", "live-retry"), 0) + 1
            )

    if axis_counts:
        total = sum(axis_counts.values())
        print(f"\n  puzzle DPO by axis ({total} pairs):")
        print(f"  {'axis':<18} {'count':>6} {'share':>7}")
        print(f"  {'─' * 33}")
        for axis in sorted(axis_counts, key=lambda a: -axis_counts[a]):
            n = axis_counts[axis]
            print(f"  {axis:<18} {n:>6} {n / total * 100:>6.1f}%")
    else:
        print("\n  (no puzzle DPO pairs found yet)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT + DPO fine-tuning datasets for world_builder and "
        "puzzle_builder via the live pipeline"
    )
    parser.add_argument(
        "--per-theme",
        type=int,
        default=120,
        help="accepted artifacts (SFT examples) to collect per theme (default: 120)",
    )
    parser.add_argument(
        "--target",
        choices=["world", "puzzle", "both"],
        default="both",
        help="which agent(s) to build a dataset for (default: both)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "hard", "env"],
        default="env",
        help="difficulty preset: 'easy' (HARD_MODE off — 2 rooms, no chain-depth "
        "target) or 'hard' (HARD_MODE on — uses NUM_ROOMS/CHAIN_DEPTH). Sets the "
        "mode AND routes output to dataset/<difficulty>/ so easy and hard never "
        "collide, and tags each example. 'env' (default) = honour existing "
        "HARD_MODE env and write to bare dataset/ (legacy behaviour).",
    )
    parser.add_argument(
        "--max-attempts-per-world",
        type=int,
        default=0,
        help="revision attempts per stage before discarding "
        "(default: settings.gen_max_attempts)",
    )
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=0,
        help="total pipeline attempts per theme (default: per-theme * 5)",
    )
    parser.add_argument(
        "--themes",
        nargs="+",
        default=None,
        metavar="THEME",
        help="subset of themes to run — quote multi-word names (default: all 10)",
    )
    parser.add_argument(
        "--dpo-axes",
        nargs="+",
        default=["all"],
        metavar="AXIS",
        choices=["all", "none"] + list(PUZZLE_CORRUPTORS),
        help="synthetic contrastive DPO axes for puzzle_builder: "
        f"{', '.join(PUZZLE_CORRUPTORS)} (default: all). Use 'none' to keep "
        "only opportunistic live-retry pairs.",
    )
    parser.add_argument(
        "--world-dpo-axes",
        nargs="+",
        default=["all"],
        metavar="AXIS",
        choices=["all", "none"] + list(WORLD_CORRUPTORS),
        help="synthetic contrastive DPO axes for world_builder: "
        f"{', '.join(WORLD_CORRUPTORS)} (default: all). Use 'none' to disable "
        "world-side synthetic DPO (world skeletons rarely yield live pairs).",
    )
    parser.add_argument(
        "--max-dpo-per-world",
        type=int,
        default=2,
        help="cap on synthetic DPO pairs kept per accepted world; axes are "
        "seeded-shuffled and the first N validated ones are kept "
        "(default: 2). 0 = no cap (keep every validated axis).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for synthetic corruption (default: 0, reproducible)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip complete themes; resume partially-done themes from current count",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print violation lists for every rejected attempt",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print SFT/DPO counts and per-axis DPO breakdown for the existing "
        "dataset, then exit (no generation)",
    )
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    themes = args.themes if args.themes else ALL_THEMES
    for t in themes:
        if t not in ALL_THEMES:
            print(f"Unknown theme: {t!r}. Choose from: {ALL_THEMES}")
            sys.exit(1)

    targets = {"world", "puzzle"} if args.target == "both" else {args.target}

    if "none" in args.dpo_axes:
        corruptors: list[str] = []
    elif "all" in args.dpo_axes:
        corruptors = list(PUZZLE_CORRUPTORS)
    else:
        corruptors = [a for a in args.dpo_axes if a in PUZZLE_CORRUPTORS]

    if "none" in args.world_dpo_axes:
        world_corruptors: list[str] = []
    elif "all" in args.world_dpo_axes:
        world_corruptors = list(WORLD_CORRUPTORS)
    else:
        world_corruptors = [a for a in args.world_dpo_axes if a in WORLD_CORRUPTORS]
    rng = random.Random(args.seed)

    # --difficulty: set the mode env BEFORE Settings() reads it, and route output
    # to dataset/<difficulty>/ so easy and hard runs never share files. 'env'
    # leaves both untouched (legacy: honour existing HARD_MODE, write to dataset/).
    global CURRENT_DIFFICULTY
    CURRENT_DIFFICULTY = args.difficulty
    if args.difficulty != "env":
        import os
        os.environ["HARD_MODE"] = "true" if args.difficulty == "hard" else "false"
        global DATASET_DIR, WORLD_DIR, PUZZLE_DIR
        DATASET_DIR = ROOT / "dataset" / args.difficulty
        WORLD_DIR = DATASET_DIR / "world"
        PUZZLE_DIR = DATASET_DIR / "puzzle"

    s = Settings()
    max_attempts = args.max_attempts_per_world or s.gen_max_attempts
    max_total_attempts = args.max_total_attempts or args.per_theme * 5
    max_rooms = s.num_rooms if s.hard_mode else MAX_ROOMS
    chain_depth = _default_chain_depth(s)
    chain_target = s.chain_depth if s.hard_mode else 0
    min_objs = _min_objects_per_room(s)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset generation via live two-stage pipeline")
    print(f"  difficulty          : {args.difficulty}")
    print(f"  targets             : {', '.join(sorted(targets))}")
    print(f"  themes              : {len(themes)}")
    print(f"  SFT target/theme    : {args.per_theme}")
    print(f"  max retries/stage   : {max_attempts}")
    print(f"  max total attempts  : {max_total_attempts} per theme")
    print(
        f"  hard mode           : {s.hard_mode}  (chain target: {chain_target}, "
        f"rooms: {max_rooms}, min objs/room: {min_objs})"
    )
    print(f"  model               : {s.builder_model}")
    print(
        f"  puzzle dpo axes     : {', '.join(corruptors) if corruptors else '(live-retry only)'}"
    )
    print(
        f"  world dpo axes      : {', '.join(world_corruptors) if world_corruptors else '(live-retry only)'}"
    )
    print(f"  max dpo / world     : {args.max_dpo_per_world or 'no cap'}")
    print(f"  output              : {DATASET_DIR}/")

    t0 = time.time()

    for theme in themes:
        slug = _slug(theme)
        existing_world = _count_existing(WORLD_DIR / "sft" / slug)
        existing_puzzle = _count_existing(PUZZLE_DIR / "sft" / slug)

        if args.resume:
            done = True
            if "world" in targets and existing_world < args.per_theme:
                done = False
            if "puzzle" in targets and existing_puzzle < args.per_theme:
                done = False
            if done:
                print(f"\n  [{theme}] already complete, skipping")
                continue

        generate_theme(
            theme=theme,
            per_theme=args.per_theme,
            max_attempts=max_attempts,
            max_total_attempts=max_total_attempts,
            max_rooms=max_rooms,
            chain_depth=chain_depth,
            chain_target=chain_target,
            min_objs=min_objs,
            targets=targets,
            corruptors=corruptors,
            world_corruptors=world_corruptors,
            max_dpo_per_world=args.max_dpo_per_world,
            rng=rng,
            resume_from_world=existing_world if args.resume else 0,
            resume_from_puzzle=existing_puzzle if args.resume else 0,
            debug=args.debug,
        )

    totals = merge_all(themes, targets)
    write_manifest(
        themes, args.per_theme, s.builder_model, chain_target, targets, totals
    )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
