"""Generate fine-tuning datasets for the two generation agents via the live pipeline.

The real game runs world generation in two stages, each owned by a different agent:

  world_builder  (agents.game_master)
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

from agents.game_master import (
    _generate_world,
    _generation_prompt,
    _eval_world_structure,
    SYSTEM_PROMPT as WORLD_SYSTEM_PROMPT,
    MAX_ROOMS,
)
from agents.puzzle_builder_node import (
    _generate_puzzle,
    _generate_puzzle_with_feedback,
    _build_prompt as _puzzle_prompt,
    _eval_puzzle,
    _default_chain_depth,
    _min_objects_per_room,
    SYSTEM_PROMPT as PUZZLE_SYSTEM_PROMPT,
)
from config.settings import Settings, get_llm
from state import GameWorld

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
        f"Your previous {kind} had the following issues detected by an automated "
        f"judge. Please generate a NEW, corrected {kind} that fixes ALL of them. "
        "Return only the JSON object — no prose.\n\n"
        f"Issues to fix:\n{block}"
    )


# ---------------------------------------------------------------------------
# Contrastive corruptors — turn an ACCEPTED (solvable + deep) world into a
# deliberately WORSE one on a single, named axis. The accepted world is always
# `chosen`; the corrupted copy is `rejected`. Because both share the identical
# prompt, the only signal DPO can learn is the defect itself — never the story.
#
# Each corruptor returns (corrupted_world, violation_label) or None when it
# cannot apply to this particular world.
# ---------------------------------------------------------------------------


from state import WorldObject


def _clone(world: GameWorld) -> GameWorld:
    return world.model_copy(deep=True)


def _corrupt_unsolvable(world: GameWorld, rng: random.Random) -> tuple[GameWorld, str] | None:
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
    return bad, f"win object '{win_id}' is locked with no unlock mechanism — world is unsolvable"


def _corrupt_shallow(world: GameWorld, rng: random.Random) -> tuple[GameWorld, str] | None:
    """Flatten the difficulty: open every gate so the win object is reachable in
    one step. Stays solvable but collapses chain depth — the 'easy' contrast.

    Returns None if there is nothing gated to flatten (already trivially easy).
    """
    bad = _clone(world)
    opened = 0
    for o in bad.objects:
        had_gate = any([
            o.state in {"locked", "locked_bolt", "locked_room", "hidden"},
            o.requires_code, o.requires_tool, o.requires_liquid,
            o.requires_power, o.fuses,
        ])
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
    return bad, f"all {opened} gate(s) removed — world is solvable but trivially shallow (no puzzle chain)"


def _corrupt_orphans(world: GameWorld, rng: random.Random) -> tuple[GameWorld, str] | None:
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
        bad.objects.append(WorldObject(
            id=oid,
            location=room.id,
            description="An ornate relic that seems important but connects to nothing.",
            state="locked",
            interactable=True,
            takeable=False,
            requires_code="9981",  # a code no object ever reveals
        ))
        added.append(oid)
    return bad, (
        f"objects {', '.join(added)} are uncorrelated — gated by a code/clue that "
        "nothing in the world produces and never used by the solution"
    )


def _corrupt_phantom_solution(world: GameWorld, rng: random.Random) -> tuple[GameWorld, str] | None:
    """Inject solution_path steps that reference object ids which do not exist.

    Re-introduces what `_scrub_ghost_ids` removes: a narrated step naming a
    phantom object id. Returns None if the world has no solution_path.
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


# Puzzle-stage corruptors keyed by axis name (CLI-selectable).
PUZZLE_CORRUPTORS = {
    "unsolvable": _corrupt_unsolvable,
    "shallow": _corrupt_shallow,
    "orphans": _corrupt_orphans,
    "phantom": _corrupt_phantom_solution,
}


def _corruption_is_real(axis: str, bad: GameWorld, chain_target: int, min_objs: int) -> bool:
    """Confirm a corrupted world is genuinely worse on its axis before keeping it.

    We do not trust the corruptor's claim blindly: the `rejected` example must be
    demonstrably degraded, or the DPO pair teaches nothing. Validation differs by
    axis because `_eval_puzzle` models solvability/depth but NOT solution-text or
    orphan defects (those are silently repaired at build time, never evaluated).
    """
    if axis == "unsolvable":
        # Oracle must now fail end-to-end.
        return any(_is_oracle_failure(i) for i in _eval_puzzle(bad, chain_target, min_objs))
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
    return False


def _is_oracle_failure(issue: str) -> bool:
    return issue.startswith("oracle failed to win") or "no win condition" in issue


def _has_phantom_solution_ids(world: GameWorld) -> bool:
    valid = {o.id for o in world.objects} | {r.id for r in world.rooms}
    for step in world.solution_path:
        for tok in re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", step):
            if tok not in valid and tok not in {"the_hidden", "hidden_code", "the_object", "solution_path"}:
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
            (info and (re.sub(r"[^0-9]", "", info) == re.sub(r"[^0-9]", "", code) and code))
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


def _world_dpo(theme: str, violations: list[str], bad: GameWorld, good: GameWorld) -> dict:
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


def _puzzle_sft(skeleton: GameWorld, full: GameWorld, chain_depth: int, min_objs: int) -> dict:
    """puzzle_builder SFT: rooms skeleton prompt -> objects + solution path."""
    return {
        "messages": [
            {"role": "system", "content": PUZZLE_SYSTEM_PROMPT},
            {"role": "user", "content": _puzzle_prompt(skeleton, chain_depth, min_objs)},
            {"role": "assistant", "content": _world_json(full)},
        ]
    }


def _puzzle_dpo(
    skeleton: GameWorld, violations: list[str], bad: GameWorld, good: GameWorld,
    chain_depth: int, min_objs: int,
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


def _write(path: Path, payload: dict) -> None:
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
                _write(out, _world_dpo(theme, _eval_world_structure(bad_world, max_rooms), bad_world, world))

        if world_issues:
            print(f"\n    => world discarded after {retry + 1} attempt(s): "
                  f"{len(world_issues)} issue(s)")
            continue

        skeleton = world  # rooms-only; keep a clean copy for the puzzle prompt

        # --- Stage 2: puzzle_builder, with eval-B retry + DPO capture ---
        try:
            full, _ = _silent(_generate_puzzle, llm, skeleton, chain_depth, min_objs)
        except Exception as exc:
            print(f" puzzle gen error: {exc}")
            continue

        puzzle_issues = _eval_puzzle(full, chain_target, min_objs)
        pretry = 0
        while puzzle_issues and pretry < max_attempts - 1:
            pretry += 1
            bad_full = full
            if debug:
                for v in puzzle_issues:
                    print(f"\n      puzzle: {v}", end="")
            try:
                full, _ = _silent(
                    _generate_puzzle_with_feedback,
                    llm, skeleton, chain_depth, min_objs, puzzle_issues,
                )
            except Exception as exc:
                print(f" puzzle regen error: {exc}")
                break
            prev_issues = puzzle_issues
            puzzle_issues = _eval_puzzle(full, chain_target, min_objs)
            if "puzzle" in targets and not puzzle_issues:
                counts["puzzle_dpo"] += 1
                out = dirs["puzzle_dpo"] / f"{counts['puzzle_dpo']:04d}.jsonl"
                _write(out, _puzzle_dpo(skeleton, prev_issues, bad_full, full, chain_depth, min_objs))

        if puzzle_issues:
            print(f"\n    => puzzle discarded after {pretry + 1} attempt(s): "
                  f"{len(puzzle_issues)} issue(s)")
            continue

        sig = _world_signature(full)
        if sig in seen:
            print(" duplicate, skipped")
            continue
        seen.add(sig)

        # --- Both stages accepted: write SFT examples for enabled targets ---
        notes = []
        if "world" in targets:
            counts["world_sft"] += 1
            out = dirs["world_sft"] / f"{counts['world_sft']:04d}.jsonl"
            _write(out, _world_sft(theme, skeleton))
            notes.append(f"world {counts['world_sft']}/{per_theme}")
        if "puzzle" in targets:
            counts["puzzle_sft"] += 1
            out = dirs["puzzle_sft"] / f"{counts['puzzle_sft']:04d}.jsonl"
            _write(out, _puzzle_sft(skeleton, full, chain_depth, min_objs))
            notes.append(f"puzzle {counts['puzzle_sft']}/{per_theme}")

            # --- Synthetic contrastive DPO: accepted world is `chosen`, a
            # deliberately degraded copy is `rejected`. Each corruptor targets a
            # single axis (solvability / difficulty / orphans / phantom ids).
            # We only keep a pair if the corruption actually fails eval — that
            # guarantees the contrast is real, not merely asserted.
            for axis in corruptors:
                made = PUZZLE_CORRUPTORS[axis](full, rng)
                if made is None:
                    continue
                bad_world, label = made
                if not _corruption_is_real(axis, bad_world, chain_target, min_objs):
                    continue  # corruption didn't actually degrade — skip
                counts["puzzle_dpo"] += 1
                out = dirs["puzzle_dpo"] / f"{counts['puzzle_dpo']:04d}.jsonl"
                pair = _puzzle_dpo(skeleton, [label], bad_world, full, chain_depth, min_objs)
                pair["axis"] = axis
                _write(out, pair)
                notes.append(f"dpo:{axis}")

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
    themes: list[str], per_theme: int, model: str, chain_target: int,
    targets: set[str], totals: dict[str, int],
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT + DPO fine-tuning datasets for world_builder and "
                    "puzzle_builder via the live pipeline"
    )
    parser.add_argument(
        "--per-theme", type=int, default=120,
        help="accepted artifacts (SFT examples) to collect per theme (default: 120)",
    )
    parser.add_argument(
        "--target", choices=["world", "puzzle", "both"], default="both",
        help="which agent(s) to build a dataset for (default: both)",
    )
    parser.add_argument(
        "--max-attempts-per-world", type=int, default=0,
        help="judge-retry attempts per stage before discarding "
             "(default: settings.gen_max_attempts)",
    )
    parser.add_argument(
        "--max-total-attempts", type=int, default=0,
        help="total pipeline attempts per theme (default: per-theme * 5)",
    )
    parser.add_argument(
        "--themes", nargs="+", default=None, metavar="THEME",
        help="subset of themes to run — quote multi-word names (default: all 10)",
    )
    parser.add_argument(
        "--dpo-axes", nargs="+", default=["all"], metavar="AXIS",
        choices=["all", "none"] + list(PUZZLE_CORRUPTORS),
        help="synthetic contrastive DPO axes for puzzle_builder: "
             f"{', '.join(PUZZLE_CORRUPTORS)} (default: all). Use 'none' to keep "
             "only opportunistic live-retry pairs.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for synthetic corruption (default: 0, reproducible)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip complete themes; resume partially-done themes from current count",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print violation lists for every rejected attempt",
    )
    args = parser.parse_args()

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
    rng = random.Random(args.seed)

    s = Settings()
    max_attempts = args.max_attempts_per_world or s.gen_max_attempts
    max_total_attempts = args.max_total_attempts or args.per_theme * 5
    max_rooms = s.num_rooms if s.hard_mode else MAX_ROOMS
    chain_depth = _default_chain_depth(s)
    chain_target = s.chain_depth if s.hard_mode else 0
    min_objs = _min_objects_per_room(s)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset generation via live two-stage pipeline")
    print(f"  targets             : {', '.join(sorted(targets))}")
    print(f"  themes              : {len(themes)}")
    print(f"  SFT target/theme    : {args.per_theme}")
    print(f"  max retries/stage   : {max_attempts}")
    print(f"  max total attempts  : {max_total_attempts} per theme")
    print(f"  hard mode           : {s.hard_mode}  (chain target: {chain_target}, "
          f"rooms: {max_rooms}, min objs/room: {min_objs})")
    print(f"  model               : {s.game_master_model}")
    print(f"  dpo contrast axes   : {', '.join(corruptors) if corruptors else '(live-retry only)'}")
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
            rng=rng,
            resume_from_world=existing_world if args.resume else 0,
            resume_from_puzzle=existing_puzzle if args.resume else 0,
            debug=args.debug,
        )

    totals = merge_all(themes, targets)
    write_manifest(themes, args.per_theme, s.game_master_model, chain_target, targets, totals)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
