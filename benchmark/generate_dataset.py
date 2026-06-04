"""Generate a fine-tuning dataset across all story themes using the live pipeline.

Drives the same generation + solvability check + LLM-judge retry loop that
``world_builder_node`` uses in the real game, but iterates per theme until the
target count of validated worlds is reached.

Two output formats are written in parallel:

  SFT  — instruction-tuning pairs (system + user prompt + accepted world JSON).
         Every accepted world produces one SFT example regardless of how many
         retries it took.

  DPO  — preference pairs (chosen = accepted world, rejected = a bad world that
         was corrected into it). Only produced when a world required at least one
         feedback-driven retry; each retry step that had violations becomes one
         DPO pair with the final accepted world as the `chosen` response.

Output layout:
  dataset/sft/<theme_slug>/<nnnn>.jsonl   — one SFT file per accepted world
  dataset/dpo/<theme_slug>/<nnnn>.jsonl   — one DPO file per rejection step
  dataset/sft_all.jsonl                   — merged SFT flat file
  dataset/dpo_all.jsonl                   — merged DPO flat file
  dataset/manifest.json                   — counts and run metadata

Usage:
    python -m benchmark.generate_dataset
    python -m benchmark.generate_dataset --per-theme 120
    python -m benchmark.generate_dataset --themes "Horror" "Pirate Adventure"
    python -m benchmark.generate_dataset --resume    # skip/continue partial themes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.game_master import (
    _generate_world,
    _generation_prompt,
    SYSTEM_PROMPT,
)
from agents.puzzle_builder_node import (
    _generate_puzzle,
    _generate_puzzle_with_feedback,
    _world_is_solvable,
    _world_meets_chain_depth,
    _default_chain_depth,
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
SFT_DIR = DATASET_DIR / "sft"
DPO_DIR = DATASET_DIR / "dpo"


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


def _to_sft(theme: str, world: GameWorld) -> dict:
    """Instruction-tuning pair: system + user(prompt) + assistant(world)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _generation_prompt(theme)},
            {"role": "assistant", "content": _world_json(world)},
        ]
    }


def _to_dpo(theme: str, violations: list[str], bad_world: GameWorld, good_world: GameWorld) -> dict:
    """Preference pair: chosen = accepted world, rejected = world that had violations.

    The `prompt` mirrors the exact conversation the model saw when it produced
    the bad world — system + original generation prompt + the correction message
    (if violations were present). This way the DPO trainer sees the full context
    that produced each response.
    """
    violation_block = "\n".join(f"  - {v}" for v in violations)
    correction = (
        "Your previous world had the following issues detected by an automated judge. "
        "Please generate a NEW, corrected world for the same theme that fixes ALL of them. "
        "Return only the JSON object — no prose.\n\n"
        f"Issues to fix:\n{violation_block}"
    )
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _generation_prompt(theme)},
            {"role": "user", "content": correction},
        ],
        "chosen": [{"role": "assistant", "content": _world_json(good_world)}],
        "rejected": [{"role": "assistant", "content": _world_json(bad_world)}],
        "violations": violations,
    }


def _count_existing(theme_dir: Path) -> int:
    return sum(1 for _ in theme_dir.glob("*.jsonl")) if theme_dir.exists() else 0


def _world_ok(world: GameWorld, chain_target: int) -> tuple[bool, str]:
    if not _world_is_solvable(world):
        return False, "unsolvable"
    if not _world_meets_chain_depth(world, chain_target):
        return False, f"chain depth < {chain_target}"
    return True, ""


def _get_violations(world: GameWorld) -> list[str]:
    try:
        from benchmark.narrative_eval import quick_eval_for_feedback
        return quick_eval_for_feedback(world).violations
    except Exception:
        return []


def _silent(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with stdout suppressed; return result or raise."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Per-theme generation
# ---------------------------------------------------------------------------

def generate_theme(
    theme: str,
    per_theme: int,
    max_attempts_per_world: int,
    max_total_attempts: int,
    chain_target: int,
    resume_from: int,
    debug: bool,
) -> tuple[int, int]:
    """Run the live pipeline for one theme.

    Returns (sft_kept, dpo_pairs) counts.
    """
    sft_dir = SFT_DIR / _slug(theme)
    dpo_dir = DPO_DIR / _slug(theme)
    sft_dir.mkdir(parents=True, exist_ok=True)
    dpo_dir.mkdir(parents=True, exist_ok=True)

    llm = get_llm("game_master")
    s = Settings()
    chain_depth = _default_chain_depth(s)
    seen: set[tuple] = set()
    sft_kept = resume_from
    dpo_kept = _count_existing(dpo_dir)
    total_attempts = 0

    print(f"\n{'─' * 64}")
    print(f"  Theme : {theme}  (SFT target {per_theme}, resuming from {resume_from})")
    print(f"{'─' * 64}")

    while sft_kept < per_theme and total_attempts < max_total_attempts:
        total_attempts += 1
        print(f"  [{theme}] attempt {total_attempts} ...", end="", flush=True)

        try:
            room_world, _ = _silent(_generate_world, llm, theme, 2)
            world, _ = _silent(_generate_puzzle, llm, room_world, chain_depth)
        except Exception as exc:
            print(f" generation error: {exc}")
            continue

        sig = _world_signature(world)
        if sig in seen:
            print(" duplicate, skipped")
            continue

        ok, why = _world_ok(world, chain_target)

        # --- Retry loop with DPO capture ---
        retry = 0
        while not ok and retry < max_attempts_per_world - 1:
            retry += 1
            violations = _get_violations(world)
            bad_world = world  # capture before overwriting

            print(f"\n    rejected ({why})", end="")

            if violations:
                print(f", {len(violations)} violation(s) — retrying with feedback", flush=True)
                if debug:
                    for v in violations:
                        print(f"      {v}")
                try:
                    world, _ = _silent(_generate_puzzle_with_feedback, llm, room_world, chain_depth, violations)
                except Exception as exc:
                    print(f"      feedback regen error: {exc}")
                    break
            else:
                print(" — retrying (no violations captured)", flush=True)
                try:
                    world, _ = _silent(_generate_puzzle, llm, room_world, chain_depth)
                except Exception as exc:
                    print(f"      regen error: {exc}")
                    break

            ok, why = _world_ok(world, chain_target)

            # Save DPO pair immediately if we have violations and the new world
            # is accepted — bad→good transition with a clear correction signal.
            if violations and ok:
                dpo_kept += 1
                dpo_pair = _to_dpo(theme, violations, bad_world, world)
                dpo_out = dpo_dir / f"{dpo_kept:04d}.jsonl"
                dpo_out.write_text(json.dumps(dpo_pair, ensure_ascii=False), encoding="utf-8")
                print(f"      DPO pair saved -> {dpo_out.name}")

        if not ok:
            print(f"\n    => discarded after {retry + 1} attempt(s): {why}")
            continue

        sig = _world_signature(world)
        if sig in seen:
            print(" duplicate after retry, skipped")
            continue

        seen.add(sig)
        sft_kept += 1
        sft_example = _to_sft(theme, world)
        sft_out = sft_dir / f"{sft_kept:04d}.jsonl"
        sft_out.write_text(json.dumps(sft_example, ensure_ascii=False), encoding="utf-8")
        size = f"{len(world.rooms)}r/{len(world.objects)}o"
        via = f", {retry} retries" if retry else ""
        print(f" SFT {sft_kept}/{per_theme} -> {sft_out.name} ({size}{via})")

    if sft_kept < per_theme:
        print(
            f"  WARNING: only {sft_kept}/{per_theme} accepted after "
            f"{total_attempts} attempt(s) — raise --max-total-attempts "
            "or check your prompt."
        )

    return sft_kept, dpo_kept


# ---------------------------------------------------------------------------
# Merge + manifest
# ---------------------------------------------------------------------------

def _merge(src_dirs: list[Path], out_path: Path) -> int:
    total = 0
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


def merge_all(themes: list[str]) -> None:
    sft_total = _merge(
        [SFT_DIR / _slug(t) for t in themes],
        DATASET_DIR / "sft_all.jsonl",
    )
    dpo_total = _merge(
        [DPO_DIR / _slug(t) for t in themes],
        DATASET_DIR / "dpo_all.jsonl",
    )
    print(f"\nMerged {sft_total} SFT examples  -> {DATASET_DIR}/sft_all.jsonl")
    print(f"Merged {dpo_total} DPO pairs     -> {DATASET_DIR}/dpo_all.jsonl")


def write_manifest(
    themes: list[str], per_theme: int, model: str, chain_target: int
) -> None:
    sft_counts = {t: _count_existing(SFT_DIR / _slug(t)) for t in themes}
    dpo_counts = {t: _count_existing(DPO_DIR / _slug(t)) for t in themes}
    manifest = {
        "themes": themes,
        "per_theme_target": per_theme,
        "chain_depth_target": chain_target,
        "model": model,
        "sft_counts": sft_counts,
        "dpo_counts": dpo_counts,
        "sft_total": sum(sft_counts.values()),
        "dpo_total": sum(dpo_counts.values()),
    }
    out = DATASET_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest -> {out}")
    print(f"  {'Theme':<30} {'SFT':>6}  {'DPO':>6}")
    print(f"  {'─' * 44}")
    for t in themes:
        sft_n = sft_counts[t]
        dpo_n = dpo_counts[t]
        status = "" if sft_n >= per_theme else f"  ← PARTIAL"
        print(f"  {t:<30} {sft_n:>6}  {dpo_n:>6}{status}")
    print(f"  {'─' * 44}")
    print(f"  {'TOTAL':<30} {manifest['sft_total']:>6}  {manifest['dpo_total']:>6}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT + DPO fine-tuning dataset via the live game_master pipeline"
    )
    parser.add_argument(
        "--per-theme", type=int, default=120,
        help="validated worlds (SFT examples) to collect per theme (default: 120)",
    )
    parser.add_argument(
        "--max-attempts-per-world", type=int, default=0,
        help="judge-retry attempts per world before discarding "
             "(default: settings.gen_max_attempts)",
    )
    parser.add_argument(
        "--max-total-attempts", type=int, default=0,
        help="total generation attempts per theme (default: per-theme * 5)",
    )
    parser.add_argument(
        "--themes", nargs="+", default=None, metavar="THEME",
        help="subset of themes to run — quote multi-word names (default: all 10)",
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

    s = Settings()
    max_attempts_per_world = args.max_attempts_per_world or s.gen_max_attempts
    max_total_attempts = args.max_total_attempts or args.per_theme * 5
    chain_target = s.chain_depth if s.hard_mode else 0

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    DPO_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset generation via live pipeline")
    print(f"  themes              : {len(themes)}")
    print(f"  SFT target/theme    : {args.per_theme}")
    print(f"  max retries/world   : {max_attempts_per_world}")
    print(f"  max total attempts  : {max_total_attempts} per theme")
    print(f"  hard mode           : {s.hard_mode}  (chain target: {chain_target})")
    print(f"  model               : {s.game_master_model}")
    print(f"  output              : {DATASET_DIR}/")

    t0 = time.time()

    for theme in themes:
        existing_sft = _count_existing(SFT_DIR / _slug(theme))

        if args.resume and existing_sft >= args.per_theme:
            print(f"\n  [{theme}] already complete ({existing_sft}/{args.per_theme}), skipping")
            continue

        resume_from = existing_sft if args.resume else 0

        generate_theme(
            theme=theme,
            per_theme=args.per_theme,
            max_attempts_per_world=max_attempts_per_world,
            max_total_attempts=max_total_attempts,
            chain_target=chain_target,
            resume_from=resume_from,
            debug=args.debug,
        )

    merge_all(themes)
    write_manifest(themes, args.per_theme, s.game_master_model, chain_target)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
