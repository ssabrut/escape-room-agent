"""Narrative evaluation of a generated GameWorld — LLM-as-judge + oracle.

Seven evaluation dimensions:

  1. narrative_quality     (LLM judge) — richness and coherence of scenario + room prose
  2. plot_twist            (LLM judge) — presence and quality of a twist
  3. tool_coherence        (LLM judge) — objects fit the theme; unlock relationships make sense
  4. required_tool_present (deterministic) — every requires_tool target exists, is takeable,
                              is reachable, and every requires_code has a clue producer
  5. solvability           (oracle) — heuristic policy reaches victory; measures chain depth
  6. prompt_compliance     (LLM judge) — world builder + puzzle builder output follows the generation prompt rules
  7. solution_path_validity (deterministic) — solution path references real ids; replay wins

All scores are floats in [0.0, 1.0].  The ``overall`` score is the unweighted mean.

Usage
-----
    from benchmark.narrative_eval import evaluate_world, print_report
    report = evaluate_world(world)           # GameWorld
    print_report(report)
    print_report(report, show_trace=True)    # include oracle solve trace
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import GameWorld


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DimensionResult:
    score: float  # [0.0, 1.0]
    label: str  # "PASS" | "WARN" | "FAIL"
    verdict: str = ""  # one-sentence summary from LLM (or deterministic message)
    notes: list[str] = field(default_factory=list)


@dataclass
class NarrativeEvalReport:
    narrative_quality: DimensionResult
    plot_twist: DimensionResult
    tool_coherence: DimensionResult
    required_tool_present: DimensionResult
    solvability: DimensionResult
    prompt_compliance: DimensionResult
    solution_path_validity: DimensionResult
    overall: float
    oracle_trace: list[str] = field(default_factory=list)
    chain_depth: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grade(score: float) -> str:
    if score >= 0.8:
        return "PASS"
    if score >= 0.5:
        return "WARN"
    return "FAIL"


def _parse_llm_json(text: str) -> dict:
    def _try_parse(src: str) -> dict | None:
        """Try json.loads, then repair common model artifacts and retry."""
        try:
            return json.loads(src)
        except json.JSONDecodeError:
            pass
        # Models sometimes escape single quotes as \' which is invalid JSON
        repaired = src.replace("\\'", "'")
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    # 1. Try fenced code block first
    fence = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        result = _try_parse(fence.group(1))
        if result is not None:
            return result

    # 2. Try the whole text verbatim
    result = _try_parse(text.strip())
    if result is not None:
        return result

    # 3. Extract the first balanced { ... } block (handles leading/trailing prose)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start : i + 1])
                    if result is not None:
                        return result
                    break

    return {}


def _clamp(v, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, float(v)))


def _fill(template: str, **kwargs: str) -> str:
    """Substitute {{key}} tokens in `template` — safe against literal { } in JSON examples."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", value)
    return result


# ---------------------------------------------------------------------------
# LLM judge helpers
# ---------------------------------------------------------------------------


def _get_llm():
    from config.settings import get_llm

    return get_llm("game_master")


def _judge_call(prompt_text: str) -> tuple[dict, bool]:
    """Call the LLM judge; return (parsed_dict, success)."""
    from langchain_core.messages import HumanMessage

    try:
        llm = _get_llm()
        response = llm.invoke([HumanMessage(content=prompt_text)])
        data = _parse_llm_json(response.content)
        if not data:
            print(
                f"[narrative_eval] WARNING: judge returned no parseable JSON "
                f"(response length={len(response.content)})",
                flush=True,
            )
        return data, bool(data)
    except Exception as exc:
        print(f"[narrative_eval] WARNING: judge call failed — {exc}", flush=True)
        return {}, False


# ---------------------------------------------------------------------------
# World serialisers for prompts
# ---------------------------------------------------------------------------


def _rooms_text(world: "GameWorld") -> str:
    lines = []
    for r in world.rooms:
        lines.append(f"  Room '{r.id}': {r.description}")
        if r.goal:
            lines.append(f"    Goal: {r.goal}")
    return "\n".join(lines) or "(none)"


def _clues_text(world: "GameWorld") -> str:
    lines = []
    for o in world.objects:
        parts = []
        if o.contains_info:
            parts.append(f"contains_info={o.contains_info!r}")
        if o.note:
            parts.append(f"note={o.note!r}")
        if parts:
            lines.append(f"  {o.id} ({o.description[:60]}): {', '.join(parts)}")
    return "\n".join(lines) or "(none)"


def _objects_text(world: "GameWorld") -> str:
    lines = []
    for o in world.objects:
        req_parts = []
        if o.requires_tool:
            req_parts.append(f"requires_tool={o.requires_tool}")
        if o.requires_code:
            req_parts.append(f"requires_code={o.requires_code!r}")
        if o.requires_liquid:
            req_parts.append(f"requires_liquid={o.requires_liquid}")
        if o.requires_power:
            req_parts.append(f"requires_power={o.requires_power}")
        if o.fuses:
            req_parts.append(f"fuses={list(o.fuses.keys())}")
        contains = f"contains_info={o.contains_info!r}" if o.contains_info else ""
        req_str = ", ".join(req_parts) if req_parts else "—"
        lines.append(
            f"  {o.id} | {o.location} | {o.description[:60]} | {req_str} | {contains}"
        )
    return "\n".join(lines) or "(none)"


def _solution_text(world: "GameWorld") -> str:
    return "\n".join(f"  {s}" for s in world.solution_path) or "(not provided)"


# ---------------------------------------------------------------------------
# Dimension 1 — Narrative quality (LLM judge)
# ---------------------------------------------------------------------------


def _eval_narrative_quality(world: "GameWorld") -> DimensionResult:
    from prompts import load_prompt

    prompt = _fill(
        load_prompt("narrative_eval", "narrative_quality"),
        scenario=world.scenario or "(empty)",
        objective=world.objective or "(empty)",
        rooms=_rooms_text(world),
    )
    data, ok = _judge_call(prompt)

    if not ok:
        return DimensionResult(
            score=0.5,
            label="WARN",
            verdict="LLM judge unavailable — narrative quality not assessed",
            notes=["Judge call failed; score is neutral placeholder"],
        )

    score = _clamp(data.get("score", 0.5))
    verdict = str(data.get("verdict", "")).strip()
    notes: list[str] = []
    for key in ("strengths", "weaknesses"):
        for item in data.get(key, []):
            prefix = "+" if key == "strengths" else "−"
            notes.append(f"{prefix} {item}")

    return DimensionResult(
        score=round(score, 3), label=_grade(score), verdict=verdict, notes=notes
    )


# ---------------------------------------------------------------------------
# Dimension 2 — Plot twist (LLM judge)
# ---------------------------------------------------------------------------


def _eval_plot_twist(world: "GameWorld") -> DimensionResult:
    from prompts import load_prompt

    prompt = _fill(
        load_prompt("narrative_eval", "plot_twist"),
        scenario=world.scenario or "(empty)",
        objective=world.objective or "(empty)",
        rooms=_rooms_text(world),
        clues=_clues_text(world),
        solution_path=_solution_text(world),
    )
    data, ok = _judge_call(prompt)

    if not ok:
        return DimensionResult(
            score=0.5,
            label="WARN",
            verdict="LLM judge unavailable — plot twist not assessed",
            notes=["Judge call failed; score is neutral placeholder"],
        )

    score = _clamp(data.get("score", 0.5))
    verdict = str(data.get("verdict", "")).strip()
    twist_present = bool(data.get("twist_present", False))
    twist_summary = str(data.get("twist_summary", "")).strip()

    notes: list[str] = []
    if twist_present and twist_summary and twist_summary.lower() != "none":
        notes.append(f"Twist: {twist_summary}")
    elif not twist_present:
        notes.append("No twist detected — scored neutral (0.5)")
    for item in data.get("notes", []):
        notes.append(str(item))

    return DimensionResult(
        score=round(score, 3), label=_grade(score), verdict=verdict, notes=notes
    )


# ---------------------------------------------------------------------------
# Dimension 3 — Tool coherence (LLM judge)
# ---------------------------------------------------------------------------


def _eval_tool_coherence(world: "GameWorld") -> DimensionResult:
    from prompts import load_prompt

    prompt = _fill(
        load_prompt("narrative_eval", "tool_coherence"),
        scenario=world.scenario or "(empty)",
        rooms=_rooms_text(world),
        objects=_objects_text(world),
    )
    data, ok = _judge_call(prompt)

    if not ok:
        return DimensionResult(
            score=0.5,
            label="WARN",
            verdict="LLM judge unavailable — tool coherence not assessed",
            notes=["Judge call failed; score is neutral placeholder"],
        )

    score = _clamp(data.get("score", 0.5))
    verdict = str(data.get("verdict", "")).strip()
    notes: list[str] = []
    for item in data.get("coherent_examples", []):
        notes.append(f"+ {item}")
    for item in data.get("incoherent_examples", []):
        notes.append(f"− {item}")
    for item in data.get("notes", []):
        notes.append(str(item))

    return DimensionResult(
        score=round(score, 3), label=_grade(score), verdict=verdict, notes=notes
    )


# ---------------------------------------------------------------------------
# Dimension 4 — Required tool present (deterministic)
# ---------------------------------------------------------------------------


def _eval_required_tool_present(world: "GameWorld") -> DimensionResult:
    issues: list[str] = []
    by_id = {o.id: o for o in world.objects}

    _LOCKED = {"locked", "locked_bolt", "locked_room", "hidden"}

    def _is_reachable(obj_id: str, _seen: frozenset[str] = frozenset()) -> bool:
        """True if obj is not in a dead-end locked/hidden state."""
        obj = by_id.get(obj_id)
        if obj is None or obj_id in _seen:
            return False
        if obj.state not in _LOCKED:
            return True
        # Reachable if some clue object can reveal it (by name match in contains_info)
        for o in world.objects:
            if o.contains_info and obj_id in o.contains_info:
                return True
        # Reachable if nested inside a container that is itself reachable
        if obj.location in by_id:
            return _is_reachable(obj.location, _seen | {obj_id})
        return False

    # Check requires_tool
    checked_tools = 0
    for obj in world.objects:
        tool_id = obj.requires_tool
        if not tool_id:
            continue
        checked_tools += 1
        tool = by_id.get(tool_id)
        if tool is None:
            issues.append(
                f"{obj.id}: requires_tool '{tool_id}' — object does not exist"
            )
            continue
        if not tool.takeable:
            issues.append(f"{obj.id}: tool '{tool_id}' exists but is not takeable")
        if tool.location == obj.id:
            issues.append(
                f"{obj.id}: tool '{tool_id}' is nested inside the gate it must open"
            )
        elif not _is_reachable(tool_id):
            issues.append(
                f"{obj.id}: tool '{tool_id}' is locked/hidden with no reveal path"
            )

    # Check requires_code has a clue producer
    for obj in world.objects:
        code = obj.requires_code
        if not code:
            continue
        digits_code = re.sub(r"[^0-9]", "", code)
        producers = [
            o
            for o in world.objects
            if o.contains_info
            and (
                o.contains_info == code
                or code in o.contains_info
                or (
                    digits_code
                    and digits_code == re.sub(r"[^0-9]", "", o.contains_info)
                )
            )
        ]
        if not producers:
            issues.append(
                f"{obj.id}: requires_code '{code}' — no clue object produces this token"
            )

    if checked_tools == 0 and not any(o.requires_code for o in world.objects):
        return DimensionResult(
            score=1.0,
            label="PASS",
            verdict="No tool or code requirements to validate",
            notes=["World has no gated interactions — trivially passes"],
        )

    penalty = min(1.0, 0.2 * len(issues))
    score = max(0.0, 1.0 - penalty)

    verdict = (
        f"All {checked_tools} tool requirement(s) valid; all codes have clue producers"
        if not issues
        else f"{len(issues)} structural issue(s) found"
    )
    return DimensionResult(
        score=round(score, 3),
        label=_grade(score),
        verdict=verdict,
        notes=(
            issues
            if issues
            else ["All requires_tool and requires_code references resolve correctly"]
        ),
    )


# ---------------------------------------------------------------------------
# Dimension 5 — Solvability (oracle)
# ---------------------------------------------------------------------------


def _eval_solvability(world: "GameWorld") -> tuple[DimensionResult, list[str], int]:
    """Run the heuristic oracle. Returns (result, trace, chain_depth)."""
    if not world.win_condition.object_id or not world.rooms:
        return (
            DimensionResult(
                score=0.0,
                label="FAIL",
                verdict="No win condition or rooms defined",
                notes=["World cannot be solved — missing win target"],
            ),
            [],
            0,
        )

    try:
        from benchmark.engine import HeadlessEpisode
        from benchmark.policies import heuristic_policy
    except Exception as exc:
        return (
            DimensionResult(
                score=0.5,
                label="WARN",
                verdict=f"Benchmark harness unavailable ({exc})",
                notes=["Solvability unknown — harness import failed"],
            ),
            [],
            0,
        )

    result = HeadlessEpisode(world).run(heuristic_policy, record_history=True)
    notes: list[str] = []

    if result.victory:
        score = 1.0
        if result.chain_depth == 0:
            score = 0.8
            notes.append(
                "Chain depth 0 — puzzle may be trivially short or repair flattened it"
            )
        elif result.chain_depth == 1:
            score = 0.9
            notes.append(
                "Chain depth 1 — shallow puzzle (consider more dependent steps)"
            )
        notes.append(
            f"Solved in {result.ticks} tick(s), "
            f"{result.objects_resolved} object(s) resolved, "
            f"rooms visited: {result.rooms_visited}"
        )
        verdict = (
            f"Oracle wins in {result.ticks} tick(s), chain depth {result.chain_depth}"
        )
    else:
        score = 0.0
        verdict = f"Oracle FAILED to win in {result.ticks} tick(s)"
        notes.append(
            f"Last room: {result.last_room}, "
            f"win object state: {result.win_object_state!r} "
            f"(target: {world.win_condition.state!r})"
        )

    return (
        DimensionResult(
            score=round(score, 3), label=_grade(score), verdict=verdict, notes=notes
        ),
        result.history,
        result.chain_depth,
    )


# ---------------------------------------------------------------------------
# Dimension 6 — Prompt compliance (LLM judge)
# ---------------------------------------------------------------------------


def _world_json_compact(world: "GameWorld") -> str:
    """Serialize only rule-relevant fields for the compliance judge.

    Omits description/notes prose to keep prompt length under the model's
    context budget — the judge checks structure and references, not prose.
    """

    def _obj_fields(o) -> dict:
        d: dict = {
            "id": o.id,
            "location": o.location,
            "state": o.state,
            "interactable": o.interactable,
            "takeable": o.takeable,
        }
        if o.scenic:
            d["scenic"] = True
        if o.requires_code:
            d["requires_code"] = o.requires_code
        if o.requires_tool:
            d["requires_tool"] = o.requires_tool
        if o.requires_liquid:
            d["requires_liquid"] = o.requires_liquid
        if o.requires_power:
            d["requires_power"] = o.requires_power
        if o.fuses:
            d["fuses"] = o.fuses
        if o.contains_info:
            d["contains_info"] = o.contains_info
        return d

    data = {
        "rooms": [
            {
                "id": r.id,
                "adjacency": r.adjacency,
                "goal": r.goal,
                "goal_completion": (
                    r.goal_completion.model_dump(exclude_none=True)
                    if r.goal_completion
                    else None
                ),
                "key_objects": r.key_objects,
                "object_count": sum(1 for o in world.objects if o.location == r.id),
                "scenic_count": sum(
                    1 for o in world.objects if o.location == r.id and o.scenic
                ),
            }
            for r in world.rooms
        ],
        "objects": [_obj_fields(o) for o in world.objects],
        "win_condition": world.win_condition.model_dump(),
        "solution_path": world.solution_path,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


_STANDARD_RULES_SUMMARY = """\
STANDARD (2-room) MODE RULES:
- Exactly 2 rooms in a linear chain, connected by a single mirrored direction pair.
- First room's goal_completion gates entry to the second room.
- Second room's goal_completion must equal the win_condition (same object_id + state).
- Every open/unlock goal_completion must use state exactly "unlocked" (no synonyms).
- Each room has 5–10 objects (puzzle pieces + 2–4 scenic props).
- Scenic objects: scenic=true, interactable=false, takeable=false, no contains_info.
- Every requires_code must be produced by a contains_info token on some object.
- Every requires_tool must reference a takeable=true object in a reachable (non-locked) state.
- No circular tool dependencies.
- No dangling clues (every contains_info must be consumed downstream).
- No duplicate clues (each info token on exactly one object).
- solution_path must not contain literal codes/combinations (use "the hidden code").
- solution_path references only real room and object ids.
- Goal prose must match goal_completion: if completion is known_info, goal says "discover the hidden clue".
"""

_HARD_RULES_SUMMARY = """\
HARD (multi-room) MODE RULES:
- Linear chain of N rooms; first room is start, last room is the win room.
- Each non-final room's goal_completion gates passage to the next room.
- Final room's goal_completion equals the win_condition.
- Every open/unlock goal_completion must use state exactly "unlocked".
- Each room has 5–10 objects (puzzle pieces + scenic props).
- Scenic objects: scenic=true, interactable=false, takeable=false, no contains_info.
- Decoy notes/items must NOT carry contains_info tokens nothing uses.
- Every requires_code produced by a contains_info token; every requires_tool is takeable + reachable.
- No circular tool deps; no dangling clues; no duplicate clues.
- solution_path must not contain literal codes.
"""


def _instructions_for_world(world: "GameWorld") -> str:
    """Return a concise rule summary for this world (standard vs hard mode)."""
    if len(world.rooms) == 2:
        return _STANDARD_RULES_SUMMARY
    return _HARD_RULES_SUMMARY.replace("N", str(len(world.rooms)))


def _eval_prompt_compliance(world: "GameWorld") -> DimensionResult:
    """Judge how well the world_builder + puzzle_builder output follows the generation instructions."""
    from prompts import load_prompt

    instructions = _instructions_for_world(world)
    world_json = _world_json_compact(world)

    prompt = _fill(
        load_prompt("narrative_eval", "prompt_compliance"),
        instructions=instructions,
        world=world_json,
    )
    if len(prompt) > 12_000:
        print(
            f"[narrative_eval] WARNING: compliance prompt is {len(prompt)} chars "
            f"— may exceed model context",
            flush=True,
        )
    data, ok = _judge_call(prompt)

    if not ok:
        return DimensionResult(
            score=0.5,
            label="WARN",
            verdict="LLM judge unavailable — prompt compliance not assessed",
            notes=["Judge call failed; score is neutral placeholder"],
        )

    score = _clamp(data.get("score", 0.5))
    verdict = str(data.get("verdict", "")).strip()
    notes: list[str] = []

    for v in data.get("violations", []):
        rule = v.get("rule", "?")
        desc = v.get("description", "")
        affected = v.get("affected", "")
        notes.append(f"Rule {rule} violated ({affected}): {desc}")

    for p in data.get("passes", []):
        notes.append(f"✓ {p}")

    return DimensionResult(
        score=round(score, 3), label=_grade(score), verdict=verdict, notes=notes
    )


# ---------------------------------------------------------------------------
# Dimension 7 — Solution path validity (deterministic)
# ---------------------------------------------------------------------------


def _eval_solution_path(
    world: "GameWorld", oracle_trace: list[str] | None = None
) -> DimensionResult:
    """Check solution path object references and replay-solvability.

    Two deterministic checks, followed by an LLM judge when issues are found:

    A) Object/room reference validity — every object or room id mentioned by
       name in the solution_path steps must actually exist in the world.
       Hallucinated ids (ids that existed in the LLM's raw output but were
       dropped by the repair pipeline) are caught here.

    B) Replay solvability — the oracle already confirmed the world is winnable
       (dimension 5). Here we go further: we replay the solution_path steps
       themselves (parsed as engine actions) through the headless engine to
       check whether the GM-authored path actually solves the world. If the
       path is incomplete or uses wrong ids, the replay fails even though the
       oracle wins via a different route.

    When either check fails, an LLM judge is called with the oracle trace as
    ground truth to produce specific, actionable GM feedback.
    """
    notes: list[str] = []
    issues: list[str] = []

    object_ids = {o.id for o in world.objects}
    room_ids = {r.id for r in world.rooms}
    all_ids = object_ids | room_ids

    path = world.solution_path
    if not path:
        return DimensionResult(
            score=0.5,
            label="WARN",
            verdict="No solution path provided",
            notes=["solution_path is empty — cannot validate"],
        )

    # --- A: object reference check ---
    path_text = " ".join(path)

    # Find ids mentioned as words in the path that DON'T resolve
    # Strategy: tokenise path steps into snake_case-like tokens and check each
    tokens = set(re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", path_text.lower()))
    ghost_ids = (
        tokens - object_ids - room_ids - {"the_hidden", "hidden_code", "solution_path"}
    )
    # Only flag tokens that look like real ids (not common words run together)
    plausible_ghosts = {t for t in ghost_ids if len(t) > 6 and "_" in t}
    if plausible_ghosts:
        issues.append(
            f"solution_path references id-like tokens not in world: "
            f"{', '.join(sorted(plausible_ghosts)[:5])}"
        )

    # --- B: replay solvability ---
    # Parse solution_path steps into candidate engine actions.
    # Each step is e.g. "1. examine clue_note in room_1 to obtain the code"
    # We look for known action verbs followed by a known id.
    _VERBS = (
        "examine",
        "take",
        "use_tool",
        "enter_code",
        "go",
        "flip_fuse",
        "insert_liquid",
        "open",
    )
    candidate_actions: list[str] = []
    for step in path:
        step_lower = step.lower()
        for verb in _VERBS:
            # find verb then nearest known id on the same step
            if verb not in step_lower:
                continue
            for id_ in all_ids:
                if id_.lower() in step_lower:
                    candidate_actions.append(f"{verb} {id_}")
                    break

    replay_ok: bool | None = None
    if candidate_actions:
        try:
            from benchmark.generate_bank import _replay_wins

            replay_ok = _replay_wins(world, candidate_actions)
            if replay_ok:
                notes.append(
                    f"Solution path replay succeeded ({len(candidate_actions)} parsed step(s))"
                )
            else:
                issues.append(
                    f"Solution path replay FAILED — parsed {len(candidate_actions)} step(s) "
                    f"but they do not reach victory (path may be incomplete or use wrong ids)"
                )
        except Exception as exc:
            notes.append(f"Replay skipped — harness unavailable ({exc})")
    else:
        issues.append(
            "No parseable engine actions found in solution_path — steps must use engine "
            "verbs (examine, take, use_tool, enter_code, go, flip_fuse, insert_liquid) "
            "followed by a real object or room id so the replay can verify the path"
        )

    # --- Score ---
    # Start at 1.0, deduct for ghost ids and failed/unparseable replay
    score = 1.0
    if plausible_ghosts:
        score -= 0.3
    if replay_ok is False or (replay_ok is None and not candidate_actions):
        score -= 0.4
    score = max(0.0, round(score, 3))

    # --- LLM judge for detailed GM feedback when issues exist ---
    if issues:
        llm_result = _eval_solution_path_llm(
            world=world,
            path=path,
            oracle_trace=oracle_trace or [],
            candidate_actions=candidate_actions,
            replay_ok=replay_ok,
        )
        if llm_result is not None:
            return llm_result

    if not issues:
        verdict = "All solution path references exist; path replay confirms solvability"
        if replay_ok is None:
            verdict = (
                "Object references look valid; replay skipped (no parseable steps)"
            )
    else:
        verdict = f"{len(issues)} solution path issue(s) found"
        notes = issues + notes

    return DimensionResult(
        score=score, label=_grade(score), verdict=verdict, notes=notes
    )


def _eval_solution_path_llm(
    world: "GameWorld",
    path: list[str],
    oracle_trace: list[str],
    candidate_actions: list[str],
    replay_ok: bool | None,
) -> "DimensionResult | None":
    """LLM judge for solution_path issues — returns detailed GM feedback or None on failure."""
    from prompts import load_prompt

    world_summary = json.dumps(
        {
            "rooms": [
                {
                    "id": r.id,
                    "goal": r.goal,
                    "goal_completion": (
                        r.goal_completion.model_dump(exclude_none=True)
                        if r.goal_completion
                        else None
                    ),
                }
                for r in world.rooms
            ],
            "objects": [
                {
                    "id": o.id,
                    "location": o.location,
                    "state": o.state,
                    "takeable": o.takeable,
                    "requires_tool": o.requires_tool,
                    "requires_code": o.requires_code,
                    "contains_info": o.contains_info,
                }
                for o in world.objects
            ],
            "win_condition": world.win_condition.model_dump(),
        },
        indent=2,
        ensure_ascii=False,
    )

    replay_result_str = (
        "PASS — replay reached victory"
        if replay_ok is True
        else (
            "FAIL — replay did not reach victory"
            if replay_ok is False
            else "SKIPPED — no parseable engine actions found in solution path"
        )
    )

    prompt = _fill(
        load_prompt("narrative_eval", "solution_path"),
        world=world_summary,
        solution_path="\n".join(path),
        oracle_trace=(
            "\n".join(oracle_trace) if oracle_trace else "(oracle trace not available)"
        ),
        parsed_actions=(
            "\n".join(candidate_actions) if candidate_actions else "(none extracted)"
        ),
        replay_result=replay_result_str,
    )

    data, ok = _judge_call(prompt)
    if not ok:
        return None

    score = _clamp(data.get("score", 0.5))
    verdict = str(data.get("verdict", "")).strip()
    feedback_notes: list[str] = []
    for item in data.get("gm_feedback", []):
        feedback_notes.append(f"GM fix needed: {item}")
    for item in data.get("passes", []):
        feedback_notes.append(f"✓ {item}")

    return DimensionResult(
        score=round(score, 3),
        label=_grade(score),
        verdict=verdict,
        notes=feedback_notes,
    )


# ---------------------------------------------------------------------------
# Fast feedback eval (used by game_master retry loop — no extra LLM calls)
# ---------------------------------------------------------------------------


@dataclass
class QuickEvalResult:
    """Lightweight result from the deterministic-only eval pass.

    Contains only the dimensions that are fast enough to run inside the
    generation retry loop without adding significant latency:
      - required_tool_present  (deterministic)
      - solvability            (oracle, already runs in puzzle_builder_node)
      - prompt_compliance      (LLM judge — included because violations are
                                 the most actionable feedback for the GM)

    ``violations`` is a flat list of human-readable strings extracted from
    whichever dimensions flagged issues.  ``score`` is the mean of the three
    dimension scores.
    """

    score: float
    solvable: bool
    violations: list[str]


def quick_eval_for_feedback(world: "GameWorld") -> QuickEvalResult:
    """Run the three dimensions most useful as GM retry feedback.

    Intentionally skips narrative_quality, plot_twist, and tool_coherence
    (subjective LLM judges) to stay fast inside the generation loop.
    Returns a QuickEvalResult with a ``violations`` list ready to paste
    into a correction prompt.
    """
    violations: list[str] = []

    tool = _eval_required_tool_present(world)
    for note in tool.notes:
        if not note.startswith("All requires"):
            violations.append(f"[tool] {note}")

    solvability, _, _ = _eval_solvability(world)
    solvable = solvability.score >= 0.8
    if not solvable:
        violations.append(f"[solvability] {solvability.verdict}")

    compliance = _eval_prompt_compliance(world)
    if compliance.label != "PASS":
        for note in compliance.notes:
            if note.startswith("Rule"):
                violations.append(f"[compliance] {note}")

    scores = [tool.score, solvability.score, compliance.score]
    return QuickEvalResult(
        score=round(sum(scores) / len(scores), 3),
        solvable=solvable,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_world(world: "GameWorld") -> NarrativeEvalReport:
    """Evaluate all six dimensions and return a NarrativeEvalReport."""
    print("[narrative_eval] judging narrative quality...", flush=True)
    narr = _eval_narrative_quality(world)

    print("[narrative_eval] judging plot twist...", flush=True)
    twist = _eval_plot_twist(world)

    print("[narrative_eval] judging tool coherence...", flush=True)
    coherence = _eval_tool_coherence(world)

    print("[narrative_eval] checking required tool presence...", flush=True)
    tool_present = _eval_required_tool_present(world)

    print("[narrative_eval] running oracle solvability check...", flush=True)
    solvability, trace, depth = _eval_solvability(world)

    print("[narrative_eval] judging prompt compliance...", flush=True)
    compliance = _eval_prompt_compliance(world)

    print("[narrative_eval] checking solution path validity...", flush=True)
    solution_path = _eval_solution_path(world, oracle_trace=trace)

    scores = [
        narr.score,
        twist.score,
        coherence.score,
        tool_present.score,
        solvability.score,
        compliance.score,
        solution_path.score,
    ]
    overall = round(sum(scores) / len(scores), 3)

    return NarrativeEvalReport(
        narrative_quality=narr,
        plot_twist=twist,
        tool_coherence=coherence,
        required_tool_present=tool_present,
        solvability=solvability,
        prompt_compliance=compliance,
        solution_path_validity=solution_path,
        overall=overall,
        oracle_trace=trace,
        chain_depth=depth,
    )


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_W = 94
_COL = 28


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.2f}"


def _banner(title: str, char: str = "=") -> None:
    line = char * _W
    pad = max(0, _W - len(title) - 4)
    print("\n" + line)
    print(f"{char}{char} {title}{' ' * pad}{char}{char}")
    print(line)


def _section(dim: str, result: DimensionResult) -> None:
    print(f"\n  {dim:<{_COL}} {_bar(result.score)}  [{result.label}]")
    if result.verdict:
        print(f"    {result.verdict}")
    for note in result.notes:
        print(f"    • {note}")


def report_to_dict(report: NarrativeEvalReport) -> dict:
    """Serialise a NarrativeEvalReport to a plain dict (JSON-safe)."""

    def _dim(d: DimensionResult) -> dict:
        return {
            "score": d.score,
            "label": d.label,
            "verdict": d.verdict,
            "notes": d.notes,
        }

    return {
        "overall": report.overall,
        "chain_depth": report.chain_depth,
        "dimensions": {
            "narrative_quality": _dim(report.narrative_quality),
            "plot_twist": _dim(report.plot_twist),
            "tool_coherence": _dim(report.tool_coherence),
            "required_tool_present": _dim(report.required_tool_present),
            "solvability": _dim(report.solvability),
            "prompt_compliance": _dim(report.prompt_compliance),
            "solution_path_validity": _dim(report.solution_path_validity),
        },
        "oracle_trace": report.oracle_trace,
    }


def write_report(report: NarrativeEvalReport, path: Path) -> None:
    """Write the report as JSON to *path*."""
    path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_report(report: NarrativeEvalReport, show_trace: bool = False) -> None:
    """Print a formatted evaluation report to stdout."""
    _banner("NARRATIVE EVALUATION REPORT")

    print(f"\n  {'OVERALL':<{_COL}} {_bar(report.overall)}  [{_grade(report.overall)}]")
    print(f"  {'Chain depth':<{_COL}} {report.chain_depth} ordered dependency link(s)")

    _banner("Dimension Scores", char="-")
    _section("1. Narrative quality", report.narrative_quality)
    _section("2. Plot twist", report.plot_twist)
    _section("3. Tool coherence", report.tool_coherence)
    _section("4. Required tool present", report.required_tool_present)
    _section("5. Solvability (oracle)", report.solvability)
    _section("6. Prompt compliance", report.prompt_compliance)
    _section("7. Solution path validity", report.solution_path_validity)

    if show_trace:
        _banner("Oracle Solve Trace", char="-")
        if report.oracle_trace:
            for line in report.oracle_trace:
                print(f"  {line}")
        else:
            print("  (no trace recorded)")

    print()
