"""Storyboard Builder agent — generates the human narrative layer for a finished world.

world_builder produces the rooms/scenario, puzzle_builder populates objects and
the solution path. This node receives the fully assembled GameWorld and
generates a Storyboard: plot, adapted personas, room stories, discovery beats,
and — only for mystery themes — the victim/killer/suspects/solution.

Generation runs in up to three focused LLM passes instead of one giant JSON
document:
  1. core   — mystery / solution / suspects (mystery themes only)
  2. beats  — discovery_beats / conversation_seeds / ending_guidance
  3. flavor — plot / adapted_personas / room_stories / phase_guidance

A local model drops random fields when asked for everything at once (attention
degrades as output grows). Each pass is small enough that critical fields
cannot get lost, and passes 2-3 receive pass 1's decisions as immutable
CASE_FACTS so they cannot contradict them.

Degrades gracefully: any LLM/parse failure for a pass leaves that section
empty and relies on _repair_mystery / _ensure_personas to backfill it, so the
pipeline never fails because of this node.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.escape_rooms.utils.settings import get_llm
from src.escape_rooms.utils.logging import get_node_logger
from src.escape_rooms.prompts import load_prompt
from src.escape_rooms.state import (
    Character,
    GameState,
    GameWorld,
    Storyboard,
    StoryboardMystery,
    StoryboardPersona,
    StoryboardSolution,
)

log = get_node_logger("storyboard_builder")

CORE_SYSTEM_PROMPT = load_prompt("storyboard_builder", "system_core")
CORE_GENERATION_PROMPT = load_prompt("storyboard_builder", "generation_core")
BEATS_SYSTEM_PROMPT = load_prompt("storyboard_builder", "system_beats")
BEATS_GENERATION_PROMPT = load_prompt("storyboard_builder", "generation_beats")
FLAVOR_SYSTEM_PROMPT = load_prompt("storyboard_builder", "system_flavor")
FLAVOR_GENERATION_PROMPT = load_prompt("storyboard_builder", "generation_flavor")

# Object ids with these prefixes are scenic/filler — never plot-critical.
_NON_PLOT_PREFIXES = ("scenic_", "gate_", "filler_")

# Forbidden vocabulary words indicating genre contamination from a tech/mechanical role.
_CONTAMINATION_WORDS = frozenset({
    "system", "power", "restore", "circuit", "override", "scan",
    "malfunction", "panel", "grid", "reboot", "diagnostic", "check",
    "mainframe", "electrical", "machinery",
})


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict | None:
    fence_match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = fence_match.group(1) if fence_match else text.strip()
    for candidate in (json_str, text.strip()):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            return None
    return None


def _s(v) -> str:
    """Coerce any LLM value to a plain string — guards against the model returning
    a nested dict or list where a string field was expected."""
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return ""
    return str(v)


# ---------------------------------------------------------------------------
# Clue-type classification + prompt building
# ---------------------------------------------------------------------------


def _classify_objects(world: GameWorld, proof_object_id: str = "") -> dict[str, str]:
    """Return {object_id: clue_type} for plot-critical objects.

    clue_type is one of: story_clue, key, code, lock, other. Classification
    uses only fields that exist on WorldObject: requires_tool, requires_code,
    contains_info.
    """
    referenced_as_tool: set[str] = set()
    referenced_as_code_source: set[str] = set()
    for obj in world.objects:
        if obj.requires_tool:
            referenced_as_tool.add(obj.requires_tool)
        if obj.requires_code:
            producer = next(
                (o.id for o in world.objects if o.contains_info == obj.requires_code),
                None,
            )
            if producer:
                referenced_as_code_source.add(producer)

    clue_types: dict[str, str] = {}
    for obj in world.objects:
        if proof_object_id and obj.id == proof_object_id:
            clue_types[obj.id] = "story_clue"
        elif obj.id in referenced_as_tool:
            clue_types[obj.id] = "key"
        elif obj.id in referenced_as_code_source:
            clue_types[obj.id] = "code"
        elif obj.contains_info:
            clue_types[obj.id] = "code" if re.search(r"\d", obj.contains_info) else "story_clue"
        elif obj.requires_tool or obj.requires_code:
            clue_types[obj.id] = "lock"
        else:
            clue_types[obj.id] = "other"

    return clue_types


def _is_plot_critical(obj, clue_types: dict[str, str]) -> bool:
    if obj.id.startswith(_NON_PLOT_PREFIXES):
        return False
    ct = clue_types.get(obj.id, "other")
    return ct != "other" or bool(obj.requires_tool or obj.requires_code or obj.contains_info)


def _build_world_data(world: GameWorld, characters: list[Character]) -> dict:
    clue_types = _classify_objects(world)
    obj_by_id = {o.id: o for o in world.objects}

    plot_objects = [
        {
            "id": o.id,
            "location": o.location,
            "description": o.description,
            "clue_type": clue_types.get(o.id, "other"),
            "contains_info_token": o.contains_info or "",
        }
        for o in world.objects
        if _is_plot_critical(o, clue_types)
    ]

    relationships: list[str] = []
    for obj in world.objects:
        if obj.requires_tool and obj.requires_tool in obj_by_id:
            key = obj_by_id[obj.requires_tool]
            relationships.append(f'"{key.id}" ({key.description[:60]}) -> unlocks "{obj.id}"')
        if obj.requires_code:
            producer = next(
                (o for o in world.objects if o.contains_info == obj.requires_code), None
            )
            if producer:
                relationships.append(f'"{producer.id}" (contains code) -> unlocks "{obj.id}"')

    return {
        "scenario": world.scenario,
        "objective": world.objective,
        "rooms": [{"id": r.id, "description": r.description} for r in world.rooms],
        "players": [
            {"name": c.name, "role": c.role, "backstory": c.backstory[:120] if c.backstory else ""}
            for c in characters
        ],
        "plot_objects": plot_objects[:12],
        "unlock_chain": relationships[:10],
    }


# ---------------------------------------------------------------------------
# LLM passes
# ---------------------------------------------------------------------------


def _call_json(system_prompt: str, user_prompt: str, llm=None) -> dict | None:
    """One LLM call returning a parsed JSON object, or None on any failure."""
    if llm is None:
        llm = get_llm("storyboard")
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception as exc:
        log.warning("storyboard pass: LLM call failed: {}", exc)
        return None
    data = _parse_json(response.content)
    if not isinstance(data, dict):
        log.warning("storyboard pass: could not parse LLM response")
        return None
    return data


def _run_core_pass(world_data: dict) -> dict | None:
    """Pass 1 — mystery/solution/suspects. Small output: victim/killer cannot be dropped."""
    user_prompt = CORE_GENERATION_PROMPT.format(world_data_json=json.dumps(world_data, indent=2))
    return _call_json(CORE_SYSTEM_PROMPT, user_prompt)


def _case_facts(data: dict) -> dict:
    """Extract the immutable case facts decided by the core pass."""
    m = data.get("mystery") if isinstance(data.get("mystery"), dict) else {}
    sol = data.get("solution") if isinstance(data.get("solution"), dict) else {}
    suspects = data.get("suspects") if isinstance(data.get("suspects"), list) else []
    return {
        "victim": _s(m.get("victim")),
        "killer_name": _s(m.get("killer_name")) or _s(sol.get("answer")),
        "proof_object_id": _s(m.get("proof_object_id")) or _s(sol.get("proof_object")),
        "motive_hint": _s(m.get("motive_hint")),
        "suspects": [
            {"name": _s(s.get("name")), "is_killer": bool(s.get("is_killer"))}
            for s in suspects
            if isinstance(s, dict) and s.get("name")
        ],
    }


def _case_facts_section(case_facts: dict | None) -> str:
    if not case_facts or not case_facts.get("killer_name"):
        return ""
    return "CASE_FACTS (immutable — use these exact names):\n" + json.dumps(case_facts, indent=2) + "\n\n"


def _run_beats_pass(world_data: dict, is_mystery: bool, case_facts: dict | None, llm=None) -> dict | None:
    """Pass 2 — clue layer, grounded in the now-fixed case facts."""
    user_prompt = BEATS_GENERATION_PROMPT.format(
        is_mystery="true" if is_mystery else "false",
        world_data_json=json.dumps(world_data, indent=2),
        case_facts_section=_case_facts_section(case_facts),
    )
    return _call_json(BEATS_SYSTEM_PROMPT, user_prompt, llm)


def _run_flavor_pass(world_data: dict, is_mystery: bool, case_facts: dict | None, llm=None) -> dict | None:
    """Pass 3 — atmosphere layer: plot, adapted personas, room stories, phase guidance."""
    user_prompt = FLAVOR_GENERATION_PROMPT.format(
        is_mystery="true" if is_mystery else "false",
        world_data_json=json.dumps(world_data, indent=2),
        case_facts_section=_case_facts_section(case_facts),
    )
    return _call_json(FLAVOR_SYSTEM_PROMPT, user_prompt, llm)


# ---------------------------------------------------------------------------
# Repair + sanitization
# ---------------------------------------------------------------------------


def _repair_mystery(data: dict, world: GameWorld) -> None:
    """Patch common LLM failures in the mystery/solution/suspects sections."""
    gen_mystery = data.get("mystery") if isinstance(data.get("mystery"), dict) else {}
    solution = data.get("solution") if isinstance(data.get("solution"), dict) else {}

    killer = _s(gen_mystery.get("killer_name")) or _s(solution.get("answer"))
    proof = _s(gen_mystery.get("proof_object_id")) or _s(solution.get("proof_object"))
    victim = _s(gen_mystery.get("victim"))
    motive = _s(gen_mystery.get("motive_hint"))

    valid_obj_ids = {o.id for o in world.objects}
    if proof and (" " in proof or proof not in valid_obj_ids):
        recovered = ""
        for o in world.objects:
            if o.id not in valid_obj_ids or o.id.startswith(_NON_PLOT_PREFIXES):
                continue
            if o.contains_info and not re.search(r"\d", o.contains_info):
                recovered = o.id
                break
        if not recovered:
            beats = data.get("discovery_beats", {})
            if isinstance(beats, dict) and killer:
                killer_last = killer.split()[-1]
                for obj_id, beat_text in beats.items():
                    if " " not in obj_id and killer_last in _s(beat_text):
                        recovered = obj_id
                        break
        if recovered:
            log.warning("REPAIR: proof_object_id was a description — recovered {!r}", recovered)
            proof = recovered
            if isinstance(data.get("mystery"), dict):
                data["mystery"]["proof_object_id"] = ""
            if isinstance(data.get("solution"), dict):
                data["solution"]["proof_object"] = ""

    if not isinstance(data.get("mystery"), dict):
        data["mystery"] = {}
    m = data["mystery"]
    if killer and not m.get("killer_name"):
        m["killer_name"] = killer
    if proof and not m.get("proof_object_id"):
        m["proof_object_id"] = proof

    if not isinstance(data.get("solution"), dict):
        data["solution"] = {}
    sol = data["solution"]
    if killer and not sol.get("answer"):
        sol["answer"] = killer
    if proof and not sol.get("proof_object"):
        sol["proof_object"] = proof
    if not sol.get("answer_type"):
        sol["answer_type"] = "person"
    if not sol.get("question") and killer:
        victim_name = victim.split(",")[0] if victim else "the victim"
        sol["question"] = f"Who is responsible for what happened to {victim_name}?"
    if not sol.get("motive") and motive:
        sol["motive"] = motive
    if not sol.get("hint_1") and killer:
        sol["hint_1"] = "Focus on who had both a reason and the opportunity — look at the relationships."
    if not sol.get("hint_2") and proof:
        sol["hint_2"] = "The proof object holds the answer — check what it reveals about its owner."
    if not sol.get("victory_narration_hook") and killer:
        sol["victory_narration_hook"] = (
            f"The evidence points unmistakably to {killer} — every clue was a thread leading here."
        )
    if not sol.get("defeat_narration_hook") and killer:
        sol["defeat_narration_hook"] = (
            f"The truth stays buried. {killer} walks free while the secret holds."
        )

    # Normalize suspects field names.
    raw_suspects = data.get("suspects", [])
    if isinstance(raw_suspects, list):
        normalized = []
        for s in raw_suspects:
            if not isinstance(s, dict) or not s.get("name"):
                continue
            name = s.get("name", "")
            connection = (
                s.get("connection_to_victim") or s.get("relationship_to_victim") or s.get("description") or ""
            )
            apparent_motive = s.get("apparent_motive") or s.get("motive") or ""
            is_killer = s.get("is_killer", name == killer)
            if not apparent_motive:
                if is_killer and motive:
                    apparent_motive = motive
                elif is_killer:
                    apparent_motive = "Their exact motive remains unclear — but they had the most to gain."
                else:
                    apparent_motive = "Their connection to the victim gave them both means and opportunity."
            normalized.append({
                "name": name,
                "connection_to_victim": connection or "Known to the victim and present near the time of the incident.",
                "apparent_motive": apparent_motive,
                "is_killer": bool(is_killer),
            })
        data["suspects"] = normalized

    # Coerce discovery_beats values to strings and patch the proof beat.
    discovery_beats = data.get("discovery_beats", {})
    if isinstance(discovery_beats, dict):
        data["discovery_beats"] = {k: _s(v) for k, v in discovery_beats.items()}
        discovery_beats = data["discovery_beats"]
        if proof and killer:
            beat = discovery_beats.get(proof, "")
            killer_last = killer.split()[-1] if killer else ""
            if not beat or (killer_last and killer_last not in beat):
                discovery_beats[proof] = (
                    f"The evidence here bears {killer}'s mark — this is the moment the investigation breaks open."
                )

    # ending_guidance: ensure won/lost/lost_by_wrong_deduction are present.
    eg = data.get("ending_guidance", {})
    if isinstance(eg, dict):
        if not eg.get("won") and killer:
            eg["won"] = sol.get("victory_narration_hook", f"The team exposes {killer}. Justice is served.")
        if not eg.get("lost"):
            eg["lost"] = sol.get("defeat_narration_hook", "Time runs out. The truth stays buried.")
        if not eg.get("lost_by_wrong_deduction") and killer:
            eg["lost_by_wrong_deduction"] = f"The wrong name was called. {killer} slips away."
        data["ending_guidance"] = eg

    # plot.victim/threat/stakes: backfill from mystery facts if the LLM omitted them
    # (qwen3:14b reliably fills timeline/tension_hint/atmosphere but often drops these).
    if not isinstance(data.get("plot"), dict):
        data["plot"] = {}
    plot = data["plot"]
    if not plot.get("victim") and victim:
        plot["victim"] = victim
    if not plot.get("threat") and killer:
        plot["threat"] = "Someone in this story has already killed once and is willing to do it again to stay hidden."
    if not plot.get("stakes") and killer:
        plot["stakes"] = "If the team cannot uncover the truth, the person responsible walks free."


def _strip_mystery_sections(data: dict) -> None:
    """Force mystery/solution/suspects to empty for non-mystery themes."""
    data["mystery"] = {}
    data["solution"] = {}
    data["suspects"] = []
    seeds = data.get("conversation_seeds")
    if isinstance(seeds, dict):
        data["conversation_seeds"] = {k: [] for k in seeds}


def _ensure_personas(data: dict, characters: list[Character]) -> None:
    personas = data.get("adapted_personas")
    if not isinstance(personas, dict):
        data["adapted_personas"] = {}
        personas = data["adapted_personas"]
    for c in characters:
        entry = personas.get(c.name)
        if not isinstance(entry, dict):
            entry = {}
            personas[c.name] = entry
        if not entry.get("world_role"):
            entry["world_role"] = (
                f"{c.name} investigates by reading the physical scene — "
                f"looking for what was moved, what is missing, and what someone left behind."
            )
        if not entry.get("voice"):
            entry["voice"] = "Precise and observational — states what the evidence implies, not what they feel."
        if not entry.get("vocabulary"):
            entry["vocabulary"] = ["who had access", "this was deliberate", "something is missing here"]
        if not entry.get("sample_lines"):
            entry["sample_lines"] = [
                "Scratches around the keyhole. Fresh ones. This was forced by someone in a hurry.",
                "The frame is intact but the hinge is bent. That takes deliberate force, not an accident.",
            ]


def _sanitize_personas(data: dict, characters: list[Character]) -> None:
    """Strip vocabulary phrases that still contain tech/mechanical contamination words."""
    personas = data.get("adapted_personas")
    if not isinstance(personas, dict):
        return
    player_roles = {c.name: c.role for c in characters}
    for name, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        vocab = persona.get("vocabulary")
        if not isinstance(vocab, list):
            continue
        clean = []
        for phrase in vocab:
            if not isinstance(phrase, str):
                continue
            phrase_lower = phrase.lower()
            if any(word in phrase_lower for word in _CONTAMINATION_WORDS):
                log.warning(
                    "contaminated vocabulary for {!r} (role={!r}): {!r} — removed",
                    name, player_roles.get(name, ""), phrase,
                )
            else:
                clean.append(phrase)
        persona["vocabulary"] = clean


def _build_storyboard(data: dict, world_id: str) -> Storyboard:
    plot = data.get("plot", {}) if isinstance(data.get("plot"), dict) else {}
    mystery_raw = data.get("mystery", {}) if isinstance(data.get("mystery"), dict) else {}
    solution_raw = data.get("solution", {}) if isinstance(data.get("solution"), dict) else {}

    personas_raw = data.get("adapted_personas", {})
    adapted_personas = {
        name: StoryboardPersona(
            world_role=_s(v.get("world_role")),
            voice=_s(v.get("voice")),
            vocabulary=[p for p in v.get("vocabulary", []) if isinstance(p, str)],
            sample_lines=[p for p in v.get("sample_lines", []) if isinstance(p, str)],
        )
        for name, v in personas_raw.items()
        if isinstance(v, dict)
    }

    suspects = [
        {k: v for k, v in s.items() if k != "is_killer"}
        for s in data.get("suspects", [])
        if isinstance(s, dict) and s.get("name")
    ]

    room_stories = {
        k: _s(v) for k, v in data.get("room_stories", {}).items()
    } if isinstance(data.get("room_stories"), dict) else {}

    discovery_beats = {
        k: _s(v) for k, v in data.get("discovery_beats", {}).items()
    } if isinstance(data.get("discovery_beats"), dict) else {}

    phase_guidance = {
        k: _s(v) for k, v in data.get("phase_guidance", {}).items()
    } if isinstance(data.get("phase_guidance"), dict) else {}

    conversation_seeds = {
        name: [s for s in seeds if isinstance(s, str)]
        for name, seeds in data.get("conversation_seeds", {}).items()
        if isinstance(seeds, list)
    } if isinstance(data.get("conversation_seeds"), dict) else {}

    ending_guidance = {
        k: _s(v) for k, v in data.get("ending_guidance", {}).items()
    } if isinstance(data.get("ending_guidance"), dict) else {}

    return Storyboard(
        world_id=world_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        plot_victim=_s(plot.get("victim")),
        plot_threat=_s(plot.get("threat")),
        plot_stakes=_s(plot.get("stakes")),
        plot_timeline=_s(plot.get("timeline")),
        plot_tension_hint=_s(plot.get("tension_hint")),
        plot_atmosphere=_s(plot.get("atmosphere")),
        plot_protagonist_context=_s(plot.get("protagonist_context")),
        adapted_personas=adapted_personas,
        room_stories=room_stories,
        discovery_beats=discovery_beats,
        phase_guidance=phase_guidance,
        conversation_seeds=conversation_seeds,
        ending_guidance=ending_guidance,
        suspects=suspects,
        mystery=StoryboardMystery(
            victim=_s(mystery_raw.get("victim")),
            killer_name=_s(mystery_raw.get("killer_name")),
            proof_object_id=_s(mystery_raw.get("proof_object_id")),
            motive_hint=_s(mystery_raw.get("motive_hint")),
        ),
        solution=StoryboardSolution(
            question=_s(solution_raw.get("question")),
            answer=_s(solution_raw.get("answer")),
            answer_aliases=[a for a in solution_raw.get("answer_aliases", []) if isinstance(a, str)],
            answer_type=_s(solution_raw.get("answer_type")) or "person",
            motive=_s(solution_raw.get("motive")),
            proof_object=_s(solution_raw.get("proof_object")),
            proof_sentence=_s(solution_raw.get("proof_sentence")),
            hint_1=_s(solution_raw.get("hint_1")),
            hint_2=_s(solution_raw.get("hint_2")),
            victory_narration_hook=_s(solution_raw.get("victory_narration_hook")),
            defeat_narration_hook=_s(solution_raw.get("defeat_narration_hook")),
        ),
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def storyboard_builder_node(state: GameState) -> dict:
    """Generate the narrative storyboard for the assembled world.

    Runs after puzzle_builder, in three focused LLM passes: core facts
    (mystery themes only), the clue layer, and the flavor layer. mystery/
    solution/suspects are populated only when the theme is a mystery theme.
    Degrades to an empty/partial Storyboard on any LLM/parse failure so the
    pipeline never fails because of this node.
    """
    world = state.world
    if not world or not world.rooms:
        log.warning("storyboard_builder_node called with no world — skipping")
        return {}

    is_mystery = "mystery" in state.theme.lower()
    world_id = world.scenario[:40] if world.scenario else ""

    world_data = _build_world_data(world, state.characters)
    data: dict = {}
    transcript_parts: list[str] = []

    case_facts: dict | None = None
    if is_mystery:
        core = _run_core_pass(world_data)
        if core:
            data.update({k: core[k] for k in ("mystery", "solution", "suspects") if k in core})
            transcript_parts.append(f"--- core pass ---\n{json.dumps(core, indent=2)}")
        else:
            log.warning("storyboard_builder: core pass failed — relying on repair")
        case_facts = _case_facts(data)

    # Passes 2 (beats) and 3 (flavor) are independent given case_facts — run them
    # concurrently, each on its own Ollama instance if Settings.ollama_workers
    # lists additional LAN instances (same fan-out pattern as
    # puzzle_graph.apply_theming).
    from concurrent.futures import ThreadPoolExecutor

    from src.escape_rooms.utils.settings import get_worker_llms

    llms = get_worker_llms("storyboard")
    with ThreadPoolExecutor(max_workers=2) as pool:
        beats_future = pool.submit(_run_beats_pass, world_data, is_mystery, case_facts, llms[0])
        flavor_future = pool.submit(_run_flavor_pass, world_data, is_mystery, case_facts, llms[1 % len(llms)])
        beats = beats_future.result()
        flavor = flavor_future.result()

    if beats:
        data.update({
            k: beats[k]
            for k in ("discovery_beats", "conversation_seeds", "ending_guidance")
            if k in beats
        })
        transcript_parts.append(f"--- beats pass ---\n{json.dumps(beats, indent=2)}")
    else:
        log.warning("storyboard_builder: beats pass failed — relying on repair")

    if flavor:
        data.update({
            k: flavor[k]
            for k in ("plot", "adapted_personas", "room_stories", "phase_guidance")
            if k in flavor
        })
        transcript_parts.append(f"--- flavor pass ---\n{json.dumps(flavor, indent=2)}")
    else:
        log.warning("storyboard_builder: flavor pass failed — relying on repair")

    if not transcript_parts:
        log.warning("storyboard_builder: all passes failed — returning empty storyboard")
        return {
            "storyboard": Storyboard(world_id=world_id),
            "messages": [AIMessage(content="=== STORYBOARD (all passes failed) ===")],
        }

    if is_mystery:
        _repair_mystery(data, world)
    else:
        _strip_mystery_sections(data)

    _ensure_personas(data, state.characters)
    _sanitize_personas(data, state.characters)

    storyboard = _build_storyboard(data, world_id)

    if is_mystery:
        if not storyboard.mystery.killer_name:
            log.warning("storyboard_builder: mystery.killer_name is empty")
        if not storyboard.mystery.proof_object_id:
            log.warning("storyboard_builder: mystery.proof_object_id is empty")
        if len(storyboard.suspects) < 2:
            log.warning("storyboard_builder: only {} suspect(s) generated", len(storyboard.suspects))

    log.success(
        "Generated storyboard — {} room stor{}, {} discovery beat(s), {} persona(s)",
        len(storyboard.room_stories),
        "y" if len(storyboard.room_stories) == 1 else "ies",
        len(storyboard.discovery_beats),
        len(storyboard.adapted_personas),
    )

    return {
        "storyboard": storyboard,
        "messages": [AIMessage(content="=== STORYBOARD ===\n\n" + "\n\n".join(transcript_parts))],
    }
