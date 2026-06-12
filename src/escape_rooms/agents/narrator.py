"""GameMasterNarrator — the storytelling LLM.

Ported from Escapee's ``app/agents/gm_narrator.py`` / ``gm_narrator_prompt.py``,
adapted to this project's sync ``get_llm(role).invoke([...])`` convention and
``Storyboard``/``GameWorld``/``Character`` schema.

This is a PRESENTATION layer: its output is never fed back into the simulator.
If the model errors, each method degrades gracefully to a deterministic
fallback so the game keeps flowing.

Standalone / best-effort for this pass — not yet wired into ``gameplay.py``'s
tick loop (see plan section 9). Importable and testable on its own, e.g.:

    storyboard = Storyboard.model_validate(data["storyboard"])
    narrator = GameMasterNarrator(storyboard=storyboard)
    print(narrator.narrate_opening(scenario=..., objective=..., rooms=[...]))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.escape_rooms.prompts import load_prompt
from src.escape_rooms.state import Storyboard
from src.escape_rooms.utils.settings import get_llm

NARRATOR_SYSTEM_PROMPT = load_prompt("narrator", "system")
NARRATOR_EVENT_SYSTEM_PROMPT = load_prompt("narrator", "event_system")

_NON_LATIN_RE = re.compile(r"[⺀-鿿豈-﫿︰-﹏＀-￯]+")
_SPEAKER_CONTINUATION_RE = re.compile(r'["\s]{2,}[A-Z][a-z]+ [A-Z][a-z]+\s*:')


def humanize_text(token: str | None) -> str:
    """Turn a snake_case id/token into readable words (supply_locker -> supply locker)."""
    if not token:
        return "something"
    return token.replace("_", " ").strip()


def _tension_level(turn: int, solved_count: int = 0, total_puzzles: int = 0) -> str:
    """Return a tension descriptor that escalates across the game arc."""
    if total_puzzles > 0:
        solved_ratio = solved_count / total_puzzles
        if solved_ratio >= 0.8:
            return "PEAK — almost out, every second counts"
        if solved_ratio >= 0.5:
            return "HIGH — team is making progress but danger is real"
        if solved_ratio >= 0.25:
            return "BUILDING — clues accumulating, the picture forming"
    if turn <= 5:
        return "LOW — disoriented, still taking in the situation"
    if turn <= 12:
        return "BUILDING — patterns emerging, urgency growing"
    if turn <= 20:
        return "HIGH — team is committed, no turning back"
    return "PEAK — running out of time, desperation is setting in"


def _mood_for(*, success: bool, looped: bool = False) -> str:
    if looped:
        return "frustrated but collaborative"
    if success:
        return "energized and urgent"
    return "tense and analytical"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_dialogue_context_package(
    *,
    scenario: str,
    objective: str,
    turn: int,
    actor_name: str,
    actor_role: str,
    actor_backstory: str,
    action_text: str,
    speech: str | None,
    outcome: str,
    success: bool,
    looped: bool = False,
    recent_story: list[str],
    lore_excerpt: str = "",
    adapted_world_role: str = "",
    persona_voice: str = "",
    persona_vocabulary: list[str] | None = None,
    persona_sample_lines: list[str] | None = None,
    conversation_seed: str = "",
    proof_revealed: bool = False,
    killer_name: str = "",
    suspects_context: str = "",
    solved_count: int = 0,
    total_puzzles: int = 0,
) -> str:
    """Build a rigid, low-drift context package for local 7B dialogue models."""
    story = "\n".join(f"- {beat}" for beat in recent_story) or "(session just started)"
    speech_text = humanize_text(speech.strip()) if speech and speech.strip() else "(not stated)"

    lore_section = (
        f"WORLD_LORE (tone/voice reference — do not quote directly):\n{lore_excerpt}\n\n"
        if lore_excerpt else ""
    )

    if adapted_world_role:
        role_line = f"- ROLE IN THIS WORLD: {adapted_world_role}"
    else:
        role_line = f"- ROLE: {actor_role}"

    voice_line = f"- VOICE: {persona_voice}\n" if persona_voice else ""
    vocabulary_line = (
        f"- VOCABULARY (work these phrasings in naturally, don't force all of them): {', '.join(persona_vocabulary)}\n"
        if persona_vocabulary else ""
    )
    sample_lines_section = (
        "- REGISTER EXAMPLES (match this rhythm and attitude, do not copy verbatim):\n"
        + "\n".join(f'  "{line}"' for line in persona_sample_lines) + "\n"
        if persona_sample_lines else ""
    )

    seed_section = (
        f"PLOT_OBSERVATION (surface this naturally if the moment fits):\n"
        f"  {conversation_seed}\n\n"
        if conversation_seed else ""
    )

    if proof_revealed and killer_name:
        investigation_status = (
            "INVESTIGATION_STATUS: REVELATION MODE\n"
            f"The proof has been found. The killer is {killer_name}. "
            f"Characters now know who is responsible. You MAY name {killer_name} explicitly.\n"
        )
    else:
        suspects_block = (
            f"KNOWN SUSPECTS (any could be responsible — do not name which is guilty):\n{suspects_context}\n"
            if suspects_context else
            "KNOWN SUSPECTS: identity unknown.\n"
        )
        investigation_status = (
            "INVESTIGATION_STATUS: MYSTERY MODE\n"
            + suspects_block
            + "FORBIDDEN: naming any specific suspect as the killer. "
            "Use 'whoever did this', 'someone with access', 'the person responsible'.\n"
        )

    tension = _tension_level(turn, solved_count, total_puzzles)
    return (
        "CONTEXT_PACKAGE\n"
        "GLOBAL_CONTEXT:\n"
        f"- SCENARIO: {scenario}\n"
        f"- OBJECTIVE: {objective}\n"
        f"- TURN: {turn} | TENSION_LEVEL: {tension}\n"
        f"STORY_SO_FAR (most recent last — react to the last entry if it's from a teammate):\n{story}\n\n"
        f"{lore_section}"
        f"{seed_section}"
        f"{investigation_status}\n"
        "CHARACTER_SHEET (shapes this character's specific lens on everything they see):\n"
        f"- NAME: {actor_name}\n"
        f"{role_line}\n"
        f"- BACKSTORY_HINT: {actor_backstory or 'ordinary survivor under pressure'}\n"
        f"- MOOD: {_mood_for(success=success, looped=looped)}\n"
        f"{voice_line}"
        f"{vocabulary_line}"
        f"{sample_lines_section}"
        "\n"
        "ACTION_EVENT:\n"
        f"- ATTEMPT: {action_text}\n"
        f"- PLAYER_MOTIVATION (background only — DO NOT echo this as your line; it is what the character is thinking, not what they say):\n"
        f"  {speech_text}\n"
        f"- SIMULATOR_OUTCOME: {outcome}\n"
        f"- SUCCESS: {'yes' if success else 'no'}\n\n"
        "WRITING_TARGET:\n"
        "- FORMAT: Character Name: \"message\"\n"
        "- LENGTH: 10 to 20 words — short, punchy, specific\n"
        "- STEP 1: react to the last STORY_SO_FAR entry if it's a teammate\n"
        "- STEP 2: add what SIMULATOR_OUTCOME just revealed, through CHARACTER_SHEET lens\n"
        "- FORBIDDEN: — or – or -- or rhetorical questions (Why...? What...? Could this...?)\n"
        "- FORBIDDEN: echoing PLAYER_MOTIVATION word-for-word\n"
        "- FORBIDDEN: invented facts not in SIMULATOR_OUTCOME\n"
        "Write the single dialogue line now. Stop after the closing quote."
    )


def build_opening_user_prompt(
    *, scenario: str, objective: str, room_ids: list[str], storyboard: Storyboard | None = None
) -> str:
    rooms_text = ", ".join(room_ids)

    story_context_parts: list[str] = []
    if storyboard and not storyboard.is_empty():
        if storyboard.plot_victim:
            story_context_parts.append(f"THE VICTIM: {storyboard.plot_victim}")
        if storyboard.plot_protagonist_context:
            story_context_parts.append(f"WHO THEY ARE: {storyboard.plot_protagonist_context}")
        if storyboard.plot_timeline:
            story_context_parts.append(f"WHAT HAPPENED BEFORE: {storyboard.plot_timeline}")
        if storyboard.plot_stakes:
            story_context_parts.append(f"WHAT'S AT STAKE: {storyboard.plot_stakes}")
        if storyboard.plot_atmosphere:
            story_context_parts.append(f"ATMOSPHERE: {storyboard.plot_atmosphere}")

    story_context = (
        "\nSTORY CONTEXT (use specific names and facts from here — do NOT quote directly):\n"
        + "\n".join(story_context_parts)
        + "\n"
    ) if story_context_parts else ""

    return (
        "Write the opening moment: the team arrives and feels the full weight of what they face.\n\n"
        f"SETTING: {scenario}\n"
        f"THEIR MISSION: {objective}\n"
        f"ROOMS THEY WILL SEARCH: {rooms_text}\n"
        + story_context
        + "\nWrite 2-3 sentences. Open with a single sharp sensory detail (smell, sound, or sight). "
        "If STORY CONTEXT includes a victim name, use it — make them real. "
        "End with what's concretely at stake — say 'the killer', 'whoever is responsible', or 'the murderer'. "
        "NEVER name the killer or any suspect in the opening — the mystery must remain unsolved. "
        "Present tense. Third person. No em-dashes. No generic words like 'eerie' or 'dark secrets'."
    )


def build_room_entry_prompt(
    *, actor_name: str, room_id: str, room_story: str = "", scenario: str
) -> str:
    """Narrator prompt for the first time a character enters a new room."""
    story_line = f"\nROOM STORY CONTEXT: {room_story}" if room_story else ""
    return (
        f"SCENARIO: {scenario}\n"
        f"CHARACTER: {actor_name} just entered: {room_id.replace('_', ' ')}"
        f"{story_line}\n\n"
        "Write 1-2 sentences describing this room as the character first sees it. "
        "If ROOM STORY CONTEXT is given, let it shape the atmosphere — but do NOT quote or summarize it directly. "
        "Focus on one striking sensory detail: light, smell, sound, or texture. "
        "Simple English. Present tense. Third person. Do not list objects or exits. No em-dashes."
    )


def build_discovery_prompt(
    *,
    actor_name: str,
    item_id: str,
    item_description: str,
    unlocks_description: str,
    connection_lore: str,
    scenario: str,
) -> str:
    """Narrator prompt for a first-discovery beat: connects item to what it unlocks."""
    lore_section = f"\nSTORY CONTEXT: {connection_lore}" if connection_lore else ""
    return (
        f"SCENARIO: {scenario}\n"
        f"CHARACTER: {actor_name}\n"
        f"ITEM FOUND: {item_id.replace('_', ' ')} — {item_description}\n"
        f"THIS ITEM ENABLES: {unlocks_description}"
        f"{lore_section}\n\n"
        "Write 1-2 sentences of atmospheric third-person narration for this discovery moment. "
        "Connect the item's physical details to what it enables — give the reader the WHY. "
        "Simple English. Present tense. No invented facts. No em-dashes. "
        "Do NOT state the solution or code directly. Focus on mood and significance."
    )


def build_system_event_prompt(*, event_kind: str, actor_name: str, detail: str, scenario: str) -> str:
    """Build a narrator prose prompt for internal system events.

    ``event_kind`` is one of: "loop_avoided", "planner_override", "critical_stuck", "milestone".
    """
    if event_kind == "loop_avoided":
        instruction = (
            f"{actor_name} just tried to repeat something that already failed or was already done. "
            f"Detail: {detail}\n\n"
            "Narrate this as a brief atmospheric beat: the character's body language, "
            "a flicker of frustration, the dead end. 1-2 sentences."
        )
    elif event_kind == "planner_override":
        instruction = (
            f"The team was forced onto a different action. Detail: {detail}\n\n"
            "Narrate this as a moment of urgency or course correction. 1-2 sentences."
        )
    elif event_kind == "critical_stuck":
        instruction = (
            f"The team is critically stuck and going in circles. Detail: {detail}\n\n"
            "Narrate rising tension, the clock ticking, desperation creeping in. 1-2 sentences."
        )
    else:  # milestone
        instruction = (
            f"The team just achieved a breakthrough: {detail}\n\n"
            "Narrate the moment of success, the shift in energy. 1-2 sentences."
        )

    return (
        f"SCENARIO: {scenario}\n\n"
        f"EVENT: {instruction}\n\n"
        "Write the narration now. No invented facts. No dashes. Present tense."
    )


def build_ending_user_prompt(*, objective: str, won: bool, ending_guidance: str = "") -> str:
    if won:
        fate = "The team has SUCCEEDED. Give the story a triumphant, exhaling close."
    else:
        fate = "The team has FAILED to escape in time. Give the story a grim, unresolved close."

    guidance_section = (
        f"\nSTORY ENDING GUIDANCE (use this to shape the close):\n{ending_guidance}\n"
        if ending_guidance else ""
    )
    return (
        f"Conclude the story.\n\nGOAL WAS: {objective}\n{fate}"
        f"{guidance_section}\n\n"
        "Write 2-3 sentences of closing narration. "
        "If STORY ENDING GUIDANCE is given, use its specific details — names, consequences, tone. "
        "No new puzzle details. No em-dashes."
    )


# ---------------------------------------------------------------------------
# GameMasterNarrator
# ---------------------------------------------------------------------------


@dataclass
class GameMasterNarrator:
    """Narrates the live game as prose, with a rolling memory for continuity."""

    storyboard: Storyboard = field(default_factory=Storyboard)
    temperature: float = 0.8
    recent_window: int = 6

    _recent: list[str] = field(default_factory=list, init=False)
    _used_seeds: set[str] = field(default_factory=set, init=False)
    _proof_revealed: bool = field(default=False, init=False)

    def reveal_proof(self) -> None:
        """Flip narrator into revelation mode.

        Called the moment the proof object is discovered. From this point on,
        the narrator is allowed to name the killer. Before this call, mystery
        mode is active and the killer's name is suppressed from all dialogue.
        """
        self._proof_revealed = True

    def _normalize_dialogue_line(self, text: str) -> str:
        """Extract message text from model output in `Name: "message"` format."""
        line = " ".join((text or "").strip().splitlines()).strip()
        if not line:
            return "..."
        line = _NON_LATIN_RE.sub("", line).strip()
        if not line:
            return "..."

        for prefix in ("assistant:", "narrator:", "dialogue:"):
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
                break

        if ":" in line:
            parts = line.split(":", 1)
            candidate = parts[1].strip().strip('"').strip()
            if candidate:
                line = candidate

        if line.startswith('"') and line.endswith('"') and len(line) > 1:
            line = line[1:-1].strip()

        line = _SPEAKER_CONTINUATION_RE.split(line)[0]
        line = line.rstrip('"').strip()

        line = line.replace("—", ",").replace("–", ",").replace("--", ",")

        return line or "..."

    def _clean_prose(self, text: str) -> str:
        """Clean raw prose output (narrator beats) — strip role prefixes and dashes."""
        line = " ".join((text or "").strip().splitlines()).strip()
        if not line:
            return "..."
        line = _NON_LATIN_RE.sub("", line).strip()
        if not line:
            return "..."
        for prefix in ("assistant:", "narrator:", "prose:"):
            if line.lower().startswith(prefix):
                line = line[len(prefix):].strip()
                break
        line = line.replace("—", ",").replace("–", ",").replace("--", ",")
        return line or "..."

    def _say(self, user_prompt: str, *, system: str, fallback: str) -> str:
        try:
            response = get_llm("narrator").invoke(
                [SystemMessage(content=system), HumanMessage(content=user_prompt)]
            )
            text = (response.content or "").strip()
        except Exception:
            return fallback
        return text or fallback

    def _remember(self, line: str) -> None:
        """Store a bare prose line (for opening, milestone, system events)."""
        self._recent.append(line)
        if len(self._recent) > self.recent_window:
            self._recent = self._recent[-self.recent_window:]

    def remember_turn(self, *, actor_name: str, speech: str) -> None:
        """Store the character's speech as a chat line so STORY_SO_FAR reads as a conversation."""
        self._recent.append(f'{actor_name}: "{speech}"')
        if len(self._recent) > self.recent_window:
            self._recent = self._recent[-self.recent_window:]

    def _lore_excerpt(self, *, actor_name: str, room_id: str, object_id: str = "") -> str:
        return self.storyboard.lore_excerpt(actor_name=actor_name, room_id=room_id, object_id=object_id)

    def connection_lore_for(self, object_id: str) -> str:
        """Return the pre-written story sentence for a specific object, or empty string."""
        return self.storyboard.discovery_beat(object_id)

    def narrate_opening(self, *, scenario: str, objective: str, room_ids: list[str]) -> str:
        text = self._say(
            build_opening_user_prompt(
                scenario=scenario, objective=objective, room_ids=room_ids, storyboard=self.storyboard
            ),
            system=NARRATOR_EVENT_SYSTEM_PROMPT,
            fallback=scenario,
        )
        text = self._clean_prose(text)
        self._remember(text)
        return text

    def narrate_turn(
        self,
        *,
        scenario: str,
        objective: str,
        turn: int,
        actor_name: str,
        actor_role: str,
        actor_backstory: str,
        action_text: str,
        speech: str | None,
        outcome: str,
        success: bool,
        looped: bool = False,
        room_id: str = "",
        object_id: str = "",
        solved_count: int = 0,
        total_puzzles: int = 0,
    ) -> str:
        lore_excerpt = self._lore_excerpt(actor_name=actor_name, room_id=room_id, object_id=object_id)
        persona = self.storyboard.persona_for(actor_name)
        seed = self.storyboard.pop_seed(actor_name, self._used_seeds)
        if seed:
            self._used_seeds.add(seed)

        # Prefer storyboard.mystery (authoritative generated source); fall back to solution.answer.
        killer_name = (
            (self.storyboard.mystery.killer_name or self.storyboard.solution.answer)
            if self._proof_revealed else ""
        )
        line = self._say(
            build_dialogue_context_package(
                scenario=scenario,
                objective=objective,
                turn=turn,
                actor_name=actor_name,
                actor_role=actor_role,
                actor_backstory=actor_backstory,
                action_text=action_text,
                speech=speech,
                outcome=outcome,
                success=success,
                looped=looped,
                recent_story=list(self._recent),
                lore_excerpt=lore_excerpt,
                adapted_world_role=persona.world_role if persona else "",
                persona_voice=persona.voice if persona else "",
                persona_vocabulary=persona.vocabulary if persona else None,
                persona_sample_lines=persona.sample_lines if persona else None,
                conversation_seed=seed or "",
                proof_revealed=self._proof_revealed,
                killer_name=killer_name,
                suspects_context=self.storyboard.suspects_context(),
                solved_count=solved_count,
                total_puzzles=total_puzzles,
            ),
            system=NARRATOR_SYSTEM_PROMPT,
            fallback=f'{actor_name}: "{outcome}"',
        )
        normalized = self._normalize_dialogue_line(line)
        self.remember_turn(actor_name=actor_name, speech=normalized)
        return normalized

    def narrate_system_event(self, *, event_kind: str, actor_name: str, detail: str, scenario: str) -> str:
        """Generate atmospheric narrator prose for internal system events (loop, override, stuck)."""
        text = self._say(
            build_system_event_prompt(event_kind=event_kind, actor_name=actor_name, detail=detail, scenario=scenario),
            system=NARRATOR_EVENT_SYSTEM_PROMPT,
            fallback="",
        )
        if text:
            text = self._clean_prose(text)
            self._remember(text)
        return text

    def narrate_room_entry(self, *, actor_name: str, room_id: str, scenario: str) -> str:
        """One-time atmospheric beat the first time a character enters a room."""
        text = self._say(
            build_room_entry_prompt(
                actor_name=actor_name,
                room_id=room_id,
                room_story=self.storyboard.room_story(room_id),
                scenario=scenario,
            ),
            system=NARRATOR_EVENT_SYSTEM_PROMPT,
            fallback="",
        )
        if text:
            text = self._clean_prose(text)
            self._remember(text)
        return text

    def narrate_discovery(
        self,
        *,
        actor_name: str,
        item_id: str,
        item_description: str,
        unlocks_description: str,
        connection_lore: str,
        scenario: str,
    ) -> str:
        """One-time discovery beat: connects a found item to what it unlocks.

        If the storyboard has a pre-written beat for this object, that sentence
        is emitted directly — no LLM call. Falls back to LLM generation only
        when the storyboard has no entry for this object.
        """
        pre_written = self.storyboard.discovery_beat(item_id)
        if pre_written:
            self._remember(pre_written)
            return pre_written

        text = self._say(
            build_discovery_prompt(
                actor_name=actor_name,
                item_id=item_id,
                item_description=item_description,
                unlocks_description=unlocks_description,
                connection_lore=connection_lore,
                scenario=scenario,
            ),
            system=NARRATOR_EVENT_SYSTEM_PROMPT,
            fallback="",
        )
        if text:
            text = self._clean_prose(text)
            self._remember(text)
        return text

    def narrate_milestone(self, *, milestone: str, scenario: str) -> str:
        """Generate a short atmospheric narrator beat for a progress milestone."""
        prompt = (
            f"SCENARIO: {scenario}\n"
            f"MILESTONE JUST ACHIEVED: {milestone}\n\n"
            "Write 1-2 sentences of tense, atmospheric third-person narration "
            "reacting to this breakthrough. Present tense. No character names. "
            "No invented facts. No em-dashes. Focus on the mood shift."
        )
        text = self._say(prompt, system=NARRATOR_EVENT_SYSTEM_PROMPT, fallback="")
        if text:
            text = self._clean_prose(text)
            self._remember(text)
        return text

    def narrate_ending(self, *, objective: str, won: bool, wrong_deduction: bool = False) -> str:
        fallback = (
            "The crew breaks free into the light." if won
            else "Time runs out. The truth stays buried."
        )
        ending_guidance = self.storyboard.ending_text(won, wrong_deduction=wrong_deduction)
        text = self._say(
            build_ending_user_prompt(objective=objective, won=won, ending_guidance=ending_guidance),
            system=NARRATOR_EVENT_SYSTEM_PROMPT,
            fallback=fallback,
        )
        text = self._clean_prose(text)
        self._remember(text)
        return text
