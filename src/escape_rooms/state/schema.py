from __future__ import annotations

import re
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Prerequisite(BaseModel):
    """A structured condition — used by Room.goal_completion to mark when a room's goal is done.

    A Prerequisite has exactly one `type`; only the fields meaningful for that
    type carry a value. The always-null sibling fields are dropped at dump time
    (see ``main._jsonable`` / ``GameWorld`` serialization with ``exclude_none``),
    so emitted JSON shows only the keys that matter for each condition shape:
      - object_state -> object_id, state
      - has_item     -> object_id
      - known_info   -> info
      - power_active  -> id
    """

    type: str  # "object_state" | "known_info" | "has_item" | "power_active"
    object_id: str | None = None
    state: str | None = None
    info: str | None = None
    id: str | None = None


class Room(BaseModel):
    """Static blueprint of a room — never mutated after generation."""

    id: str
    description: str
    adjacency: dict[str, str] = Field(default_factory=dict)
    goal: str = ""
    goal_completion: Prerequisite | None = None
    key_objects: list[str] = Field(default_factory=list)


class WorldObject(BaseModel):
    """An object placed in the world — fixed scenery, item, container, panel, door, etc."""

    id: str
    location: str  # room id or another object id (for nesting)
    description: str
    state: str = "visible"
    interactable: bool = False
    takeable: bool = False
    requires_code: str | None = None
    code_digits: int | None = None
    requires_tool: str | None = None
    requires_liquid: str | None = None
    requires_power: str | None = None
    fuses: dict[str, str] | None = None
    contains_info: str | None = None
    slot_description: str | None = None
    note: str | None = None


class WinCondition(BaseModel):
    """The target object state that ends the game in victory."""

    object_id: str = ""
    state: str = ""


def derive_win_condition(rooms: list["Room"]) -> WinCondition:
    """Build the win condition from the final room's object_state goal_completion.

    The win condition is, by construction (see the generation prompt), the last
    room's ``object_state`` goal. puzzle_builder calls this once the world is
    fully assembled to populate :attr:`GameWorld.win_condition`. Returns an empty
    WinCondition when there is no usable final ``object_state`` goal.
    """
    for room in reversed(rooms):
        gc = room.goal_completion
        if gc is not None and gc.type == "object_state" and gc.object_id:
            return WinCondition(object_id=gc.object_id, state=gc.state or "")
    return WinCondition()


class GameWorld(BaseModel):
    """The frozen dungeon blueprint produced by the game master.

    ``win_condition`` is a stored field populated by puzzle_builder (via
    :func:`derive_win_condition`) once objects exist. world_builder leaves it at
    its empty default, so a rooms-only skeleton has no win target yet.
    """

    scenario: str = ""
    objective: str = ""
    rooms: list[Room] = Field(default_factory=list)
    objects: list[WorldObject] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    solution_path: list[str] = Field(default_factory=list)
    win_condition: WinCondition = Field(default_factory=WinCondition)


class Character(BaseModel):
    """A playable character generated for this escape room."""

    name: str
    role: str
    backstory: str


class PartyMember(BaseModel):
    """A player agent's character selection with reasoning."""

    agent_id: str  # e.g., "agent_1", "agent_2"
    character: Character  # the character this agent chose
    reasoning: str  # why this agent chose this character


class ObjectObservation(BaseModel):
    """A structured cross-room observation entry for a single world object."""

    object_id: str
    state: str
    location: str  # room id where this object lives
    notes: list[str] = Field(default_factory=list)  # agent-produced bullet notes
    last_seen_tick: int = 0


class TickAction(BaseModel):
    """A single agent's action + spoken line on one tick of gameplay."""

    tick: int
    agent_id: str
    say: str
    action: str
    target_object: str | None = None  # object id this action operated on, if any
    note: str = ""  # short outcome message (e.g., "unlocked", "no effect")


class PartyState(BaseModel):
    """Shared runtime state of the co-op party."""

    current_room: str = ""
    inventory: list[str] = Field(default_factory=list)  # object ids the party carries
    # current_room/inventory double as the "active perspective" slot for
    # whichever agent is taking its turn — swapped via gameplay's
    # _load_agent_view/_save_agent_view around each agent's action.
    agent_rooms: dict[str, str] = Field(default_factory=dict)  # agent_id -> room_id
    agent_inventories: dict[str, list[str]] = Field(
        default_factory=dict
    )  # agent_id -> object ids that agent personally carries
    visited: set[str] = Field(default_factory=set)
    object_states: dict[str, str] = Field(
        default_factory=dict
    )  # object_id -> current state
    known_info: list[str] = Field(
        default_factory=list
    )  # contains_info tokens discovered
    fuse_states: dict[str, dict[str, str]] = Field(
        default_factory=dict
    )  # panel_id -> {fuse_label: "ON"|"OFF"}
    power_active: set[str] = Field(
        default_factory=set
    )  # power identifiers currently on
    tick: int = 0
    game_over: bool = False
    victory: bool = False
    log: list[TickAction] = Field(default_factory=list)
    spotted_clues: list[str] = Field(default_factory=list)
    observed_rooms: set[str] = Field(
        default_factory=set
    )  # rooms whose entry observation pass is done
    room_observations: dict[str, list[str]] = Field(
        default_factory=dict
    )  # room_id -> observed-object bullets
    room_plans: dict[str, list[str]] = Field(
        default_factory=dict
    )  # room_id -> escape-plan bullets
    last_fingerprint: str | None = None  # room snapshot from end of previous tick
    global_object_observations: dict[str, ObjectObservation] = Field(
        default_factory=dict
    )  # object_id -> latest structured observation across all rooms
    proof_found: bool = False  # storyboard.mystery.proof_object_id has been examined/taken
    deduction_attempts: int = 0  # number of 'accuse' guesses made so far
    wrong_deduction: bool = False  # deduction attempts exhausted without a correct guess
    accusation: str = ""  # the suspect name from the most recent 'accuse' action


class SolverResult(BaseModel):
    """Summary of one LLM solver run, stored on GameState after solver_node runs."""

    won: bool
    ticks: int
    optimal: int
    reward: float
    efficiency: float
    wasted: int
    history: list[str] = Field(default_factory=list)
    # True when the solver exhausted all 'accuse' attempts without the correct
    # suspect (mystery themes only — always False otherwise).
    wrong_deduction: bool = False


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for answer matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class StoryboardMystery(BaseModel):
    """The authoritative mystery facts generated by the storyboard LLM.

    Populated only when the world's theme is a mystery theme. All downstream
    consumers (narrator, deduction validation) read from here.
    """

    victim: str = ""
    killer_name: str = ""
    proof_object_id: str = ""
    motive_hint: str = ""

    def is_empty(self) -> bool:
        return not self.killer_name


class StoryboardSolution(BaseModel):
    """The sealed answer used only for human deduction validation.

    Never passed to the narrator. Populated only for mystery themes.
    """

    question: str = ""
    answer: str = ""
    answer_aliases: list[str] = Field(default_factory=list)
    answer_type: str = "person"
    motive: str = ""
    proof_object: str = ""
    proof_sentence: str = ""
    hint_1: str = ""
    hint_2: str = ""
    victory_narration_hook: str = ""
    defeat_narration_hook: str = ""

    def is_empty(self) -> bool:
        return not self.answer

    def matches(self, human_answer: str) -> bool:
        """True if the human's answer matches the canonical answer or any alias."""
        normalized = _normalize_answer(human_answer)
        if not normalized:
            return False
        canonical = _normalize_answer(self.answer)
        if normalized == canonical:
            return True
        aliases = [_normalize_answer(a) for a in self.answer_aliases]
        if normalized in aliases:
            return True
        return any(
            normalized in alias or alias in normalized
            for alias in aliases
            if alias
        )


class StoryboardPersona(BaseModel):
    """Adapted identity for one character in this specific world genre."""

    world_role: str = ""
    voice: str = ""
    vocabulary: list[str] = Field(default_factory=list)
    # Few-shot style anchors for the dialogue model: example lines in this
    # character's register. Far more effective than abstract style adjectives
    # for local models — they imitate examples, not descriptions.
    sample_lines: list[str] = Field(default_factory=list)


class Storyboard(BaseModel):
    """Narrative layer for one generated world.

    Produced by storyboard_builder after puzzle_builder. ``mystery``,
    ``solution``, and ``suspects`` are populated only for mystery themes;
    other fields (plot, adapted_personas, room_stories, discovery_beats,
    phase_guidance, conversation_seeds, ending_guidance) are genre-agnostic.
    """

    world_id: str = ""
    generated_at: str = ""
    schema_version: str = "1"

    plot_victim: str = ""
    plot_threat: str = ""
    plot_stakes: str = ""
    plot_timeline: str = ""
    plot_tension_hint: str = ""
    plot_atmosphere: str = ""
    plot_protagonist_context: str = ""

    adapted_personas: dict[str, StoryboardPersona] = Field(default_factory=dict)
    room_stories: dict[str, str] = Field(default_factory=dict)
    discovery_beats: dict[str, str] = Field(default_factory=dict)
    phase_guidance: dict[str, str] = Field(default_factory=dict)
    conversation_seeds: dict[str, list[str]] = Field(default_factory=dict)
    ending_guidance: dict[str, str] = Field(default_factory=dict)

    # Mystery-only fields — empty/default for non-mystery themes.
    suspects: list[dict] = Field(default_factory=list)
    mystery: StoryboardMystery = Field(default_factory=StoryboardMystery)
    solution: StoryboardSolution = Field(default_factory=StoryboardSolution)

    def is_empty(self) -> bool:
        return not any([self.plot_victim, self.plot_threat, self.room_stories])

    def lore_excerpt(
        self,
        *,
        actor_name: str = "",
        room_id: str = "",
        object_id: str = "",
    ) -> str:
        """Compact plot context for a single narrator beat."""
        parts: list[str] = []
        if self.plot_victim:
            parts.append(f"THE VICTIM: {self.plot_victim}")
        if self.plot_tension_hint:
            parts.append(f"TENSION: {self.plot_tension_hint}")
        if self.plot_atmosphere:
            parts.append(f"ATMOSPHERE: {self.plot_atmosphere}")
        if actor_name:
            persona = self.adapted_personas.get(actor_name)
            if persona and persona.voice:
                parts.append(f"VOICE ({actor_name}): {persona.voice}")
        if room_id and room_id in self.room_stories:
            parts.append(f"THIS ROOM: {self.room_stories[room_id]}")
        if object_id and object_id in self.discovery_beats:
            parts.append(f"THIS OBJECT: {self.discovery_beats[object_id]}")
        return "\n".join(parts)

    def persona_for(self, name: str) -> StoryboardPersona | None:
        return self.adapted_personas.get(name)

    def discovery_beat(self, object_id: str) -> str:
        return self.discovery_beats.get(object_id, "")

    def room_story(self, room_id: str) -> str:
        return self.room_stories.get(room_id, "")

    def phase_for_turn(self, turn: int, total_turns: int = 40) -> str:
        """Map a turn number to a story phase name."""
        ratio = turn / max(total_turns, 1)
        if ratio < 0.25:
            return "establishing"
        if ratio < 0.6:
            return "investigating"
        if ratio < 0.85:
            return "converging"
        return "revealing"

    def phase_guidance_for_turn(self, turn: int, total_turns: int = 40) -> str:
        phase = self.phase_for_turn(turn, total_turns)
        return self.phase_guidance.get(phase, "")

    def ending_text(self, won: bool, wrong_deduction: bool = False) -> str:
        if wrong_deduction:
            return self.ending_guidance.get("lost_by_wrong_deduction", "")
        return self.ending_guidance.get("won" if won else "lost", "")

    def pop_seed(self, actor_name: str, used: set[str]) -> str | None:
        """Return the next unused conversation seed for this actor, or None."""
        for seed in self.conversation_seeds.get(actor_name, []):
            if seed not in used:
                return seed
        return None

    def suspects_context(self) -> str:
        """Return a formatted suspects list for the narrator — without flagging who is guilty."""
        if not self.suspects:
            return ""
        lines = []
        for s in self.suspects:
            if not isinstance(s, dict):
                continue
            name = s.get("name", "")
            connection = s.get("connection_to_victim", "")
            motive = s.get("apparent_motive", "")
            if name:
                entry = f"- {name}: {connection}"
                if motive:
                    entry += f" Apparent motive: {motive}"
                lines.append(entry)
        return "\n".join(lines)


class GameState(BaseModel):
    """Top-level LangGraph state."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    theme: str = "mystery"
    world: GameWorld | None = None
    characters: list[Character] = Field(default_factory=list)
    party: list[PartyMember] = Field(default_factory=list)
    party_state: PartyState | None = None
    solver_result: SolverResult | None = None
    solve: bool = False  # when True, solver_node runs after puzzle_builder
    storyboard: Storyboard | None = None
