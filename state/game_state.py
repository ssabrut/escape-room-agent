from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field


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
    """The target object state that ends the game in victory.

    Not stored on its own anymore — :attr:`GameWorld.win_condition` derives this
    from the final room's ``goal_completion`` so the win target lives in exactly
    one place and cannot drift from the room it belongs to.
    """

    object_id: str = ""
    state: str = ""


class GameWorld(BaseModel):
    """The frozen dungeon blueprint produced by the game master."""

    scenario: str = ""
    objective: str = ""
    rooms: list[Room] = Field(default_factory=list)
    objects: list[WorldObject] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    solution_path: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_condition(self) -> WinCondition:
        """The game-ending target, derived from the final room's goal_completion.

        The win condition is, by construction (see the generation prompt), the
        last room's ``object_state`` goal. Deriving it here keeps a single source
        of truth instead of storing the same object_id/state twice. Falls back to
        an empty WinCondition when there is no usable final ``object_state`` goal.
        """
        for room in reversed(self.rooms):
            gc = room.goal_completion
            if gc is not None and gc.type == "object_state" and gc.object_id:
                return WinCondition(object_id=gc.object_id, state=gc.state or "")
        return WinCondition()


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


class GameState(BaseModel):
    """Top-level LangGraph state."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    theme: str = "mystery"
    world: GameWorld | None = None
    characters: list[Character] = Field(default_factory=list)
    party: list[PartyMember] = Field(default_factory=list)
    party_state: PartyState | None = None
