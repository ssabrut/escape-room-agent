from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RoomItem(BaseModel):
    name: str
    description: str


class Room(BaseModel):
    """Static blueprint of a room — never mutated after generation."""

    name: str
    description: str
    adjacency: dict[str, str] = Field(default_factory=dict)
    items: list[RoomItem] = Field(default_factory=list)


class Gate(BaseModel):
    """A single step in the game flow sequence."""

    room: str
    requires: str | None = None   # None means freely accessible
    unlocks: str                  # what this gate's completion opens up


class GameFlow(BaseModel):
    """Ordered sequence of gates from start to win."""

    starting_room: str = ""
    win_condition: str = ""
    gates: list[Gate] = Field(default_factory=list)


class GameWorld(BaseModel):
    """The frozen dungeon blueprint produced by the game master."""

    title: str = ""
    setup: str = ""
    atmosphere: str = ""
    objective: str = ""
    rooms: list[Room] = Field(default_factory=list)
    game_flow: GameFlow = Field(default_factory=GameFlow)


class PlayerState(BaseModel):
    """Mutable runtime state — updated as the player acts."""

    current_room: str = ""
    inventory: list[RoomItem] = Field(default_factory=list)
    visited: set[str] = Field(default_factory=set)
    # Items still present in each room, keyed by room name
    items_remaining: dict[str, list[RoomItem]] = Field(default_factory=dict)
    turn_count: int = 0
    game_over: bool = False


class Character(BaseModel):
    """A playable character generated for this escape room."""

    name: str
    role: str
    backstory: str
    special_trait: str


class PartyMember(BaseModel):
    """A player agent's character selection with reasoning."""

    agent_id: str         # e.g., "agent_1", "agent_2"
    character: Character  # the character this agent chose
    reasoning: str        # why this agent chose this character


class Mission(BaseModel):
    """An interactive mission the player must complete to progress through a room."""

    room: str
    gate_index: int
    description: str          # narrative task description shown to the player
    required_actions: list[str]  # interaction keywords that count as completing the mission
    reward_item: str          # existing room item awarded on completion
    unlocks_exit_to: str      # room name that becomes accessible after completion


class GameState(BaseModel):
    """Top-level LangGraph state."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    theme: str = "mystery"
    world: GameWorld | None = None
    player: PlayerState | None = None
    characters: list[Character] = Field(default_factory=list)
    party: list[PartyMember] = Field(default_factory=list)
    missions: list[Mission] = Field(default_factory=list)
