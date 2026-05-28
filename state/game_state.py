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


class GameWorld(BaseModel):
    """The frozen dungeon blueprint produced by the game master."""

    title: str = ""
    setup: str = ""
    atmosphere: str = ""
    objective: str = ""
    rooms: list[Room] = Field(default_factory=list)


class PlayerState(BaseModel):
    """Mutable runtime state — updated as the player acts."""

    current_room: str = ""
    inventory: list[RoomItem] = Field(default_factory=list)
    visited: set[str] = Field(default_factory=set)
    # Items still present in each room, keyed by room name
    items_remaining: dict[str, list[RoomItem]] = Field(default_factory=dict)
    turn_count: int = 0
    game_over: bool = False


class GameState(BaseModel):
    """Top-level LangGraph state."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    theme: str = "mystery"
    world: GameWorld | None = None
    player: PlayerState | None = None
