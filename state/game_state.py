from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Puzzle(BaseModel):
    id: str
    title: str
    description: str
    solution: str
    difficulty: str = "medium"
    solved: bool = False


class Clue(BaseModel):
    id: str
    description: str
    puzzle_id: str


class Role(BaseModel):
    id: str
    name: str
    description: str
    starting_items: list[str] = Field(default_factory=list)


class Item(BaseModel):
    id: str
    name: str
    description: str
    location: str = ""


class RoomItem(BaseModel):
    name: str
    description: str


class Room(BaseModel):
    name: str
    description: str
    adjacency: dict[str, str] = Field(default_factory=dict)
    items: list[RoomItem] = Field(default_factory=list)


class HintRecord(BaseModel):
    puzzle_id: str
    hint_text: str
    level: int  # 1=vague, 2=moderate, 3=direct


class NarrativeChunk(BaseModel):
    role: str  # "intro", "transition", "outro"
    text: str


class GameState(BaseModel):
    # LangGraph message history (append-only via reducer)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Game metadata
    theme: str = "mystery"

    # Active game content
    current_puzzle: Puzzle | None = None
    puzzles: list[Puzzle] = Field(default_factory=list)
    clues: list[Clue] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    hints_used: list[HintRecord] = Field(default_factory=list)
    narrative_log: list[NarrativeChunk] = Field(default_factory=list)

    # Routing / flow control
    next_agent: str = "game_master"
    game_over: bool = False

    room_layout: list[Room] = Field(default_factory=list)

    # Scratch space for inter-agent data
    context: dict[str, Any] = Field(default_factory=dict)
