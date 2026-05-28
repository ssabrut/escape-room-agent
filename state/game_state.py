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
    hints_used: list[HintRecord] = Field(default_factory=list)
    narrative_log: list[NarrativeChunk] = Field(default_factory=list)

    # Routing / flow control
    next_agent: str = "game_master"
    game_over: bool = False

    # Scratch space for inter-agent data
    context: dict[str, Any] = Field(default_factory=dict)
