"""Utility tools shared across agents."""

from langchain_core.tools import tool

from state import GameState


@tool
def check_solution(player_answer: str, correct_solution: str) -> bool:
    """Check whether the player's answer matches the puzzle solution."""
    return player_answer.strip().lower() == correct_solution.strip().lower()


@tool
def list_puzzles(puzzles: list[dict]) -> str:
    """Return a formatted summary of all puzzles and their solved status."""
    if not puzzles:
        return "No puzzles generated yet."
    lines = []
    for p in puzzles:
        status = "✓" if p.get("solved") else "○"
        lines.append(f"[{status}] {p['title']} ({p['difficulty']})")
    return "\n".join(lines)


@tool
def get_hint_count(puzzle_id: str, hints_used: list[dict]) -> int:
    """Return how many hints have been used for a specific puzzle."""
    return sum(1 for h in hints_used if h.get("puzzle_id") == puzzle_id)
