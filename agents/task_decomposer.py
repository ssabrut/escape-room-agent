"""Task Decomposer — extracts structured puzzles, clues, roles, and items from the generated scenario."""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_extractor_llm
from state import Clue, GameState, Item, Puzzle, Role

SYSTEM_PROMPT = "You are a structured data extractor. Output only valid JSON, no explanation, without preamble and postamble."

EXTRACTION_PROMPT = """Extract the puzzles, clues, roles, and items from the escape room scenario below.

Return a single JSON object with exactly these keys:
{{
  "puzzles": [
    {{"id": "p1", "title": "...", "description": "...", "solution": "...", "difficulty": "easy|medium|hard"}}
  ],
  "clues": [
    {{"id": "c1", "description": "...", "puzzle_id": "p1"}}
  ],
  "roles": [
    {{"id": "r1", "name": "...", "description": "...", "starting_items": ["..."]}}
  ],
  "items": [
    {{"id": "i1", "name": "...", "description": "...", "location": "..."}}
  ]
}}

Scenario:
{scenario}"""


def task_decomposer_node(state: GameState) -> dict:
    messages = state.messages
    if not messages:
        return {}

    scenario = messages[-1].content
    llm = get_extractor_llm()

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=EXTRACTION_PROMPT.format(scenario=scenario)),
        ]
    )

    raw = response.content.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    data = json.loads(raw)

    puzzles = [Puzzle(**p) for p in data.get("puzzles", [])]
    clues = [Clue(**c) for c in data.get("clues", [])]
    roles = [Role(**r) for r in data.get("roles", [])]
    items = [Item(**i) for i in data.get("items", [])]

    print("\n" + "=" * 60)
    print("TASK DECOMPOSER EXTRACTED:")
    print("=" * 60)
    print(f"Puzzles ({len(puzzles)}):")
    for p in puzzles:
        print(f"  [{p.id}] {p.title} — {p.description} | answer: {p.solution}")
    print(f"Clues ({len(clues)}):")
    for c in clues:
        print(f"  [{c.id}] {c.description} (helps: {c.puzzle_id})")
    print(f"Roles ({len(roles)}):")
    for r in roles:
        print(f"  [{r.id}] {r.name} — {r.description} | items: {r.starting_items}")
    print(f"Items ({len(items)}):")
    for i in items:
        print(f"  [{i.id}] {i.name} — {i.description} | location: {i.location}")
    print("=" * 60 + "\n")

    output = {
        "narrative": scenario,
        "puzzles": [p.model_dump() for p in puzzles],
        "clues": [c.model_dump() for c in clues],
        "roles": [r.model_dump() for r in roles],
        "items": [i.model_dump() for i in items],
    }
    with open("game_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved: game_output.json\n")

    return {"puzzles": puzzles, "clues": clues, "roles": roles, "items": items}
