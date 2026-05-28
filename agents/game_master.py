"""Game Master agent — sole node; generates all narrative dynamically."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from state import GameState

SYSTEM_PROMPT = (
    """You are the Game Master of a 2D text-based conversational Escape Room game."""
)

GENERATION_PROMPT = """Generate a complete escape room scenario for a 2D text-based conversational game.

Theme: {theme}

Include:
* The narrative (intro + atmosphere)
* Main objective
* Room layout (3–4 rooms)
* Roles / personas (3 total): name, description, starting items
* Items (5 total): name, description, location
* Puzzles (3 total): name, riddle, answer
* Clues (3 total): description, which puzzle it helps
* Hint system: explain how hints are awarded
* Game flow: how the player moves from start to finish"""


def game_master_node(state: GameState) -> dict:
    llm = get_llm()

    prompt = GENERATION_PROMPT.format(theme=state.theme)

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    print("\n" + "=" * 60)
    print("GAME MASTER GENERATED:")
    print("=" * 60)
    print(response.content)
    print("=" * 60 + "\n")

    return {"messages": [AIMessage(content=response.content)]}
