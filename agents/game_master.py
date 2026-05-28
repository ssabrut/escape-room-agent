"""Game Master agent — sole node; generates all narrative dynamically."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config.settings import get_llm
from prompts import load_prompt
from state import GameState

SYSTEM_PROMPT = load_prompt("game_master", "system")
GENERATION_PROMPT = load_prompt("game_master", "generation")


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
