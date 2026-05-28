"""Main entrypoint — runs game_master once and prints output."""

from graph import graph
from state import GameState
from visualization import render_room_layout


def run():
    state = GameState(theme="pirate")

    result = graph.invoke(state)

    rooms = result.get("room_layout", [])

    print("\n" + "=" * 94)
    print(" ESCAPE ROOM MAP")
    print("=" * 94 + "\n")

    if rooms:
        render_room_layout(rooms)
    else:
        print("  [No room layout could be parsed from the LLM response]\n")

    messages = result.get("messages", [])
    if messages:
        print("\n[Game Master Narrative]:")
        print(messages[-1].content)


if __name__ == "__main__":
    run()
