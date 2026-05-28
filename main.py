"""Main entrypoint — runs game_master once and prints output."""

from graph import graph
from state import GameState


def run():
    state = GameState(theme="pirate")

    result = graph.invoke(state)

    messages = result.get("messages", [])
    if messages:
        print(f"\n[Game Master]: {messages[-1].content}\n")


if __name__ == "__main__":
    run()
