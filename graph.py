"""LangGraph wiring — single game_master node for testing."""

from langgraph.graph import END, StateGraph

from agents.game_master import game_master_node
from state import GameState


def route(state: GameState) -> str:
    if state.game_over:
        return END
    return END  # single-turn: stop after each node so main.py can prompt the player


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("game_master", game_master_node)
    builder.set_entry_point("game_master")
    builder.add_conditional_edges("game_master", route, {END: END})
    return builder.compile()


graph = build_graph()
