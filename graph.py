"""LangGraph wiring."""

from langgraph.graph import END, StateGraph

from agents.character_master_node import character_master_node
from agents.game_master import game_master_node
from state import GameState


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("game_master", game_master_node)
    builder.add_node("character_master", character_master_node)
    builder.set_entry_point("game_master")
    builder.add_edge("game_master", "character_master")
    builder.add_edge("character_master", END)
    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
