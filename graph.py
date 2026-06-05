"""LangGraph wiring."""

from langgraph.graph import StateGraph

from agents.world_builder import world_builder_node
from state import GameState


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.set_entry_point("world_builder")
    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
