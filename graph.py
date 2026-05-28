"""LangGraph wiring."""

from langgraph.graph import END, StateGraph

from agents.game_master import game_master_node
from agents.task_decomposer import task_decomposer_node
from state import GameState


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("game_master", game_master_node)
    builder.add_node("task_decomposer", task_decomposer_node)
    builder.set_entry_point("game_master")
    builder.add_edge("game_master", "task_decomposer")
    builder.add_edge("task_decomposer", END)
    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
