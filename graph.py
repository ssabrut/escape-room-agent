"""LangGraph wiring — generation + solving pipeline.

world_builder -> puzzle_builder -> solver -> END
"""

from langgraph.graph import END, StateGraph

from agents.world_builder import world_builder_node
from agents.puzzle_builder_node import puzzle_builder_node
from agents.solver_node import solver_node
from state import GameState


def build_graph(num_players: int = 1) -> StateGraph:
    """Build the graph: world_builder -> puzzle_builder -> solver -> END.

    ``num_players`` is accepted for backward compatibility but no longer used.
    """
    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("puzzle_builder", puzzle_builder_node)
    builder.add_node("solver", solver_node)

    builder.set_entry_point("world_builder")
    builder.add_edge("world_builder", "puzzle_builder")
    builder.add_edge("puzzle_builder", "solver")
    builder.add_edge("solver", END)

    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
