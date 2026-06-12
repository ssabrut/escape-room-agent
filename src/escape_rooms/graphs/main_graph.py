"""LangGraph wiring — generation + optional solving pipeline.

world_builder -> puzzle_builder -> storyboard_builder -> (solver if state.solve) -> END
"""

from langgraph.graph import END, StateGraph

from src.escape_rooms.nodes.world_builder import world_builder_node
from src.escape_rooms.nodes.puzzle_builder import puzzle_builder_node
from src.escape_rooms.nodes.storyboard_builder import storyboard_builder_node
from src.escape_rooms.nodes.solver import solver_node
from src.escape_rooms.state import GameState


def _route_solver(state: GameState) -> str:
    return "solver" if state.solve else END


def build_graph(num_players: int = 1) -> StateGraph:
    """Build the graph: world_builder -> puzzle_builder -> storyboard_builder -> (solver?) -> END.

    ``num_players`` is accepted for backward compatibility but no longer used.
    The solver node only runs when ``GameState.solve`` is True.
    """
    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("puzzle_builder", puzzle_builder_node)
    builder.add_node("storyboard_builder", storyboard_builder_node)
    builder.add_node("solver", solver_node)

    builder.set_entry_point("world_builder")
    builder.add_edge("world_builder", "puzzle_builder")
    builder.add_edge("puzzle_builder", "storyboard_builder")
    builder.add_conditional_edges("storyboard_builder", _route_solver, ["solver", END])
    builder.add_edge("solver", END)

    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
