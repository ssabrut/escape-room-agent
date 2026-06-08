"""LangGraph wiring — generation pipeline (world_builder -> puzzle_builder).

The live multi-agent gameplay (character_master, player agents, gameplay loop) has
been removed; solving is now handled separately by the deterministic oracle and the
LLM solver agent (agents.solver_agent), which run against a finished world via
benchmark.engine.HeadlessEpisode rather than through this graph.
"""

from langgraph.graph import END, StateGraph

from agents.world_builder import world_builder_node
from agents.puzzle_builder_node import puzzle_builder_node
from state import GameState


def build_graph(num_players: int = 1) -> StateGraph:
    """Build the generation graph: world_builder -> puzzle_builder -> END.

    ``num_players`` is accepted for backward compatibility but no longer used
    (there are no player-agent nodes).
    """
    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("puzzle_builder", puzzle_builder_node)

    builder.set_entry_point("world_builder")
    builder.add_edge("world_builder", "puzzle_builder")
    builder.add_edge("puzzle_builder", END)

    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
