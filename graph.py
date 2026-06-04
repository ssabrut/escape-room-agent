"""LangGraph wiring."""

from langgraph.graph import END, StateGraph

from agents.character_master_node import character_master_node
from agents.game_master import world_builder_node
from agents.game_master_eval import game_master_eval_node, route_after_eval
from agents.gameplay_node import gameplay_node
from agents.player_agent_node import player_agent_1_node, player_agent_2_node
from agents.puzzle_builder_node import puzzle_builder_node
from state import GameState


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("puzzle_builder", puzzle_builder_node)
    builder.add_node("character_master", character_master_node)
    builder.add_node("player_agent_1", player_agent_1_node)
    builder.add_node("player_agent_2", player_agent_2_node)
    builder.add_node("gameplay", gameplay_node)
    builder.add_node("game_master_eval", game_master_eval_node)

    builder.set_entry_point("world_builder")

    builder.add_edge("world_builder", "puzzle_builder")
    builder.add_edge("puzzle_builder", "character_master")
    builder.add_edge("character_master", "player_agent_1")
    builder.add_edge("player_agent_1", "player_agent_2")
    builder.add_edge("player_agent_2", "gameplay")
    builder.add_edge("gameplay", "game_master_eval")
    builder.add_conditional_edges(
        "game_master_eval",
        route_after_eval,
        {"gameplay": "gameplay", "end": END},
    )

    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
