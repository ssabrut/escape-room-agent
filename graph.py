"""LangGraph wiring."""

from langgraph.graph import END, StateGraph

from agents.character_master_node import character_master_node
from agents.game_master import game_master_node
from agents.gameplay_node import gameplay_node
from agents.mission_master_node import mission_master_node
from agents.player_agent_node import player_agent_1_node, player_agent_2_node
from state import GameState


def build_graph() -> StateGraph:
    builder = StateGraph(GameState)
    builder.add_node("game_master", game_master_node)
    builder.add_node("character_master", character_master_node)
    builder.add_node("player_agent_1", player_agent_1_node)
    builder.add_node("player_agent_2", player_agent_2_node)
    builder.add_node("mission_master", mission_master_node)
    builder.add_node("gameplay", gameplay_node)
    builder.set_entry_point("game_master")
    builder.add_edge("game_master", "character_master")
    builder.add_edge("character_master", "player_agent_1")
    builder.add_edge("player_agent_1", "player_agent_2")
    builder.add_edge("player_agent_2", "mission_master")
    builder.add_edge("mission_master", "gameplay")
    builder.add_edge("gameplay", END)
    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
