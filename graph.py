"""LangGraph wiring."""

from langgraph.graph import END, StateGraph

from agents.character_master_node import character_master_node
from agents.game_master import world_builder_node
from agents.gameplay_node import gameplay_node, route_after_gameplay
from agents.player_agent_node import make_player_node
from agents.puzzle_builder_node import puzzle_builder_node
from state import GameState


def build_graph(num_players: int = 1) -> StateGraph:
    """Build the game graph with `num_players` sequential player-agent nodes.

    Each player node picks one (distinct) character; running them in sequence lets
    later agents see who has already been chosen and complement the party.
    """
    num_players = max(1, num_players)

    builder = StateGraph(GameState)
    builder.add_node("world_builder", world_builder_node)
    builder.add_node("puzzle_builder", puzzle_builder_node)
    builder.add_node("character_master", character_master_node)

    player_nodes = [f"player_agent_{i}" for i in range(1, num_players + 1)]
    for i, name in enumerate(player_nodes, 1):
        builder.add_node(name, make_player_node(f"agent_{i}"))

    builder.add_node("gameplay", gameplay_node)

    builder.set_entry_point("world_builder")

    builder.add_edge("world_builder", "puzzle_builder")
    builder.add_edge("puzzle_builder", "character_master")

    # character_master -> player_1 -> player_2 -> ... -> gameplay
    builder.add_edge("character_master", player_nodes[0])
    for prev, nxt in zip(player_nodes, player_nodes[1:]):
        builder.add_edge(prev, nxt)
    builder.add_edge(player_nodes[-1], "gameplay")

    builder.add_conditional_edges(
        "gameplay",
        route_after_gameplay,
        {"gameplay": "gameplay", "end": END},
    )

    return builder.compile()


graph = build_graph()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
