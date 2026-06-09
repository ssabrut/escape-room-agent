"""LangGraph node that runs the LLM solver on the finished world.

Sits after puzzle_builder in the generation graph. Calls solve_world (the same
ReAct policy used by the CLI), scores the result with objective(), and stores a
SolverResult on GameState so downstream nodes or callers can inspect it.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.solver_agent import objective, solve_world
from state import GameState, SolverResult


def solver_node(state: GameState) -> dict:
    world = state.world
    if world is None or not world.rooms or not world.win_condition.object_id:
        return {"solver_result": None}

    trace: list[str] = []
    result, optimal = solve_world(world, trace=trace)
    score = objective(world, result, optimal_len=len(optimal))

    verdict = "ESCAPED" if score["won"] else "FAILED"
    print(f"\n[solver] {verdict} in {score['ticks']} tick(s) "
          f"(optimal {score['optimal']}, reward {score['reward']})")

    return {
        "solver_result": SolverResult(
            won=score["won"],
            ticks=score["ticks"],
            optimal=score["optimal"],
            reward=score["reward"],
            efficiency=score["efficiency"],
            wasted=score["wasted"],
            history=result.history,
        ),
        "messages": [
            AIMessage(
                content=f"[solver] {verdict} in {score['ticks']} tick(s) | "
                        f"optimal={score['optimal']} reward={score['reward']}"
            )
        ],
    }
