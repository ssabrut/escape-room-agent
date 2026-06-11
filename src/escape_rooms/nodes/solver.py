"""LangGraph node that runs the LLM solver on the finished world.

Sits after puzzle_builder in the generation graph. Calls solve_world (the same
ReAct policy used by the CLI), scores the result with objective(), and stores a
SolverResult on GameState so downstream nodes or callers can inspect it.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.escape_rooms.agents.solver_agent import objective, solve_world
from src.escape_rooms.state import GameState, SolverResult
from src.escape_rooms.utils.logging import get_node_logger

log = get_node_logger("solver")


def solver_node(state: GameState) -> dict:
    world = state.world
    if world is None or not world.rooms or not world.win_condition.object_id:
        log.warning("solver_node: no world or win condition — skipping")
        return {"solver_result": None}

    log.info(
        "Starting — win target: {!r} -> {!r}  rooms={}  objects={}",
        world.win_condition.object_id, world.win_condition.state,
        len(world.rooms), len(world.objects),
    )

    trace: list[str] = []
    debug_log: list[dict] = []
    result, optimal = solve_world(world, trace=trace, debug_log=debug_log)
    score = objective(world, result, optimal_len=len(optimal))

    verdict = "ESCAPED" if score["won"] else "FAILED"
    log_fn = log.success if score["won"] else log.warning
    log_fn(
        "{} in {} tick(s)  optimal={}  reward={:.3f}  efficiency={:.3f}  wasted={}",
        verdict, score["ticks"], score["optimal"],
        score["reward"], score["efficiency"], score["wasted"],
    )
    for line in (trace or []):
        log.debug("  trace: {}", line)

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
        "_solver_debug_log": debug_log,
    }
