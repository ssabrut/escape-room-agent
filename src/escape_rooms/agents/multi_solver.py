"""Cognitive solver policy — TeamCognition + ActionPlanner turn loop.

Ported from Escapee's ``GameOrchestrator._take_turn``, simplified for a single
actor (escape-rooms' solver is currently single-agent — see ``cognition.py``'s
module docstring for the multi-agent extension point).

Exposes ``cognitive_solver_policy(...)``, a drop-in alternative to
``solver_agent.react_solver_policy`` with the same
``(world, ps, action_space) -> action`` contract, so it runs under
``benchmark.engine.HeadlessEpisode`` exactly like the ReAct policy.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.escape_rooms.agents.action_planner import ActionPlanner, _action_succeeded, _is_free
from src.escape_rooms.agents.cognition import (
    CognitionConfig,
    TeamCognition,
    derive_board,
    derive_current_goal,
)
from src.escape_rooms.nodes.gameplay import (
    IDLE_ACTION,
    _objects_in_room,
    _parse_json,
    _resolve_choice,
)
from src.escape_rooms.state import GameWorld, PartyState
from src.escape_rooms.utils.settings import get_llm

from src.escape_rooms.agents.solver_agent import REACT_SYSTEM, _build_prompt

# Progress-gate thresholds (ported from Escapee's orchestrator constants).
_PROGRESS_GATE_DOMINANT_MIN = 90.0
_PROGRESS_GATE_GAP = 30.0
_PROGRESS_GATE_LOW_VALUE = 3.0
_PROGRESS_GATE_EXEMPT_VERBS = {"enter_code", "use_tool", "insert_liquid", "flip_fuse"}


def cognitive_solver_policy(
    role: str = "solver",
    scratchpad_limit: int = 30,
    trace: list | None = None,
    debug_log: list[dict] | None = None,
    *,
    enforce_candidate_policy: bool = True,
):
    """Cognitive policy: TeamCognition brief + ActionPlanner recommendation each tick.

    Carries the same scratchpad/dead-ends/cycle-detection machinery as
    ``react_solver_policy``, plus a deterministic cognition layer that compiles
    policy candidates, derives the current goal/plan, and gates the LLM's choice
    against a short-horizon beam search.

    If ``debug_log`` is given, one dict per tick is appended to it, capturing the
    LLM's raw thought/plan, the planner's ranked candidates, any gate overrides,
    and the final action — everything needed to debug a run after the fact.
    """
    llm = get_llm(role)
    cognition = TeamCognition(config=CognitionConfig())
    planner = ActionPlanner()

    scratchpad: list[str] = []
    dead_ends: set[str] = set()
    blocked_exits: set[str] = set()
    current_plan: list[str] = ["(none yet)"]
    action_history: list[str] = []
    _CYCLE_REPS = 3
    _MAX_PERIOD = 4
    state = {"seen_log": 0, "prev_ps": None, "prev_action": None}

    def _is_cycling(history: list[str]) -> bool:
        for period in range(1, _MAX_PERIOD + 1):
            needed = period * _CYCLE_REPS
            if len(history) < needed:
                continue
            tail = history[-needed:]
            pattern = tail[:period]
            if tail == pattern * _CYCLE_REPS:
                return True
        return False

    def _policy(world: GameWorld, ps: PartyState, action_space: list[str]) -> str:
        if not action_space:
            return IDLE_ACTION

        # Ingest engine log (dead-ends/blocked exits), same as react_solver_policy.
        while state["seen_log"] < len(ps.log):
            entry = ps.log[state["seen_log"]]
            note = entry.note
            if note:
                parts = entry.action.split()
                if "no hidden info" in note and len(parts) >= 2:
                    dead_ends.add(parts[1])
                if "is locked" in note and len(parts) >= 2 and parts[0] == "go":
                    blocked_exits.add(parts[1])
            state["seen_log"] += 1

        # Lazily observe the outcome of the action chosen on the previous tick,
        # diffing the snapshot taken then against the current state. This keeps
        # the (world, ps, action_space) -> action contract intact: HeadlessEpisode
        # resolves the action *after* the policy returns, so we can't observe the
        # post-action state until the next call.
        prev_outcome = None
        new_milestones: list[str] = []
        if state["prev_ps"] is not None and state["prev_action"] is not None:
            last_entry = ps.log[-1] if ps.log else None
            last_note = last_entry.note if last_entry else ""
            success = _action_succeeded(last_note)
            update = cognition.observe(
                state["prev_action"], last_note, state["prev_ps"], ps, world, ps.tick, success
            )
            prev_outcome = {
                "action": state["prev_action"],
                "note": last_note,
                "success": success,
            }
            if update.new_milestones:
                new_milestones = update.new_milestones
                scratchpad.append(f"Milestones: {', '.join(update.new_milestones)}")

        board = derive_board(world, ps)
        current_goal = cognition._derive_positional_goal(world, ps, board)
        next_plan_step = cognition._next_plan_step()

        decision = planner.plan_turn(world, ps, cognition, current_goal, next_plan_step)

        critical_note = cognition.critical_guidance_for(world, ps, ps.tick)
        reflect = cognition.should_reflect(ps.tick)
        brief = cognition.brief_for(
            world, ps,
            reflect=reflect,
            critical_note=critical_note,
            recommended_action=decision.best_action,
        )

        visible = _objects_in_room(world, ps)

        pad_lines = [f"Current plan: {current_plan[0]}"] + scratchpad[-scratchpad_limit:]
        pad = "\n".join(pad_lines)
        prompt = (
            _build_prompt(world, ps, action_space, visible, dead_ends, blocked_exits)
            + f"\nTEAM COGNITION BRIEF:\n{brief.render()}\n"
            + f"\nSCRATCHPAD (plan + recent thought -> action -> observation):\n{pad}\n"
        )

        try:
            response = llm.invoke(
                [SystemMessage(content=REACT_SYSTEM), HumanMessage(content=prompt)]
            )
            data = _parse_json(response.content) or {}
            thought = str(data.get("thought", "")).strip()
            plan = str(data.get("plan", "")).strip()
            raw_action = str(data.get("action", "")).strip()
            action = _resolve_choice(raw_action, action_space)
        except Exception:
            thought, plan, raw_action, action = "(parse error)", "", "", action_space[0]

        llm_action = action
        gate_notes: list[str] = []

        # --- Gates (ported from GameOrchestrator._take_turn, single-agent) ---
        gate_note = None

        # Stall override: stuck + chose a free action -> force planner's best
        # progress action.
        if cognition.stuck and _is_free(action):
            forced = decision.best_progress_action
            if forced and forced in action_space:
                gate_note = f"[stall override] forcing '{forced}' (was '{action}')"
                action = forced

        # Out-of-policy gate: chosen action isn't a compiled candidate and a
        # PROGRESS candidate exists -> prefer the candidate.
        if (
            gate_note is None
            and enforce_candidate_policy
            and not cognition.is_policy_candidate(world, ps, current_goal, action, next_plan_step)
        ):
            forced = decision.best_progress_action
            if forced and forced in action_space and forced != action:
                gate_note = f"[policy gate] forcing '{forced}' (was '{action}', off-policy)"
                action = forced

        # Progress gate: planner is dominant and the chosen action is low-value
        # -> force the high-value action, exempting genuine puzzle attempts.
        if gate_note is None:
            chosen_verb = action.split(" ", 1)[0]
            if (
                decision.ranked
                and chosen_verb not in _PROGRESS_GATE_EXEMPT_VERBS
                and decision.top_score >= _PROGRESS_GATE_DOMINANT_MIN
            ):
                chosen_value = planner.immediate_value(world, ps, action, cognition)
                gap = decision.top_score - chosen_value
                if gap >= _PROGRESS_GATE_GAP and chosen_value <= _PROGRESS_GATE_LOW_VALUE:
                    forced = decision.best_progress_action
                    if forced and forced in action_space and forced != action:
                        gate_note = f"[progress gate] forcing '{forced}' (was '{action}', low value)"
                        action = forced

        if gate_note:
            gate_notes.append(gate_note)
            scratchpad.append(gate_note)
            if trace is not None:
                trace.append(f"t{ps.tick} {gate_note}")

        # Cycle guard (verbatim from react_solver_policy).
        action_history.append(action)
        if _is_cycling(action_history):
            cycle_set = set(action_history[-(2 * _MAX_PERIOD):])
            alternatives = [a for a in action_space if a not in cycle_set]
            if not alternatives:
                alternatives = [a for a in action_space if a != action]
            if alternatives:
                forced = alternatives[0]
                cycle_note = (
                    f"[override] cycle detected {action_history[-(2*_MAX_PERIOD):]} — forcing '{forced}'"
                )
                gate_notes.append(cycle_note)
                scratchpad.append(cycle_note)
                if trace is not None:
                    trace.append(f"t{ps.tick} {cycle_note}")
                action = forced
                action_history[-1] = forced

        if plan:
            current_plan[0] = plan
        if thought:
            scratchpad.append(f"Thought: {thought}")
            if trace is not None:
                trace.append(f"t{ps.tick} THINK: {thought}")
        scratchpad.append(f"Action: {action}")

        if debug_log is not None:
            debug_log.append({
                "tick": ps.tick,
                "room": ps.current_room,
                "prev_outcome": prev_outcome,
                "new_milestones": new_milestones,
                "current_goal": current_goal,
                "next_plan_step": next_plan_step,
                "stuck": cognition.stuck,
                "reflect": reflect,
                "critical_note": critical_note,
                "planner_recommended": decision.best_action,
                "planner_ranked": [
                    {"action": o.action, "score": o.score, "rationale": o.rationale}
                    for o in decision.ranked
                ],
                "thought": thought,
                "plan": plan,
                "llm_raw_action": raw_action,
                "llm_resolved_action": llm_action,
                "gates_fired": gate_notes,
                "final_action": action,
            })

        state["prev_ps"] = ps.model_copy(deep=True)
        state["prev_action"] = action
        return action

    return _policy


def solve_world_multi(world: GameWorld, role: str = "solver", trace: list | None = None):
    """Run the cognitive solver once. Returns (EpisodeResult, optimal_path_steps)."""
    from benchmark.engine import HeadlessEpisode
    from benchmark.policies import bfs_solution_path

    policy = cognitive_solver_policy(role, trace=trace)
    result = HeadlessEpisode(world).run(policy, record_history=True)
    optimal = bfs_solution_path(world)
    return result, optimal
