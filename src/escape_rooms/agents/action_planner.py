"""ActionPlanner — short-horizon beam search over cloned PartyState.

Ported from Escapee's ``app/cognition/action_planner.py``. The planner evaluates
``cognition.policy_candidates(...)`` with forward search on cloned
``PartyState`` snapshots (``ps.model_copy(deep=True)`` stands in for Escapee's
``deepcopy(state)``), scoring transitions via ``_resolve_action`` +
``_check_victory`` — escape-rooms' existing deterministic engine functions, reused
unchanged.

Returns ranked actions plus a best action the turn-loop can use for stall
fallback / progress-gate enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.escape_rooms.agents.cognition import (
    TeamCognition,
    _FREE_VERBS,
    action_signature,
    compute_milestones,
    derive_board,
)
from src.escape_rooms.nodes.gameplay import _check_victory, _resolve_action
from src.escape_rooms.state import GameWorld, PartyState

# Note substrings meaning the action made no progress (mirrors
# `solver_agent._NO_PROGRESS`).
_NO_PROGRESS = ("no hidden info", "is locked", "unknown", "idle", "no direct route",
                "already", "code unknown", "missing tool", "no matching liquid",
                "still locked", "no fuse", "needs no", "cannot take", "no target")


def _is_free(action: str) -> bool:
    return action.split(" ", 1)[0] in _FREE_VERBS


def _action_succeeded(note: str) -> bool:
    lnote = note.lower()
    return not any(p in lnote for p in _NO_PROGRESS)


@dataclass(frozen=True)
class PlannerOption:
    action: str
    score: float
    rationale: str


@dataclass(frozen=True)
class PlannerDecision:
    best_action: str
    ranked: list[PlannerOption] = field(default_factory=list)

    @property
    def best_non_free_action(self) -> str | None:
        for option in self.ranked:
            if not _is_free(option.action):
                return option.action
        return None

    @property
    def best_progress_action(self) -> str | None:
        for option in self.ranked:
            if not _is_free(option.action) and option.score > 0:
                return option.action
        return None

    @property
    def top_score(self) -> float:
        """Score of the single highest-ranked option (free or not)."""
        return self.ranked[0].score if self.ranked else 0.0

    def score_of(self, action: str) -> float | None:
        sig = action_signature(action)
        for option in self.ranked:
            if action_signature(option.action) == sig:
                return option.score
        return None


@dataclass
class ActionPlanner:
    """Deterministic planner for fallback and anti-stall escalation."""

    depth: int = 3
    beam_width: int = 5
    discount: float = 0.75

    def plan_turn(
        self,
        world: GameWorld,
        ps: PartyState,
        cognition: TeamCognition,
        current_goal: str,
        next_plan_step: str = "",
    ) -> PlannerDecision:
        actions = cognition.policy_candidates(world, ps, current_goal, next_plan_step)
        if not actions:
            fallback = "wait"
            return PlannerDecision(
                best_action=fallback,
                ranked=[PlannerOption(action=fallback, score=-1.0, rationale="No deterministic candidates")],
            )

        ranked: list[PlannerOption] = []
        for action in actions[: self.beam_width]:
            immediate, next_ps, note = self._simulate(world, ps, action)
            success = _action_succeeded(note)
            won = _check_victory(world, next_ps)
            future = 0.0
            if self.depth > 1 and success and not won and not next_ps.game_over:
                future = self._search(
                    world=world,
                    ps=next_ps,
                    cognition=cognition,
                    current_goal=current_goal,
                    next_plan_step=next_plan_step,
                    depth=self.depth - 1,
                    path={action_signature(action)},
                )
            score = immediate + (self.discount * future)
            ranked.append(
                PlannerOption(
                    action=action,
                    score=score,
                    rationale=f"{'success' if success else 'failed'}; score={score:.1f}; action={action}",
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return PlannerDecision(best_action=ranked[0].action, ranked=ranked[:3])

    def immediate_value(
        self,
        world: GameWorld,
        ps: PartyState,
        action: str,
        cognition: TeamCognition,
    ) -> float:
        """One-step transition score for an arbitrary action.

        Used by the turn-loop's progress gate to judge how productive the
        agent's chosen action is, even when it falls outside the ranked top
        options (e.g. examining a decoy).
        """
        immediate, _next_ps, _note = self._simulate(world, ps, action)
        return immediate

    def _search(
        self,
        *,
        world: GameWorld,
        ps: PartyState,
        cognition: TeamCognition,
        current_goal: str,
        next_plan_step: str,
        depth: int,
        path: set[str],
    ) -> float:
        if depth <= 0 or ps.game_over:
            return 0.0

        actions = cognition.policy_candidates(world, ps, current_goal, next_plan_step)
        if not actions:
            return 0.0

        best = float("-inf")
        for action in actions[: self.beam_width]:
            sig = action_signature(action)
            repeat_penalty = 2.5 if sig in path else 0.0
            immediate, next_ps, note = self._simulate(world, ps, action)
            success = _action_succeeded(note)
            won = _check_victory(world, next_ps)
            score = immediate - repeat_penalty
            if depth > 1 and success and not won and not next_ps.game_over:
                score += self.discount * self._search(
                    world=world,
                    ps=next_ps,
                    cognition=cognition,
                    current_goal=current_goal,
                    next_plan_step=next_plan_step,
                    depth=depth - 1,
                    path=path | {sig},
                )
            if score > best:
                best = score
        return 0.0 if best == float("-inf") else best

    def _simulate(
        self, world: GameWorld, ps: PartyState, action: str
    ) -> tuple[float, PartyState, str]:
        next_ps = ps.model_copy(deep=True)

        before_board = derive_board(world, ps)
        before_ms = compute_milestones(world, ps, ps)

        note, _target = _resolve_action(action, world, next_ps)

        after_board = derive_board(world, next_ps)
        after_ms = compute_milestones(world, ps, next_ps)

        won = _check_victory(world, next_ps)
        score = self._transition_score(
            action=action,
            note=note,
            unsolved_before=len(before_board.unsolved),
            unsolved_after=len(after_board.unsolved),
            new_ms=len(after_ms - before_ms),
            won=won,
        )
        return score, next_ps, note

    def _transition_score(
        self,
        *,
        action: str,
        note: str,
        unsolved_before: int,
        unsolved_after: int,
        new_ms: int,
        won: bool,
    ) -> float:
        score = 0.0
        success = _action_succeeded(note)
        if won:
            score += 120.0
        score += 3.0 if success else -5.0

        if _is_free(action):
            score -= 2.0

        if unsolved_after < unsolved_before:
            score += 18.0 * (unsolved_before - unsolved_after)
        elif unsolved_after > unsolved_before:
            score -= 4.0

        if new_ms > 0:
            score += 8.0 * new_ms
        elif success and not _is_free(action):
            score -= 1.0

        return score
