"""TeamCognition — deterministic "external brain" for the cognitive solver.

Ported from Escapee's ``app/cognition/team_cognition.py``, adapted to escape-rooms'
types: actions here are plain strings from ``_build_action_space`` (e.g.
``"enter_code keypad_1"``, ``"go room_2"``, ``"take wrench"``), and world state is a
single ``PartyState`` (escape-rooms' solver is single-actor, unlike Escapee's
multi-player ``GameState``).

It tracks what the LLM cannot be trusted to hold over dozens of turns: a symbolic
fingerprint of the world, an episodic memory of every attempt, which moves are now
provably pointless (loop detection), how much real progress has been made
(milestones), and when the agent has stalled.

Extension point: ``CandidateAction.tag == "COORDINATION"`` and per-agent functional
roles (Explorer/Solver/Critic) are intentionally not built — they only matter once
escape-rooms' solver becomes multi-agent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from src.escape_rooms.nodes.gameplay import (
    HIDDEN_STATES,
    IDLE_ACTION,
    OPENED_STATES,
    _build_action_space,
    _format_goal_completion,
    _goal_completion_satisfied,
    _objects_in_room,
    _state_satisfies,
)
from src.escape_rooms.state import GameWorld, PartyState

# Actions that are "free": pure orientation, no world-state mutation. Never
# blocked as redundant — re-examining context is cheap (escape-rooms drops
# repeat examines from the action space anyway via `_already_examined`).
_FREE_VERBS = {"examine", IDLE_ACTION}


@dataclass(frozen=True)
class CandidateAction:
    """Deterministic next-action option exposed to the LLM policy layer."""

    action: str
    tag: str  # PROGRESS | EXPLORATION | COORDINATION


def action_signature(action: str) -> str:
    """Compact, stable label for an action. Action strings are already stable."""
    return action


def describe_action_brief(action: str) -> str:
    """One-line imperative description of an action (for RECOMMENDED ACTION)."""
    parts = action.split()
    if not parts:
        return action
    verb = parts[0]
    if verb == "go" and len(parts) >= 2:
        return f"move to {parts[1]}"
    if verb == IDLE_ACTION:
        return "wait"
    return action


def world_fingerprint(ps: PartyState) -> str:
    """A short, stable hash of the symbolic world state.

    Two states with the same fingerprint are interaction-equivalent: any given
    action yields an identical outcome in both.
    """
    parts: list[str] = []
    for oid in sorted(ps.object_states):
        parts.append(f"s:{oid}={ps.object_states[oid]}")
    parts.append(f"room:{ps.current_room}")
    parts.append("inv:" + ",".join(sorted(ps.inventory)))
    parts.append("info:" + ",".join(sorted(ps.known_info)))
    parts.append("visited:" + ",".join(sorted(ps.visited)))
    parts.append("power:" + ",".join(sorted(ps.power_active)))
    for oid in sorted(ps.fuse_states):
        fuses = ",".join(f"{k}={v}" for k, v in sorted(ps.fuse_states[oid].items()))
        parts.append(f"fuse:{oid}=[{fuses}]")
    blob = "|".join(parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def compute_milestones(world: GameWorld, ps_before: PartyState, ps_after: PartyState) -> set[str]:
    """Derive newly-achieved milestones by diffing two PartyStates."""
    out: set[str] = set()
    for obj in world.objects:
        before_state = ps_before.object_states.get(obj.id, obj.state)
        after_state = ps_after.object_states.get(obj.id, obj.state)
        if after_state in OPENED_STATES and before_state not in OPENED_STATES:
            out.add(f"opened:{obj.id}")
    for oid in ps_after.inventory:
        if oid not in ps_before.inventory:
            out.add(f"took:{oid}")
    for room in ps_after.visited:
        if room not in ps_before.visited:
            out.add(f"reached:{room}")
    for info in ps_after.known_info:
        if info not in ps_before.known_info:
            out.add(f"learned:{info}")
    for flag in ps_after.power_active:
        if flag not in ps_before.power_active:
            out.add(f"power:{flag}")
    return out


def _humanize(token: str) -> str:
    """'opened:supply_locker' -> 'opened supply locker'."""
    kind, _, rest = token.partition(":")
    return f"{kind} {rest.replace('_', ' ')}".strip()


def _requirement_hint(obj) -> str:
    """A NON-spoiler note about what a locked object still needs (no codes leaked)."""
    if obj.requires_code:
        return " (needs a code)"
    if obj.requires_tool:
        return " (needs the right tool used on it)"
    if obj.requires_liquid:
        return " (needs a specific liquid)"
    if obj.requires_power:
        return " (needs power restored first)"
    if obj.fuses is not None:
        return " (a power source — set its fuses)"
    return ""


def _has_requirement(obj) -> bool:
    """True when an object has at least one defined unlock/activation mechanism."""
    return bool(
        obj.requires_code
        or obj.requires_tool
        or obj.requires_liquid
        or obj.requires_power
        or obj.fuses is not None
    )


@dataclass
class SolutionBoard:
    """A deterministic, always-correct snapshot of progress derived from state.

    Recomputed from ``PartyState`` every turn — never trusts LLM memory of what
    it has already accomplished.
    """

    objective: str = ""
    solved: list[str] = field(default_factory=list)
    unsolved: list[str] = field(default_factory=list)

    def render(self) -> str:
        solved = "; ".join(self.solved) or "(nothing yet)"
        todo = "; ".join(self.unsolved) or "(none — you may be ready to win)"
        return (
            f"GOAL: {self.objective}\n"
            f"SOLVED (already done — NEVER redo these): {solved}\n"
            f"STILL TO DO (focus here): {todo}"
        )


def derive_board(world: GameWorld, ps: PartyState) -> SolutionBoard:
    """Compute the SOLVED / STILL-TO-DO board straight from authoritative state."""
    solved: list[str] = []
    unsolved: list[str] = []

    for obj in world.objects:
        cur = ps.object_states.get(obj.id, obj.state)
        if cur in OPENED_STATES and obj.state not in OPENED_STATES:
            solved.append(f"{obj.id} is {cur}")
        elif obj.id in ps.inventory:
            solved.append(f"{obj.id} has been taken")
    for info in sorted(ps.known_info):
        solved.append(f"learned [{info}]")
    for flag in sorted(ps.power_active):
        solved.append(f"power on [{flag}]")
    for room in sorted(ps.visited):
        solved.append(f"reached {room}")

    for room in world.rooms:
        if room.goal_completion is not None and not _goal_completion_satisfied(
            room.goal_completion, ps
        ):
            unsolved.append(
                f"{room.id}: {room.goal or 'unsolved'} "
                f"(needs {_format_goal_completion(room.goal_completion)})"
            )

    win = world.win_condition
    cur_win = ps.object_states.get(win.object_id) if win.object_id else None
    if win.object_id and not _state_satisfies(cur_win, win.state):
        now = cur_win if cur_win else "unknown"
        unsolved.append(
            f"WIN CONDITION: [{win.object_id}] must reach '{win.state}' (now '{now}')"
        )

    return SolutionBoard(
        objective=world.objective or "Escape the room.",
        solved=solved,
        unsolved=unsolved,
    )


def derive_current_goal(board: SolutionBoard) -> str:
    """Pick one actionable goal for this turn from unsolved objectives."""
    for item in board.unsolved:
        if "WIN CONDITION" not in item:
            return item
    if board.unsolved:
        return board.unsolved[0]
    return "Satisfy the win condition now."


def _goal_object_id(current_goal: str) -> str | None:
    """Extract a leading object id from goal lines of various formats."""
    if "WIN CONDITION:" in current_goal and "[" in current_goal and "]" in current_goal:
        try:
            return current_goal.split("[", 1)[1].split("]", 1)[0].strip()
        except Exception:
            return None
    if current_goal.startswith("take ") and " from " in current_goal:
        return current_goal.split("take ", 1)[1].split(" from ")[0].strip()
    if " is still " in current_goal:
        return current_goal.split(" is still ", 1)[0].strip()
    return None


@dataclass
class PlanStep:
    text: str
    done: bool = False


@dataclass
class TeamPlan:
    """A shared, free-text step-by-step plan, auto-ticked from milestones.

    Steps are auto-ticked once a milestone of the matching *kind* is reached,
    so the displayed plan tracks real progress without trusting model memory.
    """

    steps: list[PlanStep] = field(default_factory=list)

    def refresh(self, milestones: set[str]) -> None:
        _TAKE_VERBS = ("take", "pick up", "grab", "retrieve", "collect", "get")
        _OPEN_VERBS = ("enter", "unlock", "open", "use", "code", "set", "flip")
        _MOVE_VERBS = ("move", "reach", "enter", "pass", "go", "advance")
        _POWER_VERBS = ("power", "fuse", "disable", "enable", "online", "activate")
        _LEARN_VERBS = ("inspect", "examine", "read", "find", "discover", "note", "clue")

        for step in self.steps:
            if step.done:
                continue
            low = step.text.lower().replace("_", " ")
            for milestone in milestones:
                kind, _, rest = milestone.partition(":")
                token = rest.replace("_", " ").strip()
                if not token:
                    continue
                matched = False
                if kind == "took":
                    if token in low and any(kw in low for kw in _TAKE_VERBS):
                        matched = True
                elif kind == "opened":
                    if token in low and any(kw in low for kw in _OPEN_VERBS):
                        matched = True
                elif kind == "reached":
                    if token in low and any(kw in low for kw in _MOVE_VERBS):
                        matched = True
                elif kind == "power":
                    if any(kw in low for kw in _POWER_VERBS):
                        matched = True
                elif kind == "learned":
                    if token in low and any(kw in low for kw in _LEARN_VERBS):
                        matched = True
                if matched:
                    step.done = True
                    break

    def render(self) -> str:
        if not self.steps:
            return "(no agreed plan yet — propose one)"
        return "\n".join(
            f"  {i + 1}. [{'x' if s.done else ' '}] {s.text}"
            for i, s in enumerate(self.steps)
        )


@dataclass
class AttemptRecord:
    """One executed action, with the world it was taken in and its outcome."""

    tick: int
    signature: str
    world_key: str  # fingerprint BEFORE the action
    success: bool
    outcome: str


@dataclass
class CognitionConfig:
    stall_threshold: int = 6
    reflect_every: int = 6
    curiosity_limit: int = 4
    failed_limit: int = 6
    hypothesis_limit: int = 4


@dataclass
class ProgressUpdate:
    """What changed cognitively after an action was executed."""

    new_milestones: list[str] = field(default_factory=list)
    became_stuck: bool = False
    became_unstuck: bool = False


@dataclass
class TeamBrief:
    """Per-turn brief assembled for the acting agent's prompt."""

    objective: str
    current_goal: str
    blockers: list[str]
    solved: list[str]
    open_puzzles: list[str]
    team_memory: list[str]
    candidate_actions: list[str]
    recommended_action: str
    already_examined: list[str]
    next_plan_step: str
    failed_attempts: list[str]
    critical_note: str | None
    stuck: bool
    reflect: bool

    def render(self) -> str:
        lines = [
            f"OBJECTIVE: {self.objective}",
            f"CURRENT GOAL: {self.current_goal}",
        ]
        if self.next_plan_step:
            lines.append(f"NEXT PLAN STEP: {self.next_plan_step}")
        if self.blockers:
            lines.append(f"BLOCKERS: {', '.join(self.blockers)}")
        lines.append(
            "SOLVED (never redo): " + ("; ".join(self.solved) or "(nothing yet)")
        )
        lines.append(
            "STILL TO DO: " + ("; ".join(self.open_puzzles) or "(none — try to win now)")
        )
        if self.team_memory:
            lines.append("MEMORY: " + "; ".join(self.team_memory))
        if self.failed_attempts:
            lines.append("RECENTLY FAILED (don't repeat): " + "; ".join(self.failed_attempts))
        if self.already_examined:
            lines.append("ALREADY EXAMINED: " + ", ".join(self.already_examined))
        if self.candidate_actions:
            lines.append("RECOMMENDED CANDIDATES:")
            for c in self.candidate_actions:
                lines.append(f"  - {c}")
        if self.recommended_action:
            lines.append(f"PLANNER RECOMMENDS: {self.recommended_action}")
        if self.critical_note:
            lines.append(f"CRITICAL: {self.critical_note}")
        if self.stuck:
            lines.append("STATUS: STUCK — no progress recently, prioritize the recommended action.")
        if self.reflect:
            lines.append("REFLECT: take stock of progress before choosing your next action.")
        return "\n".join(lines)


@dataclass
class TeamCognition:
    """Deterministic shared memory + loop/progress/stall reasoning."""

    config: CognitionConfig = field(default_factory=CognitionConfig)

    attempts: list[AttemptRecord] = field(default_factory=list)
    team_summary: str = ""
    milestones: set[str] = field(default_factory=set)
    plan: TeamPlan = field(default_factory=TeamPlan)

    _seen: set[tuple[str, str]] = field(default_factory=set, init=False)  # (sig, world_key)
    _tried_targets: set[str] = field(default_factory=set, init=False)
    _inspected_targets: set[str] = field(default_factory=set, init=False)
    _turns_since_progress: int = field(default=0, init=False)
    stuck: bool = field(default=False, init=False)
    _last_critical_tick: int = field(default=-1, init=False)

    # ------------------------------------------------------------------ #
    # Blackboard
    # ------------------------------------------------------------------ #
    def set_reflection(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.team_summary = text

    def set_plan(self, steps: list[str]) -> None:
        clean = [s.strip() for s in steps if s and s.strip()]
        self.plan = TeamPlan(steps=[PlanStep(text=s) for s in clean])

    # ------------------------------------------------------------------ #
    # Loop detection (the no-repeat guard)
    # ------------------------------------------------------------------ #
    def already_done(self, action: str, ps: PartyState) -> str | None:
        """Goal-level redundancy: is this action's objective ALREADY satisfied?

        Reads CURRENT authoritative state, catching re-takes/re-opens/re-enters
        even when the wider world fingerprint has since changed.
        """
        parts = action.split()
        if not parts:
            return None
        verb = parts[0]
        target = parts[1] if len(parts) >= 2 else None

        if verb == "take" and target:
            if target in ps.inventory:
                return f"'{target}' has already been taken — it is in your inventory."
        if verb in ("enter_code", "use_tool", "insert_liquid", "open", "flip_fuse") and target:
            if ps.object_states.get(target) in OPENED_STATES:
                return f"'{target}' is already open/unlocked — repeating this does nothing."
        if verb == "go" and target:
            if ps.current_room == target:
                return f"you are already in '{target}'."
        if verb == "examine" and target:
            if target in self._inspected_targets:
                return (
                    f"'{target}' was already examined — examining is idempotent, so "
                    f"examining it again reveals nothing new."
                )
        return None

    def blocked_reason(self, action: str, ps: PartyState) -> str | None:
        """The single gate consulted before treating an action as a candidate.

        Returns a human note if the move should be re-decided, else None. Free
        actions (examine/wait) always pass the redundancy check (the action
        space already drops re-examines via ``_already_examined``).
        """
        verb = action.split(" ", 1)[0]
        if verb in _FREE_VERBS:
            return None
        done = self.already_done(action, ps)
        if done:
            return (
                f"ALREADY DONE — {done} This objective is complete; pick the NEXT "
                f"unsolved step toward the goal instead of repeating it."
            )
        if self.is_redundant(action, ps):
            return self.redundancy_note(action)
        return None

    def is_redundant(self, action: str, ps: PartyState) -> bool:
        """True if this exact action was already tried in this exact world.

        Free actions are never redundant.
        """
        verb = action.split(" ", 1)[0]
        if verb in _FREE_VERBS:
            return False
        return (action_signature(action), world_fingerprint(ps)) in self._seen

    def redundancy_note(self, action: str) -> str:
        sig = action_signature(action)
        prior = next((a for a in reversed(self.attempts) if a.signature == sig), None)
        outcome = f" Last time: {prior.outcome}" if prior else ""
        return (
            f"BLOCKED: '{sig}' was already tried and the world has NOT changed since, "
            f"so the result is identical.{outcome} Choose a DIFFERENT action — explore "
            f"something untried, apply a clue, or work toward the next plan step."
        )

    # ------------------------------------------------------------------ #
    # Recording outcomes + progress / stall tracking
    # ------------------------------------------------------------------ #
    def observe(
        self,
        action: str,
        note: str,
        ps_before: PartyState,
        ps_after: PartyState,
        world: GameWorld,
        tick: int,
        success: bool,
    ) -> ProgressUpdate:
        sig = action_signature(action)
        world_key = world_fingerprint(ps_before)
        self.attempts.append(
            AttemptRecord(
                tick=tick,
                signature=sig,
                world_key=world_key,
                success=success,
                outcome=note,
            )
        )
        self._seen.add((sig, world_key))

        parts = action.split()
        if len(parts) >= 2:
            self._tried_targets.add(parts[1])
        if parts and parts[0] == "examine" and len(parts) >= 2:
            self._inspected_targets.add(parts[1])

        update = ProgressUpdate()
        new = compute_milestones(world, ps_before, ps_after)
        if new:
            update.new_milestones = sorted(new)
            self.milestones |= new
            self._turns_since_progress = 0
            if self.stuck:
                self.stuck = False
                update.became_unstuck = True
        else:
            self._turns_since_progress += 1
            if not self.stuck and self._turns_since_progress >= self.config.stall_threshold:
                self.stuck = True
                update.became_stuck = True
        return update

    # ------------------------------------------------------------------ #
    # Reflection scheduling
    # ------------------------------------------------------------------ #
    def should_reflect(self, tick: int) -> bool:
        return tick > 0 and tick % self.config.reflect_every == 0

    # ------------------------------------------------------------------ #
    # Code extraction from known clues
    # ------------------------------------------------------------------ #
    def _extract_known_codes(self, ps: PartyState) -> tuple[list[str], list[str]]:
        """Extract known numeric and phrase codes from discovered clues."""
        numbers: list[str] = []
        phrases: list[str] = []

        def ingest(text: str) -> None:
            for token in re.findall(r"[0-9]{2,8}", text):
                numbers.append(token)
            for token in re.findall(r"[A-Z][A-Z0-9]{1,7}", text):
                phrases.append(token)
            for quoted in re.findall(r"'([^']{3,40})'", text):
                q = quoted.strip().lower()
                if q and not q.isdigit():
                    phrases.append(q)

        for info in ps.known_info:
            ingest(info)
            phrases.append(info)

        def dedupe(values: list[str]) -> list[str]:
            out: list[str] = []
            for value in values:
                if value not in out:
                    out.append(value)
            return out

        return dedupe(numbers), dedupe(phrases)

    def _codes_for_object(self, obj, numeric_codes: list[str], phrase_codes: list[str]) -> list[str]:
        """Return only code candidates that match the lock's expected format."""
        if not obj.requires_code:
            return []
        if obj.requires_code in phrase_codes:
            return [obj.requires_code]
        if obj.requires_code.isdigit():
            if obj.code_digits:
                filtered = [c for c in numeric_codes if len(c) == obj.code_digits]
                return filtered[:2]
            return numeric_codes[:2]
        return []

    # ------------------------------------------------------------------ #
    # Candidate compilation — the core deterministic policy
    # ------------------------------------------------------------------ #
    def _compile_candidates(
        self,
        world: GameWorld,
        ps: PartyState,
        current_goal: str,
        next_plan_step: str = "",
    ) -> list[CandidateAction]:
        """Filter/rank ``_build_action_space`` into a small, tagged candidate set.

        Returns up to 8 ``CandidateAction``s with PROGRESS / EXPLORATION tags
        (COORDINATION is reserved for a future multi-agent extension).
        """
        visible = _objects_in_room(world, ps)
        space = _build_action_space(world, ps, visible)
        by_id = {o.id: o for o in world.objects}
        numeric_codes, phrase_codes = self._extract_known_codes(ps)
        focus_id = _goal_object_id(current_goal)
        plan_low = next_plan_step.lower().replace("_", " ").strip()

        out: list[CandidateAction] = []
        seen: set[str] = set()

        def push(action: str, tag: str) -> None:
            if action not in space or action in seen:
                return
            if self.blocked_reason(action, ps):
                return
            seen.add(action)
            out.append(CandidateAction(action=action, tag=tag))

        def plan_match(action: str) -> bool:
            if not plan_low:
                return False
            low = action.lower().replace("_", " ")
            return low in plan_low or any(
                tok in low for tok in plan_low.split() if len(tok) > 3
            )

        # Goal-targeted candidates first.
        if focus_id:
            focus_obj = by_id.get(focus_id)
            if focus_obj is not None:
                for code in self._codes_for_object(focus_obj, numeric_codes, phrase_codes):
                    # The action space doesn't encode the code value; just the
                    # verb+target is enough to identify the candidate slot.
                    push(f"enter_code {focus_id}", "PROGRESS")
                if focus_obj.requires_tool and focus_obj.requires_tool in ps.inventory:
                    push(f"use_tool {focus_id}", "PROGRESS")
                if focus_obj.requires_liquid:
                    push(f"insert_liquid {focus_id}", "PROGRESS")
                if focus_obj.fuses is not None:
                    for label in focus_obj.fuses:
                        push(f"flip_fuse {focus_id} {label}", "PROGRESS")
                if focus_obj.takeable:
                    push(f"take {focus_id}", "PROGRESS")
                if focus_obj.requires_power:
                    push(f"open {focus_id}", "PROGRESS")

        # Tool acquisition: a visible takeable item some still-locked object needs.
        needed_tools = {
            o.requires_tool
            for o in world.objects
            if o.requires_tool and ps.object_states.get(o.id, o.state) in HIDDEN_STATES
        }
        for obj in visible:
            if obj.id in needed_tools and obj.takeable:
                push(f"take {obj.id}", "PROGRESS")

        # Movement through open exits is usually progress-making.
        for action in space:
            if action.startswith("go "):
                push(action, "PROGRESS")

        # Per-object candidates from the visible set.
        for obj in visible:
            for action in space:
                parts = action.split()
                if len(parts) < 2 or parts[1] != obj.id:
                    continue
                verb = parts[0]
                if verb == "examine":
                    push(action, "EXPLORATION")
                elif verb == "take":
                    push(action, "PROGRESS")
                elif verb in ("enter_code", "use_tool", "insert_liquid", "flip_fuse", "open"):
                    push(action, "PROGRESS")

        # Shared-plan bias: surface candidates matching the next open plan step.
        if plan_low:
            matched = [c for c in out if plan_match(c.action)]
            rest = [c for c in out if not plan_match(c.action)]
            out = matched + rest

        return out[:8]

    def is_policy_candidate(
        self, world: GameWorld, ps: PartyState, current_goal: str, action: str, next_plan_step: str = ""
    ) -> bool:
        """True when an action is in the deterministic policy candidate set."""
        verb = action.split(" ", 1)[0]
        if verb in _FREE_VERBS:
            return True
        allowed = {
            c.action for c in self._compile_candidates(world, ps, current_goal, next_plan_step)
        }
        return action in allowed

    def policy_candidates(
        self, world: GameWorld, ps: PartyState, current_goal: str, next_plan_step: str = ""
    ) -> list[str]:
        """Deterministic valid/reachable action candidates for this turn."""
        return [
            c.action
            for c in self._compile_candidates(world, ps, current_goal, next_plan_step)
        ]

    def _candidate_actions(
        self, world: GameWorld, ps: PartyState, current_goal: str, next_plan_step: str = ""
    ) -> list[str]:
        return [
            f"[{c.tag}] {describe_action_brief(c.action)}"
            for c in self._compile_candidates(world, ps, current_goal, next_plan_step)
        ]

    def _next_plan_step(self) -> str:
        for step in self.plan.steps:
            if not step.done:
                return step.text
        return ""

    # ------------------------------------------------------------------ #
    # Critical guidance (loop escalation)
    # ------------------------------------------------------------------ #
    def critical_guidance_for(self, world: GameWorld, ps: PartyState, tick: int) -> str | None:
        """Escalate when the agent loops on examine/wait while a known code is usable."""
        board = derive_board(world, ps)
        if not board.unsolved:
            return None

        if not self.attempts:
            return None
        if self._last_critical_tick == tick:
            return None

        recent = self.attempts[-3:]
        r1 = recent[-1].signature
        wait_loop = r1.startswith(IDLE_ACTION)
        examine_loop = (
            len(recent) >= 3
            and all(a.signature.startswith("examine") for a in recent)
            and len({a.signature for a in recent}) == 1
        )
        if not (wait_loop or examine_loop):
            return None

        numeric_codes, phrase_codes = self._extract_known_codes(ps)
        by_id = {o.id: o for o in world.objects}
        for obj in world.objects:
            cur = ps.object_states.get(obj.id, obj.state)
            if not (
                obj.requires_code
                and cur in HIDDEN_STATES
                and obj.id in {o.id for o in _objects_in_room(world, ps)}
            ):
                continue
            matching = self._codes_for_object(obj, numeric_codes, phrase_codes)
            if matching:
                self._last_critical_tick = tick
                return (
                    f"You are looping. You know code '{matching[0]}'. "
                    f"Use enter_code on '{obj.id}' now to progress."
                )

        self._last_critical_tick = tick
        return (
            "You are looping with blockers active. Stop repeating examine/wait. "
            "Pick a non-redundant candidate action that changes the world state now."
        )

    # ------------------------------------------------------------------ #
    # Memory + brief assembly
    # ------------------------------------------------------------------ #
    def _team_memory(self, ps: PartyState) -> list[str]:
        out: list[str] = []
        for info in sorted(ps.known_info):
            out.append(f"Confirmed clue: {info}")
        for token in sorted(self.milestones):
            if token.startswith("opened:") or token.startswith("took:") or token.startswith("power:"):
                out.append(_humanize(token))
        seen: set[str] = set()
        for a in reversed(self.attempts):
            if a.success:
                continue
            msg = a.outcome.strip()
            if not msg or msg in seen:
                continue
            seen.add(msg)
            out.append(f"Failed before: {msg}")
            if len(out) >= 8:
                break
        return out[:8]

    def _blocked_for(self, ps: PartyState) -> list[str]:
        """Actions now provably pointless (already-tried, world unchanged)."""
        world_key = world_fingerprint(ps)
        out: list[str] = []
        seen_sigs: set[str] = set()
        for a in reversed(self.attempts):
            if a.signature in seen_sigs:
                continue
            if (a.signature, world_key) in self._seen and not a.success:
                out.append(f"{a.signature} — {a.outcome}")
                seen_sigs.add(a.signature)
            if len(out) >= self.config.failed_limit:
                break
        return out

    def _examined_here(self, world: GameWorld, ps: PartyState) -> list[str]:
        visible = _objects_in_room(world, ps)
        return [o.id for o in visible if o.id in self._inspected_targets]

    def _blockers_for_goal(self, current_goal: str) -> list[str]:
        blockers: list[str] = []
        if " is still " in current_goal or "(needs" in current_goal:
            idx = current_goal.find("(needs")
            if idx != -1:
                hint = current_goal[idx:].strip()
                blockers.append(hint.strip("()"))
        for a in reversed(self.attempts):
            if a.success:
                continue
            txt = a.outcome.lower()
            if "is locked" in txt:
                blockers.append("A required door/lock is still closed")
            elif "code unknown" in txt:
                blockers.append("Code not yet known — find the clue first")
            elif "missing tool" in txt:
                blockers.append("Required tool not yet collected")
            if len(blockers) >= 4:
                break
        uniq: list[str] = []
        for b in blockers:
            if b and b not in uniq:
                uniq.append(b)
        return uniq[:4]

    def _derive_positional_goal(self, world: GameWorld, ps: PartyState, board: SolutionBoard) -> str:
        """Return the most actionable goal from the agent's current position.

        Priority: fuse panels with OFF fuses -> locked solvable objects in the
        current room -> takeable items in the current room -> board fallback.
        """
        room = ps.current_room
        visible = _objects_in_room(world, ps)
        visible_ids = {o.id for o in visible}

        for obj in visible:
            if obj.fuses is not None and obj.id in ps.fuse_states:
                for fuse, pos in ps.fuse_states[obj.id].items():
                    if pos != "ON":
                        return f"set fuse {fuse} to ON on {obj.id} to activate power"

        for obj in visible:
            cur = ps.object_states.get(obj.id, obj.state)
            if obj.interactable and cur in HIDDEN_STATES and _has_requirement(obj):
                return f"{obj.id} is still {cur}{_requirement_hint(obj)}"

        for obj in visible:
            if (
                obj.takeable
                and obj.id not in ps.inventory
                and ps.object_states.get(obj.id, obj.state) not in HIDDEN_STATES
            ):
                return f"take {obj.id} from current room"

        return derive_current_goal(board)

    def brief_for(
        self,
        world: GameWorld,
        ps: PartyState,
        *,
        reflect: bool = False,
        critical_note: str | None = None,
        recommended_action: str = "",
    ) -> TeamBrief:
        board = derive_board(world, ps)
        # Sync the shared plan against cumulative milestones each turn.
        self.plan.refresh(self.milestones)
        next_plan_step = self._next_plan_step()
        current_goal = self._derive_positional_goal(world, ps, board)

        return TeamBrief(
            objective=board.objective,
            current_goal=current_goal,
            blockers=self._blockers_for_goal(current_goal),
            solved=board.solved,
            open_puzzles=board.unsolved,
            team_memory=self._team_memory(ps),
            candidate_actions=self._candidate_actions(world, ps, current_goal, next_plan_step),
            recommended_action=describe_action_brief(recommended_action) if recommended_action else "",
            already_examined=self._examined_here(world, ps),
            next_plan_step=next_plan_step,
            failed_attempts=self._blocked_for(ps),
            critical_note=critical_note,
            stuck=self.stuck,
            reflect=reflect,
        )
