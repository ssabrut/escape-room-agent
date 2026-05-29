# Plan — Make the Multi-Agent Escape Room More Fun & Engaging

## Context

The current game ([graph.py](graph.py)) runs a linear LangGraph DAG: `game_master → (character_master ∥ mission_master) → gameplay → END`. Inside [agents/gameplay_node.py](agents/gameplay_node.py), each tick the two player agents pick one action from a list of remaining mission keywords. There are no NPCs, no real puzzles, no consequences for wrong choices, and no mechanic that forces the two agents to coordinate beyond reading each other's last `say`. Runs are deterministic-feeling: pick the right keyword, advance the gate, repeat.

Goal: keep the LangGraph topology untouched and add four small, composable features to the tick loop that together deliver:

- **Watchability** — visible drama (failed persuasions, lock clicks, HP loss, sync misses)
- **Agent skill expression** — better prompting / reasoning visibly wins (parse clues, commit via dialogue)
- **Narrative richness** — NPCs, dispositions, in-character dialogue
- **Mechanical depth** — order-sensitive puzzles, HP, hazards, simultaneous actions

Scope is intentionally **small additions only** — extend `PartyState` / `Mission` / `Room`, expand the action space, add resolution branches inside the existing per-tick block at [agents/gameplay_node.py:337-371](agents/gameplay_node.py#L337-L371). No new graph nodes.

---

## Feature 1 — NPCs & "Persuasion Check"

Adds NPCs to missions. Agents can `talk:<npc>` to gather hints or `persuade:<npc>` using a password keyword surfaced through dialogue. Success unlocks a hidden required action; failure escalates the NPC and costs HP (composes with Feature 3).

- **State** ([state/game_state.py](state/game_state.py)):
  - new `class NPC(BaseModel)` with `name`, `disposition`, `password`, `reveals_action`
  - `Mission.npcs: list[NPC] = []`
  - `PartyState.npc_states: dict[str, str] = {}` — `"hostile" | "helped" | "hostile_escalated"`
- **Gameplay loop** ([agents/gameplay_node.py](agents/gameplay_node.py)):
  - In `_build_action_space`, append `f"talk:{n.name}"` and `f"persuade:{n.name}"` for each mission NPC not yet `"helped"`.
  - In the resolution block at [agents/gameplay_node.py:337-352](agents/gameplay_node.py#L337-L352), branch on `chosen.startswith("persuade:")`: if `decided["say"]` contains the password (case-insensitive substring) → flip state to `"helped"`, inject `reveals_action` into `mission.required_actions`. Else → `"hostile_escalated"` + call `_apply_damage` (Feature 3).
- **Content generation**: extend [prompts/mission_master/generation.txt](prompts/mission_master/generation.txt) to emit `npcs[]` with hint phrases planted in `Mission.description`; extend [prompts/gameplay_agent/action.txt](prompts/gameplay_agent/action.txt) to surface visible NPCs + dialogue history.
- **Covers**: narrative, watchability, agent skill.

## Feature 2 — Real Puzzles: "Ordered Combination Lock"

A mission may carry a `solution_order` — the existing `required_actions` must be performed in that exact order. Wrong order resets gate progress and emits a "lock clicks wrong" log line plus −1 HP.

- **State**: `Mission.solution_order: list[str] | None = None`. Reuses existing `completed_actions_by_gate`.
- **Gameplay loop**: in the `if matched:` branch at [agents/gameplay_node.py:340-348](agents/gameplay_node.py#L340-L348), if `mission.solution_order` is set, check `matched == solution_order[len(completed_list)]`. On mismatch: clear `ps.completed_actions_by_gate[mission.gate_index]`, set `note = "lock reset"`, `_apply_damage(ps, 1, "wrong combination")`.
- **Content generation**: [prompts/mission_master/generation.txt](prompts/mission_master/generation.txt) emits `solution_order` plus diegetic clues; clue items can be planted via [prompts/game_master/generation.txt](prompts/game_master/generation.txt) into `Room.items[].description`.
- **Covers**: mechanical depth, agent skill, watchability.

## Feature 3 — Risk & Consequences: "Party HP & Room Hazards"

A shared HP pool (start 5). Failed persuasions, wrong combination steps, and entering a hazardous room without the right inventory item all cost HP. At 0 → `game_over` without victory. HP is shown to agents each tick so they can weigh risk.

- **State**:
  - `PartyState.hp: int = 5`
  - `PartyState.hp_log: list[str] = []`
  - `Room.hazard: str | None = None`, `Room.hazard_cost: int = 1`
- **Gameplay loop**:
  - Add helper `_apply_damage(ps, amount, reason)` near the top of the file alongside `_build_action_space`.
  - After every movement step that updates `ps.current_room`, if the new room has a `hazard` and no inventory item name appears inside the hazard string → `_apply_damage`.
  - After the per-tick action block at [agents/gameplay_node.py:371](agents/gameplay_node.py#L371), if `ps.hp <= 0` set `ps.game_over = True; ps.victory = False` and break.
- **Content generation**: [prompts/game_master/generation.txt](prompts/game_master/generation.txt) may set `hazard` on 1–2 rooms; [prompts/gameplay_agent/action.txt](prompts/gameplay_agent/action.txt) surfaces current HP.
- **Covers**: risk, watchability, narrative, mechanical depth — and amplifies Features 1, 2, 4.

## Feature 4 — Coordination: "Simultaneous Action"

Any `required_action` prefixed `sync:` only counts when **both** agents pick it on the same tick. Forces explicit coordination through `say` ("on three… now"). Misses are logged as visible "sync failed — partner didn't join".

- **State**: `PartyState.pending_sync: dict[str, list[str]] = {}` (gate_index → list of agent_ids who chose a sync action this tick — reset each tick). Sync actions reuse `Mission.required_actions` strings.
- **Gameplay loop**: refactor the inner per-agent block at [agents/gameplay_node.py:337-363](agents/gameplay_node.py#L337-L363) to **collect** each agent's choice into a tick-scoped dict first. After both agents have decided, walk choices: non-`sync:` actions resolve as today; `sync:` actions only complete when ≥2 distinct agent_ids chose them this tick. Lone-sync gets `note = "sync failed — partner didn't join"`.
- **Content generation**: [prompts/mission_master/generation.txt](prompts/mission_master/generation.txt) emits at most one `sync:` action per game; [prompts/gameplay_agent/action.txt](prompts/gameplay_agent/action.txt) reminds agents to announce intent the tick before executing.
- **Covers**: coordination, agent skill, watchability, mechanical depth.

---

## Why these four

They compose: failed persuasion (1) → HP loss (3); wrong combination (2) → HP loss (3); a `sync:` action (4) can target an NPC (1) or be the next step in a locked order (2). Four features, all four goals, all four focus areas, no new graph nodes.

## Verification

1. Run `python /Users/michaeleko/Documents/Projects/escape-rooms/main.py` with at least one theme.
2. Confirm in the streamed output:
   - **NPC**: `talk:<npc>` / `persuade:<npc>` appear in action space; an `npc_states` transition prints.
   - **Puzzle**: at least one mission shows `solution_order`; observe a `lock reset` note followed by recovery on a later tick.
   - **HP**: HP counter prints each tick; at least one `−1 HP (reason: …)` event appears.
   - **Sync**: a `sync:` action shows in the action space; observe one `sync failed` then both agents matching on a subsequent tick.
3. Inspect the final `state.party_state.log` and `hp_log` in [smoke_runs/](smoke_runs/) for ground truth across 2–3 runs of different themes — verify variability run-to-run.

## Critical files

- [state/game_state.py](state/game_state.py) — add `NPC`, extend `Mission`, `Room`, `PartyState`
- [agents/gameplay_node.py](agents/gameplay_node.py) — `_apply_damage` helper, expand `_build_action_space`, refactor tick block at lines 337-371 for sync collection + new resolution branches
- [prompts/mission_master/generation.txt](prompts/mission_master/generation.txt) — emit `npcs`, `solution_order`, `sync:` actions
- [prompts/game_master/generation.txt](prompts/game_master/generation.txt) — emit `hazard` on rooms, plant clue items
- [prompts/gameplay_agent/action.txt](prompts/gameplay_agent/action.txt) — surface NPCs, HP, sync hint
