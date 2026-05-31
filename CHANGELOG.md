# Change Log

Chronological log of code changes. Newest entries appear first.

## 2026-05-31 08:55:02 WIB

### What changed
- Room entry gating removed — players can freely move between adjacent rooms without prerequisites. Prerequisite list field replaced with plain-text `next_step` hint per room (guides player toward that room's goal without blocking entry).
- Goal completion now tracked independently from room entry. Each room's `goal_completion` marks when that room's objective is satisfied, used only for victory/progress (not for gate logic).
- Automatic clue patching added — world generator now detects unsolvable worlds where a required code/info token has no source object, and auto-assigns it to a plausible carrier (prioritizing clue-like objects, takeable items, or interactable containers in the solution path).
- Room `next_step` field added to gameplay prompt, displayed as "Next step (what to do)" to guide agent strategy without hard-blocking exits.

### Why
Simplifying room progression: prerequisites were causing dead ends (unwinnable worlds where Room B required conditions only satisfiable in Room A, but Room A's `goal_completion` couldn't be reached without Room B's tools). Moving prerequisites out of entry logic and into goal-completion tracking decouples room unlock from puzzle difficulty. The `next_step` hint provides softer guidance via the agent's reasoning rather than hard failure messages. Auto-patching closes solvability gaps by ensuring every required code/info has a discoverable source before the solver needs it.

---

## 2026-05-29 16:02:05 WIB

### What changed
- agents/game_master.py: Added `SINGLE_ROOM_MODE = True` flag and a `_build_world` guard that keeps only the first room, clears its adjacency, and empties its prerequisites — forcing single-room generation regardless of what the LLM returns.
- main.py: `_write_node_log` now accepts a `root: Path` argument; `_run_once_captured` takes `log_nodes` and `log_root`, streaming through the graph when logs are requested so node logs are written per-run; `smoke` accepts `log_nodes` and stores per-run logs under `smoke_runs/<ts>/run_<NNN>_logs/` so smoke runs and node logs no longer clobber each other.
- prompts/game_master/generation.txt: Rewrote the rooms section for single-room mode — exactly one room with empty adjacency, empty prerequisites, and a `goal_completion` equal to the `win_condition`. Collapsed the adjacency and goal/prerequisite blocks to the single-room invariants.

### Why
Multi-room generation was overwhelming the player agent and the smoke logger overwrote `logs/<node>/raw.txt` every run, making post-hoc debugging impossible. Constraining to one room lets us validate the full loop (generation → selection → gameplay → victory) end-to-end before scaling complexity, and per-run log directories under each smoke timestamp let us inspect every run's raw LLM output without losing data.

---

## 2026-05-29 15:35:45 WIB

### What changed
- agents/gameplay_node.py: `_resolve_examine` now distinguishes repeat/no-op examines — it returns explicit dead-end notes (`examined X again — nothing new (dead end)`, `examined X — no hidden info`, `examined X; already knew <info>`) instead of a bare `examined X`; added `"dead end"` and `"nothing new"` to `_FAIL_KEYWORDS` so these outcomes badge as failures.
- prompts/gameplay_agent/action.txt: Strengthened the no-repeat guidance — agents are now explicitly forbidden from repeating any action whose outcome contained `no hidden info`, `nothing new`, `dead end`, `already`, or `code unknown`, and told to MOVE rooms once every object in the current room is exhausted.
- prompts/game_master/generation.txt: Added a CRITICAL "Solvability rule" requiring every code/liquid/power/tool precondition to be discoverable in a reachable location before the locked object, and forbidding codes/clues that only appear in the `solution_path` (which the runtime ignores).

### Why
Follow-up to the agent-history fix: even with history surfaced, examines returned the same generic `examined X` note whether or not new info was learned, so the agent couldn't tell a productive examine from a dead end. Making the outcome text explicit (and badging dead ends as failures) gives the loop-breaking prompt concrete signals to act on. The GM solvability rule prevents the generator from authoring unsolvable worlds where a required code exists only in the ignored solution path.

---

## 2026-05-29 14:57:54 WIB

### What changed
- agents/gameplay_node.py: Added `HISTORY_WINDOW` constant and `_format_agent_history()` helper that pulls each agent's last 6 actions + outcomes from `ps.log`; wired `agent_recent_history` into the action-selection prompt.
- prompts/gameplay_agent/action.txt: Added `{agent_recent_history}` block plus an explicit nudge instructing the agent not to repeat no-op examines and to try a different object, apply a tool/code/liquid/power, or move via `go <room_id>`.

### Why
Smoke-test run `smoke_runs/20260529_135622/run_001.txt` showed both party members looping on `examine chains` / `examine rusty_spoon` / `examine window` in the brig for all 40 ticks, never moving to the mess_hall. Root cause: the action-selection prompt was effectively stateless — each agent only saw its teammate's last action, not its own history, so given identical world snapshots it produced identical answers. Surfacing the agent's own recent actions and their outcomes breaks the deterministic loop without any extra state field (it reads from the existing `ps.log`).

---

## 2026-05-29 11:12:58 WIB

### What changed
- agents/mission_master_node.py: Deleted — mission generation no longer fits the object-graph world model.
- prompts/mission_master/: Deleted directory (system.txt + generation.txt).
- state/game_state.py: Removed `Mission` and `RoomItem`; reshaped `PartyState` to track `inventory: list[str]` (object ids), `object_states`, `known_info`, `fuse_states`, `power_active`; swapped `TickAction.matched_required_action` → `target_object`; dropped `missions` field from `GameState`.
- state/__init__.py: Removed `Mission`/`RoomItem` exports.
- graph.py: Removed `mission_master` node and its two edges; `gameplay` now fans in only from `player_agent_2`.
- agents/character_master_node.py: Switched prompt vars from `title`/`room.name` to `scenario`/`room.id`.
- prompts/character_master/generation.txt: Replaced `Title:` placeholder with `Scenario:`.
- agents/player_agent_node.py: Switched prompt var from `title` to `scenario`.
- prompts/player_agent/selection.txt: Replaced `Title:` placeholder with `Scenario:`.
- agents/gameplay_node.py: Full rewrite — new mechanical engine resolving verbs `examine / take / enter_code / use_tool / insert_liquid / flip_fuse <label> / open / go <room> / wait` against object preconditions; object visibility honors container chains; `_build_initial_party_state` seeds `object_states`/`fuse_states`/`power_active` from world; victory fires when `object_states[win.object_id] == win.state`; added `_liquid_token_matches` for tolerant pH/liquid matching against held bottle descriptions.
- prompts/gameplay_agent/system.txt: Rewritten around the object-graph model and discrete action menu.
- prompts/gameplay_agent/action.txt: Rewritten to surface `objective`, `win_condition`, `objects_in_room` (with state), `inventory`, `known_info`, and the new action grammar.
- visualization/renderer.py: Keyed by `Room.id`; accepts `objects: list[WorldObject]` and renders objects grouped by `location` instead of inline `room.items`; updated `_place_rooms` and `_build_cell` accordingly.
- main.py: Removed `mission_master` from `NODE_NAMES`; added `all` as a valid `--log` choice that expands to every node; rewrote `_render` to print `scenario`, `objective`, `win_condition`, `solution_path`, inventory of object ids, and `known_info`; removed mission and game_flow rendering blocks.

### Why
With the GM now emitting an object-graph world (codes, tools, liquids, power, win_condition), the old gate/mission pipeline was dead weight — the user opted to drop missions entirely and drive victory off `win_condition` directly. Gameplay was rewritten as a small deterministic engine over object preconditions so the loop can mechanically reach victory (verified end-to-end against `logs/game_setting.json`: 9-step solve produces VICTORY). Downstream prompts and the renderer were updated to match the new `Room.id` / `WorldObject` shapes. The `--log all` shortcut was added so the user can capture every node's output in one flag.

---

## 2026-05-29 11:00:31 WIB

### What changed
- prompts/game_master/generation.txt: Rewrote the GM output template to the new schema (scenario, rooms with id+adjacency, objects[] with optional precondition fields, rules, win_condition as {object_id,state}, solution_path); dropped narrative title/setup/atmosphere and game_flow/gates.
- state/game_state.py: Rekeyed `Room` by `id` (dropped `name` and inline `items`); added `WorldObject` (with state, interactable/takeable, requires_code/code_digits/requires_tool/requires_liquid/requires_power, fuses, contains_info, slot_description, note) and `WinCondition`; reshaped `GameWorld` (scenario, objects, rules, win_condition, solution_path); removed `Gate` and `GameFlow`.
- state/__init__.py: Updated exports — removed `Gate`/`GameFlow`, added `WorldObject`/`WinCondition`/`PartyState`, fixed stale `PlayerState` reference.
- agents/game_master.py: Rewrote builders/validators for the new schema — `_build_rooms` now accepts dict OR bare-string room entries and keys by `id`; `_repair_adjacency` keyed by id and no longer caps items; new `_build_objects` validates `location` against rooms+nested objects and nulls dangling `requires_tool`; new `_build_win_condition`; new `_build_world` assembles scenario/objects/rules/win_condition/solution_path; removed `_build_game_flow` and `MAX_ITEMS`.
- main.py: Added `--log NODE` flag (repeatable, choices restricted to the six graph nodes); when set, `run()` switches to `graph.stream(stream_mode="updates")` and writes `logs/<node>/output.json` (pydantic-serialized state delta) plus `logs/<node>/raw.txt` (concatenated AIMessage content) for each requested node. Added `_jsonable`, `_write_node_log`, and `_merge_update` helpers.
- .gitignore: Added `logs/` to ignored paths.

### Why
The user is migrating the Game Master to emit a richer object-graph world model (matching `logs/game_setting.json`) with mechanical preconditions like codes, tools, liquids, and power — replacing the previous narrative `gates[]` flow. Scope was deliberately limited to GM output only (downstream agents will break and be fixed in follow-ups), while keeping cardinal-direction room adjacency so the player can still navigate. The `--log` flag was added so the user can capture per-node raw LLM responses and parsed outputs to disk for debugging the new generation pipeline.

---
