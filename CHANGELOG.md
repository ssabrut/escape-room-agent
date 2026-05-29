# Change Log

Chronological log of code changes. Newest entries appear first.

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
