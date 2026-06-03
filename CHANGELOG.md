# Change Log

Chronological log of code changes. Newest entries appear first.

## 2026-06-03 08:16:37 WIB

### What changed
- Live game mode now supports hard-mode world generation: multi-room worlds with deep puzzle chains (configurable room count, chain depth, and decoys per room) are generated using a dedicated `generation_bank` prompt and validated solvable by the heuristic oracle before the live game starts, regenerating up to a configured max attempts until the world is winnable.
- Hard mode is opt-in via CLI flags (`--hard`, `--rooms N`, `--decoys N`) or environment variables (`HARD_MODE`, `NUM_ROOMS`, `CHAIN_DEPTH`, `DECOYS`, `GEN_MAX_ATTEMPTS`), defaulting to the original 2-room standard mode when not set.
- The Game Master now exposes regeneration progress and mode info in its startup log, reporting whether a world was generated in standard or hard mode and how many generation attempts were needed to find a solvable world.

### Why
The benchmark had revealed that the original single/two-room generator could produce unwinnable worlds that the oracle would reject, blocking gameplay. Hard mode enables the live game to use the same bank-quality multi-room generator and oracle validation that the benchmark uses, guaranteeing every world the party encounters is actually solvable while keeping the standard 2-room experience available as a baseline.

---

## 2026-06-02 16:52:41 WIB

### What changed
- Each kept benchmark world's `solution_path` is now rebuilt from the oracle's actual winning trace over the final (post-repair) object graph, replacing the LLM-authored path. This eliminates hallucinated or repair-drifted object references and guarantees the recorded solution is one the engine can actually replay to victory.
- The recorded solution is now the MINIMAL winning path: a greedy leave-one-out pass replays subsets of the trace through the engine and drops any step (decoy takes, dead-end drawer opens, redundant examines) that isn't required to win, leaving only the steps needed for an optimal solve.
- The oracle now always records its solve history (the trace is needed to derive the solution path), so the prior `trace`/`debug`-gated history recording was removed.

### Why
The LLM-generated `solution_path` referenced object ids that no longer matched the world after generation-time repairs drifted the object graph, so the stored solution couldn't be trusted or replayed. Deriving the path from the oracle's guaranteed-valid winning trace and then minimizing it produces a consistent, optimal solution — the target a trained policy should converge to.

---

## 2026-06-02 16:08:08 WIB

### What changed
- A clue token's required-info source room is now resolved to the room that actually holds the locked object consuming it (walking the container chain up to its room), instead of always blaming the start room. A second-room safe's clue is no longer stranded back in room one, and a `known_info` goal sources to its own room.
- The missing-clue patcher no longer buries a code clue on an object that is used as someone else's tool, and only credits a takeable object as a good clue carrier when it also reads like something you examine (a note/journal), so codes the party must READ land on readable objects.
- World generation now repairs locks whose required code is supplied by nothing but the lock's own `contains_info` (the answer written onto the lock while `requires_code` names a different, unproduced token): `requires_code` is repointed to that self-carried clue so the lock is at least solvable. Runs after the patch pass so a genuine upstream clue is preferred.
- `requires_tool` targets that name a real but non-takeable object (e.g. a fixed terminal), or a tool locked/hidden behind the very gate it opens, are now forced grabbable — made takeable and relaxed to a visible state — instead of only warned about, so the lock is no longer dead.
- `power_active` goal gates are now made satisfiable: since the engine only brings power online via a fuse flip producing a `sekring_<label>_ON` token, any power gate whose id isn't such a producible token gets a fuse panel attached to a same-room object and its goal id rewritten to the matching token.
- The headless benchmark now records each episode's `chain_depth` — the count of ordered dependency links the oracle actually cleared (unlocks, power-on, room moves) — as a truer puzzle-depth measure than raw ticks, for rejecting shallow worlds from the bank.
- The benchmark oracle policy now deprioritizes actions that failed with no state change since (so it stops head-banging a GM-blocked exit or an unsatisfiable tool), revives them once the world advances, and avoids re-flipping a fuse that's already ON (which would toggle power back off and oscillate forever).

### Why
The benchmark surfaced worlds that were unwinnable or that the oracle could never solve: clues stranded in the wrong room, codes buried on tools, locks gated only by their own answer, non-takeable required tools, and `power_active` gates with no producible fuse token (the `emergency_relay_power` spin-forever case). These generation-time repairs make benchmark target worlds reliably winnable, while the oracle's dead-action and fuse-toggle guards plus the `chain_depth` metric keep the headless solver from looping to timeout and give a real measure of puzzle depth.

---

## 2026-06-02 14:32:02 WIB

### What changed
- A cleared lock now actually satisfies a goal/win that named a "solved" synonym: goal_completion and the win condition of type `object_state` are matched with state equivalence, so an object the engine marked `"unlocked"` also satisfies a target written as `"open"`, `"opened"`, `"unsealed"`, `"dissolved"`, or `"deactivated"` (the `"visible"` state is excluded — merely visible is not a solved lock). Previously such a goal could never be met and the game became unwinnable.
- The generation prompt now constrains the open/unlock state vocabulary (target `state` MUST be the literal `"unlocked"`, not a synonym) and forbids unsolvable tool setups: a `requires_tool` target must be takeable and start in a reachable (non-locked/hidden) state and must not be gated behind the very lock it opens, and tool chains must not be circular (no A↔B cycles) — each chain must terminate at an immediately takeable tool. The self-check was expanded into a numbered checklist covering these cases.

### Why
The resolvers always write `"unlocked"` when a lock is cleared, but the generator named matching goal/win targets with synonyms like `"open"` or `"dissolved"`, so a correctly-solved puzzle never registered and the world was unwinnable. Treating the "opened" synonym family as equivalent fixes existing worlds, while the tightened prompt rules (canonical state vocabulary, no circular or self-gated tool dependencies) prevent the generator from authoring these unsolvable worlds in the first place.

---

## 2026-06-02 14:21:45 WIB

### What changed
- Code-locked doors now unlock from clues that embed the answer in a decorated token: a clue like `captain_combination_8429` now satisfies a door requiring `captain_combination`. Matching accepts equal tokens, a stem that is an underscore-boundary prefix of the other (either direction), a shared underscore-separated segment, or equal non-empty digit sequences — so the bare-stem requirement and its answer-bearing clue are recognized as the same code.
- Both the action-availability check (whether to offer `enter_code`) and the actual `enter_code` resolution now share this single code-matching rule, keeping the offered verb and its success test in sync.

### Why
The generator named a door's `requires_code` with a bare stem (e.g. `captain_combination`) but carried the matching clue as a decorated token embedding the answer (e.g. `captain_combination_8429`). The old test only matched exact equality or the clue's last segment, so this prefix case was missed and a solvable door stayed permanently locked.

---

## 2026-06-02 13:27:15 WIB

### What changed
- World generation now repairs goal-gating objects that the LLM locked with no way to unlock them: if a room's goal_completion (or the win condition) targets a locked/hidden object that carries no requires_* mechanism, it's given a `requires_tool` pointing at a same-room takeable that reads like an unlocker (wheel, key, lever, etc.), or — if no takeable exists — its initial state is relaxed to the target so the goal is reachable.
- Room exits are now held back while the room's goal is unmet and other productive work remains: since the Game Master blocks every exit until the goal is satisfied, offering `go` would be a dead end the party spins on. Exits are still offered as a last resort when nothing else is productive, so the GM's block narration can surface what the room still needs.

### Why
The party could get permanently stranded on a Game Master-blocked exit whose goal was either unwinnable (a locked gate with no unlock mechanism) or simply not worth attempting yet. Repairing unsolvable gates and withholding pointless `go` actions keep the party making progress instead of looping on guaranteed failures.

---

## 2026-06-02 12:45:00 WIB

### What changed
- Dangling `requires_tool` references are now repaired by fuzzy-matching to a takeable object rather than being nulled outright: when a lock names a near-miss id (e.g. `brg_key` vs the real `brg_key_revealed`), it's repointed to the matching object by stripping common qualifier suffixes, so the lock keeps its tool requirement instead of opening for free. Refs with no plausible match are still nulled.
- The action space now hides unlock verbs whose precondition is not currently satisfiable: `enter_code`, `use_tool`, `insert_liquid`, and power-`open` only appear once the party actually has the code, tool, liquid, or active power. The verbs reappear when the prerequisite is met.
- The `examine` verb is dropped from the action space once an object has already been examined, since re-examining can never reveal new information.

### Why
Agents were burning ticks on guaranteed dead-end actions — re-examining exhausted objects and retrying unlocks they couldn't satisfy — and a dangling tool reference could strand a door or make it open with no tool at all. Pruning impossible verbs and repairing near-miss tool ids keeps the action space honest and the world solvable.

---

## 2026-06-02 10:58:36 WIB

### What changed
- The Game Master now proactively directs the party onward: once the current room's goal is satisfied and a route to the win room exists, it announces (with LLM-generated narration, plus a templated fallback) the order to advance to the next room — rather than waiting for a player to attempt `go` on their own.
- The deterministic next hop toward the win room is computed by the engine, and that destination is surfaced into each agent's action prompt as a Game Master directive that overrides their other priorities for that tick, so the party reliably picks `go <dest>`.
- The GM directive is recorded in the tick log as a non-gameplay action (`gm_directive`), so it's excluded from teammate "last action" context and stall detection, and is rendered in its own Game Master panel.

### Why
Once a room was cleared, the party could linger instead of moving on. Having the Game Master actively call the advance — with a computed destination injected as an overriding directive — keeps the party progressing toward the win room without relying on agents to independently decide to leave.

---

## 2026-06-02 10:38:54 WIB

### What changed
- Objects are now marked "handled" (checkmark) once examined, in addition to being taken or having their state changed. Previously a mere examine never counted as resolved, so the party could keep re-examining the same objects needlessly.

### Why
Treating an examined object as handled stops the party from wasting turns re-examining objects that have already been inspected, focusing attention on objects whose puzzles are still open.

---

## 2026-06-02 10:30:44 WIB

### What changed
- The win condition is now derived automatically from the final room's `goal_completion` instead of being stored as a separate field, so the win target lives in exactly one place and can no longer drift from the room it belongs to.
- World generation now collapses redundant container/lock pairs: when the LLM emits both a container and a separate "lock"/"panel" object sharing the same location and requirement, they are merged into the single object the goal checks, with tool references repointed to the survivor.
- World generation now emits warn-only coherence checks that flag dangling puzzle pieces — clues that nothing consumes, duplicated clue tokens, and tools that are required but unreachable (not takeable, or hidden with no reveal path).
- The generation prompt now forbids dangling clues, duplicate clues, and redundant objects, requires hidden tools to be revealable, and requires the win condition to equal the second room's goal_completion.
- Logged world JSON now omits always-null fields (e.g. unused Prerequisite/WorldObject slots), showing only the keys meaningful for each record.

### Why
Generated worlds suffered from redundant and incoherent puzzle pieces (duplicate locks, orphan clues, unreachable tools) and from the win target being duplicated across two places where it could drift. Deriving the win condition from the room goal, deduping objects, and adding coherence warnings plus tighter prompt rules make generated worlds cleaner and more reliably solvable.

---

## 2026-06-02 09:59:16 WIB

### What changed
- Generated worlds now contain TWO connected rooms instead of one: a starting room and a final room joined by a single doorway (mirrored adjacency, e.g. east<->west). The win condition lives in the second room.
- The first room's goal now gates passage to the second: the party cannot reach the final room until the first room's goal_completion (e.g. unlocking the connecting door) is satisfied, and the puzzle chain flows across both rooms.
- Room truncation now anchors on BOTH the starting room and the win-object room when the LLM over-produces rooms, preserving the playable start->win chain (previously it only kept the win-object room).

### Why
Expanding from single-room to two-room worlds makes the escape rooms richer and multi-stage, with a gated progression between rooms. The selection logic was updated in lockstep so that over-generation can't drop either endpoint of the start-to-win path.

---

## 2026-06-02 09:43:25 WIB

### What changed
- Escape planning is now a multi-agent debate instead of a single per-agent plan: each party member first PROPOSES a plan (colored by their role/ability), then runs critique/revise rounds where they read every teammate's plan and improve their own, and finally a facilitator SYNTHESIZES one unified plan the whole party acts on. The number of critique rounds is tunable (`DEBATE_ROUNDS`, default 1).
- A lone agent skips the debate — its proposal becomes the plan directly with no synthesis call.
- Each planning step now surfaces a one-line "why" reasoning in its panel, and the agreed plan is shown in a dedicated "UNIFIED PLAN" panel.
- The mid-room re-plan (triggered when the room state changes) now rebuilds the plan through the same debate rather than a single lead-agent plan.
- The single `plan` prompt was split into three specialized prompts (`plan_propose`, `plan_critique`, `plan_synthesize`) driving the respective debate phases.

### Why
A single agent's escape plan didn't make use of the party's distinct roles and abilities and could lock in a flawed strategy. Having agents propose, critique each other, and converge on a synthesized plan produces a stronger, role-informed strategy that the whole party agrees on, while keeping the debate cheap (bounded rounds, skipped entirely for solo play).

---

## 2026-06-02 09:18:07 WIB

### What changed
- The lead agent now re-observes and re-plans mid-room whenever the room picture actually changes — a new object is revealed, an item is taken, a lock opens, or power comes online — keeping the standing observation and escape plan in sync with what the party can currently see. Ticks where nothing changed skip the extra LLM calls via a room-state fingerprint.
- Observation and planning passes (`observe`, `plan`, `reobserve`, `replan`) are now treated as bookkeeping rather than real moves: they're excluded from each agent's action history, from teammate "last action" context, from stall detection, and from the action-log panel (they appear in their own observation/plan panels instead).

### Why
The entry observation and plan became stale as soon as the party changed the room, and counting observation passes as actions polluted teammate context and stall detection. Refreshing the plan only when the room state genuinely changes keeps decisions grounded without wasting LLM calls, and excluding bookkeeping entries keeps action history and stall logic focused on real gameplay moves.

---

## 2026-06-02 09:07:41 WIB

### What changed
- Room entry now runs a two-stage observe-then-plan pass: agents first OBSERVE (surveying the full room and listing notable objects as bullet points) and then PLAN (forming an ordered escape plan from that observation). No action is taken on the entry tick.
- The observation now surveys the *complete* room state rather than only currently-reachable objects — every object whose container chain roots in the room is listed with its state, a `[done|pending]` status, and a `(inside <container>)` note when nested, so planning accounts for clues and tools still locked in closed containers.
- Per-tick actions are now grounded in the room's agreed escape plan (formed on entry and persisted per-room) instead of a freshly re-derived observation each tick. The action prompt now receives the escape plan.
- The lead agent's observation and escape plan persist per-room (`room_observations`, `room_plans`) and are surfaced in the live tick header under new "OBSERVED RESULT" and "ESCAPE PLAN" panels.
- Observations and plans are now structured as bullet lists (rendered as bullets and logged), coercing varied LLM output shapes into clean bullets.

### Why
Re-observing and re-reasoning from scratch every tick let agents drift off strategy and ignore puzzle pieces hidden inside closed containers. Splitting entry into an observation that captures the whole room (handled and pending, reachable and nested) and a committed escape plan keeps each subsequent action anchored to one agreed strategy rather than re-deriving it tick by tick.

---

## 2026-06-01 08:33:41 WIB

### What changed
- Agents now run a dedicated observation phase — the first tick after entering a room is spent observing (each agent enumerates objects and their states and reasons about the goal via a new `observe` prompt) before any action is taken. Rooms remember they've been observed (`observed_rooms`) so the entry pass runs once per room.
- Every subsequent tick, agents re-observe the current state and then act *from* that fresh observation, which is passed into the action prompt.
- The object checkmark in the map/tick header now means "puzzle handled" — an object is marked resolved only when its state actually changed from initial or it was taken into inventory, no longer when merely examined.

### Why
Agents were acting on stale or shallow readings of the room and treating a bare examine as "done", which inflated progress and led to poor decisions. Forcing an explicit observe-then-act cycle (and a one-time entry observation) grounds each action in the current object states, and tightening the resolved-object definition makes the progress display reflect real puzzle advancement.

---

## 2026-06-01 00:50:17 WIB

### What changed
- World generation reverted to single-room mode — exactly one room that is both start and final, with empty adjacency, and whose `goal_completion` equals the `win_condition`. The full puzzle chain (clue → code/tool → final unlock) must now resolve inside that one room.
- The per-room `next_step` hint was removed entirely — the field is gone from the room model, generation prompt, gameplay/eval prompts, and the GM exit-evaluation narration (which now nudges toward completing the room goal rather than a next-step hint).
- Room truncation is now win-aware — when the LLM over-produces rooms, the room holding the `win_condition` object is kept (anchored first) instead of blindly keeping the first room, preventing unsolvable worlds.
- Object ids that collide with a room id are now rejected during world build, and the generation prompt forbids emitting an object that represents the room itself.

### Why
The project is collapsing back to a single room to validate the core solve loop before scaling, which makes the multi-room `next_step` hint obsolete. The win-aware truncation guards against the generator's tendency to over-produce rooms and accidentally drop the one containing the win object, while the id-collision guard prevents malformed worlds where an object shadows its own room.

---

## 2026-05-31 23:56:02 WIB

### What changed
- Player-facing hints no longer leak answers — literal codes/combinations (both the raw token like `safe_code_1942` and its bare digits `1942`) are now scrubbed from each room's `next_step` and from the `solution_path`, replaced with "the hidden code". Codes remain only in objects' `contains_info`/`requires_code`.
- Room goal prose is now bound to its actual completion condition — when the LLM's narrative goal diverges from what `goal_completion` checks, the goal is rewritten to a deterministic sentence derived from the condition (e.g. "Get X into the 'unlocked' state"). `known_info` goals are always normalized to a non-leaking sentence since their subject is a secret token.
- World generation logs any spoiler redactions and goal rewrites it applies.
- Generation prompt now requires the `goal` to name the same object/outcome `goal_completion` checks, and adds an explicit SPOILER RULE forbidding literal codes/passwords in `next_step` and `solution_path`.

### Why
Player-facing hints (`next_step` is shown live during play, `solution_path` in logs) could contain the literal answer code, handing the puzzle to the agents for free. Separately, the LLM sometimes wrote a goal describing a different step than the machine condition actually required, so the displayed objective misled players. Scrubbing secrets and binding goal prose to the real completion condition keeps puzzles solvable-by-effort and the displayed objective truthful.

---

## 2026-05-31 22:07:29 WIB

### What changed
- World generation is now resilient to malformed LLM output — object id, location, and tool-reference fields that arrive as dicts (e.g. `{"id": "crowbar"}`) or lists are normalized to plain id strings, descriptions/states are coerced to strings, and nested structures in scalar fields are dropped (`fuses` is kept only when it is a dict).
- Objects that still fail validation are skipped individually rather than crashing the whole generation step, so one bad object no longer aborts building the world.

### Why
The LLM intermittently emits object references and fields in the wrong shape (dicts/lists instead of id strings, malformed nested values), which previously caused Pydantic validation errors that crashed world generation. Coercing id-like fields and skipping unrecoverable objects keeps the generator robust against these schema deviations.

---

## 2026-05-31 10:03:20 WIB

### What changed
- Stall detection added — when every party member idles for two consecutive ticks, agents are nudged with an explicit prompt to examine an un-inspected object, apply a known clue/tool/code, or move rooms instead of waiting again.
- `wait` is now offered only as a true last resort — it appears in the action space solely when no productive action exists, and unparseable LLM replies fall back to the first productive option (preferring idle only if explicitly offered).
- Same-tick teammate awareness — each agent now sees the action a teammate JUST took earlier in the same tick (published in-place as players act), with prompt guidance to split work and avoid duplicating a teammate's current-tick action.
- Action visibility improved — the tick header now shows known info and a cumulative action log (last 12 entries), and each agent panel surfaces what the teammate did ("SAW : ...").

### Why
Smoke runs showed agents falling into mutual-idle cascades and duplicating each other's actions within a tick because each player only saw the prior tick's state. Demoting `wait` to a last resort, surfacing same-tick teammate actions, and nudging on detected stalls break these loops and push the party toward productive, non-overlapping progress.

---

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
