# Change Log

Chronological log of code changes. Newest entries appear first.

## 2026-06-08 08:38:56 WIB

### What changed
- When a declared key object still can't be materialised after the puzzle budget is exhausted, `puzzle_builder` now surgically repairs just the offending room skeleton(s) and rebuilds the puzzle, instead of jumping straight to a full world regeneration. This new escalation tier (bounded by the world-regen budget) reuses the `repair` prompt via a public `repair_world` entry point, keeping every unaffected room intact, and the repair prompt now instructs the model to swap an unbuildable key-object anchor for a simpler, physically buildable one and re-point the room's goal at it.
- The solvability check now rejects trivially-won worlds: a win condition whose object already starts in its target state (won at tick 0) or whose target state is an inert default (`visible`/`fixed`) is flagged as an issue, forcing `puzzle_builder` to regenerate a goal that demands real play. The world-builder generation prompts were updated to match — the win-condition room's goal state must be `unlocked`, never `visible`/`fixed`.
- The evaluation CLI flags were renamed to make the two evaluators distinct: `--eval` → `--narrative-eval` (slow end-of-run LLM judge) and `--trace-eval` → `--struct-eval` (fast inline per-node structural check), with the old names kept as deprecated aliases. Help text and usage examples were clarified, including that `--oracle-trace` is only a verbosity modifier for the narrative evaluator.

### Why
Worlds were occasionally generated that the oracle "won" at tick 0 because the win object already sat in its target state (or targeted an inert default state), making the benchmark meaningless; rejecting these forces a goal that requires actual solving. The surgical key-object repair tier avoids throwing away an otherwise-good world just because one anchor proved unbuildable, repairing only the room at fault. The flag rename removes longstanding confusion between the cheap structural eval and the expensive narrative judge.

---

## 2026-06-05 16:20:11 WIB

### What changed
- `generate` mode now runs the full two-stage pipeline again — after `world_builder` produces the rooms skeleton, `puzzle_builder` runs immediately on it, and both stages are rendered, timed, logged, and structurally traced together (single runs, captured runs, smoke, and per-node tracing all now scope to both `world_builder` and `puzzle_builder` in generate mode, undoing the prior world-only restriction).
- The win condition is now an explicitly stored field owned by `puzzle_builder` rather than a property computed on the fly from the final room's goal. `world_builder` leaves it empty, and once objects exist `puzzle_builder` derives and writes it (via a new shared `derive_win_condition` helper), so a rooms-only skeleton carries no win target until the puzzle stage assembles one. World repair no longer re-passes the win condition (it follows from the merged rooms).
- A room's declared `key_objects` are now enforced as mandatory anchors throughout puzzle building: a new structural check rejects any world where a room's `key_objects` are not all materialised as real objects with matching ids (causing an automatic retry), and the generation prompt now lists each room's key objects and forbids dropping, renaming, or substituting them. To support this, key objects are no longer silently scrubbed from rooms and are always kept through orphan pruning even if not strictly reachable.

### Why
The two stages had been split apart so that generate mode produced only a rooms skeleton, but generating and evaluating a complete world in one pass is what the dataset pipeline needs, so the puzzle stage was rejoined to generate mode. Making the win condition a stored field owned by the puzzle stage removes the awkwardness of a "computed" win target that was meaningless for a rooms-only world and gives each stage a clear owner for it. Enforcing key objects as hard anchors prevents the puzzle builder from quietly dropping the headline objects a room's goal was designed around — previously such a drop could be masked instead of surfaced and retried.

---

## 2026-06-05 16:04:24 WIB

### What changed
- World generation now repairs failed rooms surgically instead of regenerating the whole world. When the structural eval flags issues that name specific rooms (e.g. `room '<id>': ...`), the world builder sends only those rooms — alongside the full world for context and a per-room list of their issues — to a new `repair` prompt, then merges the fixed rooms back in and re-runs adjacency repair. Only when an issue names no room (e.g. a missing scenario) does it fall back to a full regeneration. The retry log now prints whether each attempt is a targeted repair or a full regeneration.
- World generation prompts moved to a dedicated `prompts/world_builder/` set (system, generation, generation_bank, plus the new repair prompt), replacing the prompts previously loaded from the `game_master` namespace.
- Each generation node now records its per-attempt retry history (attempt number and issues) to a separate `attempts.json` under the node's log directory, and the rejected-attempt messages no longer embed the full raw LLM payload — keeping the message trail and `output.json` lean while the structured attempt log lives in its own file.
- The `generate` mode now runs only the `world_builder` stage; the `puzzle_builder` stage no longer runs in generate-only mode (single runs, captured runs, smoke, and per-node structural tracing all scope their node set to `world_builder` when in generate mode).
- When `--smoke` and `--node` are passed together, `--smoke` now takes precedence and `--node` is ignored, with the smoke runner using `--mode` to control which pipeline runs.

### Why
World skeletons were thrown away and regenerated wholesale even when only one or two rooms had structural defects, which was slow and discarded otherwise-valid rooms (and stories) on every retry. Targeted per-room repair keeps the good rooms and fixes only what failed, and moving the world-builder prompts into their own namespace plus restricting generate mode to that single stage reflects the now-independent world/puzzle split. Pulling the raw payload out of the retry messages into a dedicated attempts log keeps the conversation trail readable while preserving the full retry trace for inspection.

---

## 2026-06-05 14:45:08 WIB

### What changed
- Added a per-node structural eval trace (`--trace-eval`): as the pipeline runs, each stage's deterministic structural checks (world structure for `world_builder`, solvability for `puzzle_builder`) run inline and print PASS/FAIL plus the specific issues found, with a final summary listing any failed nodes. This is independent of `--eval`, which still runs the narrative LLM judge once at the end. The trace also works under `--smoke`, writing each run's per-node `eval.json` and a `trace_eval_summary.json` roll-up, and under `--log`.
- When tracing, the fully-assembled world (after `puzzle_builder`) additionally gets the LLM-free policy benchmark emitted as `benchmark.json`, so the trace captures how baseline policies fare on the world.
- Added the ability to run a single pipeline node in isolation (`--node NAME`): `world_builder` runs from a `--theme` alone, while every other node loads its upstream inputs from a saved `output.json`/`world.json` via `--from PATH`, validates that the inputs it depends on are present, runs the node, and writes the result to `logs/<node>/output.json`.
- Added a `--theme` flag that supplies the generation theme directly and skips the interactive theme picker, for both the normal pipeline and single-node runs.
- Solvability checking now flags goals that reference objects the party cannot actually act on: an `object_state` goal object must physically live in the room it gates, and a `has_item` goal item must live in a room reachable on the path leading to that room (so the party can carry it in). Object home rooms are resolved by walking nested-object location chains (guarding against cycles).
- Renamed the oracle solve-trace flag from `--eval-trace` to `--oracle-trace` to free up `--trace-eval` for the new per-node structural trace.

### Why
The pipeline previously only surfaced quality problems via the end-of-run narrative judge, making it hard to tell which stage produced a structurally broken world; running each stage's structural eval inline gives fast, deterministic per-stage feedback and lets smoke runs roll up pass/fail rates. Being able to run and trace a single node against saved upstream state shortens the debug loop when iterating on one stage. The new goal-reachability checks close a solvability gap where a room's win condition could point at an object that was never present or reachable, which would make the room unwinnable.

---

## 2026-06-05 14:01:23 WIB

### What changed
- World structural validation now enforces that each room's goal revolves around a real key object: when a room's `goal_completion` names an `object_id`, that id MUST appear in the room's `key_objects` list, otherwise the world is flagged as invalid. The generation prompt and JSON template were tightened in lockstep — the `goal` sentence must explicitly name the key object the party obtains/uses/manipulates, that object must be listed in `key_objects`, and it must be the same object `goal_completion` references.
- `solution_path` is no longer scrubbed of secrets. Because the path is an internal validation/debug artifact never shown to the player, the real object ids, clue tokens, and codes are now preserved in it (exactly what makes it useful for confirming solvability). The dedicated secret-token/spoiler redaction pass (`_secret_tokens`/`_scrub_spoilers`) was removed; only genuinely hallucinated ids that match nothing real are still scrubbed, with the valid-id set widened to include `contains_info` and `requires_code` tokens so legitimate clue/code references survive.
- Serialized world JSON is now wrapped under a top-level `"world"` key instead of being the bare world object at the document root.

### Why
Goals could be authored around objects that weren't actually the room's key objects, letting the puzzle's win target drift from the items the prompt told the party to focus on; requiring the goal, `key_objects`, and `goal_completion` to all name the same object keeps the objective coherent. Since `solution_path` is internal-only, blanking its codes and ids to "the hidden code"/"the object" stripped exactly the information needed to confirm a world is solvable, so spoiler scrubbing was dropped in favor of preserving real references. Wrapping the JSON under `"world"` gives the output a stable envelope for downstream consumers.

---

## 2026-06-05 12:41:38 WIB

### What changed
- Dataset generation now mints **synthetic contrastive DPO pairs for `world_builder`** by corrupting an accepted room skeleton on the structural axes the world evaluator checks: a dangling exit pointing at a nonexistent room (`broken_adjacency`), a one-directional exit with no mirror (`unmirrored_adjacency`), a room stripped of its win condition (`missing_room_goal`), and a duplicated room id (`duplicate_room`). The accepted skeleton becomes `chosen` and the corrupted copy `rejected`, with the failed axis labelled on each pair. Each corruption is only kept if it actually trips `_eval_world_structure`, and the pairs honour the same per-world cap and shuffle as puzzle DPO.
- A new `--world-dpo-axes` flag selects which world corruptors run (default `all`, or `none` to disable), mirroring the existing `--dpo-axes` for puzzles; the run header now reports puzzle and world DPO axes separately.
- A new `--difficulty` preset (`easy` / `hard` / `env`) sets `HARD_MODE` before settings load, routes output to `dataset/<difficulty>/` so easy and hard runs never share files, and stamps a `difficulty` field onto every written example. `env` (default) preserves legacy behaviour: honour the existing `HARD_MODE` env and write to bare `dataset/`.

### Why
World skeletons almost always pass structural evaluation on the first try, so the live retry loop produced essentially zero natural preference pairs for `world_builder` — the synthetic corruptors give that half of the dataset real DPO signal, matching what was already done for puzzles. The difficulty preset and per-example tagging keep easy and hard datasets cleanly separated and self-describing so the two profiles can be generated and consumed without colliding.

---

## 2026-06-05 10:42:33 WIB

### What changed
- The default game-master model shipped in `.env.example` is now `qwen3:14b` (was `qwen3.5:9b-mlx`), so fresh checkouts generate worlds with the larger, non-MLX model out of the box.
- Dataset generation now defaults to `HARD_MODE=false` in `.env.example`, making the easier generation profile the default for new setups.

### Why
The example environment was retargeted at a more capable game-master model for higher-quality world generation, and hard mode was turned off by default so the generator produces solvable worlds more reliably on a fresh setup. (Regenerated dataset artifacts — manifest counts and puzzle jsonl files — accompany this run but are produced output, not behavior changes.)

---

## 2026-06-05 10:14:55 WIB

### What changed
- The `--judge-dpo` flag is now a **hard gate** inside a single unified revision loop rather than a separate one-shot pass that ran after acceptance. When enabled, each puzzle world is revised until it satisfies BOTH the deterministic policy (solvable + deep) AND the LLM judge, or it is discarded — so any world written to the dataset is guaranteed to pass both. The deterministic check runs first and stays authoritative; the slow, stochastic judge runs only once a world is already structurally valid, saving a judge call on every broken attempt.
- Every regen that clears all issues (both judges) now captures the bad→good transition as a real DPO pair, and the pair's `axis` is labelled `compliance` when the deterministic check was already clean (a narrative/prompt-compliance fix) versus `solvability` when structure was the failure — so compliance and solvability contrasts are distinguishable in the dataset.
- The run header now reports the judge as an `llm judge gate` ("revise until policy+judge pass"), and a discarded world's message states that policy+judge were not both satisfied.

### Why
The previous design judged compliance only in a separate stage bolted on after the deterministic gate, which duplicated the regen logic and could let a world be kept as an SFT target without the judge and the policy ever being satisfied together. Folding the judge into one revision loop as a hard gate guarantees every kept world clears both checks, captures every fix as a labelled DPO pair, and removes the redundant second code path.

---

## 2026-06-05 09:56:18 WIB

### What changed
- The dataset generator can now mine *real* (non-synthetic) compliance DPO pairs via the new `--judge-dpo` flag. After a world clears the deterministic solvable+deep gate, an LLM-as-judge quick pass (`quick_eval_for_feedback`) runs once; if it flags narrative/prompt-compliance violations that a single feedback regen then fixes (re-verified to stay structurally valid AND clear the judge), the bad→good pair is captured on a new `compliance` axis and the judge-approved world replaces the original as the SFT target. A flaky or unavailable judge never blocks generation (failures return no violations).
- Added a `--min-quality SCORE` post-generation filter that scores every accepted puzzle SFT world with the full `narrative_eval` and deletes any whose overall score falls below the threshold (0.0 = off). It runs once after generation so the generation pass stays deterministic, and only puzzle worlds (which carry objects and a solution path) are scored; the subsequent merge/manifest reflect the cut.
- The run header now reports the judge-flagged-DPO and quality-post-filter settings.

### Why
The deterministic gate guarantees structural solvability but says nothing about narrative quality or prompt compliance, so the dataset lacked a contrastive signal for those failure modes. Routing the LLM judge into a single feedback-regen loop yields genuine bad→good compliance pairs, and the optional quality post-filter lets low-scoring worlds be dropped from the final SFT set without slowing or de-determinizing generation itself.

---

## 2026-06-05 09:42:37 WIB

### What changed
- Added a fifth synthetic contrastive DPO axis for `puzzle_builder`, `untakeable_tool`, which breaks the win chain at an *intermediate* dependency: it makes a tool the solution relies on impossible to pick up (re-introducing exactly what `_make_required_tools_takeable` repairs). As with the other axes, a pair is kept only after the corrupted copy is re-verified to genuinely fail the end-to-end oracle, and `untakeable_tool` is now a selectable value for `--dpo-axes`.
- Synthetic DPO generation now caps how many pairs a single accepted world can contribute, via the new `--max-dpo-per-world` flag (default 2; 0 = no cap). Corruption axes are visited in a seeded-shuffled order and only the first N validated pairs are kept, so no single accepted world gets reinforced across every axis (which would invite memorizing its story) while the rotating subset keeps every axis globally balanced. The run header reports the per-world cap.
- Added a `--stats` mode to the dataset generator that reports SFT/DPO counts per target and a per-axis breakdown (count and share) of puzzle DPO pairs from the existing dataset, then exits without any generation or LLM calls. Live-retry pairs (which carry no `axis` field) are bucketed as `live-retry`.

### Why
The dataset needed a contrastive signal for a common, subtle failure mode the existing axes missed — a required tool that exists but can't be carried — distinct from a fully unsolvable win object. Capping DPO pairs per world prevents over-reinforcing any single accepted story while keeping axis coverage balanced, and the `--stats` view makes the resulting axis distribution inspectable without re-running generation.

---

## 2026-06-05 09:27:51 WIB

### What changed
- The dataset generator now builds fine-tuning data for both generation agents independently instead of treating world generation as a single step. Each accepted run emits training examples into two separate trees — `dataset/world/` (theme → rooms-only skeleton, owned by `world_builder`) and `dataset/puzzle/` (rooms skeleton → objects, locks, clues, and solution path, owned by `puzzle_builder`) — each with its own SFT and DPO files plus merged `sft_all.jsonl`/`dpo_all.jsonl`.
- The pipeline now runs the two stages with independent validation and retry: the rooms skeleton is gated by the deterministic structural check and the full puzzle by the oracle-backed `_eval_puzzle`, each retrying up to the attempt budget and capturing a DPO pair on every real violation→fix transition with the exact correction prompt the model received.
- Added synthetic contrastive DPO generation for `puzzle_builder`. From each accepted world, corruptors deliberately degrade a single named axis — `unsolvable` (re-lock the win object), `shallow` (open every gate so the chain collapses), `orphans` (inject objects gated by codes nothing produces), and `phantom` (reference object ids that don't exist in the solution path). A pair is kept only after the corrupted copy is re-verified to genuinely fail on its axis, so each chosen/rejected pair shares an identical prompt and teaches only the defect.
- New CLI controls: `--target {world,puzzle,both}` selects which agent(s) to build data for, `--dpo-axes` selects which synthetic corruption axes to apply (or `none`), and `--seed` makes the synthetic corruption reproducible. The manifest and run summary now report per-theme world/puzzle SFT and DPO counts side by side, and `--resume` accounts for both targets when deciding whether a theme is complete.

### Why
The real game generates worlds in two stages owned by different agents, so a single combined dataset couldn't train either agent cleanly. Splitting the dataset per agent — and adding verified, single-axis synthetic DPO pairs that share the prompt with their accepted counterpart — gives each fine-tune a clean bad→good signal isolated to the exact defect rather than to incidental story differences.

---

## 2026-06-05 08:45:38 WIB

### What changed
- World generation now self-validates and retries. After building the rooms-only skeleton, the world builder runs a deterministic structural check (room count, required metadata, duplicate ids, and mirrored adjacency topology) and regenerates the world up to the configured attempt budget when violations are found, emitting one message per attempt so the retry trail is captured.
- The puzzle builder was rebuilt around a unified "eval B" gate that runs four deterministic passes with no LLM calls: static backward-chain solvability, a per-room object-count minimum, a per-room oracle that confirms each room's goal is achievable using only that room's local objects, and a global end-to-end oracle that confirms full solvability and chain depth. Violations are fed back into the next generation prompt as feedback.
- When a puzzle remains genuinely unsolvable after the puzzle-attempt budget is exhausted, the builder now discards the entire world and regenerates it from scratch (world builder → puzzle builder) rather than shipping an unwinnable world, up to a world-regen budget. Cosmetic/structural issues no longer trigger a full world discard.
- Rooms now enforce a minimum object count per room (1 in standard mode, 8 in hard mode). The generation prompt states this requirement explicitly, and after orphan-pruning the builder backfills any under-filled room with inert scenic props so the count survives pruning instead of fighting it across attempts.
- Orphan pruning now seeds reachability from object ids mentioned in the solution path, so intermediate result objects referenced only in the solution narrative are no longer incorrectly pruned.
- The separate Game Master eval node was removed from the pipeline. Room-exit gating is now a deterministic in-process check (no LLM call), and victory/time-up detection happens at the end of each gameplay tick with a conditional edge routing straight to END, eliminating an LLM round-trip and the `pending_directive` state field.
- Smoke runs and single runs now emit a per-node timing table (elapsed time and percentage per stage), write a clean human-readable run summary, and serialize the final assembled world to a `.world.json` file alongside the raw stdout capture.

### Why
The world/puzzle generation pipeline was shipping broken or unwinnable worlds and wasting LLM calls on eval steps that could be done deterministically. Splitting validation into a structural "eval A" for the room skeleton and an oracle-backed "eval B" for the full puzzle — with feedback-driven retries and a world-discard fallback — makes generation reliable for the fine-tuning dataset. Collapsing the Game Master eval node into deterministic checks removes per-tick LLM latency, and the timing/summary output makes slow runs diagnosable.

---

## 2026-06-04 15:08:33 WIB

### What changed
- Per-node timing is now printed to stdout for all major pipeline stages: `character_master` logs how long character generation took and how many characters were produced; `player_agent` logs how long character selection took per agent; `gameplay_node` logs elapsed time at the end of each tick (both the observe+plan path and the full action path); `game_master_eval` logs elapsed time at every exit point (victory, time-up, and normal tick completion); `puzzle_builder` logs total time and attempt count together instead of just attempt count.
- The puzzle builder now records a structured retry trail: each rejected attempt is captured with its attempt number, rejection reason, and violation list, then emitted as a sequence of `AIMessage` entries (one per attempt) so node logs contain the full retry history including why each attempt failed. The final accepted world is emitted as a separate message labelled `(final)`.
- The `prompt_compliance` evaluation dimension description was updated to clarify it judges both world-builder and puzzle-builder output (not just game master output), matching the split-phase architecture introduced in the prior session.

### Why
The pipeline was opaque about where time was being spent — a slow run could be caused by LLM latency in any of several nodes and there was no way to diagnose which. Adding per-node timing to stdout gives immediate visibility during live runs and smoke tests. The retry trail in puzzle builder messages was added so the fine-tuning dataset pipeline can capture rejected attempts as DPO negative examples rather than silently discarding them.

---

## 2026-06-04 14:26:51 WIB

### What changed
- A new `puzzle_builder_node` graph step has been introduced. The world-builder now produces only the room layout and goals; all puzzle objects and the solution path are constructed by the separate puzzle builder in the next pipeline step. This splits world generation into two focused phases.
- The `game_master` agent no longer generates `WorldObject` entries directly — the `WorldObject` import and the `_OPTIONAL_OBJECT_FIELDS` allow-list it relied on have been removed, and object construction responsibility is fully delegated downstream.
- A fine-tuning dataset generator (`benchmark/generate_dataset.py`) has been added. It drives the same generation + solvability-check + LLM-judge retry loop used in the live pipeline, iterating per story theme until a target count of validated worlds is reached. Each run produces SFT examples (instruction-tuning pairs) and DPO preference pairs (chosen = accepted world, rejected = a world that had violations corrected into the accepted one), with per-theme JSONL files, merged flat files, and a manifest.
- Generation prompts (`prompts/game_master/generation.txt` and `generation_bank.txt`) have been updated to reflect the narrowed scope of the world-builder (rooms and goals only, no objects).
- The graph wiring and `main.py` entry point have been updated to include the new `puzzle_builder_node` in the execution sequence.

### Why
Generating rooms, objects, and puzzle logic in a single LLM call produced worlds where the solution chain and object graph were often inconsistent. Splitting into world-builder (rooms + goals) and puzzle-builder (objects + solution path) allows each phase to be validated, retried, and fine-tuned independently — which also enables the DPO dataset to capture clean before/after correction pairs for training.

---

## 2026-06-04 14:13:13 WIB

### What changed
- The world-building node has been renamed from `game_master` to `world_builder` throughout the system — the graph entry point, all edges, log-node lists, direct invocation paths, and internal documentation now use the new name.

### Why
The `game_master` name was ambiguous given the separate `game_master_eval` node that also exists in the graph. Renaming to `world_builder` makes the node's role — generating the escape room world — explicit and distinguishable from the eval node that judges whether the generated world is acceptable.

---

## 2026-06-04 11:54:10 WIB

### What changed
- A BFS-based policy (`bfs_policy`) is now available in the benchmark module. Given a world, it explores the full reachable state space from the initial party state and returns a closure that replays the shortest winning action sequence found — or falls back to `heuristic_policy` if no solution is found within the node budget (`max_states=50_000`).
- The benchmark's single-world episode counts have been raised: `first` and `heuristic` now run 20 episodes each (previously 1) to produce statistically meaningful win percentages; `bfs` runs once (it is deterministic and stateful — the pending list drains).
- `bfs_policy` is registered as a fourth baseline in both `compute_policy_benchmark` and the `main()` multi-world runner, so every benchmark table now includes a BFS column alongside random, first, and heuristic.
- `run_policy` now supports *factory* policies — callables that take a world and return a policy closure — detected via signature inspection rather than a test call. This allows BFS (whose closure is stateful) to be regenerated fresh for each episode without special-casing it in the call site.

### Why
The existing random/first/heuristic baselines couldn't distinguish between a world that is hard and one that is structurally unsolvable because none of them are guaranteed to find the optimal path. A BFS policy, when it succeeds, proves the world is solvable and provides an optimal lower-bound on ticks-to-win, making it a ground-truth reference for evaluating the heuristic and LLM-driven policies. Raising the episode counts for the other deterministic policies provides richer aggregate statistics for comparison.

---

## 2026-06-04 11:45:07 WIB

### What changed
- A static backward-chain solvability checker (`check_solvable`) is now available in the benchmark module. It walks backward from the win condition through every prerequisite in the object graph — resolving `requires_code`, `requires_tool`, `requires_power`, `requires_liquid`, and `goal_completion` targets — and returns a `SolvabilityReport` with a boolean result and a full list of blocking issues. Runs in O(objects²) with no simulation or tick budget.
- The checker catches: missing `contains_info` suppliers for code or info goals, bare digit codes with no matching token, non-existent or non-takeable required tools, circular tool dependencies, power gates with no satisfying fuse panel, `known_info` goals with no upstream producer, `goal_completion` targets referencing non-existent objects, a missing or non-existent win-condition object, and rooms disconnected from the start room.
- After each world is generated in the live game, the solvability check now runs automatically and prints a one-line SOLVABLE confirmation or a bulleted list of structural issues to stdout before the policy benchmark table.

### Why
The policy benchmark could fail or time out for reasons that were hard to distinguish from a solvable-but-hard world versus a structurally broken one. A fast, simulation-free static check separates structural impossibilities (missing clue suppliers, unreachable tools, disconnected rooms) from runtime policy failures, giving immediate diagnosis before a single tick is simulated.

---

## 2026-06-04 11:35:18 WIB

### What changed
- The "decoy" concept has been removed from the system: the `--decoys` CLI flag, `DECOYS` env var, and `settings.decoys` field are gone; the generation prompt no longer instructs the LLM to scatter red-herring objects, and internal filters that excluded objects with `"decoy"` in their ID have been dropped.
- The generation retry loop now collects structured violation lists from a fast deterministic + compliance eval (`quick_eval_for_feedback`) before regenerating, and injects those violations as a correction message to the LLM so it can fix specific issues rather than starting from scratch blindly.
- A new `quick_eval_for_feedback` function has been added to the narrative evaluator, running only the three fastest dimensions (required-tool presence, solvability, prompt compliance) and returning a flat `violations` list suitable for pasting into a correction prompt.
- The `_generate_world` logic has been refactored into `_generation_prompt` and `_build_world_from_response` helpers, with a new `_generate_world_with_feedback` variant that sends a multi-turn correction message when violations are known.
- The bank generation script (`generate_bank.py`) has had the `--decoys` argument and `decoys` parameter removed end-to-end.

### Why
Decoy objects were conceptually at odds with the pruner, which already drops any object not on the solution path — making authored decoys inert noise the pruner would silently remove anyway. Removing the concept cleans up the API surface and aligns the prompt with actual runtime behavior. The feedback-driven retry loop replaces blind regeneration with targeted correction, giving the LLM judge violations concrete instructions rather than hoping a fresh attempt accidentally avoids the same problems.

---

## 2026-06-04 11:11:45 WIB

### What changed
- When the deterministic `solution_path` checks fail (ghost object IDs or no parseable engine actions), an LLM judge is now invoked with the oracle's winning trace as ground truth to produce specific, actionable feedback for the GM on how to fix the path.
- Unparseable solution paths (steps with no recognized engine verbs) are now scored as a failure (`-0.4`) and flagged with an explicit issue message listing the required engine verbs, rather than being silently skipped with a neutral note.
- The oracle trace is now threaded from `evaluate_world` into `_eval_solution_path`, making it available as ground truth for the LLM judge when issues are detected.

### Why
The deterministic checks could identify *that* a solution path was broken but not *how* to fix it — the GM was left with a score and no concrete guidance. Feeding the oracle's verified winning trace to an LLM judge when issues arise closes that gap with actionable repair instructions. Treating unparseable paths as scoring failures (rather than neutral skips) aligns the score with the real quality of the authored path.

---

## 2026-06-04 10:08:06 WIB

### What changed
- Ghost object IDs are now scrubbed from `solution_path` after world generation: any snake_case token that looks like an object ID but was dropped by the build pipeline is replaced with "the object" so the recorded solution stays readable and internally consistent.
- Spoiler redaction now runs a second digit-only pass: after replacing full secret tokens (e.g. `safe_code_742`), a regex pass also redacts bare digit sequences (3+ digits) that match a secret code's numeric portion, catching cases where the raw number slipped through even after the token was removed.
- Rooms are now padded to a minimum of 5 objects after orphan pruning: when the prune pass drops puzzle-irrelevant objects, harmless non-scenic orphans with no preconditions are promoted back into the keep set to satisfy the 5–10 objects-per-room density requirement from the generation prompt.
- The generation prompt now includes explicit narrative quality requirements evaluated by an LLM judge: each scenario must contain sensory detail and mood; room descriptions must include at least two vivid specifics; all rooms must share a consistent story world; and the world must embed at least one narrative plot twist hinted at through clue text and room descriptions.

### Why
The build pipeline could prune objects that the LLM referenced in `solution_path`, leaving dangling IDs in a path that was supposed to be replay-safe. The digit-only spoiler pass closes a gap where a bare number leaked even after its full token was redacted. Room padding prevents the pruner from producing sparse rooms that violate the density rule. The narrative quality requirements in the prompt directly target the dimensions scored by the LLM-as-judge evaluator, so generation is aligned with evaluation criteria from the start.

---

## 2026-06-04 09:44:13 WIB

### What changed
- The pipeline now supports a `--mode` flag with two options: `generate` (world generation only, skipping characters and gameplay) and `full` (the complete pipeline, default). This replaces the previous hardcoded full-pipeline behavior.
- An interactive theme picker is now presented at startup: the user chooses from a curated list of 10 themes (Haunted House, Murder Mystery, Prison Break, etc.) via arrow-key selection before any generation runs. The `"pirate"` hardcoded theme is gone from all code paths.
- The LLM is now capped at 4096 output tokens (`num_predict=4096`) and thinking mode is explicitly disabled (`extra_body={"think": False}`), preventing runaway or reasoning-heavy responses from the model.
- A `--eval` flag is available to evaluate world narrative quality using the LLM-as-judge + oracle pipeline. Supply a saved `output.json` path to evaluate an existing world, or omit the path to evaluate the world produced by the current run. Works in both single-run and smoke modes.
- An `--eval-trace` flag prints the oracle's tick-by-tick solve trace alongside the narrative evaluation report.
- Smoke runs now accept `--mode`, `--eval`, and `--eval-trace` so each smoke iteration can be generated in either mode and optionally evaluated, with the evaluation result written to `run_NNN.eval.json` per run.

### Why
Benchmarking world generation in isolation was slow because the full pipeline (characters, gameplay) ran every time. A dedicated `generate` mode lets the team rapidly iterate on world quality without running agents. The theme picker replaces the hardcoded `"pirate"` string so test runs reflect diverse settings. The token cap and thinking-disable prevent the model from stalling on long reasoning chains during generation. The `--eval` flag wires the narrative evaluator (added in the prior session) into the main CLI so quality scores are a first-class output rather than a separate script.

---

## 2026-06-04 08:38:37 WIB

### What changed
- Rooms now include scenic (atmosphere-only) props: the world model, generation prompts, and object builder all support a `scenic` boolean flag that marks furniture, decorations, and ambient details as pure flavor with no puzzle logic.
- Each room is now required to contain 5–10 objects total — puzzle-critical pieces plus 2–4 scenic props — so generated spaces feel populated and real.
- Scenic objects are permanently exempt from orphan pruning: the post-generation prune pass that removes objects with no path to the solution now keeps any object flagged `scenic: true`, so atmosphere props are never silently dropped.
- The action prompt now shows a type-level hint next to any locked/hidden object (e.g. "needs a 3-digit code", "needs a tool + power") so agents understand what is required to open an object without being told the answer, reducing blind fumbling while preserving discovery.

### Why
Generated rooms felt sparse and puzzle-mechanical with no environmental texture, and agents were wasting ticks attempting actions they had no means to satisfy because the object listing gave no signal about what an object needed. Adding scenic props (with an exemption from pruning so they survive) fills rooms with atmosphere, while the requirement hint gives agents just enough type information to direct their search without leaking the solution.

---

## 2026-06-03 16:22:51 WIB

### What changed
- World generation now prunes orphan objects after building: any object that no solution path touches (computed by backward closure from each room's goal subject through tool/code/liquid/power/container dependencies) is dropped so every remaining object is load-bearing and the action space carries no inert decoys.
- Decoys now default to 0 (`DECOYS=0`), since the pruner removes inert red-herrings anyway; raising it only matters if the generation prompt is changed to make decoys coherent.
- Solvability regeneration now runs in every mode, not just hard mode — a standard world can be just as unwinnable, so generation retries until the oracle confirms the world is winnable.
- Hard mode additionally rejects shallow worlds: a generated world must clear a minimum oracle-measured dependency chain depth (`CHAIN_DEPTH`) or it is regenerated, so a technically-winnable-but-trivial puzzle isn't shipped.
- The live game now prints a per-world policy benchmark (win%, ticks-to-win, objects resolved for random/first/heuristic baselines) when a world is generated, mirroring the aggregate benchmark output; the smoke runner writes the same numbers to a `run_NNN.benchmark.json` file per run.
- When a room's goal is already complete at tick start, the Game Master now issues its "advance to next room" directive in-tick — before agents choose — so the party moves on immediately instead of re-working the finished room and drifting back. The directive is computed once and reused by the eval node, avoiding a duplicate LLM call and a duplicate panel.
- The OBSERVED RESULT panel is now a live global view rebuilt every tick from observations across all visited rooms (plus inventory), grouped by room with the current room first, so object states never go stale; when a room's goal is done the ESCAPE PLAN panel shows the GM advance directive instead of the now-solved room plan.

### Why
The puzzle quality and pacing needed tightening: inert decoy objects bloated the action space without adding difficulty, unsolvable or shallow worlds were reaching live play, and per-world baseline numbers were hard to see. The in-tick GM directive and live observation panels fix agents wasting ticks on finished rooms and reasoning from stale state.

---

## 2026-06-03 14:42:10 WIB

### What changed
- The gameplay loop is now headless and tick-driven: `gameplay_node` runs exactly one tick per graph invocation and returns, and a new `game_master_eval_node` checks for victory, time-up, or a local room-goal completion after each tick — looping back to gameplay or terminating accordingly. Victory and time-up rendering have moved into the eval node.
- The LangGraph graph is wired with a conditional edge from `game_master_eval` back to `gameplay` (or to END), replacing the old single terminal edge from `gameplay` to END.
- Party room-movement is now atomic: all `go` actions within a tick are deferred and resolved together, so a later agent cannot pick a different exit mid-tick and split the party. A single departure destination is agreed on and all agents move as one.
- Each agent now receives a cross-room global object table when observing: every object the party has seen across all rooms (state, location, agent notes) is injected into the observe prompt, enabling agents to cross-reference clues and tools from earlier rooms when planning.
- Agents collect per-object notes during observation (a structured `object_notes` map alongside the bullet list), which are stored in a new `global_object_observations` table on `PartyState` and refreshed after every action tick.
- The action-space `examine` verb is now suppressed for takeable objects that haven't been picked up yet, surfacing `take` first so agents grab items in one tick rather than examining then taking on separate ticks.
- The room fingerprint (used to decide whether the lead agent should re-observe) is now stored as a string on `PartyState` (`last_fingerprint`) so it survives across graph node boundaries between ticks.
- World generation now repairs missing win conditions: if the final room's `goal_completion` is absent or not an `object_state` type, a valid win condition is synthesised from the room's locked key objects so the game can terminate correctly.
- Fuse panels can no longer be assigned to decoy objects during power-gate repair; decoys are excluded from host selection in both the primary and fallback candidate passes.

### Why
The gameplay loop was a single monolithic `while` loop inside one graph node, making it impossible to inspect or benchmark individual ticks and preventing the graph from checkpointing or branching mid-run. Splitting into a per-tick `gameplay_node` + `game_master_eval_node` pair with a conditional back-edge makes the loop observable and headless-benchmark-friendly. The cross-room observation table and global notes were added to reduce the information asymmetry agents faced when a clue from room one was needed to unlock something in room two.

---

## 2026-06-03 09:03:28 WIB

### What changed
- Characters no longer have mechanical abilities — the `Ability` model, all ability-related constants (`ABILITY_EFFECTS`, `ABILITY_TRIGGERS`), and the per-party `ability_rooms_triggered` tracking field have been removed from the data model entirely.
- Character generation no longer asks the LLM to assign or describe abilities; the generation prompt and JSON schema no longer include the `ability` block.
- Agent action and plan-proposal prompts no longer reference a character's ability; agents reason only from their role when proposing and executing plans.
- Character display in the player-selection screen and the live game startup no longer shows ability name, effect, or uses.

### Why
The ability system added complexity (effect resolution, use-tracking, per-room trigger state) without being reliably exercised or benchmarked in the current headless evaluation loop. Removing it simplifies the state model and prompts so the benchmark can focus on core puzzle-solving behavior driven by role alone.

---

## 2026-06-03 08:51:39 WIB

### What changed
- When the Game Master issues a directive to advance rooms, the action space presented to each agent is now narrowed to only that mandated move (`go <dest>`), so agents can no longer accidentally pick a wrong exit or take an unintended action when the GM has already declared the room complete.
- The stale room escape plan is now suppressed for the tick in which a GM directive is active, preventing the old plan from competing with the directive in the agent's prompt.
- The effective action space is recomputed after each agent acts within a tick, so mid-tick state changes (e.g. a teammate completing the room goal) are reflected before the next agent decides.

### Why
Agents were occasionally choosing an incorrect exit or acting on a residual room plan even after the GM had issued an explicit move directive, causing the party to stall or deviate. Narrowing the action space to the directed move and clearing the competing plan text ensures the GM directive is unambiguously acted on.

---

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
