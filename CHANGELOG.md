# Change Log

Chronological log of code changes. Newest entries appear first.

## 2026-06-12 14:52:21 WIB

### What changed
- The LAN fan-out mechanism previously limited to per-room world theming has been generalized into a shared `get_worker_llms(role, llm=None)` helper in `utils/settings.py`, and renamed from `OLLAMA_THEMING_WORKERS` to `OLLAMA_WORKERS` (a comma-separated list of additional Ollama base URLs). `Settings.ollama_workers` reads `OLLAMA_WORKERS`, falling back to `OLLAMA_THEMING_WORKERS` as a legacy alias. `get_worker_llms` returns `[llm (local)] + [get_llm(role, base_url=url) for url in worker_urls]`, defaulting `llm` to `get_llm(role)` if not given — any caller with N independent LLM calls can now round-robin them across the local Ollama instance plus any configured LAN workers.
- `puzzle_graph.apply_theming` now calls `get_worker_llms("game_master", llm)` instead of building its worker list inline from `Settings().ollama_theming_workers`, preserving its existing round-robin-across-rooms behavior but sharing the new helper.
- `storyboard_builder_node` now runs its two independent generation passes — pass 2 (`_run_beats_pass`, clue layer) and pass 3 (`_run_flavor_pass`, atmosphere layer) — concurrently instead of sequentially, via a `ThreadPoolExecutor(max_workers=2)`. Each pass now takes an optional `llm` parameter (threaded through to `_call_json(system_prompt, user_prompt, llm=None)`, which defaults to `get_llm("storyboard")` when `llm` is `None`); the two passes are dispatched against `llms = get_worker_llms("storyboard")`, with the beats pass on `llms[0]` (local) and the flavor pass on `llms[1 % len(llms)]` (a worker if one is configured, otherwise local again).
- Added two new scripts for setting up distributed Ollama inference: `scripts/advertise_ollama.py` (run on each additional Mac; advertises its local Ollama instance via Bonjour/mDNS as `_ollama-worker._tcp.local.` on the port derived from `OLLAMA_HOST`, default 11434 — Ollama itself keeps serving normally, this only registers the mDNS record) and `scripts/discover_ollama.py` (browses for `_ollama-worker._tcp.local.` services for a configurable timeout, default 3s, printing one `<ip>:<port>` per line). Added `scripts/setup_ollama_worker.sh`, a per-worker setup script (mirroring `setup_worker.sh`'s conda-env bootstrap) that sets up the `escape-rooms` conda env, installs `requirements.txt`, prints the manual `OLLAMA_WORKERS=http://<lan-ip>:<port>` value, warns if Ollama isn't reachable on the expected port, and then execs `advertise_ollama.py` (unless run with `--setup`, which only does setup).
- `scripts/setup_main.sh` was restructured: sprite-worker discovery/health-check logic (previously the whole script) is now conditional on `MERGED` being non-empty (prints "No sprite workers configured or discovered — skipping SPRITE_WORKERS." otherwise), and a new second section handles `OLLAMA_WORKERS` the same way — auto-discovering `_ollama-worker._tcp.local.` instances via `discover_ollama.py` (only when run with no arguments; manual single-worker mode now only adds a sprite worker and skips Ollama discovery), merging with any existing `OLLAMA_WORKERS`/`OLLAMA_THEMING_WORKERS` entries, health-checking each via `GET /` (Ollama's root endpoint, vs. sprite workers' `/health`), and writing only reachable URLs back to `OLLAMA_WORKERS` in `.env`. The script now resolves `PYTHON_BIN` (conda env `escape-rooms` if present, else `python3`) once up front and shares it across both sprite and Ollama discovery.
- `.env.example` documents the renamed `OLLAMA_WORKERS` setting (with `OLLAMA_THEMING_WORKERS` noted as a legacy alias), describes the new storyboard beats/flavor fan-out alongside the existing per-room theming fan-out, and points to `setup_ollama_worker.sh` (per worker Mac) + `setup_main.sh` with no arguments (auto-discover) as the setup flow.

### Why
Generalizes the existing LAN-distributed-inference mechanism (previously special-cased to per-room world theming) so the storyboard builder's two independent LLM passes (beats and flavor) can also run concurrently and be spread across additional Ollama instances on the LAN, following the same fan-out pattern already used for sprite generation (`SPRITE_WORKERS`) and theming.

### Files changed
- `.env.example` — renamed `OLLAMA_THEMING_WORKERS` documentation to `OLLAMA_WORKERS`, describing both the per-room theming and storyboard beats/flavor fan-out, noting the legacy alias, and pointing to the new setup scripts.
- `scripts/advertise_ollama.py` (new) — `_local_ip()` helper and `main(port)` async entry point; registers an `_ollama-worker._tcp.local.` `ServiceInfo` via `AsyncZeroconf` and keeps it advertised until interrupted.
- `scripts/discover_ollama.py` (new) — `main()`; browses `_ollama-worker._tcp.local.` via `ServiceBrowser` for a configurable timeout and prints `<ip>:<port>` per discovered instance.
- `scripts/setup_main.sh` — resolves `PYTHON_BIN` up front; gated sprite-worker health-check/`SPRITE_WORKERS` write behind `MERGED` being non-empty; added a new Ollama-worker section that reads/merges `OLLAMA_WORKERS`/`OLLAMA_THEMING_WORKERS`, optionally auto-discovers via `discover_ollama.py` (no-argument invocation only), health-checks each via `GET /`, and writes `OLLAMA_WORKERS` in `.env`.
- `scripts/setup_ollama_worker.sh` (new) — bootstraps the `escape-rooms` conda env, installs dependencies, prints the manual `OLLAMA_WORKERS` value derived from the LAN IP and `OLLAMA_HOST`/port, checks local Ollama reachability, and execs `advertise_ollama.py` (unless `--setup`).
- `src/escape_rooms/graphs/subgraphs/puzzle_graph.py` — `apply_theming` now builds its LLM list via `get_worker_llms("game_master", llm)` instead of `Settings().ollama_theming_workers` + manual `get_llm` calls; updated docstring/log message to reference `Settings.ollama_workers` and `len(llms)`.
- `src/escape_rooms/nodes/storyboard_builder.py` — `_call_json(system_prompt, user_prompt, llm=None)` gained an `llm` param (defaults to `get_llm("storyboard")`); `_run_beats_pass`/`_run_flavor_pass` gained an `llm` param threaded to `_call_json`; `storyboard_builder_node` now runs both passes concurrently via `ThreadPoolExecutor(max_workers=2)` against `get_worker_llms("storyboard")`.
- `src/escape_rooms/utils/settings.py` — `Settings.ollama_theming_workers` renamed to `ollama_workers`, now reads `OLLAMA_WORKERS` with `OLLAMA_THEMING_WORKERS` as a fallback env var; added `get_worker_llms(role, llm=None) -> list[ChatOllama]`; updated `get_llm`'s docstring reference.

### Key code
```python
# src/escape_rooms/utils/settings.py
def get_worker_llms(role: str, llm: ChatOllama | None = None) -> list[ChatOllama]:
    """Return `[llm (local)] + one ChatOllama per Settings.ollama_workers`."""
    if llm is None:
        llm = get_llm(role)
    worker_urls = Settings().ollama_workers
    return [llm] + [get_llm(role, base_url=url) for url in worker_urls]
```

```diff
# src/escape_rooms/nodes/storyboard_builder.py — storyboard_builder_node
-    beats = _run_beats_pass(world_data, is_mystery, case_facts)
+    from concurrent.futures import ThreadPoolExecutor
+    from src.escape_rooms.utils.settings import get_worker_llms
+
+    llms = get_worker_llms("storyboard")
+    with ThreadPoolExecutor(max_workers=2) as pool:
+        beats_future = pool.submit(_run_beats_pass, world_data, is_mystery, case_facts, llms[0])
+        flavor_future = pool.submit(_run_flavor_pass, world_data, is_mystery, case_facts, llms[1 % len(llms)])
+        beats = beats_future.result()
+        flavor = flavor_future.result()
```

```diff
# src/escape_rooms/graphs/subgraphs/puzzle_graph.py — apply_theming
-        worker_urls = Settings().ollama_theming_workers
-        llms = [llm] + [get_llm("game_master", base_url=url) for url in worker_urls]
-        if worker_urls:
+        llms = get_worker_llms("game_master", llm)
+        if len(llms) > 1:
             log.info(
                 "apply_theming: distributing {} room(s) across {} Ollama instance(s) (1 local + {} remote)",
-                len(world.rooms), len(llms), len(worker_urls),
+                len(world.rooms), len(llms), len(llms) - 1,
             )
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-12 14:01:19 WIB

### What changed
- Mystery worlds now require a final **deduction step**: once the storyboard's proof object (`storyboard.mystery.proof_object_id`) has been examined or taken, `PartyState.proof_found` flips to `True` and a new `accuse <suspect_token>` action becomes available in the action space (`_build_action_space` in `nodes/gameplay.py`), built from `storyboard.suspects` via a new `_suspect_token(name)` helper (e.g. "Marcus Webb" -> `marcus_webb`).
- Added `_resolve_accuse(token, storyboard, ps)` in `nodes/gameplay.py`: it looks up the suspect by token, and if `storyboard.solution.matches(name)` is true sets `ps.accusation = name` (a correct guess); otherwise increments `ps.deduction_attempts` and, once `MAX_DEDUCTION_ATTEMPTS` (3) guesses are wrong, sets `ps.wrong_deduction = True` to end the game. Wrong guesses before the limit return a remaining-attempts message plus a hint from `storyboard.solution.hint_1`/`hint_2` (first vs. subsequent wrong guesses).
- `PartyState` gained four new fields: `proof_found: bool`, `deduction_attempts: int`, `wrong_deduction: bool`, and `accusation: str` (the most recently accused name).
- `_check_victory(world, ps, storyboard=None)` now takes an optional `storyboard` argument: for mystery themes (non-empty `storyboard.solution`) the win condition's object-state check alone is no longer sufficient — `_check_victory` also requires `bool(ps.accusation)` (i.e. a correct `accuse` has been made). Non-mystery worlds behave exactly as before.
- `_resolve_examine` and `_resolve_take` now take an optional `storyboard` param and call a new `_mark_proof_found(obj, ps, storyboard)` helper, which sets `ps.proof_found = True` the first time the proof object is examined (with or without new info) or picked up. `_resolve_action` threads `storyboard` through to these and to the new `accuse` verb branch.
- `HeadlessEpisode` and `MultiAgentEpisode` (in `benchmark/engine.py`) both gained an optional `storyboard: Storyboard | None` constructor param, pass it to `_check_victory`/`_build_action_space`/`_resolve_action`, and break out of their run loop early if `ps.wrong_deduction` becomes true (in addition to the existing victory break). Both `EpisodeResult` and `CooperativeEpisodeResult` gained a `wrong_deduction: bool = False` field copied from `ps.wrong_deduction` at the end of the run.
- `cognitive_solver_policy`, `cognitive_solver_policy_multi`, `solve_world_multi`, and `solve_world_cooperative` (in `agents/multi_solver.py`) all gained an optional `storyboard: Storyboard | None = None` param, threaded down to `HeadlessEpisode`/`MultiAgentEpisode` and into a new `_accusation_guidance(action_space, storyboard)` call appended to each tick's prompt. `"accuse"` was added to `_PROGRESS_GATE_EXEMPT_VERBS` so the planner's dominance gate doesn't override an LLM-chosen `accuse` action in favor of a higher-scored exploration move.
- `_accusation_guidance(action_space, storyboard)` (new, in `nodes/gameplay.py`) returns "" unless an `accuse <...>` action is present in `action_space`; otherwise it renders a block: "The proof has been found — you may now accuse a suspect:", `storyboard.suspects_context()`, an optional `Motive hint: {storyboard.mystery.motive_hint}` line, and a closing caution that wrong guesses cost an attempt. This is surfaced to the live gameplay agent via a new `accusation_guidance` format field in `_agent_act` (new `storyboard` param) and a new `{accusation_guidance}` placeholder in `prompts/gameplay_agent/action.txt`, which also documents the new `"accuse <suspect_token>"` action grammar.
- `solver_node` now passes `storyboard=state.storyboard` into `solve_world_cooperative`, and `SolverResult`/`SolverLog` (in `state/schema.py` and `api/routers/generate.py`) both gained `wrong_deduction: bool = False`, populated from `result.wrong_deduction`/`solver_result.wrong_deduction`.
- `gameplay_node`'s end-of-tick check gained a new `elif ps.wrong_deduction` branch (between the victory check and the time-up check): it sets `ps.game_over = True`, prints a `"WRONG DEDUCTION"` banner, streams `"Party exhausted all {MAX_DEDUCTION_ATTEMPTS} accusations without naming the killer."`, renders the final state, and appends a `"[gameplay] WRONG DEDUCTION at tick {ps.tick}"` message.
- In `api/routers/generate.py`, `_run_pipeline` and `_run_solve_pipeline`'s `on_tick` closures now call `narrator.reveal_proof()` as soon as `ps.proof_found` is true (so subsequent narration can reference the proof/mystery solution), and both pass `wrong_deduction=solver_result.wrong_deduction` into `narrator.narrate_ending(...)` (using the existing `storyboard.ending_text(won, wrong_deduction=...)` machinery) so the ending narration distinguishes a true loss from a "ran out of accusations" loss. `_run_pipeline` now copies `state` with both `world` and `storyboard` (previously only `world`) before solving, and `_run_solve_pipeline` builds `GameState(..., storyboard=storyboard)` directly instead of constructing a plain `GameState` and assigning the narrator's storyboard separately.

### Why
Completes the mystery gameplay loop: previously a mystery world could be "won" purely by escaping (satisfying the win condition's object state) without ever identifying the killer, even though the storyboard already carried a sealed `solution`/`suspects` answer key. This adds the missing deduction gate — find the proof, then `accuse` the correct suspect — with a bounded number of wrong guesses (with hints) before the run ends in a "wrong deduction" loss distinct from a generic failure, and surfaces that distinction through to the live narration and solver logs.

### Files changed
- `api/routers/generate.py` — `SolverLog` gained `wrong_deduction: bool = False`; `_run_pipeline`'s `on_tick` calls `narrator.reveal_proof()` when `ps.proof_found`, copies `state` with `world` and `storyboard`, and passes `wrong_deduction=solver_result.wrong_deduction` to `narrator.narrate_ending(...)` and the constructed `SolverLog`; `_run_solve_pipeline` now builds `GameState(theme=..., world=world, storyboard=storyboard)` directly (removing a separate earlier `GameState` construction), and its `on_tick`/`narrate_ending` call get the same `reveal_proof()`/`wrong_deduction` treatment.
- `benchmark/engine.py` — imports `Storyboard`; `EpisodeResult` and `CooperativeEpisodeResult` gained `wrong_deduction: bool = False`; `HeadlessEpisode.__init__`/`MultiAgentEpisode.__init__` gained `storyboard: Storyboard | None = None`; both `.run()` methods pass `storyboard` to `_check_victory`/`_build_action_space`/`_resolve_action`, break early on `ps.wrong_deduction`, and return `wrong_deduction=ps.wrong_deduction`.
- `src/escape_rooms/agents/multi_solver.py` — imports `_accusation_guidance` and `Storyboard`; added `"accuse"` to `_PROGRESS_GATE_EXEMPT_VERBS`; `cognitive_solver_policy`, `cognitive_solver_policy_multi`, `solve_world_multi`, `solve_world_cooperative` all gained `storyboard: Storyboard | None = None` and thread it to `HeadlessEpisode`/`MultiAgentEpisode` and into each tick's prompt via `_accusation_guidance(action_space, storyboard)`.
- `src/escape_rooms/nodes/gameplay.py` — imports `Storyboard`; added `_suspect_token(name)`, `_mark_proof_found(obj, ps, storyboard)`, `MAX_DEDUCTION_ATTEMPTS = 3`, `_resolve_accuse(token, storyboard, ps)`, `_accusation_guidance(action_space, storyboard)`; `_build_action_space`, `_resolve_examine`, `_resolve_take`, `_resolve_action`, `_agent_act`, `_check_victory` all gained an optional `storyboard` param and use it as described above; `gameplay_node` reads `state.storyboard`, threads it through all the above calls, and adds the `elif ps.wrong_deduction` end-of-tick branch.
- `src/escape_rooms/nodes/solver.py` — `solver_node` passes `storyboard=state.storyboard` to `solve_world_cooperative` and copies `result.wrong_deduction` into the returned `SolverResult`.
- `src/escape_rooms/prompts/gameplay_agent/action.txt` — added `{accusation_guidance}` placeholder line and documented the `"accuse <suspect_token>"` action in the action grammar list.
- `src/escape_rooms/state/schema.py` — `PartyState` gained `proof_found`, `deduction_attempts`, `wrong_deduction`, `accusation` fields; `SolverResult` gained `wrong_deduction: bool = False`.

### Key code
```python
# src/escape_rooms/nodes/gameplay.py
MAX_DEDUCTION_ATTEMPTS = 3


def _resolve_accuse(token: str, storyboard: Storyboard | None, ps: PartyState) -> str:
    """Resolve an 'accuse <suspect_token>' action against the sealed solution.

    A correct guess sets ps.accusation to the matched name (checked by
    _check_victory). A wrong guess consumes one of MAX_DEDUCTION_ATTEMPTS;
    exhausting all attempts sets ps.wrong_deduction and ends the game.
    """
    ...
```

```diff
# src/escape_rooms/nodes/gameplay.py — _check_victory
-def _check_victory(world: GameWorld, ps: PartyState) -> bool:
+def _check_victory(world: GameWorld, ps: PartyState, storyboard: Storyboard | None = None) -> bool:
     win = world.win_condition
     if not win.object_id:
         return False
-    return _state_satisfies(ps.object_states.get(win.object_id), win.state)
+    if not _state_satisfies(ps.object_states.get(win.object_id), win.state):
+        return False
+    if storyboard is not None and not storyboard.solution.is_empty():
+        return bool(ps.accusation)
+    return True
```

```diff
# src/escape_rooms/nodes/gameplay.py — gameplay_node end-of-tick
     if _check_victory(world, ps, storyboard):
         ps.victory = True
         ...
+    elif ps.wrong_deduction:
+        ps.game_over = True
+        _banner("WRONG DEDUCTION", char="*")
+        _stream(f"  Party exhausted all {MAX_DEDUCTION_ATTEMPTS} accusations without naming the killer.")
+        _render_final(ps, world)
+        new_messages.append(AIMessage(content=f"[gameplay] WRONG DEDUCTION at tick {ps.tick}"))
     elif ps.tick >= MAX_TICKS:
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-12 13:15:24 WIB

### What changed
- `storyboard_builder_node` now generates the storyboard in up to three focused LLM passes instead of one large JSON document, each with its own system/generation prompt pair: pass 1 `_run_core_pass` (mystery themes only) produces `mystery`/`solution`/`suspects` via `system_core.txt`/`generation_core.txt`; pass 2 `_run_beats_pass` produces `discovery_beats`/`conversation_seeds`/`ending_guidance` via `system_beats.txt`/`generation_beats.txt`; pass 3 `_run_flavor_pass` produces `plot`/`adapted_personas`/`room_stories`/`phase_guidance` via `system_flavor.txt`/`generation_flavor.txt`. A shared `_call_json(system_prompt, user_prompt)` helper makes the LLM call and parses the JSON response, returning `None` on any invoke or parse failure (logged as a warning).
- Passes 2 and 3 are grounded in pass 1's decisions via a new immutable `CASE_FACTS` block. `_case_facts(data)` extracts `victim`, `killer_name`, `proof_object_id`, `motive_hint`, and a `suspects` list (`name`/`is_killer`) from the core pass output (falling back to `solution.answer`/`solution.proof_object` if the mystery block is missing those fields). `_case_facts_section(case_facts)` renders this as a `"CASE_FACTS (immutable — use these exact names):\n<json>\n\n"` string (empty if there's no `killer_name`), which is interpolated into the beats and flavor prompts via a new `case_facts_section` format field so later passes can't contradict the case decided in pass 1.
- `storyboard_builder_node` merges each pass's output into `data` only for that pass's expected top-level keys, and degrades per-pass: a failed pass leaves its keys absent (relying on existing `_repair_mystery`/`_ensure_personas` backfill) and logs a warning, rather than failing the whole node. Only if ALL passes fail does it return an empty `Storyboard(world_id=world_id)` with a `"=== STORYBOARD (all passes failed) ==="` message; otherwise the returned `AIMessage` content is `"=== STORYBOARD ===\n\n"` joined with each pass's `"--- <name> pass ---\n<json>"` transcript.
- Adapted personas now include `sample_lines`: each `StoryboardPersona` (in `state/schema.py`) gained a `sample_lines: list[str]` field (default `[]`), populated by the new flavor-pass prompt's `adapted_personas.<character>.sample_lines` (exactly 2 full dialogue lines, 10-18 words each, illustrating the character's register). `_ensure_personas` backfills two default sample lines (`"Scratches around the keyhole..."` / `"The frame is intact but the hinge is bent..."`) when missing, and `_build_storyboard` copies `sample_lines` (filtered to strings) from the raw persona dict into the `StoryboardPersona`.
- The narrator's dialogue prompt now incorporates persona voice/vocabulary/register. `build_dialogue_context_package` in `agents/narrator.py` gained three new optional parameters — `persona_voice: str = ""`, `persona_vocabulary: list[str] | None = None`, `persona_sample_lines: list[str] | None = None` — which render as new `- VOICE: ...`, `- VOCABULARY (work these phrasings in naturally, don't force all of them): ...`, and `- REGISTER EXAMPLES (match this rhythm and attitude, do not copy verbatim): "..."` lines inserted after the existing `- MOOD: ...` line. `GameMasterNarrator` now passes `persona.voice`, `persona.vocabulary`, and `persona.sample_lines` (each `if persona else ""`/`None`) into this call.

### Why
A local model drops random fields when asked to generate the full storyboard JSON in one shot (attention degrades as output grows). Splitting generation into three smaller, focused passes keeps each pass's output small enough that critical fields (victim, killer, proof object) can't get lost, and feeding pass 1's decisions to later passes as immutable `CASE_FACTS` prevents the beats/flavor passes from inventing a different killer or suspect set. The new `sample_lines` persona field gives the dialogue model concrete few-shot register examples, which work better than abstract style adjectives for steering a local model's voice.

### Files changed
- `src/escape_rooms/nodes/storyboard_builder.py` — replaced single `SYSTEM_PROMPT`/`GENERATION_PROMPT`/`_build_user_prompt` with `CORE_SYSTEM_PROMPT`/`CORE_GENERATION_PROMPT`, `BEATS_SYSTEM_PROMPT`/`BEATS_GENERATION_PROMPT`, `FLAVOR_SYSTEM_PROMPT`/`FLAVOR_GENERATION_PROMPT`; added `_call_json`, `_run_core_pass`, `_case_facts`, `_case_facts_section`, `_run_beats_pass`, `_run_flavor_pass`; `_ensure_personas` now backfills `sample_lines`; `_build_storyboard` copies `sample_lines` into `StoryboardPersona`; `storyboard_builder_node` rewritten to run the three passes sequentially and merge/transcript their results.
- `src/escape_rooms/prompts/storyboard_builder/generation.txt` — deleted (replaced by the three per-pass generation prompts below).
- `src/escape_rooms/prompts/storyboard_builder/system.txt` — deleted (replaced by the three per-pass system prompts below).
- `src/escape_rooms/prompts/storyboard_builder/generation_core.txt` — new prompt for pass 1: outputs `mystery`/`solution`/`suspects` only, with victim/killer-name-first rules and suspect/red-herring guidance.
- `src/escape_rooms/prompts/storyboard_builder/system_core.txt` — new system prompt for pass 1 ("STORY ARCHITECT" persona deciding victim/killer/proof/suspects).
- `src/escape_rooms/prompts/storyboard_builder/generation_beats.txt` — new prompt for pass 2: outputs `discovery_beats`/`conversation_seeds`/`ending_guidance`, referencing `{case_facts_section}` and `CASE_FACTS.suspects`/`CASE_FACTS.killer_name`/`CASE_FACTS.proof_object_id` for naming suspects and the proof beat.
- `src/escape_rooms/prompts/storyboard_builder/system_beats.txt` — new system prompt for pass 2 ("CLUE WRITER" persona).
- `src/escape_rooms/prompts/storyboard_builder/generation_flavor.txt` — new prompt for pass 3: outputs `plot`/`adapted_personas`/`room_stories`/`phase_guidance`, with new `adapted_personas.<character>.sample_lines` field and SAMPLE_LINES RULES (2 lines, 10-18 words, register contrast across characters).
- `src/escape_rooms/prompts/storyboard_builder/system_flavor.txt` — new system prompt for pass 3 ("NARRATIVE DESIGNER" persona).
- `src/escape_rooms/state/schema.py` — `StoryboardPersona` gained `sample_lines: list[str] = Field(default_factory=list)`.
- `src/escape_rooms/agents/narrator.py` — `build_dialogue_context_package` gained `persona_voice`, `persona_vocabulary`, `persona_sample_lines` params and corresponding `VOICE`/`VOCABULARY`/`REGISTER EXAMPLES` prompt lines; `GameMasterNarrator` passes `persona.voice`/`persona.vocabulary`/`persona.sample_lines` through.
- `.DS_Store` — binary, no behavioral change.

### Key code
```python
# src/escape_rooms/nodes/storyboard_builder.py
def _run_core_pass(world_data: dict) -> dict | None:
    """Pass 1 — mystery/solution/suspects. Small output: victim/killer cannot be dropped."""
    user_prompt = CORE_GENERATION_PROMPT.format(world_data_json=json.dumps(world_data, indent=2))
    return _call_json(CORE_SYSTEM_PROMPT, user_prompt)


def _case_facts_section(case_facts: dict | None) -> str:
    if not case_facts or not case_facts.get("killer_name"):
        return ""
    return "CASE_FACTS (immutable — use these exact names):\n" + json.dumps(case_facts, indent=2) + "\n\n"
```

```diff
# src/escape_rooms/agents/narrator.py — build_dialogue_context_package
     recent_story: list[str],
     lore_excerpt: str = "",
     adapted_world_role: str = "",
+    persona_voice: str = "",
+    persona_vocabulary: list[str] | None = None,
+    persona_sample_lines: list[str] | None = None,
     conversation_seed: str = "",
```

```diff
# src/escape_rooms/state/schema.py — StoryboardPersona
     world_role: str = ""
     voice: str = ""
     vocabulary: list[str] = Field(default_factory=list)
+    sample_lines: list[str] = Field(default_factory=list)
```

### Verification
Not verified in conversation.

---

## 2026-06-12 10:10:07 WIB

### What changed
- The `/generate` API endpoint now produces live narration alongside the solver run. When `solve` is requested, `_run_pipeline` instantiates a `GameMasterNarrator(storyboard=storyboard or Storyboard())` and emits an `{"type": "narration", "stage": "opening", "text": ...}` event up front via `narrator.narrate_opening(scenario=world.scenario or req.theme, objective=world.objective, room_ids=[room.id for room in world.rooms])`. After the solve completes, it emits a matching `{"type": "narration", "stage": "ending", "text": ...}` event from `narrator.narrate_ending(objective=world.objective, won=solver_result.won)`. The final `GenerateResponse` gained two new fields, `narration_opening: str | None` and `narration_ending: str | None`, carrying these same strings.
- Each solver tick emitted from `/generate` and `/generate/solve` now includes a `"narration"` string alongside the existing `render`/record fields, produced by a new shared helper `_narrate_tick(narrator, *, record, ps, scenario, objective, total_puzzles)`. It derives an `actor_name` from `record["agent_id"]` (e.g. `"agent_1"` -> `"Agent 1"`), pulls the action/outcome/success from `record["final_action"]`/`record["prev_outcome"]` (defaulting success to `True` and the outcome note to `"Nothing notable happens."` when absent), and calls `narrator.narrate_turn(...)` with `actor_role="escape room agent"`, a fixed `actor_backstory`, `looped=bool(record.get("gates_fired"))`, `solved_count=len(ps.known_info)`, and `total_puzzles` (the count of rooms with a non-`None` `goal_completion`).
- `/generate/solve`'s `SolveRequest` gained an optional `storyboard: dict | None` field (the `storyboard` field from a prior `/generate` response). `_run_solve_pipeline` builds a `Storyboard.model_validate(req.storyboard)` if provided (else an empty `Storyboard()`), constructs its own `GameMasterNarrator`, and emits the same opening/per-tick/ending narration events; `SolveResponse` gained matching `narration_opening`/`narration_ending` fields. When no storyboard is supplied, `world.scenario or "mystery"` is used as the narrator's scenario.

### Why
Wires the previously standalone, importable-but-unused `GameMasterNarrator` (added in a prior session) into the live `/generate` and `/generate/solve` streams so API clients receive game-master narration text — opening scene-setting, per-tick in-character commentary, and an ending beat — alongside the existing render/solver data, using the storyboard's plot/personas/discovery beats when available.

### Files changed
- `api/routers/generate.py` — imports `GameMasterNarrator`; added module-level `_narrate_tick(narrator, *, record, ps, scenario, objective, total_puzzles)` helper; `GenerateResponse` gained `narration_opening`/`narration_ending`; `_run_pipeline` constructs a `GameMasterNarrator`, emits opening/ending `"narration"` events, and adds `"narration"` to each `"tick"` event via `_narrate_tick`; `SolveRequest` gained `storyboard: dict | None`; `SolveResponse` gained `narration_opening`/`narration_ending`; `_run_solve_pipeline` builds a `Storyboard` from `req.storyboard` (or an empty one), constructs its own `GameMasterNarrator`, and emits the same opening/per-tick/ending narration events.

### Key code
```python
# api/routers/generate.py
def _narrate_tick(
    narrator: GameMasterNarrator,
    *,
    record: dict,
    ps: PartyState,
    scenario: str,
    objective: str,
    total_puzzles: int,
) -> str:
    """Turn one solver tick record into a single narration line."""
    agent_id = record.get("agent_id") or "agent_1"
    actor_name = agent_id.replace("_", " ").title()
    ...
    return narrator.narrate_turn(
        scenario=scenario,
        objective=objective,
        turn=record.get("tick", ps.tick),
        actor_name=actor_name,
        actor_role="escape room agent",
        ...
        looped=bool(record.get("gates_fired")),
        solved_count=len(ps.known_info),
        total_puzzles=total_puzzles,
    )
```

```diff
# api/routers/generate.py — GenerateResponse / SolveResponse
     storyboard: dict | None = None
+    narration_opening: str | None = None
+    narration_ending: str | None = None
```

```diff
# api/routers/generate.py — _run_pipeline
+        narrator = GameMasterNarrator(storyboard=storyboard or Storyboard())
+        opening_narration = narrator.narrate_opening(
+            scenario=world.scenario or req.theme,
+            objective=world.objective,
+            room_ids=[room.id for room in world.rooms],
+        )
+        emit.put({"type": "narration", "stage": "opening", "text": opening_narration})
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-12 09:21:11 WIB

### What changed
- The `/generate` API endpoint now runs `storyboard_builder_node` as part of `_run_pipeline`, after `puzzle_builder` and before sprite generation/solving. It emits a `{"type": "progress", "stage": "story", "message": "Writing the story and characters..."}` event, then calls `storyboard_builder_node(state)` (with `state` updated via `state.model_copy(update={"world": world})` so the node sees the assembled `GameWorld`) and extracts `sb_update.get("storyboard")` as a `Storyboard | None`.
- `GenerateResponse` gained a new `storyboard: dict | None = None` field. The final response now includes `storyboard.model_dump(mode="json", exclude_none=True)` when a storyboard was produced, or `None` if the node returned nothing (e.g. non-mystery degrade or world is empty) — so API clients now receive the narrative layer (plot, personas, room stories, discovery beats, mystery/solution for mystery themes) alongside the generated world.
- `./scripts/setup_main.sh`'s worker health check now runs *before* `SPRITE_WORKERS` is written to `.env`, and only workers that pass `/health` are written — unreachable workers are dropped from the saved list (previously all merged URLs, healthy or not, were written, and the script only reported failures afterward and exited non-zero if any worker was down). Reachable workers are collected into `HEALTHY` and written as `HEALTHY_CSV`; unreachable ones go into `UNHEALTHY` and are reported to stderr with a hint to start `./scripts/setup_worker.sh` on each. The script no longer exits 1 when some/all workers are unreachable — if zero workers are healthy it now prints "No workers reachable — sprite generation will run locally only." and exits 0, since local generation remains a valid fallback.

### Why
Surfaces the previously-generated-but-unused storyboard (added in the prior commit) through the public `/generate` API so clients can render narrative content alongside the world. The setup script change ensures `SPRITE_WORKERS` never contains stale/dead worker URLs that would otherwise need manual cleanup, and treats an unreachable worker as a degrade-to-local rather than a hard setup failure.

### Files changed
- `api/routers/generate.py` — imports `storyboard_builder_node` and `Storyboard`; `GenerateResponse` gained `storyboard: dict | None = None`; `_run_pipeline` calls `state.model_copy(update={"world": world})`, emits a `"story"` progress event, runs `storyboard_builder_node(state)`, and includes the serialized storyboard in the returned `GenerateResponse`.
- `scripts/setup_main.sh` — reordered the script so the `/health` check over `MERGED` runs first, splitting results into `HEALTHY`/`UNHEALTHY` arrays; `SPRITE_WORKERS` is now written from `HEALTHY_CSV` instead of `MERGED_CSV`; unreachable workers are reported to stderr with a re-add hint; removed the `exit 1` on any/all workers unreachable, replaced with a "running locally only" message when `HEALTHY` is empty.

### Key code
```diff
# api/routers/generate.py
     if world is None:
         raise RuntimeError("puzzle_builder produced no world")

+    state = state.model_copy(update={"world": world})
+
+    emit.put({"type": "progress", "stage": "story", "message": "Writing the story and characters..."})
+    sb_update = storyboard_builder_node(state)
+    storyboard: Storyboard | None = sb_update.get("storyboard")
+
     total_objects = len(world.objects)
```

```diff
# api/routers/generate.py — GenerateResponse
     solver: SolverLog | None = None
     sprites: dict[str, str] = {}  # object_id → base64 PNG
+    storyboard: dict | None = None
```

```bash
# scripts/setup_main.sh
declare -a HEALTHY=()
declare -a UNHEALTHY=()
for url in "${MERGED[@]+"${MERGED[@]}"}"; do
    if curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1; then
        HEALTHY+=("${url}")
    else
        UNHEALTHY+=("${url}")
    fi
done
HEALTHY_CSV="$(IFS=,; echo "${HEALTHY[*]+"${HEALTHY[*]}"}")"
# ... SPRITE_WORKERS=${HEALTHY_CSV} written to .env ...
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-12 09:13:28 WIB

### What changed
- A new `storyboard_builder` pipeline stage runs after `puzzle_builder` and before the solver: `world_builder -> puzzle_builder -> storyboard_builder -> (solver?) -> END`. It takes the fully assembled `GameWorld` and generates a `Storyboard` — the narrative layer (plot, per-character `adapted_personas`, `room_stories`, `discovery_beats`, `phase_guidance`, `conversation_seeds`, `ending_guidance`) plus, only when the theme contains "mystery", a `mystery`/`solution`/`suspects` set (victim, killer, proof object, hints, answer aliases for deduction matching).
- Added new Pydantic models in `state/schema.py`: `Storyboard`, `StoryboardMystery`, `StoryboardSolution`, and `StoryboardPersona`, plus a module-level `_normalize_answer(text)` helper (lowercase, strip punctuation/whitespace) used by `StoryboardSolution.matches(human_answer)` to check a player's deduction guess against the canonical answer or any alias. `GameState` gained a new `storyboard: Storyboard | None = None` field.
- `Storyboard` exposes helper methods consumed by gameplay/narration: `lore_excerpt(actor_name=, room_id=, object_id=)`, `persona_for(name)`, `discovery_beat(object_id)`, `room_story(room_id)`, `phase_for_turn(turn, total_turns=40)` / `phase_guidance_for_turn(...)`, `ending_text(won, wrong_deduction=False)`, `pop_seed(actor_name, used)`, and `suspects_context()` (formats suspects without revealing who is guilty).
- Added `storyboard_builder_node(state)` in `nodes/storyboard_builder.py`. It classifies plot-critical objects via `_classify_objects(world, proof_object_id="")` (clue types: `story_clue`, `key`, `code`, `lock`, `other`, based on `requires_tool`/`requires_code`/`contains_info`), builds a compact world summary via `_build_world_data`/`_build_user_prompt`, calls the LLM, and parses the JSON response with `_parse_json`. For mystery themes, `_repair_mystery(data, world)` patches common LLM omissions (missing killer/proof/hints/suspect fields, recovers a mis-specified `proof_object_id` by scanning objects/discovery beats); for non-mystery themes `_strip_mystery_sections(data)` forces `mystery`/`solution`/`suspects` empty. `_ensure_personas`/`_sanitize_personas` backfill default personas per character and strip vocabulary containing tech/mechanical "contamination words" (e.g. "circuit", "mainframe", "override"). On any parse failure or missing world, the node degrades to returning an empty `Storyboard` (or `{}`) rather than failing the pipeline.
- Added a standalone `GameMasterNarrator` dataclass in `agents/narrator.py` (ported from the "Escapee" project), a presentation-only LLM narrator with a rolling memory (`recent_window=6`) of recent story beats. Methods: `narrate_opening`, `narrate_turn` (builds a `CONTEXT_PACKAGE` dialogue prompt via `build_dialogue_context_package`, including a `TENSION_LEVEL` from `_tension_level(turn, solved_count, total_puzzles)` and mystery/revelation-mode gating via `reveal_proof()`/`_proof_revealed`), `narrate_system_event` (loop_avoided/planner_override/critical_stuck/milestone), `narrate_room_entry`, `narrate_discovery` (uses a pre-written `storyboard.discovery_beat` if present, else falls back to LLM), `narrate_milestone`, and `narrate_ending` (uses `storyboard.ending_text(won, wrong_deduction=)`). All LLM calls go through `_say()`, which degrades to a deterministic fallback string on any exception. Not yet wired into `gameplay.py`'s tick loop — importable/testable standalone for now.
- Added new prompt files: `prompts/storyboard_builder/system.txt` and `prompts/storyboard_builder/generation.txt` (189 lines, the storyboard generation template consumed by `storyboard_builder_node`), and `prompts/narrator/system.txt` / `prompts/narrator/event_system.txt` (dialogue vs. atmospheric-event system prompts for `GameMasterNarrator`).
- Added two new settings roles in `utils/settings.py`: `storyboard` (`STORYBOARD_MODEL`, falling back to `BUILDER_MODEL`; `STORYBOARD_TEMPERATURE` default `0.85` for creative variety, since storyboard generation runs once per world) and `narrator` (`NARRATOR_MODEL`, falling back to `PLAYER_MODEL` then `BUILDER_MODEL`; `NARRATOR_TEMPERATURE` default `0.8`).
- Wired `storyboard_builder` into `main.py`: added to `NODE_NAMES` and `RUNNABLE_NODE_NAMES` (registered in `_node_registry()` with upstream dependency `("world",)`), run after `puzzle_builder` in both `run()` and `_run_once_captured()` with its own timing entry and optional `--log` node dump, and `_write_run_summary` now prints a `STORYBOARD` section (plot victim/threat/stakes, killer/proof object for mysteries, suspects with apparent motives, per-room stories, and discovery beats).
- `graphs/main_graph.py` adds a `storyboard_builder` node between `puzzle_builder` and the solver-routing conditional edge (`_route_solver`), and `nodes/__init__.py` / `state/__init__.py` export the new node function and schema classes.

### Why
Splits narrative generation out as its own pipeline stage so the puzzle's mechanical structure (objects, locks, solution path) and its human-facing story (plot, character voices, room atmosphere, and — for mystery themes — the killer/suspects/deduction answer key) are produced and validated independently, with the storyboard degrading gracefully rather than failing generation if the LLM call or JSON parse fails.

### Files changed
- `main.py` — added `storyboard_builder` to `NODE_NAMES`/`RUNNABLE_NODE_NAMES`; `_write_run_summary` gained a `STORYBOARD` section; `run()` and `_run_once_captured()` invoke `storyboard_builder_node` after `puzzle_builder` with timing/log support; `_node_registry()` registers `"storyboard_builder": (storyboard_builder_node, ("world",))`.
- `src/escape_rooms/graphs/main_graph.py` — `build_graph()` adds a `storyboard_builder` node and edge `puzzle_builder -> storyboard_builder -> _route_solver`; updated module/function docstrings.
- `src/escape_rooms/nodes/__init__.py` — exports `storyboard_builder_node`.
- `src/escape_rooms/nodes/storyboard_builder.py` (new) — `storyboard_builder_node(state)` plus helpers `_parse_json`, `_s`, `_classify_objects`, `_is_plot_critical`, `_build_world_data`, `_build_user_prompt`, `_repair_mystery`, `_strip_mystery_sections`, `_ensure_personas`, `_sanitize_personas`, `_build_storyboard`.
- `src/escape_rooms/state/__init__.py` — exports `Storyboard`, `StoryboardMystery`, `StoryboardPersona`, `StoryboardSolution`.
- `src/escape_rooms/state/schema.py` — added `_normalize_answer`, `StoryboardMystery`, `StoryboardSolution` (with `matches()`), `StoryboardPersona`, `Storyboard` (with `lore_excerpt`, `persona_for`, `discovery_beat`, `room_story`, `phase_for_turn`, `phase_guidance_for_turn`, `ending_text`, `pop_seed`, `suspects_context`, `is_empty`); `GameState` gained `storyboard: Storyboard | None = None`.
- `src/escape_rooms/utils/settings.py` — added `storyboard_model`/`storyboard_temperature` and `narrator_model`/`narrator_temperature` fields and corresponding entries in `_ROLE_CONFIG`.
- `src/escape_rooms/agents/narrator.py` (new) — `GameMasterNarrator` dataclass and prompt builders `build_dialogue_context_package`, `build_opening_user_prompt`, `build_room_entry_prompt`, `build_discovery_prompt`, `build_system_event_prompt`, `build_ending_user_prompt`, plus helpers `humanize_text`, `_tension_level`, `_mood_for`.
- `src/escape_rooms/prompts/storyboard_builder/system.txt`, `src/escape_rooms/prompts/storyboard_builder/generation.txt` (new) — storyboard generation prompts.
- `src/escape_rooms/prompts/narrator/system.txt`, `src/escape_rooms/prompts/narrator/event_system.txt` (new) — narrator dialogue and atmospheric-event system prompts.
- `graph.png`, `.DS_Store` — regenerated/binary artifacts; non-behavioral.

### Key code
```python
# src/escape_rooms/state/schema.py
class StoryboardSolution(BaseModel):
    ...
    def matches(self, human_answer: str) -> bool:
        """True if the human's answer matches the canonical answer or any alias."""
        normalized = _normalize_answer(human_answer)
        ...
```

```python
# src/escape_rooms/graphs/main_graph.py
builder.add_edge("world_builder", "puzzle_builder")
builder.add_edge("puzzle_builder", "storyboard_builder")
builder.add_conditional_edges("storyboard_builder", _route_solver, ["solver", END])
builder.add_edge("solver", END)
```

```python
# src/escape_rooms/nodes/storyboard_builder.py
def storyboard_builder_node(state: GameState) -> dict:
    """Generate the narrative storyboard for the assembled world."""
    world = state.world
    if not world or not world.rooms:
        return {}
    is_mystery = "mystery" in state.theme.lower()
    ...
    if not isinstance(data, dict):
        return {"storyboard": Storyboard(world_id=world_id), "messages": [...]}
    if is_mystery:
        _repair_mystery(data, world)
    else:
        _strip_mystery_sections(data)
    ...
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-12 08:03:27 WIB

### What changed
- `./scripts/setup_main.sh` now supports configuring **multiple** sprite-generation workers instead of just one. With no arguments, it auto-discovers every `_sprite-worker._tcp.local.` instance currently advertising on the LAN (via `scripts/discover_worker.py`, which already prints one `<ip>:<port>` per line) and adds all of them. With a `<worker-hostname-or-ip> [port]` argument, it appends that single worker to the existing list rather than overwriting it.
- The script now reads any existing `SPRITE_WORKERS=` value from `.env`, merges (deduplicating, preserving order) newly discovered/specified worker URLs with the existing ones, and writes the merged comma-separated list back to `SPRITE_WORKERS`.
- After writing `.env`, the script checks `/health` on **every** configured worker URL (not just one), printing a per-worker `OK`/`FAIL` line, and reports overall success only if all workers are reachable (exits 1 with a hint to run `./scripts/setup_worker.sh` if any fail).
- Updated `.env.example`, `scripts/setup_main.sh`, `scripts/setup_worker.sh`, and `sprite_worker.py` docs/comments to describe `SPRITE_WORKERS` as a comma-separated list of one or more worker URLs (e.g. `SPRITE_WORKERS=http://192.168.1.50:8001,http://192.168.1.51:8001`), and to say workers can be run "on as many machines as you have" rather than assuming exactly one second Mac.

### Why
Not specified in conversation — the diff generalizes the sprite-worker setup from a fixed "one main + one worker" pairing to support an arbitrary number of LAN worker machines.

### Files changed
- `.env.example` — updated `SPRITE_WORKERS` comment block to describe a comma-separated multi-worker list and changed the example value to `SPRITE_WORKERS=http://192.168.1.50:8001,http://192.168.1.51:8001`.
- `scripts/setup_main.sh` — rewrote the script to read existing `SPRITE_WORKERS` from `.env` into `URLS`, collect `NEW_URLS` from either the CLI arg or all lines of `discover_worker.py`'s output, merge+dedupe into `MERGED`/`MERGED_CSV`, write that back to `SPRITE_WORKERS`, then loop over `MERGED` checking `/health` for each worker and printing `OK`/`FAIL` per URL; exits 1 if any worker fails health check.
- `scripts/setup_worker.sh` — updated header comments to describe running the script on "each ADDITIONAL MacBook Pro" / "as many machines as you have" instead of "the SECOND MacBook Pro"; no behavioral change.
- `sprite_worker.py` — updated module docstring to say "Run this on one or more additional machines" instead of "Run this on a second machine"; no behavioral change.

### Key code
```bash
# scripts/setup_main.sh
EXISTING="$(grep "^SPRITE_WORKERS=" "${ENV_FILE}" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
declare -a URLS=()
if [[ -n "${EXISTING}" ]]; then
    IFS=',' read -ra URLS <<< "${EXISTING}"
fi

# ... NEW_URLS populated from $1 or from `discover_worker.py` (one per line) ...

declare -a MERGED=()
for u in "${URLS[@]}" "${NEW_URLS[@]}"; do
    [[ -z "${u}" ]] && continue
    skip=0
    for existing in "${MERGED[@]}"; do
        [[ "${existing}" == "${u}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 0 ]] && MERGED+=("${u}")
done
MERGED_CSV="$(IFS=,; echo "${MERGED[*]}")"

FAILED=0
for url in "${MERGED[@]}"; do
    if curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1; then
        echo "  OK    ${url}"
    else
        echo "  FAIL  ${url}"
        FAILED=1
    fi
done
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-11 20:22:54 WIB

### What changed
- Added a `GET /generate/runs` endpoint that lists previously generated worlds saved under `api_runs/`. For each `*.json` file (newest first), it returns a `SavedRunSummary` with `filename`, a human-readable `theme` (derived by stripping the `<timestamp>_` prefix from the filename and title-casing the remaining slug), `created_at` (parsed from the `YYYYMMDD_HHMMSS` filename prefix, falling back to the file's mtime if unparseable), `num_rooms`, `num_objects`, and the run's `solver` log if present. Files that fail to read or parse as JSON are silently skipped.
- Added a `GET /generate/runs/{filename}` endpoint that returns the full API-shaped `GenerateResponse` for a single saved run. Rejects filenames containing `/` or `\` or not ending in `.json` with a 400, returns 404 if the file doesn't exist, and 500 if it can't be read or parsed as JSON.

### Why
Lets a client browse and re-load previously generated worlds (e.g. to re-run `/generate/solve` against one or inspect a past run's solver results) without needing direct filesystem access to `api_runs/`.

### Files changed
- `api/routers/generate.py` — added `SavedRunSummary` (BaseModel with `filename`, `theme`, `created_at`, `num_rooms`, `num_objects`, `solver: SolverLog | None`), `list_runs()` registered as `GET /runs`, and `get_run(filename: str)` registered as `GET /runs/{filename}`.

### Key code
```python
@router.get("/runs")
def list_runs() -> list[SavedRunSummary]:
    """List previously generated worlds saved under api_runs/."""
    ...

@router.get("/runs/{filename}")
def get_run(filename: str) -> GenerateResponse:
    """Fetch a previously generated world's full API-shaped JSON."""
    if "/" in filename or "\\" in filename or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    ...
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-11 19:28:26 WIB

### What changed
- The solver can now run with multiple cooperating agents in the same world: `solver_node(state, on_tick=..., num_agents=1)` accepts a new `num_agents` param (1-4), and `/generate` and `/generate/solve` API requests gained a matching `num_agents: int = Field(1, ge=1, le=4, ...)` field that flows through to the solver.
- Added `cognitive_solver_policy_multi(agent_ids, ...)` and `solve_world_cooperative(world, num_agents=1, ...)` in `multi_solver.py`, which spin up one `TeamCognition`/`ActionPlanner`/scratchpad per agent (agent ids `agent_1`..`agent_N`), sharing a single `PartyState` and a `shared_milestones` set so agents see each other's discovered milestones and the full shared `ps.log` (dead-ends/blocked exits).
- `PartyState` gained `agent_rooms: dict[str, str]` and `agent_inventories: dict[str, list[str]]` (agent_id -> room/inventory); `current_room`/`inventory` now act as a per-turn "active perspective" slot, swapped via two new `gameplay` helpers: `_load_agent_view(ps, agent_id)` (loads that agent's room/inventory into the active slot) and `_save_agent_view(ps, agent_id)` (writes the active slot back to that agent's dict). `_build_initial_party_state(world, agent_ids=None)` now seeds both dicts for all agents (defaulting to `["agent_1"]`).
- Added `benchmark.engine.MultiAgentEpisode` (with `AgentEpisodeResult` and `CooperativeEpisodeResult` dataclasses), which runs a world to victory/timeout with N agents taking turns per tick — checking victory before the tick and after every individual agent action — and records per-agent final room/inventory/history plus an interleaved global history.
- `TeamCognition` gained agent-to-agent coordination: `_compile_candidates`, `is_policy_candidate`, `policy_candidates`, `_candidate_actions`, and `brief_for` all take an optional `teammate_rooms: dict[str, str] | None` (other agents' current rooms). When no PROGRESS candidate exists, a `COORDINATION`-tagged `go <teammate_room>` candidate is surfaced to regroup with a teammate elsewhere; `TeamBrief` gained a `teammates` field rendered as a `"TEAMMATES: agent_2 is in room_x; ..."` line.
- `render_world(...)` gained optional `agent_rooms`/`agent_inventories` params describing a multi-agent run. The response now includes a new `"parties"` list (one entry per agent with `agentId`/`currentRoom`/`inventory`/`tick`), each room gains an `"agentsHere"` list of agent ids currently in it, `"isCurrentRoom"` is true for any room occupied by any agent, and `"interacted"` on objects is true if the object is in *any* agent's inventory. The top-level `"party"` field is preserved (populated from the first agent) for backward compatibility. `solver.py`'s `on_tick` callback signature changed from `(record, ps, world)` to `(record, ps, world, agent_id)`, and `/generate`/`/generate/solve`'s `on_tick` handlers now build the render using `ps.agent_rooms`/`ps.agent_inventories` plus the new `agent_rooms`/`agent_inventories` render params.

### Why
Extends the solver from a single agent to N cooperating agents that can split up exploration and converge on solving multi-room puzzles together, building on the per-agent perspective groundwork (`agent_rooms`/`agent_inventories`) and exposing it end-to-end through the API and tile-map renderer so the SwiftUI client can display multiple party members at once.

### Files changed
- `api/routers/generate.py` — `GenerateRequest` and new `SolveRequest` gained `num_agents: int = Field(1, ge=1, le=4, ...)`; `_run_pipeline` and `_run_solve_pipeline`'s `on_tick` closures now take `agent_id` and render via `ps.agent_rooms`/`ps.agent_inventories`; both pass `num_agents=req.num_agents` to `solver_node`.
- `benchmark/engine.py` — added `AgentEpisodeResult`, `CooperativeEpisodeResult` dataclasses and `MultiAgentEpisode` class with a `run(policy, record_history=False)` method.
- `src/escape_rooms/agents/cognition.py` — `TeamBrief` gained `teammates: dict[str, str] | None = None` field and renders a `TEAMMATES:` line; `_compile_candidates`, `is_policy_candidate`, `policy_candidates`, `_candidate_actions`, `brief_for` all gained `teammate_rooms: dict[str, str] | None = None` param; `_compile_candidates` adds a `COORDINATION`-tagged candidate when no `PROGRESS` candidate exists; updated module docstring to remove the old "future multi-agent extension" note.
- `src/escape_rooms/agents/multi_solver.py` — added `cognitive_solver_policy_multi(agent_ids, role="solver", scratchpad_limit=30, trace=None, debug_log=None, on_tick=None, *, enforce_candidate_policy=True)` returning a per-agent `_policy(agent_id, world, ps, action_space) -> action`, and `solve_world_cooperative(world, num_agents=1, role="solver", trace=None, debug_log=None, on_tick=None)` which builds `agent_ids = [f"agent_{i+1}" for i in range(num_agents)]` and runs them via `MultiAgentEpisode`; updated module docstring.
- `src/escape_rooms/nodes/gameplay.py` — `_build_initial_party_state(world, agent_ids=None)` now seeds `agent_rooms`/`agent_inventories` for all agents (default `["agent_1"]`); added `_load_agent_view(ps, agent_id)` and `_save_agent_view(ps, agent_id)` helpers.
- `src/escape_rooms/nodes/solver.py` — `solver_node(state, on_tick=None, num_agents=1)` gained `num_agents` param, now imports and delegates to `solve_world_cooperative` instead of `solve_world`; `on_tick` type updated to include `agent_id: str`; log message now includes `agents={num_agents}`.
- `src/escape_rooms/state/schema.py` — `PartyState` gained `agent_rooms: dict[str, str] = Field(default_factory=dict)` and `agent_inventories: dict[str, list[str]] = Field(default_factory=dict)`, with a comment documenting `current_room`/`inventory` as the swapped "active perspective" slot.
- `src/escape_rooms/utils/renderer.py` — `render_world(...)` gained `agent_rooms`/`agent_inventories` params; computes `all_inventories`/`occupied_rooms`; each room dict gains `"agentsHere"`, `"isCurrentRoom"` now checks room membership in `occupied_rooms`; top-level response gains a `"parties"` list alongside the existing `"party"` field.
- `.claude/skills/log/SKILL.md` — expanded the `/log` skill's entry format to require `### Files changed`, `### Key code`, and `### Verification` sections with more detailed "What changed" guidance; non-behavioral (tooling/process change).

### Key code
```python
# src/escape_rooms/nodes/gameplay.py
def _load_agent_view(ps: PartyState, agent_id: str) -> None:
    """Swap ps.current_room/inventory to agent_id's perspective for this turn."""
    ps.current_room = ps.agent_rooms.get(agent_id, ps.current_room)
    ps.inventory = ps.agent_inventories.get(agent_id, [])


def _save_agent_view(ps: PartyState, agent_id: str) -> None:
    """Persist the active perspective back into the per-agent dicts."""
    ps.agent_rooms[agent_id] = ps.current_room
    ps.agent_inventories[agent_id] = list(ps.inventory)
```

```python
# src/escape_rooms/agents/multi_solver.py
def solve_world_cooperative(
    world: GameWorld,
    num_agents: int = 1,
    role: str = "solver",
    trace: list | None = None,
    debug_log: list[dict] | None = None,
    on_tick: Callable[[dict, PartyState, GameWorld, str], None] | None = None,
):
    agent_ids = [f"agent_{i + 1}" for i in range(num_agents)]
    policy = cognitive_solver_policy_multi(
        agent_ids, role, trace=trace, debug_log=debug_log, on_tick=on_tick
    )
    result = MultiAgentEpisode(world, agent_ids).run(policy, record_history=True)
    optimal = bfs_solution_path(world)
    return result, optimal
```

```diff
# src/escape_rooms/agents/cognition.py — coordination candidate
+        if teammate_rooms and not any(c.tag == "PROGRESS" for c in out):
+            for _tid, troom in teammate_rooms.items():
+                if troom and troom != ps.current_room:
+                    push(f"go {troom}", "COORDINATION")
+                    break
```

### Verification
Not verified in conversation (no tests or manual runs were executed).

---

## 2026-06-11 14:53:50 WIB

### What changed
- Added a new `/generate/solve` API endpoint that takes a previously generated world (the `world` field from a `/generate` response) and runs only the solver against it, streaming live per-tick progress and a final render plus solver log as newline-delimited JSON — without re-running world or puzzle generation.
- The streaming machinery behind `/generate` was generalized into a reusable `_stream_pipeline` helper that wraps any pipeline function emitting progress events, so both `/generate` and the new `/generate/solve` share the same NDJSON streaming behavior.

### Why
Lets a client re-solve or re-verify an already-generated world (e.g. after editing it) without paying the cost of regenerating the world and puzzle from scratch, while reusing the existing live progress-streaming infrastructure.

---

## 2026-06-11 13:35:57 WIB

### What changed
- World theming can now distribute its per-room LLM calls across multiple Ollama instances: a new `OLLAMA_THEMING_WORKERS` setting takes a comma-separated list of additional Ollama base URLs, and rooms are split round-robin between the local Ollama instance and each configured worker, themed concurrently.
- `get_llm` now accepts an optional `base_url` override so a role's model can be pointed at a remote Ollama instance instead of the default `OLLAMA_BASE_URL`, with its cache key extended to include the override.

### Why
Theming a world with many rooms previously ran all per-room LLM calls against a single local Ollama instance even though calls are already parallelized per room; spreading rooms across additional machines on the LAN (mirroring the existing `SPRITE_WORKERS` fan-out for sprite generation) cuts wall-clock theming time roughly in proportion to the number of available Ollama instances.

---

## 2026-06-11 11:10:42 WIB

### What changed
- World theming now rejects non-ASCII LLM-generated object names and descriptions on a per-object basis: if a themed name or description contains any non-ASCII characters, it's dropped and that object falls back to its auto-generated id and default description instead of being applied.

### Why
The slugifier strips all non a-z0-9 characters, so a non-ASCII name (e.g. CJK script) collapses into a generic id like `object`/`object_2`, but its paired non-ASCII description would still be applied to that id — producing garbled or unreadable content. Rejecting both per-object before the id-mapping step ensures only the safe fallback is used for affected objects.

---

## 2026-06-11 10:55:24 WIB

### What changed
- World theming now runs one LLM call per room, concurrently, instead of a single sequential call covering the whole world's object graph. Each room's call only sees that room's own objects, dependencies, and goal; if a room's theming call fails, only that room falls back to default descriptions instead of the entire world losing its themed names/descriptions.

### Why
A single world-wide theming prompt grew with the number of rooms and objects, and any failure discarded all themed names/descriptions for the whole world. Scoping each call to one room keeps prompts smaller and lets rooms theme in parallel, while isolating failures to the affected room.

---

## 2026-06-11 10:31:27 WIB

### What changed
- The cognitive solver can now stream its per-tick debug record live as it's produced, via an optional `on_tick` callback threaded through `cognitive_solver_policy` -> `solve_world` -> `solver_node`, alongside the existing batch `debug_log` accumulation.
- The `/generate` API endpoint now wires this callback into its progress queue: when `solve` is requested, each solver tick is emitted to the client as a `{"type": "tick", ...}` event in real time, instead of only surfacing solver results once the whole solve finishes.
- Smoke runs now also write each generated world's full API-shaped JSON to `api_runs/<timestamp>_<theme>_<run-index>.json` (via a new `suffix` parameter on `_write_api_run`), with the path shown in the per-run summary line.

### Why
The `/generate` endpoint's progress stream previously went quiet during the (potentially long) solver pass, giving clients no visibility into solver progress tick-by-tick. Adding a live `on_tick` callback lets the API surface each tick as it happens. The smoke-run API JSON dump lets per-run API payloads from a batch be inspected individually rather than only the latest single-run output.

---

## 2026-06-11 10:03:41 WIB

### What changed
- The cognitive solver now records a per-tick debug log capturing everything needed to reconstruct a run after the fact: the LLM's raw thought/plan/action, the planner's recommended action and full ranked candidate list, any gate overrides (stuck/cycle/dominance), the previous tick's outcome, new milestones, and the final chosen action. `solve_world` and `solver_node` thread this log through via a new `debug_log` parameter, and `main.py` writes it to `logs/solver/debug.json` whenever the solver runs.
- The standalone `--node` CLI flag now only lists `world_builder` and `puzzle_builder` as runnable single-node choices (the new `solver` node was added to the pipeline's node list but isn't independently runnable via `--node`).
- `--solve` now also works with `--smoke`: each generated world is run through the cognitive solver, with its result written to `run_NNN.solver.json`, its debug log written alongside the run's logs, and a `+solver.json` suffix shown in the per-run summary line. The `--solve` help text was updated to describe this smoke-mode behavior.

### Why
The cognitive solver's gates (stuck override, cycle override, dominance override) and planner recommendations were opaque from the outside — when a run behaved unexpectedly there was no way to see what the LLM proposed versus what the planner ranked versus what gate fired and why. The per-tick debug log makes every decision point inspectable after a run completes, and wiring `--solve` into `--smoke` lets that debug data be captured across a batch of generated worlds rather than only a single run.

---

## 2026-06-11 09:24:44 WIB

### What changed
- The LLM solver now has a new "cognitive" strategy (the new default) that pairs the LLM with a deterministic planning layer: a `TeamCognition` module derives the current goal, a SOLVED/STILL-TO-DO board, milestones, loop/stall detection, and a ranked list of valid candidate actions each turn, while an `ActionPlanner` runs a short-horizon beam search over simulated future states to recommend and score actions.
- The solver's turn loop now enforces several gates on the LLM's chosen action: it overrides free (examine/wait) actions when the agent is stuck, redirects off-policy actions back to a recommended candidate, and forces a higher-value action when the planner finds the LLM's choice is dominated by a much better option.
- `solve_world` and the `benchmark` CLI now accept a `--strategy` option (`cognitive` or `react`) to choose between the new cognitive policy and the original single-pass ReAct policy.

### Why
Ports the cognitive architecture from the "Escapee" project (deterministic goal-tracking, milestone/loop detection, and beam-search action planning) to give the LLM solver explicit guardrails against stalling, looping, and low-value moves over long episodes, while keeping the existing ReAct policy available via `--strategy react` for comparison.

---

## 2026-06-10 14:55:40 WIB

### What changed
- Sprite generation can now report progress as it runs: `generate_world_sprites` accepts an optional progress callback that fires after each sprite finishes, reporting how many sprites are done out of the total and which object was just generated.

### Why
Lets callers stream live progress updates during what can be a long-running sprite generation batch, instead of only seeing a result once the whole batch completes.

---

## 2026-06-10 10:43:12 WIB

### What changed
- Pixel-art sprite generation can now be distributed across multiple machines: a new standalone FastAPI worker (`sprite_worker.py`) runs the SDXL pipeline on a remote machine and exposes a `/sprite` endpoint. The main process splits a batch of sprite jobs round-robin between its own local pipeline and any workers listed in the new `SPRITE_WORKERS` env var, generating all shares concurrently.
- The `/generate` API endpoint now only runs the LLM solver when the new `solve` request flag is explicitly set to true; previously every generation request ran the solver and returned its trace, adding latency to plain world-generation calls.

### Why
SDXL sprite generation for a full world's objects was slow when run sequentially on one machine; fanning jobs out to additional machines running `sprite_worker.py` cuts wall-clock time roughly in proportion to the number of workers. Making the solver opt-in via `solve` lets callers that only need a generated world skip the extra solver pass and its cost.

---

## 2026-06-09 09:32:17 WIB

### What changed
- The `oracle_solve` function now records the full action history (`record_history=True`), matching the behaviour of the smoke-run solver so callers get a complete winning trace.
- The unused bank and dataset generation scripts (`benchmark/generate_bank.py`, `benchmark/generate_dataset.py`) have been removed.

### Why
`oracle_solve` was running the same BFS-first policy as the smoke solver but discarding the action trace; enabling history recording aligns the two paths. The generation scripts had no remaining callers and were dead code.

---

## 2026-06-08 23:55:59 WIB

### What changed
- The LLM solver now tracks a permanent **dead-ends set** — objects confirmed to hold no hidden info are remembered and the solver is explicitly told never to examine them again. Both the direct policy and the ReAct policy maintain this set by scanning the engine log for "no hidden info" notes.
- The **ReAct policy** was substantially strengthened: it now carries a sticky **plan field** (one sentence written by the model each tick summarising current multi-step intent) that is prepended to the scratchpad so the model re-reads its own goal before reasoning, even when older scratchpad entries age out. The scratchpad retention window was raised from 16 to 30 entries.
- Every solver prompt now includes **ROOM PROGRESS** (each room's status — DONE or what goal condition remains — with the current room marked) so the agent knows which rooms still need work.
- **CLUES KNOWN** are now annotated with which object or room door needs each clue (`→ needed by: ...`), so the agent applies a code immediately rather than searching for where to use it.
- The theming prompt rule for clue objects was tightened: clue descriptions must make clear that *the object itself* holds the secret (e.g. "numbers scratched into the back"), rather than redirecting the player to another location.
- The benchmark generator's `_oracle_solve` now delegates to `oracle_solve` (the BFS-first solver already used in the puzzle builder), replacing the old `heuristic_policy`-backed `HeadlessEpisode` run. This means a solvable world is never falsely rejected during bank generation because a greedy policy stalled.

### Why
The LLM solver was wasting ticks re-examining dead objects, losing track of its own multi-step plans across scratchpad rollover, and failing to connect known clues to the locks that need them. The improvements give the agent persistent memory of dead ends, a durable intent signal, and annotated clue-to-consumer mappings so it can reason across ticks without losing context. The theming fix prevents the clue description from directing the player elsewhere, which would make the puzzle unsolvable. The oracle change in the bank generator ensures generation uses the same authoritative, complete-search oracle as the rest of the pipeline.

---

## 2026-06-08 23:27:29 WIB

### What changed
- The world-builder's environment variable for model selection was renamed from `GAME_MASTER_MODEL` to `BUILDER_MODEL`, and the temperature variable from `GAME_MASTER_TEMPERATURE` to `BUILDER_TEMPERATURE`. Code that read these env vars (settings, solver fallback chain, dataset generator logging) was updated accordingly.
- The benchmark bank generator now loads its generation and system prompts from the `world_builder` prompt namespace instead of the now-deleted `game_master` namespace.
- The five `game_master` prompts (`directive.txt`, `evaluation.txt`, `generation.txt`, `generation_bank.txt`, `system.txt`) have been deleted from the repository; they were made redundant when the corresponding prompts were moved to `world_builder/` in an earlier session.

### Why
The `game_master` name was a holdover from before the world-building agent was renamed to `world_builder`. The env var rename (`GAME_MASTER_MODEL` → `BUILDER_MODEL`) and prompt namespace cleanup complete that rename at the configuration and file-system level so nothing in the codebase still refers to `game_master` for world generation.

---

## 2026-06-08 23:17:48 WIB

### What changed
- The benchmark generator now supports a `--per-theme` flag that generates exactly N worlds per theme exhaustively (one theme at a time) before moving to the next, rather than cycling themes round-robin. Total worlds produced equals `N × number-of-themes`.
- Generation attempt logic was extracted into a shared `_attempt_one` helper used by both the per-theme and default count-fill modes, eliminating duplicated validation, duplicate-detection, oracle-solve, and depth-check code.
- In default (count-fill) mode, the per-attempt log no longer prints depth or the oracle trace unconditionally — those details are now only shown under `--debug`, matching the per-theme mode's behaviour.

### Why
Running the bank generator with a fixed world count caused uneven theme coverage — prolific themes crowded out harder ones before their quota was reached. The `--per-theme` mode guarantees balanced representation across themes for training and evaluation purposes.

---

## 2026-06-08 23:14:51 WIB

### What changed
- The puzzle graph spec passed to the theming LLM now includes each room's goal text and annotates every object's dependency relationships (e.g. `unlocked-by=<tool>`, `unlocked-by=code`, `unlocked-by=power(…)`, `reveals=code`, `controls-power(…)`), giving the LLM enough causal context to write names and descriptions that feel mechanically coherent rather than generic.
- The theming prompt was updated to use these annotations: it instructs the LLM to write causally-paired names (e.g. a tool and the lock it opens should feel like they belong together), to hint at hidden numbers in clue descriptions without stating digits, and to enforce cross-room vocabulary consistency so all rooms feel part of the same story world.
- World-builder prompts (`generation.txt` and `generation_bank.txt`) now ask the LLM to supply vivid, theme-appropriate snake_case nouns as `key_objects` ids (e.g. `rusted_iron_gate`, `bloodstained_ledger`) instead of structural placeholders (e.g. `room1_safe`, `room1_clue`), so the room feels grounded before the puzzle-builder even runs.
- The benchmark generator (`generate_bank.py`) was updated to use the constructive `build_solvable_world` + `apply_theming` pipeline directly, replacing the old `puzzle_builder_node._generate_puzzle` call so bank generation mirrors the live pipeline.

### Why
The theming pass was operating without knowledge of how objects depend on each other, producing names and descriptions that were thematically flavored but causally incoherent — a "rusted crowbar" unlocking a "porcelain music box" with no logical connection. Surfacing the dependency graph in both the spec and the prompt instructions gives the LLM the information it needs to write names where the tool and its lock feel like a matched pair, and where clue and code objects carry appropriate narrative hints.

---

## 2026-06-08 23:11:16 WIB

### What changed
- Every object in a room is now part of the dependency chain — forming a true linked list where each item requires the previous one to unlock it. Intermediate chain links always use the tool mechanic (so each helper becomes a locked takeable gated by the next link), and only the final (root) link uses a terminating mechanic (code or power) whose helper is directly accessible.
- The effective chain depth is now `max(chain_depth, min_objects_per_room - 1)`, so the per-room object minimum is met entirely by real chained objects. The scenic backfill pass (which padded under-filled rooms with inert props) has been removed — there are no longer any filler objects.

### Why
The previous mechanic picker could randomly assign a terminating mechanic (code or power) to an intermediate chain link, which left a helper locked but with nothing to subsequently gate it — creating an un-takeable (deadlocked) object. Forcing all intermediate links to use the tool mechanic guarantees that every object in a room is chained to and reachable from the one before it, making the dependency graph a strict linked list with no scenic dead weight.

---

## 2026-06-08 20:13:39 WIB

### What changed
- Object ids produced by the LLM theming step are now slugified to `snake_case` (lowercase letters, digits, underscores only). Both the inline `_coerce_id` helper in the puzzle builder and the `apply_theming` slug-mapping in the puzzle graph independently enforce this conversion, and duplicate slugs after renaming are resolved by appending a counter suffix.
- A room's goal text is automatically rewritten to reference the new slug after its key object is renamed during theming, keeping the displayed objective consistent with the new object id.
- Themed descriptions are now remapped from original ids to their new slugs before being applied to objects, so every object receives its LLM-authored description instead of falling back to the generic fallback.
- The theming prompt now explicitly requires names to be `snake_case` identifiers (no spaces, hyphens, or other punctuation), so the LLM is less likely to produce multi-word names that would break single-token action parsing.

### Why
The game engine splits action strings on whitespace and uses the second token as the target object id, so an object named "iron gate" (two tokens) would be unreachable via any action. Slugifying all LLM-produced names to `snake_case` at both the prompt level and in post-processing ensures ids remain single-token action targets regardless of what the model emits.

---

## 2026-06-08 14:54:28 WIB

### What changed
- Generation runs can now optionally hand the finished world to the LLM solver agent via a new `--solve` flag. In a single run this prints the solver's full trace — interleaving each `THINK` line with the action it reasoned about — its verdict (ESCAPED/FAILED) and tick count, and a score against the BFS optimum (reward, efficiency, wasted steps). A companion `--react` flag switches the solver from its direct policy to the ReAct policy (Thought→Action with a running scratchpad).
- Smoke runs honour `--solve`/`--react` too: each run's solver trace plus the BFS optimal path is written to `run_NNN.solve.txt` while only a one-line result prints to stdout. A `solve_summary.json` roll-up is then emitted reporting solvable-worlds-escaped count, solve rate, and mean reward on escapes across the batch.

### Why
With the pipeline now generation-only and the LLM solver available as a standalone agent, wiring it into the generation CLI lets the solver be exercised and benchmarked against the BFS optimum in the same pass that produces a world, and the smoke roll-up turns that into an aggregate solve-rate signal across many worlds without manual per-run inspection.

---

## 2026-06-08 14:37:32 WIB

### What changed
- Puzzle generation is now **solvable by construction**. The dependency chain (locks, keys, clues, and the object graph) is assembled deterministically in code, and the LLM is used only to *theme* the objects — describe and flavor them — so it can no longer produce an unwinnable world. The previous flow generated the entire puzzle graph with the LLM and relied on a retry/repair loop to catch breakage.
- The expensive recovery machinery is now a fallback rather than the main path. Surgical room-skeleton repair for unbuildable key objects and full world regeneration are no longer run on every build; the deterministic + oracle eval still runs as a safety net, and only if it flags an issue does the system fall back to the legacy LLM generation loop.
- Build logging reflects the new path: a successful build reports that the puzzle was built constructively, along with its object count and elapsed time, instead of attempt and world-regen counts.

### Why
Generating the puzzle graph with the LLM made solvability probabilistic, requiring retries, key-object repairs, and whole-world regenerations to salvage broken worlds. Moving graph construction into deterministic code (`agents.puzzle_graph`) guarantees winnability up front and relegates the LLM to theming, where it can't break the puzzle; the old loop is retained only as a safety net.

---

## 2026-06-08 14:19:54 WIB

### What changed
- The solver agent's ReAct trace now prints each `THINK` reasoning line directly above the action it reasoned about. Previously the thought was tagged one tick ahead of its action, so every thought displayed against the *next* tick's action, making the verbose `--world` trace read as if the agent's reasoning lagged a step behind what it actually did.

### Why
The thought tag used `ps.tick + 1` while the engine tags the corresponding action line with `ps.tick` (the policy runs after the tick counter is already incremented), so the interleave paired each thought with the wrong action. Dropping the `+ 1` aligns the two; the fix is display-only and does not change solver behavior.

---

## 2026-06-08 14:03:21 WIB

### What changed
- The live multi-agent gameplay pipeline has been removed. Character generation, player-agent character selection, the tick-driven gameplay loop, and the in-loop Game Master adjudication node are all gone (along with their prompts and the `--mode`/`--player` CLI flags). The LangGraph pipeline is now strictly generation: `world_builder -> puzzle_builder -> END`, and `main.py` only generates and displays a world. Solving a finished world is now the job of the deterministic oracle and the new LLM solver agent instead of the in-graph party.
- Added a standalone LLM solver agent (`agents.solver_agent`) that drives a finished world to escape through the same `(world, ps, action_space) -> action` policy seam as the deterministic `heuristic_policy`/`bfs_policy`, so it runs under `HeadlessEpisode` and is directly comparable to the BFS optimum. It plays under partial observability (only the current room's visible objects, inventory, and discovered clues) with gated movement, and can be run on a saved world via `python -m agents.solver_agent --world <path>`.
- The solver agent has its own model and temperature, configurable via `SOLVER_MODEL` and `SOLVER_TEMPERATURE` (both falling back to the game-master model; solver temperature defaults to a low 0.2 for determinism) and selectable through the new `solver` role in settings.
- The world-generation agent module was renamed from `agents.game_master` to `agents.world_builder` (the file and all imports), removing the long-standing name clash with the now-deleted runtime Game Master eval node.

### Why
The project has pivoted away from running an in-graph LLM party toward generating worlds and then solving them separately: the deterministic oracle proves solvability, and the new LLM solver agent plays the world through the same policy interface as the baselines so its performance can be benchmarked head-to-head against the BFS optimum. Stripping the gameplay/character/player machinery leaves a clean generation-only pipeline, and giving the solver its own model/temperature lets the (small, deterministic) solving model be tuned independently of the (large, creative) generation model. Renaming `game_master` to `world_builder` removes the ambiguity left over now that the runtime Game Master no longer exists.

---

## 2026-06-08 13:24:52 WIB

### What changed
- The recorded solution path is now always ground truth derived from the oracle's actual winning solve over the finalized object graph, never authored by the LLM. The puzzle builder ignores any `solution_path` the model emits and instead derives one via a new shared `bfs_solution_path` helper (BFS-shortest, leave-one-out minimised, annotated with each step's room and the engine's outcome note). This single helper now backs the puzzle builder, the bank generator, and the dataset generator, replacing the per-script trace-rebuilding logic and the hallucinated-id scrubbing pass (`_scrub_ghost_ids`), which were removed.
- Rooms now gate travel to the next room behind their own goal: advancing to a not-yet-visited room requires the current room's `goal_completion` to be satisfied (its locked exit door opened), while backtracking to an already-visited room is always allowed so the party can never softlock. Exits are still always offered so the locked door is visible as an option; the gate is enforced at resolution time.
- World generation prompts now mandate a locked exit door as each non-final room's goal: the door id must appear in `key_objects` and its `goal_completion` must be an `object_state` unlock, making each door the gate that turns its room into a required, sequential step. Puzzle-builder prompts and system prompt were updated to drop solution-path authoring (the runtime no longer reads an LLM path) while keeping the solvability/dependency-chain rules.
- Ollama inference is now tunable via `OLLAMA_KEEP_ALIVE` (model residency; `-1` = never unload) and `OLLAMA_NUM_PREDICT` (max tokens per call), wired through settings and applied to the game LLMs and the bank generator. The `.env.example` defaults also flip dataset generation to `HARD_MODE=true` with `GEN_MAX_ATTEMPTS=3`.

### Why
The LLM-authored solution path referenced hallucinated or repair-drifted object ids and couldn't be trusted to replay; deriving it from the oracle's guaranteed-winning solve over the actual object graph makes it a hallucination-free answer key, so the model no longer needs to author (or have scrubbed) a path at all. The per-room exit-door gate makes every non-final room a required step in the chain rather than something the party can walk past. Keeping models resident and capping tokens per call avoids per-call reloads of multi-GB models and stops a runaway model from burning minutes on one response.

---

## 2026-06-08 09:39:47 WIB

### What changed
- Solvability evaluation now actually plays the world to victory with a complete search instead of relying on a greedy policy. A new `oracle_solve` helper runs an exhaustive breadth-first search of the reachable state space first (so a win that exists within the search budget is always found, never falsely reported "unsolvable" because a greedy policy stalled), falling back to the heuristic policy only when BFS can't enumerate the state space in budget. This oracle now backs the puzzle builder's `_eval_puzzle` gate and the dynamic verdict in `main.py`'s eval node, complementing the static solvability walk.
- Smoke runs now emit a per-run `bfs_path.txt` diagnostic alongside the world/benchmark files, recording whether the world was solved, the path source (BFS shortest vs heuristic best-effort), tick/chain-depth counts, and the full winning action sequence (or the action trace when no win was reached).
- Headless episodes and the BFS/heuristic policies no longer patch a deterministic Game Master exit gate. Because movement is now free in the current model (the party may leave any room without a GM exit-gate check), the gate-patching machinery (`_deterministic_exit_gate` and the per-episode/per-search gate overrides) was removed; routing toward the win room is now a plain BFS over the room adjacency graph, implemented locally in `policies.py`.

### Why
A greedy heuristic policy could stall and make a genuinely solvable world look unsolvable, producing false rejections during generation; running a complete BFS first makes the solvability verdict authoritative. The GM exit-gate patching was dead weight once movement became free in the world model, so it was removed and replaced with a straightforward adjacency BFS for routing. The per-run BFS path file gives a concrete, replayable winning trace for inspecting smoke-generated worlds.

---

## 2026-06-08 09:14:32 WIB

### What changed
- Removed the LLM-as-judge entirely. Deleted the `benchmark/narrative_eval.py` module and its `prompts/narrative_eval/` prompts, and dropped every entry point into it: the `--narrative-eval`/`--eval` and `--oracle-trace` flags (plus `_run_narrative_eval`/`_load_world_from_json`) from `main.py`, and the `--judge-dpo` revision-loop gate, the `--min-quality` post-filter (`quality_filter`), and `_judge_violations` from `benchmark/generate_dataset.py`. World and puzzle acceptance is now gated solely by the deterministic checks — `_eval_world_structure`, `check_solvable`, and `_eval_puzzle` (static solvability + key objects + object counts + oracle).
- The fast, deterministic `--struct-eval` (alias `--trace-eval`) inline per-node check is unchanged and remains the only evaluator. Feedback/correction wording that called the deterministic checks an "automated judge" was corrected to "automated checks".

### Why
The deterministic gates already fully validate structural solvability and key-object presence, and the LLM judge added cost, latency, and non-determinism to generation and dataset building. Removing it makes both pipelines fully deterministic by default.

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
