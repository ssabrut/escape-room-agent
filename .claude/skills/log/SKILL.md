---
name: log
description: Append a detailed, timestamped entry to CHANGELOG.md describing what code changed in the current conversation, why, the per-file/function breakdown, key code snippets, and how it was verified. Newest entries on top.
disable-model-invocation: true
allowed-tools: Bash(git *), Bash(date *), Read, Write, Edit
---

# Change Log

Append a new, **detailed** entry to `CHANGELOG.md` at the repository root describing the code changes from the current conversation. The newest entry must always appear at the **top** of the file (below the title header).

## Current changes
```!
git diff HEAD
```

## Current timestamp
```!
date "+%Y-%m-%d %H:%M:%S %Z"
```

## Your task

1. **Read** `CHANGELOG.md` at the repo root. If it does not exist, create it with this header:
   ```markdown
   # Change Log

   Chronological log of code changes. Newest entries appear first.
   ```

2. **Compose a new entry** using the timestamp above, the diff, and the conversation context. The entry must follow this exact format:

   ```markdown
   ## <YYYY-MM-DD HH:MM:SS TZ>

   ### What changed
   - <feature or behavior that was added/changed/removed — described from a user/system perspective, NOT as a file edit. Be specific: name the new functions/parameters/config keys/data flow involved, not just the user-visible effect.>
   - <another feature or behavior change, same level of detail>

   ### Why
   <1–3 sentences explaining the motivation, derived from the conversation context — not just restating the diff. If the reason is not clear from context, write "Not specified in conversation.">

   ### Files changed
   - `<path/to/file>` — <one-line summary of what changed in this file, naming the specific functions/classes/constants touched, e.g. "added `_stream_pipeline()` helper; `generate()` and new `solve()` now delegate to it">
   - `<path/to/other_file>` — <...>

   ### Key code
   <For the 1-3 most significant changes, a short fenced code block (diff-style ```diff or language-tagged) showing the new/changed code. Keep each snippet focused — a function signature, a new branch, a new config key — not whole files. Omit this section entirely if the diff is trivial enough that "Files changed" already conveys it.>

   ### Verification
   <How this was checked in the conversation: tests run (with pass/fail result), manual/CLI checks performed, smoke runs, or "Not verified in conversation" if nothing was run.>

   ---
   ```

   **"What changed" rules:**
   - Each bullet describes a *feature, behavior, or capability* that changed — what the system now does differently — NOT which file was edited.
   - Do NOT lead with file paths in this section (file-level detail belongs in "Files changed"). Group related edits across files into one bullet about the single feature they implement.
   - Be concrete and detailed: include new function/parameter/config names, default values, data shapes, and edge cases handled, where relevant.
   - Good: `- Rooms now gate entry behind prerequisites that must be satisfied in an earlier room, enforced via a new _check_prerequisites(room, state) check called from enter_room() before any room-entry side effects run.`
   - Bad: `- agents/game_master.py: added _build_prerequisites helper.`

   **"Files changed" rules:**
   - List every file touched in the diff (use `git diff --stat` if helpful), one bullet each, in the order they appear in the diff.
   - For each file, name the specific functions, classes, constants, or config keys that were added/modified/removed — not just "updated logic".
   - Skip purely mechanical files (lockfiles, generated artifacts) unless their change is itself the point of the entry.

   **"Key code" rules:**
   - Pick the changes most central to understanding the feature — typically new function signatures, new branches in existing logic, or new config/schema fields.
   - Keep snippets short (a handful of lines each). Use ```diff with `+`/`-` markers when showing a modification, or a plain language-tagged block for newly added code.
   - This section is optional — omit it for small/mechanical diffs.

   **"Verification" rules:**
   - Report what was actually done in this conversation: e.g. "Ran `pytest tests/test_solver.py` — 12 passed", "Started the API and hit `/generate/solve` manually with a sample world — returned expected NDJSON stream", or "Not verified in conversation."
   - Do not invent verification that didn't happen.

3. **Insert** the new entry directly below the header block (above any existing entries) so newest is on top. Do not modify prior entries.

4. **Only describe actual behavior/feature changes** present in the diff. Skip whitespace-only, pure-refactor-with-no-behavior-change, or unrelated noise in "What changed" — but still list such files in "Files changed" if they were part of the diff, with a note that the change is non-behavioral. If there are no meaningful changes at all, reply with `No changes to log.` and do not modify the file.

## Output

After writing the file, reply with a single line: `Logged <N> file change(s) at <timestamp>.`
