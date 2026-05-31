---
name: log
description: Append a timestamped entry to CHANGELOG.md describing what code changed in the current conversation and why. Newest entries on top.
disable-model-invocation: true
allowed-tools: Bash(git *), Bash(date *), Read, Write, Edit
---

# Change Log

Append a new entry to `CHANGELOG.md` at the repository root describing the code changes from the current conversation. The newest entry must always appear at the **top** of the file (below the title header).

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

2. **Compose a new entry** using the timestamp above and the diff. The entry must follow this exact format:

   ```markdown
   ## <YYYY-MM-DD HH:MM:SS TZ>

   ### What changed
   - <feature or behavior that was added/changed/removed — described from a user/system perspective, NOT as a file edit>
   - <another feature or behavior change>

   ### Why
   <1–3 sentences explaining the motivation, derived from the conversation context — not just restating the diff. If the reason is not clear from context, write "Not specified in conversation.">

   ---
   ```

   **"What changed" rules:**
   - Each bullet describes a *feature, behavior, or capability* that changed — what the system now does differently — NOT which file was edited.
   - Do NOT lead with file paths. Group related edits across files into one bullet about the single feature they implement.
   - Good: `- Rooms now gate entry behind prerequisites that must be satisfied in an earlier room.`
   - Bad: `- agents/game_master.py: added _build_prerequisites helper.`
   - You may mention a file parenthetically only if it adds genuine clarity, but the bullet must read as a change to *what the software does*, not *which lines moved*.

3. **Insert** the new entry directly below the header block (above any existing entries) so newest is on top. Do not modify prior entries.

4. **Only describe actual behavior/feature changes** present in the diff. Skip whitespace-only, pure-refactor-with-no-behavior-change, or unrelated noise. If there are no meaningful changes, reply with `No changes to log.` and do not modify the file.

## Output

After writing the file, reply with a single line: `Logged <N> file change(s) at <timestamp>.`
