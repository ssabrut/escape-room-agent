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
   - <file_path>: <concise description of the change>
   - <file_path>: <concise description of the change>

   ### Why
   <1–3 sentences explaining the motivation, derived from the conversation context — not just restating the diff. If the reason is not clear from context, write "Not specified in conversation.">

   ---
   ```

3. **Insert** the new entry directly below the header block (above any existing entries) so newest is on top. Do not modify prior entries.

4. **Only describe actual code changes** present in the diff. Skip whitespace-only or unrelated noise. If there are no changes, reply with `No changes to log.` and do not modify the file.

## Output

After writing the file, reply with a single line: `Logged <N> file change(s) at <timestamp>.`
