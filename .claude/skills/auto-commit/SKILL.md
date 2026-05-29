---
name: auto-commit
description: Auto-generate a concise commit message based on current git changes. Use when staging changes and needing a quick, descriptive commit message.
disable-model-invocation: true
allowed-tools: Bash(git *)
model: claude-haiku-4-5-20251001
---

# Auto-Commit Message Generator

Generate a short, semantic commit message from your current staged and unstaged changes.

## Current changes
```!
git diff HEAD
```

## Your task

Analyze the diff above and generate a single-line commit message (≤50 characters) that:

1. **Starts with a type**: feat, fix, refactor, docs, test, chore, style, perf
2. **Is concise and descriptive**: explains WHAT changed, not HOW
3. **Omits punctuation**: no period at the end
4. **Uses lowercase**: except proper nouns

## Examples

Good:
- `feat: add character ability system with active effects`
- `fix: resolve null reference in game master state`
- `refactor: simplify player state management`

Bad:
- `update files` (too vague)
- `fixed the bug` (not semantic)
- `add character ability system with three active effects and update game master` (too long)

## Output

Reply with ONLY the commit message, nothing else. No markdown formatting, no explanation.
