---
name: auto-pr
description: Auto-generate a pull request title and description from the current branch's git changes. Use when opening a PR and needing a clear summary of what changed and why.
disable-model-invocation: true
allowed-tools: Bash(git *)
model: claude-haiku-4-5-20251001
---

# Auto-PR Title & Description Generator

Generate a pull request title and description from the commits and diff on the current branch versus its base.

## Branch metadata
```!
git rev-parse --abbrev-ref HEAD
```

## Base branch (origin/main if it exists, else main)
```!
git rev-parse --verify --quiet origin/main >/dev/null && echo origin/main || echo main
```

## Commit log on this branch
```!
git log --no-merges --pretty=format:"%h %s%n%b%n---" $(git rev-parse --verify --quiet origin/main >/dev/null && echo origin/main || echo main)..HEAD
```

## Diff summary (files + line counts)
```!
git diff --stat $(git rev-parse --verify --quiet origin/main >/dev/null && echo origin/main || echo main)..HEAD
```

## Full diff
```!
git diff $(git rev-parse --verify --quiet origin/main >/dev/null && echo origin/main || echo main)..HEAD
```

## Your task

Analyze the branch above and produce a PR title and description.

### Title rules

1. **Starts with a type**: feat, fix, refactor, docs, test, chore, style, perf
2. **≤ 70 characters**
3. **Imperative mood**: "add X", not "added X"
4. **No trailing punctuation, lowercase except proper nouns**

### Description rules

The description has exactly two sections:

**## What changed**
- 2–5 bullet points
- Each bullet names a concrete change (a file, a system, a behavior) — not the commit message verbatim
- Group related file changes into one bullet

**## Why**
- 1–3 short paragraphs (or bullets)
- Explain the motivation: what problem this solves, what need it addresses, what the previous behavior was missing
- Infer the "why" from the diff and commit messages — code that adds validation suggests fixing invalid inputs; renaming suggests clarity; deleting suggests cleanup
- If the "why" is genuinely unclear from the diff, write `Why: (motivation not captured in commits — fill in before merging)` so the human knows to edit it

## Examples

Good title:
- `feat: add character ability system with five mechanical effects`
- `fix: handle null unlocks in game master gate parsing`
- `refactor: replace special_trait string with Ability model`

Good description:
```
## What changed
- Added `Ability` model and `ABILITY_EFFECTS` / `ABILITY_TRIGGERS` constants to `state/game_state.py`
- Replaced `Character.special_trait: str` with `Character.ability: Ability`
- Updated `character_master_node.py` to parse and validate ability fields against the closed effect set
- Rewrote `prompts/character_master/generation.txt` to instruct the LLM to pick from 5 fixed effect slots
- Updated display sites in `main.py`, `player_agent_node.py`, `gameplay_node.py` to render ability info

## Why
Previously `special_trait` was a free-text one-liner with no mechanical hook — it appeared in prompts and printouts but the tick loop never resolved it, so character choice had no gameplay impact. This change makes abilities a closed enum the gameplay loop can dispatch on, so a "rogue" picking `extra_action` actually behaves differently from a "scholar" picking `spot_clue`.
```

## Output

Reply with the title on the first line, a blank line, then the description in markdown.
No leading prose, no code fences around the whole output, no trailing explanation.
