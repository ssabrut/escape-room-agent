---
name: auto-pr
description: Auto-generate a pull request title and description (summary, what changes, why it changed, benefits) based on git diff.
disable-model-invocation: true
allowed-tools: Bash(git *)
model: claude-haiku-4-5-20251001
---

# Auto-PR Generator

Generate a pull request title and description from your current branch changes.

## Current changes

```!
git diff main...HEAD
```

## Recent commits

```!
git log main...HEAD --oneline -10
```

## Your task

Analyze the diff and commits above and generate a PR title and description that:

### Title (≤70 characters)
1. **Clear and concise**: summarizes the main feature or fix
2. **Action-oriented**: what does this PR accomplish?
3. **No markdown**: plain text only
4. **Examples**:
   - `Add character ability system with active effects`
   - `Fix null reference in game master gate parsing`
   - `Improve player state initialization`

### Description

Structure your response with these sections:

**Summary**
- One sentence describing what this PR does at a high level

**What Changes**
- Bullet list of the main changes made
- Be specific about files or components modified
- Focus on user-facing or architectural impact

**Why It Changed**
- Explain the motivation or problem being solved
- Reference any bugs, features, or improvements
- Be clear about the "why", not just the "what"

**Benefits**
- List the positive outcomes
- Improved performance, user experience, maintainability, etc.
- Keep each benefit concise

## Output

Format exactly as shown below, with no additional commentary:

```
Title: [your title here]

Summary
[one sentence summary]

What Changes
- [change 1]
- [change 2]
- [change 3]

Why It Changed
[explanation of motivation/problem]

Benefits
- [benefit 1]
- [benefit 2]
- [benefit 3]
```

Do not include any text outside this format.
