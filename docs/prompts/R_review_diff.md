# Prompt R — Review current diff

Goal:
Review the current uncommitted diff before I accept or commit it.

Read first:
- `AGENTS.md`
- `docs/review_checklist.md`
- current git diff

Tasks:
1. Identify correctness risks, scope creep, stale docs, dead code, and fragile assumptions.
2. Check whether commands/tests were run and whether failures were documented.
3. Suggest minimal fixes; do not rewrite large parts unless necessary.
4. End with a concise commit message suggestion.
