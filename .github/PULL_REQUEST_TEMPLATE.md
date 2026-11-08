<!-- Title: the task ID (`CELL-3a: …`) or a conventional prefix (`fix(io): …`, `docs: …`). The squash commit on main takes this title. -->

## Summary

<!-- The problem, the related issue or task ID, and what changes. A before/after example helps. -->

## Testing

<!-- The commands you ran and their results (passes, skips), and anything not verified. For numerical changes: the reference, tolerances, dt and solver. -->

```
pre-commit run --all-files
pytest braincell/...
```

## Checklist

- [ ] One task or problem; split if the diff is over ~1500 lines (deletions and generated files aside).
- [ ] A bug fix comes with a test that fails without it.
- [ ] Docstrings, docs, examples and `changelog.md` (under `Unreleased`) are updated in this PR, or nothing user-visible changed.
- [ ] Breaking changes: none, or listed below with their deprecation and migration.

## Breaking changes

<!-- None, or what breaks, the deprecation warning, and how to migrate. -->
