# Fix the release-triggered PyPI publish workflow

## Problem

`.github/workflows/Publish.yml` runs on `release: [published]`. Every release
event so far has failed — five for five:

| Tag | Date | Run | Failure |
| --- | --- | --- | --- |
| `v0.0.5` | 2025-10-09 | 18364503846 | Legacy `python setup.py bdist_wheel` step |
| `v0.0.6` | 2025-10-13 | 18470126089 | `invalid-publisher` |
| `v0.0.7` | 2026-01-19 | 21125492639 | `invalid-publisher` |
| `v0.0.8` | 2026-03-11 | 22963659534 | `invalid-publisher` |
| `v0.1.0` | 2026-06-18 | 27735433497 | `invalid-publisher` |

The `v0.1.0` run is the only one whose logs have not expired. Its
`Build distribution` job succeeded — including the tag/version verification
step — and `Publish to PyPI` failed with:

```
Trusted publishing exchange failure:
* `invalid-publisher`: valid token, but no corresponding publisher
  (Publisher with matching claims was not found)
* `sub`: `repo:chaobrain/braincell:environment:pypi`
* `workflow_ref`: `chaobrain/braincell/.github/workflows/Publish.yml@refs/tags/v0.1.0`
```

PyPI had no Trusted Publisher registered for this repository, so the OIDC
token exchange had nothing to match against.

## What is already fixed

Commit `8349d5b` ("ci: revert PyPI publish to API token auth", #107, 2026-06-18
21:34 +0800) dropped `environment: pypi` and `permissions: id-token: write` and
switched the publish step to `password: ${{ secrets.PYPI_API_TOKEN }}`. That
secret is an **organisation-level** secret on `chaobrain` — it is present in
neither the repository secrets nor the `pypi` environment secrets, but it
resolves at runtime.

A manual `workflow_dispatch` run (27763271548, 2026-06-18 13:35 UTC) using the
reverted workflow uploaded `braincell-0.1.0` to PyPI successfully. That is how
`v0.1.0` actually shipped.

So the authentication defect is fixed. But **no release event has occurred
since**, which leaves three problems.

## Remaining problems

1. **The release path is untested.** Between releases nothing exercises
   `python -m build`, so a packaging regression is only discovered on the day a
   release is cut — exactly the failure pattern above.
2. **Re-runs and manual dispatches are not idempotent.** Without
   `skip-existing`, re-running a partially-failed release job, or dispatching
   manually after a release already uploaded, fails with
   `400 File already exists`. The manual-dispatch path is the emergency release
   hatch that rescued `v0.1.0`, so it must stay usable.
3. **sdist contents are implicit.** The published `braincell-0.1.0.tar.gz` is
   24.98 MB against a 2.6 MB wheel, because at that time
   `build-system.requires` still listed `setuptools-scm[toml]>=6.2`. That pulls
   `vcs_versioning` into the isolated build environment, which registers a git
   file-finder, which sweeps *every git-tracked file* into the sdist —
   `examples/neuron_compare` alone is 15 MB of benchmark CSVs and notebooks —
   and auto-promotes git-tracked non-`.py` files to package data.

   Commit `532311f` (#112, 2026-08-13) already dropped `setuptools-scm`, leaving
   `requires = ["setuptools>=77.0.3"]`. A local build on that tree produces a
   594 KB sdist and a 767 KB wheel, so **the bloat is already gone**. What
   remains is that sdist contents are decided implicitly by whichever file
   finders happen to be installed in the build environment: re-adding any
   VCS-based build dependency silently re-bloats the sdist by 24 MB with no
   signal. Nothing under `braincell/` references `examples/` (verified by grep),
   so the intended contents can be pinned explicitly.

## Approach

### 1. `Publish.yml`

- Build job: after `python -m build`, run `twine check --strict dist/*`. Bad
  metadata or an unrenderable README fails the build job instead of failing the
  upload after PyPI has already been contacted.
- Publish step gains:
  - `skip-existing: true` — makes re-runs and manual dispatch idempotent.
  - `attestations: false` — password auth disables Trusted Publishing, so the
    action's default `attestations: true` is ignored and emits a warning. Say so
    explicitly rather than carrying the noise.
  - `verbose: true` — the next genuine failure should be readable.

The publish job is deliberately **not** gated on
`if: github.event_name == 'release'`. Manual dispatch is the only recovery path
when a release run fails, and it is the path that actually shipped `v0.1.0`.
`skip-existing: true` addresses the duplicate-upload failure without removing
the hatch.

Trusted Publishing is not restored here. It is the better long-term posture, but
it requires registering a publisher at
<https://pypi.org/manage/project/braincell/settings/publishing/> (owner
`chaobrain`, repository `braincell`, workflow `Publish.yml`, environment `pypi`)
— a PyPI-side action outside this change.

### 2. `CI.yml`

Add a `build_package` job (ubuntu-latest, single Python) that runs
`python -m build` plus `twine check --strict dist/*` on every push. This moves
packaging validation from "once per release" to "every commit", which is the
structural fix for problem 1.

### 3. `MANIFEST.in`

New file pruning `examples/`, `docs/`, `dev/`, `data/`, `.github/`, and
repository-admin markdown from the sdist, while explicitly keeping `LICENSE`,
`README.md` and `braincell/py.typed`.

This is insurance, not a size win — on today's tree the prunes are no-ops
because no git file-finder is installed. Its job is to make the sdist contents
the same whether or not one is: with a finder present the prunes bite, without
one the explicit includes match what setuptools already does. The `build_package`
job in step 2 is what turns a future regression here into a visible failure.

### 4. `data/vis_outputs/`

Delete the two tracked visualisation PNGs (648 KB) and add a `.gitignore` rule
so regenerated output does not get committed again.

## Edge cases and verification

- `skip-existing: true` silently succeeds on a duplicate upload. This cannot
  mask a forgotten version bump, because the `Verify release tag matches package
  version` step fails first when `braincell/_version.py` does not match the tag.
- `twine check --strict` promotes README rendering warnings to errors. Verified
  against a local build of this tree: both the wheel and the sdist PASS, so no
  README change is needed. Should it ever fail, fix the README rather than
  relaxing `--strict`.
- `MANIFEST.in` pruning must not drop files the package needs. Nothing under
  `braincell/` references `examples/` (verified by grep), and `LICENSE` /
  `py.typed` are covered by `license-files` and `package-data` in
  `pyproject.toml` — both confirmed by inspecting the built artefacts.
- Verification: build before and after, compare sdist size, run
  `twine check --strict`, install the wheel into a clean environment and
  `import braincell`, and parse both workflow YAML files.
