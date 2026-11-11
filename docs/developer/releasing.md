---
myst:
  heading_anchors: 2
---

# Releasing

A release is a version bump, a reviewed changelog, a tag, and a green Publish run. Work through the list below in order and paste it into the release PR.
Past releases went out with a missing version section, with new packages unmentioned, and with a feature claimed that did not exist; each step below closes one of those gaps.

## Changelog discipline between releases

`changelog.md` keeps an `## Unreleased` section at the top. A PR with a user-visible change adds its entry there, in the same PR ([contributing](contributing.md#make-and-validate-changes)). The release PR renames that section; it should not have to write the entries.

## Checklist

1. **Choose the version number.** Follow [semantic versioning](https://semver.org/): a new package or feature is a minor release, a fix-only release is a patch, and a removal or rename of a public name is allowed only in a minor release, after the [deprecation period](contributing.md#breaking-changes). Set `__version__` in `braincell/_version.py` and nowhere else; the package metadata and the Publish workflow read it from there. Before opening the release PR, run `python -m braincell._deprecation X.Y` with the new version: each deprecation it lists is due, and is removed or postponed in its own PR first ([conventions](conventions.md#deprecation)); the deprecation tests fail on the release PR while one remains.
2. **Raise the coverage floor (minor releases only).** In a PR merged before the release PR, set `report.fail_under` in `[tool.coverage]` (`pyproject.toml`) to the total coverage that the `coverage` CI job reports on `main`, rounded down ([coverage gates](testing.md#meet-the-coverage-gates)).
3. **Review the changelog against the history.** Run `git log --oneline vPREVIOUS..origin/main` and `git diff --stat vPREVIOUS..origin/main -- braincell`, and check that:
   - every user-visible change, every new package or top-level module (`ls braincell/`), and every new dependency floor or Python version change appears;
   - every breaking change gives the migration, and every deprecation names the release that may remove it;
   - no entry names a version that was not released, and a stub or `NotImplementedError` path (`grep -rn NotImplementedError braincell`) is not described as working.
4. **Check each claimed feature.** Every feature in the new section has a test or an example that runs ([testing](testing.md)); if it has neither, add one or remove the claim.
5. **Cut the section.** Rename `## Unreleased` to `## Version X.Y.Z`, add a new empty `## Unreleased` above it, and write a two-to-four sentence summary of the release at the top of the section.
6. **Run the checks** from a clean checkout of the release commit: `pre-commit run --all-files`, `python -m pytest braincell/ -q`, and `python -m sphinx -b html docs docs/_build/html` with no warnings.
7. **Merge the release PR.** Title it `Release X.Y.Z: version bump and changelog`; it contains the version bump and the changelog edits only, and merges under the usual rules (`ci-ok` green, one approval, through the merge queue).
8. **Tag and publish.** Create a GitHub release on `main` with the tag `vX.Y.Z` (the tag must equal `v` plus `__version__`) and the new changelog section as its notes. Publishing the release starts the [Publish workflow](https://github.com/chaobrain/braincell/blob/main/.github/workflows/Publish.yml) and the Deploy Docs workflow.
9. **Check the Publish result.** In the Actions tab, confirm that the `Publish to PyPI` run is green on the new tag. Its build job fails when the tag does not match `__version__`, and when `twine check --strict` rejects the metadata; its publish job uploads with `skip-existing`, so a re-run cannot fail on files already uploaded. If the run fails after the files were built, fix the cause and re-run it, or start it by hand (`workflow_dispatch`) from the tag.
10. **Verify the artifact.** Once PyPI lists the version, install it in a new virtual environment (`pip install braincell==X.Y.Z`) and import it. Confirm that the Deploy Docs run is green and the documentation site shows the new version.
