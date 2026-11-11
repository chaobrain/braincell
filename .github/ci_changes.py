# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Decide which CI jobs a pull request needs from the files it changes.

Usage: ``python .github/ci_changes.py [--all] < changed-files`` prints
``braincell=…``, ``neuron=…`` and ``bench=…`` lines for ``$GITHUB_OUTPUT``;
``--self-test`` checks the rules below and exits non-zero on a mismatch.

- Any build or CI input (``.github/``, ``pyproject.toml``, ``requirements*``,
  ``setup.*``, ``MANIFEST.in``, any ``conftest.py``, ``data/``) runs everything.
- ``braincell``: the package test suite with coverage, for ``braincell/``.
  The suite reads no file outside the package.
- ``neuron``: the NEURON comparisons, for ``braincell/`` or ``validation/neuron/``.
- ``bench``: ``validation/optim`` and the benchmark scripts' tests, for
  ``braincell/``, ``validation/`` outside ``validation/neuron/``, or ``benchmarks/``.

Everything else (``docs/``, ``examples/``, Markdown, the pre-commit config)
is covered by the always-run ``checks`` job (lint, package build, docs).
"""

import fnmatch
import sys

RUN_ALL = (
    ".github/*",
    "pyproject.toml",
    "requirements*",
    "setup.*",
    "MANIFEST.in",
    "conftest.py",
    "*/conftest.py",
    "data/*",
)


def jobs(files: list[str]) -> dict[str, bool]:
    every = any(fnmatch.fnmatch(f, p) for f in files for p in RUN_ALL)
    pkg = any(f.startswith("braincell/") for f in files)
    neuron = any(f.startswith("validation/neuron/") for f in files)
    bench = any(
        f.startswith("benchmarks/") or (f.startswith("validation/") and not f.startswith("validation/neuron/"))
        for f in files
    )
    return {"braincell": every or pkg, "neuron": every or pkg or neuron, "bench": every or pkg or bench}


CASES = [
    (["docs/index.md", "examples/a.py", "changelog.md"], (False, False, False)),
    (["validation/neuron/channel_no_conc/tolerances.py"], (False, True, False)),
    (["validation/optim/x.py"], (False, False, True)),
    (["benchmarks/performance/a.py"], (False, False, True)),
    (["braincell/channel/sodium.py"], (True, True, True)),
    ([".github/workflows/CI.yml"], (True, True, True)),
    (["pyproject.toml"], (True, True, True)),
    (["validation/neuron/conftest.py"], (True, True, True)),
    (["data/mechanisms/testing/clleak.mod"], (True, True, True)),
    ([], (False, False, False)),
]


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        bad = [(f, want) for f, want in CASES if tuple(jobs(f).values()) != want]
        for f, want in bad:
            print(f"ci_changes: {f} -> {jobs(f)}, expected {want}", file=sys.stderr)
        return 1 if bad else 0
    files = [line.strip() for line in sys.stdin if line.strip()]
    result = {k: True for k in ("braincell", "neuron", "bench")} if "--all" in argv else jobs(files)
    for key, value in result.items():
        print(f"{key}={'true' if value else 'false'}")
    print(f"ci_changes: {len(files)} changed files -> {result}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
