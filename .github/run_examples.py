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

"""Run the example scripts and the tests under ``examples/`` (daily ``examples`` job).

Usage, from the repository root::

    python .github/run_examples.py [--timeout S] [--only PATH ...]
    python .github/run_examples.py --pytest REPORT.xml
    python .github/run_examples.py --self-test

Scripts: every tracked ``examples/**/*.py`` except ``*_test.py``,
``__init__.py`` and ``conftest.py`` runs in its own process from the root,
with ``MPLBACKEND=Agg``, ``JAX_PLATFORMS=cpu`` and the root on
``PYTHONPATH``: as ``python -m examples.<dotted.name>`` when its folder is a
package (it may use relative imports), as ``python <path>`` otherwise. A
script fails on a non-zero exit or after ``--timeout`` seconds (a
``[[timeout]]`` entry sets its own). Scripts run in path order, so a script
may read what an earlier one in its folder wrote.

Tests: ``--pytest`` runs ``pytest examples`` with the ``[[deselect]]`` entries
deselected and fails when the JUnit report has no test or a skipped one (a
test that needs an optional dependency is listed, not skipped silently).

``examples/nightly_skip.toml`` holds the ``[[skip]]``, ``[[timeout]]`` and
``[[deselect]]`` entries, each skip and deselect with its reason. An entry
for a script that is not tracked, or for a test that pytest does not
collect, is an error, so the list cannot go stale. Exits 1 on a failure or
an invalid list. ``--self-test`` checks the list rules on generated lists and
that ``examples/nightly_skip.toml`` names only tracked scripts.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import tomllib

from junit_counts import problems

ROOT = Path(__file__).resolve().parents[1]
SKIP_LIST = ROOT / "examples" / "nightly_skip.toml"
NOT_SCRIPTS = ("__init__.py", "conftest.py")
KEYS = {"skip": {"path", "reason"}, "timeout": {"path", "seconds"}, "deselect": {"nodeid", "reason"}}


def scripts() -> list[str]:
    """Every tracked example script, as a POSIX path relative to the root."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "examples/*.py"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return sorted(n for n in out.split("\0") if n and not n.endswith("_test.py") and Path(n).name not in NOT_SCRIPTS)


def load_list(path: Path) -> tuple[dict[str, list[dict]], list[str]]:
    """Return the list's tables and its format errors (unknown tables or keys, a missing reason, a duplicate)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    errors = [f"{path.name}: unknown table [[{name}]]" for name in sorted(set(data) - set(KEYS))]
    tables = {name: data.get(name, []) for name in KEYS}
    for name, entries in tables.items():
        key = "nodeid" if name == "deselect" else "path"
        seen = set()
        for entry in entries:
            target = entry.get(key, "")
            if set(entry) != KEYS[name]:
                errors.append(f"[[{name}]] {target!r}: keys must be {sorted(KEYS[name])}, got {sorted(entry)}")
            if "reason" in KEYS[name] and not str(entry.get("reason", "")).strip():
                errors.append(f"[[{name}]] {target!r}: no reason")
            if target in seen:
                errors.append(f"[[{name}]] {target!r}: listed twice")
            seen.add(target)
    for entry in tables["timeout"]:
        if not isinstance(entry.get("seconds"), (int, float)) or entry["seconds"] <= 0:
            errors.append(f"[[timeout]] {entry.get('path')!r}: seconds must be positive")
    skipped = {e.get("path") for e in tables["skip"]}
    errors += [f"[[timeout]] {e.get('path')!r}: also skipped" for e in tables["timeout"] if e.get("path") in skipped]
    return tables, errors


def command(script: str) -> list[str]:
    """The command that runs ``script``: as a module inside a package, else by path."""
    path = Path(script)
    if (ROOT / path.parent / "__init__.py").is_file():
        return [sys.executable, "-m", ".".join(path.with_suffix("").parts)]
    return [sys.executable, script]


def _env() -> dict[str, str]:
    env = dict(os.environ, MPLBACKEND="Agg", JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    return env


def _text(output) -> str:
    return output.decode(errors="replace") if isinstance(output, bytes) else (output or "")


def run(script: str, timeout: float) -> tuple[str, float, str]:
    """Run one script; return its status (pass, fail, timeout), its duration and the end of its output."""
    start = time.perf_counter()
    try:
        result = subprocess.run(command(script), cwd=ROOT, env=_env(), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return "timeout", time.perf_counter() - start, (_text(exc.stdout) + _text(exc.stderr))[-3000:]
    status = "pass" if result.returncode == 0 else "fail"
    return status, time.perf_counter() - start, (_text(result.stdout) + _text(result.stderr))[-3000:]


def run_scripts(tables: dict[str, list[dict]], default_timeout: float, only: list[str] | None) -> int:
    known = scripts()
    errors = [
        f"[[{n}]] {e['path']!r}: not a tracked example script"
        for n in ("skip", "timeout")
        for e in tables[n]
        if e.get("path") not in known
    ]
    skips = {e["path"]: e["reason"] for e in tables["skip"]}
    timeouts = {e["path"]: e["seconds"] for e in tables["timeout"]}
    counts = dict.fromkeys(("pass", "fail", "timeout", "skipped"), 0)
    selected = known if only is None else [s for s in known if s in only]
    start = time.perf_counter()
    for script in selected:
        if script in skips:
            counts["skipped"] += 1
            print(f"SKIP     {script}: {skips[script]}", flush=True)
            continue
        status, seconds, tail = run(script, timeouts.get(script, default_timeout))
        counts[status] += 1
        print(f"{status.upper():8} {script} ({seconds:.1f} s)", flush=True)
        if status != "pass":
            print("    " + tail.replace("\n", "\n    "), flush=True)
    for error in errors:
        print(f"skip list: {error}", flush=True)
    print(
        f"{len(selected)} example scripts: {counts['pass']} passed, {counts['fail']} failed, "
        f"{counts['timeout']} timed out, {counts['skipped']} skipped; {time.perf_counter() - start:.0f} s"
    )
    return 1 if errors or counts["fail"] or counts["timeout"] else 0


def run_pytest(tables: dict[str, list[dict]], report: Path) -> int:
    base = [sys.executable, "-m", "pytest", "examples", "-p", "no:cacheprovider"]
    collected = subprocess.run([*base, "--collect-only", "-q"], cwd=ROOT, env=_env(), capture_output=True, text=True)
    ids = set(collected.stdout.split())
    errors = [
        f"[[deselect]] {e['nodeid']!r}: pytest does not collect it"
        for e in tables["deselect"]
        if e["nodeid"] not in ids
    ]
    for entry in tables["deselect"]:
        print(f"DESELECT {entry['nodeid']}: {entry['reason']}", flush=True)
    deselect = [arg for e in tables["deselect"] for arg in ("--deselect", e["nodeid"])]
    status = subprocess.run(
        [*base, "-rs", "--durations=10", f"--junitxml={report}", *deselect], cwd=ROOT, env=_env()
    ).returncode
    line, bad = problems(str(report))
    print(line)
    for error in errors:
        print(f"skip list: {error}", flush=True)
    if bad:
        print(
            "pytest examples ran no test or skipped one: list a test that needs an optional dependency under [[deselect]]."
        )
    return 1 if status or bad or errors else 0


def _self_test() -> int:
    bad = []
    cases = {
        '[[skip]]\npath = "a.py"\nreason = "x"\n[[timeout]]\npath = "b.py"\nseconds = 5\n': 0,
        '[[skip]]\npath = "a.py"\nreason = " "\n': 1,
        '[[skip]]\npath = "a.py"\nreason = "x"\n[[skip]]\npath = "a.py"\nreason = "y"\n': 1,
        '[[skip]]\npath = "a.py"\n': 2,
        '[[timeout]]\npath = "a.py"\nseconds = 0\n': 1,
        '[[skip]]\npath = "a.py"\nreason = "x"\n[[timeout]]\npath = "a.py"\nseconds = 5\n': 1,
        '[[deselect]]\nnodeid = "t.py::test"\nreason = "x"\nextra = 1\n': 1,
        '[[skipped]]\npath = "a.py"\nreason = "x"\n': 1,
    }
    with tempfile.TemporaryDirectory() as directory:
        for i, (text, expected) in enumerate(cases.items()):
            path = Path(directory) / f"{i}.toml"
            path.write_text(text)
            errors = load_list(path)[1]
            if len(errors) != expected:
                bad.append(f"{text!r}: {len(errors)} errors, expected {expected}: {errors}")
    if command("examples/x/y.py")[1:] != ["examples/x/y.py"]:
        bad.append("a script outside a package should run by path")
    package = next((s for s in scripts() if (ROOT / Path(s).parent / "__init__.py").is_file()), None)
    if package is None or command(package)[1] != "-m":
        bad.append(f"a script inside a package should run with -m: {package}")
    tables, errors = load_list(SKIP_LIST)
    known = set(scripts())
    errors += [
        f"{e['path']}: not a tracked example script"
        for n in ("skip", "timeout")
        for e in tables[n]
        if e["path"] not in known
    ]
    bad += [f"{SKIP_LIST.name}: {e}" for e in errors]
    for line in bad:
        print(f"run_examples self-test: {line}", file=sys.stderr)
    print(
        f"run_examples self-test: {len(cases)} lists, {len(known)} scripts, {len(tables['skip'])} skipped; {'FAIL' if bad else 'ok'}"
    )
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per script (default 600)")
    parser.add_argument("--only", nargs="*", default=None, help="run only these scripts")
    parser.add_argument("--pytest", type=Path, metavar="REPORT", help="run pytest examples, writing this JUnit report")
    parser.add_argument("--skip-list", type=Path, default=SKIP_LIST)
    parser.add_argument("--self-test", action="store_true", help="check the list rules and the skip list")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    tables, errors = load_list(args.skip_list)
    for error in errors:
        print(f"skip list: {error}", flush=True)
    status = run_pytest(tables, args.pytest) if args.pytest else run_scripts(tables, args.timeout, args.only)
    return 1 if errors else status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
