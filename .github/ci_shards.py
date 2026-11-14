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

"""Split the test files of a pytest session into balanced CI shards.

The `coverage` job runs ``pytest braincell/`` as several shards in parallel.
Setting ``BRAINCELL_TEST_SHARD=<index>/<count>`` makes the root ``conftest.py``
keep only the tests of the files assigned to that shard. Every shard collects
the whole session and assigns the collected files with :func:`assign`, so each
collected file is in exactly one shard by construction, including test files
that no list names yet.

The assignment balances the measured seconds per file in
``.github/test_durations.json`` (longest first, each to the least-loaded shard).
A file missing from that table is estimated from its size, at the table's
average seconds per byte.

Usage:

``python .github/ci_shards.py --self-test``
    Checks the assignment rules on generated inputs.
``python .github/ci_shards.py --update <junit.xml>...``
    Rewrites ``.github/test_durations.json`` from JUnit reports of a full run
    (the coverage job uploads one per shard as the ``junit-shard-*`` artifacts).
``python .github/ci_shards.py --show <count>``
    Prints the estimated seconds of each shard for the current test files.
"""

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DURATIONS = Path(__file__).with_name("test_durations.json")


def parse_shard(value: str) -> tuple[int, int]:
    """Parse ``"<index>/<count>"`` into ``(index, count)`` with ``0 <= index < count``."""
    try:
        index, count = (int(part) for part in value.split("/"))
    except ValueError:
        raise ValueError(f"BRAINCELL_TEST_SHARD must look like '0/3', got {value!r}") from None
    if not 0 <= index < count:
        raise ValueError(f"BRAINCELL_TEST_SHARD index must be in [0, {count}), got {value!r}")
    return index, count


def load_durations(path: Path = DURATIONS) -> dict[str, float]:
    return json.loads(path.read_text()) if path.exists() else {}


def estimate(files: list[str], durations: dict[str, float], root: Path = ROOT) -> dict[str, float]:
    """Seconds per file: measured when known, else the file size times the table's seconds per byte."""
    sizes = {}
    for name in files:
        try:
            sizes[name] = (root / name).stat().st_size
        except OSError:
            sizes[name] = 0
    known = [name for name in durations if (root / name).exists()]
    known_bytes = sum((root / name).stat().st_size for name in known)
    rate = sum(durations[name] for name in known) / known_bytes if known_bytes else 1.0
    return {name: durations[name] if name in durations else sizes[name] * rate for name in files}


def assign(seconds: dict[str, float], count: int) -> dict[str, int]:
    """Assign each file to a shard: longest first, each to the least-loaded shard (ties by index)."""
    loads = [0.0] * count
    shard_of = {}
    for name in sorted(seconds, key=lambda n: (-seconds[n], n)):
        target = min(range(count), key=lambda i: (loads[i], i))
        shard_of[name] = target
        loads[target] += seconds[name]
    return shard_of


def shard_files(files: list[str], index: int, count: int, durations: dict[str, float] | None = None) -> set[str]:
    """The files of ``files`` that shard ``index`` of ``count`` runs."""
    durations = load_durations() if durations is None else durations
    shard_of = assign(estimate(sorted(set(files)), durations), count)
    return {name for name, shard in shard_of.items() if shard == index}


def junit_seconds(reports: list[str], root: Path = ROOT) -> dict[str, float]:
    """Seconds per test file, summed over the test cases of JUnit reports."""
    seconds: dict[str, float] = {}
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            parts = case.get("classname", "").split(".")
            # The longest dotted prefix naming a file: "pkg.mod_test.Class" -> pkg/mod_test.py.
            for end in range(len(parts), 0, -1):
                name = "/".join(parts[:end]) + ".py"
                if (root / name).is_file():
                    seconds[name] = seconds.get(name, 0.0) + float(case.get("time", 0.0))
                    break
    return {name: round(value, 2) for name, value in sorted(seconds.items())}


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = [f"pkg/m{i}_test.py" for i in range(10)]
        (root / "pkg").mkdir()
        for i, name in enumerate(files):
            (root / name).write_text("x" * (100 * (i + 1)))
        durations = {files[0]: 50.0, files[1]: 10.0, "pkg/gone_test.py": 99.0}
        seconds = estimate(files, durations, root)
        assert seconds[files[0]] == 50.0 and seconds[files[1]] == 10.0, seconds
        # Unmeasured files are scaled by the measured seconds per byte: 60 s over 300 bytes.
        assert abs(seconds[files[9]] - 1000 * 0.2) < 1e-9, seconds[files[9]]
        for count in (1, 2, 3, 7, 12):
            shard_of = assign(seconds, count)
            assert sorted(shard_of) == sorted(files), "every file is assigned"
            parts = [{n for n, s in shard_of.items() if s == i} for i in range(count)]
            assert set().union(*parts) == set(files), "the shards cover every file"
            assert sum(len(p) for p in parts) == len(files), "no file is in two shards"
            assert assign(dict(reversed(list(seconds.items()))), count) == shard_of, (
                "the order of the input does not matter"
            )
        loads = [0.0, 0.0, 0.0]
        for name, shard in assign(seconds, 3).items():
            loads[shard] += seconds[name]
        assert max(loads) <= 1.5 * min(loads), loads
        for bad in ("3/3", "-1/3", "1", "a/b"):
            try:
                parse_shard(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} was accepted")
        assert parse_shard("2/3") == (2, 3)
        report = root / "report.xml"
        report.write_text(
            '<testsuites><testsuite>'
            '<testcase classname="pkg.m0_test.Case" name="a" time="1.5"/>'
            '<testcase classname="pkg.m0_test" name="b" time="0.5"/>'
            '<testcase classname="pkg.m3_test.Case" name="c" time="2"/>'
            '</testsuite></testsuites>'
        )
        assert junit_seconds([str(report)], root) == {"pkg/m0_test.py": 2.0, "pkg/m3_test.py": 2.0}
    print("ci_shards self-test passed")


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        self_test()
        return 0
    if len(argv) >= 2 and argv[0] == "--update":
        seconds = junit_seconds(argv[1:])
        if not seconds:
            print("no test case maps to a file", file=sys.stderr)
            return 1
        DURATIONS.write_text(json.dumps(seconds, indent=1, sort_keys=True) + "\n")
        print(f"{DURATIONS.name}: {len(seconds)} files, {sum(seconds.values()):.0f} s")
        return 0
    if len(argv) == 2 and argv[0] == "--show":
        count = int(argv[1])
        files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "braincell").rglob("*_test.py"))
        seconds = estimate(files, load_durations())
        loads = [0.0] * count
        sizes = [0] * count
        for name, shard in assign(seconds, count).items():
            loads[shard] += seconds[name]
            sizes[shard] += 1
        for i in range(count):
            print(f"shard {i}/{count}: {sizes[i]} files, {loads[i]:.0f} s")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
