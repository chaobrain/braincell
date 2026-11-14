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

"""Fail when a pytest JUnit report ran no test or skipped one.

Usage: ``python .github/junit_counts.py <report.xml>...`` prints each report's
test and skip counts and exits non-zero when a report is missing or
unreadable, holds no test, or has a skipped test. pytest exits 0 when every
test skips, so a CI step whose tests must run (they skip only when NEURON or
a source file is missing) writes ``--junitxml`` and checks it with this.
``--self-test`` checks the rules on generated reports.
"""

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def problems(path: str) -> tuple[str, bool]:
    """Return a report's summary line and whether it fails the rule."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return f"{path}: no report ({exc})", True
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    tests = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return f"{path}: {tests} tests, {skipped} skipped", tests == 0 or skipped > 0


def _self_test() -> int:
    cases = {
        '<testsuites><testsuite tests="3" skipped="0"/></testsuites>': False,
        '<testsuite tests="3" skipped="0"/>': False,
        '<testsuites><testsuite tests="3" skipped="1"/></testsuites>': True,
        '<testsuites><testsuite tests="0" skipped="0"/></testsuites>': True,
        "<testsuites>": True,
    }
    bad = 0
    with tempfile.TemporaryDirectory() as directory:
        for i, (text, fails) in enumerate(cases.items()):
            path = Path(directory) / f"{i}.xml"
            path.write_text(text)
            if problems(str(path))[1] != fails:
                print(f"junit_counts: {text!r} should {'fail' if fails else 'pass'}", file=sys.stderr)
                bad += 1
        if not problems(str(Path(directory) / "missing.xml"))[1]:
            print("junit_counts: a missing report should fail", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    if not argv:
        print("usage: junit_counts.py <report.xml>...", file=sys.stderr)
        return 2
    failed = False
    for path in argv:
        line, bad = problems(path)
        print(line)
        failed |= bad
    if failed:
        print("A report above ran no test or skipped one.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
