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

"""Pinned third-party sources: the SHA-256 check and the ``data/modeldb`` provenance READMEs."""

import hashlib
from pathlib import Path
import re
import tempfile
import unittest

from validation.neuron._paths import MODELDB_ROOT, REPO_ROOT, pinned_source


def _folders() -> list[Path]:
    return sorted(p for p in MODELDB_ROOT.iterdir() if p.is_dir())


class PinnedSourceTest(unittest.TestCase):
    def test_the_recorded_file_passes_and_a_changed_or_missing_one_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.mod"
            path.write_bytes(b"NEURON { SUFFIX x }\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(pinned_source(path, digest), path)
            path.write_bytes(b"NEURON { SUFFIX y }\n")  # one byte changed
            with self.assertRaisesRegex(ValueError, f"not the recorded {digest}"):
                pinned_source(path, digest)
            path.write_bytes(b"NEURON { SUFFIX x }\r\n")  # a line-ending conversion
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pinned_source(path, digest)
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "is missing"):
                pinned_source(path, digest)


class ModelDBFolderTest(unittest.TestCase):
    """Every file under ``data/modeldb/<accession>/`` is listed, with its SHA-256, in that folder's README."""

    def test_every_file_is_in_its_readme_with_its_hash(self):
        folders = _folders()
        files = [p for folder in folders for p in sorted(folder.iterdir()) if p.name != "README.md"]
        print(f"\n{len(folders)} ModelDB folders, {len(files)} files")
        self.assertGreaterEqual(len(folders), 3)
        self.assertGreaterEqual(len(files), 6)
        index = (MODELDB_ROOT / "README.md").read_text(encoding="utf-8")
        for folder in folders:
            readme = (folder / "README.md").read_text(encoding="utf-8")
            rows = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", readme, flags=re.MULTILINE))
            on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.name != "README.md"}
            with self.subTest(folder=folder.name):
                self.assertEqual(rows, on_disk)
                self.assertIn(f"https://modeldb.science/{folder.name}", readme)
                self.assertRegex(readme, rf"ModelDBRepository/{folder.name} at commit\s+`[0-9a-f]{{40}}`")
                self.assertRegex(readme, r"Retrieved \d{4}-\d{2}-\d{2}")
                self.assertIn("No license statement", readme)
                self.assertIn("D13", readme)
                self.assertIn(f"[`{folder.name}/`]({folder.name}/README.md)", index)

    def test_line_ending_conversion_is_off_for_the_folder(self):
        # Otherwise a Windows checkout rewrites the files and every hash check fails.
        rules = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(re.fullmatch(r"data/modeldb/\*\* -text( .*)?", line) for line in rules))
