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

"""Build isolation and cache invalidation for reference mechanisms."""

from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from validation.neuron import _mechanisms
from validation.neuron._paths import DATA_ROOT

_COMMENT_BLOCK = re.compile(r"^\s*COMMENT\b.*?^\s*ENDCOMMENT\b", re.M | re.S)
_USEION = re.compile(r"\bUSEION\s+\w+\s+([^\n}]*)")
_CONSTANT = re.compile(r"^\s*CONSTANT\s*\{(.*?)\}", re.M | re.S)


def _useion_redeclared_as_constant(text: str) -> set[str]:
    """Return the ion variables that a MOD source also declares in a CONSTANT block."""
    text = _COMMENT_BLOCK.sub("", text)
    text = re.sub(r"[:?].*", "", text)
    ion = {
        name
        for statement in _USEION.findall(text)
        for name in re.findall(r"[A-Za-z_]\w*", statement)
        if name not in ("READ", "WRITE", "VALENCE", "REPRESENTS")
    }
    constant = {m.group(1) for body in _CONSTANT.findall(text) for m in re.finditer(r"^\s*([A-Za-z_]\w*)", body, re.M)}
    return ion & constant


class MechanismBuildTest(unittest.TestCase):
    def test_source_changes_rebuild_without_writing_to_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data" / "mechanisms" / "channel" / "example.mod"
            source.parent.mkdir(parents=True)
            source.write_text("NEURON { SUFFIX example }\n")
            build_root = root / "artifacts"

            def compile_stub(command, *, cwd, env, check):
                library = cwd / "x86_64" / "libnrnmech.so"
                library.parent.mkdir()
                library.touch()

            with (
                patch.object(_mechanisms, "model_dir", return_value=root / "data"),
                patch.object(_mechanisms, "mechanism_build_dir", return_value=build_root),
                patch.object(_mechanisms.subprocess, "run", side_effect=compile_stub) as run,
                patch.object(_mechanisms.shutil, "which", return_value="nrnivmodl"),
            ):
                self.assertEqual(_mechanisms.compile_mechanisms("fixture", "channel"), build_root)
                _mechanisms.compile_mechanisms("fixture", "channel")
                self.assertEqual(run.call_count, 1)
                source.write_text("NEURON { SUFFIX example_v2 }\n")
                _mechanisms.compile_mechanisms("fixture", "channel")
                self.assertEqual(run.call_count, 2)
                self.assertEqual((build_root / source.name).read_bytes(), source.read_bytes())
            self.assertEqual(list((root / "data").rglob("*.*")), [source])

    def test_distinct_subsets_have_distinct_build_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "one.mod"
            other = root / "two.mod"
            source.write_text("NEURON { SUFFIX one }\n")
            other.write_text("NEURON { SUFFIX two }\n")

            def compile_stub(command, *, cwd, env, check):
                library = cwd / "x86_64" / "libnrnmech.so"
                library.parent.mkdir()
                library.touch()

            with (
                patch.object(_mechanisms, "model_dir", return_value=root),
                patch.object(_mechanisms, "mechanism_build_dir", side_effect=lambda model, variant: root / "artifacts" / variant),
                patch.object(_mechanisms.subprocess, "run", side_effect=compile_stub),
                patch.object(_mechanisms.shutil, "which", return_value="nrnivmodl"),
            ):
                one = _mechanisms.compile_mechanisms("fixture", files=[source])
                two = _mechanisms.compile_mechanisms("fixture", files=[source, other])
                self.assertNotEqual(one, two)


class MechanismSourceTest(unittest.TestCase):
    def test_no_ion_variable_is_redeclared_as_constant(self):
        # NEURON 9's nocmodl rejects it ("cao used in USEION statement can not
        # be re-declared in a CONSTANT block"), which breaks the whole build.
        self.assertEqual(
            _useion_redeclared_as_constant("NEURON { USEION ca READ cao, cai WRITE cai }\nCONSTANT { cao = 2 (mM) }\n"),
            {"cao"},
        )
        self.assertEqual(
            _useion_redeclared_as_constant("NEURON { USEION ca READ cao }\n: CONSTANT { cao = 2 (mM) }\n"), set()
        )
        sources = sorted(DATA_ROOT.rglob("*.mod"))
        self.assertGreaterEqual(len(sources), 100)  # the scan saw the vendored sources
        offenders = {}
        for source in sources:
            names = _useion_redeclared_as_constant(source.read_text(encoding="utf-8", errors="replace"))
            if names:
                offenders[source.relative_to(DATA_ROOT).as_posix()] = sorted(names)
        self.assertEqual(offenders, {}, f"checked {len(sources)} MOD files")
