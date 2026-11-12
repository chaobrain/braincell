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

"""Run with ``pytest examples/convert_mod/nmodl/steps/artifacts_test.py``; it needs no NMODL parser."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

NMODL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(NMODL_ROOT) not in sys.path:
    sys.path.insert(0, str(NMODL_ROOT))

from steps.artifacts import save_pipeline_artifacts  # noqa: E402


def _fake_result(source_file: str) -> dict:
    return {
        "spec": SimpleNamespace(pipeline_name="fake"),
        "step1_result": {
            "ast_json": {"type": "Program"},
            "raw_blocks": {},
            "canonical_blocks": {},
            "bc_ast": {"source_file": source_file, "nested": [{"note": f"read from {source_file}"}]},
        },
        "step2_result": {
            "semantic_ir": {"source_file": source_file},
            "target_ir": {"source_file": source_file},
        },
        "step3_result": {
            "source_file": source_file,
            "rendered_text": f'"""Generated from {source_file}."""\n',
            "validation": {"ok": True},
        },
    }


class SavePipelineArtifactsTest(unittest.TestCase):
    def test_a_source_file_inside_the_repository_is_saved_relative_to_it(self):
        source = NMODL_ROOT / "mod_files" / "kv.mod"
        relative = "examples/convert_mod/nmodl/mod_files/kv.mod"
        with tempfile.TemporaryDirectory() as tmp:
            summary = save_pipeline_artifacts(_fake_result(str(source)), tmp)
            saved = {name: (Path(tmp) / name).read_text(encoding="utf-8") for name in summary["saved_files"]}

        for name, text in saved.items():
            self.assertNotIn(str(REPO_ROOT), text, name)
        self.assertEqual(json.loads(saved["metadata.json"])["source_file"], relative)
        bc_ast = json.loads(saved["bc_ast.json"])
        self.assertEqual(bc_ast["source_file"], relative)
        self.assertEqual(bc_ast["nested"][0]["note"], f"read from {relative}")
        for name in ("semantic_ir.json", "target_ir.json"):
            self.assertEqual(json.loads(saved[name])["source_file"], relative, name)
        self.assertEqual(saved["rendered_channel.py"], f'"""Generated from {relative}."""\n')

    def test_a_source_file_outside_the_repository_is_saved_as_given(self):
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as tmp:
            source = str(Path(src_dir).resolve() / "kv.mod")
            save_pipeline_artifacts(_fake_result(source), tmp)
            metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
            rendered = (Path(tmp) / "rendered_channel.py").read_text(encoding="utf-8")

        self.assertEqual(metadata["source_file"], source)
        self.assertEqual(rendered, f'"""Generated from {source}."""\n')


if __name__ == "__main__":
    unittest.main()
