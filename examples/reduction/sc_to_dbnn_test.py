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

import unittest
import json
from pathlib import Path
import tempfile

import brainunit as u
import numpy as np

from examples.reduction import sc_to_dbnn
from braincell.mech import Synapse
from braincell.reduction.dbnn import ChannelLayout, DBNNGIF


class ScToDbnnExampleTest(unittest.TestCase):
    def test_default_profile_is_full(self):
        self.assertEqual(sc_to_dbnn.parse_args([]).profile, "full")

    def test_quick_profile_and_output_directory_are_configurable(self):
        output = Path("/tmp/sc-to-dbnn")
        args = sc_to_dbnn.parse_args(("--profile", "quick", "--output-dir", str(output)))

        self.assertEqual(args.profile, "quick")
        self.assertEqual(args.output_dir, output)
        self.assertLess(sc_to_dbnn.PROFILES["quick"].train_traces, sc_to_dbnn.PROFILES["full"].train_traces)

    def test_source_fingerprint_tracks_sc_channel_selection(self):
        first = sc_to_dbnn._source_fingerprint("quick", (0,), (8,), ("E",))
        second = sc_to_dbnn._source_fingerprint("quick", (1,), (12,), ("E",))

        self.assertNotEqual(first, second)

    def test_migrates_version_six_gif_asset_to_current_format(self):
        layout = ChannelLayout((Synapse("ExpSyn", name="input"),), (0,), (0,), (0,), (0,))
        model = DBNNGIF(layout, dt=1.0 * u.ms)
        params = model.get_params()
        manifest = {
            "format_version": 6,
            "model_type": "DBNNGIF",
            "n_channels": 1,
            "dt_ms": 1.0,
            "mode": "f",
            "input_sign_mode": "channel_type",
            "input_alignment": "target-step",
            "layout": {key: value for key, value in layout.to_dict().items() if key not in {"branch_ids", "branch_xs"}},
            "metadata": {"source_fingerprint": "legacy-sc"},
            "reset_enabled": False,
        }
        values = {f"param_{name}": np.asarray(value) for name, value in params.items() if name != "dt_ms"}
        values["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.npz"
            destination = Path(directory) / "migrated.npz"
            np.savez_compressed(source, **values)
            migrated = sc_to_dbnn.migrate_v3_gif_model(source, destination, layout)
            restored = DBNNGIF.load(destination)

        self.assertEqual(migrated.mode, "r")
        self.assertEqual(restored.mode, "r")
        self.assertEqual(restored.layout, layout)
        self.assertEqual(restored.source_fingerprint, "legacy-sc")


if __name__ == "__main__":
    unittest.main()
