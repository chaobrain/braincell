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

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from examples.experimental.optim_gradient_scaling.profile_case import (
    RTRLGradientWorkload,
    _pad_seed_roots,
)
from examples.profiling.profile_simulation import main


class RTRLGradientWorkloadTest(unittest.TestCase):
    def test_seed_padding_preserves_prefix_and_repeats_roots(self) -> None:
        roots = (jnp.arange(6).reshape((2, 3)), jnp.asarray([[10.0], [20.0]]))

        padded = _pad_seed_roots(roots, execution_seed_count=5)

        np.testing.assert_array_equal(padded[0], roots[0][jnp.asarray([0, 1, 0, 1, 0])])
        np.testing.assert_array_equal(padded[1], roots[1][jnp.asarray([0, 1, 0, 1, 0])])

    def test_execution_seed_count_cannot_drop_requested_seeds(self) -> None:
        args = argparse.Namespace(
            duration_ms=0.1,
            dt_ms=0.025,
            n_seed=2,
            execution_seed_count=1,
        )

        with self.assertRaisesRegex(ValueError, "at least --n-seed"):
            RTRLGradientWorkload(args)

    def test_profile_smoke_writes_hlo_and_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profile.json"
            hlo = root / "compiled.hlo.txt"
            main(
                [
                    "--case",
                    "rtrl_bptt_gradient",
                    "--platform",
                    "cpu",
                    "--method",
                    "bptt",
                    "--n-cv",
                    "1",
                    "--duration-ms",
                    "0.1",
                    "--batch-size",
                    "2",
                    "--n-seed",
                    "2",
                    "--execution-seed-count",
                    "4",
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                    "--hlo-out",
                    str(hlo),
                    "--out",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text())
            hlo_text = hlo.read_text()

        self.assertEqual(payload["metadata"]["requested_seed_count"], 2)
        self.assertEqual(payload["metadata"]["execution_seed_count"], 4)
        self.assertEqual(payload["materialized"]["gradient_shape"], [2, 3])
        self.assertGreater(payload["metadata"]["memory_analysis"]["temporary_bytes"], 0)
        self.assertIn("HloModule", hlo_text)


if __name__ == "__main__":
    unittest.main()
