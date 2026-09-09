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

"""Offline figure generation with explicit errors and saved output."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from .plotting import plot_comparison, save_plot
from .results import ComparisonResult, Trace


class PlottingTest(unittest.TestCase):
    def test_overlay_error_and_export(self):
        import matplotlib.pyplot as plt

        result = ComparisonResult(Trace([0, 1, 2], [-2, 2, -2]), Trace([0, 1, 2], [-3, 1, -2]), {"model": "toy"})
        fig, axes = plot_comparison(result)
        try:
            self.assertEqual(len(axes), 2)
            np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [-1, -1, 0])
            self.assertEqual(axes[0].get_title(), "toy")
            with tempfile.TemporaryDirectory() as tmp:
                path = save_plot(fig, Path(tmp) / "comparison.png")
                self.assertGreater(path.stat().st_size, 0)
        finally:
            plt.close(fig)
