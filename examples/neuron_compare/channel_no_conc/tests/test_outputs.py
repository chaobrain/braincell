import os
from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np

from ._helpers import TEMPLATES_ROOT, load_module


matplotlib.use("Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


outputs = load_module(
    TEMPLATES_ROOT / "outputs.py",
    "channel_no_conc_outputs_test",
)


class OutputsTest(unittest.TestCase):
    def test_save_case_plot_uses_gate_alignment_when_gate_names_differ(self) -> None:
        result = {
            "time_ms": [0.0, 0.1, 0.2],
            "braincell": {
                "voltage_mV": [-65.0, -64.0, -63.5],
                "current": {"ix": [0.0, 0.1, 0.2]},
                "gates": {"p": [0.1, 0.2, 0.3]},
            },
            "neuron": {
                "voltage_mV": [-65.1, -64.2, -63.7],
                "current": {"ix": [0.0, 0.1, 0.2]},
                "gates": {"Y": [0.11, 0.21, 0.31]},
            },
            "aligned": {
                "current": {
                    "time_ms": [0.0, 0.1, 0.2],
                    "braincell_ix": [0.0, 0.1, 0.2],
                    "neuron_ix": [0.0, 0.1, 0.2],
                }
            },
            "alignment": {
                "gates": [
                    {
                        "canonical_name": "act",
                        "braincell_gate": "p",
                        "neuron_gate": "Y",
                    }
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "case.png"
            outputs.save_case_plot(out_path, result)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
