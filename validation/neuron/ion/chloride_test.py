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

"""``Cldif_JE2011`` against NEURON's ``cldifus`` (ModelDB 148253) on one segment.

Whether ``cldif.mod`` (no license statement) may be committed awaits a
maintainers' decision; until then the test runs only when
``BRAINCELL_JE2011_MOD_DIR`` holds the file pinned below
(``ModelDBRepository/148253@1d737a0``). One segment, L = 20 um, diam = 1 um,
35 degC, no current; after ``finitialize(-70)`` the outer shell is set to 8 mM,
then 1000 ms at dt = 0.025 ms. Sample i of BrainCell (float64) is compared with
point i of NEURON's record, each in its own process; the limit is T0.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from validation.neuron._tolerance import T0_LIMITS, relative_current_error

CLDIF_SHA256 = "077dc33c1f42fa85c67f6a5b3408844fd73c268f97c9c09620612b1759aa24ee"
STEPS = 40000

_NEURON = """
import json, sys
from neuron import h
h.nrn_load_dll(sys.argv[1]); h.load_file("stdrun.hoc")
s = h.Section(name="s"); s.L = 20; s.diam = 1; s.nseg = 1
s.insert("cldifus")
s(0.5).cldifus.cli0 = 5
h.cli0_cl_ion = 5; h.clo0_cl_ion = 133.5; h.celsius = 35; h.dt = 0.025
cli = h.Vector().record(s(0.5)._ref_cli)
h.finitialize(-70)
s(0.5).cldifus.cl[0] = 8.0; s(0.5).cli = 8.0
h.fcurrent()
cli.resize(0); h.frecord_init()
h.continuerun(1000)
print(json.dumps(list(cli)))
"""

_BRAINCELL = """
import json
import brainstate, brainunit as u, numpy as np
import braincell as bc
brainstate.environ.set(precision=64)
branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=-70.0 * u.mV)
cell.paint(bc.mech.Ion("Cldif_JE2011", name="cl", species_initializers={"Ci": 8.0 * u.mM}))
cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
result = cell.run(dt=0.025 * u.ms, duration=%d * 0.025 * u.ms)
print(json.dumps(np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0].tolist()))
"""


def _run(code: str, *args: str) -> np.ndarray:
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return np.asarray(json.loads(result.stdout.strip().splitlines()[-1]))


class CldifusAgreementTest(unittest.TestCase):
    def test_zero_current_relaxation_matches_neuron(self) -> None:
        mod_dir = os.environ.get("BRAINCELL_JE2011_MOD_DIR")
        if not mod_dir:
            self.skipTest("cldif.mod awaits a maintainers' decision on committing it; set BRAINCELL_JE2011_MOD_DIR.")
        source = Path(mod_dir) / "cldif.mod"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), CLDIF_SHA256)

        from validation.neuron._mechanisms import compile_mechanisms
        from validation.neuron.cell._nrnmech import nrnmech_path

        library = nrnmech_path(compile_mechanisms("testing", files=[source]) / "x86_64")
        neuron = _run(_NEURON, str(library))
        braincell = _run(_BRAINCELL % (STEPS + 1))
        self.assertEqual(neuron.shape, (STEPS + 1,))
        self.assertEqual(neuron[0], 8.0)
        error = relative_current_error([(braincell, neuron)])
        print(f"cldifus: e_Ci = {error:.3e}, Ci(1000 ms) = {neuron[-1]:.7f} mM")
        self.assertLessEqual(error, T0_LIMITS["e_s"])


if __name__ == "__main__":
    unittest.main()
