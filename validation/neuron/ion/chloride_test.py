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

``cldif.mod`` (``ModelDBRepository/148253@1d737a0``, no license statement) is
committed unmodified in ``data/modeldb/148253`` (MECH-48); the test checks the
SHA-256 pinned below before compiling it. Each side runs in its own process
(BrainCell in float64); the limits are T0.

- Relaxation: one segment, L = 20 um, diam = 1 um, 35 degC, no current; after
  ``finitialize(-70)`` the outer shell is set to 8 mM, then 1000 ms at
  dt = 0.025 ms. Sample i of BrainCell is compared with point i of NEURON.
- Chloride leak under current clamp: the same segment with ``pas``/``IL``,
  ``clleak``/``Cl_Leak`` (``data/mechanisms/testing/clleak.mod``) and the pool,
  0.003 nA from 5 to 505 ms, 600 ms. Sample i of BrainCell (t = i dt) is
  compared with point i of NEURON; NEURON's extra last point is not compared.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from validation.neuron._paths import DATA_ROOT, MODELDB_ROOT, pinned_source
from validation.neuron._tolerance import T0_LIMITS, max_abs_error, relative_current_error

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


_NEURON_CLAMP = """
import json, sys
from neuron import h
h.nrn_load_dll(sys.argv[1]); h.load_file("stdrun.hoc")
s = h.Section(name="soma"); s.L = 20; s.diam = 1; s.nseg = 1; s.cm = 1; s.Ra = 160
s.insert("pas"); s.g_pas = 2.74578e-5; s.e_pas = -70
s.insert("clleak"); s.g_clleak = 1e-4
s.insert("cldifus"); s(0.5).cldifus.cli0 = 5
h.cli0_cl_ion = 5; h.clo0_cl_ion = 133.5; h.celsius = 35; h.dt = 0.025; h.steps_per_ms = 40
ic = h.IClamp(s(0.5)); ic.delay = 5; ic.dur = 500; ic.amp = 0.003
v = h.Vector().record(s(0.5)._ref_v); cli = h.Vector().record(s(0.5)._ref_cli)
h.finitialize(-70); h.continuerun(600)
print(json.dumps([list(v), list(cli)]))
"""

_BRAINCELL_CLAMP = """
import json
import brainstate, brainunit as u, numpy as np
import braincell as bc
from braincell.filter import RootLocation
brainstate.environ.set(precision=64)
branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um, type="soma")
cell = bc.Cell(bc.Morphology.from_root(branch, name="soma"), cv_policy=bc.CVPerBranch(1), V_init=-70.0 * u.mV)
cell.paint(bc.mech.CableProperty(resting_potential=-70.0 * u.mV, membrane_capacitance=1.0 * u.uF / u.cm**2,
                                 axial_resistivity=160.0 * u.ohm * u.cm, temperature=u.celsius2kelvin(35.0)))
cell.paint(bc.mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
cell.paint(bc.mech.Ion("Cldif_JE2011", name="cl"))
cell.paint(bc.mech.Channel("Cl_Leak", g_max=0.1 * u.mS / u.cm**2))
cell.place(RootLocation(0.5), bc.mech.CurrentClamp(delay=5.0 * u.ms, durations=500.0 * u.ms, amplitudes=0.003 * u.nA))
cell.record("v", bc.observe.state("V"))
cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
result = cell.run(dt=0.025 * u.ms, duration=600.0 * u.ms)
print(json.dumps([np.asarray(result.samples["v"].values.to_decimal(u.mV))[:, 0].tolist(),
                  np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0].tolist()]))
"""


def _cldif_source() -> Path:
    return pinned_source(MODELDB_ROOT / "148253" / "cldif.mod", CLDIF_SHA256)


def _library(*sources: Path) -> str:
    from validation.neuron._mechanisms import compile_mechanisms
    from validation.neuron.cell._nrnmech import nrnmech_path

    return str(nrnmech_path(compile_mechanisms("testing", files=list(sources)) / "x86_64"))


def _run(code: str, *args: str) -> np.ndarray:
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return np.asarray(json.loads(result.stdout.strip().splitlines()[-1]))


class CldifusAgreementTest(unittest.TestCase):
    def test_zero_current_relaxation_matches_neuron(self) -> None:
        neuron = _run(_NEURON, _library(_cldif_source()))
        braincell = _run(_BRAINCELL % (STEPS + 1))
        self.assertEqual(neuron.shape, (STEPS + 1,))
        self.assertEqual(neuron[0], 8.0)
        error = relative_current_error([(braincell, neuron)])
        print(f"cldifus: e_Ci = {error:.3e}, Ci(1000 ms) = {neuron[-1]:.7f} mM")
        self.assertLessEqual(error, T0_LIMITS["e_s"])

    def test_chloride_leak_under_current_clamp_matches_neuron(self) -> None:
        library = _library(_cldif_source(), DATA_ROOT / "mechanisms" / "testing" / "clleak.mod")
        v_nrn, ci_nrn = _run(_NEURON_CLAMP, library)
        v_bc, ci_bc = _run(_BRAINCELL_CLAMP)
        self.assertEqual(v_nrn.shape, (24001,))
        self.assertEqual(v_bc.shape, (24000,))
        e_V = max_abs_error([(v_bc, v_nrn[:-1])])
        e_Ci = relative_current_error([(ci_bc, ci_nrn[:-1])])
        print(f"clleak + cldifus: e_V = {e_V:.3e} mV, e_Ci = {e_Ci:.3e}, Ci(600 ms) = {ci_nrn[-1]:.6f} mM")
        self.assertLessEqual(e_V, T0_LIMITS["e_V"])
        self.assertLessEqual(e_Ci, T0_LIMITS["e_s"])
        self.assertAlmostEqual(ci_nrn[-1], 5.830631, delta=5e-7)


if __name__ == "__main__":
    unittest.main()
