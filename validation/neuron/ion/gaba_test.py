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

"""``GabaA_JE2011`` + ``Cldif_JE2011`` against NEURON's ``gaba`` + ``cldifus`` (ModelDB 148253).

Whether the two ``.mod`` files (no license statement) may be committed awaits
a maintainers' decision; until then the test runs only when
``BRAINCELL_JE2011_MOD_DIR`` holds both files pinned below
(``ModelDBRepository/148253@1d737a0``). Each side runs in its own process
(BrainCell in float64); the limits are T0.

One segment, L = 20 um, diam = 1 um, ``pas``/``IL`` (2.74578e-5 S/cm^2,
-70 mV), 35 degC, one synapse (tau1 = 0.5 ms, tau2 = 6 ms, P = 0.18) driven
by a NetStim from 5 ms every 25 ms with weight 0.0005 uS, 1005 ms at
dt = 0.025 ms. Sample i of BrainCell (t = i dt) is compared with point i of
NEURON; NEURON's extra last point is not compared.
"""

import hashlib
import os
from pathlib import Path
import unittest

import numpy as np

from validation.neuron._tolerance import T0_LIMITS, max_abs_error, relative_current_error
from validation.neuron.ion.chloride_test import _cldif_source, _library, _run

GABA_SHA256 = "6a7e8e810f02c2d9de67b18e158afc54e0971f5d7d5aafe3711244940fec1e24"

_NEURON = """
import json, sys
from neuron import h
h.nrn_load_dll(sys.argv[1]); h.load_file("stdrun.hoc")
s = h.Section(name="soma"); s.L = 20; s.diam = 1; s.nseg = 1; s.cm = 1; s.Ra = 160
s.insert("pas"); s.g_pas = 2.74578e-5; s.e_pas = -70
s.insert("cldifus"); s(0.5).cldifus.cli0 = 5
h.cli0_cl_ion = 5; h.clo0_cl_ion = 133.5; h.celsius = 35; h.dt = 0.025; h.steps_per_ms = 40
syn = h.gaba(s(0.5)); syn.tau1 = 0.5; syn.tau2 = 6; syn.P = 0.18
ns = h.NetStim(); ns.start = 5; ns.interval = 25; ns.number = 1e9; ns.noise = 0
nc = h.NetCon(ns, syn, 0, 0, 0.0005)
v = h.Vector().record(s(0.5)._ref_v); cli = h.Vector().record(s(0.5)._ref_cli)
ecl = h.Vector().record(s(0.5)._ref_ecl)
h.finitialize(-70); h.continuerun(1005)
print(json.dumps([list(v), list(cli), list(ecl)]))
"""

_BRAINCELL = """
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
cell.place(RootLocation(0.5), bc.mech.Synapse("GabaA_JE2011", name="gaba"))
cell.record("v", bc.observe.state("v"))
cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
net = bc.Network("je2011")
stim = net.add_population("stim", bc.NetStim(size=1, start=5.0 * u.ms, number=100, interval=25.0 * u.ms))
post = net.add_population("post", cell)
net.connect("in", source=stim, synapse=post.synapses["gaba"], weight=0.0005 * u.uS)
samples = net.run(dt=0.025 * u.ms, duration=1005.0 * u.ms).samples["post"]
print(json.dumps([np.asarray(samples["v"].values.to_decimal(u.mV))[:, 0].tolist(),
                  np.asarray(samples["ci"].values.to_decimal(u.mM))[:, 0].tolist()]))
"""


def _gaba_source(test: unittest.TestCase) -> Path:
    mod_dir = os.environ.get("BRAINCELL_JE2011_MOD_DIR")
    if not mod_dir:
        test.skipTest(
            "cldif.mod and gabaA_Cl.mod await a maintainers' decision on committing them; "
            "set BRAINCELL_JE2011_MOD_DIR."
        )
    source = Path(mod_dir) / "gabaA_Cl.mod"
    test.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), GABA_SHA256)
    return source


class GabaChlorideAgreementTest(unittest.TestCase):
    def test_gaba_synapse_drives_cldifus_as_in_neuron(self) -> None:
        gaba = _gaba_source(self)
        library = _library(_cldif_source(self), gaba)
        v_nrn, ci_nrn, ecl_nrn = _run(_NEURON, library)
        v_bc, ci_bc = _run(_BRAINCELL)
        self.assertEqual(v_nrn.shape, (40201,))
        self.assertEqual(v_bc.shape, (40200,))
        e_V = max_abs_error([(v_bc, v_nrn[:-1])])
        e_Ci = relative_current_error([(ci_bc, ci_nrn[:-1])])
        print(
            f"gaba + cldifus: e_V = {e_V:.3e} mV, e_Ci = {e_Ci:.3e}, Ci {ci_nrn[0]:.4f} -> {ci_nrn[-1]:.4f} mM, "
            f"E_Cl {ecl_nrn[0]:.3f} -> {ecl_nrn[-1]:.3f} mV"
        )
        self.assertLessEqual(e_V, T0_LIMITS["e_V"])
        self.assertLessEqual(e_Ci, T0_LIMITS["e_s"])
        np.testing.assert_allclose([ecl_nrn[0], ecl_nrn[-1]], [-87.222, -82.336], atol=5e-4)


if __name__ == "__main__":
    unittest.main()
