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

import functools
import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell as bc
from braincell._base_channel import IonInfo
from braincell.channel import IL, Cl_Leak, OhmicHH
from braincell.ion import Chloride
from braincell.ion._testing import run_float64

_E_CL_35C = -87.22198260005996  # mV, Ci 5 / Co 133.5 mM, z = -1, 35 degC
_E_HCO3_35C = -12.892326235873629  # mV, HCO3 16 / 26 mM, z = -1, 35 degC
_UA_CM2 = u.uA / u.cm**2


def _cl_info(E_mV: float) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((1,), 5.0) * u.mM, Co=jnp.full((1,), 133.5) * u.mM, E=jnp.full((1,), E_mV) * u.mV, valence=-1
    )


def _tonic_gaba_currents() -> dict:
    """Return the Cl_Leak + IL sum and the tonic GABA_A current at -60 mV, in uA/cm2."""
    g_t, P, V = 0.1, 0.18, jnp.full((1,), -60.0) * u.mV
    cl = Cl_Leak(size=1, g_max=(1.0 - P) * g_t * (u.mS / u.cm**2))
    hco3 = IL(size=1, g_max=P * g_t * (u.mS / u.cm**2), E=_E_HCO3_35C * u.mV)
    total = cl.current(V, _cl_info(_E_CL_35C)) + hco3.current(V)
    v = -60.0
    tonic = g_t * ((1.0 - P) * (_E_CL_35C - v) + P * (_E_HCO3_35C - v))  # mS/cm2 * mV = uA/cm2
    return {"sum": float(np.asarray(total.to_decimal(_UA_CM2)).reshape(-1)[0]), "tonic": tonic}


@functools.cache
def _float64_report() -> dict:
    """The float64 tonic GABA measurements."""
    return run_float64(_tonic_gaba_currents)


def _cable_cell() -> bc.Cell:
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um)
    return bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=-70.0 * u.mV)


class ClLeakCurrentTest(unittest.TestCase):
    """Ohm's law with the reversal potential of the attached chloride ion."""

    def test_template_and_root_type(self) -> None:
        self.assertTrue(issubclass(Cl_Leak, OhmicHH))
        self.assertIs(Cl_Leak.root_type, Chloride)
        self.assertEqual(Cl_Leak.gates, ())
        np.testing.assert_allclose(np.asarray(Cl_Leak(size=1).g_max.to_decimal(u.mS / u.cm**2)), 0.1)

    def test_current_is_inward_positive(self) -> None:
        ch = Cl_Leak(size=1, g_max=0.1 * (u.mS / u.cm**2))
        V = jnp.full((1,), -60.0) * u.mV
        current = ch.current(V, _cl_info(_E_CL_35C))
        np.testing.assert_allclose(np.asarray(current.to_decimal(_UA_CM2)), [-2.722198260005996], rtol=1e-6)
        self.assertIsNone(ch.compute_derivative(V, _cl_info(_E_CL_35C)))

    def test_current_enters_the_chloride_pool(self) -> None:
        cell = _cable_cell()
        cell.paint(bc.mech.Ion("Cldif_JE2011", name="cl"))
        cell.paint(bc.mech.Channel("Cl_Leak", g_max=0.1 * (u.mS / u.cm**2)))
        cell.init_state()
        ion = cell.get_ion("cl")
        self.assertEqual(list(ion.channels), ["Cl_Leak"])
        # 0.1 mS/cm2 * (E_Cl(35 degC) - (-70 mV)).
        expected = 0.1 * (_E_CL_35C + 70.0)
        np.testing.assert_allclose(
            np.asarray(ion.current(cell.V.value).to_decimal(_UA_CM2)).reshape(-1), [expected], rtol=1e-5
        )

    def test_tonic_gaba_a_is_cl_leak_plus_il(self) -> None:
        report = _float64_report()
        np.testing.assert_allclose(report["sum"], -1.3842644454506423, rtol=1e-12)
        np.testing.assert_allclose(report["tonic"], -1.3842644454506423, rtol=1e-12)


class ClLeakWithoutChlorideTest(unittest.TestCase):
    """A Cell that paints Cl_Leak without a chloride ion fails at init_state."""

    def test_cable_and_lumped_cells_raise_key_error(self) -> None:
        cells = {
            "cable": _cable_cell(),
            "lumped": bc.Cell.point(1, length=20.0 * u.um, radius=0.5 * u.um, V_init=-70.0 * u.mV),
        }
        for kind, cell in cells.items():
            with self.subTest(cell=kind):
                cell.paint(bc.mech.Channel("Cl_Leak"))
                with self.assertRaisesRegex(KeyError, "No ion candidates are registered for family 'cl'."):
                    cell.init_state()


if __name__ == "__main__":
    unittest.main()
