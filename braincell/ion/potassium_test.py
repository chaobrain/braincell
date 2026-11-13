# -*- coding: utf-8 -*-
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

from braincell._base_ion import Ion
from braincell._base_neuron import HHTypedNeuron
from braincell.channel.potassium import K_TM1991
from braincell.ion._base import InitNernstIon
from braincell.ion._testing import V as _V, DynamicShellIonContractTests, FixedIonContractTests, run_float64
from braincell.ion.potassium import (
    Potassium,
    PotassiumDetailed,
    PotassiumDetailedFH,
    PotassiumFixed,
    PotassiumInitNernst,
)


class PotassiumBaseTest(unittest.TestCase):
    """Tests for the abstract :class:`Potassium` base class."""

    def test_potassium_is_subclass_of_ion(self) -> None:
        self.assertTrue(issubclass(Potassium, Ion))

    def test_root_type_is_hh_typed_neuron(self) -> None:
        self.assertIs(Potassium.root_type, HHTypedNeuron)


class PotassiumFixedDefaultsTest(unittest.TestCase):
    """Defaults and parameter storage for :class:`PotassiumFixed`."""

    def test_custom_scalar_parameters_are_honoured(self) -> None:
        k = PotassiumFixed(size=2, E=-80.0 * u.mV, Ci=0.1 * u.mM, Co=3.0 * u.mM, valence=1)
        self.assertTrue(u.math.allclose(k.E, -80.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(k.Ci, 0.1 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.Co, 3.0 * u.mM, atol=1e-9 * u.mM))


class PotassiumFixedContainerTest(unittest.TestCase):
    """Ion-as-container behaviour."""

    def test_channels_kwarg_is_attached(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        self.assertIn("IK", k.channels)
        self.assertIsInstance(k.channels["IK"], K_TM1991)

    def test_current_with_channel_delegates_to_channel(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        i = k.current(V)
        self.assertEqual(i.shape, (1,))

    def test_current_with_include_external_adds_registered_fn(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        base = k.current(V)
        delta = jnp.array([2.5]) * u.uA / u.cm**2

        def external(V_local, info):
            return delta

        k.register_external_current("ext", external)
        total = k.current(V, include_external=True)
        got = (total - base).to_decimal(u.uA / u.cm**2)
        self.assertTrue(u.math.allclose(got, jnp.array([2.5]), atol=1e-9))


class PotassiumFixedLifecycleTest(unittest.TestCase):
    """init_state / reset_state / compute_derivative propagate to children.

    ``PotassiumFixed`` overrides ``reset_state`` with its own implementation
    that performs a hierarchy check plus per-channel ``reset_state`` calls –
    this suite exercises that override specifically.
    """

    def test_init_state_without_children_is_a_no_op(self) -> None:
        k = PotassiumFixed(size=1)
        k.init_state(_V([-60.0]))

    def test_reset_state_without_children_is_a_no_op(self) -> None:
        k = PotassiumFixed(size=1)
        # The override calls check_hierarchies which should succeed even with
        # zero channels.
        k.reset_state(_V([-60.0]))

    def test_init_state_initialises_child_channel_gate(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        ch = k.channels["IK"]
        # ``K_TM1991`` uses a single ``p`` gate for the n^4 formula.
        self.assertEqual(ch.p.value.shape, (1,))

    def test_reset_state_forwards_to_child(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        first = k.channels["IK"].p.value

        # Mutate and reset again – should recover exactly the same state.
        k.channels["IK"].p.value = jnp.array([0.999])
        k.reset_state(V)
        self.assertTrue(u.math.allclose(k.channels["IK"].p.value, first, atol=1e-9))

    def test_compute_derivative_populates_child_derivative(self) -> None:
        k = PotassiumFixed(size=1, IK=K_TM1991(size=1))
        V = _V([-60.0])
        k.init_state(V)
        k.reset_state(V)
        k.compute_derivative(V)
        self.assertEqual(k.channels["IK"].p.derivative.shape, (1,))


class PotassiumInitNernstTest(unittest.TestCase):
    def test_is_init_nernst_ion(self) -> None:
        self.assertTrue(issubclass(PotassiumInitNernst, Potassium))
        self.assertTrue(issubclass(PotassiumInitNernst, InitNernstIon))

    def test_E_is_initialized_from_nernst_on_init_and_reset(self) -> None:
        k = PotassiumInitNernst(size=1)
        V = _V([-65.0])

        self.assertIsNone(k.E)
        k.init_state(V)
        expected = u.gas_constant * k.temp / (k.valence * u.faraday_constant) * u.math.log(k.Co / k.Ci)
        self.assertTrue(u.math.allclose(k.E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        first_E = k.E
        k.Ci = jnp.array([10.0]) * u.mM
        self.assertTrue(u.math.allclose(k.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        k.reset_state(V)
        self.assertFalse(u.math.allclose(k.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_custom_Ci_and_Co_are_respected(self) -> None:
        k = PotassiumInitNernst(size=1, Ci=60.0 * u.mM, Co=3.0 * u.mM)
        self.assertTrue(u.math.allclose(k.Ci, 60.0 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(k.Co, 3.0 * u.mM, atol=1e-9 * u.mM))


class PotassiumFixedContractTest(FixedIonContractTests, unittest.TestCase):
    """The shared ``FixedIon`` contract, for potassium."""

    ION_CLASS = PotassiumFixed
    FAMILY_CLASS = Potassium
    DEFAULT_E = -95.0 * u.mV
    DEFAULT_CI = 54.4 * u.mM
    DEFAULT_CO = 2.5 * u.mM
    DEFAULT_VALENCE = 1


class PotassiumDetailedTest(DynamicShellIonContractTests, unittest.TestCase):
    """``PotassiumDetailed``: the shared shell-ion tests and its family defaults."""

    ION_CLASS = PotassiumDetailed
    LEAK = "K_Leak"
    INWARD_V_INIT = -100.0 * u.mV  # below E_K (-82 mV at the defaults)

    def test_defaults(self) -> None:
        ion = PotassiumDetailed(size=1)
        self.assertTrue(u.math.allclose(ion.Co, 2.5 * u.mM))
        self.assertTrue(u.math.allclose(ion.C_rest, 54.4 * u.mM))
        ion.init_state(_V([-65.0]))
        self.assertTrue(u.math.allclose(ion.Ci.value, 54.4 * u.mM))


# Botta et al. (2010) K_conc.mod at one point: ki 140, ko 5 mM, 23 degC, the
# .mod file's d = 0.2 um (0.4 um at z = 1) and beta = 1.3 /ms; a 1 mS/cm2
# potassium leak at V = -60 mV gives ik = 2.5038634e-2 mA/cm2 (outward).
_BOTTA_POINT = dict(
    temp=u.celsius2kelvin(23.0),
    d=0.4 * u.um,
    d_o=0.4 * u.um,
    beta=1.3 / u.ms,
    beta_o=1.3 / u.ms,
    C_rest=140.0 * u.mM,
    Co_rest=5.0 * u.mM,
    Ci_initializer=140.0 * u.mM,
    Co_initializer=5.0 * u.mM,
)
_BOTTA_I = -2.5038634e-2 * u.mA / u.cm**2


def _fh_cable_cell(V_init=-100.0 * u.mV, **ion_kwargs):
    """A one-CV cable Cell, L = diam = 15 um, with the FH ion ``k`` and ``K_Leak``."""
    import braincell as bc

    branch = bc.Branch.from_lengths(lengths=[15.0] * u.um, radii=[7.5, 7.5] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=V_init)
    cell.paint(bc.mech.Ion("PotassiumDetailedFH", name="k", **ion_kwargs))
    cell.paint(bc.mech.Channel("K_Leak", g_max=1.0 * u.mS / u.cm**2))
    cell.record("ci", bc.observe.ion(name="k").state("Ci"))
    cell.record("co", bc.observe.ion(name="k").state("Co"))
    return cell


def _fh_x64_report() -> dict:
    """Float64 measurements; ``_fh_float64`` runs it at float64."""
    V = _V([-60.0])
    ion = PotassiumDetailedFH(size=1, **_BOTTA_POINT)
    ion.init_state(V)
    rates = []
    for current in (_BOTTA_I, _BOTTA_I.to(u.amp / u.meter**2), _BOTTA_I.in_unit(u.uA / u.cm**2)):
        ion._cached_total_current = jnp.ones(1) * current
        ion.compute_derivative(V)
        rates.append([float(np.asarray(s.derivative.to_decimal(u.mM / u.ms))[0]) for s in (ion.Ci, ion.Co)])
    d, d_o = 1.0, 0.25  # um
    result = _fh_cable_cell(d=d * u.um, d_o=d_o * u.um).run(dt=0.025 * u.ms, duration=10.0 * u.ms)
    ci = np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0]
    co = np.asarray(result.samples["co"].values.to_decimal(u.mM))[:, 0]
    return {
        "E": float(np.asarray(ion.E.to_decimal(u.mV))[0]),
        "rates": rates,
        "content": (ci * d + co * d_o).tolist(),
        "co_change": float(co[-1] - co[0]),
    }


@functools.cache
def _fh_float64() -> dict:
    return run_float64(_fh_x64_report)


class PotassiumDetailedFHTest(DynamicShellIonContractTests, unittest.TestCase):
    """``PotassiumDetailedFH``: the shell-ion tests plus the extracellular ``Co`` state."""

    ION_CLASS = PotassiumDetailedFH
    LEAK = "K_Leak"
    INWARD_V_INIT = -100.0 * u.mV  # below E_K (-82 mV at the defaults)

    def test_defaults(self) -> None:
        ion = PotassiumDetailedFH(size=1)
        for name, value in (("C_rest", 54.4 * u.mM), ("Co_rest", 2.5 * u.mM), ("d", 1.0 * u.um), ("d_o", 1.0 * u.um)):
            self.assertTrue(u.math.allclose(getattr(ion, name), value), name)
        ion.init_state(_V([-65.0]))
        self.assertTrue(u.math.allclose(ion.Ci.value, 54.4 * u.mM))
        self.assertTrue(u.math.allclose(ion.Co.value, 2.5 * u.mM))
        self.assertEqual(ion.Co.value.shape, (1,))

    def test_botta_point_rates_and_reversal(self) -> None:
        report = _fh_float64()
        np.testing.assert_allclose(report["E"], -85.038634, rtol=1e-7)
        # -ik / (2 F 0.2 um): the outward current lowers Ci and raises Co by as much.
        np.testing.assert_allclose(report["rates"], [[-6.4876788e-3, 6.4876788e-3]] * 3, rtol=1e-7)

    def test_content_is_conserved_without_relaxation(self) -> None:
        report = _fh_float64()
        content = np.asarray(report["content"])
        self.assertLess(report["co_change"], -1e-3)  # the inward leak current drains the space
        np.testing.assert_allclose(content, np.full_like(content, content[0]), rtol=1e-10)

    def test_co_is_recorded_and_finite_in_a_cell(self) -> None:
        result = _fh_cable_cell().run(dt=0.025 * u.ms, duration=5.0 * u.ms)
        co = np.asarray(result.samples["co"].values.to_decimal(u.mM))[:, 0]
        ci = np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0]
        self.assertTrue(np.all(np.isfinite(co)) and np.all(np.isfinite(ci)))
        self.assertTrue(np.all(np.diff(co) <= 0.0) and co[-1] < co[0] < 2.5 + 1e-6)

    def test_reset_state_restores_co_to_its_initializer(self) -> None:
        V = _V([-65.0, -60.0])
        ion = PotassiumDetailedFH(size=2, Co_initializer=4.0 * u.mM)
        ion.init_state(V)
        ion.Co.value = jnp.array([7.0, 8.0]) * u.mM
        ion.reset_state(V)
        np.testing.assert_allclose(np.asarray(ion.Co.value.to_decimal(u.mM)), [4.0, 4.0])
        ion.reset_state(V, batch_size=3)
        self.assertEqual(ion.Co.value.shape, (3, 2))

    def test_declared_co_initializer_reaches_the_state_in_a_cell(self) -> None:
        cell = _fh_cable_cell(V_init=-82.0 * u.mV, Co_initializer=4.0 * u.mM, Co_rest=4.0 * u.mM, beta_o=0.5 / u.ms)
        result = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
        co = np.asarray(result.samples["co"].values.to_decimal(u.mM))[:, 0]
        np.testing.assert_allclose(co[0], 4.0, rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(co)))


if __name__ == "__main__":
    unittest.main()
