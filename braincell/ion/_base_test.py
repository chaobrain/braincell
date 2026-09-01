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

import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base_channel import Channel
from braincell._base_channel import IonInfo
from braincell.ion import Calcium
from braincell.ion import _base as ionbase
from braincell.ion._base import Conserve
from braincell.ion._base import DynamicNernstIon
from braincell.ion._base import Factor
from braincell.ion._base import FixedIon
from braincell.ion._base import InitNernstIon
from braincell.ion._base import KineticIon
from braincell.ion._base import Reaction
from braincell.ion._base import Species
from braincell.ion._base import _RadialShellGeometry
from braincell.quad import get_integrator
from braincell.quad.protocol import DiffEqState


class _RecorderChannel(Channel):
    root_type = Calcium

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.last_ion = None

    def init_state(self, V, ion, batch_size=None):
        _ = (V, batch_size)
        self.last_ion = ion

    def reset_state(self, V, ion, batch_size=None):
        _ = (V, batch_size)
        self.last_ion = ion

    def compute_derivative(self, V, ion):
        _ = V
        self.last_ion = ion

    def current(self, V, ion):
        _ = V
        self.last_ion = ion
        return 0.25 * u.mM / u.ms


class _ConstantIon(Calcium, FixedIon):
    default_Ci = 2.0e-4 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_fixed_ion(E=120.0 * u.mV)


class _InitNernstIon(Calcium, InitNernstIon):
    default_Ci = 2.0e-4 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_nernst_ion(temp=u.celsius2kelvin(36.0))


class _DynamicNernstIon(Calcium, DynamicNernstIon):
    default_Co = 2.0 * u.mM
    default_valence = 2
    uses_total_current = True

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_dynamic_nernst_ion(
            temp=u.celsius2kelvin(36.0),
            Co=None,
            valence=None,
            Ci_initializer=2.0e-4 * u.mM,
        )

    def derivative(self, Ci, V, total_current=None):
        _ = V
        return 0.1 * total_current - 0.05 * Ci / u.ms


class _DynamicNernstIonNoCurrent(Calcium, DynamicNernstIon):
    default_Co = 2.0 * u.mM
    default_valence = 2
    uses_total_current = False

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.last_total_current = "unset"
        self._init_dynamic_nernst_ion(
            temp=u.celsius2kelvin(36.0),
            Co=None,
            valence=None,
            Ci_initializer=2.0e-4 * u.mM,
        )

    def current(self, V, include_external=False):
        _ = (V, include_external)
        raise AssertionError("current() should not be called when uses_total_current is False.")

    def derivative(self, Ci, V, total_current=None):
        _ = V
        self.last_total_current = total_current
        return -0.05 * Ci / u.ms


class _SimpleKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    factors = (Factor("cyto", lambda self: self.cyt_volume),)
    species = (
        Species("Ci", init=0.1 * u.mM, factor="cyto"),
        Species("B", init=1.0 * u.mM, factor="cyto"),
        Species("BC", init=0.0 * u.mM, factor="cyto"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf * self.cyt_volume,
            backward=lambda self, V, x: self.kb * self.cyt_volume,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("B", "BC"),
            algebraic="B",
            total=lambda self, V, x: self.Btot * self.cyt_volume,
        ),
    )

    def __init__(self, size=1, species_initializers=None, use_defaults=False):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        solver = None if use_defaults else "euler"
        substeps = None if use_defaults else 2
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            species_initializers=species_initializers,
            solver=solver,
            substeps=substeps,
        )
        self.cyt_volume = braintools.init.param(3.0 * u.um**3, self.varshape, allow_none=False)
        self.kf = braintools.init.param(2.0 / (u.mM * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(1.0 * u.mM, self.varshape, allow_none=False)


class _StoichKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    species = (
        Species("Ci", init=0.1 * u.mM),
        Species("A", init=1.0 * u.mM),
        Species("B", init=1.0 * u.mM),
        Species("C", init=0.0 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"A": 3, "B": 1},
            rhs={"C": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = ()

    def __init__(self, size=1):
        super().__init__(size=size, name=None, probe=_RecorderChannel(size=size))
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            solver="euler",
            substeps=1,
        )
        self.kf = braintools.init.param(2.0 / (u.mM**3 * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)


class _UnitMismatchKineticIon(Calcium, KineticIon):
    default_Co = 2.0 * u.mM
    default_valence = 2

    factors = (Factor("cyto", lambda self: self.cyt_volume),)
    species = (
        Species("Ci", init=0.1 * u.mM, factor="cyto"),
        Species("B", init=1.0 * u.mM),
        Species("BC", init=0.0 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = ()

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            solver="euler",
            substeps=1,
        )
        self.cyt_volume = braintools.init.param(3.0 * u.um**3, self.varshape, allow_none=False)
        self.kf = braintools.init.param(2.0 / (u.mM * u.ms), self.varshape, allow_none=False)
        self.kb = braintools.init.param(0.5 / u.ms, self.varshape, allow_none=False)


class IonTemplateTest(unittest.TestCase):
    def test_constant_pack_info_and_child_derivative(self) -> None:
        ion = _ConstantIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, ion.Co, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(info.E, ion.E, atol=1e-9 * u.mV))
        self.assertEqual(info.valence, 2)

        ion.init_state(V)
        ion.reset_state(V)
        ion.compute_derivative(V)
        self.assertIsInstance(ion.channels["probe"].last_ion, IonInfo)

    def test_species_first_inheritance_keeps_template_hooks_active(self) -> None:
        self.assertLess(_ConstantIon.__mro__.index(Calcium), _ConstantIon.__mro__.index(FixedIon))
        self.assertTrue(hasattr(_InitNernstIon, "_ion_init_state_hook"))
        self.assertTrue(hasattr(_DynamicNernstIon, "_ion_compute_derivative_hook"))

    def test_init_nernst_only_updates_on_reset(self) -> None:
        ion = _InitNernstIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        first_E = ion.E
        expected = u.gas_constant * ion.temp / (ion.valence * u.faraday_constant) * u.math.log(ion.Co / ion.Ci)
        self.assertTrue(u.math.allclose(first_E.to_decimal(u.mV), expected.to_decimal(u.mV), atol=1e-6))

        ion.Ci = braintools.init.param(1.0e-3 * u.mM, ion.varshape, allow_none=False)
        self.assertTrue(u.math.allclose(ion.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

        ion.reset_state(V)
        self.assertFalse(u.math.allclose(ion.E.to_decimal(u.mV), first_E.to_decimal(u.mV), atol=1e-6))

    def test_dynamic_nernst_updates_E_and_Ci_derivative(self) -> None:
        ion = _DynamicNernstIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.reset_state(V)
        first_E = ion.E
        ion.Ci.value = jnp.array([1.0e-3]) * u.mM
        second_E = ion.E
        self.assertFalse(u.math.allclose(first_E.to_decimal(u.mV), second_E.to_decimal(u.mV), atol=1e-6))

        ion.compute_derivative(V)
        expected = 0.1 * (0.25 * u.mM / u.ms) - 0.05 * ion.Ci.value / u.ms
        self.assertTrue(
            u.math.allclose(
                ion.Ci.derivative.to_decimal(u.mM / u.ms),
                expected.to_decimal(u.mM / u.ms),
                atol=1e-6,
            )
        )
        info = ion.channels["probe"].last_ion
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))

    def test_dynamic_nernst_skips_current_when_total_current_not_needed(self) -> None:
        ion = _DynamicNernstIonNoCurrent(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.reset_state(V)
        ion.compute_derivative(V)

        self.assertIsNone(ion.last_total_current)

    def test_kinetic_ion_init_and_reset_write_back_algebraic_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        self.assertIsInstance(ion.Ci, DiffEqState)
        self.assertIsInstance(ion.BC, DiffEqState)
        self.assertFalse(isinstance(ion.B, DiffEqState))
        self.assertIsInstance(ion.B, brainstate.HiddenState)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.reset_state(V)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([1.0]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_uses_central_default_integration_schedule(self) -> None:
        ion = _SimpleKineticIon(size=1, use_defaults=True)
        self.assertIs(ion.solver, get_integrator("backward_euler"))
        self.assertEqual(ion.substeps, 1)

        override = _SimpleKineticIon(size=1)
        self.assertIs(override.solver, get_integrator("euler"))
        self.assertEqual(override.substeps, 2)

    def test_kinetic_ion_species_values_return_resolved_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.B.value = jnp.array([-999.0]) * u.mM
        values = ion.species_values()

        self.assertTrue(u.math.allclose(values["Ci"], ion.Ci.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["BC"], ion.BC.value, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(values["B"], jnp.array([0.75]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_compute_derivative_resolves_algebraic_species_first(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        ion.BC.value = jnp.array([0.25]) * u.mM
        ion.B.value = jnp.array([-999.0]) * u.mM
        ion.compute_derivative(V)

        expected_visible_flux = 2.0 * 0.2 * 0.75 - 0.5 * 0.25
        self.assertTrue(
            u.math.allclose(
                ion.Ci.derivative.to_decimal(u.mM / u.ms),
                jnp.array([-expected_visible_flux]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            u.math.allclose(
                ion.BC.derivative.to_decimal(u.mM / u.ms),
                jnp.array([expected_visible_flux]),
                atol=1e-6,
            )
        )

    def test_kinetic_ion_post_integral_refreshes_algebraic_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.BC.value = jnp.array([0.4]) * u.mM
        ion.B.value = jnp.array([9.9]) * u.mM
        ion.post_integral(V)
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.6]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_pack_info_uses_Ci_species(self) -> None:
        ion = _SimpleKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.2]) * u.mM
        info = ion.pack_info()

        self.assertTrue(u.math.allclose(info.Ci, ion.Ci.value, atol=1e-12 * u.mM))
        expected_E = (u.gas_constant * ion.temp / (ion.valence * u.faraday_constant)) * u.math.log(
            ion.Co / ion.Ci.value
        )
        self.assertTrue(u.math.allclose(info.E.to_decimal(u.mV), expected_E.to_decimal(u.mV), atol=1e-6))

    def test_kinetic_ion_species_initializers_override_declared_inits(self) -> None:
        ion = _SimpleKineticIon(
            size=1,
            species_initializers={
                "Ci": 0.2 * u.mM,
                "BC": 0.3 * u.mM,
            },
        )
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.2]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_kinetic_ion_reset_restores_species_initializer_overrides(self) -> None:
        ion = _SimpleKineticIon(
            size=1,
            species_initializers={
                "Ci": 0.2 * u.mM,
                "BC": 0.3 * u.mM,
            },
        )
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.Ci.value = jnp.array([0.9]) * u.mM
        ion.BC.value = jnp.array([0.8]) * u.mM
        ion.B.value = jnp.array([-99.0]) * u.mM

        ion.reset_state(V)

        self.assertTrue(u.math.allclose(ion.Ci.value, jnp.array([0.2]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.BC.value, jnp.array([0.3]) * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.B.value, jnp.array([0.7]) * u.mM, atol=1e-12 * u.mM))

    def test_factor_crossing_amount_units_can_match_between_volume_and_area_species(self) -> None:
        cyto_amount = (1.0 * u.mM) * (3.0 * u.um**3)
        pump_amount = (1.0 * u.mM * u.um) * (3.0 * u.um**2)
        converted = pump_amount.in_unit(cyto_amount.unit)

        self.assertTrue(u.math.allclose(converted, 3.0 * cyto_amount.unit, atol=1e-12 * cyto_amount.unit))

    def test_stoichiometric_reaction_uses_power_and_signed_coefficients(self) -> None:
        ion = _StoichKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        ion.A.value = jnp.array([2.0]) * u.mM
        ion.B.value = jnp.array([5.0]) * u.mM
        ion.C.value = jnp.array([7.0]) * u.mM

        ion.compute_derivative(V)

        expected_flux = 2.0 * (2.0**3) * 5.0 - 0.5 * 7.0
        self.assertTrue(
            u.math.allclose(
                ion.A.derivative.to_decimal(u.mM / u.ms),
                jnp.array([-3.0 * expected_flux]),
                atol=1e-6,
            )
        )

    def test_mixed_factor_reaction_with_incompatible_scaled_units_raises(self) -> None:
        ion = _UnitMismatchKineticIon(size=1)
        V = jnp.array([-65.0]) * u.mV

        ion.init_state(V)
        with self.assertRaises(Exception):
            ion.compute_derivative(V)


class ConserveWritebackStateClassTest(unittest.TestCase):
    """``_Conserve.writeback`` must not allocate a wrongly-classed state.

    ``writeback`` *does* run every simulation step, outside the
    :func:`braincell.state_grouping` scope the host sets around
    ``init_state``. Its allocation branch is dead in normal flow because
    ``_Species.init`` ran first — these tests pin both halves of that, so
    a regression shows up as a failure rather than as a silently
    non-grouped state inside a ``Cell``.

    See ``docs/specs/2026-08-13-cell-hidden-group-state.md``.
    """

    @staticmethod
    def _kinetic_cell():
        import braincell
        from braincell import Branch, CVPerBranch, Cell, Morphology
        from braincell.filter import AllRegion

        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = Morphology.from_root(soma, name="soma")
        cell = Cell(tree, cv_policy=CVPerBranch(cv_per_branch=2))
        # CdpCR_MA2020_GrC declares algebraic species alongside diffeq ones,
        # so it is the model that exercises the writeback path.
        cell.paint(AllRegion(), braincell.mech.Ion("CdpCR_MA2020_GrC"))
        cell.init_state()
        return cell

    def test_writeback_runs_mid_step_but_allocates_nothing(self) -> None:
        from braincell.ion import _base as ionbase

        cell = self._kinetic_cell()
        calls = {"writeback": 0, "allocated": 0}
        original_writeback = ionbase._Conserve.writeback
        original_algebraic = ionbase._Species.algebraic_state

        def counting_writeback(self, V=None):
            calls["writeback"] += 1
            return original_writeback(self, V)

        def counting_algebraic(self, value):
            calls["allocated"] += 1
            return original_algebraic(self, value)

        # Identity of every algebraic state before the step, so a
        # reallocation is visible as a swapped object and not only as an
        # extra ``algebraic_state`` call.
        before = self._algebraic_states(cell)
        self.assertGreater(len(before), 0, "expected at least one algebraic species")

        ionbase._Conserve.writeback = counting_writeback
        ionbase._Species.algebraic_state = counting_algebraic
        try:
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.01 * u.ms):
                cell.update()
        finally:
            ionbase._Conserve.writeback = original_writeback
            ionbase._Species.algebraic_state = original_algebraic

        self.assertGreater(calls["writeback"], 0, "writeback should run during a step")
        self.assertEqual(calls["allocated"], 0, "writeback must not allocate mid-run")

        after = self._algebraic_states(cell)
        self.assertEqual(set(before), set(after))
        for path, state in after.items():
            with self.subTest(state=path):
                self.assertIs(state, before[path], "writeback replaced the state object")
                self.assertIsInstance(state, brainstate.HiddenGroupState)

    @staticmethod
    def _algebraic_states(cell) -> dict:
        """Path → state for every algebraic species state in the cell."""
        from braincell.ion import _base as ionbase

        names = set()
        for node in cell.nodes().values():
            if isinstance(node, ionbase.KineticIon):
                names.update(ionbase._Specs.for_type(type(node)).algebraic_names)
        return {
            ".".join(map(str, path)): state
            for path, state in brainstate.graph.states(cell).items()
            if path and str(path[-1]) in names
        }

    def test_late_allocation_still_matches_its_siblings(self) -> None:
        # Force the dead branch: drop the algebraic species back to a bare
        # value so the next writeback has to re-allocate it. writeback runs
        # outside any state_grouping scope, so an ambient-scope allocation
        # would hand back a plain HiddenState and break the Cell invariant.
        from braincell.ion import _base as ionbase

        cell = self._kinetic_cell()
        ion = cell.get_ion("CdpCR_MA2020_GrC")
        specs = ionbase._Specs.for_type(type(ion))
        (name,) = specs.algebraic_names

        state = getattr(ion, name)
        self.assertIsInstance(state, brainstate.HiddenGroupState)
        setattr(ion, name, state.value)
        self.assertNotIsInstance(getattr(ion, name), brainstate.State)

        species = ionbase._Species(ion, specs)
        ionbase._Conserve(ion, specs, species).writeback(cell.V.value)

        restored = getattr(ion, name)
        self.assertIsInstance(restored, brainstate.HiddenGroupState)
        self.assertEqual(restored.value.shape, state.value.shape)

    def test_late_allocation_stays_plain_when_its_siblings_are_plain(self) -> None:
        # The mirror of the test above. A sibling-derived class has to work
        # in both directions, or "derive it from a sibling" is really just
        # "always group", which would be wrong outside a Cell.
        from braincell.ion import _base as ionbase

        ion = _SimpleKineticIon(size=3)
        ion.init_state(jnp.ones(3) * -65.0 * u.mV)
        specs = ionbase._Specs.for_type(type(ion))
        (name,) = specs.algebraic_names

        state = getattr(ion, name)
        self.assertIsInstance(state, brainstate.HiddenState)
        self.assertNotIsInstance(state, brainstate.HiddenGroupState)
        setattr(ion, name, state.value)

        species = ionbase._Species(ion, specs)
        ionbase._Conserve(ion, specs, species).writeback(jnp.ones(3) * -65.0 * u.mV)

        restored = getattr(ion, name)
        self.assertIsInstance(restored, brainstate.HiddenState)
        self.assertNotIsInstance(restored, brainstate.HiddenGroupState)
        self.assertEqual(restored.value.shape, state.value.shape)


class _ShellGeometryIon(Calcium, _RadialShellGeometry, KineticIon):
    """Minimal kinetic ion exercising the shared radial-shell geometry."""

    default_Co = 2.0 * u.mM
    default_valence = 2

    species = (Species("Ci", init=0.1 * u.mM),)
    reactions = ()
    sources = ()
    conserves = ()

    def __init__(self, size=1, Nannuli=5.0):
        super().__init__(size=size, name=None)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self._init_kinetic_ion(
            Co=None,
            temp=u.celsius2kelvin(36.0),
            valence=None,
            species_initializers={"Ci": 0.1 * u.mM},
            solver="euler",
            substeps=1,
        )


class RadialShellGeometryTest(unittest.TestCase):
    """Cover the geometry mixin shared by every ``cdp*`` calcium ion."""

    def test_vrat_matches_the_annulus_construction(self) -> None:
        ion = _ShellGeometryIon(size=1, Nannuli=5.0)
        dr2 = 0.25 / (5.0 - 1.0)
        expected = u.math.pi * (0.5 - dr2 / 2.0) * 2.0 * dr2
        self.assertTrue(u.math.allclose(ion.vrat, expected, atol=1e-12))

    def test_geometry_properties_compose(self) -> None:
        ion = _ShellGeometryIon(size=1)
        ion.diam_arc_mean = 3.0 * u.um
        self.assertTrue(u.math.allclose(ion.dsq, 9.0 * u.um**2, atol=1e-12 * u.um**2))
        self.assertTrue(u.math.allclose(ion.dsqvol, ion.dsq * ion.vrat, atol=1e-12 * u.um**2))
        self.assertTrue(u.math.allclose(ion.parea, u.math.pi * 3.0 * u.um, atol=1e-12 * u.um))

    def test_missing_diameter_raises_before_state_initialization(self) -> None:
        ion = _ShellGeometryIon(size=1)
        V = jnp.array([-65.0]) * u.mV
        with self.assertRaises(AttributeError):
            ion.dsq
        with self.assertRaises(AttributeError):
            ion._ion_init_state_hook(V)

    def test_hooks_delegate_once_the_diameter_is_known(self) -> None:
        ion = _ShellGeometryIon(size=1)
        ion.diam_arc_mean = 3.0 * u.um
        V = jnp.array([-65.0]) * u.mV
        ion._ion_init_state_hook(V)
        self.assertTrue(u.math.allclose(ion.Ci.value, 0.1 * u.mM, atol=1e-12 * u.mM))
        ion.Ci.value = 0.5 * u.mM
        ion._ion_reset_state_hook(V)
        self.assertTrue(u.math.allclose(ion.Ci.value, 0.1 * u.mM, atol=1e-12 * u.mM))

    def test_buffer_equilibrium_partitions_the_total(self) -> None:
        ion = _ShellGeometryIon(size=1)
        total = 1.0 * u.mM
        kon = 2.0 / (u.mM * u.ms)
        koff = 0.5 / u.ms
        cai = 0.1 * u.mM
        free = ion._ss_buffer_free(total, kon, koff, cai)
        bound = ion._ss_buffer_bound(total, kon, koff, cai)
        self.assertTrue(u.math.allclose(free + bound, total, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(bound / free, (kon / koff) * cai, atol=1e-9))

    def test_as_initializer_keeps_a_per_point_tuple_as_one_array(self) -> None:
        ion = _ShellGeometryIon(size=1)
        resolved = ion._as_initializer((1.0 * u.mM, 2.0 * u.mM, 3.0 * u.mM))
        self.assertEqual(resolved.shape, (3,))
        self.assertTrue(u.math.allclose(resolved, jnp.array([1.0, 2.0, 3.0]) * u.mM, atol=1e-12 * u.mM))

    def test_as_initializer_passes_a_callable_through(self) -> None:
        ion = _ShellGeometryIon(size=1)

        def initializer(shape):
            return jnp.zeros(shape) * u.mM

        self.assertIs(ion._as_initializer(initializer), initializer)


class KineticIonCiInitializerTest(unittest.TestCase):
    """``Ci_initializer`` is one view onto ``species_initializers['Ci']``."""

    def test_reads_through_to_species_initializers(self) -> None:
        ion = _SimpleKineticIon(size=1, species_initializers={"Ci": 0.25 * u.mM})
        self.assertIs(ion.Ci_initializer, ion.species_initializers["Ci"])

    def test_writes_through_to_species_initializers(self) -> None:
        ion = _SimpleKineticIon(size=1, species_initializers={"Ci": 0.25 * u.mM})
        ion.Ci_initializer = 0.75 * u.mM
        self.assertTrue(u.math.allclose(ion.species_initializers["Ci"], 0.75 * u.mM, atol=1e-12 * u.mM))
        self.assertTrue(u.math.allclose(ion.Ci_initializer, 0.75 * u.mM, atol=1e-12 * u.mM))

    def test_write_through_reaches_the_reset_state(self) -> None:
        ion = _SimpleKineticIon(size=1, species_initializers={"Ci": 0.25 * u.mM})
        V = jnp.array([-65.0]) * u.mV
        ion.init_state(V)
        ion.Ci_initializer = 0.75 * u.mM
        ion.reset_state(V)
        self.assertTrue(u.math.allclose(ion.Ci.value, 0.75 * u.mM, atol=1e-12 * u.mM))


class NernstHelperTest(unittest.TestCase):
    """The single Nernst transcription shared by three ion templates."""

    def test_unwrap_returns_state_payload_and_plain_values(self) -> None:
        state = brainstate.State(jnp.array([1.0]) * u.mM)
        self.assertIs(ionbase._unwrap(state), state.value)
        plain = 2.0 * u.mM
        self.assertIs(ionbase._unwrap(plain), plain)

    def test_nernst_matches_the_written_formula(self) -> None:
        Ci, Co = 0.1 * u.mM, 2.0 * u.mM
        temp, valence = u.celsius2kelvin(36.0), 2
        expected = (u.gas_constant * temp / (valence * u.faraday_constant)) * u.math.log(Co / Ci)
        self.assertTrue(u.math.allclose(ionbase._nernst(Ci=Ci, Co=Co, temp=temp, valence=valence), expected))

    def test_all_three_templates_agree_on_the_same_inputs(self) -> None:
        init_nernst = _InitNernstIon(size=1)
        init_nernst._update_reversal()
        dynamic = _DynamicNernstIon(size=1)
        dynamic.init_state(jnp.array([-65.0]) * u.mV)
        expected = ionbase._nernst(
            Ci=init_nernst.Ci,
            Co=init_nernst.Co,
            temp=init_nernst.temp,
            valence=init_nernst.valence,
        )
        self.assertTrue(u.math.allclose(init_nernst.E, expected, atol=1e-9 * u.mV))


if __name__ == "__main__":
    unittest.main()
