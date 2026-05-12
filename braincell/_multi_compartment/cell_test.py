"""Unit tests for :class:`braincell.Cell`."""

import unittest

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell.mech as mech
from braincell import Branch, CVPerBranch, Cell, CurrentClamp, Morphology
from braincell.filter import AllRegion, RootLocation
from braincell.mech import register_channel, register_ion
from braincell.mech import StateProbe
from braincell.ion import Calcium
from braincell.ion._base import KineticIon, Species
from braincell.quad.protocol import DiffEqState


def _simple_cell() -> Cell:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())


def _cell_with_probe() -> Cell:
    cell = _simple_cell()
    cell.place(RootLocation(0.0), StateProbe(field="v", name="V_root"))
    return cell


class TestCellDeclaration(unittest.TestCase):
    def test_constructs_with_defaults(self):
        cell = _simple_cell()
        self.assertGreater(len(cell.paint_rules), 0)
        self.assertEqual(len(cell.place_rules), 0)
        self.assertFalse(cell._initialized)

    def test_rejects_non_morphology(self):
        with self.assertRaises(TypeError):
            Cell("not-a-morpho")  # type: ignore[arg-type]

    def test_cv_policy_mutation_invalidates_cache(self):
        cell = _simple_cell()
        _ = cell.cvs
        cell.cv_policy = CVPerBranch()
        _ = cell.cvs

    def test_paint_returns_self_for_chaining(self):
        cell = _simple_cell()
        from braincell import CableProperty
        result = cell.paint(
            cell.paint_rules[0].region,
            CableProperty(
                resting_potential=-70 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm ** 2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
        )
        self.assertIs(result, cell)

    def test_place_dedups_identical_rules(self):
        cell = _simple_cell()
        clamp = CurrentClamp.step(0.1 * u.nA, 10 * u.ms, delay=1 * u.ms)
        cell.place(RootLocation(0.0), clamp)
        cell.place(RootLocation(0.0), clamp)
        self.assertEqual(len(cell.place_rules), 1)


class TestCellLifecycle(unittest.TestCase):

    def test_declaration_phase_flag(self):
        cell = _cell_with_probe()
        self.assertFalse(cell._initialized)

    def test_init_state_flips_flag_and_populates_runtime(self):
        cell = _cell_with_probe()
        cell.init_state()
        self.assertTrue(cell._initialized)
        self.assertIsNotNone(cell._runtime)
        self.assertIsNotNone(cell._point_tree)
        self.assertIsNotNone(cell._axial_jax)
        self.assertTrue(hasattr(cell, "V"))
        self.assertTrue(hasattr(cell, "spike"))

    def test_init_state_twice_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
            cell.init_state()

    def test_paint_after_init_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV))

    def test_place_after_init_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.place(RootLocation(0.5), StateProbe(field="v", name="V_mid"))

    def test_config_setters_after_init_raise(self):
        cell = _cell_with_probe()
        cell.init_state()
        for name, value in (
            ("V_th", -60 * u.mV),
            ("V_init", -65 * u.mV),
            ("cv_policy", CVPerBranch()),
            ("solver", "staggered"),
        ):
            with self.subTest(attr=name):
                with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
                    setattr(cell, name, value)

    def test_reset_from_declaring_raises(self):
        cell = _cell_with_probe()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.reset()

    def test_reset_round_trip(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.reset()
        self.assertFalse(cell._initialized)
        self.assertIsNone(cell._runtime)
        self.assertIsNone(cell._point_tree)
        self.assertIsNone(cell._axial_jax)
        self.assertFalse(hasattr(cell, "V"))
        self.assertFalse(hasattr(cell, "spike"))

        # Paint after reset works.
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV))
        cell.init_state()
        self.assertTrue(cell._initialized)

    def test_reset_restores_scalar_V_th(self):
        cell = _cell_with_probe()
        original_V_th = cell.V_th
        cell.init_state()
        # After init V_th has been vectorised by install_cell_runtime.
        self.assertNotEqual(cell.V_th.shape if hasattr(cell.V_th, "shape") else (), ())
        cell.reset()
        # Back to scalar declaration value.
        self.assertEqual(cell.V_th, original_V_th)

    def test_runtime_method_requires_init(self):
        cell = _cell_with_probe()
        for method_name in (
            "sample_probes",
            "mech_table",
            "point_tree",
        ):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, method_name)()

    def test_run_auto_inits_from_declaring(self):
        cell = _cell_with_probe()
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertTrue(cell._initialized)
        self.assertIn("V_root", result.traces)

    def test_run_does_not_reinit(self):
        cell = _cell_with_probe()
        cell.init_state()
        first_runtime = cell._runtime
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertIs(cell._runtime, first_runtime)

    def test_axial_operator_cache_tracks_precision(self):
        cell = _simple_cell()
        with brainstate.environ.context(precision=32):
            cell.init_state()
            cache32 = cell.runtime.axial_operator_cache
            self.assertEqual(cell._axial_jax.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell.runtime.axial_operator_np.dtype, np.float64)
            self.assertIsNotNone(cache32)

        with brainstate.environ.context(precision=64):
            operator64 = cell._get_axial_operator()
            cache64 = cell.runtime.axial_operator_cache
            self.assertEqual(operator64.dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)


class CellDoesNotAllocatePlaceholderIonsEagerlyTest(unittest.TestCase):
    """MED-09: Cell.__init__ must not allocate a throwaway ion container."""

    def test_build_placeholder_ions_not_called_in_init(self) -> None:
        from unittest.mock import patch

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        with patch(
            "braincell._multi_compartment.cell.build_placeholder_ions",
            side_effect=AssertionError(
                "placeholder must not be called at __init__"
            ),
        ):
            _ = Cell(tree)


_PHASE_LOG = []


@register_channel("__phase_channel__")
class _PhaseChannel(braincell.Channel):
    root_type = Calcium

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self.p = DiffEqState(jnp.zeros(self.varshape))

    def init_state(self, V, ion, batch_size=None):
        _ = (V, ion, batch_size)

    def reset_state(self, V, ion, batch_size=None):
        _ = (V, ion, batch_size)
        self.p.value = jnp.zeros(self.varshape)

    def compute_derivative(self, V, ion):
        self.p.derivative = u.math.zeros_like(self.p.value) / u.ms
        _PHASE_LOG.append("channel_compute")

    def current(self, V, ion):
        _ = V
        return ion.Ci * (u.nA / (u.cm ** 2) / u.mM)


@register_ion("__phase_ion__")
class _PhaseIon(Calcium, KineticIon):
    uses_total_current = True
    species = (
        Species("Ci", init=1.0 * u.mM),
    )

    def __init__(self, size, name=None, **channels):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=2.0 * u.mM,
            temp=u.celsius2kelvin(36.0),
            valence=2,
            solver="euler",
            substeps=1,
        )
        self.seen_total_current = None

    def _ion_compute_derivative_hook(self, V):
        self.seen_total_current = self._cached_total_current
        self.Ci.derivative = u.math.zeros_like(self.Ci.value) / u.ms
        _PHASE_LOG.append("ion_compute")


class TestCellFamilyPhasedUpdate(unittest.TestCase):
    def setUp(self) -> None:
        _PHASE_LOG.clear()

    def _build_family_phased_cell(self) -> Cell:
        cell = _simple_cell()
        region = AllRegion()
        from braincell import CableProperty

        cell.paint(
            region,
            CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm ** 2,
                axial_resistivity=100.0 * u.ohm * u.cm,
                temperature=u.celsius2kelvin(36.0),
            ),
        )
        cell.paint(region, mech.Ion("__phase_ion__", name="ca_dyn"))
        cell.paint(region, mech.Channel("__phase_channel__", ion_name="ca_dyn", name="ca_dep"))
        return cell

    def test_default_update_policy_is_legacy(self):
        cell = _simple_cell()
        self.assertEqual(cell.update_policy, "legacy")

    def test_rejects_unknown_update_policy(self):
        with self.assertRaises(ValueError):
            Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), update_policy="bad")

    def test_family_phased_channel_runs_before_ion(self):
        cell = self._build_family_phased_cell()
        cell.update_policy = "family_phased"
        cell.init_state()
        ion = cell.get_ion("ca_dyn")
        channel = ion.channels["ca_dep"]
        ion.Ci.value = jnp.asarray([1.0]) * u.mM
        with brainstate.environ.context(dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(_PHASE_LOG, ["channel_compute", "ion_compute"])

    def test_family_phased_ion_uses_cached_current(self):
        cell = self._build_family_phased_cell()
        cell.update_policy = "family_phased"
        cell.init_state()
        ion = cell.get_ion("ca_dyn")
        channel = ion.channels["ca_dep"]
        ion.Ci.value = jnp.asarray([1.0]) * u.mM
        channel.p.value = jnp.asarray([999.0])
        with brainstate.environ.context(dt=0.1 * u.ms):
            cell.update()

        expected = jnp.asarray([1.0]) * (u.nA / (u.cm ** 2))
        self.assertTrue(
            u.math.allclose(
                ion.seen_total_current.to_decimal(u.nA / (u.cm ** 2)),
                expected.to_decimal(u.nA / (u.cm ** 2)),
                atol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
