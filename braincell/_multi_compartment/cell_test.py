"""Unit tests for :class:`braincell.Cell`."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell.mech as mech
from braincell import Branch, CVPerBranch, Cell, CurrentClamp, IonChannel, Morphology
from braincell.filter import AllRegion, RootLocation
from braincell.mech import StateProbe


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
        self.assertGreater(len(cell.node_tree.nodes), 0)
        self.assertGreater(len(cell.runtime_nodes), 0)
        self.assertGreater(len(cell.runtime_cvs), 0)
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
        self.assertGreater(len(cell.node_tree.nodes), 0)
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
        for method_name in ("sample_probes", "mech_table"):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, method_name)()
        for property_name in ("runtime_cvs", "runtime_nodes"):
            with self.subTest(property=property_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, property_name)

    def test_static_topology_is_available_before_init(self):
        cell = _cell_with_probe()
        self.assertGreater(len(cell.cvs), 0)
        self.assertGreater(len(cell.cv_tree.cvs), 0)
        self.assertGreater(len(cell.node_tree.nodes), 0)

    def test_runtime_views_bind_static_objects_after_init(self):
        cell = _cell_with_probe()
        cell.init_state()
        runtime_cv = cell.runtime_cvs[0]
        runtime_node = cell.runtime_nodes[0]
        self.assertIs(runtime_cv.declaration, cell.cvs[0])
        self.assertIs(runtime_node.declaration, cell.node_tree.nodes[0])
        self.assertIn("na", runtime_cv.ions)
        self.assertIn("na", runtime_node.ions)

    def test_nodes_query_api_is_restored_after_init(self):
        cell = _cell_with_probe()
        cell.init_state()
        nodes = cell.nodes(IonChannel, allowed_hierarchy=(1, 1))
        self.assertGreater(len(nodes), 0)

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

    def test_scalar_v_init_broadcasts_to_voltage_shape(self):
        cell = _simple_cell()
        cell.V_init = -60.0 * u.mV
        cell.init_state()
        self.assertEqual(cell.V.value.shape, (cell.n_cv,))
        self.assertTrue(u.math.allclose(cell.V.value, jnp.full((cell.n_cv,), -60.0) * u.mV, atol=1e-9 * u.mV))

    def test_run_supports_scalar_v_init(self):
        cell = _cell_with_probe()
        cell.V_init = -60.0 * u.mV
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertIn("V_root", result.traces)


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


if __name__ == "__main__":
    unittest.main()
