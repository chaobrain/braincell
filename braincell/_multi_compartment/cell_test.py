"""Unit tests for :class:`braincell.Cell`."""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell.mech as mech
from braincell import Branch, CVPerBranch, Cell, Channel, CurrentClamp, Ion, IonChannel, Morphology
from braincell.filter import AllRegion, RootLocation
from braincell.mech import StateProbe
from braincell.quad import DiffEqState, get_integrator


def _soma_tree() -> Morphology:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def _simple_cell() -> Cell:
    return Cell(_soma_tree(), cv_policy=CVPerBranch())


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

    def test_subsolver_defaults_to_backward_euler_once(self):
        cell = _simple_cell()
        self.assertEqual(cell.subsolver_name, "backward_euler")
        self.assertIs(cell.subsolver, get_integrator("backward_euler"))
        self.assertEqual(cell.substeps, 1)

    def test_accepts_explicit_subsolver_schedule(self):
        cell = Cell(_simple_cell().morpho, subsolver="rk4", substeps=3)
        self.assertEqual(cell.subsolver_name, "rk4")
        self.assertIs(cell.subsolver, get_integrator("rk4"))
        self.assertEqual(cell.substeps, 3)

    def test_subsolver_schedule_must_be_an_atomic_pair(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            Cell(_simple_cell().morpho, subsolver="rk4")
        with self.assertRaisesRegex(ValueError, "provided together"):
            Cell(_simple_cell().morpho, substeps=2)

    def test_subsolver_schedule_validates_substeps(self):
        with self.assertRaises(TypeError):
            Cell(_simple_cell().morpho, subsolver="rk4", substeps=True)
        with self.assertRaises(ValueError):
            Cell(_simple_cell().morpho, subsolver="rk4", substeps=0)

    def test_rejects_non_morphology(self):
        with self.assertRaises(TypeError):
            Cell("not-a-morpho")  # type: ignore[arg-type]

    def test_rejects_unknown_ion_channel_update_order(self):
        with self.assertRaisesRegex(ValueError, "ion_channel_update_order"):
            Cell(
                Morphology.from_root(
                    Branch.from_lengths(
                        lengths=[20.0] * u.um,
                        radii=[10.0, 10.0] * u.um,
                        type="soma",
                    ),
                    name="soma",
                ),
                ion_channel_update_order="legacy",
            )

    def test_accepts_family_and_integration_update_orders(self):
        tree = Morphology.from_root(
            Branch.from_lengths(
                lengths=[20.0] * u.um,
                radii=[10.0, 10.0] * u.um,
                type="soma",
            ),
            name="soma",
        )
        self.assertEqual(
            Cell(tree, ion_channel_update_order="family").ion_channel_update_order,
            "family",
        )
        self.assertEqual(
            Cell(tree, ion_channel_update_order="integration").ion_channel_update_order,
            "integration",
        )

    def test_membrane_linearizer_defaults_to_point_and_validates(self):
        cell = _simple_cell()
        self.assertEqual(cell.membrane_linearizer, "point")
        cell.membrane_linearizer = "generic"
        self.assertEqual(cell.membrane_linearizer, "generic")
        with self.assertRaisesRegex(ValueError, "membrane_linearizer"):
            Cell(cell.morpho, membrane_linearizer="finite_difference")

    def test_membrane_linearizer_is_frozen_after_initialization(self):
        cell = _simple_cell()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "membrane_linearizer"):
            cell.membrane_linearizer = "generic"

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
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
        )
        self.assertIs(result, cell)

    def test_place_dedups_identical_rules(self):
        cell = _simple_cell()
        clamp = CurrentClamp(delay=1 * u.ms, durations=10 * u.ms, amplitudes=0.1 * u.nA)
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
        self.assertIsNone(cell._axial_jax)
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
            cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV))

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
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV))
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

    def test_run_loop_cache_reuses_matching_shape(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 1)
        first_runner = next(iter(cell._run_loop_cache.values()))

        cell.reset_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)

        self.assertEqual(len(cell._run_loop_cache), 1)
        self.assertIs(next(iter(cell._run_loop_cache.values())), first_runner)

    def test_run_loop_cache_separates_step_count(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        cell.reset_state()
        cell.run(dt=0.1 * u.ms, duration=0.4 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 2)

    def test_reset_clears_run_loop_cache(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 1)

        cell.reset()

        self.assertEqual(cell._run_loop_cache, {})

    def test_axial_operator_cache_tracks_precision(self):
        cell = _simple_cell()
        with brainstate.environ.context(precision=32):
            cell.init_state()
            self.assertIsNone(cell.runtime.axial_operator_np)
            self.assertIsNone(cell.runtime.axial_operator_cache)
            self.assertIsNone(cell._axial_jax)

            operator32 = cell._get_axial_operator()
            cache32 = cell.runtime.axial_operator_cache
            self.assertEqual(operator32.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell._axial_jax.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell.runtime.axial_operator_np.dtype, np.float64)
            self.assertIsNotNone(cache32)

        with brainstate.environ.context(precision=64):
            operator64 = cell._get_axial_operator()
            cache64 = cell.runtime.axial_operator_cache
            self.assertEqual(operator64.dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)

    def test_staggered_run_does_not_build_dense_axial_operator(self):
        cell = _cell_with_probe()
        cell.init_state()

        cell.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertIsNone(cell.runtime.axial_operator_np)
        self.assertIsNone(cell.runtime.axial_operator_cache)
        self.assertIsNone(cell._axial_jax)
        self.assertIsNotNone(cell.runtime.dhs_static_source_np)

    def test_scalar_v_init_broadcasts_to_voltage_shape(self):
        cell = _simple_cell()
        cell.V_init = -60.0 * u.mV
        cell.init_state()
        self.assertEqual(cell.V.value.shape, cell.pop_size + (cell.n_cv,))
        self.assertTrue(
            u.math.allclose(
                cell.V.value,
                jnp.full(cell.pop_size + (cell.n_cv,), -60.0) * u.mV,
                atol=1e-9 * u.mV,
            )
        )

    def test_run_supports_scalar_v_init(self):
        cell = _cell_with_probe()
        cell.V_init = -60.0 * u.mV
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertIn("V_root", result.traces)

    def test_pop_size_extends_voltage_shape(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4, 4))
        cell.V_init = -60.0 * u.mV
        cell.init_state()
        self.assertEqual(cell.pop_size, (4, 4))
        self.assertEqual(cell.varshape, (4, 4, cell.n_cv))
        self.assertEqual(cell.V.value.shape, (4, 4, cell.n_cv))

    def test_pop_size_with_batch_size_keeps_batch_leading(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(2, 3))
        cell.V_init = -60.0 * u.mV
        cell.init_state(batch_size=5)
        self.assertEqual(cell.V.value.shape, (5, 2, 3, cell.n_cv))

    def test_bind_synapse_input_registers_source(self):
        cell = _simple_cell()
        source = lambda: jnp.asarray([1.0])

        returned = cell.bind_synapse_input("ampa", source, weight=2.0)

        self.assertIs(returned, cell)
        self.assertEqual(len(cell._synapse_input_bindings["ampa"]), 1)


class CellMembraneLinearizerTest(unittest.TestCase):
    @staticmethod
    def _leak_cell(mode: str) -> Cell:
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            V_init=-65.0 * u.mV,
            membrane_linearizer=mode,
        )
        cell.paint(
            AllRegion(),
            mech.Channel(
                "IL",
                name="leak",
                g_max=0.3 * (u.mS / u.cm**2),
                E=-54.3 * u.mV,
            ),
        )
        cell.init_state()
        return cell

    def test_point_matches_generic_values_and_higher_order_gradient(self):
        results = {}
        for mode in ("point", "generic"):
            cell = self._leak_cell(mode)
            voltage_unit = u.get_unit(cell.V.value)
            with brainstate.environ.context(t=0.0 * u.ms):
                linear, derivative = cell._voltage_linearizer()(cell.V.value)

                def objective(voltage_mantissa):
                    _, value = cell._voltage_linearizer()(u.Quantity(voltage_mantissa, voltage_unit))
                    return jnp.sum(u.get_mantissa(value))

                gradient = jax.grad(objective)(u.get_mantissa(cell.V.value))
            results[mode] = (linear, derivative, gradient)

        point = results["point"]
        generic = results["generic"]
        for point_value, generic_value in zip(point, generic):
            np.testing.assert_allclose(
                np.asarray(u.get_mantissa(point_value)),
                np.asarray(u.get_mantissa(generic_value)),
                rtol=1e-6,
                atol=1e-7,
            )


class CellIonChannelUpdateOrderTest(unittest.TestCase):
    def test_update_does_not_add_post_voltage_dispatch_for_custom_solver(self):
        calls = []

        def solver(target):
            calls.append("solver")

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
        )
        cell.init_state()

        def family(point_V):
            calls.append("family")

        def integration(point_V):
            calls.append("integration")

        cell._update_ion_channel_families = family
        cell._update_ion_channels_by_integration = integration
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver"])

    def test_staggered_solver_runs_family_dispatch_after_synapse_dynamics(self):
        calls = []

        def solver(target):
            calls.append("solver")
            point_V = target._cv_to_point(target.V.value)
            target._integrate_runtime_synapse_dynamics(point_V)
            target._update_ion_channel_families(point_V)

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
        )
        cell.init_state()
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._integrate_runtime_synapse_dynamics = lambda point_V: calls.append("synapse_dynamics")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver", "synapse_dynamics", "family"])

    def test_staggered_solver_can_use_integration_order(self):
        calls = []

        def solver(target):
            calls.append("solver")
            point_V = target._cv_to_point(target.V.value)
            target._update_ion_channels_by_integration(point_V)

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
            ion_channel_update_order="integration",
        )
        cell.init_state()
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._prepare_runtime_synapse_inputs = lambda point_V: calls.append("prepare")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver", "integration", "prepare"])

    def test_integration_phase_integrates_dependent_runtime_synapse(self):
        from unittest.mock import patch

        from braincell._base import Synapse as RuntimeSynapse

        calls = []

        class _RuntimeSynapse(RuntimeSynapse):
            def current(self, V_post):
                return 0.0

        cell = _simple_cell()
        cell.init_state()
        synapse = _RuntimeSynapse(size=1, name=None)

        cell.runtime_objects = lambda *args, **kwargs: {("layout_0",): synapse}
        cell._runtime_node_phase_args = lambda path, node, point_V: ("syn_arg",)
        cell._top_level_ion_channel_nodes = lambda: ((("layout_0",), synapse),)
        synapse.ind_update = lambda *args, **kwargs: calls.append(("ind", args))

        def _step(target, *args):
            calls.append(("dependent", target, args))

        with patch("braincell._multi_compartment.cell.ind_exp_euler_step", _step):
            point_V = cell._cv_to_point(cell.V.value)
            cell._update_ion_channels_by_integration(point_V)

        self.assertEqual(calls[0][0], "dependent")
        self.assertIs(calls[0][1], synapse)
        self.assertEqual(calls[0][2], ("syn_arg",))
        self.assertEqual(calls[1][0], "ind")
        self.assertEqual(len(calls[1][1]), 1)

    def test_family_phase_runs_ion_hooks_before_child_channel_hooks(self):
        calls = []

        class _Ion(Ion):
            def __init__(self):
                super().__init__(size=1, name=None)
                self.Ci = DiffEqState(jnp.asarray([1.0]))
                self.Co = jnp.asarray([2.0])
                self.valence = 1

            @property
            def E(self):
                return jnp.asarray([0.0])

            def _ion_compute_derivative_hook(self, V):
                calls.append("ion")
                self.Ci.derivative = jnp.asarray([0.0]) / u.ms

        class _Channel(Channel):
            root_type = _Ion

            def __init__(self):
                super().__init__(size=1, name=None)
                self.x = DiffEqState(jnp.asarray([1.0]))

            def compute_derivative(self, V, ion):
                calls.append("channel")
                self.x.derivative = jnp.asarray([0.0]) / u.ms

            def current(self, V, ion):
                return 0.0

        cell = _simple_cell()
        ion = _Ion()
        ion.add(ch=_Channel())
        cell.init_state()
        cell.ion_channels["test_ion"] = ion

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell._update_ion_channel_families(cell._cv_to_point(cell.V.value))

        self.assertEqual(calls, ["ion", "channel"])

    def test_family_phase_does_not_pass_recursive_child_to_channels(self):
        channel_args = []

        class _Ion(Ion):
            def __init__(self):
                super().__init__(size=1, name=None)
                self.Ci = DiffEqState(jnp.asarray([1.0]))
                self.Co = jnp.asarray([2.0])
                self.valence = 1

            @property
            def E(self):
                return jnp.asarray([0.0])

            def _ion_compute_derivative_hook(self, V):
                self.Ci.derivative = jnp.asarray([0.0]) / u.ms

        class _Channel(Channel):
            root_type = _Ion

            def __init__(self):
                super().__init__(size=1, name=None)
                self.x = DiffEqState(jnp.asarray([1.0]))

            def pre_integral(self, V, *ions):
                channel_args.append(("pre", ions))

            def compute_derivative(self, V, *ions):
                channel_args.append(("compute", ions))
                self.x.derivative = jnp.asarray([0.0]) / u.ms

            def post_integral(self, V, *ions):
                channel_args.append(("post", ions))

            def current(self, V, ion):
                return 0.0

        cell = _simple_cell()
        ion = _Ion()
        ion.add(ch=_Channel())
        cell.init_state()
        cell.ion_channels["test_ion"] = ion

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell._update_ion_channel_families(cell._cv_to_point(cell.V.value))

        self.assertGreaterEqual(len(channel_args), 3)
        for _, ions in channel_args:
            self.assertEqual(len(ions), 1)
            self.assertFalse(any(isinstance(arg, bool) for arg in ions))


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
            side_effect=AssertionError("placeholder must not be called at __init__"),
        ):
            _ = Cell(tree)


class CellPopulationAxisIsMandatoryTest(unittest.TestCase):
    """``Cell`` always carries a population axis, so state is rank >= 2."""

    def test_default_pop_size_is_one(self) -> None:
        self.assertEqual(_simple_cell().pop_size, (1,))

    def test_none_pop_size_means_unspecified_and_becomes_one(self) -> None:
        cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=None)
        self.assertEqual(cell.pop_size, (1,))

    def test_scalar_and_tuple_pop_size_agree(self) -> None:
        self.assertEqual(Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=4).pop_size, (4,))
        self.assertEqual(Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=(2, 3)).pop_size, (2, 3))

    def test_empty_pop_size_is_rejected(self) -> None:
        for empty in ((), []):
            with self.subTest(pop_size=empty):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=empty)

    def test_voltage_carries_the_population_axis(self) -> None:
        for pop_size, expected in ((1, (1,)), (4, (4,)), ((2, 3), (2, 3))):
            with self.subTest(pop_size=pop_size):
                cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=pop_size)
                cell.init_state()
                self.assertEqual(tuple(cell.V.value.shape), expected + (cell.n_cv,))


def _hh_cell(pop_size, *, calcium: bool = True) -> Cell:
    """A Cell exercising gate channels, a kinetic ion, and a placed synapse.

    ``calcium=False`` drops the bare ``CalciumDetailed`` ion, which has no
    current source and therefore cannot be stepped; it is only there so the
    kinetic-species allocation path is covered.
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(dend, name="dend", parent_x=0.5)

    cell = Cell(tree, pop_size=pop_size, cv_policy=CVPerBranch(cv_per_branch=2))
    cell.paint(AllRegion(), mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)))
    cell.paint(AllRegion(), mech.Channel("K_HH1952", g_max=3.6 * (u.mS / u.cm**2)))
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.0 * u.mV))
    if calcium:
        cell.paint(AllRegion(), mech.Ion("CalciumDetailed"))
    cell.place(
        RootLocation(x=0.5),
        mech.Synapse("Exp2Syn", tau1=0.5 * u.ms, tau2=2.0 * u.ms, e=0.0 * u.mV, weight=1.0 * u.uS),
    )
    return cell


class CellHiddenStatesAreGroupStatesTest(unittest.TestCase):
    """Every ``Cell`` hidden state groups its trailing spatial axis.

    See ``docs/specs/2026-08-13-cell-hidden-group-state.md``.
    """

    def _hidden_states(self, cell: Cell):
        return [
            (".".join(map(str, path)), state)
            for path, state in brainstate.graph.states(cell).items()
            if isinstance(state, brainstate.HiddenState)
        ]

    def _assert_all_grouped(self, cell: Cell) -> None:
        hidden = self._hidden_states(cell)
        self.assertGreater(len(hidden), 1, "expected channel/ion/synapse states beyond V")
        for name, state in hidden:
            with self.subTest(state=name):
                self.assertIsInstance(state, brainstate.HiddenGroupState)
                # ``num_state`` is the compartment (or point) count, so a
                # grouped state must have something beyond the population axis.
                self.assertEqual(state.varshape, cell.pop_size)
                self.assertIn(state.num_state, (cell.n_cv, cell.n_point, 1))

    def test_every_hidden_state_is_grouped(self) -> None:
        for pop_size in (1, 4):
            with self.subTest(pop_size=pop_size):
                cell = _hh_cell(pop_size)
                cell.init_state()
                self._assert_all_grouped(cell)

    def test_still_grouped_after_reset_state(self) -> None:
        cell = _hh_cell(4)
        cell.init_state()
        cell.reset_state()
        self._assert_all_grouped(cell)

    def test_voltage_is_a_diffeq_group_state(self) -> None:
        from braincell.quad import DiffEqGroupState

        cell = _hh_cell(1)
        cell.init_state()
        self.assertIsInstance(cell.V, DiffEqGroupState)
        # Solvers still select it, because DiffEqGroupState is a DiffEqState.
        self.assertIsInstance(cell.V, DiffEqState)
        self.assertEqual(cell.V.num_state, cell.n_cv)


class CellPopulationWideningIsNumericallyNeutralTest(unittest.TestCase):
    """Widening the population axis must not change the simulated trace."""

    @staticmethod
    def _trace(pop_size, n_steps: int = 20):
        cell = _hh_cell(pop_size, calcium=False)
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=0.1 * u.ms, durations=1.0 * u.ms, amplitudes=0.5 * u.nA),
        )
        cell.init_state()

        dt = 0.01 * u.ms

        def step(i):
            with brainstate.environ.context(t=i * dt, dt=dt):
                cell.update()
            return cell.V.value

        with brainstate.environ.context(dt=dt):
            return brainstate.transform.for_loop(step, jnp.arange(n_steps))

    def test_multi_axis_population_steps_and_jits(self) -> None:
        # ``pop_size`` may have more than one axis; only the trailing
        # compartment axis is the group axis.
        cell = _hh_cell((2, 3), calcium=False)
        cell.init_state()
        self.assertEqual(cell.V.varshape, (2, 3))
        self.assertEqual(cell.V.num_state, cell.n_cv)

        dt = 0.01 * u.ms

        @brainstate.transform.jit
        def step():
            with brainstate.environ.context(t=0.0 * u.ms, dt=dt):
                cell.update()
            return cell.V.value

        self.assertEqual(tuple(step().shape), (2, 3, cell.n_cv))

    def test_three_members_reproduce_the_single_member_trace(self) -> None:
        single = np.asarray(self._trace(1).to_decimal(u.mV), dtype=float)
        population = np.asarray(self._trace(3).to_decimal(u.mV), dtype=float)

        self.assertEqual(single.shape[1], 1)
        self.assertEqual(population.shape, single.shape[:1] + (3,) + single.shape[2:])
        self.assertTrue(np.all(np.isfinite(single)))
        for member in range(3):
            np.testing.assert_allclose(population[:, member], single[:, 0], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
