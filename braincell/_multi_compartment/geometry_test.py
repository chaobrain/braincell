# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
import numpy as np
import brainstate
import jax
import jax.numpy as jnp

import brainunit as u

from braincell import Cell
from braincell._discretization.policy import CVPerBranch
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology
from braincell.trainable import scale, parameter


def _cell(pop_size=(1,), n_cv=1):
    branch = Branch.from_lengths(
        lengths=[20.0, 20.0] * u.um,
        radii=[5.0, 4.0, 3.0] * u.um,
        type="soma",
    )
    return Cell(
        Morphology.from_root(branch, name="soma"),
        cv_policy=CVPerBranch(n_cv),
        pop_size=pop_size,
        V_init=-65.0 * u.mV,
    )


class GeometryViewTest(unittest.TestCase):
    def test_batched_trainable_gradient_checks_for_all_groupings(self):
        from braincell.experimental.optim.gradients import build_trajectory_value_and_grad

        with brainstate.environ.context(precision=64, dt=0.025 * u.ms):
            times = jnp.arange(3, dtype=jnp.float64) * 0.025
            fields = ("length", "radius_scale", "Ra", "cm")
            initial = jnp.asarray([[-65., -50., -40.], [-55., -70., -45.]]) * u.mV

            def build(group, method):
                model = _cell(pop_size=(2,), n_cv=3)
                model.V_init = initial
                model.init_state()
                model.geometry.trainable(**{
                    field: scale(group_by=group, name=field) for field in fields
                })

                def step(time):
                    with brainstate.environ.context(t=time * u.ms):
                        model.update()
                    return model.V.value.to_decimal(u.mV)

                def loss(voltage, _times):
                    return jnp.mean((voltage - jnp.asarray([-60., -55., -50.]))**2)

                return model, build_trajectory_value_and_grad(model, step=step, loss=loss, method=method)

            for group, count in (("row", 6), ("cv", 3), ("population", 2), ("all", 1)):
                with self.subTest(group=group):
                    model, bptt = build(group, "bptt")
                    _, rtrl = build(group, "rtrl")
                    expected = bptt(times)
                    actual = rtrl(times)
                    np.testing.assert_allclose(actual.loss, expected.loss, rtol=1e-10, atol=1e-10)
                    roots = tuple(state.value for state in rtrl.parameter_states.values())
                    names = tuple(rtrl.parameter_states)
                    for index, field in enumerate(names):
                        self.assertEqual(np.size(roots[index]), count)
                        np.testing.assert_allclose(actual.gradients[field], expected.gradients[field], rtol=1e-8, atol=1e-8)
                        direction = jnp.linspace(0.3, 1.0, count).reshape(roots[index].shape)
                        epsilon = 1e-4
                        def loss_at(delta):
                            values = tuple(value + delta * direction if i == index else value for i, value in enumerate(roots))
                            voltage = rtrl._observation_rollout(values, times)
                            return rtrl.loss(voltage, times)
                        finite = (loss_at(epsilon) - loss_at(-epsilon)) / (2 * epsilon)
                        automatic = jnp.sum(actual.gradients[field] * direction)
                        self.assertGreater(abs(float(automatic)), 1e-7, (group, field))
                        np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-6)

                    # Roots are materialized through the complete registered
                    # tree; the gradient checks above cover their updates.

    def test_four_groupings_have_expected_root_shapes(self):
        for group, expected in (("row", (2,)), ("cv", ()), ("population", (2,)), ("all", ())):
            cell = _cell(pop_size=(2,))
            cell.init_state()
            cell.geometry.length.trainable(scale(group_by=group))
            root = next(iter(cell.trainables.roots.values()))
            self.assertEqual(tuple(root.value().shape), expected, group)

    def test_geometry_scales_refresh_derived_cable_values(self):
        cell = _cell()
        cell.init_state()
        area = cell.runtime.cable.area
        resistance = cell.runtime.cable.resistance_prox
        cell.geometry.length.set(cell.geometry.length.get() * 2.0)
        cell.geometry.radius_scale.set(0.5)
        self.assertTrue(bool((cell.runtime.cable.area == area).all()))
        self.assertTrue(bool((cell.runtime.cable.resistance_prox == resistance * 8.0).all()))

    def test_direct_physical_geometry_fields_are_available(self):
        cell = _cell()
        cell.init_state()
        self.assertEqual(cell.geometry.length.get().shape, (1,))
        self.assertEqual(set(cell.geometry.parameter_info()), {"length", "radius_scale", "Ra", "cm"})
        np.testing.assert_allclose(cell.geometry.radius_scale.get(), [1.0])
        for field in ("radius", "length_scale", "radius_prox", "radius_dist"):
            with self.assertRaises(AttributeError):
                getattr(cell.geometry, field)
        cell.geometry.length.set(2.0 * u.um)
        cell.geometry.radius_scale.set(0.5)
        np.testing.assert_allclose(cell.geometry.length.get().to_decimal(u.um), [2.0])
        np.testing.assert_allclose(cell.geometry.radius_mid.get().to_decimal(u.um), [2.0])
        np.testing.assert_allclose(cell.geometry.diam_arc_mean.get().to_decimal(u.um), [4.0])
        self.assertLess(float(cell.geometry.area.get().to_decimal(u.um**2)[0]), 1.0e3)
        for field in ("radius_mid", "diam_arc_mean"):
            with self.assertRaises(KeyError):
                getattr(cell.geometry, field).set(1.0 * u.um)
            with self.assertRaises(KeyError):
                getattr(cell.geometry, field).trainable(parameter())

    def test_geometry_is_runtime_only_and_reset_drops_override(self):
        cell = _cell()
        with self.assertRaises(RuntimeError):
            _ = cell.geometry
        cell.init_state()
        cell.geometry.cm.set(2.0 * u.uF / u.cm**2)
        self.assertEqual(float(cell.geometry.cm.get()[0].to_decimal(u.uF / u.cm**2)), 2.0)
        cell.reset_state()
        self.assertEqual(float(cell.geometry.cm.get()[0].to_decimal(u.uF / u.cm**2)), 2.0)
        cell.reset()
        with self.assertRaises(RuntimeError):
            _ = cell.geometry

    def test_population_specific_runtime_geometry_updates_each_cell(self):
        cell = _cell(pop_size=(2,))
        cell.init_state()
        cell.geometry.length.set([40.0, 80.0] * u.um)
        np.testing.assert_allclose(cell.geometry.length.get().to_decimal(u.um), [40, 80])
        area = cell.geometry.area.get().to_decimal(u.um**2)
        np.testing.assert_allclose(area[1], 2 * area[0])

    def test_heterogeneous_population_rollout_matches_independent_cells(self):
        with brainstate.environ.context(precision=64, dt=0.025 * u.ms):
            cell = _cell(pop_size=(2,), n_cv=3)
            cell.init_state()
            length = np.asarray([[10., 14., 16.], [20., 9., 11.]]) * u.um
            ra = np.asarray([[90., 100., 110.], [120., 80., 130.]]) * u.ohm * u.cm
            cm = np.asarray([[0.8, 1.2, 1.], [1.4, 0.7, 1.1]]) * u.uF / u.cm**2
            radius_scale = np.asarray([[0.8, 1.2, 1.], [1.4, 0.7, 1.1]])
            fields = dict(length=length, Ra=ra, cm=cm, radius_scale=radius_scale)
            cell.geometry.set(**{name: value.reshape(-1) for name, value in fields.items()})
            initial = jnp.asarray([[-65., -50., -40.], [-55., -70., -45.]]) * u.mV

            def trajectory(model, voltage):
                def step(time):
                    with brainstate.environ.context(t=time * u.ms):
                        model.update()
                    return model.V.value.to_decimal(u.mV)

                @brainstate.transform.jit
                def rollout():
                    model.reset_state()
                    model.V.value = voltage
                    return brainstate.transform.for_loop(step, jnp.arange(4) * 0.025)
                return rollout

            rollout = trajectory(cell, initial)
            batched = rollout()
            for population in range(2):
                other = _cell(n_cv=3)
                other.init_state()
                other.geometry.set(**{name: value[population] for name, value in fields.items()})
                expected = trajectory(other, initial[population:population + 1])()
                np.testing.assert_allclose(batched[:, population], expected[:, 0], rtol=1e-12, atol=1e-10)

            cell.geometry.length.set((length * 1.3).reshape(-1))
            changed = rollout()
            self.assertGreater(float(jnp.max(jnp.abs(changed - batched))), 1e-4)

    def test_length_parameter_and_scale_share_one_runtime_field(self):
        for source, update in ((parameter(name="L"), 60.0 * u.um), (scale(name="L"), 1.5)):
            cell = _cell()
            cell.init_state()
            cell.geometry.length.trainable(source)
            cell.trainables.parameters().set_physical_values({"L": update})
            cell.trainables.materialize()
            np.testing.assert_allclose(cell.geometry.length.get().to_decimal(u.um), [60.0])
            self.assertNotIn("length_scale", cell.runtime.geometry)
            cell.reset_state()
            np.testing.assert_allclose(cell.geometry.length.get().to_decimal(u.um), [60.0])
            cell.reset()
            cell.init_state()
            np.testing.assert_allclose(cell.geometry.length.get().to_decimal(u.um), [40.0])

    def test_cv_collection_exposes_geometry_content_view(self):
        cell = _cell(pop_size=(2,), n_cv=3)
        cell.init_state()
        self.assertEqual(len(cell.cvs), cell.n_cv)
        self.assertEqual(tuple(cell.cvs.length.get().shape), (6,))
        selected = cell.cvs[0:1]
        self.assertEqual(len(selected), 1)
        self.assertEqual(tuple(selected.length.get().shape), (2,))
        selected.length.set(60.0 * u.um)
        np.testing.assert_allclose(selected.length.get().to_decimal(u.um), 60.0)


if __name__ == "__main__":
    unittest.main()
