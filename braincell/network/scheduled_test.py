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

"""``Network.run``/``prepare_run(..., scheduled_delivery=...)`` against the default path (CELL-9).

Model M: 4 cells, one 20 um branch of radius 10 um (``CVPerBranch(1)``),
``IL`` 0.1 mS/cm2 at E -65 mV, ``V_init`` -65 mV; 6 independent ExpSyn per
cell (tau 2 ms, e 0 mV, x 0.5); ``NetStim(size=24, start=0.3 ms,
interval=0.5 ms, number=160, noise=1, seed=5)``, ``Network(seed=1)``; one
row per synapse, weight 0.002 uS, delays ``linspace(0, 1, 24)`` ms; dt
0.025 ms. Its ``"padded"`` plans need 4.82, 4.39, 4.00 and 4.34 slots per
record for the windows [0, 300), [300, 600), [600, 1000) and [0, 1000)
(width 5 in each), below the limit of 8.

The reference is always the default path (``scheduled_delivery=None``)
run on the same model; a plan delivers each event at the step and with
the weight that path does, so the trajectories agree to ``rtol=1e-12``
(on this machine they are bitwise equal, the layouts having one row per
synapse).
"""

import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell as bc
from braincell.experimental.scheduled_delivery import PaddingLimitError as ExperimentalPaddingLimitError
from braincell.filter import AllRegion, LocsetMask
from braincell.network import scheduled
from braincell.network.scheduled import METHODS, PaddingLimitError

DT = 0.025 * u.ms
N_CELLS, M_SYN = 4, 6
SIZE = N_CELLS * M_SYN
RTOL, ATOL = 1e-12, 1e-15
RUNS = (1000, (300, 300, 400), 1000)

#: Sparse schedule: every source at 0.3 (twice) and 1.2 ms. Its padded plan
#: needs 12.5 slots per record in [0, 300) (48 records of width 2) and 41.7
#: in [0, 1000), above the limit of 8, so ``"padded"`` raises there.
SPARSE_MS = (0.3, 0.3, 1.2)
#: Dense schedule: every source at 0.3 (twice), 0.8 + 0.5 k ms (k = 0 .. 48)
#: and once more at 12.3 ms (twice at 12.3). Width 2 in every window; 1.80,
#: 1.67, 1.67 and 1.71 slots per record for [0, 300), [300, 600), [600,
#: 1000) and [0, 1000); 24 records of multiplicity 2 in each of the first two.
DENSE_MS = tuple(sorted((0.3, 0.3, *(0.8 + 0.5 * k for k in range(49)), 12.3)))


def _float64(test):
    """Run ``test`` in float64."""

    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    wrapped.__doc__ = test.__doc__
    return wrapped


def _netstim():
    return bc.NetStim(size=SIZE, start=0.3 * u.ms, interval=0.5 * u.ms, number=160, noise=1.0, seed=5)


def _sequence(times_ms, size=SIZE):
    return bc.EventSequence.from_times(tuple(np.asarray(times_ms, dtype=float) * u.ms for _ in range(size)))


def _rule():
    # The parameters of the lumped-target rule test (CELL-12a acceptance 7).
    return bc.mech.ConnectionRule(
        "PairSTDP",
        A_plus=1e-4 * u.uS,
        A_minus=1.2e-4 * u.uS,
        tau_plus=20.0 * u.ms,
        tau_minus=20.0 * u.ms,
        w_min=0.0 * u.uS,
        w_max=0.002 * u.uS,
    )


def _model(*, source=None, layout="independent", synapse="ExpSyn", v_th=None, ring=False, rule=False):
    """Model M, or a variant of it; returns ``(network, cell)``."""
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    extra = {} if v_th is None else {"V_th": v_th * u.mV}
    cell = bc.Cell(
        bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=N_CELLS, V_init=-65.0 * u.mV, **extra
    )
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    per_cell = M_SYN if layout == "independent" else 1
    if synapse == "Exp2Syn":
        declaration = bc.mech.Synapse("Exp2Syn", name="exp", tau1=0.5 * u.ms, tau2=5.0 * u.ms, e=0.0 * u.mV)
    else:
        declaration = bc.mech.Synapse(synapse, name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(LocsetMask.from_columns(np.zeros(per_cell, dtype=int), np.full(per_cell, 0.5)), declaration)
    net = bc.Network("m", seed=1)
    post = net.add_population("post", cell)
    stim = net.add_population("stim", _netstim() if source is None else source)
    # Synapse id = placement * N_CELLS + member: row r reaches member r % 4.
    targets = np.arange(SIZE) if layout == "independent" else np.arange(SIZE) % N_CELLS
    net.connect(
        "drive",
        source=stim,
        synapse=post.synapses["exp"][targets],
        weight=0.002 * u.uS,
        delay=np.linspace(0.0, 1.0, SIZE) * u.ms,
    )
    if ring:
        # Live: the spike of cell i reaches synapse 0 of cell i + 1.
        net.connect(
            "ring",
            source=post.event_outputs["spike"][np.arange(N_CELLS)],
            synapse=post.synapses["exp"][(np.arange(N_CELLS) + 1) % N_CELLS],
            weight=0.001 * u.uS,
            delay=1.0 * u.ms,
        )
    if rule:
        learn = net.add_population(
            "learn_stim", bc.NetStim(size=4, start=1.0 * u.ms, number=40, interval=2.0 * u.ms, noise=0.8)
        )
        net.connect(
            "learn",
            source=learn,
            synapse=post.synapses["exp"][np.arange(N_CELLS)],
            weight=0.0005 * u.uS,
            delay=0.5 * u.ms,
            plasticity=_rule(),
        )
    cell.record("v", "V")
    cell.record("g", bc.observe.synapse(name="exp").state("g"))
    if synapse == "ExpSynSTD":
        cell.record("x", bc.observe.synapse(name="exp").state("x"))
    return net, cell


def _values(result, cell, *, rule=False):
    samples = result.samples["post"]
    out = {name: np.asarray(u.get_mantissa(block.values)) for name, block in samples.items()}
    spikes = result.events["post"]["spike"]
    out["spike_ms"] = np.asarray(spikes.time.to_decimal(u.ms))
    out["spike_src"] = np.asarray(spikes.source_id)
    if rule:
        out["w"] = np.asarray(cell.connections["learn"].rule_state("w").to_decimal(u.uS))
    return out


def _concat(parts):
    return {
        name: (parts[-1][name] if name == "w" else np.concatenate([part[name] for part in parts])) for name in parts[0]
    }


def _three_runs(net, cell, method, *, rule=False):
    """One 1000-step run; after reset, 300 + 300 + 400; after reset, 1000 again."""
    extra = {} if method is None else {"scheduled_delivery": method}
    out = []
    for index, steps in enumerate(RUNS):
        if index:
            net.reset_state()
        parts = [
            _values(net.run(dt=DT, duration=n * DT, **extra), cell, rule=rule)
            for n in (steps if isinstance(steps, tuple) else (steps,))
        ]
        out.append(_concat(parts))
    return out


def _assert_same(test, got, reference, *, rtol=RTOL, atol=ATOL, v_atol=None):
    test.assertEqual(set(got), set(reference))
    for name in reference:
        if name.startswith("spike"):
            np.testing.assert_array_equal(got[name], reference[name], err_msg=name)
            continue
        tolerance = atol if (name != "v" or v_atol is None) else v_atol
        np.testing.assert_allclose(got[name], reference[name], rtol=rtol, atol=tolerance, err_msg=name)


class EquivalenceTest(unittest.TestCase):
    """Acceptance 1: one run, three segments and a run after reset agree with the default path."""

    def _check(self, *, methods=METHODS, rtol=RTOL, v_atol=None, rule=False, **model):
        net, cell = _model(rule=rule, **model)
        reference = _values(net.run(dt=DT, duration=1000 * DT), cell, rule=rule)
        for method in methods:
            with self.subTest(method=method):
                net, cell = _model(rule=rule, **model)
                for got in _three_runs(net, cell, method, rule=rule):
                    _assert_same(self, got, reference, rtol=rtol, v_atol=v_atol)
        return reference

    @_float64
    def test_model_m(self):
        reference = self._check()
        self.assertGreater(np.max(reference["g"]), 0.01)  # the events arrive

    @_float64
    def test_padded_stores_a_power_of_two_width_whose_extra_slots_are_masked(self):
        net, cell = _model()
        reference = _values(net.run(dt=DT, duration=1000 * DT), cell)
        net, cell = _model()
        got = _values(net.run(dt=DT, duration=1000 * DT, scheduled_delivery="padded"), cell)
        (layout,) = net._scheduled_plans.preparation["layouts"]
        self.assertEqual((layout["width"], layout["stored_width"]), (5, 8))
        self.assertAlmostEqual(layout["padding_ratio"], 1000 * 5 / 1153)
        _assert_same(self, got, reference)

    @_float64
    def test_dense_event_sequence(self):
        """(a) An EventSequence source, with two events of a row on one step."""
        self._check(source=_sequence(DENSE_MS))

    @_float64
    def test_sparse_event_sequence_direct(self):
        """(a) The sparse schedule, whose padded plan is over its limit (``PaddingLimitTest``)."""
        self._check(source=_sequence(SPARSE_MS), methods=("direct",))

    @_float64
    def test_shared_synapses(self):
        """(b) 24 rows onto one ExpSyn per cell: the additions come in another order."""
        self._check(layout="shared")

    @_float64
    def test_exp2syn(self):
        """(c)"""
        self._check(synapse="Exp2Syn")

    @_float64
    def test_live_ring_keeps_its_path(self):
        """(d) V_th -60 mV and a live ring, delay 1 ms: four live events, one per cell."""
        reference = self._check(v_th=-60.0, ring=True)
        self.assertEqual(sorted(reference["spike_src"].tolist()), [0, 1, 2, 3])

    def test_float32(self):
        """(e) The three runs of model M in float32."""
        with brainstate.environ.context(precision=32):
            self._check(rtol=1e-6, v_atol=1e-6)


class CacheTest(unittest.TestCase):
    """Acceptance 2: plan builds, input-stage traces and the window cache."""

    def _counting(self):
        builds = mock.patch.object(scheduled, "_build_plans", wraps=scheduled._build_plans)
        traces = mock.patch.object(
            scheduled._LayoutInput, "deliver", autospec=True, side_effect=scheduled._LayoutInput.deliver
        )
        return builds, traces

    @_float64
    def test_windows_reuse_the_loop_and_the_cache_keeps_four(self):
        for method in METHODS:
            with self.subTest(method=method):
                net, cell = _model()
                builds, traces = self._counting()
                with builds as built, traces as traced:
                    counts = []
                    for steps in (300, 300, 400):
                        net.run(dt=DT, duration=steps * DT, scheduled_delivery=method)
                        counts.append((built.call_count, traced.call_count))
                    net.reset_state()
                    net.run(dt=DT, duration=1000 * DT, scheduled_delivery=method)
                    counts.append((built.call_count, traced.call_count))
                    # The second 300-step window reuses the loop; 400 and 1000 steps trace their own.
                    self.assertEqual(counts, [(1, 1), (2, 1), (3, 2), (4, 3)])
                    windows = [key[:2] for key in net._scheduled_plans.windows]
                    self.assertEqual(windows, [(0, 300), (300, 300), (600, 400), (0, 1000)])

                    # After reset_state, [0, 1000) is a cache hit.
                    net.reset_state()
                    first = _values(net.run(dt=DT, duration=1000 * DT, scheduled_delivery=method), cell)
                    self.assertEqual((built.call_count, traced.call_count), (4, 3))

                    # A new weight is read on every step: no rebuild, no trace.
                    cell.connections["drive"].set(weight=0.003 * u.uS)
                    net.reset_state()
                    heavier = _values(net.run(dt=DT, duration=1000 * DT, scheduled_delivery=method), cell)
                    self.assertEqual((built.call_count, traced.call_count), (4, 3))

                    # A fifth window pushes out the least recently used, (0, 300).
                    net.reset_state()
                    net.run(dt=DT, duration=100 * DT, scheduled_delivery=method)
                    self.assertEqual(built.call_count, 5)
                    windows = [key[:2] for key in net._scheduled_plans.windows]
                    self.assertEqual(windows, [(300, 300), (600, 400), (0, 1000), (0, 100)])
                    net.reset_state()
                    net.run(dt=DT, duration=300 * DT, scheduled_delivery=method)
                    self.assertEqual(built.call_count, 6)

                ref_net, ref_cell = _model()
                ref_net.init_state()
                ref_cell.connections["drive"].set(weight=0.003 * u.uS)
                reference = _values(ref_net.run(dt=DT, duration=1000 * DT), ref_cell)
                _assert_same(self, heavier, reference)
                self.assertGreater(np.max(heavier["g"]), np.max(first["g"]))

    @_float64
    def test_a_growing_netstim_schedule_rebuilds_the_plan_not_the_loop(self):
        """A schedule generated on demand grows with the runs; the plan is an argument of the loop."""
        out = {}
        for method in (None, *METHODS):
            stim = bc.NetStim(size=2, start=0.1 * u.ms, interval=0.05 * u.ms, number=2**20)
            self.assertFalse(stim._schedule_complete)
            net, cell = self._growing(stim)
            extra = {} if method is None else {"scheduled_delivery": method}
            builds, traces = self._counting()
            with builds as built, traces as traced:
                parts = [_values(net.run(dt=DT, duration=400 * DT, **extra), cell) for _ in range(3)]
            out[method] = _concat(parts)
            if method is not None:
                # The schedule grew twice (three tokens), the loop traced once.
                self.assertEqual((built.call_count, traced.call_count), (3, 1))
        for method in METHODS:
            _assert_same(self, out[method], out[None])

    @staticmethod
    def _growing(stim):
        branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV)
        cell.place(LocsetMask.from_columns([0, 0], [0.5, 0.5]), bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms))
        net = bc.Network("grow", seed=1)
        post = net.add_population("post", cell)
        source = net.add_population("stim", stim)
        net.connect("drive", source=source, synapse=post.synapses["exp"][[0, 1]], weight=0.001 * u.uS)
        cell.record("v", "V")
        cell.record("g", bc.observe.synapse(name="exp").state("g"))
        return net, cell


def _grad_model():
    net, cell = _model()
    net.init_state()
    cell.synapses["exp"].trainable(tau=bc.trainable.scale(name="tau_scale"))
    cell.connections["drive"].trainable(weight=bc.trainable.scale(name="weight_scale"))
    return net, cell


def _rollout_loss(net, cell):
    """Reset, then two ``for_loop(update)`` of 450 and 550 steps; Σ g (uS) + 1e-3 Σ V (mV) over the steps."""
    node = cell.runtime.get_runtime_node(cell.synapses["exp"]._store.layout_id("ExpSyn"))

    def step(_):
        net.update()
        return node.g.value.to_decimal(u.uS).sum() + 1e-3 * cell.V.value.to_decimal(u.mV).sum()

    def loss():
        net.reset_state()
        first = brainstate.transform.for_loop(step, jnp.arange(450)).sum()
        return first + brainstate.transform.for_loop(step, jnp.arange(550)).sum()

    return loss


class GradientTest(unittest.TestCase):
    """Acceptance 3: reverse gradients through ``prepare_run`` + ``update``."""

    @_float64
    def test_gradients_match_the_default_path_and_central_differences(self):
        grads = {}
        for method in (None, *METHODS):
            net, cell = _grad_model()
            net.prepare_run(dt=DT, **({} if method is None else {"scheduled_delivery": method}))
            loss = _rollout_loss(net, cell)
            params = net.trainables.parameters()
            grad = brainstate.transform.jit(brainstate.transform.grad(loss, grad_states=params.states()))()
            grads[method] = {name: float(value) for name, value in grad.items()}
            if method is None:
                for name, value in grads[method].items():
                    shifted = []
                    for h in (1e-6, -1e-6):
                        physical = params.physical_values()
                        base = float(physical[name])
                        physical[name] = base + h
                        params.set_physical_values(physical)
                        shifted.append(float(brainstate.transform.jit(loss)()))
                        physical[name] = base
                        params.set_physical_values(physical)
                    central = (shifted[0] - shifted[1]) / 2e-6
                    self.assertLessEqual(abs(value - central) / abs(central), 1e-6, name)
        self.assertEqual(set(grads[None]), {"post.tau_scale", "post.weight_scale"})
        for method in METHODS:
            for name, value in grads[None].items():
                np.testing.assert_allclose(grads[method][name], value, rtol=1e-9, err_msg=f"{method} {name}")


class RuleAndCountTest(unittest.TestCase):
    """Acceptance 4: rows with a rule keep the rule's path; ``"sum_count"`` layouts get their counts."""

    @_float64
    def test_a_rule_call_on_the_same_cell(self):
        net, cell = _model(v_th=-60.0, rule=True)
        reference = _values(net.run(dt=DT, duration=1000 * DT), cell, rule=True)
        self.assertFalse(np.allclose(reference["w"], 0.0005))  # the rule learns
        for method in METHODS:
            with self.subTest(method=method):
                net, cell = _model(v_th=-60.0, rule=True)
                got = _values(net.run(dt=DT, duration=1000 * DT, scheduled_delivery=method), cell, rule=True)
                _assert_same(self, got, reference)
                (route,) = net._scheduled_plans.routes
                self.assertEqual(route.call_rows, (SIZE,))  # the rule call is not planned

    @_float64
    def test_sum_count_layout(self):
        net, cell = _model(synapse="ExpSynSTD")
        reference = _values(net.run(dt=DT, duration=1000 * DT), cell)
        self.assertLess(np.min(reference["x"]), 0.9)  # the resource depletes
        for method in METHODS:
            with self.subTest(method=method):
                net, cell = _model(synapse="ExpSynSTD")
                for got in _three_runs(net, cell, method):
                    _assert_same(self, got, reference)


def _overflow_model():
    """1 cell, 10 independent ExpSyn, one event every 10 ms per source: at most one record per step."""
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV)
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(
        LocsetMask.from_columns(np.zeros(10, dtype=int), np.full(10, 0.5)),
        bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
    )
    net = bc.Network("overflow", seed=1)
    post = net.add_population("post", cell)
    stim = net.add_population(
        "stim", bc.NetStim(size=10, start=0.5 * np.arange(10) * u.ms, interval=10.0 * u.ms, number=10)
    )
    net.connect("drive", source=stim, synapse=post.synapses["exp"][np.arange(10)], weight=0.002 * u.uS)
    cell.record("v", "V")
    cell.record("g", bc.observe.synapse(name="exp").state("g"))
    return net, cell


def _state(net, cell):
    """``(t in ms, step, V, g)``: what a run that raised at its entry must leave as it was."""
    net.init_state()
    node = cell.runtime.get_runtime_node(
        cell.synapses["exp"]._store.layout_id(str(cell.synapses["exp"].synapse_type[0]))
    )
    return (
        float(cell.current_time.to_decimal(u.ms)),
        int(cell._clock_step.value),
        np.asarray(cell.V.value.to_decimal(u.mV)),
        np.asarray(node.g.value.to_decimal(u.uS)),
    )


class PaddingLimitTest(unittest.TestCase):
    """Acceptance 5: a padded plan over its limit raises at the entry and changes nothing."""

    def _assert_unchanged(self, before, after):
        self.assertEqual(before[:2], after[:2])
        for first, second in zip(before[2:], after[2:]):
            np.testing.assert_array_equal(first, second)

    @_float64
    def test_one_record_per_step_needs_forty_slots_per_record(self):
        net, cell = _overflow_model()
        reference = _values(net.run(dt=DT, duration=100.0 * u.ms), cell)
        net, cell = _overflow_model()
        net.run(dt=DT, duration=10 * DT)  # it has run: the state is not the initial one
        net, cell = _overflow_model()
        before = _state(net, cell)
        with mock.patch.object(scheduled, "_build_plans", wraps=scheduled._build_plans) as built:
            with self.assertRaises(PaddingLimitError) as caught:
                net.run(dt=DT, duration=100.0 * u.ms, scheduled_delivery="padded")
        self.assertEqual(built.call_count, 1)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIs(PaddingLimitError, ExperimentalPaddingLimitError)
        self.assertIn('"direct"', str(caught.exception))
        self.assertIn(" 40 slots per record", str(caught.exception))
        self._assert_unchanged(before, _state(net, cell))
        # A first run that raised fixes no configuration: "direct" can run.
        _assert_same(self, _values(net.run(dt=DT, duration=100.0 * u.ms, scheduled_delivery="direct"), cell), reference)

    @_float64
    def test_after_a_run_the_configuration_stays_fixed(self):
        net, cell = _overflow_model()
        net.run(dt=DT, duration=8 * DT, scheduled_delivery="padded")  # [0, 8) holds one record: 8 slots
        before = _state(net, cell)
        with self.assertRaises(PaddingLimitError):
            net.run(dt=DT, duration=100.0 * u.ms, scheduled_delivery="padded")
        self._assert_unchanged(before, _state(net, cell))
        with self.assertRaises(RuntimeError):
            net.run(dt=DT, duration=100.0 * u.ms, scheduled_delivery="direct")

    @_float64
    def test_prepare_run_plans_every_event(self):
        net, cell = _overflow_model()
        with self.assertRaises(PaddingLimitError):
            net.prepare_run(dt=DT, scheduled_delivery="padded")
        self.assertIsNone(net._runtime_config)
        net.prepare_run(dt=DT, scheduled_delivery="direct")

    @_float64
    def test_sparse_event_sequence(self):
        """(a) The sparse schedule needs 12.5 slots per record in its first 300 steps."""
        net, cell = _model(source=_sequence(SPARSE_MS))
        before = _state(net, cell)
        with self.assertRaisesRegex(PaddingLimitError, "12.5 slots per record"):
            net.run(dt=DT, duration=300 * DT, scheduled_delivery="padded")
        self._assert_unchanged(before, _state(net, cell))

    @_float64
    def test_an_empty_window_has_no_ratio_limit(self):
        net, cell = _model(source=_sequence([20.0]))
        reference = _values(net.run(dt=DT, duration=100 * DT), cell)
        self.assertEqual(np.max(reference["g"]), 0.0)
        net, cell = _model(source=_sequence([20.0]))
        _assert_same(self, _values(net.run(dt=DT, duration=100 * DT, scheduled_delivery="padded"), cell), reference)


class ConfigurationTest(unittest.TestCase):
    """Acceptance 5: the values ``scheduled_delivery`` accepts, and the fixed runtime configuration."""

    def test_every_value_runs_or_is_rejected_at_the_call(self):
        # A new method needs its own run test here (L71).
        self.assertEqual(METHODS, ("direct", "padded"))
        for value in (None, *METHODS):
            with self.subTest(value=value):
                net, cell = _model()
                net.run(dt=DT, duration=300 * DT, scheduled_delivery=value)
                self.assertEqual(net._runtime_config[-1], value)
                self.assertEqual(net._scheduled_plans is None, value is None)
        for value in ("bucket", "Direct", True, 0):
            for entry in ("run", "prepare_run"):
                with self.subTest(value=value, entry=entry):
                    net, cell = _model()
                    call = (
                        (lambda: net.run(dt=DT, duration=40 * DT, scheduled_delivery=value))
                        if entry == "run"
                        else (lambda: net.prepare_run(dt=DT, scheduled_delivery=value))
                    )
                    with self.assertRaisesRegex(ValueError, "scheduled_delivery must be None, 'direct', or 'padded'"):
                        call()
                    self.assertFalse(net._initialized)
                    net.run(dt=DT, duration=40 * DT)
                    before = float(cell.current_time.to_decimal(u.ms))
                    with self.assertRaises(ValueError):
                        call()
                    self.assertEqual(float(cell.current_time.to_decimal(u.ms)), before)

    def test_the_method_is_fixed_after_the_first_run(self):
        for first, then, entry in (
            ("direct", None, "run"),
            (None, "direct", "run"),
            ("direct", "padded", "run"),
            ("padded", None, "prepare_run"),
        ):
            with self.subTest(first=first, then=then, entry=entry):
                net, _ = _model()
                net.run(dt=DT, duration=300 * DT, scheduled_delivery=first)
                with self.assertRaisesRegex(RuntimeError, "configuration is fixed"):
                    if entry == "run":
                        net.run(dt=DT, duration=40 * DT, scheduled_delivery=then)
                    else:
                        net.prepare_run(dt=DT, scheduled_delivery=then)

    def test_a_window_past_the_int32_step_count_raises_before_any_plan(self):
        net, cell = _model()
        with mock.patch.object(scheduled, "_build_plans", wraps=scheduled._build_plans) as built:
            with self.assertRaisesRegex(ValueError, "past its limit"):
                net.run(dt=DT, duration=2**31 * DT, scheduled_delivery="direct")
        self.assertEqual(built.call_count, 0)
        self.assertEqual(int(cell._clock_step.value), 0)


class UpdatePathTest(unittest.TestCase):
    """``prepare_run`` plans every event once; ``update()`` loops read it at any step."""

    @staticmethod
    def _update_values(net, cell, steps):
        node = cell.runtime.get_runtime_node(cell.synapses["exp"]._store.layout_id("ExpSyn"))

        def step(_):
            net.update()
            return cell.V.value.to_decimal(u.mV), node.g.value.to_decimal(u.uS)

        v, g = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, jnp.arange(steps)))()
        return {"v": np.asarray(v), "g": np.asarray(g)}

    def _check(self, *, source=None, run_first=0, steps=1000):
        out = {}
        for method in (None, *METHODS):
            net, cell = _model(source=source)
            extra = {} if method is None else {"scheduled_delivery": method}
            net.prepare_run(dt=DT, **extra)
            parts = []
            if run_first:
                parts.append(_values(net.run(dt=DT, duration=run_first * DT, **extra), cell))
            parts.append(self._update_values(net, cell, steps))
            net.reset_state()
            parts.append(self._update_values(net, cell, steps))
            out[method] = parts
        for method in METHODS:
            for got, reference in zip(out[method], out[None]):
                _assert_same(self, got, reference)
        return out

    @_float64
    def test_update_loops(self):
        self._check()

    @_float64
    def test_a_run_and_then_update_loops(self):
        self._check(run_first=300, steps=700)

    @_float64
    def test_steps_after_the_last_arrival(self):
        """The sparse schedule's last arrival is near step 88: later steps read no record."""
        out = self._check(source=_sequence(SPARSE_MS), steps=200)
        self.assertGreater(np.max(out[None][0]["g"]), 0.0)
        self.assertEqual(out[None][0]["g"].shape[0], 200)


class ForwardModeTest(unittest.TestCase):
    """Acceptance 5: forward mode through ``update()`` -- ``"padded"`` only."""

    @_float64
    def test_jvp(self):
        tangents = {}
        for method in (None, *METHODS):
            net, cell = _grad_model()
            net.prepare_run(dt=DT, **({} if method is None else {"scheduled_delivery": method}))
            node = cell.runtime.get_runtime_node(cell.synapses["exp"]._store.layout_id("ExpSyn"))
            state = net.trainables.parameters().states()["post.weight_scale"]
            base = state.value

            def g_sum(scale):
                state.value = scale
                net.reset_state()
                return brainstate.transform.for_loop(
                    lambda _: (net.update(), node.g.value.to_decimal(u.uS).sum())[1], jnp.arange(100)
                ).sum()

            try:
                if method == "direct":
                    with self.assertRaisesRegex(TypeError, "custom_vjp"):
                        brainstate.transform.jvp(g_sum, (base,), (jnp.ones_like(base),))
                    continue
                tangents[method] = brainstate.transform.jvp(g_sum, (base,), (jnp.ones_like(base),))
            finally:
                state.value = base
        np.testing.assert_allclose(tangents["padded"], tangents[None], rtol=RTOL)


def _two_populations():
    """Two Cell populations, one with an ExpSyn and an Exp2Syn layout, each fed all 24 NetStim sources."""
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    net = bc.Network("two", seed=1)
    stim = net.add_population("stim", _netstim())
    cells = {}
    for name, synapses in (("a", ("ExpSyn", "Exp2Syn")), ("b", ("ExpSyn",))):
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=2, V_init=-65.0 * u.mV)
        cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        for kind in synapses:
            cell.place(LocsetMask.from_columns([0, 0, 0], [0.5, 0.5, 0.5]), bc.mech.Synapse(kind, name=kind))
        population = net.add_population(name, cell)
        for kind in synapses:
            net.connect(
                f"{name}_{kind}",
                source=stim,
                synapse=population.synapses[kind][np.arange(SIZE) % 6],
                weight=0.001 * u.uS,
                delay=np.linspace(0.0, 1.0, SIZE) * u.ms,
            )
            cell.record(kind, bc.observe.synapse(name=kind).state("g" if kind == "ExpSyn" else "B"))
        cell.record("V", "V")
        cells[name] = cell
    return net


class ClockOriginTest(unittest.TestCase):
    """A Cell that ran at another ``dt`` before joining starts the Network's window at its time."""

    @_float64
    def test_the_window_starts_at_the_cell_time_on_the_dt_grid(self):
        out = {}
        for method in (None, "direct"):
            branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
            cell = bc.Cell(
                bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV
            )
            cell.place(LocsetMask.from_columns([0, 0], [0.5, 0.5]), bc.mech.Synapse("ExpSyn", name="exp"))
            cell.record("g", bc.observe.synapse(name="exp").state("g"))
            sequence = _sequence([0.5, 1.2, 1.5, 2.0, 2.05], size=2)
            bc.connect(
                "drive",
                source=sequence,
                synapse=cell.synapses["exp"][[0, 1]],
                weight=0.001 * u.uS,
                delay=np.array([0.0, 0.3]) * u.ms,
            )
            cell.init_state()
            cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)  # the clock's origin moves to 1 ms below
            net = bc.Network("origin", seed=1)
            net.add_population("post", cell)
            net.add_population("stim", sequence)
            result = net.run(dt=DT, duration=2.0 * u.ms, **({} if method is None else {"scheduled_delivery": method}))
            out[method] = np.asarray(u.get_mantissa(result.samples["post"]["g"].values))
            if method is not None:
                self.assertEqual(cell._clock_host(), (1.0, 80, 0.025))
                self.assertEqual(net._scheduled_plans.preparation["start_step"], 40)
        self.assertGreater(np.max(out[None]), 0.0)
        np.testing.assert_allclose(out["direct"], out[None], rtol=RTOL, atol=ATOL)


class LayoutsTest(unittest.TestCase):
    """A plan per layout and population, in one loop."""

    @_float64
    def test_two_populations_three_layouts(self):
        out = {}
        for method in (None, *METHODS):
            net = _two_populations()
            result = net.run(dt=DT, duration=400 * DT, **({} if method is None else {"scheduled_delivery": method}))
            out[method] = {
                (population, name): np.asarray(block.values.to_decimal(block.values.unit))
                for population, blocks in result.samples.items()
                for name, block in blocks.items()
            }
            if method is not None:
                self.assertEqual(len(net._scheduled_plans.routes), 3)
        self.assertEqual(len(out[None]), 5)
        for method in METHODS:
            for key, reference in out[None].items():
                np.testing.assert_allclose(out[method][key], reference, rtol=RTOL, atol=ATOL, err_msg=str(key))


if __name__ == "__main__":
    unittest.main()
