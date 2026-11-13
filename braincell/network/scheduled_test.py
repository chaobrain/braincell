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
from braincell.filter import AllRegion, LocsetMask, at
from braincell.network import engine, scheduled
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

    @staticmethod
    def _plain_and_ruled():
        """Synapse 0 takes a plain row from NetStim ``plain``, synapse 1 a PairSTDP row from NetStim ``ruled``."""
        branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV)
        cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        cell.place(LocsetMask.from_columns([0, 0], [0.5, 0.5]), bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms))
        net = bc.Network("rule", seed=1)
        post = net.add_population("post", cell)
        # Generated on demand (number 10**9): the 10 ms runs double both schedules twice.
        plain = net.add_population("plain", bc.NetStim(size=1, start=0.3 * u.ms, interval=0.5 * u.ms, number=10**9))
        ruled = net.add_population("ruled", bc.NetStim(size=1, start=0.2 * u.ms, interval=0.5 * u.ms, number=10**9))
        net.connect("drive", source=plain, synapse=post.synapses["exp"][[0]], weight=0.001 * u.uS)
        net.connect("learn", source=ruled, synapse=post.synapses["exp"][[1]], weight=0.001 * u.uS, plasticity=_rule())
        cell.record("g", bc.observe.synapse(name="exp").state("g"))
        return net

    def _four_runs(self, method):
        """Per-step g of four 10 ms runs, and the loop traces of each run."""
        net = self._plain_and_ruled()
        extra = {} if method is None else {"scheduled_delivery": method}
        builds, traces = self._counting()
        parts, counts = [], []
        with builds, traces as traced:
            for _ in range(4):
                before = traced.call_count
                g = np.asarray(u.get_mantissa(net.run(dt=DT, duration=400 * DT, **extra).samples["post"]["g"].values))
                self.assertEqual(g.shape[0], 400)
                parts.append(g.reshape(400, 2))
                counts.append(traced.call_count - before)
        return np.concatenate(parts), counts

    @_float64
    def test_a_netstim_read_by_a_rule_keys_the_planned_loop(self):
        """The rule reads its source's events when the loop is traced, so that schedule keys the loop (L40)."""
        reference, _ = self._four_runs(None)
        self.assertTrue(np.all(np.isfinite(reference)))
        self.assertEqual(np.count_nonzero(np.diff(reference, axis=0) > 0, axis=0).tolist(), [80, 80])
        got, traces = self._four_runs("direct")
        # Both schedules double in runs 2 and 3; the rule row's source traces the loop again.
        self.assertEqual(traces, [1, 1, 1, 0])
        np.testing.assert_array_equal(got, reference)
        # Were the rule row's source out of the key, the loop of run 1 would keep its 21 events.
        with mock.patch.object(engine, "loop_schedule_key", lambda populations, plans: ()):
            dropped, traces = self._four_runs("direct")
        self.assertEqual(traces, [1, 0, 0, 0])
        np.testing.assert_array_equal(dropped[:, 0], reference[:, 0])
        self.assertEqual(np.count_nonzero(np.diff(dropped[:, 1]) > 0), 21)
        self.assertFalse(np.array_equal(dropped[:, 1], reference[:, 1]))


#: Records of the 40-step (1 ms) windows of :meth:`WindowSlotsTest._model`, by hand. Window ``w`` takes
#: the arrivals in ``[w - 0.0125, w + 0.9875)`` ms; source ``i`` starts at ``(i + 0.25) / 1000`` ms and the
#: delay is 0.15 ms, so window 0 takes sources 0..837 (start below 0.8375 ms), windows 1-4 one event of
#: every source, window 5 the fifth event of sources 838..999, and windows 6 and 7 none.
SLOT_RECORDS = (838, 1000, 1000, 1000, 1000, 162, 0, 0)
#: Slots of every such window: twice the first window's records plus a block. On main ``84378a3`` they
#: followed each window, ``_power_of_two(records + 128)``: 1024, 2048, ..., 512, 128.
SLOTS = 2 * 838 + scheduled.BLOCK_SIZE


class WindowSlotsTest(unittest.TestCase):
    """CELL-73: "direct" windows of one length keep the record slots, and the compiled loop, of the largest."""

    ROWS, WINDOW = 1000, 40

    def _model(self):
        branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=1, V_init=-65.0 * u.mV)
        # One synapse per row, so that the default path's sums are the plan's (see the module docstring).
        cell.place(
            LocsetMask.from_columns(np.zeros(self.ROWS, dtype=int), np.full(self.ROWS, 0.5)),
            bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms),
        )
        net = bc.Network("slots", seed=1)
        post = net.add_population("post", cell)
        start = (np.arange(self.ROWS) + 0.25) / self.ROWS * u.ms
        stim = net.add_population("stim", bc.NetStim(size=self.ROWS, start=start, interval=1.0 * u.ms, number=5))
        net.connect(
            "drive",
            source=stim,
            synapse=post.synapses["exp"][np.arange(self.ROWS)],
            weight=0.001 * u.uS,
            delay=0.15 * u.ms,
        )
        cell.record("g", bc.observe.synapse(name="exp").state("g"))
        return net, cell

    def _runs(self, net, lengths, method):
        """Run windows of ``lengths`` steps; per run the g samples, the plan's slots, its records if built, its traces."""
        traces = mock.patch.object(
            scheduled._LayoutInput, "deliver", autospec=True, side_effect=scheduled._LayoutInput.deliver
        )
        builds = mock.patch.object(scheduled, "_build_plans", wraps=scheduled._build_plans)
        extra = {} if method is None else {"scheduled_delivery": method}
        out = []
        with traces as traced, builds as built:
            for steps in lengths:
                before = (traced.call_count, built.call_count)
                result = net.run(dt=DT, duration=steps * DT, **extra)
                g = np.asarray(u.get_mantissa(result.samples["post"]["g"].values))
                self.assertEqual(g.shape[0], steps)  # one sample per step
                if method is None:
                    out.append((g, None, None, None))
                    continue
                plans = net._scheduled_plans
                ids = plans.windows[next(reversed(plans.windows))][0][0]
                records = plans.preparation["layouts"][0]["records"] if built.call_count > before[1] else None
                out.append((g, ids.shape, records, traced.call_count - before[0]))
        return out

    def _check(self, rtol=RTOL):
        lengths = (self.WINDOW,) * len(SLOT_RECORDS)
        ref_net, _ = self._model()
        reference = np.concatenate([run[0] for run in self._runs(ref_net, lengths, None)])
        self.assertTrue(np.all(np.isfinite(reference)))
        self.assertGreater(np.count_nonzero(np.diff(reference, axis=0)), 0)
        net, _ = self._model()
        first = self._runs(net, lengths, "direct")
        # The cache keeps four windows, so after reset_state each of the eight is built again.
        net.reset_state()
        again = self._runs(net, lengths, "direct")
        self.assertEqual([run[2] for run in first], list(SLOT_RECORDS))
        self.assertEqual([run[2] for run in again], list(SLOT_RECORDS))
        # The first window traces the loop and no later one retraces it (main: 1, 1, 0, 0, 0, 1, 1, 0, then
        # 0 after reset_state): every window has one shape.
        self.assertEqual([run[3] for run in first + again], [1] + [0] * 15)
        self.assertEqual({run[1] for run in first + again}, {(SLOTS,)})
        self.assertEqual(net._scheduled_plans.slots, {("post", net._scheduled_plans.routes[0].layout_id, 40): SLOTS})
        for runs in (first, again):
            _assert_same(self, {"g": np.concatenate([run[0] for run in runs])}, {"g": reference}, rtol=rtol)

    @_float64
    def test_windows_of_one_length_share_one_loop(self):
        self._check()

    def test_float32(self):
        with brainstate.environ.context(precision=32):
            self._check(rtol=1e-6)

    @_float64
    def test_each_window_length_keeps_its_own_slots(self):
        """A longer window's slots do not reach the shorter windows, whose loop it would retrace."""
        net, _ = self._model()
        runs = self._runs(net, (40, 40, 400, 40), "direct")
        # [2, 12) ms: three events of sources 0..837 and four of 838..999; [12, 13) ms: none.
        self.assertEqual([run[2] for run in runs], [838, 1000, 838 * 3 + 162 * 4, 0])
        # main: 1, 1, 1, 1 (the first 1000 records took 2048 slots, the empty window 128).
        self.assertEqual([run[3] for run in runs], [1, 0, 1, 0])
        self.assertEqual([run[1] for run in runs], [(SLOTS,), (SLOTS,), (2 * 3162 + 128,), (SLOTS,)])


class ArrivalsTest(unittest.TestCase):
    """The window readers against every arrival computed directly, for consecutive and seeking windows."""

    DT_MS = 0.025
    SOURCE_INDEX = np.array([0, 0, 1, 2, 3, 4, 4, 2])
    DELAY_MS = np.array([0.0, 0.5, 0.0, 1.0, 0.25, 0.0, 0.0, 1.0])  # rows 5, 6 and 3, 7 share a group

    def _expected(self, times, mask, start, stop):
        from braincell.network.event import _host_arrival_steps

        pairs = []
        for row, (source, delay) in enumerate(zip(self.SOURCE_INDEX, self.DELAY_MS)):
            steps = _host_arrival_steps(times[source][mask[source]], delay, self.DT_MS)
            pairs += [(int(step), row) for step in steps if start <= step < stop]
        return sorted(pairs)

    def _check(self, table, times, mask, windows):
        for start, stop in windows:
            steps, rows = table.window(start, stop)
            self.assertEqual(sorted(zip(steps.tolist(), rows.tolist())), self._expected(times, mask, start, stop))

    def test_netstim_windows_follow_a_growing_schedule(self):
        # Too many events to generate at once (see NetStim), so the schedule grows on demand.
        stim = bc.NetStim(size=5, start=0.2 * u.ms, interval=0.3 * u.ms, number=2**20, noise=0.7, seed=3)
        groups = scheduled._Groups.build(self.SOURCE_INDEX, self.DELAY_MS)
        self.assertEqual(groups.source.size, 6)
        stim._ensure_schedule(until_ms=10.0)
        table = scheduled._ScheduleArrivals(groups, stim, self.DT_MS)
        # Consecutive windows, a seek back, one from step 0, and one past the generated events.
        times, mask = stim._event_times_ms, stim._event_mask
        self._check(table, times, mask, [(0, 100), (100, 250), (250, 400), (37, 90), (0, 400), (10**8, 10**8 + 9)])
        # A stale or wrong cursor is checked, not trusted.
        table.cursor = (150, np.zeros(groups.source.size, dtype=np.int64))
        self._check(table, times, mask, [(150, 300)])
        table.cursor = (300, np.full(groups.source.size, 10**6))
        self._check(table, times, mask, [(300, 380)])

        width = times.shape[1]
        stim._ensure_schedule(until_ms=120.0)
        self.assertGreater(stim._event_times_ms.shape[1], width)
        # The wider schedule keeps the first events, so the next window starts at the last cursor.
        np.testing.assert_array_equal(stim._event_times_ms[:, :width][mask], times[mask])
        grown = scheduled._ScheduleArrivals(groups, stim, self.DT_MS, table.cursor)
        times, mask = stim._event_times_ms, stim._event_mask
        self._check(grown, times, mask, [(380, 1000), (1000, 4000), (123, 4800), (0, 2**31 - 1)])

    def test_a_window_after_the_schedule_grew_reads_its_own_events(self):
        """The work of a window is that of its arrivals, not of every generated event (acceptance 7)."""
        size = 50
        stim = bc.NetStim(size=size, start=0.05 * u.ms, interval=0.1 * u.ms, number=2**20, seed=3)
        groups = scheduled._Groups.build(np.arange(size), np.zeros(size))
        stim._ensure_schedule(until_ms=10.0)
        table = scheduled._ScheduleArrivals(groups, stim, self.DT_MS)
        table.window(0, 400)
        stim._ensure_schedule(until_ms=1000.0)
        grown = scheduled._ScheduleArrivals(groups, stim, self.DT_MS, table.cursor)
        with mock.patch.object(scheduled, "_host_arrival_steps", wraps=scheduled._host_arrival_steps) as rounded:
            steps, _ = grown.window(400, 800)
        self.assertEqual(steps.size, size * 100)  # 10 ms of events at 0.1 ms
        computed = sum(np.size(call.args[0]) for call in rounded.call_args_list)
        # The cursor's two checks, the window's events, and one event past its stop, per source.
        self.assertEqual(computed, size * (2 + 100 + 1))
        self.assertGreater(np.count_nonzero(stim._event_mask), 50 * computed)

    def test_event_sequence_table(self):
        rng = np.random.default_rng(2)
        events = [np.sort(rng.uniform(0.0, 30.0, size)) for size in (0, 3, 40, 7, 1)]
        source = bc.EventSequence.from_times(tuple(t * u.ms for t in events))
        groups = scheduled._Groups.build(self.SOURCE_INDEX, self.DELAY_MS)
        table = scheduled._SequenceArrivals.build(groups, source, self.DT_MS)
        width = max(map(len, events))
        times = np.array([np.pad(t, (0, width - len(t))) for t in events])
        mask = np.arange(width)[None, :] < np.array([len(t) for t in events])[:, None]
        self._check(table, times, mask, [(0, 300), (300, 1300), (17, 29), (0, 2**31 - 1)])


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


def _overflow_model(interval_ms=10.0, number=10):
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
        "stim", bc.NetStim(size=10, start=0.5 * np.arange(10) * u.ms, interval=interval_ms * u.ms, number=number)
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
        """Its plan holds only the 100 steps with arrivals: one slot per record (CELL-65).

        On main 6629a87 it held the 3781 steps to the last arrival, 37.8
        slots per record, and ``prepare_run`` raised.
        """
        reference, cell = _overflow_model()
        reference.prepare_run(dt=DT)
        expected = UpdatePathTest._update_values(reference, cell, 4000)
        net, cell = _overflow_model()
        net.prepare_run(dt=DT, scheduled_delivery="padded")
        (layout,) = net._scheduled_plans.preparation["layouts"]
        self.assertEqual(
            (layout["records"], layout["stored_steps"], layout["n_steps"], layout["padding_ratio"]),
            (100, 100, 3781, 1.0),
        )
        _assert_same(self, UpdatePathTest._update_values(net, cell, 4000), expected)

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


#: The tail model's windows: 10 ms each (400 steps).
TAIL_STEPS = 400
TAIL_ROWS = 1000


def _tail_starts_ms():
    return np.asarray(brainstate.random.RandomState(7).uniform(0.0, 1.0, TAIL_ROWS))


def _tail_model(*, source="netstim", layout="independent", synapse="ExpSyn"):
    """CELL-53: a schedule whose second window holds only its tail, on 10 cells of 100 ExpSyn.

    ``NetStim(size=1000, start=U[0, 1) ms, interval=1 ms, number=11)``,
    delay 0.1 ms, weight 0.001 uS: the window [0, 400) holds the first ten
    events of each row (1.26 slots per record), [400, 800) only the
    eleventh (11.1 slots per record, above the limit of 8) at the same
    width, so its padded plan is no larger than the first one's.
    ``source="sequence"`` gives the same times as an EventSequence;
    ``layout="shared"`` sends the 1000 rows to one synapse per cell.
    """
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), pop_size=10, V_init=-65.0 * u.mV)
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    per_cell = 100 if layout == "independent" else 1
    cell.place(
        LocsetMask.from_columns(np.zeros(per_cell, dtype=int), np.full(per_cell, 0.5)),
        bc.mech.Synapse(synapse, name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
    )
    net = bc.Network("tail", seed=1)
    post = net.add_population("post", cell)
    starts = _tail_starts_ms()
    if source == "netstim":
        model = bc.NetStim(size=TAIL_ROWS, start=starts * u.ms, interval=1.0 * u.ms, number=11, seed=7)
    else:
        model = bc.EventSequence.from_times(tuple((start + np.arange(11.0)) * u.ms for start in starts))
    stim = net.add_population("stim", model)
    # Synapse id = placement * 10 + member: row r reaches member r % 10.
    targets = np.arange(TAIL_ROWS) if layout == "independent" else np.arange(TAIL_ROWS) % 10
    net.connect("drive", source=stim, synapse=post.synapses["exp"][targets], weight=0.001 * u.uS, delay=0.1 * u.ms)
    cell.record("v", "V")
    cell.record("g", bc.observe.synapse(name="exp").state("g"))
    if synapse == "ExpSynSTD":
        cell.record("x", bc.observe.synapse(name="exp").state("x"))
    return net, cell


def _tail_windows(net, cell, method=None):
    extra = {} if method is None else {"scheduled_delivery": method}
    parts, layouts = [], []
    for _ in range(2):
        parts.append(_values(net.run(dt=DT, duration=TAIL_STEPS * DT, **extra), cell))
        if method is not None:
            layouts.append(net._scheduled_plans.preparation["layouts"][0])
    return _concat(parts), layouts


def _tail_key(net):
    (route,) = net._scheduled_plans.routes
    return route.population, route.layout_id


class LaterWindowTest(unittest.TestCase):
    """CELL-53: a later window no larger than a padded plan the Network accepted has no ratio limit."""

    _assert_unchanged = PaddingLimitTest._assert_unchanged

    def _check(self, rtol=RTOL, v_atol=None, **model):
        net, cell = _tail_model(**model)
        reference, _ = _tail_windows(net, cell)
        self.assertEqual(reference["v"].shape[0], 2 * TAIL_STEPS)  # one sample per step
        self.assertGreater(np.max(reference["g"]), 0.0)
        net, cell = _tail_model(**model)
        got, (first, second) = _tail_windows(net, cell, "padded")
        self.assertLessEqual(first["padding_ratio"], 8.0)
        self.assertGreater(second["padding_ratio"], 8.0)
        self.assertLessEqual(second["array_bytes"], first["array_bytes"])
        _assert_same(self, got, reference, rtol=rtol, v_atol=v_atol)
        return net, cell, reference

    @_float64
    def test_the_tail_of_a_schedule_runs_padded(self):
        net, cell, reference = self._check()
        # A run again after reset_state hits the cache: no plan is built.
        with mock.patch.object(scheduled, "_build_plans", wraps=scheduled._build_plans) as built:
            net.reset_state()
            again, _ = _tail_windows(net, cell, "padded")
        self.assertEqual(built.call_count, 0)
        _assert_same(self, again, reference)

    @_float64
    def test_shared_synapses(self):
        self._check(layout="shared")

    @_float64
    def test_sum_count_layout(self):
        self._check(synapse="ExpSynSTD")

    @_float64
    def test_event_sequence_source(self):
        self._check(source="sequence")

    def test_float32(self):
        with brainstate.environ.context(precision=32):
            self._check(rtol=1e-6, v_atol=1e-6)

    @_float64
    def test_the_exemption_is_up_to_the_accepted_bytes(self):
        net, cell = _tail_model()
        net.run(dt=DT, duration=TAIL_STEPS * DT)
        reference = _values(net.run(dt=DT, duration=TAIL_STEPS * DT), cell)
        for margin, raises in ((0, False), (1, True)):
            with self.subTest(margin=margin):
                net, cell = _tail_model()
                net.run(dt=DT, duration=TAIL_STEPS * DT, scheduled_delivery="padded")
                plans = net._scheduled_plans
                accepted = plans.preparation["layouts"][0]["array_bytes"]  # the second window's bytes too
                self.assertEqual(plans.accepted, {_tail_key(net): accepted})
                with mock.patch.dict(plans.accepted, {_tail_key(net): accepted - margin}):
                    if raises:
                        before = _state(net, cell)
                        with self.assertRaises(PaddingLimitError) as caught:
                            net.run(dt=DT, duration=TAIL_STEPS * DT, scheduled_delivery="padded")
                        self.assertEqual(plans.accepted, {_tail_key(net): accepted - margin})
                        self._assert_unchanged(before, _state(net, cell))
                        self.assertIn("11.1 slots per record", str(caught.exception))
                    else:
                        got = _values(net.run(dt=DT, duration=TAIL_STEPS * DT, scheduled_delivery="padded"), cell)
                        _assert_same(self, got, reference)

    @_float64
    def test_a_first_sparse_window_still_raises(self):
        # The same tail with no accepted plan before it: after a topology
        # change the plans, and what they accepted, are dropped.
        net, cell = _tail_model()
        net.run(dt=DT, duration=TAIL_STEPS * DT, scheduled_delivery="padded")
        net._mark_topology_changed()
        self.assertIsNone(net._scheduled_plans)
        before = _state(net, cell)
        with self.assertRaisesRegex(PaddingLimitError, "11.1 slots per record"):
            net.run(dt=DT, duration=TAIL_STEPS * DT, scheduled_delivery="padded")
        self._assert_unchanged(before, _state(net, cell))

    @_float64
    def test_a_refused_plan_is_not_accepted(self):
        net, cell = _overflow_model()
        with self.assertRaises(PaddingLimitError):
            net.run(dt=DT, duration=100.0 * u.ms, scheduled_delivery="padded")
        self.assertEqual(net._scheduled_plans.accepted, {})
        net.run(dt=DT, duration=8 * DT, scheduled_delivery="padded")
        self.assertEqual(net._scheduled_plans.accepted, {_tail_key(net): (2 * 8 * 1 + 10 + 1) * 4})


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
    def test_a_schedule_far_longer_than_the_loops(self):
        """The dense schedule and one more event at 100 s, 4e6 steps on (CELL-65).

        On main 6629a87 the plan of every event held every step to it, and
        ``"padded"`` raised (``PaddingLimitError``, 6.54e3 slots per record).
        """
        self._check(source=_sequence((*DENSE_MS, 1.0e5)), steps=1000)
        # The plan holds only its steps with arrivals; a run between loops reuses it.
        self._check(source=_sequence((*DENSE_MS, 1.0e5)), run_first=300, steps=700)

    @_float64
    def test_a_run_before_prepare_run_and_a_schedule_far_longer(self):
        # CELL-56 (d) with the plan of the steps with arrivals: 4 steps run, then the update() loops.
        out = {}
        for method in (None, *METHODS):
            net, cell = _model(source=_sequence((*DENSE_MS, 1.0e5)))
            extra = {} if method is None else {"scheduled_delivery": method}
            head = _values(net.run(dt=DT, duration=4 * DT, **extra), cell)
            net.prepare_run(dt=DT, **extra)
            if method is not None:
                (layout,) = net._scheduled_plans.preparation["layouts"]
                self.assertLess(layout["stored_steps"], layout["n_steps"])
            out[method] = (head, self._update_values(net, cell, 996))
        self.assertGreater(np.max(out[None][1]["g"]), 0.0)
        for method in METHODS:
            for got, reference in zip(out[method], out[None]):
                _assert_same(self, got, reference)

    @_float64
    def test_a_schedule_without_events(self):
        out = self._check(source=bc.NetStim(size=SIZE, number=0), steps=100)
        self.assertEqual(np.max(np.abs(out[None][0]["g"])), 0.0)

    @_float64
    def test_steps_after_the_last_arrival(self):
        """The sparse schedule's last arrival is near step 88: later steps read no record."""
        out = self._check(source=_sequence(SPARSE_MS), steps=200)
        self.assertGreater(np.max(out[None][0]["g"]), 0.0)
        self.assertEqual(out[None][0]["g"].shape[0], 200)


#: The step limit model (CELL-56): ``prepare_run`` generates the events up to 2**31 steps of 0.5 ms.
LIMIT_DT = 0.5 * u.ms


def _limit_model(noise=0.0):
    """``NetStim(size=2, start=[1, 3] ms, interval=1e4 ms, number=2**21)`` into one ExpSyn, delay 0.3 ms.

    At dt 0.5 ms ``prepare_run`` generates the 107 375 events of each
    source up to 2**31 steps; event k arrives at step 2.6 + 2e4 k and 6.6 +
    2e4 k, rounded to 3 + 2e4 k and 7 + 2e4 k, the last ones at
    2 147 480 003 and 2 147 480 007.
    """
    cell = bc.Cell.point(1, V_init=-65.0 * u.mV)
    cell.place(at("soma", 0.5), bc.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV))
    net = bc.Network("limit", seed=1)
    post = net.add_population("post", cell)
    stim = net.add_population(
        "stim",
        bc.NetStim(size=2, start=[1.0, 3.0] * u.ms, interval=1.0e4 * u.ms, number=2**21, noise=noise, seed=11),
    )
    net.connect("drive", source=stim, synapse=post.synapses["exp"][[0, 0]], weight=0.001 * u.uS, delay=0.3 * u.ms)
    return net, cell


class EveryEventPlanTest(unittest.TestCase):
    """CELL-65: the plan of ``prepare_run`` holds the steps with arrivals, not every step to the last one."""

    @staticmethod
    def _plan(method, **model):
        net, _ = _overflow_model(**model)
        with mock.patch.object(
            scheduled._ScheduleArrivals, "_steps", autospec=True, side_effect=scheduled._ScheduleArrivals._steps
        ) as reads:
            net.prepare_run(dt=DT, scheduled_delivery=method)
        (layout,) = net._scheduled_plans.preparation["layouts"]
        (arrays,) = net._prepared_plans
        return [np.shape(array) for array in arrays], layout, reads.call_count

    def test_the_plan_grows_with_the_events_not_the_steps(self):
        # 10 sources of 10 events, 0.5 ms apart: 100 records on 100 steps. An
        # interval 1000 times longer spans 1000 times the steps. On main
        # 6629a87 "direct" stored an offset per step (3782 and 3600182) and
        # "padded" raised (37.8 slots per record).
        for method in METHODS:
            with self.subTest(method=method):
                short, short_layout, _ = self._plan(method)
                long, long_layout, _ = self._plan(method, interval_ms=1.0e4)
                self.assertEqual(short, long)
                self.assertEqual((short_layout["n_steps"], long_layout["n_steps"]), (3781, 3600181))
                for layout in (short_layout, long_layout):
                    self.assertEqual((layout["records"], layout["stored_steps"]), (100, 100))
                limit = scheduled._power_of_two(100 + scheduled.BLOCK_SIZE)
                self.assertLessEqual(max(max(shape, default=1) for shape in long), limit)
                # Twice the events: twice the steps stored.
                _, more, _ = self._plan(method, number=20)
                self.assertEqual((more["records"], more["stored_steps"]), (200, 200))

    def test_a_plan_within_budget_stores_every_step_as_a_window(self):
        # Model M: 3730 records whose steps span [0, 3679), within either budget.
        for method in METHODS:
            with self.subTest(method=method):
                net, _ = _model()
                net.prepare_run(dt=DT, scheduled_delivery=method)
                (layout,) = net._scheduled_plans.preparation["layouts"]
                self.assertEqual((layout["records"], layout["n_steps"]), (3730, 3679))
                self.assertEqual(layout["stored_steps"], 3679)
                (full,) = net._prepared_plans
                (window,) = net._scheduled_plans.window(0, 3679)
                full, window = [np.asarray(x) for x in full], [np.asarray(x) for x in window]
                if method == "direct":
                    # The same records in other slots: the plan of every event has the power of two
                    # above its records and a block, a window twice its records and a block (CELL-73).
                    self.assertEqual((full[0].shape, window[0].shape), ((4096,), (2 * 3730 + 128,)))
                    for index in (0, 1):
                        np.testing.assert_array_equal(full[index][:3730], window[index][:3730])
                        self.assertFalse(full[index][3730:].any() or window[index][3730:].any())
                    full, window = full[2:], window[2:]
                for got, expected in zip(full, window, strict=True):
                    np.testing.assert_array_equal(got, expected)

    def test_where_a_plan_of_every_event_stops_storing_every_step(self):
        route = mock.Mock(targets=np.zeros(1, dtype=np.int32), population="post")

        def stored(method, steps, accepted=0):
            steps = np.asarray(steps, dtype=np.int64)
            _, stats = scheduled._layout_arrays(method, route, steps, np.zeros_like(steps), 0, None, accepted)
            return stats["stored_steps"], stats["n_steps"]

        # "direct": no more steps than its 256 record slots (1 record + 128, a power of two).
        self.assertEqual(stored("direct", [254]), (255, 255))
        self.assertEqual(stored("direct", [255]), (1, 256))
        # "padded": the window [0, span) within the limits, 8 slots per record at most.
        self.assertEqual(stored("padded", [*range(9), 79]), (80, 80))
        self.assertEqual(stored("padded", [*range(9), 80]), (10, 81))
        # ... or no larger than a plan accepted before (CELL-53): (2 * 81 + 1 + 1) * 4 bytes.
        self.assertEqual(stored("padded", [*range(9), 80], accepted=656), (81, 81))
        self.assertEqual(stored("padded", [*range(9), 80], accepted=655), (10, 81))

    def test_every_arrival_is_read_in_one_pass(self):
        # On main 6629a87 a pass per event of the longest row: 10 and 1000.
        for number in (10, 1000):
            with self.subTest(number=number):
                _, layout, reads = self._plan("direct", number=number)
                self.assertEqual(layout["records"], 10 * number)
                self.assertEqual(reads, 1)

    @_float64
    def test_a_schedule_to_the_step_limit(self):
        last = 2 * (3 + 107374 * 10**4) + 1  # 2 147 480 007
        values = jnp.asarray([0.25, 0.5])
        for method in METHODS:
            with self.subTest(method=method):
                net, cell = _limit_model()
                net.prepare_run(dt=LIMIT_DT, scheduled_delivery=method)
                (layout,) = net._scheduled_plans.preparation["layouts"]
                # The events past the limit, clipped to it, are not planned: one record per stored step.
                self.assertEqual(
                    (layout["records"], layout["stored_steps"], layout["n_steps"], layout["width"]),
                    (214750, 214750, last + 1, 1),
                )
                # The rows' arrays and one int32 per stored step: no array of the span.
                self.assertLess(layout["array_bytes"], 32 * 214750)
                plans = net._scheduled_plans
                (route,) = plans.routes
                full = scheduled._LayoutInput(method, route, net._prepared_plans[0])
                for start in (0, 1073740000, 2147480000, 2**31 - 65):
                    steps = np.arange(start, start + 64)
                    expected = np.where(steps % 20000 == 3, 0.25, np.where(steps % 20000 == 7, 0.5, 0.0))
                    expected[steps > last] = 0.0
                    window = scheduled._LayoutInput(method, route, plans.window(start, 64)[0])
                    for plan in (full, window):
                        got = jax.lax.map(lambda step: plan._deliver(values, step), jnp.asarray(steps, jnp.int32))
                        np.testing.assert_array_equal(np.asarray(got)[:, 0], expected, err_msg=f"{method} {start}")
                g = UpdatePathTest._update_values(net, cell, 20)["g"]
                reference, cell = _limit_model()
                reference.prepare_run(dt=LIMIT_DT)
                np.testing.assert_array_equal(g, UpdatePathTest._update_values(reference, cell, 20)["g"])
                self.assertGreater(g[3, 0], 0.0)

    def test_a_noisy_schedule_to_the_step_limit(self):
        # As CELL-56's sentinel test: no step past the limit is stored, and no step holds the clipped arrivals.
        for method in METHODS:
            with self.subTest(method=method):
                net, _ = _limit_model(noise=1.0)
                net.prepare_run(dt=LIMIT_DT, scheduled_delivery=method)
                (layout,) = net._scheduled_plans.preparation["layouts"]
                self.assertLessEqual(layout["width"], 2)
                self.assertLess(layout["stored_steps"], layout["n_steps"])
                self.assertLess(layout["n_steps"], 2**31)
                steps = np.asarray(net._prepared_plans[0][-1])
                self.assertEqual(steps.shape, (layout["stored_steps"],))
                self.assertTrue(np.all(np.diff(steps) > 0))


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
