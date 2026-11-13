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

"""Connection plasticity rules on Network and standalone Cell runs (CELL-12a).

Most targets are a toy Cell: a callable solver sets ``V`` to +40 mV in the
steps that end at the chosen post times and to -65 mV otherwise, and
integrates nothing, so a synapse's ``g`` changes only at event boundaries and
every delivered payload reads off exactly. "post at ``t``" means the toy
solver crosses threshold in the step that ends at ``t``. The rule is
``PairSTDP(A_plus=0.01 uS, A_minus=0.0105 uS, tau_plus=tau_minus=20 ms)`` and
the initial weight 0.05 uS unless a test says otherwise; ``dt`` is 0.1 ms.
"""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import CVPerBranch, Cell
from braincell.filter import at
from braincell.network._testing import make_soma_tree

DT = 0.1 * u.ms
W0 = 0.05 * u.uS
RTOL = 1e-12
E_HALF = float(np.exp(-0.5))


def _reference(value: float, published: float, decimals: int = 12) -> float:
    """``value``, the closed form of an acceptance number, after checking it rounds to ``published``.

    The acceptance numbers are rounded to ``decimals`` places, coarser than
    ``rtol=1e-12``; the tests compare with the closed form at that tolerance.
    """
    assert abs(value - published) <= 0.5 * 10.0**-decimals, (value, published)
    return value


A_PLUS, A_MINUS, TAU = 0.01, 0.0105, 20.0
DW_PRE_POST = _reference(A_PLUS * np.exp(-10.0 / TAU), 0.006065306597)


def _float64(test):
    """Run ``test`` in float64."""

    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    wrapped.__doc__ = test.__doc__
    return wrapped


def _toy_solver(post_ms):
    """Set V to +40 mV in the steps that end at ``post_ms`` (dt 0.1 ms), else -65 mV."""
    marks = np.asarray(post_ms, dtype=float)

    def solver(cell):
        t = brainstate.environ.get("t").to_decimal(u.ms)
        hit = jnp.any(jnp.abs(t - (marks - 0.1)) < 0.05) if marks.size else False
        cell.V.value = u.math.where(hit, 40.0, -65.0) * u.mV + 0.0 * cell.V.value

    return solver


def _toy_cell(post_ms=(), *, n_sites=1, size=1):
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=_toy_solver(post_ms),
    )
    for _ in range(n_sites):
        # Each place call adds an independent synapse at the same site.
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
    cell.soma.record("g", braincell.observe.synapse(name="s").state("g"))
    return cell


def _live_pre(spike_ms):
    """A toy source Cell that crosses threshold in the steps ending at ``spike_ms``."""
    return Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(1,),
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=_toy_solver(spike_ms),
    )


def _real_cell(*, e=0.0 * u.mV):
    cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
    cell.paint(braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=e))
    cell.soma.record("g", braincell.observe.synapse(name="s").state("g"))
    cell.soma.record("V", "V")
    return cell


def _stdp(**kwargs):
    params = dict(A_plus=0.01 * u.uS, A_minus=0.0105 * u.uS, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms)
    params.update(kwargs)
    signals = params.pop("signals", None)
    return braincell.mech.ConnectionRule("PairSTDP", signals=signals, **params)


def _seq(*times_ms):
    return braincell.EventSequence.from_times(tuple(np.asarray(row, dtype=float) * u.ms for row in times_ms))


def _w(cell, name="c"):
    return np.asarray(cell.connections[name].rule_state("w").to_decimal(u.uS))


def _g(result, population="post"):
    return np.asarray(result.samples[population]["g"].values.to_decimal(u.uS))


def _network(pre_model, post_cell, *, delay=0.0 * u.ms, rule=None, weight=W0, synapse_index=None):
    net = braincell.Network("plastic")
    pre = net.add_population("pre", pre_model)
    net.add_population("post", post_cell)
    synapse = post_cell.synapses["s"] if synapse_index is None else post_cell.synapses["s"][synapse_index]
    source = pre.event_outputs["spike"] if isinstance(pre_model, Cell) else pre
    net.connect("c", source=source, synapse=synapse, weight=weight, delay=delay, plasticity=rule)
    return net


#: Pre 0 and 12 ms, post 10 and 14 ms (acceptance 6e).
W_STANDALONE = _reference(
    0.05
    + A_PLUS * np.exp(-10.0 / TAU)
    - A_MINUS * np.exp(-2.0 / TAU)
    + A_PLUS * (np.exp(-14.0 / TAU) + np.exp(-2.0 / TAU)),
    0.060578740926,
)


class PairReferenceTest(unittest.TestCase):
    """Acceptance 1: the pair rule against the reference and its pairing contract."""

    @_float64
    def test_pre_then_post_potentiates_by_the_reference(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [DW_PRE_POST], rtol=RTOL)

    @_float64
    def test_a_scheduled_teacher_post_signal_gives_the_same_change(self) -> None:
        cell = _toy_cell()
        rule = _stdp(signals={"post": _seq([10.0])})
        _network(_seq([0.0]), cell, rule=rule).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [DW_PRE_POST], rtol=RTOL)

    @_float64
    def test_post_then_pre_depresses(self) -> None:
        cell = _toy_cell(post_ms=[5.0])
        _network(_seq([15.0]), cell, rule=_stdp()).run(dt=DT, duration=17.0 * u.ms)
        np.testing.assert_allclose(
            _w(cell) - 0.05, [_reference(-A_MINUS * np.exp(-10.0 / TAU), -0.006368571927)], rtol=RTOL
        )

    @_float64
    def test_pre_and_post_on_one_boundary_do_not_pair(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([10.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.0], rtol=0.0, atol=1e-15)

    @_float64
    def test_two_pre_events_on_one_boundary_count_twice(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        result = _network(_seq([0.0, 0.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [_reference(2 * DW_PRE_POST, 0.012130613194)], rtol=RTOL)
        # The 0 ms boundary delivers both events with the weight before its update.
        np.testing.assert_allclose(_g(result)[1], [0.1], rtol=RTOL)

    @_float64
    def test_every_earlier_pre_pairs_with_a_post(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0, 5.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        expected = _reference(A_PLUS * (np.exp(-10.0 / TAU) + np.exp(-5.0 / TAU)), 0.013853314428)
        np.testing.assert_allclose(_w(cell) - 0.05, [expected], rtol=RTOL)


class DelayTest(unittest.TestCase):
    """Acceptance 2: the rule's pre time is the arrival boundary."""

    @_float64
    def test_a_scheduled_delay_moves_the_pre_time(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0]), cell, delay=2.0 * u.ms, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(
            _w(cell) - 0.05, [_reference(A_PLUS * np.exp(-8.0 / TAU), 0.006703200460)], rtol=RTOL
        )

    @_float64
    def test_live_arrivals_and_first_conductance_samples(self) -> None:
        # The source crosses in step 20 (its EventSeries time is 2.0 ms), so
        # its events arrive at 2.1, 2.1 and 2.3 ms.
        at_2_1 = _reference(A_PLUS * np.exp(-7.9 / TAU), 0.006736800392)
        at_2_3 = _reference(A_PLUS * np.exp(-7.7 / TAU), 0.006804506362)
        expected = {0.0: (at_2_1, 21), 0.1: (at_2_1, 22), 0.3: (at_2_3, 24)}
        for delay_ms, (dw, first) in expected.items():
            with self.subTest(delay_ms=delay_ms):
                firsts = []
                for rule in (_stdp(), None):
                    cell = _toy_cell(post_ms=[10.0])
                    net = _network(_live_pre([2.1]), cell, delay=delay_ms * u.ms, rule=rule)
                    result = net.run(dt=DT, duration=12.0 * u.ms)
                    np.testing.assert_allclose(
                        np.asarray(result.events["pre"]["spike"].time.to_decimal(u.ms)), [2.0], rtol=RTOL
                    )
                    firsts.append(int(np.flatnonzero(_g(result)[:, 0] > 0)[0]))
                    if rule is not None:
                        np.testing.assert_allclose(_w(cell) - 0.05, [dw], rtol=RTOL)
                self.assertEqual(firsts, [first, first])

    @_float64
    def test_two_step_live_delay_reads_its_arrival_and_delivers_once(self) -> None:
        # Spikes in steps 20 and 22 arrive at 2.2 and 2.4 ms, each with w0,
        # once: as without a rule.
        for rule in (_stdp(A_plus=0.0 * u.uS), None):
            with self.subTest(rule=rule is not None):
                net = _network(_live_pre([2.1, 2.3]), _toy_cell(), delay=0.2 * u.ms, rule=rule)
                g = _g(net.run(dt=DT, duration=3.0 * u.ms))[:, 0]
                np.testing.assert_allclose(g[[22, 23, 24, 25]], [0.0, 0.05, 0.05, 0.1], rtol=RTOL)


class FanOutTest(unittest.TestCase):
    """Rows of one live call that share a source read its count history at their own delays."""

    @_float64
    def test_fan_out_rows_with_their_own_delays(self) -> None:
        # One source crossing in step 20 reaches four synapses; every row
        # pairs with the teaching post event at 10 ms from its own arrival.
        # The zero-delay row stays on the Network's immediate path, whose
        # brainevent operator reads the dynamic weight too.
        cases = {
            ("heterogeneous", "scatter"): ([0.0, 0.1, 0.3, 0.5], [2.1, 2.1, 2.3, 2.5], [21, 22, 24, 26]),
            ("heterogeneous", "brainevent"): ([0.0, 0.1, 0.3, 0.5], [2.1, 2.1, 2.3, 2.5], [21, 22, 24, 26]),
            ("uniform", "scatter"): ([0.3] * 4, [2.3] * 4, [24] * 4),
        }
        for (label, backend), (delay_ms, arrival_ms, first) in cases.items():
            with self.subTest(label=label, backend=backend):
                firsts = []
                for rule in (_stdp(A_minus=0.0 * u.uS, signals={"post": _seq([10.0])}), None):
                    cell = _toy_cell(n_sites=4)
                    net = _network(_live_pre([2.1]), cell, delay=delay_ms * u.ms, rule=rule)
                    g = _g(net.run(dt=DT, duration=12.0 * u.ms, event_backend=backend))
                    firsts.append([int(np.flatnonzero(g[:, site] > 0)[0]) for site in range(4)])
                    np.testing.assert_allclose(g[-1], [0.05] * 4, rtol=RTOL)
                    if rule is not None:
                        expected = A_PLUS * np.exp(-(10.0 - np.asarray(arrival_ms)) / TAU)
                        np.testing.assert_allclose(_w(cell) - 0.05, expected, rtol=RTOL)
                self.assertEqual(firsts, [first, first])

    @_float64
    def test_a_zero_delay_live_row_delivers_its_current_weight(self) -> None:
        # Spikes in steps 20 and 80 arrive at 2.1 and 8.1 ms; the post event
        # at 5 ms potentiates the row in between, so the second delivery
        # carries the new weight on either immediate-path backend.
        dw = A_PLUS * np.exp(-2.9 / TAU)
        for backend in ("scatter", "brainevent"):
            with self.subTest(backend=backend):
                cell = _toy_cell()
                rule = _stdp(A_minus=0.0 * u.uS, signals={"post": _seq([5.0])})
                net = _network(_live_pre([2.1, 8.1]), cell, rule=rule)
                g = _g(net.run(dt=DT, duration=9.0 * u.ms, event_backend=backend))[:, 0]
                np.testing.assert_allclose(g[[21, 80, 81]], [0.05, 0.05, 0.1 + dw], rtol=RTOL)
                np.testing.assert_allclose(_w(cell) - 0.05, [dw], rtol=RTOL)

    @_float64
    def test_a_removed_row_neither_delivers_nor_learns(self) -> None:
        cell = _toy_cell(n_sites=3)
        net = _network(
            _live_pre([2.1]),
            cell,
            delay=[0.3, 0.5, 0.3] * u.ms,
            rule=_stdp(A_minus=0.0 * u.uS, signals={"post": _seq([10.0])}),
        )
        cell.connections["c"][1].remove()
        g = _g(net.run(dt=DT, duration=12.0 * u.ms))
        np.testing.assert_allclose(g[-1], [0.05, 0.0, 0.05], rtol=RTOL)
        np.testing.assert_array_equal(cell.connections["c"].id, [0, 2])
        np.testing.assert_allclose(_w(cell) - 0.05, [A_PLUS * np.exp(-7.7 / TAU)] * 2, rtol=RTOL)


class OrderTest(unittest.TestCase):
    """Acceptance 3: a boundary delivers with the weight before its own update."""

    @_float64
    def test_the_payload_uses_the_weight_before_the_boundary(self) -> None:
        cell = _toy_cell(post_ms=[5.0, 10.0])
        g = _g(_network(_seq([0.0, 10.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms))[:, 0]
        w_after_5 = _reference(0.05 + A_PLUS * np.exp(-5.0 / TAU), 0.057788007831)
        np.testing.assert_allclose(g[101] - g[100], w_after_5, rtol=RTOL)
        w_end = w_after_5 + A_PLUS * np.exp(-10.0 / TAU) - A_MINUS * np.exp(-5.0 / TAU)
        np.testing.assert_allclose(_w(cell), [_reference(w_end, 0.055675906206)], rtol=RTOL)


class SharedSynapseTest(unittest.TestCase):
    """Acceptance 4: each row keeps its own weight, on separate or shared synapses."""

    @_float64
    def test_row_weights_evolve_alike_on_independent_and_shared_synapses(self) -> None:
        for label, n_sites, index in (("independent", 2, [0, 1]), ("shared", 1, [0, 0])):
            with self.subTest(label):
                cell = _toy_cell(post_ms=[10.0], n_sites=n_sites)
                net = _network(
                    _seq([0.0, 20.0], [0.0, 20.0]),
                    cell,
                    rule=_stdp(),
                    weight=[0.05, 0.03] * u.uS,
                    synapse_index=index,
                )
                net.run(dt=DT, duration=22.0 * u.ms)
                dw = (A_PLUS - A_MINUS) * np.exp(-10.0 / TAU)
                expected = [_reference(0.05 + dw, 0.04969673467, 11), _reference(0.03 + dw, 0.02969673467, 11)]
                np.testing.assert_allclose(_w(cell), expected, rtol=RTOL)


def _zero_rule_model(delay, rule):
    net = braincell.Network("zero")
    pre = net.add_population("pre", _live_pre([0.4, 1.9, 3.3]))
    stim = net.add_population("stim", _seq([0.5, 0.5, 2.0]))
    cell = _real_cell()
    net.add_population("post", cell)
    net.connect(
        "live",
        source=pre.event_outputs["spike"],
        synapse=cell.synapses["s"],
        weight=0.01 * u.uS,
        delay=delay,
        plasticity=rule,
    )
    net.connect(
        "sched", source=stim, synapse=cell.synapses["s"], weight=0.004 * u.uS, delay=0.3 * u.ms, plasticity=rule
    )
    return net


class NoRuleUnchangedTest(unittest.TestCase):
    """Acceptance 5(c): a rule that never changes the weight gives the no-rule run."""

    @_float64
    def test_a_zero_rule_matches_no_rule(self) -> None:
        zero = _stdp(A_plus=0.0 * u.uS, A_minus=0.0 * u.uS)
        for delay_ms in (0.0, 0.1, 0.3):
            with self.subTest(delay_ms=delay_ms):
                results = [
                    _zero_rule_model(delay_ms * u.ms, rule).run(dt=DT, duration=5.0 * u.ms) for rule in (None, zero)
                ]
                for name in ("V", "g"):
                    plain, ruled = (np.asarray(u.get_mantissa(r.samples["post"][name].values)) for r in results)
                    self.assertGreater(np.ptp(plain), 0.0)
                    np.testing.assert_allclose(ruled, plain, rtol=RTOL, atol=0.0)


def _lifecycle_model():
    net = braincell.Network("life")
    pre = net.add_population("pre", _live_pre([3.1, 17.4, 33.0, 48.2]))
    stim = net.add_population("stim", braincell.NetStim(size=1, start=1.0 * u.ms, number=50, interval=1.3 * u.ms))
    cell = _real_cell()
    net.add_population("post", cell)
    teacher = _seq(np.arange(2.0, 60.0, 3.1))
    net.connect(
        "live",
        source=pre.event_outputs["spike"],
        synapse=cell.synapses["s"],
        weight=0.002 * u.uS,
        delay=0.7 * u.ms,
        plasticity=_stdp(signals={"post": teacher}),
    )
    net.connect(
        "sched",
        source=stim,
        synapse=cell.synapses["s"],
        weight=0.001 * u.uS,
        delay=0.3 * u.ms,
        plasticity=_stdp(
            A_plus=1e-4 * u.uS,
            A_minus=1.05e-4 * u.uS,
            w_min=0.0 * u.uS,
            w_max=0.0015 * u.uS,
            signals={"post": teacher},
        ),
    )
    return net, cell


def _trajectory(result, cell):
    return (
        np.asarray(result.samples["post"]["V"].values.to_decimal(u.mV)),
        _g(result),
        _w(cell, "live"),
        _w(cell, "sched"),
    )


class LifecycleTest(unittest.TestCase):
    """Acceptance 6: segments, reset, set, Cell.reset, the state tree, Cell.run, update."""

    @_float64
    def test_segmented_and_reset_runs_match_one_run(self) -> None:
        net, cell = _lifecycle_model()
        full = _trajectory(net.run(dt=DT, duration=60.0 * u.ms), cell)
        net2, cell2 = _lifecycle_model()
        parts = [net2.run(dt=DT, duration=d * u.ms) for d in (13.0, 0.1, 21.9, 25.0)]
        split = _trajectory(braincell.NetworkResult.concat(parts), cell2)
        for got, want in zip(split, full):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
        self.assertNotEqual(float(full[2][0]), 0.002)
        self.assertTrue(np.all((full[3] >= 0.0) & (full[3] <= 0.0015)))
        net.reset_state()
        again = _trajectory(net.run(dt=DT, duration=60.0 * u.ms), cell)
        for got, want in zip(again, full):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)

    @_float64
    def test_set_weight_moves_the_current_and_the_reset_weight(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        net = _network(_seq([0.0]), cell, rule=_stdp(w_max=0.1 * u.uS))
        net.run(dt=DT, duration=12.0 * u.ms)
        view = cell.connections["c"]
        view.set(weight=0.07 * u.uS)
        np.testing.assert_allclose(_w(cell), [0.07], rtol=RTOL)
        np.testing.assert_allclose(view.weight.to_decimal(u.uS), [0.07], rtol=RTOL)
        net.run(dt=DT, duration=1.0 * u.ms)
        net.reset_state()
        np.testing.assert_allclose(_w(cell), [0.07], rtol=RTOL)

    @_float64
    def test_cell_reset_keeps_the_rule_and_starts_from_the_declared_weight(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        braincell.connect("c", source=_seq([0.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
        cell.run(dt=DT, duration=12.0 * u.ms)
        cell.connections["c"].set(weight=0.08 * u.uS)
        cell.reset()
        with self.assertRaises(RuntimeError):
            cell.connections["c"].rule_state("w")
        cell.init_state()
        np.testing.assert_allclose(_w(cell), [0.05], rtol=RTOL)
        cell.run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [DW_PRE_POST], rtol=RTOL)

    @_float64
    def test_rule_states_are_in_the_cell_state_tree(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_live_pre([2.1]), cell, delay=0.3 * u.ms, rule=_stdp()).run(dt=DT, duration=1.0 * u.ms)
        tree = {id(state) for state in brainstate.graph.states(cell).values()}
        rule_states = brainstate.graph.states(cell._connection_rules)
        self.assertGreaterEqual(len(rule_states), 3)
        self.assertTrue({id(state) for state in rule_states.values()} <= tree)
        values = {path[-1]: state.value for path, state in rule_states.items()}
        np.testing.assert_allclose(u.get_mantissa(values["w"]), _w(cell), rtol=0.0)

    @_float64
    def test_standalone_cell_run_with_its_own_spikes_as_post(self) -> None:
        weights = []
        for durations in ((16.0,), (7.0, 9.0)):
            cell = _toy_cell(post_ms=[10.0, 14.0])
            braincell.connect("c", source=_seq([0.0, 12.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
            for duration in durations:
                cell.run(dt=DT, duration=duration * u.ms)
            weights.append(_w(cell))
        for w in weights:
            np.testing.assert_allclose(w, [W_STANDALONE], rtol=RTOL)

    @_float64
    def test_standalone_update_loop_matches_cell_run(self) -> None:
        def build():
            cell = _toy_cell(post_ms=[10.0, 14.0])
            braincell.connect("c", source=_seq([0.0, 12.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
            cell.init_state()
            return cell

        cell = build()

        def step(t):
            with brainstate.environ.context(t=t):
                cell.update()

        with brainstate.environ.context(dt=DT):
            brainstate.transform.for_loop(step, u.math.arange(160) * DT)
        np.testing.assert_allclose(_w(cell), [W_STANDALONE], rtol=RTOL)

    @_float64
    def test_prepare_run_and_update_match_run(self) -> None:
        net, cell = _lifecycle_model()
        net.run(dt=DT, duration=20.0 * u.ms)
        net2, cell2 = _lifecycle_model()
        net2.prepare_run(dt=DT)
        brainstate.transform.for_loop(lambda _: net2.update(), np.arange(200))
        for name in ("live", "sched"):
            np.testing.assert_allclose(_w(cell2, name), _w(cell, name), rtol=RTOL)


def _point_target(lumped):
    morphology = braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
    if lumped:
        cell = Cell(morphology, cv_policy=braincell.Lumped(), pop_size=(2,), V_init=-65.0 * u.mV, V_th=-55.0 * u.mV)
    else:
        cell = Cell(
            morphology,
            cv_policy=CVPerBranch(1),
            pop_size=(2,),
            V_init=-65.0 * u.mV,
            V_th=-55.0 * u.mV,
            solver="ralston2",
        )
    cell.paint(braincell.mech.Channel("IL", g_max=0.5 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV))
    cell.record("V", "V")
    return cell


class LumpedTargetTest(unittest.TestCase):
    """Acceptance 7: a lumped target learns as the single-CV cable target does."""

    @_float64
    def test_lumped_and_single_cv_cable_targets_agree(self) -> None:
        out = {}
        for lumped in (True, False):
            net = braincell.Network("lumped", seed=1)
            stim = net.add_population(
                "stim",
                braincell.NetStim(size=4, start=1.0 * u.ms, number=40, interval=2.0 * u.ms, noise=0.8),
            )
            cell = _point_target(lumped)
            net.add_population("post", cell)
            net.connect(
                "c",
                source=stim,
                synapse=cell.synapses["s"][[0, 1, 0, 1]],
                weight=0.0005 * u.uS,
                delay=0.5 * u.ms,
                plasticity=_stdp(
                    A_plus=1e-4 * u.uS,
                    A_minus=1.2e-4 * u.uS,
                    w_min=0.0 * u.uS,
                    w_max=0.002 * u.uS,
                ),
            )
            result = net.run(dt=0.025 * u.ms, duration=60.0 * u.ms)
            v = np.asarray(result.samples["post"]["V"].values.to_decimal(u.mV))
            out[lumped] = (v.reshape(v.shape[0], -1), _w(cell))
        self.assertFalse(np.allclose(out[True][1], 0.0005))
        np.testing.assert_allclose(out[True][1], out[False][1], rtol=RTOL, atol=1e-15)
        np.testing.assert_allclose(out[True][0], out[False][0], rtol=0.0, atol=1e-9)


class TrainingTest(unittest.TestCase):
    """Acceptance 8: a trainable initial weight reaches the loss through the dynamic weight."""

    @_float64
    def test_autodiff_matches_central_differences(self) -> None:
        net = braincell.Network("grad")
        src = net.add_population("src", _seq([0.2, 1.0, 1.6]))
        cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
        cell.paint(braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV))
        net.add_population("post", cell)
        rule = _stdp(
            A_plus=0.002 * u.uS,
            A_minus=0.001 * u.uS,
            tau_plus=5.0 * u.ms,
            tau_minus=5.0 * u.ms,
            signals={"post": _seq([0.6, 1.3])},
        )
        net.connect("c", source=src, synapse=cell.synapses["s"], weight=0.004 * u.uS, plasticity=rule)
        net.init_state()
        cell.connections["c"].trainable(weight=braincell.trainable.scale(name="w"))
        net.prepare_run(dt=DT)
        synapse = cell.runtime.get_runtime_node(cell.synapses["s"]._store.layout_id("ExpSyn"))

        def loss():
            net.reset_state()
            brainstate.transform.for_loop(lambda _: net.update(), np.arange(20))
            return synapse.g.value.to_decimal(u.uS).sum()

        run = brainstate.transform.jit(loss)
        states = net.trainables.parameters().states()
        grad = float(brainstate.transform.jit(brainstate.transform.grad(loss, grad_states=states))()["post.w"])
        params = net.trainables.parameters()
        values = params.physical_values()
        h = 1e-6
        values["post.w"] = 1.0 + h
        params.set_physical_values(values)
        up = float(run())
        values["post.w"] = 1.0 - h
        params.set_physical_values(values)
        down = float(run())
        central = (up - down) / (2 * h)
        self.assertNotEqual(grad, 0.0)
        self.assertLessEqual(abs(grad - central) / abs(central), 1e-6)


class _TriggerSynapse(braincell.Synapse):
    """A trigger-only synapse: events carry no weight."""

    root_type = braincell.HHTypedNeuron
    states = {"g": braincell.mech.StateSpec(0.0 * u.uS)}
    event_input = braincell.mech.TriggerEventInput()

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self._init_parameters()

    def apply_events(self, payload, V_post=None):
        self.g.value = self.g.value + payload * u.uS

    def compute_derivative(self, V_post=None):
        self.g.derivative = 0.0 * self.g.value / u.ms

    def current(self, V_post):
        return self.g.value * (0.0 * u.mV - V_post)


class CounterExampleTest(unittest.TestCase):
    """Acceptance 9: each rejected declaration, with the step that rejects it."""

    def _connect(self, cell, rule, *, source=None, weight=W0, name="c"):
        source = _seq([1.0]) if source is None else source
        return braincell.connect(name, source=source, synapse=cell.synapses["s"], weight=weight, plasticity=rule)

    def test_a_trigger_target_rejects_a_rule_at_connect(self) -> None:
        cell = _toy_cell()
        cell.place(at("soma", 0.5), braincell.mech.Synapse(_TriggerSynapse, name="t"))
        with self.assertRaisesRegex(TypeError, "trigger"):
            braincell.connect("c", source=_seq([1.0]), synapse=cell.synapses["t"], weight=None, plasticity=_stdp())
        self.assertEqual(len(cell.connections), 0)

    def test_an_unregistered_rule_name_raises_key_error_at_connect(self) -> None:
        with self.assertRaisesRegex(KeyError, "NoSuchRule"):
            self._connect(_toy_cell(), braincell.mech.ConnectionRule("NoSuchRule"))

    def test_an_undeclared_signal_raises_type_error_at_connect(self) -> None:
        for name in ("voltage", "pre"):
            with self.subTest(name):
                with self.assertRaisesRegex(TypeError, "post"):
                    self._connect(_toy_cell(), _stdp(signals={name: _seq([1.0])}))

    def test_a_post_source_must_match_the_members_and_belong_to_the_target(self) -> None:
        cell = _toy_cell()
        with self.assertRaisesRegex(ValueError, "2 endpoints"):
            self._connect(cell, _stdp(signals={"post": _seq([1.0], [2.0])}))
        other = _live_pre([1.0])
        with self.assertRaisesRegex(ValueError, "target Cell"):
            self._connect(cell, _stdp(signals={"post": other.event_outputs["spike"]}))
        self.assertEqual(len(cell.connections), 0)

    def test_parameter_units_signs_bounds_and_initial_weight(self) -> None:
        cases = {
            "A_plus": _stdp(A_plus=0.01 * u.mV),
            "tau_plus": _stdp(tau_plus=0.0 * u.ms),
            "tau_minus": _stdp(tau_minus=-1.0 * u.ms),
            "w_min": _stdp(w_min=0.2 * u.uS, w_max=0.1 * u.uS),
            "outside": _stdp(w_max=0.04 * u.uS),
        }
        for label, rule in cases.items():
            with self.subTest(label):
                cell = _toy_cell()
                with self.assertRaises(ValueError):
                    self._connect(cell, rule)
                self.assertEqual(len(cell.connections), 0)
        with self.assertRaisesRegex(TypeError, "A_plus"):
            self._connect(_toy_cell(), _stdp(A_plus=0.01))

    def test_set_weight_outside_the_bounds_raises(self) -> None:
        cell = _toy_cell()
        self._connect(cell, _stdp(w_min=0.0 * u.uS, w_max=0.1 * u.uS))
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "w_max"):
            cell.connections["c"].set(weight=0.2 * u.uS)
        np.testing.assert_allclose(cell.connections["c"].weight.to_decimal(u.uS), [0.05])
        np.testing.assert_allclose(_w(cell), [0.05])

    def test_batched_init_state_is_not_implemented(self) -> None:
        cell = _toy_cell()
        self._connect(cell, _stdp())
        with self.assertRaisesRegex(NotImplementedError, "batch"):
            cell.init_state(batch_size=2)
        self.assertFalse(cell._initialized)

    def test_a_reduction_target_raises_type_error_at_init_state(self) -> None:
        cell = _toy_cell()
        cell.add_reduction("acc", braincell.PayloadAccumulatorReduction(threshold=1.0 * u.uS))
        cell.use_model("acc")
        self._connect(cell, _stdp())
        with self.assertRaisesRegex(TypeError, "reduction"):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_standalone_run_rejects_live_rule_rows(self) -> None:
        cell = _toy_cell(post_ms=[1.0])
        self._connect(cell, _stdp(), source=cell.event_outputs["spike"], weight=W0)
        with self.assertRaisesRegex(NotImplementedError, "Network"):
            cell.run(dt=DT, duration=1.0 * u.ms)

    def test_standalone_update_rejects_live_rule_rows_at_its_first_boundary(self) -> None:
        cell = _toy_cell(post_ms=[1.0])
        self._connect(cell, _stdp(), source=cell.event_outputs["spike"], weight=W0)
        cell.init_state()
        with brainstate.environ.context(dt=DT, t=0.0 * u.ms):
            with self.assertRaisesRegex(NotImplementedError, "Network"):
                cell.update()

    def test_a_declaration_must_be_a_connection_rule(self) -> None:
        with self.assertRaisesRegex(TypeError, "ConnectionRule"):
            self._connect(_toy_cell(), object())


class RuleStateTest(unittest.TestCase):
    """``ConnectionView.rule_state`` reads the rows of a view in view order."""

    def test_names_order_and_errors(self) -> None:
        cell = _toy_cell(n_sites=2)
        braincell.connect(
            "c",
            source=_seq([1.0], [2.0]),
            synapse=cell.synapses["s"],
            weight=[0.01, 0.02] * u.uS,
            plasticity=_stdp(),
        )
        braincell.connect("plain", source=_seq([1.0]), synapse=cell.synapses["s"][[0]], weight=W0)
        cell.init_state()
        view = cell.connections["c"]
        np.testing.assert_allclose(view.rule_state("w").to_decimal(u.uS), [0.01, 0.02])
        np.testing.assert_allclose(view[::-1].rule_state("w").to_decimal(u.uS), [0.02, 0.01])
        np.testing.assert_allclose(np.asarray(view.rule_state("x_pre")), [0.0, 0.0])
        with self.assertRaisesRegex(KeyError, "x_pre"):
            view.rule_state("nope")
        with self.assertRaisesRegex(TypeError, "plain"):
            cell.connections.rule_state("w")


# ---------------------------------------------------------------------------
# Recording rule states (CELL-12c). The references are closed forms and a
# float64 host rule written from the documented PairSTDP equations; neither
# calls the code under test.

STATES = ("w", "x_pre", "y_post")


def _host_pair_stdp(pre, post, w0, *, A_plus, A_minus, tau, dt, w_min=-np.inf, w_max=np.inf, set_at=None, w_set=None):
    """Float64 pair STDP on the host; row ``b`` is the state the update at boundary ``b`` receives.

    ``pre`` and ``post`` are ``(boundaries, rows)`` event counts. Each
    update decays both traces by ``exp(-dt / tau)``, adds
    ``A_plus * x_pre * post - A_minus * y_post * pre`` to the clipped
    weight, then adds the counts to the traces. ``w_set`` replaces the
    weight before the update at boundary ``set_at`` (``ConnectionView.set``
    between two runs). Returns ``(w, x_pre, y_post)``.
    """
    boundaries, rows = pre.shape
    w = np.broadcast_to(np.asarray(w0, dtype=np.float64), (rows,)).copy()
    x = np.zeros(rows)
    y = np.zeros(rows)
    decay = np.exp(-dt / tau)
    out = np.zeros((3, boundaries, rows))
    for b in range(boundaries):
        if b == set_at:
            w = np.broadcast_to(np.asarray(w_set, dtype=np.float64), (rows,)).copy()
        out[:, b] = w, x, y
        x, y = x * decay, y * decay
        w = np.clip(w + A_plus * x * post[b] - A_minus * y * pre[b], w_min, w_max)
        x, y = x + pre[b], y + post[b]
    return out[0], out[1], out[2]


def _arrivals(times_ms, delay_ms, dt_ms, boundaries):
    """Counts per boundary of events at ``times_ms`` that arrive ``delay_ms`` later, rounded half up to a boundary."""
    counts = np.zeros(boundaries)
    for t in np.atleast_1d(np.asarray(times_ms, dtype=np.float64)):
        boundary = int(np.floor((t + delay_ms) / dt_ms + 0.5))
        if 0 <= boundary < boundaries:
            counts[boundary] += 1
    return counts


def _decimal(value, unit=None):
    """``value`` as a float64 array, in ``unit`` when it has one."""
    return np.asarray(value.to_decimal(unit) if unit is not None else value, dtype=np.float64)


def _rec(samples, name, unit=None):
    """The values of the recording ``name`` as float64, in ``unit`` when it has one."""
    return _decimal(samples[name].values, unit)


def _g_of(samples):
    return _rec(samples, "g", u.uS)[:, 0]


def _unit(state):
    return u.uS if state == "w" else None


def _record_states(view, states=STATES, prefix=""):
    for state in states:
        view.record(prefix + state, state)


def _standalone_model():
    """Pre 0 and 12 ms, post 10 and 14 ms (12a acceptance 6e), every state recorded, on a standalone Cell."""
    cell = _toy_cell(post_ms=[10.0, 14.0])
    view = braincell.connect("c", source=_seq([0.0, 12.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
    _record_states(view)
    return cell


def _recorded_lifecycle_model():
    """The 12a acceptance 6(a) model, every state of both calls recorded as ``<call>_<state>``."""
    net, cell = _lifecycle_model()
    for name in ("live", "sched"):
        _record_states(cell.connections[name], prefix=f"{name}_")
    return net, cell


LIFECYCLE_RECORDINGS = tuple(f"{name}_{state}" for name in ("live", "sched") for state in STATES)


def _assert_same_recordings(test, got, want, names, rtol=RTOL, atol=1e-15):
    for name in names:
        unit = u.uS if name.endswith("w") else None
        test.assertEqual(got[name].values.shape, want[name].values.shape, name)
        np.testing.assert_allclose(_rec(got, name, unit), _rec(want, name, unit), rtol=rtol, atol=atol, err_msg=name)


class RecordRuleStateTest(unittest.TestCase):
    """``ConnectionView.record``: sample ``k`` is the state before the update at ``time[k]`` (CELL-12c)."""

    @staticmethod
    def _both_drivers(pre_ms, post_ms, states, *, duration=12.0 * u.ms):
        """The samples of the same one-row model, run by a standalone ``Cell.run`` and by a Network."""
        cell = _toy_cell(post_ms=post_ms)
        view = braincell.connect("c", source=_seq(pre_ms), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
        _record_states(view, states)
        post = _toy_cell(post_ms=post_ms)
        net = _network(_seq(pre_ms), post, rule=_stdp())
        _record_states(post.connections["c"], states)
        return cell.run(dt=DT, duration=duration).samples, net.run(dt=DT, duration=duration).samples["post"]

    @_float64
    def test_both_drivers_record_the_closed_form(self) -> None:
        k = np.arange(120)
        expected = {
            "w": np.where(k <= 100, 0.05, 0.05 + A_PLUS * E_HALF),
            "x_pre": np.where(k == 0, 0.0, np.exp(-(k - 1) / 200.0)),
            "y_post": np.where(k <= 100, 0.0, np.exp(-(k - 101) / 200.0)),
        }
        standalone, network = self._both_drivers([0.0], [10.0], STATES)
        for samples in (standalone, network):
            for state, reference in expected.items():
                self.assertEqual(samples[state].values.shape, (120, 1))
                np.testing.assert_allclose(_rec(samples, state, _unit(state))[:, 0], reference, rtol=RTOL, atol=1e-15)
                (row,) = samples[state].schema.rows
                self.assertEqual(
                    (row.population_index, row.cv_id, row.synapse_id, row.connection_id, row.source_index), (0,) * 5
                )
                self.assertEqual(
                    (row.mechanism_category, row.mechanism_type, row.mechanism_name, row.field),
                    ("connection_rule", "PairSTDP", "c", state),
                )
                self.assertEqual(row.unit, _unit(state))
            self.assertEqual(float(samples["w"].time[0].to_decimal(u.ms)), 0.0)
        _assert_same_recordings(self, standalone, network, STATES)

    @_float64
    def test_the_recorded_weight_is_the_delivered_weight(self) -> None:
        w100 = _reference(0.05 + A_PLUS * np.exp(-0.25), 0.057788007831)
        w101 = _reference(w100 + A_PLUS * E_HALF - A_MINUS * np.exp(-0.25), 0.055675906206)
        for samples in self._both_drivers([0.0, 10.0], [5.0, 10.0], ("w",)):
            w = _rec(samples, "w", u.uS)[:, 0]
            g = _g_of(samples)
            np.testing.assert_allclose([w[0], w[100], w[101]], [0.05, w100, w101], rtol=RTOL)
            # The jump of g at a boundary is the weight sampled there.
            np.testing.assert_allclose([g[1] - g[0], g[101] - g[100]], [w[0], w[100]], rtol=RTOL)

    @_float64
    def test_live_arrivals_with_their_delays(self) -> None:
        # The source crosses threshold in the step ending at 2.1 ms (detected at step 20).
        late = _reference(A_PLUS * np.exp(-7.9 / TAU), 0.006736800392)
        later = _reference(A_PLUS * np.exp(-7.7 / TAU), 0.006804506362)
        # delay, first x_pre row, first g row (a zero delay acts at the end of the detecting step), change of w
        for delay, first_x, first_g, change in ((0.0, 22, 21, late), (0.1, 22, 22, late), (0.3, 24, 24, later)):
            post = _toy_cell(post_ms=[10.0])
            net = _network(_live_pre([2.1]), post, delay=delay * u.ms, rule=_stdp())
            post.connections["c"].record("w", "w")
            post.connections["c"].record("x", "x_pre")
            samples = net.run(dt=DT, duration=12.0 * u.ms).samples["post"]
            x, w, g = _rec(samples, "x")[:, 0], _rec(samples, "w", u.uS)[:, 0], _g_of(samples)
            self.assertEqual((int(np.flatnonzero(x)[0]), x[first_x]), (first_x, 1.0))
            self.assertEqual(int(np.flatnonzero(g)[0]), first_g)
            np.testing.assert_allclose(g[first_g] - g[first_g - 1], w[first_g], rtol=RTOL)
            self.assertEqual(w[100], 0.05)
            np.testing.assert_allclose(w[101] - 0.05, change, rtol=RTOL)

    @_float64
    def test_period_frequency_and_start_choose_the_samples(self) -> None:
        for driver in ("cell", "network"):
            cell = _toy_cell(post_ms=[10.0])
            if driver == "cell":
                braincell.connect("c", source=_seq([0.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
                model = cell
            else:
                model = _network(_seq([0.0]), cell, rule=_stdp())
            view = cell.connections["c"]
            view.record("every", "x_pre")
            view.record("coarse", "x_pre", period=1.0 * u.ms, start=0.5 * u.ms)
            view.record("w_khz", "w", frequency=1.0 * u.kHz)
            result = model.run(dt=DT, duration=12.0 * u.ms)
            samples = result.samples if driver == "cell" else result.samples["post"]
            self.assertEqual(samples["coarse"].values.shape, (12, 1))
            np.testing.assert_allclose(samples["coarse"].time.to_decimal(u.ms), np.arange(0.5, 12.0, 1.0), rtol=RTOL)
            np.testing.assert_array_equal(_rec(samples, "coarse"), _rec(samples, "every")[5::10])
            np.testing.assert_allclose(
                _rec(samples, "w_khz", u.uS)[10:12, 0], [0.05, 0.05 + A_PLUS * E_HALF], rtol=RTOL
            )

    @_float64
    def test_rows_follow_their_synapse_onto_a_new_grid(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        view = braincell.connect("c", source=_seq([0.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
        view.record("w", "w")
        cell.cv_policy = CVPerBranch(3)
        samples = cell.run(dt=DT, duration=12.0 * u.ms).samples
        (row,), (g_row,) = samples["w"].schema.rows, samples["g"].schema.rows
        self.assertEqual((row.cv_id, row.point_id), (1, 2))
        self.assertEqual((row.cv_id, row.point_id, row.branch_id), (g_row.cv_id, g_row.point_id, g_row.branch_id))
        k = np.arange(120)
        np.testing.assert_allclose(
            _rec(samples, "w", u.uS)[:, 0], np.where(k <= 100, 0.05, 0.05 + A_PLUS * E_HALF), rtol=RTOL
        )


class RecordRuleStateRowsTest(unittest.TestCase):
    """The rows of a rule-state recording are the view's connected rows, in view order."""

    @staticmethod
    def _model():
        """Rule calls ``ee`` (IDs 0-2) and ``ff`` (IDs 3-4) and a call ``plain`` without a rule (ID 5)."""
        net = braincell.Network("rows")
        pre = net.add_population("pre", _seq([0.0], [1.0], [2.0]))
        stim = net.add_population("stim", braincell.NetStim(size=1, start=4.0 * u.ms, number=3, interval=2.0 * u.ms))
        cell = _toy_cell(post_ms=[5.0, 9.0], n_sites=3)
        net.add_population("post", cell)
        synapse = cell.synapses["s"]
        net.connect(
            "ee",
            source=pre,
            synapse=synapse,
            weight=[0.05, 0.04, 0.03] * u.uS,
            delay=[0.0, 0.2, 0.4] * u.ms,
            plasticity=_stdp(),
        )
        net.connect(
            "ff",
            source=pre.event_outputs["event"][[2, 0]],
            synapse=synapse[[0, 1]],
            weight=[20.0, 10.0] * u.nS,
            plasticity=_stdp(A_plus=2.0 * u.nS, A_minus=1.0 * u.nS),
        )
        net.connect("plain", source=stim, synapse=synapse[2], weight=0.01 * u.uS)
        return net, cell

    @staticmethod
    def _across(cell):
        """The ``ff`` rows, then the ``ee`` rows."""
        return cell.connections[[3, 4, 0, 1, 2]]

    @_float64
    def test_rows_follow_the_view_across_calls(self) -> None:
        net, cell = self._model()
        across = self._across(cell)
        np.testing.assert_array_equal(across.id, [3, 4, 0, 1, 2])
        across.record("w", "w")
        across.record("x", "x_pre")
        cell.connections["ee"][[2, 0]].record("picked", "w")
        samples = net.run(dt=DT, duration=10.0 * u.ms).samples["post"]

        rows = samples["w"].schema.rows
        self.assertEqual([row.connection_id for row in rows], [3, 4, 0, 1, 2])
        self.assertEqual([row.source_index for row in rows], [2, 0, 0, 1, 2])
        self.assertEqual([row.synapse_id for row in rows], [0, 1, 0, 1, 2])
        self.assertEqual([row.mechanism_name for row in rows], ["ff", "ff", "ee", "ee", "ee"])
        # connect gives the weights the synapse's event unit, uS.
        self.assertEqual({row.unit for row in rows}, {u.uS})
        self.assertEqual([row.connection_id for row in samples["picked"].schema.rows], [2, 0])

        post = np.zeros(100)
        post[[50, 90]] = 1.0
        ee_pre = np.stack([_arrivals([t], d, 0.1, 100) for t, d in ((0.0, 0.0), (1.0, 0.2), (2.0, 0.4))], axis=1)
        ff_pre = np.stack([_arrivals([2.0], 0.0, 0.1, 100), _arrivals([0.0], 0.0, 0.1, 100)], axis=1)
        ee_w, ee_x, _ = _host_pair_stdp(
            ee_pre, np.stack([post] * 3, axis=1), [0.05, 0.04, 0.03], A_plus=A_PLUS, A_minus=A_MINUS, tau=TAU, dt=0.1
        )
        ff_w, ff_x, _ = _host_pair_stdp(
            ff_pre, np.stack([post] * 2, axis=1), [20.0, 10.0], A_plus=2.0, A_minus=1.0, tau=TAU, dt=0.1
        )
        np.testing.assert_allclose(
            _rec(samples, "w", u.uS), np.concatenate([ff_w / 1000.0, ee_w], axis=1), rtol=RTOL, atol=1e-15
        )
        np.testing.assert_allclose(_rec(samples, "x"), np.concatenate([ff_x, ee_x], axis=1), rtol=RTOL, atol=1e-15)
        np.testing.assert_allclose(_rec(samples, "picked", u.uS), ee_w[:, [2, 0]], rtol=RTOL, atol=1e-15)

        # The same views read by rule_state before each Network.update().
        net2, cell2 = self._model()
        net2.prepare_run(dt=DT)
        across2, picked2 = self._across(cell2), cell2.connections["ee"][[2, 0]]

        def step(_):
            values = (across2.rule_state("w"), across2.rule_state("x_pre"), picked2.rule_state("w"))
            net2.update()
            return values

        w, x, picked = brainstate.transform.for_loop(step, np.arange(100))
        np.testing.assert_allclose(_rec(samples, "w", u.uS), _decimal(w, u.uS), rtol=RTOL, atol=1e-15)
        np.testing.assert_allclose(_rec(samples, "x"), _decimal(x), rtol=RTOL, atol=1e-15)
        np.testing.assert_allclose(_rec(samples, "picked", u.uS), _decimal(picked, u.uS), rtol=RTOL, atol=1e-15)

    @_float64
    def test_removed_rows_are_not_recorded(self) -> None:
        cell = _toy_cell(post_ms=[10.0], n_sites=3)
        view = braincell.connect(
            "c",
            source=_seq([0.0], [0.5], [1.0]),
            synapse=cell.synapses["s"],
            weight=[0.05, 0.04, 0.03] * u.uS,
            plasticity=_stdp(),
        )
        view.record("w", "w")
        view[[1]].record("only_removed", "w")
        cell.connections["c"][[1]].remove()
        first = cell.run(dt=DT, duration=12.0 * u.ms).samples
        self.assertEqual([row.connection_id for row in first["w"].schema.rows], [0, 2])
        self.assertEqual(first["w"].values.shape, (120, 2))
        self.assertEqual(first["only_removed"].values.shape, (120, 0))
        post = np.zeros((120, 2))
        post[100] = 1.0
        pre = np.stack([_arrivals([0.0], 0.0, 0.1, 120), _arrivals([1.0], 0.0, 0.1, 120)], axis=1)
        reference, _, _ = _host_pair_stdp(pre, post, [0.05, 0.03], A_plus=A_PLUS, A_minus=A_MINUS, tau=TAU, dt=0.1)
        np.testing.assert_allclose(_rec(first, "w", u.uS), reference, rtol=RTOL, atol=1e-15)
        cell.reset()
        second = cell.run(dt=DT, duration=12.0 * u.ms).samples
        _assert_same_recordings(self, second, first, ("w", "only_removed"))

    def test_only_a_cell_outside_a_network_keeps_a_copy(self) -> None:
        def count_states(*, record, in_network):
            cell = _toy_cell(post_ms=[10.0])
            if in_network:
                model = _network(_seq([0.0]), cell, rule=_stdp())
            else:
                model = cell
                braincell.connect("c", source=_seq([0.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
            if record:
                cell.connections["c"].record("w", "w")
                cell.connections["c"].record("x", "x_pre")
            model.init_state()
            return len(brainstate.graph.states(cell))

        alone = count_states(record=False, in_network=False)
        self.assertEqual(count_states(record=True, in_network=False), alone + 2)
        in_network = count_states(record=False, in_network=True)
        self.assertEqual(count_states(record=True, in_network=True), in_network)


class RecordRuleStateLoopTest(unittest.TestCase):
    """Rule-state recordings against update loops, segments, reset_state and scheduled delivery."""

    @_float64
    def test_network_update_loop_reads_sample_k_before_update_k(self) -> None:
        net, _ = _recorded_lifecycle_model()
        samples = net.run(dt=DT, duration=60.0 * u.ms).samples["post"]
        net2, cell2 = _recorded_lifecycle_model()
        net2.prepare_run(dt=DT)

        def step(_):
            values = {
                f"{name}_{state}": cell2.connections[name].rule_state(state)
                for name in ("live", "sched")
                for state in STATES
            }
            net2.update()
            return values

        read = brainstate.transform.for_loop(step, np.arange(600))
        for name, value in read.items():
            unit = u.uS if name.endswith("w") else None
            np.testing.assert_allclose(_decimal(value, unit), _rec(samples, name, unit), rtol=RTOL, atol=1e-15)
        # The scheduled call's weight reaches its bound.
        self.assertEqual(_rec(samples, "sched_w", u.uS).max(), 0.0015)

    @_float64
    def test_standalone_update_loop_reads_two_samples_ahead(self) -> None:
        cell = _standalone_model()
        samples = cell.run(dt=DT, duration=16.0 * u.ms).samples
        np.testing.assert_allclose(_w(cell), [W_STANDALONE], rtol=RTOL)
        np.testing.assert_allclose(_rec(samples, "w", u.uS)[-1], [W_STANDALONE], rtol=RTOL)
        post = np.zeros((160, 1))
        post[[100, 140]] = 1.0
        pre = _arrivals([0.0, 12.0], 0.0, 0.1, 160)[:, None]
        reference = _host_pair_stdp(pre, post, [0.05], A_plus=A_PLUS, A_minus=A_MINUS, tau=TAU, dt=0.1)
        for state, want in zip(STATES, reference):
            np.testing.assert_allclose(_rec(samples, state, _unit(state)), want, rtol=RTOL, atol=1e-15)

        cell2 = _standalone_model()
        cell2.init_state()

        def step(t):
            with brainstate.environ.context(t=t):
                cell2.update()
            return {state: cell2.connections["c"].rule_state(state) for state in STATES}

        with brainstate.environ.context(dt=DT):
            read = brainstate.transform.for_loop(step, u.math.arange(159) * DT)
        for state in STATES:
            # rule_state after update j (from 0) is sample j + 2.
            np.testing.assert_allclose(
                _decimal(read[state], _unit(state))[:158], _rec(samples, state, _unit(state))[2:], rtol=RTOL, atol=1e-15
            )

    @_float64
    def test_scheduled_delivery_keeps_the_rule_rows(self) -> None:
        def lifecycle(drive):
            net, cell = _recorded_lifecycle_model()
            stim = net.add_population("drive", drive)
            net.connect("plain", source=stim, synapse=cell.synapses["s"], weight=0.0005 * u.uS, delay=0.2 * u.ms)
            return net

        def sparse():
            return lifecycle(braincell.NetStim(size=1, start=0.5 * u.ms, number=40, interval=1.1 * u.ms))

        def dense():
            return lifecycle(braincell.NetStim(size=1, start=0.0 * u.ms, number=600, interval=0.1 * u.ms))

        def across_calls():
            net, cell = RecordRuleStateRowsTest._model()
            RecordRuleStateRowsTest._across(cell).record("w", "w")
            return net

        cases = (
            (across_calls, 10.0, ("w",), ("direct",)),
            (sparse, 60.0, LIFECYCLE_RECORDINGS, ("direct",)),
            # "padded" needs about one record per step (CELL-9).
            (dense, 60.0, LIFECYCLE_RECORDINGS, ("direct", "padded")),
        )
        for build, duration, names, methods in cases:
            default = build().run(dt=DT, duration=duration * u.ms).samples["post"]
            for method in methods:
                planned = build().run(dt=DT, duration=duration * u.ms, scheduled_delivery=method).samples["post"]
                _assert_same_recordings(self, planned, default, names)
        net = dense()
        net.prepare_run(dt=DT, scheduled_delivery="direct")
        cell = net.populations["post"].cell

        def step(_):
            values = (cell.connections["live"].rule_state("w"), cell.connections["sched"].rule_state("w"))
            net.update()
            return values

        live, sched = brainstate.transform.for_loop(step, np.arange(600))
        default = dense().run(dt=DT, duration=60.0 * u.ms).samples["post"]
        np.testing.assert_allclose(_decimal(live, u.uS), _rec(default, "live_w", u.uS), rtol=RTOL)
        np.testing.assert_allclose(_decimal(sched, u.uS), _rec(default, "sched_w", u.uS), rtol=RTOL)

    @_float64
    def test_segments_and_reset_state_match_one_run(self) -> None:
        net, _ = _recorded_lifecycle_model()
        one = net.run(dt=DT, duration=60.0 * u.ms)
        net.reset_state()
        joined = braincell.NetworkResult.concat([net.run(dt=DT, duration=d * u.ms) for d in (13.0, 0.1, 21.9, 25.0)])
        net.reset_state()
        again = net.run(dt=DT, duration=60.0 * u.ms)
        for other in (joined, again):
            _assert_same_recordings(self, other.samples["post"], one.samples["post"], LIFECYCLE_RECORDINGS)
        np.testing.assert_array_equal(
            joined.samples["post"]["live_w"].time.to_decimal(u.ms), one.samples["post"]["live_w"].time.to_decimal(u.ms)
        )

        cell = _standalone_model()
        one = cell.run(dt=DT, duration=16.0 * u.ms)
        cell.reset_state()
        joined = braincell.RunResult.concat(
            [cell.run(dt=DT, duration=7.0 * u.ms), cell.run(dt=DT, duration=9.0 * u.ms)]
        )
        cell.reset_state()
        again = cell.run(dt=DT, duration=16.0 * u.ms)
        for other in (joined, again):
            _assert_same_recordings(self, other.samples, one.samples, STATES)

    @_float64
    def test_a_weight_set_between_runs_shows_when_it_is_delivered(self) -> None:
        # The boundary at 5 ms ends the first run. A standalone Cell has
        # prepared it before set(), so its event still carries 0.05 uS
        # (CELL-55); a Network prepares it in the next run.
        for driver, rows in (("cell", [0.05, 0.08]), ("network", [0.08, 0.08])):
            cell = _toy_cell()
            if driver == "cell":
                braincell.connect("c", source=_seq([5.0]), synapse=cell.synapses["s"], weight=W0, plasticity=_stdp())
                model = cell
            else:
                model = _network(_seq([5.0]), cell, rule=_stdp())
            cell.connections["c"].record("w", "w")
            model.run(dt=DT, duration=5.0 * u.ms)
            cell.connections["c"].set(weight=0.08 * u.uS)
            result = model.run(dt=DT, duration=1.0 * u.ms)
            samples = result.samples if driver == "cell" else result.samples["post"]
            w, g = _rec(samples, "w", u.uS)[:, 0], _g_of(samples)
            np.testing.assert_allclose(w[:2], rows, rtol=RTOL)
            np.testing.assert_allclose(g[1] - g[0], w[0], rtol=RTOL)


class WeightSetBetweenRunsTest(unittest.TestCase):
    """CELL-55: ``set(weight=)`` between two runs acts from the next run's first boundary, in either driver.

    dt 0.1 ms; the first run ends at t_n = 5 ms (n = 50), where the weight
    goes from 0.05 to 0.08 uS. The source's events at 2 ms and around t_n
    (L94) arrive, rounded half up, at boundary 20; 49 (4.94 ms); 50 (4.95
    ms, a tie, 4.96, 5.0 and 5.04 ms); 51 (5.05 ms, a tie, and 5.06 ms).
    The toy target does not integrate its synapse, so g only accumulates:
    the payload of boundary b is sample b + 1 minus sample b. The Cell
    crosses threshold in the steps ending at 3 and 5 ms, its post events at
    boundaries 30 and 50. A standalone Cell used to have prepared t_n, with
    the old weight and the rule already run there, before ``set`` (main
    34973d6: 4 x 0.05 uS at t_n, and the rule's update at t_n lost).
    """

    N1, N2 = 50, 20
    W1 = 0.08
    TIMES = (2.0, 4.94, 4.95, 4.96, 5.0, 5.04, 5.05, 5.06)
    #: Arrivals per boundary, worked out by hand above.
    COUNTS = {20: 1, 49: 1, 50: 4, 51: 2}

    def _model(self, driver, *, source, delay, rule):
        cell = _toy_cell(post_ms=[3.0, 5.0])
        if source == "sequence":
            model = _seq([t - delay for t in self.TIMES])
        else:
            # 4.94 + 0.03 k ms (k < 5): boundaries 49, 50 (4.97, 5.0, 5.03 ms) and 51 (5.06 ms).
            model = braincell.NetStim(size=1, start=4.94 * u.ms, interval=0.03 * u.ms, number=5)
        if driver == "cell":
            braincell.connect(
                "c", source=model, synapse=cell.synapses["s"], weight=W0, delay=delay * u.ms, plasticity=rule
            )
            owner = cell
        else:
            owner = _network(model, cell, delay=delay * u.ms, rule=rule)
        if rule is not None:
            cell.connections["c"].record("w", "w")
        return owner, cell

    def _counts(self, source):
        if source == "sequence":
            return self.COUNTS
        return {49: 1, 50: 3, 51: 1}

    def _reference(self, source, rule_kwargs, *, set_at=N1):
        """Payload per boundary and ``w(t_b-)`` on the host, with the new weight from boundary ``set_at``.

        One boundary more than the two runs step over: ``w(t_70-)`` is what
        ``rule_state`` reads after them (the rule has run at t_69).
        """
        boundaries = self.N1 + self.N2 + 1
        pre = np.zeros((boundaries, 1))
        for b, n in self._counts(source).items():
            pre[b] = n
        if rule_kwargs is None:
            w = np.where(np.arange(boundaries) < self.N1, 0.05, self.W1)[:, None]
        else:
            post = np.zeros((boundaries, 1))
            post[[30, 50]] = 1.0
            w = _host_pair_stdp(
                pre,
                post,
                [0.05],
                A_plus=A_PLUS,
                A_minus=A_MINUS,
                tau=TAU,
                dt=0.1,
                set_at=set_at,
                w_set=[self.W1],
                **rule_kwargs,
            )[0]
        return (pre * w)[:, 0], w[:, 0]

    def _run_twice(self, owner, cell, driver):
        samples, g_after = [], []
        for steps in (self.N1, self.N2):
            if samples:
                cell.connections["c"].set(weight=self.W1 * u.uS)
            result = owner.run(dt=DT, duration=steps * DT)
            block = result.samples if driver == "cell" else result.samples["post"]
            # One sample per step before reading jumps from them (L93).
            self.assertEqual(block["g"].values.shape[0], steps)
            samples.append(block)
            g_after.append(float(_decimal(cell.synapses["s"].g, u.uS).reshape(-1)[0]))
        return samples, g_after

    @_float64
    def test_both_drivers_deliver_the_new_weight_from_the_next_boundary(self) -> None:
        cases = [
            ("sequence", 0.0, None),
            ("sequence", 0.3, None),
            ("netstim", 0.0, None),
            ("sequence", 0.0, {}),
            ("sequence", 0.3, {}),
            ("netstim", 0.0, {}),
            # w_min binds at t_n only on the old weight: 0.04996 - 0.01945 < 0.035 < 0.08 - 0.01945.
            ("sequence", 0.0, {"w_min": 0.035}),
        ]
        for source, delay, rule_kwargs in cases:
            payload, w_before = self._reference(source, rule_kwargs)
            rule = None if rule_kwargs is None else _stdp(**{k: v * u.uS for k, v in rule_kwargs.items()})
            for driver in ("network", "cell"):
                with self.subTest(source=source, delay=delay, rule=rule_kwargs, driver=driver):
                    owner, cell = self._model(driver, source=source, delay=delay, rule=rule)
                    samples, g_after = self._run_twice(owner, cell, driver)
                    g = np.concatenate([_g_of(samples[0]), _g_of(samples[1]), [g_after[1]]])
                    self.assertAlmostEqual(g_after[0], g[self.N1], delta=1e-15)
                    np.testing.assert_allclose(np.diff(g), payload[:-1], rtol=RTOL, atol=1e-15)
                    if rule is not None:
                        rows = np.concatenate([_rec(block, "w", u.uS)[:, 0] for block in samples])
                        np.testing.assert_allclose(rows, w_before[:-1], rtol=RTOL, atol=1e-15)
                        np.testing.assert_allclose(_w(cell), w_before[-1:], rtol=RTOL, atol=1e-15)

    @_float64
    def test_an_update_loop_delivers_the_new_weight_from_the_next_boundary(self) -> None:
        payload, w_before = self._reference("sequence", {})
        for driver in ("network", "cell"):
            with self.subTest(driver=driver):
                owner, cell = self._model(driver, source="sequence", delay=0.0, rule=_stdp())
                if driver == "network":
                    owner.prepare_run(dt=DT)
                else:
                    cell.init_state()

                def step(k):
                    if driver == "network":
                        owner.update()
                    else:
                        with brainstate.environ.context(t=k * DT):
                            cell.update()
                    return cell.synapses["s"].g.to_decimal(u.uS), cell.connections["c"].rule_state("w").to_decimal(u.uS)

                read = []
                with brainstate.environ.context(dt=DT):
                    for first, steps in ((0, self.N1), (self.N1, self.N2)):
                        if first:
                            cell.connections["c"].set(weight=self.W1 * u.uS)
                        read.append(brainstate.transform.for_loop(step, np.arange(first, first + steps)))
                g = np.concatenate([[0.0], *(_decimal(part[0]).reshape(-1) for part in read)])
                w = np.concatenate([_decimal(part[1]).reshape(-1) for part in read])
                np.testing.assert_allclose(np.diff(g), payload[:-1], rtol=RTOL, atol=1e-15)
                # After update k (from 0) the rule has run at boundary k: w(t_{k+1}-). The read
                # after update n - 1 comes before set(), so it is w(t_n-) of the old weight.
                expected = w_before[1:].copy()
                expected[self.N1 - 1] = self._reference("sequence", {}, set_at=None)[1][self.N1]
                np.testing.assert_allclose(w, expected, rtol=RTOL, atol=1e-15)


class RecordRuleStateFloat32Test(unittest.TestCase):
    """A float32 run of 32010 steps (800.25 ms) with two NetStim rows into one synapse (L77)."""

    DT = 0.025 * u.ms
    STEPS = 32010
    RULE = dict(
        A_plus=1e-4 * u.uS,
        A_minus=1.05e-4 * u.uS,
        tau_plus=20.0 * u.ms,
        tau_minus=20.0 * u.ms,
        w_min=0.0 * u.uS,
        w_max=0.002 * u.uS,
    )

    @staticmethod
    def _sources():
        pre = braincell.NetStim(size=2, start=[1.0, 2.0] * u.ms, number=[205, 117], interval=[4.0, 7.0] * u.ms)
        post = braincell.NetStim(size=1, start=3.0 * u.ms, number=165, interval=5.0 * u.ms)
        return pre, post

    def _connect(self, connect, cell, pre, post):
        view = connect(
            "c",
            source=pre,
            synapse=cell.synapses["s"],
            weight=[0.001, 0.0015] * u.uS,
            delay=[0.3, 1.1] * u.ms,
            plasticity=_stdp(signals={"post": post}, **self.RULE),
        )
        _record_states(view)

    def _network(self):
        net = braincell.Network("f32")
        pre, post = self._sources()
        source = net.add_population("pre", pre)
        cell = _real_cell()
        net.add_population("post", cell)
        self._connect(net.connect, cell, source, post)
        return net, cell

    @staticmethod
    def _arrays(samples):
        return {state: _rec(samples, state, _unit(state)) for state in STATES}

    def test_long_float32_run(self) -> None:
        steps, dt = self.STEPS, self.DT
        net, _ = self._network()
        one = net.run(dt=dt, duration=steps * dt).samples["post"]
        self.assertEqual(one["w"].values.dtype, jnp.float32)
        self.assertEqual(one["w"].values.shape, (steps, 2))
        want = self._arrays(one)

        def same(got):
            for state in STATES:
                atol = 1e-6 * float(np.abs(want[state]).max())
                np.testing.assert_allclose(got[state], want[state], rtol=0.0, atol=atol, err_msg=state)

        net.reset_state()
        parts = [net.run(dt=dt, duration=12000 * dt), net.run(dt=dt, duration=20010 * dt)]
        same(self._arrays(braincell.NetworkResult.concat(parts).samples["post"]))

        net2, cell2 = self._network()
        net2.prepare_run(dt=dt)

        def step(_):
            values = {state: cell2.connections["c"].rule_state(state) for state in STATES}
            net2.update()
            return values

        read = brainstate.transform.for_loop(step, np.arange(steps))
        same({state: _decimal(read[state], _unit(state)) for state in STATES})
        self.assertEqual(float(cell2.current_time.to_decimal(u.ms)), 800.25)

        pre, post = self._sources()
        cell3 = _real_cell()
        self._connect(braincell.connect, cell3, pre, post)
        same(self._arrays(cell3.run(dt=dt, duration=steps * dt).samples))
        cell3.reset_state()
        parts = [cell3.run(dt=dt, duration=12000 * dt), cell3.run(dt=dt, duration=20010 * dt)]
        same(self._arrays(braincell.RunResult.concat(parts).samples))

        pre_counts = np.stack(
            [
                _arrivals(1.0 + 4.0 * np.arange(205), 0.3, 0.025, steps),
                _arrivals(2.0 + 7.0 * np.arange(117), 1.1, 0.025, steps),
            ],
            axis=1,
        )
        post_counts = np.stack([_arrivals(3.0 + 5.0 * np.arange(165), 0.0, 0.025, steps)] * 2, axis=1)
        reference = _host_pair_stdp(
            pre_counts,
            post_counts,
            [0.001, 0.0015],
            A_plus=1e-4,
            A_minus=1.05e-4,
            tau=TAU,
            dt=0.025,
            w_min=0.0,
            w_max=0.002,
        )
        for state, ref, bound in zip(STATES, reference, (1e-4, 2e-4, 2e-4)):
            self.assertLessEqual(float(np.abs(want[state] - ref).max()), bound * float(np.abs(ref).max()), state)
        self.assertEqual(int(np.sum(reference[0] == 0.0)), 4552)


class RecordRuleStatePytreeTest(unittest.TestCase):
    """Results that hold rule-state recordings are pytrees as other results are."""

    @_float64
    def test_jit_and_tree_utilities(self) -> None:
        cell = _standalone_model()
        cell.init_state()
        jitted = brainstate.transform.jit(lambda: cell.run(dt=DT, duration=16.0 * u.ms))()
        eager = _standalone_model().run(dt=DT, duration=16.0 * u.ms)
        _assert_same_recordings(self, jitted.samples, eager.samples, STATES)
        other = _standalone_model().run(dt=DT, duration=16.0 * u.ms)
        self.assertEqual(jax.tree.structure(other), jax.tree.structure(eager))
        self.assertEqual(hash(jax.tree.structure(other)), hash(jax.tree.structure(eager)))
        self.assertIn((160, 1), [np.shape(leaf) for leaf in jax.tree.leaves(eager.samples["w"])])
        self.assertIs(jax.block_until_ready(eager), eager)

        net, _ = _recorded_lifecycle_model()
        result = net.run(dt=DT, duration=5.0 * u.ms)
        mapped = jax.tree.map(lambda leaf: leaf, result)
        _assert_same_recordings(self, mapped.samples["post"], result.samples["post"], LIFECYCLE_RECORDINGS)


if __name__ == "__main__":
    unittest.main()
