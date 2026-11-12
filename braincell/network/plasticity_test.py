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


class PairReferenceTest(unittest.TestCase):
    """Acceptance 1: the pair rule against the reference and its pairing contract."""

    @_float64
    def test_pre_then_post_potentiates_by_the_reference(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.006065306597], rtol=RTOL)

    @_float64
    def test_a_scheduled_teacher_post_signal_gives_the_same_change(self) -> None:
        cell = _toy_cell()
        rule = _stdp(signals={"post": _seq([10.0])})
        _network(_seq([0.0]), cell, rule=rule).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.006065306597], rtol=RTOL)

    @_float64
    def test_post_then_pre_depresses(self) -> None:
        cell = _toy_cell(post_ms=[5.0])
        _network(_seq([15.0]), cell, rule=_stdp()).run(dt=DT, duration=17.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [-0.006368571927], rtol=RTOL)

    @_float64
    def test_pre_and_post_on_one_boundary_do_not_pair(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([10.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.0], rtol=0.0, atol=1e-15)

    @_float64
    def test_two_pre_events_on_one_boundary_count_twice(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        result = _network(_seq([0.0, 0.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.012130613194], rtol=RTOL)
        # The 0 ms boundary delivers both events with the weight before its update.
        np.testing.assert_allclose(_g(result)[1], [0.1], rtol=RTOL)

    @_float64
    def test_every_earlier_pre_pairs_with_a_post(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0, 5.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.013853314428], rtol=RTOL)


class DelayTest(unittest.TestCase):
    """Acceptance 2: the rule's pre time is the arrival boundary."""

    @_float64
    def test_a_scheduled_delay_moves_the_pre_time(self) -> None:
        cell = _toy_cell(post_ms=[10.0])
        _network(_seq([0.0]), cell, delay=2.0 * u.ms, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms)
        np.testing.assert_allclose(_w(cell) - 0.05, [0.006703200460], rtol=RTOL)

    @_float64
    def test_live_arrivals_and_first_conductance_samples(self) -> None:
        # The source crosses in step 20 (its EventSeries time is 2.0 ms).
        expected = {0.0: (0.006736800392, 21), 0.1: (0.006736800392, 22), 0.3: (0.006804506362, 24)}
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
        cell = _toy_cell()
        net = _network(_live_pre([2.1, 2.2]), cell, delay=0.2 * u.ms, rule=_stdp(A_plus=0.0 * u.uS))
        g = _g(net.run(dt=DT, duration=3.0 * u.ms))[:, 0]
        # Spikes in steps 20 and 21 arrive at 2.2 and 2.3 ms, each with w0.
        np.testing.assert_allclose(g[[22, 23, 24]], [0.0, 0.05, 0.1], rtol=RTOL)


class OrderTest(unittest.TestCase):
    """Acceptance 3: a boundary delivers with the weight before its own update."""

    @_float64
    def test_the_payload_uses_the_weight_before_the_boundary(self) -> None:
        cell = _toy_cell(post_ms=[5.0, 10.0])
        g = _g(_network(_seq([0.0, 10.0]), cell, rule=_stdp()).run(dt=DT, duration=12.0 * u.ms))[:, 0]
        np.testing.assert_allclose(g[101] - g[100], 0.057788007831, rtol=RTOL)
        np.testing.assert_allclose(_w(cell), [0.055675906206], rtol=RTOL)


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
                np.testing.assert_allclose(_w(cell), [0.04969673467, 0.02969673467], rtol=RTOL)


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
    net.connect("sched", source=stim, synapse=cell.synapses["s"], weight=0.004 * u.uS, delay=0.3 * u.ms, plasticity=rule)
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
        np.testing.assert_allclose(_w(cell) - 0.05, [0.006065306597], rtol=RTOL)

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
            np.testing.assert_allclose(w, [0.060578740926], rtol=RTOL)

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
        np.testing.assert_allclose(_w(cell), [0.060578740926], rtol=RTOL)

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


if __name__ == "__main__":
    unittest.main()
