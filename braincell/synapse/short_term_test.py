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

"""``ExpSynSTD`` and the ``"sum_count"`` event input on every delivery path (CELL-12b).

Most targets are a toy Cell, as in the plasticity tests: a callable solver
sets ``V`` to +40 mV in the steps that end at the chosen spike times and to
-65 mV otherwise, and integrates nothing, so ``g`` and ``x`` change only at
event boundaries. The synapse is ``ExpSynSTD(tau=2 ms, U=0.5, tau_rec=800 ms)``,
the weight 0.001 uS and ``dt`` 0.1 ms unless a test says otherwise. A
sample at ``t`` holds the state before the events of the boundary ``t``.
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
from braincell.synapse import ExpSynSTD

DT = 0.1 * u.ms
W = 0.001
U = 0.5
TAU_REC = 800.0
RTOL = 1e-12


def _reference(value: float, published: float, decimals: int = 12) -> float:
    """``value``, the closed form of an acceptance number, after checking it rounds to ``published``."""
    assert abs(value - published) <= 0.5 * 10.0**-decimals, (value, published)
    return value


def _float64(test):
    """Run ``test`` in float64."""

    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    wrapped.__doc__ = test.__doc__
    return wrapped


def _std(**kwargs):
    params = dict(tau=2.0 * u.ms, U=U, tau_rec=TAU_REC * u.ms)
    params.update(kwargs)
    return braincell.mech.Synapse("ExpSynSTD", name="s", **params)


def _toy_solver(spike_ms):
    """Set V to +40 mV in the steps that end at ``spike_ms`` (dt 0.1 ms), else -65 mV."""
    marks = np.asarray(spike_ms, dtype=float)

    def solver(cell):
        t = brainstate.environ.get("t").to_decimal(u.ms)
        hit = jnp.any(jnp.abs(t - (marks - 0.1)) < 0.05) if marks.size else False
        cell.V.value = u.math.where(hit, 40.0, -65.0) * u.mV + 0.0 * cell.V.value

    return solver


def _toy_cell(spike_ms=(), *, n_sites=1):
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(1,),
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=_toy_solver(spike_ms),
    )
    for _ in range(n_sites):
        # Each place call adds an independent synapse, with its own resource, at the same site.
        cell.place(at("soma", 0.5), _std())
    _record(cell)
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


def _record(cell):
    cell.record("g", braincell.observe.synapse(name="s").state("g"))
    cell.record("x", braincell.observe.synapse(name="s").state("x"))


def _seq(*times_ms):
    return braincell.EventSequence.from_times(tuple(np.asarray(row, dtype=float) * u.ms for row in times_ms))


def _samples(samples):
    """``(g in uS, x)`` of one Cell's samples, ``(time, synapse)``."""
    return (
        np.asarray(samples["g"].values.to_decimal(u.uS)),
        np.asarray(u.get_mantissa(samples["x"].values)),
    )


def _network(pre_model, post_cell, *, synapse_index, delay=0.0 * u.ms, weight=W * u.uS, rule=None):
    net = braincell.Network("std")
    pre = net.add_population("pre", pre_model)
    net.add_population("post", post_cell)
    source = pre.event_outputs["spike"] if isinstance(pre_model, Cell) else pre
    net.connect(
        "c",
        source=source,
        synapse=post_cell.synapses["s"][synapse_index],
        weight=weight,
        delay=delay,
        plasticity=rule,
    )
    return net


def _run_network(net, duration_ms, **kwargs):
    return _samples(net.run(dt=DT, duration=duration_ms * u.ms, **kwargs).samples["post"])


#: Released by one event from x = 1, by a second one, and by two on one boundary.
DG_FIRST = _reference(W * U, 0.0005)
DG_SECOND = _reference(W * U * (1.0 - U), 0.00025)
DG_TWO = _reference(2 * W / 2 * (1.0 - (1.0 - U) ** 2), 0.00075)


class ModelTest(unittest.TestCase):
    """The release formula on the runtime class itself."""

    @_float64
    def test_one_call_releases_as_successive_events(self) -> None:
        syn = ExpSynSTD(size=4, U=0.3)
        syn.init_state()
        syn.apply_events(np.array([0.0, 0.001, 0.003, 0.004]) * u.uS, count=np.array([0.0, 1.0, 3.0, 2.0]))
        kept = 0.7 ** np.array([0.0, 1.0, 3.0, 2.0])
        mean = np.array([0.0, 0.001, 0.001, 0.002])
        np.testing.assert_allclose(syn.g.value.to_decimal(u.uS), mean * (1.0 - kept), rtol=RTOL, atol=0.0)
        np.testing.assert_allclose(syn.x.value, kept, rtol=RTOL)

    @_float64
    def test_the_release_is_differentiable_in_U_even_at_one(self) -> None:
        # Rows without events keep (1 - U)^0 = 1, whose derivative jax would give as 0 * inf at U = 1.
        count = jnp.asarray([0.0, 1.0, 2.0])
        syn = ExpSynSTD(size=3)

        def g_total(U_value):
            syn.U = jnp.full((3,), U_value)  # parameters are checked at construction, outside the trace
            syn.init_state()
            syn.apply_events(jnp.asarray([0.0, 0.001, 0.002]) * u.uS, count=count)
            return jnp.sum(syn.g.value.to_decimal(u.uS))

        for U_value, expected in ((1.0, 0.001), (0.5, 0.001 + 0.002 / 2 * 2 * 0.5)):
            with self.subTest(U=U_value):
                grad = jax.grad(g_total)(U_value)
                self.assertTrue(np.isfinite(grad))
                np.testing.assert_allclose(grad, expected, rtol=RTOL)

    def test_parameters_are_checked_at_construction(self) -> None:
        cases = {"U above one": dict(U=1.5), "U zero": dict(U=0.0), "tau_rec zero": dict(tau_rec=0.0 * u.ms)}
        cases["tau_rec negative"] = dict(tau_rec=-1.0 * u.ms)
        cases["tau zero"] = dict(tau=0.0 * u.ms)
        for label, kwargs in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    ExpSynSTD(size=1, **kwargs)
        ExpSynSTD(size=1, U=1.0)  # the upper bound is allowed

    def test_the_count_is_required(self) -> None:
        syn = ExpSynSTD(size=1)
        syn.init_state()
        with self.assertRaisesRegex(TypeError, "count"):
            syn.apply_events(np.array([0.001]) * u.uS)


class ReferenceTest(unittest.TestCase):
    """Acceptance 1: successive, same-boundary, independent and shared releases."""

    @_float64
    def test_successive_events_release_half_then_a_quarter(self) -> None:
        g, x = _run_network(_network(_seq([1.0, 2.0]), _toy_cell(), synapse_index=[0]), 3.0)
        np.testing.assert_allclose(g[[11, 21], 0] - g[[10, 20], 0], [DG_FIRST, DG_SECOND], rtol=RTOL)
        np.testing.assert_allclose(x[[11, 21], 0], [1.0 - U, (1.0 - U) ** 2], rtol=RTOL)

    @_float64
    def test_two_events_on_one_boundary_release_as_two_in_a_row(self) -> None:
        # One row with two events at 1.0 ms, and two rows with one event each.
        for label, source, index in (("one row", _seq([1.0, 1.0]), [0]), ("two rows", _seq([1.0], [1.0]), [0, 0])):
            with self.subTest(label):
                g, x = _run_network(_network(source, _toy_cell(), synapse_index=index), 2.0)
                np.testing.assert_allclose(g[[10, 11], 0], [0.0, DG_TWO], rtol=RTOL, atol=0.0)
                np.testing.assert_allclose(x[11, 0], 0.25, rtol=RTOL)

    @_float64
    def test_independent_synapses_have_their_own_resource(self) -> None:
        # A at 1 ms and B at 2 ms, to two placed synapses or to one.
        cases = {"independent": (2, [0, 1], [DG_FIRST, DG_FIRST]), "shared": (1, [0, 0], [DG_FIRST, DG_SECOND])}
        for label, (n_sites, index, expected) in cases.items():
            with self.subTest(label):
                g, _ = _run_network(_network(_seq([1.0], [2.0]), _toy_cell(n_sites=n_sites), synapse_index=index), 3.0)
                jumps = [g[11].sum() - g[10].sum(), g[21].sum() - g[20].sum()]
                np.testing.assert_allclose(jumps, expected, rtol=RTOL)


def _passive_soma(lumped=False, *, e=-65.0 * u.mV):
    """A passive soma; with ``e = -65 mV`` the synapse carries no current and ``V`` stays at rest."""
    if lumped:
        cell = Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
    else:
        cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
    cell.paint(braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    site = () if lumped else (at("soma", 0.5),)
    cell.place(*site, _std(e=e))
    _record(cell)
    cell.record("V", "V")
    return cell


class RecoveryTest(unittest.TestCase):
    """Acceptance 2: the resource recovers with tau_rec between events."""

    @_float64
    def test_recovery_between_events_at_0_and_50_ms(self) -> None:
        x_before = _reference(1.0 - (1.0 - U) * np.exp(-50.0 / TAU_REC), 0.530293468593)
        x_after = _reference(1.0 - (1.0 - (1.0 - U) * x_before) * np.exp(-0.1 / TAU_REC), 0.265238585214)
        for lumped in (False, True):
            with self.subTest(lumped=lumped):
                cell = _passive_soma(lumped)
                braincell.connect("c", source=_seq([0.0, 50.0]), synapse=cell.synapses["s"], weight=W * u.uS)
                result = cell.run(dt=DT, duration=50.2 * u.ms)
                _, x = _samples(result.samples)
                np.testing.assert_allclose(x[[500, 501], 0], [x_before, x_after], rtol=1e-9)
                v = np.asarray(result.samples["V"].values.to_decimal(u.mV))
                np.testing.assert_allclose(v, -65.0, rtol=0.0, atol=1e-9)


def _stdp(**kwargs):
    params = dict(A_plus=0.01 * u.uS, A_minus=0.0105 * u.uS, tau_plus=20.0 * u.ms, tau_minus=20.0 * u.ms)
    params.update(kwargs)
    return braincell.mech.ConnectionRule("PairSTDP", **params)


class DeliveryPathTest(unittest.TestCase):
    """Acceptance 3: every delivery path passes the count."""

    def _check(self, g, x) -> None:
        np.testing.assert_allclose(g[-1, 0], DG_TWO, rtol=RTOL)
        np.testing.assert_allclose(x[-1, 0], 0.25, rtol=RTOL)

    @_float64
    def test_live_rows_through_the_queue_and_the_zero_delay_path(self) -> None:
        # The source crosses in step 5 (0.5 to 0.6 ms); delays of 0 and 2 steps
        # arrive at 0.6 ms (end of that step) and at 0.7 ms.
        for delay_ms in ([0.0, 0.2], [0.0, 0.0], [0.2, 0.2]):
            for backend in ("scatter", "brainevent"):
                with self.subTest(delay_ms=delay_ms, backend=backend):
                    net = _network(_live_pre([0.6]), _toy_cell(), synapse_index=[0, 0], delay=delay_ms * u.ms)
                    self._check(*_run_network(net, 1.0, event_backend=backend))

    @_float64
    def test_rule_rows_on_live_and_scheduled_sources(self) -> None:
        # A rule that keeps the weight: its rows deliver through the rule boundary (live, k >= 1;
        # scheduled) or the zero-delay block, and every one of them counts.
        rule = _stdp(A_plus=0.0 * u.uS, A_minus=0.0 * u.uS)
        cases = {
            "live": (lambda: _live_pre([0.6]), [0.0, 0.2]),
            "live, both delayed": (lambda: _live_pre([0.6]), [0.2, 0.3]),
            "scheduled": (lambda: _seq([0.5], [0.5]), [0.1, 0.1]),
        }
        for label, (source, delay_ms) in cases.items():
            with self.subTest(label):
                net = _network(source(), _toy_cell(), synapse_index=[0, 0], delay=delay_ms * u.ms, rule=rule)
                self._check(*_run_network(net, 1.0))

    @_float64
    def test_scheduled_rows(self) -> None:
        sources = {
            "EventSequence": lambda: _seq([0.5], [0.5]),
            "NetStim": lambda: braincell.NetStim(size=2, start=0.5 * u.ms, number=1, interval=10.0 * u.ms),
        }
        for label, source in sources.items():
            for backend in ("scatter", "brainevent"):
                with self.subTest(label, backend=backend):
                    net = _network(source(), _toy_cell(), synapse_index=[0, 0], delay=0.1 * u.ms)
                    self._check(*_run_network(net, 1.0, event_backend=backend))

    @_float64
    def test_standalone_cell_run(self) -> None:
        cell = _toy_cell()
        braincell.connect("c", source=_seq([0.5], [0.5]), synapse=cell.synapses["s"][[0, 0]], weight=W * u.uS)
        self._check(*_samples(cell.run(dt=DT, duration=1.0 * u.ms).samples))
        # A live self-connection: the Cell crosses in step 5, rows with delays of 0 and 2 steps.
        cell = _toy_cell([0.6])
        braincell.connect(
            "self",
            source=cell.event_outputs["spike"],
            synapse=cell.synapses["s"][[0, 0]],
            weight=W * u.uS,
            delay=[0.0, 0.2] * u.ms,
        )
        self._check(*_samples(cell.run(dt=DT, duration=1.0 * u.ms).samples))

    @_float64
    def test_standalone_update_loop(self) -> None:
        cell = _toy_cell()
        braincell.connect("c", source=_seq([0.5], [0.5]), synapse=cell.synapses["s"][[0, 0]], weight=W * u.uS)
        cell.init_state()

        def step(t):
            with brainstate.environ.context(t=t):
                cell.update()

        with brainstate.environ.context(dt=DT):
            brainstate.transform.for_loop(step, u.math.arange(10) * DT)
        runtime = cell.runtime
        node = next(node for layout, node in runtime.iter_synapse_layouts())
        np.testing.assert_allclose(node.g.value.to_decimal(u.uS), [DG_TWO], rtol=RTOL)
        np.testing.assert_allclose(node.x.value, [0.25], rtol=RTOL)


class PlasticityRuleTest(unittest.TestCase):
    """Acceptance 4: with a pair STDP rule on the rows, an event adds ``w`` times the released fraction."""

    @_float64
    def test_rule_weights_on_independent_and_shared_resources(self) -> None:
        w0 = np.array([0.05, 0.03])
        # The rows potentiate at the post event (10 ms), so 20 ms delivers w0 + A_plus e^(-1/2).
        w20 = w0 + 0.01 * np.exp(-10.0 / 20.0)
        expected = {
            "independent": (2, [0, 1], _reference(w0.sum() * U, 0.04), _reference(w20.sum() * U * U, 0.023032653299)),
            "shared": (
                1,
                [0, 0],
                _reference(w0.sum() / 2 * (1.0 - (1.0 - U) ** 2), 0.03),
                _reference(w20.sum() / 2 * 0.25 * (1.0 - (1.0 - U) ** 2), 0.008637244987),
            ),
        }
        for label, (n_sites, index, jump_0, jump_20) in expected.items():
            with self.subTest(label):
                cell = _toy_cell([10.0], n_sites=n_sites)
                net = _network(
                    _seq([0.0, 20.0], [0.0, 20.0]), cell, synapse_index=index, weight=w0 * u.uS, rule=_stdp()
                )
                g, _ = _run_network(net, 21.0)
                total = g.sum(axis=1)
                np.testing.assert_allclose([total[1] - total[0], total[201] - total[200]], [jump_0, jump_20], rtol=RTOL)


def _lifecycle_model(lumped=False):
    net = braincell.Network("std_life", seed=4)
    pre = net.add_population("pre", _live_pre([1.1, 3.4, 3.6, 9.0, 17.3]))
    stim = net.add_population(
        "stim", braincell.NetStim(size=2, start=0.5 * u.ms, number=20, interval=1.3 * u.ms, noise=0.5)
    )
    cell = _passive_soma(lumped, e=0.0 * u.mV)
    net.add_population("post", cell)
    net.connect(
        "live", source=pre.event_outputs["spike"], synapse=cell.synapses["s"], weight=0.002 * u.uS, delay=0.2 * u.ms
    )
    net.connect("stim", source=stim, synapse=cell.synapses["s"][[0, 0]], weight=0.001 * u.uS, delay=0.3 * u.ms)
    return net


def _trajectory(result):
    samples = result.samples["post"]
    g, x = _samples(samples)
    return g, x, np.asarray(samples["V"].values.to_decimal(u.mV))


class LifecycleTest(unittest.TestCase):
    """Acceptance 5: segmented runs, reset_state, and Network.update."""

    @_float64
    def test_segmented_and_reset_runs_match_one_run(self) -> None:
        # A cable target and a lumped one (Cell.point).
        for lumped in (False, True):
            with self.subTest(lumped=lumped):
                net = _lifecycle_model(lumped)
                full = _trajectory(net.run(dt=DT, duration=25.0 * u.ms))
                self.assertLess(float(full[1].min()), 0.5)  # several events depleted the resource
                net2 = _lifecycle_model(lumped)
                parts = [net2.run(dt=DT, duration=d * u.ms) for d in (3.5, 0.1, 9.4, 12.0)]
                split = _trajectory(braincell.NetworkResult.concat(parts))
                for got, want in zip(split, full):
                    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
                net.reset_state()
                again = _trajectory(net.run(dt=DT, duration=25.0 * u.ms))
                for got, want in zip(again, full):
                    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)

    @_float64
    def test_prepare_run_and_update_match_run(self) -> None:
        net = _lifecycle_model()
        net.run(dt=DT, duration=20.0 * u.ms)
        net2 = _lifecycle_model()
        net2.prepare_run(dt=DT)
        brainstate.transform.for_loop(lambda _: net2.update(), np.arange(200))
        for name in ("g", "x"):
            nodes = [
                next(node for _, node in model.populations["post"].cell.runtime.iter_synapse_layouts())
                for model in (net, net2)
            ]
            got, want = (np.asarray(u.get_mantissa(getattr(node, name).value)) for node in nodes)
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)


class _NoCountSTD(braincell.Synapse):
    """A ``"sum_count"`` synapse whose handler does not take ``count``."""

    root_type = braincell.HHTypedNeuron
    states = {"g": braincell.mech.StateSpec(0.0 * u.uS)}
    event_input = braincell.mech.ScalarEventInput(u.uS, aggregation="sum_count")

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self._init_parameters()

    def apply_events(self, payload, V_post=None):
        self.g.value = self.g.value + payload


class CounterExampleTest(unittest.TestCase):
    """Acceptance 7: each rejected use, with the step that rejects it."""

    def test_a_reduction_model_rejects_a_sum_count_synapse_at_init_state(self) -> None:
        cell = _toy_cell()
        cell.add_reduction("acc", braincell.PayloadAccumulatorReduction(threshold=1.0 * u.uS))
        cell.use_model("acc")
        braincell.connect("c", source=_seq([1.0]), synapse=cell.synapses["s"], weight=W * u.uS)
        with self.assertRaisesRegex(TypeError, "'ExpSynSTD'.*sum_count.*reduction"):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_a_handler_without_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "keyword argument count"):
            braincell.mech.register_synapse("_CELL12bNoCount")(_NoCountSTD)
        self.assertFalse(braincell.mech.get_registry().contains("synapse", "_CELL12bNoCount"))
        # The same class, unregistered, in a declaration.
        with self.assertRaisesRegex(TypeError, "keyword argument count"):
            braincell.mech.Synapse(_NoCountSTD, name="bad")


class NoCountUnchangedTest(unittest.TestCase):
    """Acceptance 6 (in part): only "sum_count" layouts get count buffers and count rings."""

    def test_count_buffers_and_rings_exist_for_sum_count_layouts_only(self) -> None:
        cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="plain", tau=2.0 * u.ms))
        cell.place(at("soma", 0.5), _std())
        net = braincell.Network("mixed")
        pre = net.add_population("pre", _live_pre([0.6]))
        net.add_population("post", cell)
        for name in ("plain", "s"):
            net.connect(
                name, source=pre.event_outputs["spike"], synapse=cell.synapses[name], weight=W * u.uS, delay=0.2 * u.ms
            )
        net.run(dt=DT, duration=1.0 * u.ms)
        runtime = cell.runtime
        kinds = {layout.id: layout.kind for layout, _ in runtime.iter_synapse_layouts()}
        self.assertEqual({kinds[key] for key in runtime.event_counts}, {"synapse:ExpSynSTD"})
        (state,) = net._delivery_state_cache.values()
        counted = {state.queue_keys[index][1] for index in range(len(state.queue_keys)) if state.count_ring(index)}
        self.assertEqual({kinds[key] for key in counted}, {"synapse:ExpSynSTD"})

        plain = braincell.Network("plain")
        plain_cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
        plain_cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="plain", tau=2.0 * u.ms))
        plain_pre = plain.add_population("pre", _live_pre([0.6]))
        plain.add_population("post", plain_cell)
        plain.connect(
            "plain",
            source=plain_pre.event_outputs["spike"],
            synapse=plain_cell.synapses["plain"],
            weight=W * u.uS,
            delay=0.2 * u.ms,
        )
        plain.run(dt=DT, duration=1.0 * u.ms)
        self.assertEqual(plain_cell.runtime.event_counts, {})
        (state,) = plain._delivery_state_cache.values()
        self.assertEqual(state.count_rings, ())


if __name__ == "__main__":
    unittest.main()
