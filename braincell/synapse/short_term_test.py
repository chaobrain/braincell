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

"""``ExpSynSTD``, ``ExpSynSTP`` and the ``"sum_count"`` event input on every delivery path (CELL-12b, CELL-12e).

Most targets are a toy Cell, as in the plasticity tests: a callable solver
sets ``V`` to +40 mV in the steps that end at the chosen spike times and to
-65 mV otherwise, and integrates nothing, so ``g``, ``u`` and ``x`` change
only at event boundaries. The synapse is
``ExpSynSTD(tau=2 ms, U=0.5, tau_rec=800 ms)`` or
``ExpSynSTP(tau=2 ms, U=0.25, tau_fac=100 ms, tau_rec=800 ms)``, the weight
0.001 uS and ``dt`` 0.1 ms unless a test says otherwise. A sample at ``t``
holds the state before the events of the boundary ``t``.

Reference values are closed forms evaluated on the host (exact fractions
where the inputs are dyadic) or efficacies recorded from NEST, never the
models' own expressions.
"""

import unittest
from fractions import Fraction

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


def _toy_cell(spike_ms=(), *, n_sites=1, synapse=None, states=("g", "x")):
    synapse = _std if synapse is None else synapse
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
        cell.place(at("soma", 0.5), synapse())
    _record(cell, states)
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


def _record(cell, states=("g", "x")):
    for state in states:
        cell.record(state, braincell.observe.synapse(name="s").state(state))


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


def _run_network(net, duration_ms, *, reader=None, **kwargs):
    reader = _samples if reader is None else reader
    return reader(net.run(dt=DT, duration=duration_ms * u.ms, **kwargs).samples["post"])


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


def _passive_soma(lumped=False, *, e=-65.0 * u.mV, synapse=None, states=("g", "x"), solver=None, **params):
    """A passive soma; with ``e = -65 mV`` the synapse carries no current and ``V`` stays at rest."""
    synapse = _std if synapse is None else synapse
    options = {} if solver is None else {"solver": solver}
    if lumped:
        cell = Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, **options)
    else:
        cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV, **options)
    cell.paint(braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    site = () if lumped else (at("soma", 0.5),)
    cell.place(*site, synapse(e=e, **params))
    _record(cell, states)
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
        # arrive at 0.6 ms (end of that step) and at 0.8 ms (CELL-75).
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
        # The crossing step 5 with 3 delay steps arrives at 0.9 ms, after the
        # last sample of a 1 ms run (CELL-75): that case runs 1.1 ms.
        cases = {
            "live": (lambda: _live_pre([0.6]), [0.0, 0.2], 1.0),
            "live, both delayed": (lambda: _live_pre([0.6]), [0.2, 0.3], 1.1),
            "scheduled": (lambda: _seq([0.5], [0.5]), [0.1, 0.1], 1.0),
        }
        for label, (source, delay_ms, duration_ms) in cases.items():
            with self.subTest(label):
                net = _network(source(), _toy_cell(), synapse_index=[0, 0], delay=delay_ms * u.ms, rule=rule)
                self._check(*_run_network(net, duration_ms))

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


def _lifecycle_model(lumped=False, *, synapse=None, states=("g", "x")):
    net = braincell.Network("std_life", seed=4)
    pre = net.add_population("pre", _live_pre([1.1, 3.4, 3.6, 9.0, 17.3]))
    stim = net.add_population(
        "stim", braincell.NetStim(size=2, start=0.5 * u.ms, number=20, interval=1.3 * u.ms, noise=0.5)
    )
    cell = _passive_soma(lumped, e=0.0 * u.mV, synapse=synapse, states=states)
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
    """Acceptance 6 (in part): only "sum_count" layouts get count buffers and count their delayed arrivals."""

    def test_count_buffers_and_counted_queues_exist_for_sum_count_layouts_only(self) -> None:
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
        counted = {
            state.queue_keys[index][1] for index in range(len(state.queue_keys)) if state.count_dtype(index) is not None
        }
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
        self.assertEqual(state.count_dtypes, ())


# ---------------------------------------------------------------- ExpSynSTP (CELL-12e)

U_STP = 0.25
TAU_FAC = 100.0
#: The states an ExpSynSTP test records.
STP_STATES = ("g", "u", "x")


def _stp(**kwargs):
    params = dict(tau=2.0 * u.ms, U=U_STP, tau_fac=TAU_FAC * u.ms, tau_rec=TAU_REC * u.ms)
    params.update(kwargs)
    return braincell.mech.Synapse("ExpSynSTP", name="s", **params)


def _stp_samples(samples):
    """``(g in uS, u, x)`` of one Cell's samples, ``(time, synapse)``."""
    g = np.asarray(samples["g"].values.to_decimal(u.uS))
    return (g,) + tuple(np.asarray(u.get_mantissa(samples[name].values)) for name in ("u", "x"))


def _stp_toy(spike_ms=(), *, n_sites=1, synapse=None):
    return _toy_cell(spike_ms, n_sites=n_sites, synapse=_stp if synapse is None else synapse, states=STP_STATES)


def _stp_soma(lumped=False, **kwargs):
    """The passive soma with an ExpSynSTP; a lumped target uses ``exp_euler`` (exact for the linear relaxations)."""
    kwargs.setdefault("synapse", _stp)
    kwargs.setdefault("states", STP_STATES)
    if lumped:
        kwargs.setdefault("solver", "exp_euler")
    return _passive_soma(lumped, **kwargs)


def _events(steps, g, tau_ms=2.0):
    """The jump of ``g`` on each boundary in ``steps``: ``g`` decays exactly by ``exp(-dt/tau)`` over the step."""
    steps = np.asarray(steps)
    return g[steps + 1] * np.exp(0.1 / tau_ms) - g[steps]


def _tsodyks2_efficacies(times_ms, U, tau_fac, tau_rec):
    """``x u`` of each event by NEST's ``tsodyks2_synapse`` rule (Maass & Markram 2002, Eqs. 4-5), on the host."""
    x, u_, out = 1.0, U, []
    for k, t in enumerate(times_ms):
        if k:
            h = t - times_ms[k - 1]
            x = 1.0 + (x - x * u_ - 1.0) * np.exp(-h / tau_rec)
            u_ = U + u_ * (1.0 - U) * (np.exp(-h / tau_fac) if tau_fac else 0.0)
        out.append(x * u_)
    return np.asarray(out)


def _stp_by_hand(u0, x0, count, U, facilitates=True):
    """Release ``count`` events one at a time on the host; returns ``(released fraction, u, x)``."""
    released = 0.0
    for _ in range(int(count)):
        if facilitates:
            u0 = u0 + U * (1.0 - u0)
            r = u0 * x0
        else:
            r = U * x0
        x0 -= r
        released += r
    return released, (u0 if facilitates else 0.0), x0


W_ = Fraction(1, 1000)
#: ExpSynSTP, U = 1/4, from u = 0 and x = 1: one event releases 1/4; the next 7/16 of the 3/4 left.
STP_FIRST = float(W_ / 4)
STP_SECOND = float(W_ * Fraction(21, 64))
#: Two events on one boundary: x keeps (1 - U)^3 = 27/64, u = 1 - (1 - U)^2 = 7/16.
STP_TWO = float(W_ * Fraction(37, 64))


class STPModelTest(unittest.TestCase):
    """Acceptance 7: the release formula on the runtime class itself."""

    @_float64
    def test_one_call_releases_as_successive_events(self) -> None:
        tau_fac = np.array([100.0, 100.0, 100.0, 0.0, 0.0])
        syn = braincell.synapse.ExpSynSTP(size=5, U=0.3, tau_fac=tau_fac * u.ms)
        syn.init_state()
        u0, x0 = np.array([0.2, 0.2, 0.0, 0.2, 0.0]), np.array([0.9, 0.9, 1.0, 0.9, 1.0])
        syn.u.value, syn.x.value = jnp.asarray(u0), jnp.asarray(x0)
        count = np.array([0.0, 3.0, 1.0, 3.0, 2.0])
        payload = np.array([0.0, 0.003, 0.001, 0.003, 0.004])
        syn.apply_events(payload * u.uS, count=count)
        expected = [_stp_by_hand(u0[i], x0[i], count[i], 0.3, tau_fac[i] > 0) for i in range(5)]
        mean = payload / np.maximum(count, 1.0)
        g = np.asarray(syn.g.value.to_decimal(u.uS))
        np.testing.assert_allclose(g, mean * [e[0] for e in expected], rtol=RTOL, atol=0.0)
        # Rows without events keep u (row 0); rows without facilitation hold u at 0 (rows 3 and 4).
        np.testing.assert_allclose(syn.u.value, [0.2] + [e[1] for e in expected[1:]], rtol=RTOL, atol=0.0)
        np.testing.assert_allclose(syn.x.value, [e[2] for e in expected], rtol=RTOL)
        syn.compute_derivative()
        np.testing.assert_allclose(
            u.get_mantissa(syn.u.derivative * u.ms),
            -np.asarray(syn.u.value) / np.where(tau_fac > 0, tau_fac, 1.0),
            rtol=RTOL,
        )
        np.testing.assert_array_equal(u.get_mantissa(syn.u.derivative * u.ms)[3:], 0.0)

    @_float64
    def test_the_release_is_differentiable_in_U_even_at_one(self) -> None:
        count = jnp.asarray([0.0, 1.0, 2.0, 2.0])
        syn = braincell.synapse.ExpSynSTP(size=4, tau_fac=np.array([100.0, 100.0, 100.0, 0.0]) * u.ms)

        def g_total(U_value):
            syn.U = jnp.full((4,), U_value)
            syn.init_state()
            syn.apply_events(jnp.asarray([0.0, 0.001, 0.002, 0.002]) * u.uS, count=count)
            return jnp.sum(syn.g.value.to_decimal(u.uS))

        # d/dU of 0.001 U + 0.001 (1 - (1 - U)^3) + 0.001 (1 - (1 - U)^2).
        for U_value in (1.0, 0.5):
            with self.subTest(U=U_value):
                grad = jax.grad(g_total)(U_value)
                self.assertTrue(np.isfinite(grad))
                expected = 0.001 + 0.003 * (1.0 - U_value) ** 2 + 0.002 * (1.0 - U_value)
                np.testing.assert_allclose(grad, expected, rtol=RTOL)

    @_float64
    def test_the_recovery_is_differentiable_in_tau_fac_even_at_zero(self) -> None:
        syn = braincell.synapse.ExpSynSTP(size=2, tau_fac=np.array([100.0, 0.0]) * u.ms)

        def du_total(tau_fac):
            syn.tau_fac = tau_fac * u.ms
            syn.init_state()
            syn.u.value = jnp.asarray([0.5, 0.0])
            syn.compute_derivative()
            return jnp.sum(u.get_mantissa(syn.u.derivative * u.ms))

        grad = jax.grad(du_total)(jnp.asarray([100.0, 0.0]))
        np.testing.assert_allclose(grad, [0.5 / 100.0**2, 0.0], rtol=RTOL, atol=0.0)

    def test_the_count_is_required(self) -> None:
        syn = braincell.synapse.ExpSynSTP(size=1)
        syn.init_state()
        with self.assertRaisesRegex(TypeError, "count"):
            syn.apply_events(np.array([0.001]) * u.uS)


class STPReferenceTest(unittest.TestCase):
    """Acceptance 1: successive, same-boundary, independent and shared releases."""

    @_float64
    def test_successive_events_facilitate(self) -> None:
        g, u_, x = _run_network(_network(_seq([1.0, 2.0]), _stp_toy(), synapse_index=[0]), 3.0, reader=_stp_samples)
        np.testing.assert_allclose(g[[11, 21], 0] - g[[10, 20], 0], [STP_FIRST, STP_SECOND], rtol=RTOL)
        np.testing.assert_allclose(u_[[11, 21], 0], [0.25, 0.4375], rtol=RTOL)
        np.testing.assert_allclose(x[[11, 21], 0], [0.75, 0.421875], rtol=RTOL)

    @_float64
    def test_two_events_on_one_boundary_release_as_two_in_a_row(self) -> None:
        for label, source, index in (("one row", _seq([1.0, 1.0]), [0]), ("two rows", _seq([1.0], [1.0]), [0, 0])):
            with self.subTest(label):
                g, u_, x = _run_network(_network(source, _stp_toy(), synapse_index=index), 2.0, reader=_stp_samples)
                np.testing.assert_allclose(g[[10, 11], 0], [0.0, STP_TWO], rtol=RTOL, atol=0.0)
                np.testing.assert_allclose([u_[11, 0], x[11, 0]], [0.4375, 0.421875], rtol=RTOL)

    @_float64
    def test_independent_synapses_have_their_own_state(self) -> None:
        cases = {"independent": (2, [0, 1], [STP_FIRST, STP_FIRST]), "shared": (1, [0, 0], [STP_FIRST, STP_SECOND])}
        for label, (n_sites, index, expected) in cases.items():
            with self.subTest(label):
                net = _network(_seq([1.0], [2.0]), _stp_toy(n_sites=n_sites), synapse_index=index)
                g, _, _ = _run_network(net, 3.0, reader=_stp_samples)
                np.testing.assert_allclose([g[11].sum() - g[10].sum(), g[21].sum() - g[20].sum()], expected, rtol=RTOL)

    @_float64
    def test_without_facilitation_the_numbers_are_expsynstd_ones(self) -> None:
        # tau_fac = 0, U = 0.5: acceptance 1 of CELL-12b.
        def no_facilitation():
            return _stp(U=0.5, tau_fac=0.0 * u.ms)

        net = _network(_seq([1.0, 2.0]), _stp_toy(synapse=no_facilitation), synapse_index=[0])
        g, u_, _ = _run_network(net, 3.0, reader=_stp_samples)
        np.testing.assert_allclose(g[[11, 21], 0] - g[[10, 20], 0], [DG_FIRST, DG_SECOND], rtol=RTOL)
        np.testing.assert_array_equal(u_, 0.0)
        net = _network(_seq([1.0], [1.0]), _stp_toy(synapse=no_facilitation), synapse_index=[0, 0])
        g, u_, x = _run_network(net, 2.0, reader=_stp_samples)
        np.testing.assert_allclose([g[11, 0], x[11, 0]], [DG_TWO, 0.25], rtol=RTOL)
        np.testing.assert_array_equal(u_, 0.0)


#: NEST 3.10.0 ``tsodyks2_synapse`` efficacies (``weight_recorder``, weight 1) for the two parameter sets of
#: NEST's ``evaluate_tsodyks2_synapse`` example and events at 5 + 20 k ms (k < 8) and 645 ms.
NEST_TRAIN_MS = [5.0 + 20.0 * k for k in range(8)] + [645.0]
NEST_EFFICACY = {
    "facilitation": (
        dict(U=0.1, tau_fac=1000.0, tau_rec=100.0),
        [
            0.1,
            0.17280790388516587,
            0.21056807411702216,
            0.2198566409066093,
            0.2130766092111271,
            0.20109273084539298,
            0.19018096109978372,
            0.18250228970242455,
            0.39112470222109164,
        ],
    ),
    "depression": (
        dict(U=0.67, tau_fac=0.0, tau_rec=450.0),
        [
            0.67,
            0.24061424901665016,
            0.10507671139426952,
            0.062293679906458045,
            0.04878902418232437,
            0.04452621904942177,
            0.043180645495212076,
            0.04275590916940311,
            0.4540854301790566,
        ],
    ),
}


class STPRecoveryTest(unittest.TestCase):
    """Acceptance 2: recovery on a real soma, NEST's efficacies and every solver's discrete decay."""

    @_float64
    def test_recovery_between_events_at_0_and_50_ms(self) -> None:
        u_50 = _reference(0.25 * np.exp(-0.5), 0.151632664928158356, 15)
        x_50 = _reference(1.0 - 0.25 * np.exp(-1.0 / 16.0), 0.765146734296631053, 15)
        u_plus = 0.25 + 0.75 * u_50
        released = _reference(u_plus * x_50, 0.278302612361014514, 15)
        u_501 = _reference(u_plus * np.exp(-0.1 / TAU_FAC), 0.363360955999066399, 15)
        x_501 = _reference(1.0 - (1.0 - (x_50 - released)) * np.exp(-0.1 / TAU_REC), 0.486908262411511328, 15)
        for lumped in (False, True):
            with self.subTest(lumped=lumped):
                cell = _stp_soma(lumped)
                braincell.connect("c", source=_seq([0.0, 50.0]), synapse=cell.synapses["s"], weight=W * u.uS)
                result = cell.run(dt=DT, duration=50.2 * u.ms)
                g, u_, x = _stp_samples(result.samples)
                np.testing.assert_allclose([u_[500, 0], x[500, 0]], [u_50, x_50], rtol=RTOL)
                np.testing.assert_allclose(_events([500], g[:, 0]) / W, [released], rtol=RTOL)
                np.testing.assert_allclose([u_[501, 0], x[501, 0]], [u_501, x_501], rtol=RTOL)
                v = np.asarray(result.samples["V"].values.to_decimal(u.mV))
                np.testing.assert_allclose(v, -65.0, rtol=0.0, atol=1e-9)

    def test_the_nest_efficacies_follow_the_tsodyks2_rule(self) -> None:
        for label, (params, efficacy) in NEST_EFFICACY.items():
            with self.subTest(label):
                np.testing.assert_allclose(
                    _tsodyks2_efficacies(NEST_TRAIN_MS, params["U"], params["tau_fac"], params["tau_rec"]),
                    efficacy,
                    rtol=1e-15,
                )

    @_float64
    def test_each_event_adds_the_nest_efficacy(self) -> None:
        steps = np.round(np.asarray(NEST_TRAIN_MS) / 0.1).astype(int)
        for label, (params, efficacy) in NEST_EFFICACY.items():
            models = {"ExpSynSTP": dict(U=params["U"], tau_fac=params["tau_fac"] * u.ms)}
            if params["tau_fac"] == 0.0:
                models["ExpSynSTD"] = dict(U=params["U"])
            for model, extra in models.items():
                for lumped in (False, True):
                    with self.subTest(label, model=model, lumped=lumped):
                        cell = _stp_soma(
                            lumped,
                            synapse=_stp if model == "ExpSynSTP" else _std,
                            states=STP_STATES if model == "ExpSynSTP" else ("g", "x"),
                            tau_rec=params["tau_rec"] * u.ms,
                            **extra,
                        )
                        braincell.connect("c", source=_seq(NEST_TRAIN_MS), synapse=cell.synapses["s"], weight=W * u.uS)
                        g = np.asarray(cell.run(dt=DT, duration=646.0 * u.ms).samples["g"].values.to_decimal(u.uS))
                        np.testing.assert_allclose(_events(steps, g[:, 0]) / W, efficacy, rtol=RTOL)

    @_float64
    def test_each_solver_matches_its_discrete_decay(self) -> None:
        # u(50-) = U R(z_f)^500 and x(50-) = 1 - (1 - U) R(z_r)^500 with the method's one-step factor R.
        def taylor4(z):
            return 1 - z + z**2 / 2 - z**3 / 6 + z**4 / 24

        rational = {
            "rk4": taylor4,
            "crank_nicolson": lambda z: (1 - z / 2) / (1 + z / 2),
            "ralston2": lambda z: 1 - z + z**2 / 2,
            "euler": lambda z: 1 - z,
            "backward_euler": lambda z: 1 / (1 + z),
        }
        z_f, z_r = Fraction(1, 1000), Fraction(1, 8000)
        exact = (0.25 * np.exp(-0.5), 1.0 - 0.25 * np.exp(-1.0 / 16.0))
        discrete = {name: exact for name in ("staggered", "exp_euler", "ind_exp_euler")}
        for name, factor in rational.items():
            discrete[name] = (
                float(Fraction(1, 4) * factor(z_f) ** 500),
                float(1 - Fraction(1, 4) * factor(z_r) ** 500),
            )
        for lumped in (False, True):
            for solver, expected in discrete.items():
                if lumped and solver == "staggered":
                    continue  # staggered steps the cable tree only
                with self.subTest(solver=solver, lumped=lumped):
                    cell = _stp_soma(lumped, solver=solver)
                    braincell.connect("c", source=_seq([0.0, 50.0]), synapse=cell.synapses["s"], weight=W * u.uS)
                    _, u_, x = _stp_samples(cell.run(dt=DT, duration=50.1 * u.ms).samples)
                    np.testing.assert_allclose([u_[500, 0], x[500, 0]], expected, rtol=RTOL)


class STPDeliveryPathTest(unittest.TestCase):
    """Acceptance 3: every delivery path passes the count to ExpSynSTP."""

    def _check(self, g, u_, x) -> None:
        np.testing.assert_allclose([g[-1, 0], u_[-1, 0], x[-1, 0]], [STP_TWO, 0.4375, 0.421875], rtol=RTOL)

    @_float64
    def test_live_rows(self) -> None:
        for delay_ms in ([0.0, 0.2], [0.0, 0.0], [0.2, 0.2]):
            for backend in ("scatter", "brainevent"):
                with self.subTest(delay_ms=delay_ms, backend=backend):
                    net = _network(_live_pre([0.6]), _stp_toy(), synapse_index=[0, 0], delay=delay_ms * u.ms)
                    self._check(*_run_network(net, 1.0, reader=_stp_samples, event_backend=backend))

    @_float64
    def test_rule_rows(self) -> None:
        rule = _stdp(A_plus=0.0 * u.uS, A_minus=0.0 * u.uS)
        # As in DeliveryPathTest: 3 delay steps from step 5 arrive at 0.9 ms (CELL-75).
        cases = {
            "live": (lambda: _live_pre([0.6]), [0.0, 0.2], 1.0),
            "live, both delayed": (lambda: _live_pre([0.6]), [0.2, 0.3], 1.1),
            "scheduled": (lambda: _seq([0.5], [0.5]), [0.1, 0.1], 1.0),
        }
        for label, (source, delay_ms, duration_ms) in cases.items():
            with self.subTest(label):
                net = _network(source(), _stp_toy(), synapse_index=[0, 0], delay=delay_ms * u.ms, rule=rule)
                self._check(*_run_network(net, duration_ms, reader=_stp_samples))

    @_float64
    def test_scheduled_rows(self) -> None:
        sources = {
            "EventSequence": lambda: _seq([0.5], [0.5]),
            "NetStim": lambda: braincell.NetStim(size=2, start=0.5 * u.ms, number=1, interval=10.0 * u.ms),
        }
        for label, source in sources.items():
            for backend in ("scatter", "brainevent"):
                with self.subTest(label, backend=backend):
                    net = _network(source(), _stp_toy(), synapse_index=[0, 0], delay=0.1 * u.ms)
                    self._check(*_run_network(net, 1.0, reader=_stp_samples, event_backend=backend))

    @_float64
    def test_standalone_cell(self) -> None:
        cell = _stp_toy()
        braincell.connect("c", source=_seq([0.5], [0.5]), synapse=cell.synapses["s"][[0, 0]], weight=W * u.uS)
        self._check(*_stp_samples(cell.run(dt=DT, duration=1.0 * u.ms).samples))
        cell = _stp_toy([0.6])
        braincell.connect(
            "self",
            source=cell.event_outputs["spike"],
            synapse=cell.synapses["s"][[0, 0]],
            weight=W * u.uS,
            delay=[0.0, 0.2] * u.ms,
        )
        self._check(*_stp_samples(cell.run(dt=DT, duration=1.0 * u.ms).samples))
        # The compiled update() loop of a standalone Cell.
        cell = _stp_toy()
        braincell.connect("c", source=_seq([0.5], [0.5]), synapse=cell.synapses["s"][[0, 0]], weight=W * u.uS)
        cell.init_state()

        def step(t):
            with brainstate.environ.context(t=t):
                cell.update()

        with brainstate.environ.context(dt=DT):
            brainstate.transform.for_loop(step, u.math.arange(10) * DT)
        node = next(node for _, node in cell.runtime.iter_synapse_layouts())
        state = [node.g.value.to_decimal(u.uS)[0], node.u.value[0], node.x.value[0]]
        np.testing.assert_allclose(state, [STP_TWO, 0.4375, 0.421875], rtol=RTOL)


class STPPlasticityRuleTest(unittest.TestCase):
    """Acceptance 4: with a pair STDP rule on the rows, an event adds ``w`` times the released fraction."""

    @_float64
    def test_rule_weights_on_independent_and_shared_state(self) -> None:
        w0 = np.array([0.05, 0.03])
        w20 = w0 + 0.01 * np.exp(-10.0 / 20.0)  # potentiated at the post event (10 ms)
        expected = {
            # Independent: each synapse releases 1/4, then 7/16 of 3/4 = 21/64.
            "independent": (2, [0, 1], w0.sum() / 4, _reference(w20.sum() * 21 / 64, 0.0302303574543641568, 15)),
            # Shared: two events from u = 0, x = 1 release 37/64; then from u = 7/16, x = 27/64,
            # x (1 - (9/16)^2 (3/4)^3) = 383319/1048576.
            "shared": (
                1,
                [0, 0],
                w0.sum() / 2 * 37 / 64,
                _reference(w20.sum() / 2 * 383319 / 1048576, 0.0168397019000090306, 15),
            ),
        }
        for label, (n_sites, index, jump_0, jump_20) in expected.items():
            with self.subTest(label):
                cell = _stp_toy([10.0], n_sites=n_sites)
                net = _network(
                    _seq([0.0, 20.0], [0.0, 20.0]), cell, synapse_index=index, weight=w0 * u.uS, rule=_stdp()
                )
                g, _, _ = _run_network(net, 21.0, reader=_stp_samples)
                total = g.sum(axis=1)
                np.testing.assert_allclose([total[1] - total[0], total[201] - total[200]], [jump_0, jump_20], rtol=RTOL)


def _stp_trajectory(result):
    samples = result.samples["post"]
    return _stp_samples(samples) + (np.asarray(samples["V"].values.to_decimal(u.mV)),)


class STPLifecycleTest(unittest.TestCase):
    """Acceptance 5: segmented runs, reset_state, Network.update, a long float32 run and a batch."""

    def _model(self, lumped=False):
        return _lifecycle_model(lumped, synapse=_stp, states=STP_STATES)

    @_float64
    def test_segmented_and_reset_runs_match_one_run(self) -> None:
        for lumped in (False, True):
            with self.subTest(lumped=lumped):
                net = self._model(lumped)
                full = _stp_trajectory(net.run(dt=DT, duration=25.0 * u.ms))
                self.assertLess(float(full[2].min()), 0.5)  # several events depleted the resource
                self.assertGreater(float(full[1].max()), 0.5)  # and facilitated u
                net2 = self._model(lumped)
                parts = [net2.run(dt=DT, duration=d * u.ms) for d in (3.5, 0.1, 9.4, 12.0)]
                split = _stp_trajectory(braincell.NetworkResult.concat(parts))
                for got, want in zip(split, full):
                    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)
                net.reset_state()
                again = _stp_trajectory(net.run(dt=DT, duration=25.0 * u.ms))
                for got, want in zip(again, full):
                    np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)

    @_float64
    def test_prepare_run_and_update_match_run(self) -> None:
        net = self._model()
        net.run(dt=DT, duration=20.0 * u.ms)
        net2 = self._model()
        net2.prepare_run(dt=DT)
        brainstate.transform.for_loop(lambda _: net2.update(), np.arange(200))
        nodes = [
            next(node for _, node in model.populations["post"].cell.runtime.iter_synapse_layouts())
            for model in (net, net2)
        ]
        for name in STP_STATES:
            got, want = (np.asarray(u.get_mantissa(getattr(node, name).value)) for node in nodes)
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=0.0)

    def test_a_long_float32_run_matches_its_segments(self) -> None:
        def model():
            net = braincell.Network("stp32", seed=1)
            stim = net.add_population(
                "stim", braincell.NetStim(size=3, start=1.0 * u.ms, number=60, interval=50.0 * u.ms, noise=0.3)
            )
            cell = _stp_soma(e=0.0 * u.mV)
            net.add_population("post", cell)
            net.connect("stim", source=stim, synapse=cell.synapses["s"][[0, 0, 0]], weight=W * u.uS, delay=0.3 * u.ms)
            return net

        with jax.enable_x64(False), brainstate.environ.context(precision=32):
            full = _stp_trajectory(model().run(dt=DT, duration=3000.0 * u.ms))
            net = model()
            split = _stp_trajectory(
                braincell.NetworkResult.concat([net.run(dt=DT, duration=1000.0 * u.ms) for _ in range(3)])
            )
        self.assertEqual(full[0].shape[0], 30000)
        self.assertEqual(full[0].dtype, np.float32)
        self.assertGreater(float(full[1].max()), 0.5)
        for got, want in zip(split, full):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=0.0)

    @_float64
    def test_batch_elements_keep_their_own_state(self) -> None:
        cell = Cell(
            make_soma_tree(),
            cv_policy=CVPerBranch(),
            pop_size=(2,),
            V_init=-65.0 * u.mV,
            V_th=0.0 * u.mV,
            solver=_toy_solver(()),
        )
        cell.place(at("soma", 0.5), _stp())
        _record(cell, STP_STATES)
        # Member 0 gets one event at 1.0 ms and member 1 two.
        braincell.connect("c", source=_seq([1.0], [1.0], [1.0]), synapse=cell.synapses["s"][[0, 1, 1]], weight=W * u.uS)
        cell.init_state(batch_size=3)
        u0 = np.array([[0.0, 0.1], [0.3, 0.4], [0.6, 0.7]])
        x0 = np.array([[1.0, 0.9], [0.8, 0.7], [0.6, 0.5]])
        node = next(node for _, node in cell.runtime.iter_synapse_layouts())
        node.u.value, node.x.value = jnp.asarray(u0), jnp.asarray(x0)
        g, u_, x = _stp_samples(cell.run(dt=DT, duration=1.2 * u.ms).samples)
        self.assertEqual(g.shape, (12, 3, 2))
        np.testing.assert_allclose(
            g[11] - g[10], W * np.array([[0.25, 0.592453125], [0.38, 0.5936875], [0.42, 0.481015625]]), rtol=RTOL
        )
        np.testing.assert_allclose(u_[11], [[0.25, 0.49375], [0.475, 0.6625], [0.7, 0.83125]], rtol=RTOL)
        np.testing.assert_allclose(x[11], [[0.75, 0.307546875], [0.42, 0.1063125], [0.18, 0.018984375]], rtol=RTOL)


class STPRelationTest(unittest.TestCase):
    """Acceptance 6: where ExpSynSTP and ExpSynSTD give the same increments, and where they do not."""

    def _run(self, model, lumped, **params):
        synapse = _stp if model == "ExpSynSTP" else _std
        states = STP_STATES if model == "ExpSynSTP" else ("g", "x")
        cell = _stp_soma(lumped, e=0.0 * u.mV, synapse=synapse, states=states, tau_rec=200.0 * u.ms, **params)
        # Rows 0 and 1 both fire at 25 and 400 ms, so those boundaries carry two events.
        source = _seq([5.0, 25.0, 45.0, 65.0, 400.0], [25.0, 400.0, 401.0])
        braincell.connect(
            "c", source=source, synapse=cell.synapses["s"][[0, 0]], weight=np.array([0.001, 0.003]) * u.uS
        )
        samples = cell.run(dt=DT, duration=420.0 * u.ms).samples
        out = {name: np.asarray(u.get_mantissa(samples[name].values)) for name in states + ("V",)}
        return out

    @_float64
    def test_no_facilitation_or_full_release_reproduce_expsynstd(self) -> None:
        cases = {
            "tau_fac = 0": (dict(U=0.3, tau_fac=0.0 * u.ms), dict(U=0.3)),
            "U = 1": (dict(U=1.0, tau_fac=500.0 * u.ms), dict(U=1.0)),
        }
        for label, (stp_params, std_params) in cases.items():
            for lumped in (False, True):
                with self.subTest(label, lumped=lumped):
                    got = self._run("ExpSynSTP", lumped, **stp_params)
                    want = self._run("ExpSynSTD", lumped, **std_params)
                    for name in ("g", "x", "V"):
                        np.testing.assert_allclose(got[name], want[name], rtol=RTOL, atol=0.0)
                    if label == "tau_fac = 0":
                        np.testing.assert_array_equal(got["u"], 0.0)

    @_float64
    def test_with_facilitation_the_increments_differ(self) -> None:
        got = self._run("ExpSynSTP", False, U=0.3, tau_fac=500.0 * u.ms)
        want = self._run("ExpSynSTD", False, U=0.3)
        self.assertGreater(float(np.max(np.abs(got["g"] - want["g"]) / np.maximum(want["g"], 1e-300))), 0.1)


class STPCounterExampleTest(unittest.TestCase):
    """Acceptance 8: each rejected input, with the step that rejects it."""

    def test_parameters_are_checked_at_construction(self) -> None:
        value_errors = {
            "U above one": (dict(U=1.5), "0 < U <= 1"),
            "U zero": (dict(U=0.0), "0 < U <= 1"),
            "U negative": (dict(U=-0.1), "0 < U <= 1"),
            "tau_rec zero": (dict(tau_rec=0.0 * u.ms), "'tau_rec' must be > 0"),
            "tau_rec negative": (dict(tau_rec=-1.0 * u.ms), "'tau_rec' must be > 0"),
            "tau zero": (dict(tau=0.0 * u.ms), "'tau' must be > 0"),
            "tau_fac negative": (dict(tau_fac=-1.0 * u.ms), "'tau_fac' must be >= 0"),
            "tau_fac nan": (dict(tau_fac=float("nan") * u.ms), "finite"),
            "tau_fac in mV": (dict(tau_fac=1.0 * u.mV), "incompatible"),
        }
        for label, (kwargs, message) in value_errors.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, message):
                    braincell.synapse.ExpSynSTP(size=1, **kwargs)
        for label, kwargs in {"tau_fac bare": dict(tau_fac=100.0), "U with a unit": dict(U=0.5 * u.ms)}.items():
            with self.subTest(label):
                with self.assertRaises(TypeError):
                    braincell.synapse.ExpSynSTP(size=1, **kwargs)
        braincell.synapse.ExpSynSTP(size=1, U=1.0)  # the upper bound is allowed
        braincell.synapse.ExpSynSTP(size=1, tau_fac=0.0 * u.ms)  # no facilitation

    def _placed(self, **kwargs):
        cell = Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(1,), V_init=-65.0 * u.mV)
        cell.place(at("soma", 0.5), _stp(**kwargs))
        return cell

    def test_a_placed_synapse_is_checked_at_init_state(self) -> None:
        cases = {
            "tau_fac": (
                dict(tau_fac=-1.0 * u.ms),
                r"^Synapse 's' \(ExpSynSTP\) parameter 'tau_fac' must be >= 0\. Placed on AtLocation",
            ),
            "U": (dict(U=1.5), r"^Synapse 's' \(ExpSynSTP\): ExpSynSTP requires 0 < U <= 1\. Placed on AtLocation"),
        }
        for label, (kwargs, message) in cases.items():
            with self.subTest(label):
                cell = self._placed(**kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    cell.init_state()

    def test_set_checks_the_new_values_and_keeps_the_old_ones(self) -> None:
        cell = self._placed()
        cell.init_state()
        for kwargs in (dict(tau_fac=-1.0 * u.ms), dict(U=0.0)):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                with self.assertRaisesRegex(ValueError, r"Synapse 's' \(ExpSynSTP\)"):
                    cell.synapses["s"].set(**kwargs)
        node = next(node for _, node in cell.runtime.iter_synapse_layouts())
        np.testing.assert_allclose(node.tau_fac.to_decimal(u.ms), [TAU_FAC])
        np.testing.assert_allclose(node.U, [U_STP])

    @_float64
    def test_turning_facilitation_off_after_a_run(self) -> None:
        # Events at 1.0 and 2.0 ms; tau_fac is set to 0 at 1.5 ms. The 2.0 ms event then uses U:
        # 0.25 of the 0.75 left, and u is held at 0 from the next boundary on.
        cell = _stp_toy()
        braincell.connect("c", source=_seq([1.0, 2.0]), synapse=cell.synapses["s"], weight=W * u.uS)
        first = _stp_samples(cell.run(dt=DT, duration=1.5 * u.ms).samples)
        cell.synapses["s"].set(tau_fac=0.0 * u.ms)
        second = _stp_samples(cell.run(dt=DT, duration=1.5 * u.ms).samples)
        g, u_, x = (np.concatenate([a, b])[:, 0] for a, b in zip(first, second))
        np.testing.assert_allclose([g[11] - g[10], g[21] - g[20]], [STP_FIRST, W * 0.1875], rtol=RTOL)
        np.testing.assert_allclose(u_[14], 0.25, rtol=RTOL)
        np.testing.assert_array_equal(u_[16:], 0.0)
        np.testing.assert_allclose(x[21], 0.5625, rtol=RTOL)

    def test_a_reduction_model_rejects_it_at_init_state(self) -> None:
        cell = _stp_toy()
        cell.add_reduction("acc", braincell.PayloadAccumulatorReduction(threshold=1.0 * u.uS))
        cell.use_model("acc")
        braincell.connect("c", source=_seq([1.0]), synapse=cell.synapses["s"], weight=W * u.uS)
        with self.assertRaisesRegex(TypeError, "'ExpSynSTP'.*sum_count.*reduction"):
            cell.init_state()


if __name__ == "__main__":
    unittest.main()
