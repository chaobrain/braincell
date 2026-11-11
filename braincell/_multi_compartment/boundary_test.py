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

"""CELL-2a: clamps and synapses on branch ends and branch points under the derivative solvers.

Models (passive leak 0.1 mS/cm^2 at -65 mV, cm 1 uF/cm^2, Ra 100 ohm cm,
V_init -65 mV; ExpSyn tau 2 ms, e 0 mV, g written after init_state; clamps on
from 0 ms for 100 ms):

- M1: one 600 um x 0.5 um dendrite, 3 CVs. Inputs at x = 0: a 0.05 nA clamp
  ("clamp"), an ExpSyn with g(0) = 0.005 uS ("syn"), or both ("both").
- M2: M1's dendrite as 1 CV, the clamp at x = 0 and the ExpSyn at x = 1.
- M3: parent "p" (600 um x 0.5 um) with children "c1" (300 um x 0.4 um) and
  "c2" (450 um x 0.3 um) at its x = 1, 3 CVs each; at the branch point a
  0.05 nA clamp and the ExpSyn, at the c2 tip a -0.02 nA clamp.

The reference V(5 ms) values eliminate the boundary rows of the full point
system exactly at every evaluation and integrate with scipy DOP853 at
rtol = atol = 1e-13. "Same scheme" references step that reduced right-hand
side, written here in numpy, with the braintools kernel braincell wraps.
"""

import contextlib
import types
import unittest
import warnings
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import mech
from braincell._multi_compartment import boundary
from braincell.filter import AllRegion, AtLocation, RootLocation
from braincell.quad import integrator_info, list_integrators
from braincell.synapse import ExpSyn

CHECK = (1.0, 2.0, 3.0, 4.0, 5.0)
G0 = 0.005  # uS
CLAMP = 0.05  # nA

#: V(5 ms), mV, CV order.
REFERENCE = {
    ("M1", "clamp"): [-48.54954775746005, -55.90591218368505, -59.233244827934136],
    ("M1", "syn"): [-47.76349241200588, -52.62298265150734, -55.56330940397372],
    ("M1", "both"): [-35.73484647488816, -46.39768785806978, -51.81315195531077],
    ("M2", "both"): [-44.43180289809936],
    ("M3", "both"): [
        -57.18419124122931,
        -54.05355019839961,
        -48.05274809938648,
        -45.16509085571075,
        -46.99843488336557,
        -47.96140785715219,
        -49.86039245292369,
        -60.34277533863139,
        -70.28766327222891,
    ],
}
#: Boundary voltage at V(5 ms), mV, in point-id order of the algebraic points.
REFERENCE_BOUNDARY = {
    "M2": [-25.333209727071917, -38.41020444198748],
    "M3": [-57.18419124122931, -43.89031940643614, -47.96140785715219, -75.59282804195877],
}

_RUNS = {}


@contextlib.contextmanager
def _float64():
    with jax.enable_x64(True), brainstate.environ.context(precision=64):
        yield


def _branch(length, radius):
    return braincell.Branch.from_lengths(lengths=[length] * u.um, radii=[radius, radius] * u.um, type="dendrite")


def _morphology(model):
    if model in ("M1", "M2"):
        return braincell.Morphology.from_root(_branch(600.0, 0.5), name="dend")
    morpho = braincell.Morphology.from_root(_branch(600.0, 0.5), name="p")
    morpho.attach(parent="p", child_branch=_branch(300.0, 0.4), child_name="c1", parent_x=1.0)
    morpho.attach(parent="p", child_branch=_branch(450.0, 0.3), child_name="c2", parent_x=1.0)
    return morpho


def _inputs(model, case):
    """``(location, "clamp" | "syn", value)`` of one model and case."""
    if model == "M1":
        left = RootLocation(0.0)
        return {
            "none": (),
            "clamp": ((left, "clamp", CLAMP),),
            "syn": ((left, "syn", G0),),
            "both": ((left, "syn", G0), (left, "clamp", CLAMP)),
            "mid_clamp": ((RootLocation(0.5), "clamp", CLAMP),),
            "event": ((left, "syn", 0.0),),
        }[case]
    if model == "M2":
        return ((RootLocation(0.0), "clamp", CLAMP), (RootLocation(1.0), "syn", G0))
    junction = AtLocation(branch="p", x=1.0)
    return (
        (junction, "clamp", CLAMP),
        (junction, "syn", G0),
        (AtLocation(branch="c2", x=1.0), "clamp", -0.02),
    )


def _synapse_node(cell):
    nodes = [node for node in cell.runtime_objects(ExpSyn, allowed_hierarchy=(1, 1)).values()]
    assert len(nodes) == 1, nodes
    return nodes[0]


def _build(
    model,
    case="both",
    solver="rk4",
    *,
    pop_size=None,
    batch_size=None,
    synapse=ExpSyn,
    Ra=100.0,
    netstim=None,
    autapse=False,
):
    kwargs = {} if pop_size is None else {"pop_size": pop_size}
    cell = braincell.Cell(
        _morphology(model),
        cv_policy=braincell.CVPerBranch(cv_per_branch=1 if model == "M2" else 3),
        V_init=-65.0 * u.mV,
        solver=solver,
        **kwargs,
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=Ra * u.ohm * u.cm,
        ),
        mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
    )
    g0 = None
    declaration = None
    for location, kind, value in _inputs(model, case):
        if kind == "clamp":
            cell.place(
                location, braincell.CurrentClamp(delay=0.0 * u.ms, durations=100.0 * u.ms, amplitudes=value * u.nA)
            )
        else:
            declaration = mech.Synapse(synapse, name="syn", tau=2.0 * u.ms, e=0.0 * u.mV)
            cell.place(location, declaration)
            g0 = value
    if netstim is not None:
        start, weight = netstim
        braincell.connect(
            "drive",
            source=braincell.NetStim(start=start * u.ms, number=1),
            synapse=cell.synapses[declaration],
            weight=weight * u.uS,
        )
    if autapse:
        braincell.connect(
            "self", source=cell.event_outputs["spike"], synapse=cell.synapses[declaration], weight=G0 * u.uS
        )
    cell.init_state(batch_size=batch_size)
    if g0:
        node = _synapse_node(cell)
        node.g.value = u.Quantity(jnp.full(node.g.value.shape, g0), u.uS)
    return cell


def _run(cell, dt, *, segments=CHECK):
    """V after each segment end, ``(len(segments),) + V.shape`` in mV."""
    out, previous = [], 0.0
    for end in segments:
        cell.run(dt=dt * u.ms, duration=(end - previous) * u.ms)
        previous = end
        out.append(np.asarray(cell.V.value.to_decimal(u.mV)))
    return np.stack(out)


def _cached_run(model, case, solver, dt):
    """``(V at 1, ..., 5 ms of the single cell, cell)``, cached across tests."""
    key = (model, case, solver, dt)
    if key not in _RUNS:
        cell = _build(model, case, solver)
        _RUNS[key] = (_run(cell, dt)[:, 0, :], cell)
    return _RUNS[key]


def _final_voltage(model, case, solver, dt):
    return _cached_run(model, case, solver, dt)[0][-1]


def _error(model, case, solver, dt):
    return float(np.max(np.abs(_final_voltage(model, case, solver, dt) - np.asarray(REFERENCE[(model, case)]))))


def _algebraic_points(cell):
    midpoints = set(np.asarray(cell.runtime.node_tree.cv_to_mid_node_id).tolist())
    return [pid for pid in range(cell.runtime.n_point) if pid not in midpoints]


class _NumpyCable:
    """The point system of a uniform-cylinder tree in nF, uS, nA, mV, ms."""

    def __init__(self, model):
        self.capacitance, self.leak, self.edges, self.mids, self.n = [], [], [], [], 0
        if model == "M1":
            self.junction = self._add(600.0, 0.5)[0]
        else:
            p0, p1 = self._add(600.0, 0.5)
            self._add(300.0, 0.4, prox=p1)
            _, c2 = self._add(450.0, 0.3, prox=p1)
            self.junction, self.c2_tip = p1, c2

    def _node(self, capacitance=0.0, leak=0.0):
        self.capacitance.append(capacitance)
        self.leak.append(leak)
        self.n += 1
        return self.n - 1

    def _add(self, length_um, radius_um, prox=None, n_cv=3):
        prox = self._node() if prox is None else prox
        h, r = length_um * 1e-4 / n_cv, radius_um * 1e-4
        area = 2 * np.pi * r * h
        half = 1e6 * np.pi * r**2 / (100.0 * h / 2)
        previous = prox
        for j in range(n_cv):
            mid = self._node(1e3 * area, 1e3 * 0.1 * area)
            self.mids.append(mid)
            self.edges.append((previous, mid, half if j == 0 else half / 2))
            previous = mid
        dist = self._node()
        self.edges.append((previous, dist, half))
        return prox, dist

    def rhs(self, clamps, synapse_node):
        """``f(y)`` for ``y = (V of every CV, g)``, the boundary rows eliminated."""
        G = np.zeros((self.n, self.n))
        for i, j, g in self.edges:
            G[i, i] += g
            G[j, j] += g
            G[i, j] -= g
            G[j, i] -= g
        G += np.diag(self.leak)
        source = np.asarray(self.leak) * -65.0
        for node, value in clamps.items():
            source[node] += value
        D = np.asarray(self.mids)
        A = np.asarray([i for i in range(self.n) if i not in set(self.mids)])
        C = np.asarray(self.capacitance)[D]

        def f(y):
            V, g = y[:-1], y[-1]
            Gs, b = G.copy(), source.copy()
            Gs[synapse_node, synapse_node] += g
            Gr = Gs[np.ix_(D, D)] - Gs[np.ix_(D, A)] @ np.linalg.solve(Gs[np.ix_(A, A)], Gs[np.ix_(A, D)])
            br = b[D] - Gs[np.ix_(D, A)] @ np.linalg.solve(Gs[np.ix_(A, A)], b[A])
            return np.concatenate([(br - Gr @ V) / C, [-g / 2.0]])

        return f


def _same_scheme(model, case, solver, dt):
    cable = _NumpyCable(model)
    if model == "M1":
        clamps = {cable.junction: CLAMP} if case in ("clamp", "both") else {}
        g0 = G0 if case in ("syn", "both") else 0.0
    else:
        clamps = {cable.junction: CLAMP, cable.c2_tip: -0.02}
        g0 = G0
    f = cable.rhs(clamps, cable.junction)
    step = integrator_info(solver).fn
    y = (np.concatenate([np.full(len(cable.mids), -65.0), [g0]]),)
    out = []
    with brainstate.environ.context(dt=dt):
        for i in range(round(CHECK[-1] / dt)):
            y = (np.asarray(step(lambda state, t: (f(np.asarray(state[0])),), y, i * dt)[0]),)
            if any(abs((i + 1) * dt - end) < 1e-9 for end in CHECK):
                out.append(y[0][:-1])
    return np.stack(out)


class _Float64Case(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        stack.enter_context(_float64())
        self.addCleanup(stack.close)


def _explicit_solvers():
    return [name for name in list_integrators() if integrator_info(name).kind == "ode"]


class SameSchemeTest(_Float64Case):
    """Acceptance 1 and 4: the derivative is the reduced system's, to rounding."""

    def _assert_same(self, model, case, solver, dt):
        np.testing.assert_allclose(
            _cached_run(model, case, solver, dt)[0], _same_scheme(model, case, solver, dt), rtol=0, atol=1e-10
        )

    def test_every_explicit_solver(self):
        names = _explicit_solvers()
        self.assertGreaterEqual(len(names), 19)
        for name in names:
            with self.subTest(solver=name):
                self._assert_same("M1", "both", name, 0.25 if integrator_info(name).order >= 8 else 0.025)

    def test_clamp_only_synapse_only_and_branching(self):
        for model, case in (("M1", "clamp"), ("M1", "syn"), ("M3", "both")):
            with self.subTest(model=model, case=case):
                self._assert_same(model, case, "rk4", 0.025)


class ConvergenceTest(_Float64Case):
    """Acceptance 2 (M1 both), 3 (M2) and 4 (M3)."""

    def _ratios(self, model, solver, dts, case="both"):
        errors = [_error(model, case, solver, dt) for dt in dts]
        return errors, [a / b for a, b in zip(errors, errors[1:])]

    def test_m1_low_order(self):
        cases = {
            "euler": ((1.9, 2.1), 5e-2),
            "ralston2": ((3.8, 4.3), 2.5e-4),
            "rk3": ((7.5, 8.7), 1.2e-6),
            "rk4": ((14.0, 18.0), 5e-9),
        }
        for solver, ((low, high), bound) in cases.items():
            with self.subTest(solver=solver):
                errors, ratios = self._ratios("M1", solver, (0.05, 0.025, 0.0125))
                self.assertLessEqual(errors[1], bound)
                for ratio in ratios:
                    self.assertTrue(low <= ratio <= high, ratios)

    def test_m1_embedded_pairs(self):
        cases = {"bs32": (7.5, 9.5), "rk23": (7.5, 9.5), "rk4_38": (14.0, 18.0)}
        for solver, (low, high) in cases.items():
            with self.subTest(solver=solver):
                _, ratios = self._ratios("M1", solver, (0.2, 0.1, 0.05))
                for ratio in ratios:
                    self.assertTrue(low <= ratio <= high, ratios)
        for solver in ("rk45", "rkf45", "dopri5"):
            with self.subTest(solver=solver):
                _, (ratio,) = self._ratios("M1", solver, (0.1, 0.05))
                self.assertTrue(28.0 <= ratio <= 45.0, ratio)
        errors, (ratio,) = self._ratios("M1", "dopri8", (0.5, 0.25))
        self.assertGreaterEqual(ratio, 150.0)
        self.assertLessEqual(errors[0], 1e-9)

    def test_single_cv(self):
        errors, ratios = self._ratios("M2", "rk4", (0.05, 0.025, 0.0125))
        self.assertLessEqual(errors[1], 1.5e-9)
        for ratio in ratios:
            self.assertTrue(14.0 <= ratio <= 18.0, ratios)
        _, ratios = self._ratios("M2", "euler", (0.05, 0.025, 0.0125))
        for ratio in ratios:
            self.assertTrue(1.9 <= ratio <= 2.1, ratios)

    def test_branching(self):
        errors, ratios = self._ratios("M3", "rk4", (0.05, 0.025, 0.0125))
        self.assertLessEqual(errors[1], 4e-9)
        for ratio in ratios:
            self.assertTrue(14.0 <= ratio <= 18.0, ratios)
        _, ratios = self._ratios("M3", "ralston2", (0.05, 0.025, 0.0125))
        for ratio in ratios:
            self.assertTrue(3.8 <= ratio <= 4.3, ratios)

    def test_cached_boundary_voltage(self):
        for model in ("M2", "M3"):
            with self.subTest(model=model):
                cell = _cached_run(model, "both", "rk4", 0.0125)[1]
                point = np.asarray(cell._point_voltage_for_mechanisms().to_decimal(u.mV))[0]
                np.testing.assert_allclose(point[_algebraic_points(cell)], REFERENCE_BOUNDARY[model], rtol=0, atol=1e-8)


class ConservationTest(_Float64Case):
    """Acceptance 5: every boundary input reaches the CVs."""

    def test_total_current(self):
        cell = _build("M3")
        V = -65.0 + 10.0 * np.random.default_rng(0).standard_normal(9)
        cell.V.value = u.Quantity(V[None, :], u.mV)
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            cell._prepare_step_clamps()
            derivative = np.asarray(cell.compute_voltage_derivative(cell.V.value).to_decimal(u.mV / u.ms))[0]
        capacitance = np.asarray([cv.area.to_decimal(u.cm**2) for cv in cell.cvs]) * 1e3  # nF at 1 uF/cm^2
        self.assertAlmostEqual(float(np.sum(capacitance * derivative)), 0.2559119282055959, delta=1e-12)


class PopulationAndBatchTest(_Float64Case):
    """Acceptance 6."""

    def test_population_and_batch_match_one_cell(self):
        single = _run(_build("M3"), 0.025)[:, 0, :]
        for kwargs in ({"pop_size": (2,)}, {"batch_size": 2}):
            with self.subTest(**kwargs):
                values = _run(_build("M3", **kwargs), 0.025).reshape(len(CHECK), 2, 9)
                np.testing.assert_allclose(
                    values, np.broadcast_to(single[:, None, :], values.shape), rtol=0, atol=1e-12
                )

    def test_per_member_placement(self):
        def build(pop_size, locations):
            kwargs = {} if pop_size is None else {"pop_size": pop_size}
            cell = braincell.Cell(
                _morphology("M1"),
                cv_policy=braincell.CVPerBranch(cv_per_branch=3),
                V_init=-65.0 * u.mV,
                solver="rk4",
                **kwargs,
            )
            cell.paint(
                AllRegion(),
                mech.CableProperty(
                    resting_potential=-65.0 * u.mV,
                    membrane_capacitance=1.0 * u.uF / u.cm**2,
                    axial_resistivity=100.0 * u.ohm * u.cm,
                ),
                mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
            )
            cell.place(locations, mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
            cell.init_state()
            node = _synapse_node(cell)
            node.g.value = u.Quantity(jnp.full(node.g.value.shape, G0), u.uS)
            return _run(cell, 0.025, segments=(5.0,))[-1]

        members = build((2,), [RootLocation(0.0), RootLocation(0.5)])
        np.testing.assert_allclose(members[0], build(None, RootLocation(0.0))[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(members[1], build(None, RootLocation(0.5))[0], rtol=0, atol=1e-12)


class SegmentsAndResetTest(_Float64Case):
    """Acceptance 7."""

    def test_segments_and_reset(self):
        whole = _build("M1")
        initial_point = np.asarray(whole._point_V.value.to_decimal(u.mV))
        whole_V = _run(whole, 0.025, segments=(5.0,))[-1]
        split = _build("M1")
        split_V = _run(split, 0.025)[-1]
        np.testing.assert_array_equal(split_V, whole_V)
        np.testing.assert_array_equal(split._point_V.value.to_decimal(u.mV), whole._point_V.value.to_decimal(u.mV))

        whole.reset_state()
        np.testing.assert_array_equal(whole._point_V.value.to_decimal(u.mV), initial_point)
        node = _synapse_node(whole)
        node.g.value = u.Quantity(jnp.full(node.g.value.shape, G0), u.uS)
        np.testing.assert_array_equal(_run(whole, 0.025, segments=(5.0,))[-1], whole_V)


class EventTest(_Float64Case):
    """Acceptance 8: a NetStim event at 1 ms into the boundary ExpSyn."""

    def test_event_converges(self):
        reference = np.asarray([-46.02427839636623, -52.88874609950384, -56.8186080770654])
        for solver, (low, high), bound in (("rk4", (14.0, 18.0), 1.5e-8), ("ralston2", (3.8, 4.3), None)):
            with self.subTest(solver=solver):
                # One 5 ms run: Cell.run split at the event time delivers it twice
                # (the same on every solver; not part of this change).
                errors = [
                    float(
                        np.max(
                            np.abs(
                                _run(_build("M1", "event", solver, netstim=(1.0, G0)), dt, segments=(5.0,))[-1]
                                - reference
                            )
                        )
                    )
                    for dt in (0.05, 0.025, 0.0125)
                ]
                if bound is not None:
                    self.assertLessEqual(errors[1], bound)
                for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
                    self.assertTrue(low <= ratio <= high, errors)


class RecordingTest(_Float64Case):
    """Acceptance 9: a boundary synapse current is evaluated at the boundary voltage."""

    def test_synapse_current_record(self):
        cell = braincell.Cell(
            _morphology("M1"), cv_policy=braincell.CVPerBranch(cv_per_branch=3), V_init=-65.0 * u.mV, solver="rk4"
        )
        cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
            mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
        )
        cell.place(RootLocation(0.0), mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
        cell.loc(RootLocation(0.0)).record("isyn", braincell.observe.synapse(name="syn").current(reduce="none"))
        cell.init_state()
        node = _synapse_node(cell)
        node.g.value = u.Quantity(jnp.full(node.g.value.shape, G0), u.uS)
        result = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
        times = np.asarray(result.samples["isyn"].time.to_decimal(u.ms)).ravel()
        values = np.asarray(result.samples["isyn"].values.to_decimal(u.nA)).ravel()
        sample = values[int(np.argmin(np.abs(times - 0.5)))]
        self.assertAlmostEqual(float(sample), 0.14015649654334897, delta=1e-8)


class GradientTest(_Float64Case):
    """Acceptance 10: training sees the boundary synapse."""

    def test_gradient_matches_central_difference(self):
        cell = _build("M1")
        node = _synapse_node(cell)

        def objective(g0):
            cell.reset_state()
            cell.V.value = u.Quantity(jnp.full(cell.V.value.shape, -65.0), u.mV)
            node.g.value = u.Quantity(jnp.broadcast_to(g0, node.g.value.shape), u.uS)

            def step(i):
                with brainstate.environ.context(t=i * 0.025 * u.ms):
                    cell.update()
                return cell.V.value

            trace = brainstate.transform.for_loop(step, jnp.arange(40))
            return jnp.sum(trace[-1].to_decimal(u.mV))

        with brainstate.environ.context(dt=0.025 * u.ms):
            g0 = jnp.asarray(G0)
            gradient = float(brainstate.transform.grad(objective)(g0))
            eps = 1e-6
            central = float((objective(g0 + eps) - objective(g0 - eps)) / (2 * eps))
        self.assertGreater(abs(central), 1e3)
        self.assertLessEqual(abs(gradient - central) / abs(central), 1e-6)


class RuntimeGeometryTest(_Float64Case):
    """Acceptance 11: boundary coefficients follow geometry set at runtime."""

    def test_axial_resistivity_set_at_runtime(self):
        changed = _build("M1", "event", netstim=(0.1, G0))
        changed.geometry.Ra.set(120.0 * u.ohm * u.cm)
        fresh = _build("M1", "event", netstim=(0.1, G0), Ra=120.0)
        changed_V = _run(changed, 0.025, segments=(2.0,))[-1].ravel()
        fresh_V = _run(fresh, 0.025, segments=(2.0,))[-1].ravel()
        np.testing.assert_allclose(changed_V, fresh_V, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fresh_V, [-44.7866885, -57.68084012, -62.58027591], rtol=0, atol=1e-6)


class MgBlockedExpSyn(ExpSyn):
    """Test-only ExpSyn with an NMDA-like magnesium block: I = g B(V) (e - V)."""

    def current(self, V_post):
        block = 1.0 / (1.0 + u.math.exp(-0.062 * V_post.to_decimal(u.mV)) / 3.57)
        return self.g.value * block * (self.e - V_post)


class NonlinearBoundaryTest(_Float64Case):
    """Acceptance 12: a nonlinear boundary current keeps the solver's order."""

    REFERENCE = [-46.91191867156262, -51.440052486691414, -54.43799841874587]
    REFERENCE_TIGHT = [-46.91191867156312, -51.4400524866912, -54.43799841874637]

    def _errors(self, solver, dts, *, netstim=False, reference=None):
        errors = []
        for dt in dts:
            if netstim:
                cell = _build("M1", "event", solver, synapse=MgBlockedExpSyn, netstim=(0.0, 0.02))
            else:
                cell = _build("M1", "syn", solver, synapse=MgBlockedExpSyn)
                node = _synapse_node(cell)
                node.g.value = u.Quantity(jnp.full(node.g.value.shape, 0.02), u.uS)
            V = _run(cell, dt, segments=(5.0,))[-1].ravel()
            errors.append(float(np.max(np.abs(V - np.asarray(reference or self.REFERENCE)))))
        return errors

    def _assert_ratios(self, errors, low, high):
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertTrue(low <= ratio <= high, errors)

    def test_rk4_ralston2_staggered(self):
        dts = (0.05, 0.025, 0.0125)
        for netstim in (False, True):
            with self.subTest(netstim=netstim):
                errors = self._errors("rk4", dts, netstim=netstim)
                self.assertLessEqual(errors[1], 1.5e-7)
                self._assert_ratios(errors, 14.0, 18.0)
        self._assert_ratios(self._errors("ralston2", dts), 3.8, 4.5)
        self._assert_ratios(self._errors("staggered", dts), 1.9, 2.1)

    def test_dopri8(self):
        errors = self._errors("dopri8", (0.25, 0.125), reference=self.REFERENCE_TIGHT)
        self.assertGreaterEqual(errors[0] / errors[1], 100.0)

    def test_one_newton_step_or_no_step_start_solve_loses_order(self):
        dts = (0.05, 0.025, 0.0125)
        with mock.patch.object(boundary, "NEWTON_STEPS", 1):
            errors = self._errors("rk4", dts)
        self.assertLessEqual(max(errors[0] / errors[1], errors[1] / errors[2]), 4.0)
        with mock.patch.object(braincell.Cell, "pre_integral", _pre_integral_without_boundary_solve):
            errors = self._errors("rk4", dts, netstim=True)
        self.assertLessEqual(max(errors[0] / errors[1], errors[1] / errors[2]), 4.0)


_PRE_INTEGRAL = braincell.Cell.pre_integral


def _pre_integral_without_boundary_solve(self):
    with mock.patch.object(braincell.Cell, "_update_boundary_voltage", lambda cell: None):
        return _PRE_INTEGRAL(self)


class NoBoundaryInputTest(_Float64Case):
    """Acceptance 13: a model without a boundary input never solves a boundary row."""

    def test_no_solve_without_boundary_input(self):
        with mock.patch.object(boundary, "solve_boundary", wraps=boundary.solve_boundary) as solve:
            for case in ("none", "mid_clamp"):
                for solver in ("rk4", "exp_euler", "staggered"):
                    with self.subTest(case=case, solver=solver):
                        cell = _build("M1", case, solver)
                        _run(cell, 0.05, segments=(0.1,))
            self.assertEqual(solve.call_count, 0)
            _run(_build("M1", "clamp", "rk4"), 0.05, segments=(0.1,))
            self.assertGreater(solve.call_count, 0)

    def test_boundary_rows(self):
        rows = _build("M3")._boundary_rows()
        np.testing.assert_array_equal(rows.input_ids, [4, 12])
        # The branch point joins three CVs (p's last, c1's and c2's first), the tip one.
        np.testing.assert_array_equal(rows.edge_slot, [0, 0, 0, 1])
        np.testing.assert_array_equal(rows.edge_cv, [2, 3, 6, 8])
        self.assertEqual(len(rows.synapse_layout_ids), 1)
        self.assertFalse(_build("M1", "mid_clamp")._boundary_rows().has_inputs)

    def test_adjacent_algebraic_points_are_rejected(self):
        # Points 2 and 3 are both outside the one CV midpoint (point 1).
        edges = [types.SimpleNamespace(id=i, parent_node_id=i, child_node_id=i + 1) for i in range(3)]
        runtime = types.SimpleNamespace(
            node_tree=types.SimpleNamespace(cv_to_mid_node_id=np.asarray([1]), edges=edges),
            n_point=4,
            point_to_representative_cv_np=np.zeros(4, dtype=np.int32),
            clamp_routing_table=types.SimpleNamespace(boundary_ids=np.asarray([3])),
            iter_synapse_layouts=lambda: iter(()),
        )
        with self.assertRaisesRegex(RuntimeError, "joins two algebraic points"):
            boundary.build_boundary_rows(runtime, cvs=())


class OtherSolversTest(_Float64Case):
    """Acceptance 14: the block and exponential solvers read the same derivative."""

    def test_clamp_only(self):
        dts = (0.05, 0.025, 0.0125)
        for dt in dts:
            self.assertLessEqual(_error("M1", "clamp", "exp_euler", dt), 1e-9)
        for solver, (low, high) in (
            ("ind_exp_euler", (1.9, 2.1)),
            ("backward_euler", (1.9, 2.1)),
            ("crank_nicolson", (3.8, 4.2)),
        ):
            with self.subTest(solver=solver):
                errors = [_error("M1", "clamp", solver, dt) for dt in dts]
                for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
                    self.assertTrue(low <= ratio <= high, errors)

    def test_ind_exp_euler_with_synapse(self):
        errors = [_error("M1", "both", "ind_exp_euler", dt) for dt in (0.05, 0.025, 0.0125)]
        for ratio in (errors[0] / errors[1], errors[1] / errors[2]):
            self.assertTrue(1.9 <= ratio <= 2.1, errors)

    def test_block_solvers_still_reject_synapses(self):
        for solver in ("exp_euler", "backward_euler", "crank_nicolson"):
            with self.subTest(solver=solver):
                with self.assertRaisesRegex(ValueError, "point synapses"):
                    _build("M1", "syn", solver)


class EventVoltageTest(_Float64Case):
    """Event delivery hands a boundary synapse its point's voltage, not 0 mV."""

    def test_both_delivery_paths_pass_the_boundary_voltage(self):
        cell = _build("M1", "both", "rk4", autapse=True)
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            cell.update()
        expected = float(cell._point_voltage_for_mechanisms()[0, 0].to_decimal(u.mV))
        self.assertGreater(expected, -60.0)  # moved away from V_init by the inputs
        node = _synapse_node(cell)
        seen = []

        def apply_events(payload, V_post=None):
            seen.append(float(np.ravel(V_post.to_decimal(u.mV))[0]))

        layout_id = cell.synapses["syn"]._store.layout_id("ExpSyn")
        with mock.patch.object(node, "apply_events", side_effect=apply_events):
            with brainstate.environ.context(t=0.025 * u.ms, dt=0.025 * u.ms):
                cell._apply_direct_live_connection_events()
                cell._apply_synapse_layout_event_drive(layout_id, u.Quantity(jnp.zeros((1,)), u.uS))
        self.assertEqual(seen, [expected, expected])


class SynapseLayoutFilterTest(_Float64Case):
    """The boundary solve evaluates only synapse layouts with a row on a boundary."""

    def test_midpoint_layout_is_skipped(self):
        reference = _cached_run("M1", "both", "rk4", 0.025)[0]
        cell = braincell.Cell(
            _morphology("M1"), cv_policy=braincell.CVPerBranch(cv_per_branch=3), V_init=-65.0 * u.mV, solver="rk4"
        )
        cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
            mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
        )
        cell.place(RootLocation(0.0), mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
        cell.place(
            RootLocation(0.0), braincell.CurrentClamp(delay=0.0 * u.ms, durations=100.0 * u.ms, amplitudes=CLAMP * u.nA)
        )
        cell.place(RootLocation(0.5), mech.Synapse("Exp2Syn", name="mid", tau1=0.5 * u.ms, tau2=2.0 * u.ms))
        cell.init_state()
        node = _synapse_node(cell)
        node.g.value = u.Quantity(jnp.full(node.g.value.shape, G0), u.uS)
        rows = cell._boundary_rows()
        self.assertEqual(len(rows.synapse_layout_ids), 1)
        # The midpoint synapse carries no conductance, so the run is M1's.
        np.testing.assert_allclose(_run(cell, 0.025)[:, 0, :], reference, rtol=0, atol=1e-12)


def _state_probe():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mech.StateProbe(field="v")


class StateProbeTest(_Float64Case):
    """The deprecated StateProbe reads the solved boundary voltage under every solver."""

    def test_state_probe_at_a_boundary(self):
        for solver in ("rk4", "staggered"):
            with self.subTest(solver=solver):
                cell = _build("M1", "clamp", solver)
                _run(cell, 0.025, segments=(1.0,))
                point = cell._point_voltage_for_mechanisms()
                sampled = braincell._multi_compartment.probes._sample_state_probe_point(
                    cell, cell.runtime, declaration=_state_probe(), point_id=0
                )
                self.assertEqual(float(np.ravel(sampled.to_decimal(u.mV))[0]), float(point[0, 0].to_decimal(u.mV)))
                # The clamp current enters CV 0 through the half-CV conductance: V_b > V_0.
                self.assertGreater(float(point[0, 0].to_decimal(u.mV)), float(cell.V.value[0, 0].to_decimal(u.mV)))


if __name__ == "__main__":
    unittest.main()
