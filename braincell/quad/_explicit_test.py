# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


import math
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.quad import (
    euler_step,
    get_integrator,
    heun2_step,
    heun3_step,
    midpoint_step,
    module_step_from_ode,
    ralston2_step,
    ralston3_step,
    ralston4_step,
    rk3_step,
    rk4_step,
    ssprk3_step,
)
from braincell.quad._testing import LinearDecay, drive, integrate
from braincell.quad.protocol import DiffEqModule, DiffEqSingleState, IndependentIntegration


class RungeKuttaConvergenceTest(unittest.TestCase):
    """Verifies each RK step on a linear ODE with known analytical solution."""

    # The short registry name, not ``func.__name__``, so a subTest label
    # reads ``rk4`` rather than ``rk4_step``.
    METHODS_AND_ORDERS = [
        ('euler', euler_step, 1),
        ('midpoint', midpoint_step, 2),
        ('heun2', heun2_step, 2),
        ('ralston2', ralston2_step, 2),
        ('rk3', rk3_step, 3),
        ('heun3', heun3_step, 3),
        ('ssprk3', ssprk3_step, 3),
        ('ralston3', ralston3_step, 3),
        ('rk4', rk4_step, 4),
        ('ralston4', ralston4_step, 4),
    ]

    def _final_value(self, method, dt_ms, n_steps, tau_ms=10.0):
        return drive(method, dt_ms=dt_ms, n_steps=n_steps, tau_ms=tau_ms)[0]

    def test_each_method_matches_analytical_solution(self):
        # 100 steps of dt=0.1 ms over a tau=10 ms decay → final value
        # should be close to exp(-1) ≈ 0.367879.
        target = math.exp(-1.0)
        for name, method, order in self.METHODS_AND_ORDERS:
            with self.subTest(method=name, order=order):
                final = self._final_value(method, dt_ms=0.1, n_steps=100)
                # Allow looser tolerance for low-order methods.
                tol = {1: 5e-3, 2: 1e-4, 3: 1e-5, 4: 1e-6}[order]
                self.assertAlmostEqual(final, target, delta=tol)

    def test_global_error_decreases_with_step_size(self):
        # Each integrator should be consistent: shrinking the step size
        # from 0.4 ms down to 0.1 ms should not *increase* the global error.
        # (Going below 0.05 ms makes float32 noise dominate the higher-order
        # methods, so we stop there.)
        target = math.exp(-1.0)
        for name, method, order in self.METHODS_AND_ORDERS:
            with self.subTest(method=name):
                err_coarse = abs(self._final_value(method, dt_ms=0.4, n_steps=25) - target)
                err_fine = abs(self._final_value(method, dt_ms=0.1, n_steps=100) - target)
                # Small noise margin so float32 round-off near the noise
                # floor doesn't trip an otherwise consistent method.
                self.assertLessEqual(err_fine, err_coarse + 1e-5)

    def test_convergence_order_estimate(self):
        # Empirically estimate convergence order on the linear decay using
        # only first- and second-order methods, where the error stays well
        # above the float32 noise floor for the step sizes considered.
        target = math.exp(-1.0)
        order_methods = [(name, m, o) for name, m, o in self.METHODS_AND_ORDERS if o <= 2]
        for name, method, order in order_methods:
            with self.subTest(method=name, order=order):
                err1 = abs(self._final_value(method, dt_ms=0.4, n_steps=25) - target)
                err2 = abs(self._final_value(method, dt_ms=0.2, n_steps=50) - target)
                if err1 < 1e-6 or err2 == 0:
                    continue
                empirical = math.log(err1 / err2, 2)
                # Require at least half the theoretical order to allow for
                # higher-order error terms on this small problem.
                self.assertGreater(empirical, 0.5 * order)

    def test_pre_and_post_integral_called_once_per_step(self):
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        self.assertEqual(m.pre_calls, 1)
        self.assertEqual(m.post_calls, 1)

    def test_aux_state_unchanged_after_step(self):
        m = LinearDecay()
        before = np.array(m.aux.value)
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        np.testing.assert_array_equal(m.aux.value, before)


# Trajectories of the braincell kernels these steps replaced, recorded on main
# at bbf73a3 (before braintools' kernels were adopted) with float64. Per method:
# the first element of LinearDecay after steps 10, 20, ..., 100 (dt = 0.1 ms),
# and the HH voltage of ``integrate(name, dt=0.025 ms)`` after steps
# 40, 80, ..., 400 (indices 39, 79, ..., 399). Values in mV.
_REFERENCE = {
    'euler': (
        [
            0.9043820750088045,
            0.8179069375972309,
            0.7397003733882804,
            0.6689717585696805,
            0.6050060671375367,
            0.5471566423907614,
            0.4948386596002073,
            0.447523213763811,
            0.4047319726783242,
            0.3660323412732296,
        ],
        [
            -62.01760354837533,
            -61.31699167877802,
            -60.20191797320214,
            -58.37901209464455,
            -54.646055429578034,
            -36.53046927295392,
            32.440537842938575,
            -14.480652310912717,
            -47.36078130145155,
            -73.61418637870561,
        ],
    ),
    'midpoint': (
        [
            0.9048389374553066,
            0.8187335027352484,
            0.7408219526740235,
            0.6703245485011288,
            0.6065357522159694,
            0.5488171655637529,
            0.4965911409459393,
            0.44933500032324214,
            0.40657580425396234,
            0.367885618716192,
        ],
        [
            -62.01752081858469,
            -61.31445136415253,
            -60.19301604275633,
            -58.35184942491948,
            -54.5228008615376,
            -33.71374591049709,
            30.121062711237556,
            -15.837519908190975,
            -48.07086184750026,
            -73.54531503074976,
        ],
    ),
    'heun2': (
        [
            0.9048389374553066,
            0.8187335027352484,
            0.7408219526740235,
            0.6703245485011288,
            0.6065357522159696,
            0.548817165563753,
            0.4965911409459393,
            0.44933500032324214,
            0.40657580425396234,
            0.367885618716192,
        ],
        [
            -62.01745059832647,
            -61.314362310714316,
            -60.19288888614721,
            -58.35160617938835,
            -54.52191398780325,
            -33.68934569049437,
            30.115346847944416,
            -15.837539561895795,
            -48.07349335578558,
            -73.54367447085689,
        ],
    ),
    'ralston2': (
        [
            0.9048389374553066,
            0.8187335027352484,
            0.7408219526740235,
            0.6703245485011288,
            0.6065357522159696,
            0.548817165563753,
            0.4965911409459393,
            0.44933500032324214,
            0.40657580425396234,
            0.367885618716192,
        ],
        [
            -62.0174977510476,
            -61.31442205342813,
            -60.19297411940455,
            -58.35176903537673,
            -54.5225068544936,
            -33.705618188214075,
            30.11910441254974,
            -15.83753608859627,
            -48.071757661762526,
            -73.54479278694353,
        ],
    ),
    'rk3': (
        [
            0.9048374142355164,
            0.8187307462004155,
            0.7408182113470988,
            0.670320034773889,
            0.6065306469750671,
            0.5488116222635144,
            0.4965852891913175,
            0.4493289490192679,
            0.40656964437175647,
            0.3678794257199936,
        ],
        [
            -62.017447977563165,
            -61.314321181015224,
            -60.19283393547869,
            -58.351506333862964,
            -54.52140544954786,
            -33.65267757285895,
            30.09178586752673,
            -15.884198073831566,
            -48.112988611171396,
            -73.5573674538803,
        ],
    ),
    'heun3': (
        [
            0.9048374142355164,
            0.8187307462004155,
            0.7408182113470988,
            0.670320034773889,
            0.6065306469750671,
            0.5488116222635144,
            0.4965852891913175,
            0.4493289490192679,
            0.40656964437175647,
            0.3678794257199936,
        ],
        [
            -62.01744648371446,
            -61.31431947645017,
            -60.19283190507137,
            -58.35150372144353,
            -54.52140418000219,
            -33.653186229798905,
            30.09153587498345,
            -15.884269873232228,
            -48.113000201356975,
            -73.55730954525382,
        ],
    ),
    'ssprk3': (
        [
            0.9048374142355164,
            0.8187307462004155,
            0.7408182113470987,
            0.6703200347738889,
            0.606530646975067,
            0.5488116222635144,
            0.4965852891913175,
            0.4493289490192679,
            0.4065696443717564,
            0.36787942571999355,
        ],
        [
            -62.01744521883437,
            -61.31431804637165,
            -60.19283013475954,
            -58.351500980350906,
            -54.52139637208522,
            -33.65299822328776,
            30.09287082444258,
            -15.88409059130276,
            -48.112827789306884,
            -73.55714647757279,
        ],
    ),
    'ralston3': (
        [
            0.9048374142355164,
            0.8187307462004155,
            0.7408182113470987,
            0.6703200347738889,
            0.606530646975067,
            0.5488116222635144,
            0.4965852891913175,
            0.4493289490192679,
            0.40656964437175647,
            0.3678794257199936,
        ],
        [
            -62.017446858053994,
            -61.31431990238206,
            -60.19283240557011,
            -58.35150431324455,
            -54.52140372326914,
            -33.65298864920143,
            30.091815834051598,
            -15.884230887176717,
            -48.11298281677746,
            -73.55731183787496,
        ],
    ),
    'rk4': (
        [
            0.904837418043563,
            0.8187307530917416,
            0.7408182207003933,
            0.6703200460581703,
            0.6065306597381169,
            0.5488116361216965,
            0.49658530382061933,
            0.44932896414742757,
            0.40656965977134696,
            0.3678794412023554,
        ],
        [
            -62.0174478180204,
            -61.31432202993343,
            -60.192835174519516,
            -58.35150833940611,
            -54.52140961965558,
            -33.65227501003754,
            30.089850146377497,
            -15.88500937839205,
            -48.11347787174507,
            -73.55739606286348,
        ],
    ),
    'ind_exp_euler': (
        [
            0.9048374180359597,
            0.8187307530779819,
            0.7408182206817179,
            0.6703200460356392,
            0.6065306597126333,
            0.5488116360940263,
            0.4965853037914093,
            0.4493289641172214,
            0.40656965974059894,
            0.3678794411714422,
        ],
        [
            -62.021126247765906,
            -61.324525341035375,
            -60.216701166551196,
            -58.41511455780663,
            -54.77965093598487,
            -38.395494629491324,
            34.92843561456723,
            -11.421065116336626,
            -44.469117886009286,
            -72.2679597013316,
        ],
    ),
}


class _Float64Decay(brainstate.nn.Module, DiffEqModule):
    """LinearDecay in float64 (``LinearDecay`` fixes its dtype at import)."""

    def __init__(self):
        super().__init__()
        self.tau = 10.0 * u.ms
        self.x = DiffEqSingleState(jnp.ones(3, dtype=jnp.float64) * u.mV)

    def compute_derivative(self, *args):
        self.x.derivative = -self.x.value / self.tau


def _decay_trace(step):
    m = _Float64Decay()
    dt = 0.1 * u.ms
    out = []
    with brainstate.environ.context(dt=dt):
        for i in range(100):
            with brainstate.environ.context(t=i * dt):
                step(m)
            out.append(float(m.x.value.to_decimal(u.mV)[0]))
    return np.asarray(out)[9::10]


class UnchangedNumericsTest(unittest.TestCase):
    """The braintools kernels reproduce the replaced braincell kernels (float64)."""

    def test_reference_trajectories(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name, (decay_ref, hh_ref) in _REFERENCE.items():
                with self.subTest(method=name):
                    tol = 1e-14 if name == 'ind_exp_euler' else 1e-12
                    step = get_integrator(name)
                    np.testing.assert_allclose(_decay_trace(step), decay_ref, rtol=tol, atol=0)
                    hh = np.asarray(integrate(name, dt=0.025 * u.ms).to_decimal(u.mV))[39::40, 0]
                    np.testing.assert_allclose(hh, hh_ref, rtol=tol, atol=0)

    def test_ralston4_uses_exact_coefficients(self):
        # braincell's former ralston4 truncated its coefficients to 8 digits;
        # the exact ones move LinearDecay by ~1e-11 relative, not more.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            got = _decay_trace(ralston4_step)
        np.testing.assert_allclose(got, np.exp(-np.arange(1, 11) / 10.0), rtol=1e-9)
        self.assertIs(get_integrator('ralston4'), ralston4_step)


class _TimeDriven(brainstate.nn.Module, DiffEqModule):
    """dx/dt = t * (1 mV / ms**2), reading t from the environment."""

    def __init__(self):
        super().__init__()
        self.x = DiffEqSingleState(jnp.zeros(1) * u.mV)

    def compute_derivative(self, *args):
        t = brainstate.environ.get('t')
        self.x.derivative = jnp.ones(1) * t * (u.mV / u.ms**2)


class _WritesAux(LinearDecay):
    def compute_derivative(self, *args, **kwargs):
        super().compute_derivative()
        self.aux.value = self.aux.value + 1.0


class _Pool(brainstate.nn.Module, DiffEqModule, IndependentIntegration):
    """Advances itself (``ind_update``); its host only reads ``c``."""

    def __init__(self):
        IndependentIntegration.__init__(self, 'euler')
        brainstate.nn.Module.__init__(self)
        self.c = DiffEqSingleState(jnp.full(1, 2.0) * u.mM)

    def compute_derivative(self, *args):
        self.c.derivative = -self.c.value / (1.0 * u.ms)


class _ReadsPool(brainstate.nn.Module, DiffEqModule):
    """dx/dt = -c x / (10 mM ms), with ``c`` owned by an independently integrated pool."""

    def __init__(self):
        super().__init__()
        self.x = DiffEqSingleState(jnp.ones(1) * u.mV)
        self.pool = _Pool()

    def compute_derivative(self, *args):
        self.x.derivative = -self.pool.c.value * self.x.value / (10.0 * u.mM * u.ms)


class _ForgetsState(LinearDecay):
    """Reads a second DiffEqState of its own but never sets its derivative."""

    def __init__(self):
        super().__init__(shape=(1,))
        self.y = DiffEqSingleState(jnp.ones(1) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -(self.x.value + 0.0 * self.y.value) / self.tau


class ModuleStepFromOdeTest(unittest.TestCase):
    def test_independent_states_are_read_as_constants(self):
        # A Markov channel or kinetic ion is read by its host's derivative and
        # advanced by ``ind_update``; the explicit step used to reject it.
        for name in ('euler', 'ralston2', 'rk4', 'dopri5'):
            with self.subTest(method=name):
                m = _ReadsPool()
                with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                    get_integrator(name)(m)
                np.testing.assert_allclose(m.pool.c.value.to_decimal(u.mM), [2.0])
                np.testing.assert_allclose(m.x.value.to_decimal(u.mV), np.exp(-0.02), rtol=1e-3)

    def test_an_unwritten_dependent_state_still_raises(self):
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            with self.assertRaisesRegex(ValueError, 'must be written by compute_derivative'):
                rk4_step(_ForgetsState())

    def test_same_object_for_same_ode_step(self):
        self.assertIs(module_step_from_ode(braintools.quad.ode_rk4_step), rk4_step)
        self.assertIs(get_integrator('rk4'), rk4_step)
        self.assertIs(get_integrator('heun3'), get_integrator('heun'))
        self.assertIs(get_integrator('heun3'), heun3_step)
        self.assertIs(get_integrator('ssprk3'), ssprk3_step)
        self.assertIs(get_integrator('heun2'), heun2_step)

    def test_public_identity(self):
        for name in ('euler_step', 'rk4_step', 'ralston2_step', 'heun2_step'):
            with self.subTest(name=name):
                fun = getattr(braincell.quad, name)
                self.assertEqual(fun.__name__, name)
                self.assertEqual(fun.__module__, 'braincell.quad')

    def test_stage_times_are_set_in_the_environment(self):
        m = _TimeDriven()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        np.testing.assert_allclose(m.x.value.to_decimal(u.mV), [0.005], rtol=1e-6)

    def test_writing_a_non_diffeq_state_raises(self):
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            with self.assertRaisesRegex(ValueError, 'not for integral'):
                rk4_step(_WritesAux())

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            module_step_from_ode(1.0)

    def test_embedded_pair_as_fixed_step_on_hh(self):
        dopri5 = np.asarray(integrate('dopri5', dt=0.01 * u.ms).to_decimal(u.mV))
        rk4 = np.asarray(integrate('rk4', dt=0.01 * u.ms).to_decimal(u.mV))
        self.assertLess(np.max(np.abs(dopri5 - rk4)), 1e-2)

    def test_custom_ode_method_registered_in_braintools(self):
        def my_euler(f, y, t, *args, **kwargs):
            dt = brainstate.environ.get_dt()
            return jax.tree.map(lambda a, b: a + dt * b, y, f(y, t), is_leaf=u.math.is_quantity)

        braintools.quad.register_method('quad2b_test_euler', my_euler, overwrite=True)
        try:
            value, _ = drive(get_integrator('quad2b_test_euler'), dt_ms=0.1, n_steps=10)
            ref, _ = drive(euler_step, dt_ms=0.1, n_steps=10)
            self.assertAlmostEqual(value, ref, places=6)
        finally:
            from braintools.quad import _registry as bt_registry

            bt_registry._REGISTRY['ode'].pop('quad2b_test_euler', None)
            bt_registry._META['ode'].pop('quad2b_test_euler', None)


if __name__ == "__main__":
    unittest.main()
