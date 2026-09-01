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

"""Shared fixtures for the :mod:`braincell.quad` test modules.

The leading underscore keeps pytest from collecting this file as a test
module, matching the convention used by ``braincell/io/_testing.py`` and
``braincell/vis/_testing.py``.

Two fixtures live here:

- :class:`LinearDecay` — a scalar linear ODE with a known closed-form
  solution, used wherever a solver has to be checked against an exact
  answer without dragging in the full Hodgkin-Huxley stack.
- :class:`HH` plus :func:`integrate` and :func:`compare` — a one-compartment
  Hodgkin-Huxley model and the convergence harness built on it.
"""

import brainstate
import brainunit as u
import jax.numpy as jnp

import braincell
from braincell.quad.protocol import DiffEqModule, DiffEqSingleState

__all__ = [
    'FLOAT_DTYPE',
    'LinearDecay',
    'drive',
    'HH',
    'integrate',
    'compare',
]

#: The default floating-point dtype, used to build fixture states that match
#: whatever precision the test session is running at.
FLOAT_DTYPE = jnp.asarray(0.0).dtype


class LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Scalar linear ODE ``dx/dt = -x/tau``.

    The exact solution is ``x(t) = x0 * exp(-t / tau)``, so any solver can be
    checked against a closed form. The module also carries one non-
    :class:`DiffEqState` (``aux``) that integrators must leave alone, and
    counts its ``pre_integral`` / ``post_integral`` calls.

    Parameters
    ----------
    x0 : float, optional
        Initial value of every element of the state, in mV. Default 1.0.
    tau_ms : float, optional
        Decay time constant in milliseconds. Default 10.0.
    shape : tuple of int, optional
        Shape of the state array. Default ``(3,)``.
    """

    def __init__(self, x0: float = 1.0, tau_ms: float = 10.0, shape=(3,)):
        super().__init__()
        self.tau = tau_ms * u.ms
        self.x = DiffEqSingleState(jnp.full(shape, x0, dtype=FLOAT_DTYPE) * u.mV)
        # A non-DiffEqState should be ignored by the integrator.
        self.aux = brainstate.ShortTermState(jnp.zeros(shape, dtype=FLOAT_DTYPE))
        self.pre_calls = 0
        self.post_calls = 0

    def pre_integral(self, *args, **kwargs):
        self.pre_calls += 1

    def post_integral(self, *args, **kwargs):
        self.post_calls += 1

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


def drive(method, dt_ms: float = 0.1, n_steps: int = 100, x0: float = 1.0, tau_ms: float = 10.0):
    """Drive ``method`` ``n_steps`` times on a fresh :class:`LinearDecay`.

    Parameters
    ----------
    method : Callable
        An integrator step, called as ``method(module)``.
    dt_ms : float, optional
        Time step in milliseconds. Default 0.1.
    n_steps : int, optional
        Number of steps to take. Default 100.
    x0 : float, optional
        Initial state value in mV. Default 1.0.
    tau_ms : float, optional
        Decay time constant in milliseconds. Default 10.0.

    Returns
    -------
    value : float
        The first element of the final state, in mV.
    module : LinearDecay
        The driven module, so callers can inspect ``pre_calls`` / ``post_calls``.
    """
    m = LinearDecay(x0=x0, tau_ms=tau_ms)
    dt = dt_ms * u.ms
    with brainstate.environ.context(dt=dt):
        for i in range(n_steps):
            with brainstate.environ.context(t=i * dt):
                method(m)
    return float(m.x.value.to_decimal(u.mV)[0]), m


class HH(braincell.SingleCompartment):
    """One-compartment Hodgkin-Huxley neuron used as a solver reference model."""

    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50.0 * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77.0 * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm**2))


def integrate(method: str, dt=0.01 * u.ms):
    """Run :class:`HH` for 10 ms under ``method`` and return the voltage trace.

    Parameters
    ----------
    method : str
        Registered integrator name passed as ``solver=``.
    dt : u.Quantity[u.second], optional
        Integration time step. Default ``0.01 * u.ms``.

    Returns
    -------
    u.Quantity
        The membrane-voltage trace over the 10 ms window.
    """
    brainstate.random.seed(1)
    hh = HH(1, solver=method)
    hh.init_state()

    def step_fun(t):
        with brainstate.environ.context(t=t):
            hh.update(10 * u.nA / u.cm**2)
        return hh.V.value

    with brainstate.environ.context(dt=dt):
        times = u.math.arange(0.0 * u.ms, 10 * u.ms, brainstate.environ.get_dt())
        vs = brainstate.transform.for_loop(step_fun, times)
    return vs


def compare(method: str):
    """Measure ``method``'s deviation from ``exp_euler`` across six time steps.

    Parameters
    ----------
    method : str
        Registered integrator name to compare against the ``exp_euler``
        reference.

    Returns
    -------
    dts : jax.Array
        The six time steps, as bare mantissas in ms.
    norms : jax.Array
        The corresponding voltage-trace error norms, as bare mantissas.

    Notes
    -----
    Units are stripped from both return values so that matplotlib can convert
    them via ``np.asarray``; newer ``saiunit`` rejects
    ``np.asarray(dimensional_quantity)``.
    """
    norm = []
    dts = [1e-3 * u.ms, 2e-3 * u.ms, 4e-3 * u.ms, 8e-3 * u.ms, 1e-2 * u.ms, 2e-2 * u.ms]
    for dt in dts:
        gold_vs = integrate('exp_euler', dt=dt)
        vs = integrate(method, dt=dt)
        norm.append(u.linalg.norm(gold_vs - vs))
    return u.math.asarray(dts).mantissa, u.math.asarray(norm).mantissa
