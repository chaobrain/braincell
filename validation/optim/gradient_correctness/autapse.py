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

"""Full-state BPTT/RTRL checks for a one-CV Hodgkin-Huxley autapse."""

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.filter import AllRegion, RootLocation
from braincell.experimental.optim.gradients import build_rollout_value_and_grad

DT = 0.025 * u.ms


def build_autapse(*, delay=0.1 * u.ms, network=False, paired=False):
    """Build an initialized spiking Cell with a trainable self-connection.

    Parameters
    ----------
    delay : brainunit.Quantity, optional
        Static self-connection delay.
    network : bool, optional
        Wrap the same Cell in the public Network execution path.
    paired : bool, optional
        Add a second postsynaptic Cell and observe it instead.

    Returns
    -------
    tuple
        Execution target and underlying Cell.
    """
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"), pop_size=(1,), V_init=-65.0 * u.mV, V_th=-20.0 * u.mV
    )
    cell.paint(
        AllRegion(),
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        ),
        braincell.mech.Ion("SodiumFixed", name="sodium", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("IL", name="leak"),
        braincell.mech.Channel("Na_HH1952", name="na"),
        braincell.mech.Channel("K_HH1952", name="k"),
    )
    cell.place(
        RootLocation(0.5), braincell.mech.CurrentClamp(delay=0.25 * u.ms, durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
    )
    cell.place(RootLocation(0.5), braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms))
    connection = braincell.connect(
        "self", source=cell.event_outputs["spike"], synapse=cell.synapses["syn"], weight=0.001 * u.uS, delay=delay
    )
    cell.synapses["syn"].trainable(tau=braincell.trainable.scale(name="tau"))
    connection.trainable(weight=braincell.trainable.scale(name="weight"))
    cell.event_outputs["spike"].trainable(threshold=braincell.trainable.parameter(group_by="all", name="threshold"))
    cell.channels["na"].trainable(g_max=braincell.trainable.scale(name="na"))
    cell.ions["sodium"].trainable(E=braincell.trainable.scale(name="sodium"))
    if network:
        target = braincell.Network("autapse")
        target.add_population("cell", cell)
        if paired:
            from examples.optim.parameter_learning.synapse_learning import build_cell

            post = build_cell()
            braincell.connect(
                "forward",
                source=cell.event_outputs["spike"],
                synapse=post.synapses["syn"],
                weight=0.001 * u.uS,
                delay=delay,
            )
            target.add_population("post", post)
        target.prepare_run(dt=DT, event_backend="scatter")
        if paired:
            cell = post
    else:
        cell.init_state()
        cell.connections.prepare_runtime(DT)
        target = cell
    return target, cell


def compare(*, loss_kind="voltage", delay=0.1 * u.ms, network=False, paired=False, steps=160):
    """Compare full-state derivatives using identical surrogate rules.

    Parameters
    ----------
    loss_kind : str, optional
        ``voltage`` or ``spike`` additive loss.
    delay : brainunit.Quantity, optional
        Fixed event delay.
    network : bool, optional
        Exercise Network.update instead of standalone Cell.run.
    paired : bool, optional
        Compare gradients from a second Cell back to the presynaptic Cell.
    steps : int, optional
        Number of time steps.

    Returns
    -------
    dict
        Loss, parameter gradients, and maximum absolute gradient difference.
    """
    results = {}
    for method in ("bptt", "rtrl"):
        target, cell = build_autapse(delay=delay, network=network, paired=paired)

        def step(_):
            if network:
                target.update()
            else:
                cell.run(dt=DT, duration=DT)
            value = cell.V.value.to_decimal(u.mV) + 60.0 if loss_kind == "voltage" else cell.spike.value - 1.0
            return jnp.mean(value**2)

        engine = build_rollout_value_and_grad(target, step=step, method=method)
        inputs = jnp.arange(steps)
        engine.prepare(inputs[0])
        if method == "rtrl":
            import jax

            roots = tuple(state.value for state in engine.parameter_states.values())
            values, tangents = engine._initial_full_carry(roots)
            state_shapes = tuple(np.shape(leaf) for leaf in jax.tree.leaves(values))
            sensitivity_shapes = tuple(np.shape(leaf) for leaf in jax.tree.leaves(tangents))
        results[method] = brainstate.transform.jit(engine)(inputs)
    lhs, rhs = results["bptt"], results["rtrl"]
    np.testing.assert_allclose(lhs.losses, rhs.losses, rtol=1e-10, atol=1e-10)
    errors = []
    for name in lhs.gradients:
        a, b = u.get_mantissa(lhs.gradients[name]), u.get_mantissa(rhs.gradients[name])
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
        errors.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    return {
        "loss": float(lhs.loss),
        "max_abs_error": max(errors),
        "state_shapes": state_shapes,
        "sensitivity_shapes": sensitivity_shapes,
        "gradients": {k: float(u.get_mantissa(v)) for k, v in lhs.gradients.items()},
    }


if __name__ == "__main__":
    with brainstate.environ.context(dt=DT, precision=64):
        for mode in ("voltage", "spike"):
            print(mode, compare(loss_kind=mode))
