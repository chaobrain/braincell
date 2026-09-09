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

"""Small single-parameter fits used by synapse_learning.ipynb."""

import braincell
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.filter import AllRegion, RootLocation

DT = 0.025 * u.ms
STEPS = 240


def build_cell():
    """Build one Hodgkin-Huxley CV with one ExpSyn and a short current pulse.

    Returns
    -------
    braincell.Cell
        Uninitialized spiking model.
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
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("IL", name="leak"),
        braincell.mech.Channel("Na_HH1952", name="na"),
        braincell.mech.Channel("K_HH1952", name="k"),
    )
    cell.place(
        RootLocation(0.5), braincell.mech.CurrentClamp(delay=0.25 * u.ms, durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
    )
    cell.place(RootLocation(0.5), braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms))
    return cell


def build_experiment(field, *, train=False):
    """Create a fixed-input fit or an autaptic threshold fit.

    Parameters
    ----------
    field : str
        ``tau``, ``weight``, or ``threshold``.
    train : bool, optional
        Register exactly one scalar parameter at a perturbed initial value.

    Returns
    -------
    tuple
        Prepared Network and Cell.
    """
    cell = build_cell()
    source = (
        cell.event_outputs["spike"]
        if field == "threshold"
        else braincell.NetStim(start=1.0 * u.ms, number=1, interval=10.0 * u.ms)
    )
    connection = braincell.connect(
        "input", source=source, synapse=cell.synapses["syn"], weight=0.001 * u.uS, delay=0.1 * u.ms
    )
    if train:
        if field == "tau":
            cell.synapses["syn"].trainable(tau=braincell.trainable.scale(brainstate.nn.Param(0.8), name="factor"))
        elif field == "weight":
            connection.trainable(weight=braincell.trainable.scale(brainstate.nn.Param(0.8), name="factor"))
        elif field == "threshold":
            source.trainable(
                threshold=braincell.trainable.parameterized(
                    lambda ctx, delta: -20.0 * u.mV + delta * u.mV, delta=brainstate.nn.Param(-10.0)
                )
            )
        else:
            raise ValueError(field)
    net = braincell.Network("fit")
    net.add_population("cell", cell)
    if field != "threshold":
        net.add_population("input", source)
    net.prepare_run(dt=DT, event_backend="scatter")
    return net, cell


def simulate(net, cell):
    """Reset and collect one differentiable, six-millisecond voltage trace.

    Parameters
    ----------
    net : braincell.Network
        Prepared execution target.
    cell : braincell.Cell
        Observed Cell.

    Returns
    -------
    array
        Voltage in mV, one scalar per time step.
    """
    net.reset_state()

    def step(_):
        net.update()
        return cell.V.value.to_decimal(u.mV)[0, 0]

    return brainstate.transform.for_loop(step, jnp.arange(STEPS))


def fit_one(field, *, epochs=100):
    """Fit one parameter to one synthetic spiking voltage trace.

    Parameters
    ----------
    field : str
        Target field.
    epochs : int, optional
        Number of Adam updates.

    Returns
    -------
    dict
        Traces, losses, fitted root, and initial gradient.
    """
    reference, reference_cell = build_experiment(field)
    target = brainstate.transform.jit(lambda: simulate(reference, reference_cell))()
    net, cell = build_experiment(field, train=True)
    states = net.trainables.parameters().states()
    assert len(states) == 1
    predict = brainstate.transform.jit(lambda: simulate(net, cell))
    initial = predict()

    def loss():
        return jnp.mean((simulate(net, cell) - target) ** 2)

    gradient = brainstate.transform.grad(loss, grad_states=states, return_value=True)
    initial_gradient, _ = brainstate.transform.jit(gradient)()
    optimizer = braintools.optim.Adam(lr=0.5 if field == "threshold" else 0.03)
    optimizer.register_trainable_weights(states)

    @brainstate.transform.jit
    def optimize():
        def epoch(_):
            gradients, value = gradient()
            optimizer.update(gradients)
            return value

        return brainstate.transform.for_loop(epoch, jnp.arange(epochs))

    history = optimize()
    fitted = predict()
    initial_mse = float(jnp.mean((initial - target) ** 2))
    final_mse = float(jnp.mean((fitted - target) ** 2))
    spikes = int(np.count_nonzero((np.asarray(target)[:-1] < 0.0) & (np.asarray(target)[1:] >= 0.0)))
    assert spikes >= 1
    assert np.isfinite(np.asarray(history)).all()
    assert final_mse < initial_mse * (1.0 if field == "threshold" else 0.1), (field, initial_mse, final_mse)
    return dict(
        field=field,
        target=np.asarray(target),
        initial=np.asarray(initial),
        fitted=np.asarray(fitted),
        history=np.asarray(history),
        initial_mse=initial_mse,
        final_mse=final_mse,
        spikes=spikes,
        initial_gradient=float(next(iter(initial_gradient.values()))),
        fitted_root=float(next(iter(net.trainables.parameters().physical_values().values()))),
    )


if __name__ == "__main__":
    with brainstate.environ.context(dt=DT):
        for field in ("tau", "weight", "threshold"):
            result = fit_one(field)
            print({k: v for k, v in result.items() if not isinstance(v, np.ndarray)}, flush=True)
