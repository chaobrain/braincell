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

"""Full-state gradient and fitting probes for two bidirectionally coupled populations."""

from dataclasses import dataclass

import braincell
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.filter import AllRegion, RootLocation
from braincell.network.event import VoltageCrossingSource
from examples.experimental.optim.gradients import build_rollout_value_and_grad

DT = 0.025 * u.ms
STEPS = 800
RTOL = 1e-7
ATOL = 1e-8


class _ProbeSource(VoltageCrossingSource):
    def __init__(self, cell, *, detached):
        super().__init__(cell, name="probe")
        self.detached = detached

    def current_event_count(self, source_index):
        event = super().current_event_count(source_index)
        return jax.lax.stop_gradient(event) if self.detached else event


def _shift(base, group, name):
    return braincell.trainable.parameterized(
        lambda ctx, delta: base + delta * u.mV,
        delta=braincell.trainable.parameter(0.0, group_by=group, name=name),
    )


def _scale(group, name, shared=None):
    if shared is not None:
        return braincell.trainable.scale(shared, group_by="all", name=name)
    return braincell.trainable.scale(group_by=group, name=name, transform=brainstate.nn.TanhT(0.5, 1.5))


def _population(size, *, label, grouped, shared, detached):
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"), pop_size=(size,), V_init=-65.0 * u.mV, V_th=-20.0 * u.mV
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
    for member in range(size):
        cell[member].channels["na"].set(g_max=(108.0 + 4.0 * member + (3.0 if label == "B" else 0.0)) * u.mS / u.cm**2)
    offset = 0.25 if label == "A" else 2.25
    for pulse in (0.0, 10.0):
        cell.place(
            RootLocation(0.5),
            braincell.mech.CurrentClamp(
                delay=(offset + pulse) * u.ms, durations=1.0 * u.ms, amplitudes=(0.1 if pulse == 0.0 else 0.4) * u.nA
            ),
        )
    kind = "ExpSyn" if label == "A" else "Exp2Syn"
    kinetics = {"tau": 2.0 * u.ms} if label == "A" else {"tau1": 0.2 * u.ms, "tau2": 3.0 * u.ms}
    reversal = 0.0 * u.mV if label == "A" else -5.0 * u.mV
    cell.place(RootLocation(0.5), braincell.mech.Synapse(kind, name="syn", e=reversal, **kinetics))
    group = "all" if grouped else "population"
    cell.channels["na"].trainable(g_max=_scale(group, "gmax", shared), V_sh=_shift(-45.0 * u.mV, group, "shift"))
    cell.ions["sodium"].trainable(E=_scale(group, "ion"))
    cell.synapses["syn"].trainable(
        **{key: _scale(group, key) for key in kinetics}, e=_shift(reversal, group, "reversal")
    )
    source = _ProbeSource(cell, detached=detached)
    source.trainable(threshold=_shift(-20.0 * u.mV, group, "threshold"))
    return cell, source


@dataclass
class Experiment:
    """Hold the network and explicit observation/parameter ownership maps."""

    network: object
    cells: dict
    connections: dict
    nodes: dict

    def observation(self):
        """Read post-step voltages, spikes and receiving synapse conductances."""
        return {
            "voltage": jnp.concatenate([self.cells[name].V.value.to_decimal(u.mV)[:, 0] for name in ("A", "B")]),
            "spike": jnp.concatenate([self.cells[name].spike.value[:, 0] for name in ("A", "B")]),
            "conductance": jnp.concatenate(
                [u.get_mantissa(brainstate.maybe_state(self.nodes[name].g)) for name in ("A", "B")]
            ),
        }

    def rollout(self, steps=STEPS):
        """Reset and collect a compiled-loop trajectory without updating roots."""
        self.network.reset_state()

        def step(_):
            self.network.update()
            return self.observation()

        return brainstate.transform.for_loop(step, jnp.arange(steps))


def build(*, grouped=False, delay="heterogeneous", backend="scatter", shared=False, detached=False):
    """Build and prepare a two-by-three bidirectional population experiment.

    Parameters
    ----------
    grouped : bool, optional
        Share each field inside its population and each direction's weights.
    delay : str, optional
        ``zero``, ``positive``, or ``heterogeneous`` fixed delays.
    backend : str, optional
        ``scatter`` or ``brainevent``.
    shared : bool, optional
        Tie the two populations' conductance factors to one original root.
    detached : bool, optional
        Stop only event derivatives, leaving forward events unchanged.

    Returns
    -------
    Experiment
        Initialized network and its explicit ownership maps.
    """
    shared_root = brainstate.nn.Param(1.0, t=brainstate.nn.TanhT(0.5, 1.5)) if shared else None
    cells, sources = {}, {}
    net = braincell.Network("bidirectional")
    for label, size in (("A", 2), ("B", 3)):
        cells[label], sources[label] = _population(
            size, label=label, grouped=grouped, shared=shared_root, detached=detached
        )
        net.add_population(label, cells[label])
    delay_values = {
        "zero": np.zeros(6),
        "positive": np.full(6, 0.1),
        "heterogeneous": np.array([0.0, 0.025, 0.1, 0.05, 0.1, 0.025]),
    }
    connections = {}
    for pre, post in (("A", "B"), ("B", "A")):
        n_pre, n_post = len(sources[pre]), len(sources[post])
        pre_ids = np.repeat(np.arange(n_pre), n_post)
        post_ids = np.tile(np.arange(n_post), n_pre)
        name = f"{pre}_to_{post}"
        connection = net.connect(
            name,
            source=sources[pre][pre_ids],
            synapse=cells[post].synapses["syn"][post_ids],
            weight=np.linspace(0.0003, 0.0005, 6) * u.uS,
            delay=delay_values[delay] * u.ms,
        )
        connection.trainable(weight=_scale("all" if grouped else "row", "weight"))
        connections[name] = connection
    net.prepare_run(dt=DT, event_backend=backend)
    nodes = {
        label: cell.runtime.get_runtime_node(
            cell.synapses["syn"]._store.layout_id("ExpSyn" if label == "A" else "Exp2Syn")
        )
        for label, cell in cells.items()
    }
    return Experiment(net, cells, connections, nodes)


def engine_for(experiment, method, *, loss_kind="joint"):
    """Create the existing gradient engine for an additive voltage/spike loss.

    Parameters
    ----------
    experiment : Experiment
        Prepared model.
    method : str
        ``bptt`` or ``rtrl``.
    loss_kind : str, optional
        ``A``, ``B``, ``joint``, or ``spike``. Inputs are per-step targets.

    Returns
    -------
    RolloutGradientEngine
        Untraced gradient engine.
    """
    selection = {"A": slice(0, 2), "B": slice(2, 5), "joint": slice(None), "spike": slice(None)}[loss_kind]

    def step(target):
        experiment.network.update()
        key = "spike" if loss_kind == "spike" else "voltage"
        observed = experiment.observation()[key]
        return jnp.mean((observed[selection] - target[selection]) ** 2)

    return build_rollout_value_and_grad(experiment.network, step=step, method=method)


def gradient_comparison(*, loss_kind="joint", steps=STEPS, **configuration):
    """Compare every root coordinate using full RTRL and BPTT.

    Parameters
    ----------
    loss_kind : str, optional
        Loss selection accepted by engine_for.
    steps : int, optional
        Rollout length.
    **configuration
        Keyword arguments forwarded to build.

    Returns
    -------
    dict
        Per-root gradients, maximum absolute error, loss and carry shapes.
    """
    values = {}
    target = jnp.full((steps, 5), 1.0 if loss_kind == "spike" else -60.0)
    for method in ("bptt", "rtrl"):
        experiment = build(**configuration)
        engine = engine_for(experiment, method, loss_kind=loss_kind)
        engine.prepare(target[0])
        if method == "rtrl":
            roots = tuple(state.value for state in engine.parameter_states.values())
            state, sensitivity = engine._initial_full_carry(roots)
            shapes = (
                tuple(np.shape(x) for x in jax.tree.leaves(state)),
                tuple(np.shape(x) for x in jax.tree.leaves(sensitivity)),
            )
        values[method] = brainstate.transform.jit(engine)(target)
    left, right = values["bptt"], values["rtrl"]
    np.testing.assert_allclose(left.losses, right.losses, rtol=1e-10, atol=1e-10)
    errors = {}
    gradients = {}
    for name, value in left.gradients.items():
        a, b = np.asarray(value), np.asarray(right.gradients[name])
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)
        errors[name] = float(np.max(np.abs(a - b)))
        gradients[name] = a
    return dict(
        loss=float(left.loss),
        gradients=gradients,
        errors=errors,
        max_abs_error=max(errors.values()),
        carry_shapes=shapes,
    )


def fit(*, method="bptt", epochs=200):
    """Fit both populations and both connections to one spiking trajectory.

    Parameters
    ----------
    method : str, optional
        BPTT or full-state RTRL.
    epochs : int, optional
        Maximum number of Adam updates.

    Returns
    -------
    dict
        Loss history, trajectories and original/fitted physical root values.
    """
    reference = build(grouped=True)
    target = brainstate.transform.jit(reference.rollout)()["voltage"]
    experiment = build(grouped=True)
    parameters = experiment.network.trainables.parameters()
    initial_roots = parameters.physical_values()
    initial_roots = {
        name: value
        + (0.25 if name.endswith(("shift", "reversal", "threshold")) else -0.02 if name.startswith("A.") else 0.02)
        for name, value in initial_roots.items()
    }
    parameters.set_physical_values(initial_roots)
    predict = brainstate.transform.jit(experiment.rollout)
    initial = predict()["voltage"]
    engine = engine_for(experiment, method)
    engine.prepare(target[0])
    optimizer = braintools.optim.Adam(lr=0.01)
    optimizer.register_trainable_weights(parameters.states())

    @brainstate.transform.jit
    def optimize():
        def epoch(_):
            result = engine(target)
            optimizer.update(jax.tree.map(lambda value: value / STEPS, result.gradients))
            return result.loss / STEPS

        return brainstate.transform.for_loop(epoch, jnp.arange(epochs))

    history = np.asarray(optimize())
    fitted = predict()["voltage"]
    initial_mse = float(jnp.mean((initial - target) ** 2))
    final_mse = float(jnp.mean((fitted - target) ** 2))
    return dict(
        method=method,
        initial_mse=initial_mse,
        final_mse=final_mse,
        history=history,
        target=np.asarray(target),
        initial=np.asarray(initial),
        fitted=np.asarray(fitted),
        initial_roots=initial_roots,
        fitted_roots=parameters.physical_values(),
    )


if __name__ == "__main__":
    with brainstate.environ.context(dt=DT, precision=64):
        experiment = build()
        trajectory = brainstate.transform.jit(experiment.rollout)()
        print("spikes", np.asarray(trajectory["spike"]).sum(axis=0), flush=True)
        result = gradient_comparison()
        print("gradient error", result["max_abs_error"], flush=True)
        print("gradients", result["gradients"], flush=True)
