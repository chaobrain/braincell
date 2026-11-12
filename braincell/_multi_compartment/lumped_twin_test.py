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

"""CELL-24: a lumped Cell against its one-CV cable twin.

Views, trainables, recordings, synapses and ``Network`` populations of
``Cell.point`` give what the same declarations give on
``Cell(Morphology.from_soma(...), cv_policy=CVPerBranch())``: the one CV of
the cable twin has the lumped Cell's area and capacitance, so the two
integrate the same equations (float64, ``rk4``).
"""

import contextlib
import gc
import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell as bc
from braincell.experimental.optim import build_rollout_value_and_grad
from braincell.filter import AllRegion, RootLocation

DT = 0.025 * u.ms
GEOMETRY = dict(length=20.0 * u.um, radius=10.0 * u.um)
AMPLITUDES = np.asarray([0.10, 0.12, 0.14]) * u.nA
KINDS = ("lumped", "cable")


@contextlib.contextmanager
def _float64():
    # JAX 0.12 reuses a NumPy constant that a live program of the other
    # precision captured; free earlier programs on both sides of the switch.
    gc.collect()
    jax.clear_caches()
    try:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            yield
    finally:
        gc.collect()
        jax.clear_caches()


def _new_cell(kind, n, *, solver="rk4", V_init=-65.0 * u.mV, **geometry):
    geometry = {**GEOMETRY, **geometry}
    if kind == "lumped":
        return bc.Cell.point(n, V_init=V_init, solver=solver, **geometry)
    return bc.Cell(
        bc.Morphology.from_soma(**geometry), pop_size=n, cv_policy=bc.CVPerBranch(), V_init=V_init, solver=solver
    )


def _hh(kind, n=3, *, synapse=False, clamp=None, solver="rk4", **geometry):
    """A Hodgkin-Huxley soma: lumped, or its one-CV cable twin."""
    cell = _new_cell(kind, n, solver=solver, **geometry)
    cell.paint(
        bc.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        bc.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        bc.mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        bc.mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        bc.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    if clamp is not None:
        cell.place(RootLocation(0.5), clamp)
    if synapse:
        cell.place(RootLocation(0.5), bc.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
    return cell


def _clamp(delay=1.0, duration=8.0, amplitudes=AMPLITUDES):
    return bc.mech.CurrentClamp(delay=delay * u.ms, durations=duration * u.ms, amplitudes=amplitudes)


def _pair(lumped, cable):
    """Mantissas, the cable value without its one-CV axis where the lumped value has none."""
    lumped, cable = np.asarray(u.get_mantissa(lumped)), np.asarray(u.get_mantissa(cable))
    if cable.ndim == lumped.ndim + 1 and cable.shape[-1] == 1:
        cable = cable[..., 0]
    return lumped, cable


def _mv(samples):
    return np.asarray(samples.values.to_decimal(u.mV))


def _mean_voltage_loss(cell, duration=5.0 * u.ms, *, reduce="mean"):
    times = u.math.arange(0.0 * u.ms, duration, DT)

    def loss():
        cell.reset_state()

        def step(t):
            with brainstate.environ.context(t=t):
                spike = cell.update()
            return cell.V.value.to_decimal(u.mV), spike

        with brainstate.environ.context(dt=DT):
            voltages, spikes = brainstate.transform.for_loop(step, times)
        return u.math.mean(voltages) if reduce == "mean" else u.math.sum(spikes)

    return loss


def _gradient(cell, loss):
    states = cell.trainables.parameters().states()
    grads = brainstate.transform.grad(loss, grad_states=states)()
    (key,) = grads.keys()
    return np.asarray(u.get_mantissa(grads[key])).reshape(-1), states[key]


class LumpedTwinTestCase(unittest.TestCase):
    def assertTwin(self, lumped, cable, bound, msg=None):
        lumped, cable = _pair(lumped, cable)
        self.assertEqual(lumped.shape, cable.shape, msg)
        self.assertLessEqual(float(np.max(np.abs(lumped - cable), initial=0.0)), bound, msg)


class LumpedViewTwinTest(LumpedTwinTestCase):
    """Density, geometry and Cell views read and write the one CV of the twin."""

    def test_channel_and_ion_writes(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.record("v", bc.observe.state("V"))
                cell.init_state()
                before = cell.channels["na"].g_max
                cell.channels["na"].set(g_max=np.asarray([100.0, 110.0, 120.0]) * u.mS / u.cm**2)
                cell.ions["SodiumFixed"].set(E=np.asarray([45.0, 50.0, 55.0]) * u.mV)
                cell.channels["leak"].set(E=-60.0 * u.mV)
                reads = (before, cell.channels["na"].g_max, cell.ions["SodiumFixed"].E, cell.channels["leak"].E)
                out[kind] = reads, _mv(cell.run(dt=DT, duration=10.0 * u.ms).samples["v"])
            for index, (lumped, cable) in enumerate(zip(out["lumped"][0], out["cable"][0])):
                self.assertTwin(lumped, cable, 0.0, msg=f"read {index}")
            np.testing.assert_array_equal(_pair(out["lumped"][0][1], 0)[0], np.asarray([100.0, 110.0, 120.0]))
            self.assertTwin(out["lumped"][1], out["cable"][1], 1e-9)

    def test_ion_initializer_write(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _new_cell(kind, 2)
                cell.paint(
                    AllRegion(),
                    bc.mech.Ion("CalciumDetailed", name="ca"),
                    bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
                    bc.mech.Channel("CaHVA_MA2020_GoC", g_max=0.5 * u.mS / u.cm**2, ion_name="ca"),
                )
                cell.init_state()
                before = cell.ions["ca"].Ci_initializer
                cell.ions["ca"].set(Ci_initializer=np.asarray([2e-4, 3e-4]) * u.mM)
                cell.reset_state()
                ci = cell.ions["ca"].Ci
                cell.run(dt=DT, duration=1.0 * u.ms)
                out[kind] = before, cell.ions["ca"].Ci_initializer, ci, cell.V.value.to_decimal(u.mV)
            for index, (lumped, cable) in enumerate(zip(out["lumped"], out["cable"])):
                self.assertTwin(lumped, cable, 1e-9, msg=f"value {index}")
            np.testing.assert_allclose(_pair(out["lumped"][2], 0)[0], [2e-4, 3e-4], rtol=1e-12)

    def test_nernst_ion_write_updates_reversal(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _new_cell(kind, 2, V_init=-70.0 * u.mV, length=20.0 * u.um, radius=0.5 * u.um)
                cell.paint(bc.mech.Ion("ChlorideInitNernst", name="cl", temp=u.celsius2kelvin(35.0)))
                cell.init_state()
                before = cell.ions["cl"].E
                cell.ions["cl"].set(Co=np.asarray([100.0, 130.0]) * u.mM)
                out[kind] = before, cell.ions["cl"].E
            for index, (lumped, cable) in enumerate(zip(out["lumped"], out["cable"])):
                self.assertTwin(lumped, cable, 1e-12, msg=f"E {index}")
            self.assertFalse(np.allclose(_pair(out["lumped"][0], 0)[0], _pair(out["lumped"][1], 0)[0]))

    def test_geometry_reads_writes_and_length(self) -> None:
        fields = ("cm", "length", "Ra", "radius_scale", "area", "diam_arc_mean", "radius_mid")
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.record("v", bc.observe.state("V"))
                cell.init_state()
                reads = [cell.geometry.get(field) for field in fields]
                cell.geometry.set(cm=np.asarray([0.8, 1.0, 1.2]) * u.uF / u.cm**2)
                out[kind] = reads, _mv(cell.run(dt=DT, duration=10.0 * u.ms).samples["v"])
            for field, lumped, cable in zip(fields, out["lumped"][0], out["cable"][0]):
                self.assertTwin(lumped, cable, 0.0, msg=field)
            self.assertTwin(out["lumped"][1], out["cable"][1], 1e-9)

            traces = {}
            for kind, mode in (("lumped", "set"), ("lumped", "declared"), ("cable", "set"), ("cable", "declared")):
                length = (18.0 if mode == "declared" else 20.0) * u.um
                cell = _hh(kind, 1, clamp=_clamp(amplitudes=0.1 * u.nA), length=length)
                cell.record("v", bc.observe.state("V"))
                cell.init_state()
                if mode == "set":
                    cell.geometry.set(length=18.0 * u.um)
                traces[kind, mode] = _mv(cell.run(dt=DT, duration=10.0 * u.ms).samples["v"]).reshape(-1)
            self.assertTwin(traces["lumped", "set"], traces["lumped", "declared"], 1e-9)
            self.assertTwin(traces["lumped", "set"], traces["cable", "declared"], 1e-9)
            # The one CV of the cable passes no axial current at any length.
            self.assertTwin(traces["cable", "set"], traces["cable", "declared"], 1e-12)

    def test_cell_views(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.record("v", bc.observe.state("V"))
                cell[1].record("v1", bc.observe.state("V"))
                cell.init_state()
                reads = {
                    "cell[1].V": cell[1].V,
                    "cell[1:].V": cell[1:].V,
                    "soma.V": cell.soma.V,
                    "on.V": cell.on(AllRegion()).V,
                    "loc.V": cell.loc(RootLocation(0.5)).V,
                    "cv[0].V": cell.cv[0].V,
                    "branch[0].V": cell.branch[0].V,
                    "cell[2].spike": cell[2].spike,
                    "V_init": cell.V_init,
                    "V_th": cell.V_th,
                    "cell[1:].V_th": cell[1:].V_th,
                    "soma.V_init": cell.soma.V_init,
                    "cell[0] g_max": cell[0].channels["na"].g_max,
                    "soma g_max": cell.soma.channels["k"].g_max,
                }
                cell[0].channels["na"].set(g_max=100.0 * u.mS / u.cm**2)
                cell[2].set(V_init=-60.0 * u.mV)
                cell[0:3].set(V_th=np.asarray([-10.0, 0.0, 10.0]) * u.mV)
                cell.reset_state()
                reads["V after set"] = cell.V.value
                result = cell.run(dt=DT, duration=10.0 * u.ms)
                reads["v"] = _mv(result.samples["v"])
                reads["v1"] = _mv(result.samples["v1"])
                out[kind] = reads
            for key, cable in out["cable"].items():
                self.assertTwin(out["lumped"][key], cable, 1e-9, msg=key)


class LumpedTrainableTwinTest(LumpedTwinTestCase):
    """Trainable parameters and their first-order gradients."""

    def test_channel_scale_gradient(self) -> None:
        with _float64():
            grads = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.init_state()
                cell.channels["na"].trainable(g_max=bc.trainable.scale(name="s", group_by="population"))
                loss = _mean_voltage_loss(cell)
                grads[kind], state = _gradient(cell, loss)
            # Central finite difference on member 0's factor of the lumped Cell.
            base, eps = state.value, 1e-6
            state.value = base.at[0].add(eps)
            plus = float(loss())
            state.value = base.at[0].add(-eps)
            minus = float(loss())
            state.value = base
            np.testing.assert_allclose(grads["lumped"], grads["cable"], rtol=1e-8)
            np.testing.assert_allclose(grads["lumped"][0], (plus - minus) / (2 * eps), rtol=1e-5)

    def test_capacitance_scale_gradient(self) -> None:
        with _float64():
            grads = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.init_state()
                cell.geometry.trainable(cm=bc.trainable.scale(name="cm", group_by="population"))
                grads[kind], _ = _gradient(cell, _mean_voltage_loss(cell))
            np.testing.assert_allclose(grads["lumped"], grads["cable"], rtol=1e-8)
            self.assertTrue(np.all(grads["lumped"] != 0.0))

    def test_spike_threshold_gradient(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp())
                cell.init_state()
                cell.event_outputs["spike"].trainable(
                    threshold=bc.trainable.parameter(group_by="population", name="th")
                )
                loss = _mean_voltage_loss(cell, 6.0 * u.ms, reduce="sum")
                grad, _ = _gradient(cell, loss)
                out[kind] = grad, float(loss())
            self.assertEqual(out["lumped"][1], out["cable"][1])
            self.assertTwin(out["lumped"][0], out["cable"][0], 1e-9)

    def test_rollout_engine_schedule_and_gradients(self) -> None:
        sources = {
            "scale": lambda name: bc.trainable.scale(group_by="population", name=name),
            "parameter": lambda name: bc.trainable.parameter(group_by="population", name=name),
        }
        for source, make in sources.items():
            with self.subTest(source=source), _float64():
                out = {}
                for kind in KINDS:
                    cell = _hh(kind, 2)
                    cell.init_state()
                    for name in ("leak", "k", "na"):
                        cell.channels[name].trainable(g_max=make(name))

                    def step(target, cell=cell):
                        with brainstate.environ.context(dt=DT):
                            cell.update()
                        error = cell.V.value.to_decimal(u.mV) - target
                        return jnp.mean(error * error)

                    engine = build_rollout_value_and_grad(cell, step=step, method="bptt")
                    with brainstate.environ.context(dt=DT, t=0.0 * u.ms):
                        result = engine(jnp.full((40,), -60.0))
                    out[kind] = engine.materialization_mode, float(result.loss), result.gradients
                self.assertEqual(out["lumped"][0], out["cable"][0])
                self.assertEqual(out["lumped"][0], "rollout")
                np.testing.assert_allclose(out["lumped"][1], out["cable"][1], rtol=1e-10)
                for key, cable in out["cable"][2].items():
                    lumped, cable = _pair(out["lumped"][2][key], cable)
                    np.testing.assert_allclose(lumped.reshape(cable.shape), cable, rtol=1e-8, err_msg=key)


#: Every observable a Cell records, on the post population of the Network below.
OBSERVABLES = {
    "v": lambda: bc.observe.state("V"),
    "na_p": lambda: bc.observe.channel(name="na").state("p"),
    "na_I": lambda: bc.observe.channel(name="na").current(),
    "channel_I_sum": lambda: bc.observe.channel().current(reduce="sum"),
    "ion_E": lambda: bc.observe.ion(species="na").state("E"),
    "ion_I": lambda: bc.observe.ion(species="na").current(),
    "membrane_I": lambda: bc.observe.membrane_current(),
    "clamp_I": lambda: bc.observe.clamp_current(),
    "synapse_g": lambda: bc.observe.synapse(name="ampa").state("g"),
    "synapse_I": lambda: bc.observe.synapse(name="ampa").current(),
}


def _flat(values):
    values = np.asarray(u.get_mantissa(values))
    return values.reshape(values.shape[0], -1)


class LumpedNetworkTwinTest(LumpedTwinTestCase):
    """A lumped Cell as a Network source and target."""

    def _network_twins(self, source=None):
        results = {}
        self._post_cells = {}
        for kind in KINDS:
            pre_cell = _hh(kind, clamp=_clamp(delay=2.0, duration=6.0))
            post_cell = _hh(kind, synapse=True, clamp=_clamp(delay=1.0, duration=2.0, amplitudes=0.01 * u.nA))
            pre_cell.record("v", bc.observe.state("V"))
            for name, make in OBSERVABLES.items():
                post_cell.record(name, make())
            network = bc.Network(f"twin_{kind}")
            self._post_cells[kind] = network, post_cell
            pre = network.add_population("pre", pre_cell)
            post = network.add_population("post", post_cell)
            spikes = pre.event_outputs["spike"] if source is None else source(pre_cell)
            network.connect("drive", source=spikes, synapse=post.synapses["ampa"], weight=0.03 * u.uS, delay=0.2 * u.ms)
            results[kind] = network.run(dt=DT, duration=12.0 * u.ms)
        return results

    def assertSameEvents(self, lumped, cable):
        self.assertGreater(len(cable.time), 0)
        np.testing.assert_array_equal(np.asarray(lumped.source_id), np.asarray(cable.source_id))
        np.testing.assert_allclose(lumped.time.to_decimal(u.ms), cable.time.to_decimal(u.ms), rtol=0, atol=1e-12)

    def test_spike_source_and_synapse_target(self) -> None:
        with _float64():
            results = self._network_twins()
            for network, post_cell in self._post_cells.values():
                # reset_state clears the delivered events and the synapse state.
                self.assertGreater(float(np.max(np.asarray(u.get_mantissa(post_cell.synapses["ampa"].g)))), 0.0)
                network.reset_state()
                self.assertFalse(np.any(np.asarray(u.get_mantissa(post_cell.synapses["ampa"].g))))
        lumped, cable = results["lumped"], results["cable"]
        for population in ("pre", "post"):
            self.assertSameEvents(lumped.events[population]["spike"], cable.events[population]["spike"])
        self.assertTwin(_mv(lumped.samples["pre"]["v"]), _mv(cable.samples["pre"]["v"]), 1e-9)
        for name in OBSERVABLES:
            with self.subTest(observable=name):
                lumped_values = lumped.samples["post"][name].values
                cable_values = cable.samples["post"][name].values
                unit = u.get_unit(cable_values)
                lumped_values = _flat(u.Quantity(lumped_values).to_decimal(unit))
                cable_values = _flat(u.Quantity(cable_values).to_decimal(unit))
                self.assertEqual(lumped_values.shape, cable_values.shape)
                scale = max(1.0, float(np.max(np.abs(cable_values))))
                self.assertLessEqual(float(np.max(np.abs(lumped_values - cable_values))) / scale, 1e-10)
        self.assertGreater(float(np.max(_flat(lumped.samples["post"]["synapse_g"].values.to_decimal(u.uS)))), 0.0)

    def test_voltage_crossing_source(self) -> None:
        def crossing(cell):
            return bc.VoltageCrossingSource(cell, threshold=-20.0 * u.mV, name="crossing")

        with _float64():
            results = self._network_twins(crossing)
        lumped, cable = results["lumped"], results["cable"]
        self.assertSameEvents(lumped.events["pre"]["crossing"], cable.events["pre"]["crossing"])
        self.assertSameEvents(lumped.events["post"]["spike"], cable.events["post"]["spike"])
        self.assertTwin(_mv(lumped.samples["post"]["v"]), _mv(cable.samples["post"]["v"]), 1e-9)

    def test_lumped_and_cable_populations_mix(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                network = bc.Network(f"mixed_{kind}")
                pre = network.add_population("pre", _hh(kind, clamp=_clamp(delay=2.0, duration=6.0)))
                post_cell = _hh("cable" if kind == "lumped" else "lumped", synapse=True)
                post_cell.record("v", bc.observe.state("V"))
                post = network.add_population("post", post_cell)
                network.connect(
                    "drive", source=pre.event_outputs["spike"], synapse=post.synapses["ampa"], weight=0.03 * u.uS
                )
                out[kind] = network.run(dt=DT, duration=12.0 * u.ms)
        # lumped -> cable against cable -> lumped: each side is the other's twin.
        self.assertSameEvents(out["lumped"].events["pre"]["spike"], out["cable"].events["pre"]["spike"])
        self.assertTwin(_mv(out["cable"].samples["post"]["v"]), _mv(out["lumped"].samples["post"]["v"]), 1e-9)

    def test_population_view_places_synapses(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                cell = _hh(kind, clamp=_clamp(amplitudes=0.05 * u.nA))
                synapse = bc.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV)
                if kind == "lumped":
                    cell[1:].place(synapse)
                else:
                    cell[1:].place(RootLocation(0.5), synapse)
                cell.record("v", bc.observe.state("V"))
                cell.record("g", bc.observe.synapse(name="ampa").state("g"))
                events = bc.EventTable(source_index=np.asarray([0, 1]), time=np.asarray([1.0, 2.0]) * u.ms)
                network = bc.Network(f"subset_{kind}")
                source = network.add_population("src", bc.EventSequence(size=2, events=events))
                post = network.add_population("post", cell)
                network.connect(
                    "drive", source=source.event_outputs["event"], synapse=post.synapses["ampa"], weight=0.01 * u.uS
                )
                samples = network.run(dt=DT, duration=5.0 * u.ms).samples["post"]
                out[kind] = (
                    len(cell.synapses["ampa"]),
                    _mv(samples["v"]),
                    np.asarray(samples["g"].values.to_decimal(u.uS)),
                )
        self.assertEqual(out["lumped"][0], 2)
        self.assertEqual(out["cable"][0], 2)
        self.assertTwin(out["lumped"][1], out["cable"][1], 1e-9)
        self.assertTwin(out["lumped"][2], out["cable"][2], 1e-12)
        self.assertGreater(float(np.max(out["lumped"][2])), 0.0)

    def test_reduction_target(self) -> None:
        with _float64():
            out = {}
            for kind in KINDS:
                events = bc.EventTable(
                    source_index=np.asarray([0, 1, 2, 0]), time=np.asarray([1.0, 2.0, 2.5, 3.0]) * u.ms
                )
                cell = _hh(kind, synapse=True)
                cell.add_reduction("acc", bc.EventAccumulatorReduction(alpha=0.5, threshold=0.05))
                cell.use_model("acc")
                cell.record("out", bc.observe.output("value"))
                network = bc.Network(f"reduced_{kind}")
                source = network.add_population("src", bc.EventSequence(size=3, events=events))
                target = network.add_population("tgt", cell)
                network.connect(
                    "drive", source=source.event_outputs["event"], synapse=target.synapses["ampa"], weight=0.03 * u.uS
                )
                out[kind] = np.asarray(
                    u.get_mantissa(network.run(dt=DT, duration=4.0 * u.ms).samples["tgt"]["out"].values)
                )
        np.testing.assert_array_equal(out["lumped"], out["cable"])
        self.assertGreater(float(np.max(out["lumped"])), 0.0)


def _gaba_cell(kind, ion, solver):
    cell = _new_cell(kind, 2, solver=solver, V_init=-70.0 * u.mV, length=20.0 * u.um, radius=0.5 * u.um)
    cell.paint(bc.mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
    if ion == "ChlorideInitNernst":
        cell.paint(bc.mech.Ion(ion, name="cl", temp=u.celsius2kelvin(35.0)))
    else:
        cell.paint(bc.mech.Ion(ion, name="cl"))
    cell.place(RootLocation(0.5), bc.mech.Synapse("GabaA_JE2011", name="s"))
    cell.record("v", bc.observe.state("V"))
    cell.record("i", bc.observe.synapse(name="s").current())
    cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
    network = bc.Network("gaba")
    stim = network.add_population("stim", bc.NetStim(size=1, start=1.0 * u.ms, number=100, interval=5.0 * u.ms))
    post = network.add_population("post", cell)
    network.connect("s", source=stim, synapse=post.synapses["s"], weight=np.asarray([0.0005, 0.001]) * u.uS)
    samples = network.run(dt=DT, duration=40.0 * u.ms).samples["post"]
    units = {"v": u.mV, "i": u.nA, "ci": u.mM}
    return {key: np.asarray(value.values.to_decimal(units[key])) for key, value in samples.items()}


class LumpedSynapseTwinTest(LumpedTwinTestCase):
    """A synapse that reads and writes chloride, and the solver rule."""

    def test_chloride_synapse(self) -> None:
        for ion in ("ChlorideInitNernst", "Cldif_JE2011"):
            for solver in ("rk4", "ralston2"):
                with self.subTest(ion=ion, solver=solver), _float64():
                    out = {kind: _gaba_cell(kind, ion, solver) for kind in KINDS}
                    lumped, cable = out["lumped"], out["cable"]
                    if ion == "ChlorideInitNernst":
                        for key in ("v", "i", "ci"):
                            self.assertTwin(lumped[key], cable[key], 1e-10, msg=key)
                        continue
                    # A kinetic ion advances in ``ind_update``: before the solver on
                    # a lumped Cell (the point-neuron schedule), after it on the
                    # cable Cell, so the recorded Ci leads by one step.
                    cable_ci = cable["ci"].reshape(lumped["ci"].shape)
                    np.testing.assert_array_equal(lumped["ci"][1:], cable_ci[:-1])
                    self.assertGreater(float(np.max(cable_ci - cable_ci[0])), 1e-3)
                    for key in ("v", "i"):
                        change = float(np.max(np.abs(cable[key] - cable[key][0])))
                        self.assertTwin(lumped[key], cable[key], 1e-2 * change, msg=key)

    def test_split_run_delivers_a_boundary_event_once(self) -> None:
        # CELL-34 on a lumped Cell: a run split at the event time matches one
        # run and the twin, and the first run (after init_state or
        # reset_state) delivers an event at t = 0.
        def drive(cell, start):
            bc.connect(
                "drive",
                source=bc.NetStim(start=start * u.ms, number=1),
                synapse=cell.synapses["ampa"],
                weight=0.005 * u.uS,
            )

        def samples(runs):
            return {
                key: np.concatenate([np.asarray(u.get_mantissa(run.samples[key].values)) for run in runs])
                for key in ("v", "g")
            }

        with _float64():
            out = {}
            for kind in KINDS:
                for start in (0.0, 1.0):
                    for steps in ((200,), (40,) * 5):
                        cell = _hh(kind, 1, synapse=True)
                        cell.record("v", bc.observe.state("V"))
                        cell.record("g", bc.observe.synapse(name="ampa").state("g"))
                        drive(cell, start)
                        cell.init_state()
                        out[kind, start, len(steps)] = samples([cell.run(dt=DT, duration=n * DT) for n in steps])
                        if start == 0.0 and len(steps) == 1:
                            cell.reset_state()
                            out[kind, "reset"] = samples([cell.run(dt=DT, duration=200 * DT)])
            for start, first_step in ((0.0, 1), (1.0, 41)):
                lumped = out["lumped", start, 1]
                for key in ("v", "g"):
                    np.testing.assert_allclose(out["lumped", start, 5][key], lumped[key], rtol=1e-12, atol=0.0)
                    self.assertTwin(out["lumped", start, 5][key], out["cable", start, 5][key], 1e-12, msg=key)
                # Sampled at each step start: an event at t acts in the step that starts at t.
                g = lumped["g"].reshape(200, -1)[:, 0]
                self.assertEqual(int(np.flatnonzero(g > 0.0)[0]), first_step)
            for key in ("v", "g"):
                np.testing.assert_array_equal(out["lumped", "reset"][key], out["lumped", 0.0, 1][key])

    def test_block_solvers_advance_point_synapses(self) -> None:
        # The lumped Cell used to reject them at init_state; now each synapse
        # row sits in its member's block (CELL-31c), as in the cable twin's
        # cell block (CELL-31b), and the two agree.
        for solver in ("exp_euler", "backward_euler", "crank_nicolson"):
            with self.subTest(solver=solver), _float64():
                V = {}
                for kind in KINDS:
                    cell = _hh(kind, synapse=True, solver=solver)
                    cell.init_state()
                    for _, node in cell._runtime.iter_synapse_layouts():
                        node.g.value = u.math.zeros_like(node.g.value) + 0.005 * u.uS
                    cell.run(dt=DT, duration=2.0 * u.ms)
                    V[kind] = cell.V.value
                self.assertGreater(float(np.max(_pair(V["lumped"], V["cable"])[0])), -60.0)  # the synapse acted
                self.assertTwin(V["lumped"], V["cable"], 1e-10)


if __name__ == "__main__":
    unittest.main()
