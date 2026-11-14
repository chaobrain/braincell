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

"""``observe.output(...)`` is sampled at the start of each step, as every other recording (CELL-82).

Row ``j`` is the output at ``t_j``: the value step ``j - 1`` produced, and
row 0 the value after initialization.
"""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell import mech
from braincell.filter import AllRegion, at

DT = 0.025 * u.ms
N = 120


def _float64(test):
    def wrapped(self, *args, **kwargs):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            return test(self, *args, **kwargs)

    wrapped.__name__ = test.__name__
    return wrapped


def _hh(kind="cable"):
    """An HH soma (or a lumped HH Cell) that fires under a 0.1 nA clamp from 0.25 ms."""
    if kind == "lumped":
        cell = braincell.Cell.point(1, V_init=-65.0 * u.mV, V_th=0.0 * u.mV)
        region = ()
    else:
        soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        cell = braincell.Cell(
            braincell.Morphology.from_root(soma, name="soma"),
            cv_policy=braincell.CVPerBranch(),
            V_init=-65.0 * u.mV,
            V_th=0.0 * u.mV,
        )
        region = (AllRegion(),)
    cell.paint(
        *region,
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", g_max=120.0 * (u.mS / u.cm**2)),
        mech.Channel("K_HH1952", g_max=36.0 * (u.mS / u.cm**2)),
        mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.387 * u.mV),
    )
    clamp = mech.CurrentClamp(delay=0.25 * u.ms, durations=1e3 * u.ms, amplitudes=0.1 * u.nA)
    cell.place(*((at("soma", 0.5),) if kind != "lumped" else ()), clamp)
    cell.record("out", braincell.observe.output("voltage"))
    cell.record("V", braincell.observe.state("V"))
    return cell


def _rows(block, unit=u.mV):
    return np.asarray(block.values.to_decimal(unit), dtype=np.float64)


class _Count(braincell.ReductionModel):
    """A reduction whose ``value`` is 5 after initialization and grows by one per step."""

    def init_state(self, context, batch_size=None):
        self.context = context
        self.state = brainstate.ShortTermState(jnp.full(context.pop_size, 5.0))
        return self._output()

    def update(self, inputs):
        del inputs
        self.state.value = self.state.value + 1.0
        return self._output()

    def reset_state(self, batch_size=None):
        self.state.value = jnp.full(self.context.pop_size, 5.0)
        return self._output()

    def reset(self):
        self.state = None

    def _output(self):
        return braincell.ReductionOutput({"value": self.state.value}, jnp.zeros_like(self.state.value, dtype=jnp.int32))


def _reduced():
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"), cv_policy=braincell.CVPerBranch(), pop_size=(2,)
    )
    cell.add_reduction("count", _Count())
    cell.use_model("count")
    cell.record("value", braincell.observe.output("value"))
    return cell


class DetailedOutputTest(unittest.TestCase):
    @_float64
    def test_the_voltage_output_row_is_the_voltage_state_row(self):
        # main 8710fe5: output row j was V(t_{j+1}), the voltage after step j.
        for kind in ("cable", "lumped"):
            for driver in ("Cell.run", "Network.run"):
                with self.subTest(kind=kind, driver=driver):
                    cell = _hh(kind)
                    if driver == "Network.run":
                        net = braincell.Network("n")
                        net.add_population("c", cell)
                        samples = net.run(dt=DT, duration=N * DT).samples["c"]
                    else:
                        samples = cell.run(dt=DT, duration=N * DT).samples
                    out, V = _rows(samples["out"]), _rows(samples["V"])
                    self.assertEqual(out.shape[0], N)
                    self.assertGreater(np.ptp(V), 50.0)  # it fires
                    np.testing.assert_array_equal(out, V)
                    np.testing.assert_array_equal(
                        samples["out"].time.to_decimal(u.ms), samples["V"].time.to_decimal(u.ms)
                    )


class ReducedOutputTest(unittest.TestCase):
    def test_the_first_row_is_the_value_after_initialization(self):
        # main 8710fe5: [[6, 6], [7, 7], [8, 8]], the values after steps 0, 1, 2.
        samples = _reduced().run(dt=DT, duration=3 * DT).samples
        np.testing.assert_array_equal(np.asarray(samples["value"].values), [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])

    def test_a_network_samples_it_alike(self):
        net = braincell.Network("n")
        net.add_population("r", _reduced())
        samples = net.run(dt=DT, duration=3 * DT).samples["r"]
        np.testing.assert_array_equal(np.asarray(samples["value"].values), [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])


class SplitRunTest(unittest.TestCase):
    @_float64
    def test_split_runs_join_into_the_whole_run(self):
        for driver in ("Cell.run", "Network.run"):
            for build, name in ((_hh, "out"), (_reduced, "value")):
                with self.subTest(driver=driver, model=build.__name__):

                    def runs(lengths, build=build):
                        cell = build()
                        if driver == "Network.run":
                            net = braincell.Network("n")
                            net.add_population("c", cell)
                            go = lambda n: net.run(dt=DT, duration=n * DT).samples["c"][name]  # noqa: E731
                        else:
                            go = lambda n: cell.run(dt=DT, duration=n * DT).samples[name]  # noqa: E731
                        blocks = [go(n) for n in lengths]
                        values = np.concatenate([np.asarray(u.get_mantissa(b.values)) for b in blocks])
                        times = np.concatenate([np.asarray(b.time.to_decimal(u.ms)) for b in blocks])
                        return values, times

                    whole, whole_t = runs((N,))
                    split, split_t = runs((37, N - 37))
                    np.testing.assert_array_equal(split, whole)
                    np.testing.assert_allclose(split_t, whole_t, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
