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

"""``GabaA_JE2011`` and the Cell path for synapses that read and write an ion."""

import functools
import json
import os
import subprocess
import sys
import unittest

import brainstate
import brainunit as u
import numpy as np

import braincell as bc
from braincell import IonInfo
from braincell._multi_compartment import currents
from braincell.filter import RootLocation
from braincell.mech import register_synapse
from braincell.synapse import Exp2Syn, GabaA_JE2011

_E_CL_5 = -87.22198260005996  # mV, Ci 5 mM, Co 133.5 mM, 35 degC
_E_CL_10 = -68.81593579550584  # mV, Ci 10 mM
_E_GABA = -73.84264445450641  # mV, P = 0.18 with _E_CL_5
_DT = 0.025 * u.ms


@register_synapse("UserExp2SynGabaTest")
class _UserExp2Syn(Exp2Syn):
    """A user subclass written against the one-argument ``current``."""

    def current(self, V_post):
        return super().current(V_post)

    def ion_current(self, V_post, ion_info):
        raise AssertionError("a synapse without ion_type must not be asked for an ion current")


def _cell(*placements, ion="ChlorideInitNernst", n_cv=1, length=20.0, **cell_kw):
    branch = bc.Branch.from_lengths(lengths=[length] * u.um, radii=[0.5, 0.5] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(n_cv), V_init=-70.0 * u.mV, **cell_kw)
    cell.paint(bc.mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
    if ion == "ChlorideInitNernst":
        cell.paint(bc.mech.Ion(ion, name="cl", temp=u.celsius2kelvin(35.0)))
    elif ion is not None:
        cell.paint(bc.mech.Ion(ion, name="cl"))
    for x, synapse in placements:
        cell.place(RootLocation(x), synapse)
    cell.record("v", bc.observe.state("V"))
    return cell


def _network(cell, *weights, start=5.0, interval=25.0):
    net = bc.Network("gaba")
    stim = net.add_population("stim", bc.NetStim(size=1, start=start * u.ms, number=100, interval=interval * u.ms))
    post = net.add_population("post", cell)
    for name, weight in weights:
        net.connect(name, source=stim, synapse=post.synapses[name], weight=weight * u.uS)
    return net


def _samples(net, duration):
    samples = net.run(dt=_DT, duration=duration * u.ms).samples["post"]
    units = {"v": u.mV, "i": u.nA, "ci": u.mM}
    return {key: np.asarray(value.values.to_decimal(units[key])) for key, value in samples.items()}


def _synapse_layout(cell):
    return next(layout for layout in cell.runtime.layouts if layout.synapse_index is not None)


def _equivalence(solver, P, e):
    traces = []
    for synapse in (
        bc.mech.Synapse("GabaA_JE2011", name="s", P=P),
        bc.mech.Synapse("Exp2Syn", name="s", tau1=0.5 * u.ms, tau2=6.0 * u.ms, e=e * u.mV),
    ):
        cell = _cell((0.5, synapse), solver=solver)
        cell.record("i", bc.observe.synapse(name="s").current())
        traces.append(_samples(_network(cell, ("s", 0.0005)), 200.0))
    return [float(np.max(np.abs(traces[0][key] - traces[1][key]))) for key in ("v", "i")]


def _legacy_trace():
    """Exp2Syn and ExpSyn (one at a CV interior, one at the branch end) next to a chloride ion, staggered."""
    cell = _cell(
        (0.25, bc.mech.Synapse("Exp2Syn", name="a", tau1=0.5 * u.ms, tau2=6.0 * u.ms, e=-75.0 * u.mV)),
        (1.0, bc.mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV)),
        n_cv=2,
        length=40.0,
    )
    cell.record("i", bc.observe.synapse(name="a").current())
    samples = _samples(_network(cell, ("a", 0.002), ("b", 0.001), start=1.0, interval=7.0), 40.0)
    return {key: value[::200].tolist() for key, value in samples.items()}


def _x64_report() -> dict:
    """Float64 measurements, run in a fresh process by ``_float64_report`` (see ``ion/chloride_test.py``)."""
    brainstate.environ.set(precision=64)
    report = {
        "equivalence": {
            f"{solver}-{P}": _equivalence(solver, P, e)
            for solver in ("staggered", "ralston2")
            for P, e in ((0.18, _E_GABA), (0.0, _E_CL_5))
        },
        "legacy": _legacy_trace(),
    }

    # Reading the ion: each row gets its own CV (and population).
    for pop_size in (None, 2):
        cell = _cell(
            (0.25, bc.mech.Synapse("GabaA_JE2011", name="a")),
            (0.75, bc.mech.Synapse("GabaA_JE2011", name="b")),
            ion=None,
            n_cv=2,
            **({} if pop_size is None else {"pop_size": pop_size}),
        )
        ci = lambda ctx: (5.0 if ctx.cv_id == 0 else 10.0) * u.mM  # noqa: E731
        cell.paint(bc.mech.Ion("ChlorideInitNernst", name="cl", temp=u.celsius2kelvin(35.0), Ci=ci))
        cell.init_state()
        ion = cell.runtime.get_ion("cl")
        layout = _synapse_layout(cell)
        rows = lambda: currents._synapse_row_ion_info(layout, ion.pack_info()).E.to_decimal(u.mV).tolist()  # noqa: E731
        report[f"rows-{pop_size}"] = rows()
        if pop_size is not None:
            ion._E.value = u.Quantity(np.array([[-80.0, -60.0], [-50.0, -40.0]]), u.mV)
            report["rows-population"] = rows()

    # Writing the ion: GABA chloride influx drives Cldif_JE2011.
    for remove in (False, True):
        cell = _cell((0.5, bc.mech.Synapse("GabaA_JE2011", name="s")), ion="Cldif_JE2011")
        cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
        net = _network(cell, ("s", 0.0005))
        net.init_state()
        if remove:
            cell.runtime.get_ion("cl").external_currents.clear()
        report[f"ci-removed-{remove}"] = float(_samples(net, 1005.0)["ci"][-1, 0])

    # CELL-19: generic Cell solvers against the Richardson limit of staggered
    # (first order), V (mV) and Ci (mM) at 1 ms.
    def final(solver, dt):
        cell = _cell((0.5, bc.mech.Synapse("GabaA_JE2011", name="s")), ion="Cldif_JE2011", solver=solver)
        _network(cell, ("s", 0.0005), start=0.1, interval=0.3).run(dt=dt * u.ms, duration=1.0 * u.ms)
        values = (cell.V.value.to_decimal(u.mV), cell.runtime.get_ion("cl").Ci.value.to_decimal(u.mM))
        return [float(np.asarray(value).reshape(-1)[0]) for value in values]

    half, quarter = final("staggered", 0.0005), final("staggered", 0.00025)
    limit = [2.0 * b - a for a, b in zip(half, quarter)]
    report["generic"] = {
        f"{solver}-{dt}": [got - want for got, want in zip(final(solver, dt), limit)]
        for solver, dt in (("rk4", 0.005), ("ralston2", 0.001), ("ind_exp_euler", 0.001))
    }
    report["generic-ci-rise"] = limit[1] - 5.0

    # At a branch end the chloride current is divided by the area that
    # normalizes the synapse's point current.
    cell = _cell((1.0, bc.mech.Synapse("GabaA_JE2011", name="s", P=0.0)), ion="Cldif_JE2011", n_cv=2)
    cell.init_state()
    runtime, layout = cell.runtime, _synapse_layout(cell)
    syn = runtime.runtime_nodes[layout.id]
    syn.B.value = u.Quantity(np.full(syn.B.value.shape, 0.001), u.uS)
    ion = runtime.get_ion("cl")
    density = ion.external_currents[("synapse", layout.id)](cell.V.value, ion.pack_info())
    point_V = cell._point_voltage_for_mechanisms(cell.V.value)
    point = currents._synapse_contrib_to_point(runtime, layout, syn, point_V)
    report["end"] = {
        "cv_index": layout.cv_index.tolist(),
        "ion": density.to_decimal(u.nA / u.cm**2).reshape(-1).tolist(),
        "point": float(point[..., int(layout.point_index[0])].to_decimal(u.nA / u.cm**2).reshape(-1)[0]),
    }
    return report


@functools.cache
def _float64_report() -> dict:
    code = "import json; from braincell.synapse.gaba_test import _x64_report; print(json.dumps(_x64_report()))"
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class GabaAModelTest(unittest.TestCase):
    """Currents, reversal potentials and parameter checks of the synapse alone."""

    def test_currents_and_reversal_potentials(self) -> None:
        syn = GabaA_JE2011(size=1)
        syn.init_state()
        syn.B.value = 0.001 * u.uS
        self.assertIs(syn.ion_type, bc.ion.Chloride)
        np.testing.assert_allclose(syn.E_HCO3.to_decimal(u.mV), -12.892326, rtol=1e-6)
        cl = IonInfo(Ci=5.0 * u.mM, Co=133.5 * u.mM, E=_E_CL_5 * u.mV, valence=-1)
        V = -60.0 * u.mV
        np.testing.assert_allclose(syn.ion_current(V, cl).to_decimal(u.nA), 0.82 * 0.001 * (_E_CL_5 + 60.0), rtol=1e-6)
        np.testing.assert_allclose(
            (syn.current(V, cl) - syn.ion_current(V, cl)).to_decimal(u.nA),
            0.18 * 0.001 * (-12.892326 + 60.0),
            rtol=1e-6,
        )
        np.testing.assert_allclose(syn.current(_E_GABA * u.mV, cl).to_decimal(u.nA), 0.0, atol=1e-9)

    def test_base_synapse_reads_no_ion(self) -> None:
        self.assertIsNone(bc.Synapse.ion_type)
        self.assertIsNone(Exp2Syn(size=1).ion_current(-70.0 * u.mV, None))

    def test_invalid_parameters_raise(self) -> None:
        for kwargs in ({"tau1": 6.0 * u.ms}, {"tau1": 7.0 * u.ms}, {"P": -0.1}, {"P": 1.1}):
            with self.subTest(**{key: str(value) for key, value in kwargs.items()}):
                with self.assertRaises(ValueError):
                    GabaA_JE2011(size=1, **kwargs)
        GabaA_JE2011(size=1, P=0.0)
        GabaA_JE2011(size=1, P=1.0)

    def test_invalid_parameters_through_a_cell_name_the_synapse(self) -> None:
        # CELL-44: the checks above, reached through init_state and set, say which synapse failed.
        for kwargs, problem in (
            ({"tau1": 6.0 * u.ms}, "tau1 < tau2"),
            ({"P": 1.1}, "0 <= P <= 1"),
            ({"HCO3i": -1.0 * u.mM}, "'HCO3i' must be > 0"),
        ):
            with self.subTest(next(iter(kwargs))):
                cell = _cell((0.5, bc.mech.Synapse("GabaA_JE2011", name="g", **kwargs)))
                with self.assertRaisesRegex(ValueError, rf"^Synapse 'g' \(GabaA_JE2011\).*{problem}.*Placed on"):
                    cell.init_state()
        cell = _cell((0.5, bc.mech.Synapse("GabaA_JE2011", name="g")))
        cell.init_state()
        with self.assertRaisesRegex(ValueError, r"^Synapse 'g' \(GabaA_JE2011\): GabaA_JE2011 requires 0 <= P <= 1\.$"):
            cell.synapses["g"].set(P=-0.5)
        np.testing.assert_allclose(cell.synapses["g"].P, [0.18])


class GabaACellTest(unittest.TestCase):
    """The Cell passes each row's chloride state and adds the chloride current to the ion."""

    def test_matches_exp2syn_with_the_same_reversal(self) -> None:
        for case, (dv, di) in _float64_report()["equivalence"].items():
            with self.subTest(case=case):
                self.assertLessEqual(dv, 1e-12)  # mV
                self.assertLessEqual(di, 1e-15)  # nA

    def test_each_row_reads_its_own_cv_and_population(self) -> None:
        report = _float64_report()
        for key in ("rows-None", "rows-2"):
            expected = [_E_CL_5, _E_CL_10] * (1 if key == "rows-None" else 2)
            np.testing.assert_allclose(report[key], expected, rtol=1e-9)
        np.testing.assert_allclose(report["rows-population"], [-80.0, -60.0, -50.0, -40.0])

    def test_chloride_influx_raises_ci(self) -> None:
        report = _float64_report()
        np.testing.assert_allclose(report["ci-removed-False"], 6.0100, rtol=1e-4)
        # Counter-example: without the registered callback Ci stays at rest.
        np.testing.assert_allclose(report["ci-removed-True"], 5.0, rtol=1e-12)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(report["ci-removed-True"], 6.0100, rtol=1e-4)

    def test_branch_end_current_uses_the_end_cv_area(self) -> None:
        end = _float64_report()["end"]
        self.assertEqual(end["cv_index"], [1])
        self.assertEqual(end["ion"][0], 0.0)
        np.testing.assert_allclose(end["ion"][1], end["point"], rtol=1e-12)

    def test_synapses_without_ion_type_are_unchanged(self) -> None:
        # Reference values from the code before ``ion_type`` existed (float64).
        legacy = _float64_report()["legacy"]
        for key, values in _LEGACY.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(legacy[key], values, rtol=1e-13, atol=0.0)

    def test_user_subclass_keeps_one_argument_current(self) -> None:
        traces = []
        for synapse_type in ("Exp2Syn", "UserExp2SynGabaTest"):
            cell = _cell((0.5, bc.mech.Synapse(synapse_type, name="s", tau1=0.5 * u.ms, tau2=6.0 * u.ms)))
            cell.record("i", bc.observe.synapse(name="s").current())
            net = _network(cell, ("s", 0.0005))
            net.init_state()
            self.assertEqual(cell.runtime.bound_ion_keys[_synapse_layout(cell).id], ())
            self.assertEqual(cell.runtime.get_ion("cl").external_currents, {})
            traces.append(_samples(net, 10.0))
        for key in ("v", "i"):
            np.testing.assert_array_equal(traces[0][key], traces[1][key])


class GabaAFloat32CellTest(unittest.TestCase):
    def test_chloride_accumulates_and_the_current_is_recorded(self) -> None:
        cell = _cell((0.5, bc.mech.Synapse("GabaA_JE2011", name="s")), ion="Cldif_JE2011")
        cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
        cell.record("i", bc.observe.synapse(name="s").current())
        samples = _samples(_network(cell, ("s", 0.0005)), 40.0)
        self.assertTrue(np.all(np.isfinite(samples["ci"])))
        self.assertGreater(samples["ci"][-1, 0], 5.0 + 1e-3)
        self.assertLess(np.min(samples["i"]), 0.0)  # outward at V > E_GABA


class GabaAGenericSolverTest(unittest.TestCase):
    """CELL-19: a generic Cell solver snapshots the chloride current after its step.

    ``Cldif_JE2011`` integrates its total current, which includes the GABA
    current; under ``ralston2`` and ``ind_exp_euler`` it used to raise in
    ``run``. Measured in the float64 report.
    """

    def test_generic_solvers_converge_to_the_staggered_limit(self) -> None:
        report = _float64_report()
        self.assertGreater(report["generic-ci-rise"], 4e-3)  # mM
        # Limits about 4x the measured gaps (comments). Ci's gap is first order
        # through the pool's own backward Euler subsolver, as under staggered
        # (2.2e-5 mM at dt = 0.005 ms).
        for case, (tol_v, tol_ci) in {
            "rk4-0.005": (5e-6, 1e-4),  # 1.2e-6 mV, 2.4e-5 mM
            "ralston2-0.001": (4e-6, 2e-5),  # 9.3e-7 mV, 4.8e-6 mM
            "ind_exp_euler-0.001": (7e-3, 2e-5),  # 1.6e-3 mV (first order), 5.2e-6 mM
        }.items():
            with self.subTest(case=case):
                gap_v, gap_ci = report["generic"][case]
                self.assertLessEqual(abs(gap_v), tol_v)
                self.assertLessEqual(abs(gap_ci), tol_ci)


class GabaAGuardTest(unittest.TestCase):
    """Missing or ambiguous chloride, a missing snapshot, and a lumped Cell."""

    def test_dynamic_chloride_needs_the_snapshot(self) -> None:
        gaba = (0.5, bc.mech.Synapse("GabaA_JE2011", name="s"))
        cell = _cell(gaba, ion="Cldif_JE2011", cache_ion_total_current=False)
        with self.assertRaisesRegex(ValueError, "'GabaA_JE2011'.*'cl'"):
            cell.init_state()
        cell = _cell(gaba, ion="Cldif_JE2011")
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "'GabaA_JE2011'.*'cl'.*snapshot"):
            cell.runtime.get_ion("cl").compute_derivative(cell.V.value)
        # A fixed-concentration ion consumes no current: no snapshot is needed.
        _cell(gaba, cache_ion_total_current=False).init_state()

    def test_chloride_ion_must_be_unique(self) -> None:
        gaba = (0.5, bc.mech.Synapse("GabaA_JE2011", name="s"))
        with self.assertRaisesRegex(KeyError, "No ion candidates are registered for family 'cl'."):
            _cell(gaba, ion=None).init_state()
        cell = _cell(gaba)
        cell.paint(bc.mech.Ion("ChlorideFixed", name="cl2"))
        with self.assertRaisesRegex(ValueError, "Ion family 'cl' is ambiguous"):
            cell.init_state()

    def test_lumped_cell_places_the_synapse(self) -> None:
        # CELL-24; the traces are checked against the one-CV twin in lumped_twin_test.py.
        cell = bc.Cell.point(1)
        cell.paint(bc.mech.Ion("ChlorideInitNernst", name="cl", temp=u.celsius2kelvin(35.0)))
        cell.place(bc.mech.Synapse("GabaA_JE2011", name="s"))
        cell.init_state()
        self.assertEqual(cell.runtime.bound_ion_keys[_synapse_layout(cell).id], ("cl",))


#: ``_legacy_trace()`` before ``Synapse.ion_type`` existed (every 200th step, float64).
_LEGACY = {
    "v": [
        [-70.0, -70.0],
        [-66.66413946318491, -66.3980586603182],
        [-63.68426141053313, -63.01142620439888],
        [-72.07741391792784, -72.00657433011047],
        [-70.46941498459844, -70.29523663993392],
        [-67.34199118144146, -66.90968115821363],
        [-60.56480342308205, -59.539062217176344],
        [-71.63589849896005, -71.52555189178827],
    ],
    "i": [
        [-0.0],
        [-0.011696687060889338],
        [-0.028512251741391742],
        [-0.0032640118098576686],
        [-0.007583231436647879],
        [-0.018221780774477777],
        [-0.043044191033815744],
        [-0.004900686788628389],
    ],
}

if __name__ == "__main__":
    unittest.main()
