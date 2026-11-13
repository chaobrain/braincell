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

"""Tests for the NumPy-scalar physical constants and the precision switch they fix (OPT-17)."""

import ast
import pathlib
import unittest

import brainstate
import brainunit as u
import numpy as np

import braincell as bc
from braincell._constants import FARADAY_CONSTANT, GAS_CONSTANT
from braincell.filter import AllRegion, RootLocation

_PACKAGE = pathlib.Path(bc.__file__).resolve().parent

#: brainunit attributes held as one shared ndarray: a jaxlib cache entry per object.
_SHARED_ARRAY_CONSTANTS = frozenset(
    name
    for name in dir(u)
    if isinstance(getattr(u, name), u.Quantity) and isinstance(getattr(u, name).mantissa, np.ndarray)
)


def shared_constant_references(source: str) -> list[str]:
    """Return ``"<line>: <name>"`` for each read of a shared-ndarray brainunit constant in ``source``."""
    tree = ast.parse(source)
    aliases = {"brainunit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(a.asname or a.name for a in node.names if a.name == "brainunit")
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "brainunit":
            found += [f"{node.lineno}: {a.name}" for a in node.names if a.name in _SHARED_ARRAY_CONSTANTS]
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in _SHARED_ARRAY_CONSTANTS
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            found.append(f"{node.lineno}: {node.value.id}.{node.attr}")
    return found


class ConstantValueTest(unittest.TestCase):
    """The constants are brainunit's, as ``numpy.float64`` scalars."""

    def test_values_and_units_are_brainunits(self) -> None:
        for ours, theirs in ((GAS_CONSTANT, u.gas_constant), (FARADAY_CONSTANT, u.faraday_constant)):
            with self.subTest(unit=str(theirs.unit)):
                self.assertIs(type(ours.mantissa), np.float64)
                self.assertEqual(ours.unit, theirs.unit)
                self.assertEqual(float(ours.mantissa), float(theirs.mantissa))

    def test_values_are_codata_2018(self) -> None:
        # Exact since the 2019 SI: R = N_A k, F = N_A e.
        n_a, k, e = 6.02214076e23, 1.380649e-23, 1.602176634e-19
        np.testing.assert_allclose(GAS_CONSTANT.to_decimal(u.joule / (u.kelvin * u.mole)), n_a * k, rtol=1e-15)
        np.testing.assert_allclose(FARADAY_CONSTANT.to_decimal(u.coulomb / u.mole), n_a * e, rtol=1e-15)


class LibraryCodeAvoidsSharedConstantsTest(unittest.TestCase):
    """Library code reads R and F from ``braincell._constants``, never brainunit's shared arrays."""

    maxDiff = None

    def test_the_brainunit_constants_braincell_used_are_shared_arrays(self) -> None:
        self.assertLessEqual({"gas_constant", "faraday_constant"}, _SHARED_ARRAY_CONSTANTS)

    def test_no_library_module_reads_a_shared_constant(self) -> None:
        sources = [p for p in sorted(_PACKAGE.rglob("*.py")) if not p.name.endswith("_test.py")]
        self.assertGreater(len(sources), 100)
        offenders = {
            str(path.relative_to(_PACKAGE)): hits
            for path in sources
            if path.name != "_constants.py"
            and (hits := shared_constant_references(path.read_text(encoding="utf-8")))
        }
        self.assertEqual(offenders, {}, "use braincell._constants.GAS_CONSTANT / FARADAY_CONSTANT")

    def test_the_scan_reports_each_form(self) -> None:
        source = (
            "import brainunit as bu\nimport brainunit\nfrom brainunit import faraday_constant\n"
            "a = bu.gas_constant\nb = brainunit.faraday_constant\nc = bu.mV\n"
        )
        self.assertEqual(
            shared_constant_references(source), ["3: faraday_constant", "4: bu.gas_constant", "5: brainunit.faraday_constant"]
        )


def _point_cell(channel):
    # CELL-64: Nernst E (R, F) and the calcium drive (F); Cav3p1 adds the GHK flux (R, F).
    cell = bc.Cell.point(1, V_init=-60.0 * u.mV)
    cell.paint(bc.mech.Ion("CalciumDetailed", name="ca"))
    cell.paint(bc.mech.Channel(channel, name="c"))
    cell.init_state()
    cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
    return cell, cell


def _kinetic_pool_cell():
    # CELL-32: a kinetic Ca pool under "integration", one update() of the staggered solver.
    soma = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = bc.Cell(
        bc.Morphology.from_root(soma, name="soma"),
        cv_policy=bc.CVPerBranch(),
        solver="staggered",
        ion_channel_update_order="integration",
    )
    cell.paint(bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.paint(bc.mech.Ion("CdpStC_MA2020_GoC", name="ca"))
    cell.paint(bc.mech.Channel("CaHVA_MA2020_GoC", g_max=0.5 * u.mS / u.cm**2, ion_name="ca"))
    cell.init_state()
    with brainstate.environ.context(t=0.0 * u.ms, dt=0.005 * u.ms):
        cell.update()
    return cell, cell


def _gaba_network():
    # CELL-31b "G": GabaA_JE2011 writing Cldif_JE2011, driven by a NetStim.
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=-70.0 * u.mV, solver="rk4")
    cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.paint(AllRegion(), bc.mech.Ion("Cldif_JE2011", name="cl"))
    cell.place(RootLocation(0.5), bc.mech.Synapse("GabaA_JE2011", name="s"))
    net = bc.Network(name="opt17_gaba")
    stim = net.add_population("stim", bc.NetStim(size=1, start=0.05 * u.ms, interval=0.05 * u.ms, number=2))
    post = net.add_population("post", cell)
    net.connect("s", source=stim, synapse=post.synapses["s"], weight=0.0005 * u.uS)
    net.init_state()
    net.run(dt=0.005 * u.ms, duration=0.2 * u.ms)
    return net, cell


_MODELS = {
    "CalciumDetailed + CaT_HM1992": lambda: _point_cell("CaT_HM1992"),
    "CalciumDetailed + Cav3p1_MA2020_GoC (GHK)": lambda: _point_cell("Cav3p1_MA2020_GoC"),
    "CdpStC_MA2020_GoC + CaHVA_MA2020_GoC, integration": _kinetic_pool_cell,
    "Cldif_JE2011 + GabaA_JE2011 Network": _gaba_network,
}


class PrecisionSwitchTest(unittest.TestCase):
    """A model runs at one precision while one of the other precision is still alive (OPT-17).

    Before, the second ``init_state`` raised ``JaxRuntimeError``
    (``RuntimeProgramInputMismatch``) in both directions: the live model's
    compiled program kept jaxlib's cached device form of brainunit's
    ``gas_constant`` / ``faraday_constant`` array at its own precision.
    """

    def _check(self, first: int, second: int) -> None:
        dtype = {32: np.float32, 64: np.float64}
        for name, build in _MODELS.items():
            with self.subTest(model=name):
                with brainstate.environ.context(precision=first):
                    kept, kept_cell = build()
                with brainstate.environ.context(precision=second):
                    _, cell = build()
                V = np.asarray(cell.V.value.to_decimal(u.mV))
                self.assertEqual(V.dtype, dtype[second])
                self.assertTrue(np.all(np.isfinite(V)))
                self.assertEqual(np.asarray(kept_cell.V.value.to_decimal(u.mV)).dtype, dtype[first])
                del kept, kept_cell

    def test_float32_with_a_float64_model_alive(self) -> None:
        self._check(64, 32)

    def test_float64_with_a_float32_model_alive(self) -> None:
        self._check(32, 64)


if __name__ == "__main__":
    unittest.main()
