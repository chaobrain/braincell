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

"""Offline regression of BrainCell's morphometry against NeuroMorpho.Org.

``data/morphology/neuromorpho/`` holds seven NeuroMorpho.Org neurons in the
:class:`NeuroMorphoCache` layout: each standardized SWC with the
``metadata.json`` that :meth:`NeuroMorphoClient.download` writes, which carries
NeuroMorpho's published morphometry. Sources, hashes and the CC BY 4.0
citations are in that directory's ``README.md``.

NeuroMorpho computes its morphometry with L-Measure, compartment by compartment
from the SWC points. Surface and volume use a cylinder with the child point's
radius, so the comparison recomputes them from each segment's distal radius
instead of using BrainCell's frustum areas; the frustum values are pinned by a
snapshot of BrainCell's own numbers. Differences that come from a known
convention are registered in :data:`KNOWN_EXCEPTIONS`; a registered row that
starts to agree with NeuroMorpho fails the test, so the table cannot go stale.

``examples/io/neuromorpho.ipynb`` copies the formulas and thresholds of
:func:`compare`.
"""

import hashlib
import math
import unittest
from collections.abc import Iterable, Mapping

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.io._testing import FIXTURE_DIR
from braincell.io.neuromorpho import NeuroMorphoCache, NeuroMorphoMeasurement

DATA_DIR = FIXTURE_DIR / "neuromorpho"

SWC_SHA256 = {
    100: "39711b83c3793466ec13c90fcbe68e25df9b6d661ea2364eca5930b48b492af6",
    1013: "26247c9d947ace7d7bb529c0d174a649dcd34a790d3877be7bfc418858fff68f",
    10036: "cf59917cae264e397522be011e63701af5903f41f2a7fe6d7f3086e201ef60f0",
    10074: "98ee4f752385b119c7532cce71cf5ef053030312bb9a1b5b15c64b2661f5a48c",
    102085: "5d09c3b987329b57563bd5003cafb010c41d55f149486d53d401b58bfa277575",
    107134: "5cfe270e49e51f7615cdd3d807b59389f9b489727e7f6cbf1c9d50b2c6fdd2a2",
    122127: "016b1c8f1621371501025b39311159a7838e62a5274d244342b14d43bad74c76",
}
"""SHA-256 of each standardized SWC as published by NeuroMorpho.Org."""

NEURON_IDS = tuple(sorted(SWC_SHA256))

INTEGER_METRICS = ("n_stems", "n_branch", "n_bifs")

RTOL = {
    "length": 1e-4,
    "pathDistance": 1e-4,
    "eucDistance": 1e-4,
    "surface": 1e-4,
    "volume": 1e-4,
    "soma_Surface": 2e-2,
}
"""Relative tolerance against NeuroMorpho, per metric.

NeuroMorpho publishes six significant digits, and its residuals on the
fixture reach 5.9e-5 (1013 ``eucDistance``); 1e-4 keeps a margin above
that and still fails when the radii of a branch holding 0.5% of the surface
change by 10%. ``soma_Surface`` only holds the soma convention: BrainCell
reads a three-point soma as a cylinder of area 4 pi r^2, and L-Measure's
formula equals it only when the side points lie exactly at +-r (122127:
0.985 r, 1.54e-2). Soma geometry changes are caught by the snapshot.
"""

MEASUREMENT_FIELDS = {
    "n_stems": "n_stems",
    "n_branch": "n_branch",
    "n_bifs": "n_bifs",
    "length": "length",
    "pathDistance": "path_distance",
    "eucDistance": "euclidean_distance",
    "surface": "surface",
    "volume": "volume",
    "soma_Surface": "soma_surface",
}
"""NeuroMorpho metric name -> :class:`NeuroMorphoMeasurement` attribute."""

SNAPSHOT_RTOL = 1e-9

SNAPSHOT = {
    102085: (9098.382188156236, 5934.847284224976, 400.03604040270716),
    10074: (33781.08621449734, 15906.016234653089, 180.13330822811193),
    107134: (13429.669340690285, 1010.3691888889916, 553.5457227309096),
    122127: (8635.016915875029, 539.6885572421893, 12.566370614359172),
    100: (24176.831556390676, 4335.048471317433, 696.1589874236679),
    1013: (7476.020546150411, 2069.2948671473396, 922.4289385480871),
    10036: (4511.947253405163, 233.28628766564262, 1387.5543147440303),
}
"""BrainCell's frustum surface (um^2) and volume (um^3) of the non-soma
branches, and the soma branches' area (um^2).

The cylinder formulas use only ``radii_distal``; the snapshot also covers
``radii_proximal``, zero-length jump segments and the soma. 1e-9 is far above
the summation-order error (about 1e-13) and far below any geometry change.
"""

_MULTI_POINT_SOMA = (
    "multi-point soma ({n} points): n_stems counts the root branch's children, which are soma "
    "branches; NeuroMorpho measures distances from the SWC root through the soma, BrainCell's "
    "_excluding_soma distances from the soma boundary; NeuroMorpho's soma_Surface method for "
    "multi-point somata is not known"
)
_CON2PROX = (
    "con2prox (point {point}): the point joins the soma and has two children; L-Measure counts "
    "a bifurcation there, the neuromorpho mode attaches the second child at x = 0 of its sibling"
)

KNOWN_EXCEPTIONS = (
    (100, "n_stems", _MULTI_POINT_SOMA.format(n=14), 4),
    (100, "pathDistance", _MULTI_POINT_SOMA.format(n=14), 1074.9354884018971),
    (100, "eucDistance", _MULTI_POINT_SOMA.format(n=14), 526.2572397601766),
    (100, "soma_Surface", _MULTI_POINT_SOMA.format(n=14), 696.1589874236679),
    (1013, "n_stems", _MULTI_POINT_SOMA.format(n=12), 4),
    (1013, "n_branch", _CON2PROX.format(point=915), 216),
    (1013, "n_bifs", _CON2PROX.format(point=915), 104),
    (1013, "pathDistance", _MULTI_POINT_SOMA.format(n=12), 256.66176668439414),
    (1013, "soma_Surface", _MULTI_POINT_SOMA.format(n=12), 922.4289385480871),
    (10036, "n_branch", _CON2PROX.format(point=4267), 105),
    (10036, "n_bifs", _CON2PROX.format(point=4267), 50),
)
"""Known disagreements: (neuron id, metric, reason, BrainCell value).

Each row must still disagree with NeuroMorpho, and BrainCell's value must stay
the registered one (equal for counts, ``SNAPSHOT_RTOL`` otherwise).
"""


def braincell_metrics(morpho: Morphology) -> dict[str, float | int]:
    """Return BrainCell's value of each NeuroMorpho metric for ``morpho``."""

    soma = [branch for branch in morpho.branches if branch.type == "soma"]
    neurites = [branch for branch in morpho.branches if branch.type != "soma"]
    surface = volume = 0.0
    for branch in neurites:
        lengths = np.asarray(branch.branch.lengths.to_decimal(u.um))
        radii = np.asarray(branch.branch.radii_distal.to_decimal(u.um))
        surface += float(np.sum(2.0 * math.pi * radii * lengths))
        volume += float(np.sum(math.pi * radii * radii * lengths))
    return {
        "n_stems": int(morpho.n_stems),
        "n_branch": len(neurites),
        "n_bifs": sum(branch.n_children >= 2 for branch in neurites),
        "length": float(sum(branch.length.to_decimal(u.um) for branch in neurites)),
        "pathDistance": float(morpho.max_path_distance_excluding_soma.to_decimal(u.um)),
        "eucDistance": float(morpho.max_euclidean_distance_excluding_soma.to_decimal(u.um)),
        "surface": surface,
        "volume": volume,
        "soma_Surface": float(sum(branch.area.to_decimal(u.um**2) for branch in soma)),
    }


def compare(
    morpho: Morphology,
    measurement: NeuroMorphoMeasurement,
) -> dict[str, tuple[float | int, float | int, bool]]:
    """Compare ``morpho`` with NeuroMorpho's ``measurement``.

    Returns
    -------
    dict
        Metric -> (BrainCell value, NeuroMorpho value, passed). Counts must be
        equal; the other metrics pass when ``|bc - nm| / |nm| <= RTOL[metric]``.
    """

    results = {}
    for metric, value in braincell_metrics(morpho).items():
        reference = getattr(measurement, MEASUREMENT_FIELDS[metric])
        if metric in INTEGER_METRICS:
            passed = value == int(reference)
        else:
            passed = abs(value - reference) <= RTOL[metric] * abs(reference)
        results[metric] = (value, reference, passed)
    return results


def snapshot(morpho: Morphology) -> tuple[float, float, float]:
    """Frustum surface and volume of the non-soma branches, and the soma area."""

    neurites = [branch for branch in morpho.branches if branch.type != "soma"]
    soma = [branch for branch in morpho.branches if branch.type == "soma"]
    return (
        float(sum(branch.area.to_decimal(u.um**2) for branch in neurites)),
        float(sum(branch.volume.to_decimal(u.um**3) for branch in neurites)),
        float(sum(branch.area.to_decimal(u.um**2) for branch in soma)),
    )


def _same_value(value: float | int, registered: float | int) -> bool:
    if isinstance(registered, int):
        return value == registered
    return math.isclose(value, registered, rel_tol=SNAPSHOT_RTOL, abs_tol=0.0)


def check(
    results: Mapping[int, Mapping[str, tuple[float | int, float | int, bool]]],
    exceptions: Iterable[tuple[int, str, str, float | int]] = KNOWN_EXCEPTIONS,
) -> list[str]:
    """Return one message per problem in ``results`` (neuron id -> :func:`compare`).

    A failing metric must be a registered exception; a registered exception
    must still fail, with BrainCell's registered value. An empty list means the
    regression holds.
    """

    registered = {(neuron_id, metric): value for neuron_id, metric, _, value in exceptions}
    problems = []
    for (neuron_id, metric), value in registered.items():
        braincell_value, reference, passed = results[neuron_id][metric]
        if passed:
            problems.append(
                f"{neuron_id} {metric}: BrainCell {braincell_value!r} now agrees with NeuroMorpho "
                f"{reference!r}; the exception no longer holds, delete this row."
            )
        elif not _same_value(braincell_value, value):
            problems.append(
                f"{neuron_id} {metric}: BrainCell {braincell_value!r} differs from the registered "
                f"value {value!r}; find the change, then update the row."
            )
    for neuron_id, metrics in results.items():
        for metric, (braincell_value, reference, passed) in metrics.items():
            if passed or (neuron_id, metric) in registered:
                continue
            if metric in INTEGER_METRICS:
                detail = "counts differ"
            else:
                detail = (
                    f"relative difference {abs(braincell_value - reference) / abs(reference):.3e} > {RTOL[metric]:g}"
                )
            problems.append(
                f"{neuron_id} {metric}: BrainCell {braincell_value!r}, NeuroMorpho {reference!r} ({detail})."
            )
    return problems


def _scaled_radii(morpho: Morphology, name: str, factor: float) -> Morphology:
    edited = morpho.clone()
    branch = edited.branch(name=name).branch
    edited.replace(
        name,
        Branch(
            lengths=branch.lengths,
            radii_proximal=branch.radii_proximal * factor,
            radii_distal=branch.radii_distal * factor,
            points_proximal=branch.points_proximal,
            points_distal=branch.points_distal,
            type=branch.type,
        ),
    )
    return edited


class NeuroMorphoRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache = NeuroMorphoCache(DATA_DIR)
        cls.morphologies = {neuron_id: cls.cache.load(neuron_id) for neuron_id in NEURON_IDS}
        cls.measurements = {neuron_id: cls.cache.measurement(neuron_id) for neuron_id in NEURON_IDS}
        cls.results = {
            neuron_id: compare(cls.morphologies[neuron_id], cls.measurements[neuron_id]) for neuron_id in NEURON_IDS
        }

    def test_fixture_is_the_published_data(self) -> None:
        self.assertEqual(self.cache.list_neurons(), (100, 1013, 10036, 10074, 102085, 107134, 122127))
        for neuron_id in NEURON_IDS:
            with self.subTest(neuron_id=neuron_id):
                path = self.cache.standard_swc_path(neuron_id)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), SWC_SHA256[neuron_id])
                self.assertIsNotNone(self.measurements[neuron_id])
        readme = (DATA_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("CC BY 4.0", readme)
        for neuron_id in NEURON_IDS:
            self.assertIn(str(neuron_id), readme)
            self.assertIn(SWC_SHA256[neuron_id], readme)

    def test_reference_values_come_from_the_metadata(self) -> None:
        measurement = self.measurements[102085]
        self.assertEqual(
            {metric: getattr(measurement, field) for metric, field in MEASUREMENT_FIELDS.items()},
            {
                "n_stems": 1,
                "n_branch": 15,
                "n_bifs": 7,
                "length": 1116.79,
                "pathDistance": 252.892,
                "eucDistance": 196.417,
                "surface": 9083.6,
                "volume": 5909.05,
                "soma_Surface": 399.001,
            },
        )

    def test_metrics_agree_with_neuromorpho(self) -> None:
        self.assertEqual(check(self.results), [])

    def test_snapshot(self) -> None:
        for neuron_id in NEURON_IDS:
            with self.subTest(neuron_id=neuron_id):
                for value, registered in zip(snapshot(self.morphologies[neuron_id]), SNAPSHOT[neuron_id], strict=True):
                    self.assertTrue(math.isclose(value, registered, rel_tol=SNAPSHOT_RTOL), (value, registered))

    def test_scaled_radii_of_the_smallest_branch_fail(self) -> None:
        # basal_dendrite_3 of 102085 holds 0.49% of the non-soma surface (2 segments).
        morpho = self.morphologies[102085]
        edited = _scaled_radii(morpho, "basal_dendrite_3", 1.1)

        results = compare(edited, self.measurements[102085])

        surface, reference_surface, surface_passed = results["surface"]
        volume, reference_volume, volume_passed = results["volume"]
        self.assertFalse(surface_passed)
        self.assertFalse(volume_passed)
        self.assertAlmostEqual(abs(surface - reference_surface) / reference_surface, 4.918e-4, delta=5e-7)
        self.assertAlmostEqual(abs(volume - reference_volume) / reference_volume, 1.020e-3, delta=5e-7)
        problems = check({**self.results, 102085: results})
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("102085 surface:"), problems)
        self.assertTrue(problems[1].startswith("102085 volume:"), problems)
        frustum_surface = snapshot(edited)[0]
        change = abs(frustum_surface - SNAPSHOT[102085][0]) / SNAPSHOT[102085][0]
        self.assertAlmostEqual(change, 4.908e-4, delta=5e-7)
        self.assertGreater(change, SNAPSHOT_RTOL)

    def test_an_exception_that_agrees_is_reported_as_stale(self) -> None:
        exceptions = KNOWN_EXCEPTIONS + ((102085, "length", "fake", 1116.79),)

        problems = check(self.results, exceptions)

        self.assertEqual(len(problems), 1)
        self.assertIn("102085 length", problems[0])
        self.assertIn("the exception no longer holds, delete this row", problems[0])

    def test_a_changed_exception_value_is_reported(self) -> None:
        exceptions = tuple(
            (neuron_id, metric, reason, 5 if (neuron_id, metric) == (100, "n_stems") else value)
            for neuron_id, metric, reason, value in KNOWN_EXCEPTIONS
        )

        problems = check(self.results, exceptions)

        self.assertEqual(len(problems), 1)
        self.assertIn("100 n_stems: BrainCell 4 differs from the registered value 5", problems[0])

    def test_an_unregistered_failure_is_reported(self) -> None:
        exceptions = tuple(row for row in KNOWN_EXCEPTIONS if row[:2] not in {(1013, "n_bifs"), (100, "eucDistance")})

        problems = check(self.results, exceptions)

        self.assertEqual(
            problems,
            [
                "100 eucDistance: BrainCell 526.2572397601766, NeuroMorpho 529.358 (relative difference 5.858e-03 > 0.0001).",
                "1013 n_bifs: BrainCell 104, NeuroMorpho 105 (counts differ).",
            ],
        )


if __name__ == "__main__":
    unittest.main()
