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

"""Shared fixture builders for ``braincell.filter`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

The per-type branch builders are re-exported rather than redefined:
``braincell/morph/_testing.py`` owns them, because the objects they build
are morphology objects with no selection content.

The second half is the selection-cache layer of the morphology edit table
(``braincell.morph._testing``): :data:`SELECTION_BATTERY` and
:func:`assert_selection_fresh`.
"""

import warnings
from typing import Callable, Mapping

import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell.filter.cache import SelectionCache
from braincell.filter.locset import (
    BranchPoints,
    LocsetMask,
    RegionAnchors,
    RootLocation,
    StepSamples,
    Terminals,
    UniformSamples,
    at,
)
from braincell.filter.region import AllRegion, EmptyRegion, RadiusRangeRegion, RegionMask, SubtreeRegion, branch_in
from braincell.io._testing import FIXTURE_DIR
from braincell.morph._testing import EditCase, assert_layer_fresh, make_tree_t

# Re-exported so the filter tests keep a single import site for fixtures.
from braincell.morph._testing import (  # noqa: F401
    make_apical,
    make_axon,
    make_basal,
    make_dendrite,
    make_soma,
    make_soma_dend_tree,
)

__all__ = [
    "SELECTION_BATTERY",
    "assert_selection_fresh",
    "brute_force_mismatches",
    "crossing_errors",
    "fixture_morphologies",
    "make_apical",
    "make_axon",
    "make_basal",
    "make_dendrite",
    "make_soma",
    "make_single_branch_tree",
    "make_soma_dend_tree",
    "make_tree_r",
    "radius_um",
]


def make_single_branch_tree(*, length: float = 10.0, radius: float = 1.0) -> Morphology:
    """A one-branch morphology whose single segment has a uniform radius."""
    root = Branch.from_lengths(lengths=[length] * u.um, radii=[radius, radius] * u.um, type="soma")
    return Morphology.from_root(root, name="soma")


def fixture_morphologies() -> list[tuple[str, Morphology]]:
    """Every morphology under ``FIXTURE_DIR``: the SWC files, then the ASC files."""
    paths = sorted(FIXTURE_DIR.glob("*.swc")) + sorted(FIXTURE_DIR.glob("*.asc"))
    with warnings.catch_warnings():
        # Some fixtures repeat a sample; the reader keeps it as a zero-length segment and says so.
        warnings.filterwarnings("ignore", r"from_points\(\) produced \d+ zero-length segment", UserWarning)
        return [
            (path.name, Morphology.from_swc(path) if path.suffix == ".swc" else Morphology.from_asc(path))
            for path in paths
        ]


def _from_points(points, radii=None, *, radii_proximal=None, radii_distal=None, type: str) -> Branch:
    as_um = lambda values: np.asarray(values, dtype=float) * u.um  # noqa: E731
    if radii is not None:
        return Branch.from_points(points=as_um(points), radii=as_um(radii), type=type)
    return Branch.from_points(
        points=as_um(points), radii_proximal=as_um(radii_proximal), radii_distal=as_um(radii_distal), type=type
    )


def make_tree_r(second_soma: bool = False) -> Morphology:
    """Tapers, a radius jump, a fork, a reversed child and a U-shaped axon, for the range regions.

    Every branch is built with :meth:`Branch.from_points` from float64
    arrays (micrometres):

    =====  =====  ===================================  ======================  ==========================
    index  name   points                               radii                   joint
    =====  =====  ===================================  ======================  ==========================
    0      soma   (0,0,0)-(20,0,0)                     5, 5                    root
    1      dend   (20,0,0)-(60,30,0)-(60,30,0)-        2, 1, 0.6, 0.6          soma, parent_x 1
                  (100,30,0)
    2      d1     (100,30,0)-(130,70,0)                0.6, 0.3                dend, parent_x 1
    3      d2     (130,-10,0)-(100,30,0)               0.3, 0.6                dend, parent_x 1, child_x 1
    4      axon   (0,0,0)-(-40,0,0)-(-40,40,0)-        proximal = distal =     soma, parent_x 0
                  (0,40,0)                             0.5, 0.8, 0.5
    =====  =====  ===================================  ======================  ==========================

    ``dend`` is 90 um: a 50 um taper from 2 to 1 um, a zero-length jump to
    0.6 um at ``x = 5/9``, then a 40 um cylinder. ``d2`` is attached by its
    ``x = 1`` end, so it is thickest there. ``from_points`` stores the
    axon's two radius discontinuities as zero-length segments (lengths 40,
    0, 40, 0, 40 um). With ``second_soma`` (tree R2), ``soma2``
    (130,70,0)-(140,70,0), radius 4 um, is attached at the end of ``d1``.
    """
    tree = Morphology.from_root(_from_points([[0, 0, 0], [20, 0, 0]], [5, 5], type="soma"), name="soma")
    with warnings.catch_warnings():
        # The repeated point is the radius jump this tree exists to test.
        warnings.filterwarnings("ignore", r"from_points\(\) produced \d+ zero-length segment", UserWarning)
        dend = _from_points(
            [[20, 0, 0], [60, 30, 0], [60, 30, 0], [100, 30, 0]], [2.0, 1.0, 0.6, 0.6], type="basal_dendrite"
        )
    tree.attach(parent="soma", child_name="dend", child_branch=dend)
    tree.attach(
        parent="dend",
        child_name="d1",
        child_branch=_from_points([[100, 30, 0], [130, 70, 0]], [0.6, 0.3], type="basal_dendrite"),
    )
    tree.attach(
        parent="dend",
        child_name="d2",
        child_x=1.0,
        child_branch=_from_points([[130, -10, 0], [100, 30, 0]], [0.3, 0.6], type="basal_dendrite"),
    )
    tree.attach(
        parent="soma",
        child_name="axon",
        parent_x=0.0,
        child_branch=_from_points(
            [[0, 0, 0], [-40, 0, 0], [-40, 40, 0], [0, 40, 0]],
            radii_proximal=[0.5, 0.8, 0.5],
            radii_distal=[0.5, 0.8, 0.5],
            type="axon",
        ),
    )
    if second_soma:
        tree.attach(
            parent="d1",
            child_name="soma2",
            child_branch=_from_points([[130, 70, 0], [140, 70, 0]], [4, 4], type="soma"),
        )
    return tree


# --------------------------------------------------------------------------- #
# Brute-force checks of the range regions
# --------------------------------------------------------------------------- #

#: A length along a branch, ``metric(morpho, branch_index, xs) -> f`` in um,
#: computed independently of the region under test.
Metric = Callable[[Morphology, int, np.ndarray], np.ndarray]

#: Samples whose ``f`` is within this many um of a bound, or whose ``x`` is
#: within this of an interval end, may fall either way: open and closed ends
#: and tangencies are decided there.
BAND = 1e-9


def radius_um(morpho: Morphology, index: int, xs: np.ndarray) -> np.ndarray:
    """The local radius (um) at each ``x``, read segment by segment from the branch arrays.

    Linear along each positive-length segment; a sample at a segment end
    belongs to the first segment that reaches it, and the last positive
    segment also takes the rounding overshoot at ``x = 1``. This does not use
    ``interpolate_branch``, which reports a jump's mean for a short window of
    ``x`` past a zero-length segment.
    """
    branch = morpho.branches[index].branch
    lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    proximal = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    distal = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
    s = np.asarray(xs, dtype=float) * lengths.sum()
    out = np.full(s.shape, np.nan)
    positive = np.flatnonzero(lengths > 0.0)
    start = 0.0
    for i in range(len(lengths)):
        if lengths[i] > 0.0:
            take = np.isnan(out) & (s >= start) & ((s <= start + lengths[i]) | (i == positive[-1]))
            fraction = np.minimum((s[take] - start) / lengths[i], 1.0)
            out[take] = proximal[i] + fraction * (distal[i] - proximal[i])
        start += lengths[i]
    assert not np.isnan(out).any(), f"branch {index}: a sample fell outside every segment"
    return out


def _in_range(f: np.ndarray, lo: float | None, hi: float | None, closed: str) -> np.ndarray:
    keep = np.ones(f.shape, dtype=bool)
    if lo is not None:
        keep &= (f >= lo) if closed in ("both", "left") else (f > lo)
    if hi is not None:
        keep &= (f <= hi) if closed in ("both", "right") else (f < hi)
    return keep


def _branch_intervals(mask: RegionMask, index: int) -> list[tuple[float, float]]:
    return [(float(p), float(d)) for b, p, d in mask.intervals if b == index]


def brute_force_mismatches(
    morpho: Morphology,
    mask: RegionMask,
    metric: Metric,
    lo: float | None,
    hi: float | None,
    closed: str,
    *,
    samples: int,
    metric_values: Mapping[int, np.ndarray] | None = None,
) -> list[str]:
    """Compare ``mask`` with ``{x : lo <= f(x) <= hi}`` on ``samples`` evenly spaced ``x`` per branch.

    Samples within :data:`BAND` of a bound in ``f``, or of an interval end in
    ``x``, are not counted. Returns one line per branch with a disagreement
    beyond that, so an empty list means the two agree. ``metric_values``
    may hold ``metric`` already evaluated on the same samples, by branch.
    """
    xs = np.linspace(0.0, 1.0, samples)
    problems = []
    for index in range(len(morpho.branches)):
        f = metric_values[index] if metric_values is not None else metric(morpho, index, xs)
        want = _in_range(f, lo, hi, closed)
        got = np.zeros(samples, dtype=bool)
        ends = [-1.0]
        for p, d in _branch_intervals(mask, index):
            got |= (xs >= p) & (xs <= d)
            ends += [p, d]
        near = np.zeros(samples, dtype=bool)
        for bound in (lo, hi):
            if bound is not None:
                near |= np.abs(f - bound) <= BAND
        near |= np.min(np.abs(xs[:, None] - np.asarray(ends)[None, :]), axis=1) <= BAND
        bad = np.flatnonzero((want != got) & ~near)
        if len(bad):
            problems.append(
                f"branch {index} ({morpho.branches[index].name}): {len(bad)} samples, "
                f"e.g. x={xs[bad[:3]].tolist()} f={f[bad[:3]].tolist()} expected {want[bad[:3]].tolist()}"
            )
    return problems


def crossing_errors(
    morpho: Morphology,
    mask: RegionMask,
    metric: Metric,
    lo: float | None,
    hi: float | None,
) -> tuple[int, list[str]]:
    """Check every interior interval end of ``mask`` is where ``f`` crosses a bound.

    An end not at ``0``, ``1`` or a segment boundary must have ``f`` within
    :data:`BAND` of the nearest bound; where ``f - bound`` changes sign over
    ``[e - 1e-4, e + 1e-4]``, 80 bisection steps must land within ``1e-12``
    of the end. Returns the number of ends checked and one line per failure.
    """
    bounds = [bound for bound in (lo, hi) if bound is not None]
    checked = 0
    problems = []
    for index in range(len(morpho.branches)):
        lengths = np.asarray(morpho.branches[index].branch.lengths.to_decimal(u.um), dtype=float)
        segment_ends = np.concatenate(([0.0], np.cumsum(lengths))) / lengths.sum()
        for end in (e for p, d in _branch_intervals(mask, index) for e in (p, d)):
            if np.min(np.abs(segment_ends - end)) <= 1e-12:
                continue
            checked += 1
            value = float(metric(morpho, index, np.array([end]))[0])
            bound = min(bounds, key=lambda b: abs(value - b)) if bounds else None
            if bound is None or abs(value - bound) > BAND:
                problems.append(f"branch {index}: end {end!r} has f = {value!r}, not at a bound {bounds}")
                continue

            def g(x: float, bound: float = bound) -> float:
                return float(metric(morpho, index, np.array([x]))[0]) - bound

            a, b = max(0.0, end - 1e-4), min(1.0, end + 1e-4)
            if g(a) * g(b) < 0.0:
                for _ in range(80):
                    mid = 0.5 * (a + b)
                    if g(a) * g(mid) <= 0.0:
                        b = mid
                    else:
                        a = mid
                if abs(0.5 * (a + b) - end) > 1e-12:
                    problems.append(f"branch {index}: end {end!r} is {abs(0.5 * (a + b) - end):.3g} from the root")
    return checked, problems


# --------------------------------------------------------------------------- #
# Selection-cache layer of the edit table
# --------------------------------------------------------------------------- #

#: Expressions a ``SelectionCache`` serves after an edit, one per thing a mask
#: can depend on: segment lengths, topology, branch type, the name -> index
#: mapping, subtree membership by index and by name, the intervals a region
#: resolves to, a fixed physical step, and the local radius. Each is a
#: composite whose operands are all hashable, because the cache memoizes
#: operands only; a top-level leaf would never be cached and could not go
#: stale.
SELECTION_BATTERY: tuple[tuple[str, object], ...] = (
    ("lengths", UniformSamples(AllRegion(), 5) + RootLocation(0.5)),
    ("topology", Terminals() + BranchPoints()),
    ("types", (branch_in("type", "basal_dendrite") | branch_in("type", "axon")) & AllRegion()),
    ("names", at("soma", 0.5) + at("axon", 1.0)),
    # Two entries, not one union: "reattach d2 under the axon" moves d2 from
    # one subtree to the other, and a union of both would not change.
    ("subtree by index", SubtreeRegion(1) | EmptyRegion()),
    ("subtree by name", SubtreeRegion("axon") | EmptyRegion()),
    # The axon survives every edit; a basal region would be empty after
    # "remove the subtree at dend", and RegionAnchors raises on that.
    ("region anchors", RegionAnchors(AllRegion(), 0.5) + RegionAnchors(branch_in("type", "axon"), 1.0)),
    # Counts its steps over the whole tree's length, so every edit that
    # changes a length changes it; a Quantity step keys the cache by value.
    ("step samples", StepSamples(AllRegion(), 25 * u.um) + RootLocation(0.5)),
    # Local radius: moves with any edit that changes a branch's radii or its
    # length; a Quantity bound keys the cache by value.
    ("radius", RadiusRangeRegion(maximum=1.5 * u.um) & AllRegion()),
)


def _selection_fingerprint(morpho: Morphology, cache: SelectionCache | None) -> dict[str, object]:
    """Evaluate every battery expression on ``morpho`` through ``cache``.

    A locset reads as its ``(branch_id, branch_x)`` rows and display names, a
    region as its ``(branch_id, prox, dist)`` intervals; both are plain
    Python numbers, so the comparison is exact.
    """
    fingerprint: dict[str, object] = {}
    for label, expr in SELECTION_BATTERY:
        mask = morpho.select(expr, cache=cache)
        if isinstance(mask, LocsetMask):
            fingerprint[label] = (mask.points, mask.display_names)
        else:
            fingerprint[label] = tuple((int(b), float(p), float(d)) for b, p, d in mask.intervals)
    return fingerprint


def assert_selection_fresh(
    testcase,
    case: EditCase,
    build: Callable[[], Morphology] = make_tree_t,
) -> None:
    """Selection-cache layer: after ``case.edit``, a warm cache answers like no cache on a cold rebuild.

    Evaluates :data:`SELECTION_BATTERY` once through one
    :class:`~braincell.filter.SelectionCache` (filling it with the operand
    masks), applies the edit, evaluates the battery again through the same
    cache and compares the result with the battery evaluated without a cache
    on ``cold_rebuild`` of the edited tree.

    Parameters
    ----------
    testcase : unittest.TestCase
        Reports the failure.
    case : EditCase
        Row of the edit table to check.
    build : callable
        Builds the tree to edit.

    Raises
    ------
    AssertionError
        Naming every battery expression whose result differs.
    """

    def warm(morpho: Morphology) -> tuple[Morphology, SelectionCache]:
        return morpho, SelectionCache()

    def observe(consumer: tuple[Morphology, SelectionCache]) -> Mapping[str, object]:
        return _selection_fingerprint(*consumer)

    def reference(cold: Morphology) -> Mapping[str, object]:
        return _selection_fingerprint(cold, None)

    assert_layer_fresh(
        testcase,
        case,
        layer="selection cache",
        warm=warm,
        observe=observe,
        reference=reference,
        build=build,
    )
