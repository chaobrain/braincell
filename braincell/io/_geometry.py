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

"""Format-independent geometry shared by the morphology readers.

Everything here is pure ``numpy``: no file paths, no report objects, no
reader state. A *contour* is an ``(N, 3)`` float array of xyz coordinates
in micrometres, and a *stack* is a tuple of such contours ordered along
the soma axis. Diagnostics stay in the readers, which own the wording and
the source location; this module only answers geometric questions.

Two things live here because both the SWC and the ASC reader need them and
had drifted into private copies:

* :data:`MIN_SYNTHETIC_LENGTH_UM` and :func:`synthetic_soma_geometry`,
  the three-point stand-in soma both readers synthesize when a file has no
  usable soma;
* :func:`should_copy_attach_point`, the branch-boundary rule that decides
  whether a child branch repeats its parent's terminal point.

The contour math is currently exercised only by the ASC reader — ASC is
the only format that stores a soma as traced outlines — but it is
reader-agnostic and tested on its own here rather than through a parser.
"""

import numpy as np

__all__ = [
    "MIN_SYNTHETIC_LENGTH_UM",
    "approximate_contour_by_circle",
    "bbox_xy",
    "bboxes_intersect_xy",
    "constant_z",
    "contour_center",
    "contour_stack_center",
    "contour_stack_to_centroid",
    "contour_to_centroid",
    "group_contour_stacks",
    "loose_bbox_xy",
    "point_inside_bbox_xy",
    "principal_axis_sampling",
    "should_copy_attach_point",
    "synthetic_soma_geometry",
]

MIN_SYNTHETIC_LENGTH_UM = 1e-6
"""Floor applied to a synthesized soma radius, in micrometres.

A reconstruction may record a zero radius for the point a stand-in soma is
built from. Without a floor the resulting branch would have zero length
and be rejected downstream.
"""

#: Padding added around a single-contour soma bounding box, in micrometres.
_SINGLE_CONTOUR_BBOX_PAD_UM = 0.5


# ---------------------------------------------------------------------------
# Branch-boundary geometry
# ---------------------------------------------------------------------------


def should_copy_attach_point(
    *,
    allow_copy: bool,
    same_xyz: bool,
    same_radius: bool,
    keep_radius_jump: bool,
) -> bool:
    """Decide whether a child branch repeats its parent's terminal point.

    Parameters
    ----------
    allow_copy : bool
        Format-level permission, already decided by the caller. ``False``
        vetoes the copy outright — the SWC reader uses it for NEURON-style
        soma midpoint attachments, the ASC reader for the same case.
    same_xyz : bool
        Whether the child's first point already sits on the attachment
        coordinate.
    same_radius : bool
        Whether the child's first radius already equals the radius the
        copied point would carry.
    keep_radius_jump : bool
        Policy switch for the ``same_xyz`` case, and the one place the two
        readers genuinely disagree. See Notes.

    Returns
    -------
    bool
        ``True`` when the attachment point should be inserted ahead of the
        child's own points.

    Notes
    -----
    The readers do **not** share a ``keep_radius_jump`` value, and this
    function deliberately does not pick a winner:

    * The SWC reader passes ``True``. A child whose first point coincides
      with the attachment but carries a different radius still gets the
      copied point, so the branch boundary keeps its radius jump as a
      zero-length first segment. That behaviour is an invariant recorded in
      ``docs/design/io-swc-reader-invariants.md``.
    * The ASC reader passes ``False``. Neurolucida traces repeat the parent
      terminal coordinate routinely, and NEURON's ``read_nlcda3.hoc``
      suppresses the duplicate on coincident xyz regardless of diameter.

    Whether the ASC side should adopt the SWC rule is a parity question
    about NEURON, not a refactoring decision.
    """
    if not allow_copy:
        return False
    if not same_xyz:
        return True
    return keep_radius_jump and not same_radius


def synthetic_soma_geometry(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the three-point stand-in soma used when a file has none.

    Parameters
    ----------
    center : numpy.ndarray
        Soma centre, shape ``(3,)``, in micrometres.
    radius : float
        Soma radius in micrometres. Callers are expected to have clamped
        it with :data:`MIN_SYNTHETIC_LENGTH_UM` already.

    Returns
    -------
    points : numpy.ndarray
        Shape ``(3, 3)``: the centre offset by ``radius`` along -x, the
        centre, and the centre offset by ``radius`` along +x.
    radii : numpy.ndarray
        Shape ``(3,)``, constant ``radius``.

    Notes
    -----
    The cylinder is laid out along x purely by convention; NEURON's own
    ``Import3d`` stand-in soma does the same, and nothing downstream reads
    meaning into the orientation.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> points, radii = synthetic_soma_geometry(np.zeros(3), 2.0)
        >>> points[:, 0]
        array([-2.,  0.,  2.])
    """
    center = np.asarray(center, dtype=float)
    offset = np.array([float(radius), 0.0, 0.0], dtype=float)
    points = np.array((center - offset, center, center + offset), dtype=float)
    radii = np.full(3, float(radius), dtype=float)
    return points, radii


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def bbox_xy(contour: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` of one contour.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates.

    Returns
    -------
    tuple of float
        The axis-aligned bounding box, ignoring z.
    """
    contour = np.asarray(contour, dtype=float)
    return (
        float(contour[:, 0].min()),
        float(contour[:, 0].max()),
        float(contour[:, 1].min()),
        float(contour[:, 1].max()),
    )


def bboxes_intersect_xy(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Return whether two xy bounding boxes overlap, touching included.

    Parameters
    ----------
    left, right : tuple of float
        Boxes as returned by :func:`bbox_xy`.

    Returns
    -------
    bool
    """
    xmin1, xmax1, ymin1, ymax1 = left
    xmin2, xmax2, ymin2, ymax2 = right
    return not (xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1)


def loose_bbox_xy(stack: tuple[np.ndarray, ...]) -> tuple[float, float, float, float]:
    """Return the xy box a neurite root must fall inside to count as attached.

    Parameters
    ----------
    stack : tuple of numpy.ndarray
        One soma contour stack.

    Returns
    -------
    tuple of float
        ``(xmin, xmax, ymin, ymax)``. A single contour is padded by
        ``0.5 um`` on every side, matching NEURON; a real stack is used as
        traced, since its layers already spread the box out.
    """
    if len(stack) == 1:
        xmin, xmax, ymin, ymax = bbox_xy(stack[0])
        pad = _SINGLE_CONTOUR_BBOX_PAD_UM
        return xmin - pad, xmax + pad, ymin - pad, ymax + pad

    boxes = [bbox_xy(contour) for contour in stack]
    return (
        min(box[0] for box in boxes),
        max(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def point_inside_bbox_xy(xyz: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """Return whether ``xyz`` falls inside an xy bounding box.

    Parameters
    ----------
    xyz : numpy.ndarray
        A single point, shape ``(3,)``. Its z coordinate is ignored.
    box : tuple of float
        ``(xmin, xmax, ymin, ymax)``.

    Returns
    -------
    bool
    """
    xmin, xmax, ymin, ymax = box
    return xmin <= float(xyz[0]) <= xmax and ymin <= float(xyz[1]) <= ymax


# ---------------------------------------------------------------------------
# Contour stacks
# ---------------------------------------------------------------------------


def group_contour_stacks(contours: tuple[np.ndarray, ...]) -> tuple[tuple[np.ndarray, ...], ...]:
    """Group consecutive contours into soma stacks by xy overlap.

    Parameters
    ----------
    contours : tuple of numpy.ndarray
        Contours in file order.

    Returns
    -------
    tuple of tuple of numpy.ndarray
        One entry per stack. Two adjacent contours join the same stack when
        their xy boxes overlap; a gap starts a new stack. More than one
        stack means the file traced several disjoint cell bodies.
    """
    if not contours:
        return ()

    stacks: list[tuple[np.ndarray, ...]] = []
    current: list[np.ndarray] = [contours[0]]
    previous_box = bbox_xy(contours[0])
    for contour in contours[1:]:
        box = bbox_xy(contour)
        if bboxes_intersect_xy(previous_box, box):
            current.append(contour)
        else:
            stacks.append(tuple(current))
            current = [contour]
        previous_box = box
    stacks.append(tuple(current))
    return tuple(stacks)


def constant_z(contour: np.ndarray, *, tol: float = 1e-6) -> float | None:
    """Return the shared z of a planar contour, or ``None`` if it has none.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates.
    tol : float, optional
        Absolute tolerance on the z spread.

    Returns
    -------
    float or None
        The first point's z when every point matches it within ``tol``,
        otherwise ``None``. Callers raise; the wording belongs to whichever
        format is being read.
    """
    contour = np.asarray(contour, dtype=float)
    z0 = float(contour[0, 2])
    if np.any(np.abs(contour[1:, 2] - z0) > tol):
        return None
    return z0


def contour_center(contour: np.ndarray, *, num: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Resample a contour by arc length and return its centroid.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates.
    num : int, optional
        Number of arc-length samples.

    Returns
    -------
    center : numpy.ndarray
        Shape ``(3,)`` mean of the resampled points.
    resampled : numpy.ndarray
        Shape ``(num, 3)`` uniformly spaced samples along the traced path.

    Notes
    -----
    Uniform *arc-length* resampling is what makes the centroid independent
    of how densely the tracer clicked along each part of the outline; a
    plain mean over the raw points would be pulled towards dense regions.
    """
    contour = np.asarray(contour, dtype=float)
    x, y, z = contour[:, 0], contour[:, 1], contour[:, 2]
    segment_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    perimeter = np.zeros(len(contour), dtype=float)
    perimeter[1:] = np.cumsum(segment_lengths)
    uniform = np.linspace(0.0, perimeter[-1], num)
    x_new = np.interp(uniform, perimeter, x)
    y_new = np.interp(uniform, perimeter, y)
    z_new = np.interp(uniform, perimeter, z)
    center = np.array([x_new.mean(), y_new.mean(), z_new.mean()], dtype=float)
    return center, np.stack([x_new, y_new, z_new], axis=1)


def _rotate(values: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([values[k:], values[:k]])


def _keep_strictly_monotonic(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    increasing: bool,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    keep_indices = [0]
    for index in range(1, len(x_values)):
        if increasing:
            if x_values[index] > x_values[keep_indices[-1]] + tol:
                keep_indices.append(index)
        elif x_values[index] < x_values[keep_indices[-1]] - tol:
            keep_indices.append(index)
    keep = np.asarray(keep_indices, dtype=int)
    return x_values[keep], y_values[keep]


def _interp_strict(xp: np.ndarray, fp: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    if len(xp) == 1:
        return np.full_like(x_values, fp[0], dtype=float)
    if xp[0] > xp[-1]:
        xp = xp[::-1]
        fp = fp[::-1]
    return np.interp(x_values, xp, fp)


def principal_axis_sampling(
    contour: np.ndarray,
    *,
    n_samples: int = 21,
    arclength_resample: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a closed outline into a cable sampled along its major axis.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates of one closed contour.
    n_samples : int, optional
        Number of points along the major axis.
    arclength_resample : int, optional
        Resolution of the arc-length resampling done first.

    Returns
    -------
    xy : numpy.ndarray
        Shape ``(n_samples, 2)`` positions along the major axis.
    diameters : numpy.ndarray
        Shape ``(n_samples,)`` outline width measured across the minor
        axis at each position.

    Notes
    -----
    This reproduces NEURON's 21-point principal-axis conversion of a single
    ``CellBody`` contour.

    The sign of an eigenvector is arbitrary, so the major axis needs a
    deterministic orientation convention. Making the dominant component
    positive is sufficient to reproduce NEURON's pt3d ordering.

    Two earlier attempts at this are deliberately gone:

    1. Aligning against NEURON's own eigenvector (via ``Import3d_Section``).
       That imported ``neuron`` at runtime from inside the reader — NEURON
       is a dev-only comparator, not a runtime dependency — and it was
       wrong anyway: NEURON reverses its raw eigenvector downstream, so
       aligning to the raw vector inverts the result. ``GrC.asc`` came out
       exactly reversed, point for point.
    2. A follow-up heuristic that flipped the axis when the contour's first
       point projected positive and its last projected negative. On a
       closed contour those two points are adjacent, and when they straddle
       the centre their projections are near-exact negatives (``GrC.asc``:
       ``+0.18209`` and ``-0.18209``), so the test fires on numerical noise
       rather than on traversal direction.
    """
    mean, resampled = contour_center(contour, num=arclength_resample)
    mean_xy = mean[:2]

    pts = np.ascontiguousarray(resampled[:, :2])
    cov = np.cov(pts - mean_xy, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]
    minor = eigvecs[:, 0]
    if major[np.argmax(np.abs(major))] < 0.0:
        major = -major
    major = major / np.linalg.norm(major)
    minor = minor / np.linalg.norm(minor)

    d = (pts - mean_xy) @ major
    rad = (pts - mean_xy) @ minor

    index_max = int(np.argmax(d))
    index_min = int(np.argmin(d))
    d_rot = _rotate(d, index_max)
    rad_rot = _rotate(rad, index_max)
    index_min_rot = int(np.where(d_rot == d[index_min])[0][0])

    d_side1 = d_rot[:index_min_rot][::-1]
    rad_side1 = rad_rot[:index_min_rot][::-1]
    d_side2 = d_rot[index_min_rot:]
    rad_side2 = rad_rot[index_min_rot:]

    inc1 = len(d_side1) > 1 and bool(d_side1[1] > d_side1[0])
    inc2 = len(d_side2) > 1 and bool(d_side2[1] > d_side2[0])
    d_side1_new, rad_side1_new = _keep_strictly_monotonic(d_side1, rad_side1, increasing=inc1)
    d_side2_new, rad_side2_new = _keep_strictly_monotonic(d_side2, rad_side2, increasing=inc2)

    d_all_sorted = np.sort(np.concatenate([d_side1_new, d_side2_new]))
    d_interp = np.linspace(float(d_all_sorted[1]), float(d_all_sorted[-2]), n_samples)
    xy_interp = mean_xy[None, :] + d_interp[:, None] * major[None, :]

    rad1_interp = _interp_strict(d_side1_new, rad_side1_new, d_interp)
    rad2_interp = _interp_strict(d_side2_new, rad_side2_new, d_interp)
    diam_interp = np.abs(rad1_interp - rad2_interp)
    diam_interp[0] = 0.5 * (diam_interp[0] + diam_interp[1])
    diam_interp[-1] = 0.5 * (diam_interp[-1] + diam_interp[-2])
    return xy_interp, diam_interp


def contour_to_centroid(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a single contour into a 21-point soma cable.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates.

    Returns
    -------
    points : numpy.ndarray
        Shape ``(21, 3)`` cable points, held at the contour's own z.
    radii : numpy.ndarray
        Shape ``(21,)`` radii, half the sampled diameters.
    center : numpy.ndarray
        Shape ``(3,)`` arc-length centroid of the contour.
    """
    xy, diameters = principal_axis_sampling(contour, n_samples=21)
    center, _ = contour_center(contour)
    contour = np.asarray(contour, dtype=float)
    z_value = float(contour[0, 2]) if len(contour) else 0.0
    points = np.column_stack([xy, np.full(len(diameters), z_value, dtype=float)])
    return points, 0.5 * np.asarray(diameters, dtype=float), center


def approximate_contour_by_circle(contour: np.ndarray, *, num: int = 101) -> tuple[np.ndarray, float]:
    """Reduce one contour of a stack to a centre and an effective diameter.

    Parameters
    ----------
    contour : numpy.ndarray
        Shape ``(N, 3)`` xyz coordinates.
    num : int, optional
        Arc-length resampling resolution.

    Returns
    -------
    center : numpy.ndarray
        Shape ``(3,)``.
    diameter : float
        NEURON's stack estimate: the mean radial distance plus the closed
        perimeter divided by ``2*pi``.
    """
    center, resampled = contour_center(contour, num=num)
    contour = np.asarray(contour, dtype=float)
    perimeter = float(np.sum(np.linalg.norm(np.roll(contour, -1, axis=0) - contour, axis=1)))
    mean_radius = float(np.mean(np.linalg.norm(resampled - center[None, :], axis=1)))
    return center, mean_radius + perimeter / (2.0 * np.pi)


def contour_stack_to_centroid(stack: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a contour stack into a cable, one point per traced layer.

    Parameters
    ----------
    stack : tuple of numpy.ndarray
        Contours ordered along the soma axis.

    Returns
    -------
    points : numpy.ndarray
        Shape ``(len(stack), 3)`` layer centres.
    radii : numpy.ndarray
        Shape ``(len(stack),)`` half the per-layer effective diameters.
    """
    centers = []
    radii = []
    for contour in stack:
        center, diameter = approximate_contour_by_circle(contour)
        centers.append(center)
        radii.append(0.5 * float(diameter))
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def contour_stack_center(stack: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the point halfway along a contour stack, by path length.

    Parameters
    ----------
    stack : tuple of numpy.ndarray
        Contours ordered along the soma axis.

    Returns
    -------
    numpy.ndarray
        Shape ``(3,)``. This is the attachment anchor neurites connect to,
        so it follows the traced path rather than averaging the layer
        centres — a curved stack's average can sit off the soma entirely.
    """
    centers = np.asarray([contour_center(contour)[0] for contour in stack], dtype=float)
    if len(centers) == 1:
        return centers[0]

    cumulative = [0.0]
    total = 0.0
    for index in range(1, len(centers)):
        total += float(np.linalg.norm(centers[index] - centers[index - 1]))
        cumulative.append(total)

    if total <= 0.0:
        return centers[0]

    target = 0.5 * total
    for index in range(1, len(cumulative)):
        if cumulative[index] > target:
            fraction = (target - cumulative[index - 1]) / (cumulative[index] - cumulative[index - 1])
            return fraction * centers[index] + (1.0 - fraction) * centers[index - 1]
    return centers[-1]
