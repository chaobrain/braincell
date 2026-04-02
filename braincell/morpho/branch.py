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

"""Immutable geometry primitives for the layered architecture."""

import warnings
from dataclasses import dataclass
from typing import ClassVar

import brainunit as u
import numpy as np

from braincell._misc import normalize_param

_ALLOWED_BRANCH_TYPES = {
    "soma",
    "dendrite",
    "axon",
    "basal_dendrite",
    "apical_dendrite",
    "custom",
}

_UNSET = object()


@dataclass(frozen=True, eq=False)
class Branch:
    """Immutable geometry primitive representing an anatomical branch.

    A ``Branch`` stores segment-wise cable properties — lengths, proximal and
    distal radii, and optional 3-D point coordinates — for one contiguous
    stretch of neuronal morphology (soma, dendrite, axon, etc.).

    All geometric quantities are stored as ``brainunit`` values and normalised
    to ``u.um`` internally.  Bare numeric inputs are **rejected** with
    ``TypeError``; every value must carry explicit units.

    Branches are **frozen dataclasses**: once constructed their geometry
    cannot be changed.  Use :class:`~braincell.morpho.Morpho` to compose
    branches into a mutable morphology tree.

    Parameters
    ----------
    lengths : Quantity[u.um]
        Per-segment lengths, shape ``(n_segments,)``.  Each value must be
        ``>= 0``; the total must be ``> 0``.
    radii_proximal : Quantity[u.um]
        Proximal radius of each segment, shape ``(n_segments,)``.
        Must be ``> 0``.
    radii_distal : Quantity[u.um]
        Distal radius of each segment, shape ``(n_segments,)``.
        Must be ``> 0``.
    points_proximal : Quantity[u.um] or None
        Proximal 3-D endpoint of each segment, shape ``(n_segments, 3)``.
        Must be provided together with *points_distal* or both set to
        ``None``.
    points_distal : Quantity[u.um] or None
        Distal 3-D endpoint of each segment, shape ``(n_segments, 3)``.
        Must be provided together with *points_proximal* or both set to
        ``None``.
    type : str
        Anatomical branch type.  One of ``"soma"``, ``"dendrite"``,
        ``"axon"``, ``"basal_dendrite"``, ``"apical_dendrite"``,
        or ``"custom"`` (default).

    Raises
    ------
    TypeError
        If any geometric input is a bare number without ``brainunit`` units.
    ValueError
        If *lengths* is empty, total length is zero, array shapes are
        inconsistent, *type* is not in the allowed set, or point-derived
        lengths do not match *lengths*.

    See Also
    --------
    Branch.from_lengths : Preferred constructor from segment lengths.
    Branch.from_points : Preferred constructor from 3-D point sequences.
    Morpho : Mutable morphology tree that owns branches.

    Notes
    -----
    Prefer the factory classmethods :meth:`from_lengths` and
    :meth:`from_points` over the raw constructor.  The factories handle
    shared-radius expansion and automatically insert zero-length jump
    segments at radius discontinuities between adjacent segments.

    Each segment is modelled as a truncated cone (frustum).  Surface areas
    are computed with the lateral-surface formula
    :math:`A = \\pi (r_0 + r_1) \\sqrt{L^2 + (r_1 - r_0)^2}`
    and volumes with the frustum formula
    :math:`V = \\frac{\\pi L}{3} (r_0^2 + r_0 r_1 + r_1^2)`.

    Examples
    --------

    Create a branch from segment lengths and shared radii:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch
        >>> b = Branch.from_lengths(
        ...     lengths=[10.0, 15.0, 20.0] * u.um,
        ...     radii=[3.0, 2.5, 2.0, 1.5] * u.um,
        ...     type="dendrite",
        ... )
        >>> b.n_segments
        3
        >>> b.length
        45.0 * umetre

    Create a branch from 3-D coordinates:

    .. code-block:: python

        >>> import numpy as np
        >>> pts = np.array([[0, 0, 0], [10, 0, 0], [10, 20, 0]]) * u.um
        >>> b = Branch.from_points(
        ...     points=pts,
        ...     radii=[2.0, 1.5, 1.0] * u.um,
        ...     type="axon",
        ... )
        >>> b.points is not None
        True
    """

    lengths: u.Quantity[u.um]
    radii_proximal: u.Quantity[u.um]
    radii_distal: u.Quantity[u.um]
    points_proximal: u.Quantity[u.um] | None = None
    points_distal: u.Quantity[u.um] | None = None
    type: str = "custom"

    _BRANCH_TYPE: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        for name, kwargs in [
            ("lengths", {"unit": u.um, "shape": (None,), "bounds": {"ge": 0 * u.um}}),
            ("radii_proximal", {"unit": u.um, "shape": (None,), "bounds": {"gt": 0 * u.um}}),
            ("radii_distal", {"unit": u.um, "shape": (None,), "bounds": {"gt": 0 * u.um}}),
            ("points_proximal", {"unit": u.um, "shape": (None, 3), "allow_none": True}),
            ("points_distal", {"unit": u.um, "shape": (None, 3), "allow_none": True}),
        ]:
            object.__setattr__(self, name, normalize_param(getattr(self, name), name=name, **kwargs))

        expected_type = type(self)._BRANCH_TYPE
        if expected_type is not None and self.type != expected_type:
            raise TypeError(
                f"{type(self).__name__} requires type={expected_type!r}, got {self.type!r}. "
                f"Do not pass the 'type' parameter when constructing {type(self).__name__}."
            )
        if self.type not in _ALLOWED_BRANCH_TYPES:
            raise ValueError(f"type must be one of {sorted(_ALLOWED_BRANCH_TYPES)!r}, got {self.type!r}.")
        if self.lengths.shape[0] == 0:
            raise ValueError("Branch must contain at least one segment.")
        lengths_um = np.asarray(self.lengths.to_decimal(u.um), dtype=float)
        if float(np.sum(lengths_um)) <= 0.0:
            raise ValueError(
                "Branch total length must be > 0. "
                "All segment lengths are zero; at least one segment must have a positive length."
            )
        if (self.points_proximal is None) != (self.points_distal is None):
            raise ValueError("points_proximal and points_distal must both be provided or both be None.")

        for name in ["radii_proximal", "radii_distal", "points_proximal", "points_distal"]:
            arr = getattr(self, name)
            if arr is not None and arr.shape[0] != self.lengths.shape[0]:
                raise ValueError(f"{name} must match lengths segment count.")

        if self.points_proximal is not None and self.points_distal is not None:
            points_proximal_um = np.asarray(self.points_proximal.to_decimal(u.um), dtype=float)
            points_distal_um = np.asarray(self.points_distal.to_decimal(u.um), dtype=float)
            point_lengths = u.Quantity(np.linalg.norm(points_distal_um - points_proximal_um, axis=1), u.um)
            if not np.allclose(
                lengths_um,
                np.asarray(point_lengths.to_decimal(u.um), dtype=float),
            ):
                raise ValueError("lengths must match point-derived segment lengths.")

    @classmethod
    def from_lengths(
        cls,
        *,
        lengths: u.Quantity[u.um],
        radii: u.Quantity[u.um] | None = None,
        radii_proximal: u.Quantity[u.um] | None = None,
        radii_distal: u.Quantity[u.um] | None = None,
        type: str = _UNSET,
    ) -> "Branch":
        """Create a branch from segment lengths and radii.

        Parameters
        ----------
        lengths : Quantity[u.um]
            Per-segment lengths, shape ``(n_segments,)``.
        radii : Quantity[u.um] or None
            Shared radii at segment boundaries, shape ``(n_segments + 1,)``.
            Mutually exclusive with *radii_proximal* and *radii_distal*.
        radii_proximal : Quantity[u.um] or None
            Proximal radius of each segment, shape ``(n_segments,)``.
            Must be provided together with *radii_distal*.
        radii_distal : Quantity[u.um] or None
            Distal radius of each segment, shape ``(n_segments,)``.
            Must be provided together with *radii_proximal*.
        type : str
            Branch type (default ``"custom"``).  Not accepted by typed
            subclasses such as :class:`Soma`, :class:`Dendrite`, etc.

        Returns
        -------
        Branch
            New branch with no 3-D point geometry.

        Raises
        ------
        TypeError
            If neither *radii* nor the pair (*radii_proximal*, *radii_distal*)
            is provided, or if both are provided simultaneously.
            Also raised if *type* is passed to a typed subclass.
        ValueError
            If array shapes are inconsistent or radius discontinuities exist.

        See Also
        --------
        from_points : Create a branch from 3-D point coordinates.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch
            >>> b = Branch.from_lengths(
            ...     lengths=[10.0, 20.0] * u.um,
            ...     radii=[2.0, 1.5, 1.0] * u.um,
            ...     type="dendrite",
            ... )
            >>> b.n_segments
            2
        """
        if cls._BRANCH_TYPE is not None:
            if type is not _UNSET:
                raise TypeError(
                    f"{cls.__name__}.from_lengths() does not accept 'type'. "
                    f"The type is fixed to {cls._BRANCH_TYPE!r}."
                )
            type = cls._BRANCH_TYPE
        elif type is _UNSET:
            type = "custom"
        lengths = normalize_param(lengths, name="lengths", unit=u.um, shape=(None,), bounds={"ge": 0 * u.um})
        radii_proximal, radii_distal = cls._resolve_radius_inputs(
            "from_lengths",
            radii=radii,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            n_shared=len(lengths) + 1,
        )
        lengths, radii_proximal, radii_distal, _, _ = cls._canonicalize_segments(
            lengths=lengths,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            points_proximal=None,
            points_distal=None,
        )
        return cls(
            lengths=lengths,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            type=type,
        )

    @classmethod
    def from_points(
        cls,
        *,
        points: u.Quantity[u.um],
        radii: u.Quantity[u.um] | None = None,
        radii_proximal: u.Quantity[u.um] | None = None,
        radii_distal: u.Quantity[u.um] | None = None,
        type: str = _UNSET,
    ) -> "Branch":
        """Create a branch from ordered 3-D point coordinates.

        Segment lengths are computed automatically from consecutive point
        distances.  Warns if any pair of consecutive points coincide
        (producing zero-length segments).

        Parameters
        ----------
        points : Quantity[u.um]
            Ordered 3-D coordinates, shape ``(n_points, 3)`` with
            ``n_points >= 2``.
        radii : Quantity[u.um] or None
            Shared radii at each point, shape ``(n_points,)``.
            Mutually exclusive with *radii_proximal* and *radii_distal*.
        radii_proximal : Quantity[u.um] or None
            Proximal radius of each segment, shape ``(n_points - 1,)``.
            Must be provided together with *radii_distal*.
        radii_distal : Quantity[u.um] or None
            Distal radius of each segment, shape ``(n_points - 1,)``.
            Must be provided together with *radii_proximal*.
        type : str
            Branch type (default ``"custom"``).  Not accepted by typed
            subclasses such as :class:`Soma`, :class:`Dendrite`, etc.

        Returns
        -------
        Branch
            New branch with 3-D point geometry attached.

        Raises
        ------
        TypeError
            If neither *radii* nor the pair (*radii_proximal*, *radii_distal*)
            is provided, or if both are provided simultaneously.
            Also raised if *type* is passed to a typed subclass.
        ValueError
            If *points* has fewer than two rows or array shapes are
            inconsistent.

        Warns
        -----
        UserWarning
            If consecutive points coincide, producing zero-length segments.

        See Also
        --------
        from_lengths : Create a branch from segment lengths (no 3-D points).

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch
            >>> pts = np.array([[0, 0, 0], [10, 0, 0], [10, 20, 0]]) * u.um
            >>> b = Branch.from_points(
            ...     points=pts,
            ...     radii=[2.0, 1.5, 1.0] * u.um,
            ...     type="axon",
            ... )
            >>> b.n_segments
            2
            >>> b.points.shape
            (3, 3)
        """
        if cls._BRANCH_TYPE is not None:
            if type is not _UNSET:
                raise TypeError(
                    f"{cls.__name__}.from_points() does not accept 'type'. "
                    f"The type is fixed to {cls._BRANCH_TYPE!r}."
                )
            type = cls._BRANCH_TYPE
        elif type is _UNSET:
            type = "custom"
        points = normalize_param(points, name="points", unit=u.um, shape=(None, 3))
        if points.shape[0] < 2:
            raise ValueError("from_points() requires at least two points.")

        radii_proximal, radii_distal = cls._resolve_radius_inputs(
            "from_points",
            radii=radii,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            n_shared=points.shape[0],
        )

        points_proximal = points[:-1]
        points_distal = points[1:]
        points_um = np.asarray(points.to_decimal(u.um), dtype=float)
        lengths = u.Quantity(np.linalg.norm(points_um[1:] - points_um[:-1], axis=1), u.um)

        zero_mask = np.isclose(np.asarray(lengths.to_decimal(u.um), dtype=float), 0.0)
        if np.any(zero_mask):
            indices = np.flatnonzero(zero_mask).tolist()
            warnings.warn(
                f"from_points() produced {int(np.sum(zero_mask))} zero-length segment(s) "
                f"from coincident consecutive points at index pair(s) {indices}. "
                "These degenerate segments are kept but contribute zero volume.",
                stacklevel=2,
            )
        lengths, radii_proximal, radii_distal, points_proximal, points_distal = cls._canonicalize_segments(
            lengths=lengths,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            points_proximal=points_proximal,
            points_distal=points_distal,
        )

        return cls(
            lengths=lengths,
            radii_proximal=radii_proximal,
            radii_distal=radii_distal,
            points_proximal=points_proximal,
            points_distal=points_distal,
            type=type,
        )

    @staticmethod
    def _resolve_radius_inputs(
        method_name: str,
        *,
        radii: u.Quantity[u.um] | None,
        radii_proximal: u.Quantity[u.um] | None,
        radii_distal: u.Quantity[u.um] | None,
        n_shared: int,
    ) -> tuple[u.Quantity[u.um], u.Quantity[u.um]]:
        if radii is not None:
            if radii_proximal is not None or radii_distal is not None:
                raise TypeError(
                    "`radii` and `radii_proximal`/`radii_distal` cannot be provided together"
                )
            shared_radii = normalize_param(radii, name="radii", unit=u.um, shape=(None,))
            if len(shared_radii) != n_shared:
                raise ValueError(
                    f"{method_name}() shared `radii` must have length {n_shared}."
                )
            return shared_radii[:-1], shared_radii[1:]

        if radii_proximal is not None or radii_distal is not None:
            if radii_proximal is None or radii_distal is None:
                raise TypeError(
                    "`radii_proximal` and `radii_distal` must be provided together"
                )
            return radii_proximal, radii_distal

        raise TypeError(
            "one of `radii` or (`radii_proximal` and `radii_distal`) is required"
        )

    @staticmethod
    def _canonicalize_segments(
        *,
        lengths: u.Quantity[u.um],
        radii_proximal: u.Quantity[u.um],
        radii_distal: u.Quantity[u.um],
        points_proximal: u.Quantity[u.um] | None,
        points_distal: u.Quantity[u.um] | None,
    ) -> tuple[
        u.Quantity[u.um],
        u.Quantity[u.um],
        u.Quantity[u.um],
        u.Quantity[u.um] | None,
        u.Quantity[u.um] | None,
    ]:
        if len(lengths) <= 1:
            return lengths, radii_proximal, radii_distal, points_proximal, points_distal

        lengths_um = np.asarray(lengths.to_decimal(u.um), dtype=float)
        radii_prox_um = np.asarray(radii_proximal.to_decimal(u.um), dtype=float)
        radii_dist_um = np.asarray(radii_distal.to_decimal(u.um), dtype=float)
        points_prox_um = None if points_proximal is None else np.asarray(points_proximal.to_decimal(u.um), dtype=float)
        points_dist_um = None if points_distal is None else np.asarray(points_distal.to_decimal(u.um), dtype=float)

        discontinuous = ~np.isclose(radii_dist_um[:-1], radii_prox_um[1:])
        n_segments = len(lengths_um)
        n_jumps = int(np.count_nonzero(discontinuous))
        out_size = n_segments + n_jumps

        out_lengths = np.empty(out_size, dtype=float)
        out_radii_prox = np.empty(out_size, dtype=float)
        out_radii_dist = np.empty(out_size, dtype=float)
        out_points_prox = None if points_prox_um is None else np.empty((out_size, 3), dtype=float)
        out_points_dist = None if points_dist_um is None else np.empty((out_size, 3), dtype=float)

        out_index = 0
        for index in range(n_segments):
            out_lengths[out_index] = lengths_um[index]
            out_radii_prox[out_index] = radii_prox_um[index]
            out_radii_dist[out_index] = radii_dist_um[index]
            if out_points_prox is not None and out_points_dist is not None:
                out_points_prox[out_index] = points_prox_um[index]
                out_points_dist[out_index] = points_dist_um[index]
            out_index += 1

            if index == n_segments - 1 or not discontinuous[index]:
                continue

            out_lengths[out_index] = 0.0
            out_radii_prox[out_index] = radii_dist_um[index]
            out_radii_dist[out_index] = radii_prox_um[index + 1]
            if out_points_prox is not None and out_points_dist is not None:
                out_points_prox[out_index] = points_dist_um[index]
                out_points_dist[out_index] = points_dist_um[index]
            out_index += 1

        normalized_points_prox = None if out_points_prox is None else u.Quantity(out_points_prox, u.um)
        normalized_points_dist = None if out_points_dist is None else u.Quantity(out_points_dist, u.um)
        return (
            u.Quantity(out_lengths, u.um),
            u.Quantity(out_radii_prox, u.um),
            u.Quantity(out_radii_dist, u.um),
            normalized_points_prox,
            normalized_points_dist,
        )

    @property
    def radii(self) -> u.Quantity[u.um]:
        """Shared radii at segment boundaries when continuous.

        Returns
        -------
        Quantity[u.um]
            Shared radii array, shape ``(n_segments + 1,)``.

        Raises
        ------
        ValueError
            If any adjacent boundary has
            ``radii_distal[i] != radii_proximal[i+1]``.

        Notes
        -----
        For a single-segment branch the boundary check is vacuously satisfied
        (there are no inter-segment boundaries), so this property always
        succeeds even if the segment is tapered
        (``radii_proximal != radii_distal``).
        """

        if not u.math.allclose(self.radii_distal[:-1], self.radii_proximal[1:]):
            raise ValueError(
                "Branch radii are discontinuous across segment boundaries; use "
                "radii_proximal and radii_distal."
            )
        return u.math.concatenate((self.radii_proximal[:1], self.radii_distal), axis=0)

    @property
    def points(self) -> u.Quantity[u.um] | None:
        """Shared 3-D points at segment boundaries when continuous.

        Returns
        -------
        Quantity[u.um] or None
            Shared points array, shape ``(n_segments + 1, 3)``, or ``None``
            for branches created by :meth:`from_lengths` (no point geometry).

        Raises
        ------
        ValueError
            If any adjacent boundary has
            ``points_distal[i] != points_proximal[i+1]``.

        Notes
        -----
        For a single-segment branch the boundary check is vacuously satisfied
        (there are no inter-segment boundaries), so this property always
        succeeds regardless of whether the two endpoint coordinates coincide.
        """
        if self.points_proximal is None or self.points_distal is None:
            return None
        if not u.math.allclose(self.points_distal[:-1], self.points_proximal[1:]):
            raise ValueError(
                "Branch points are discontinuous across segment boundaries; use "
                "points_proximal and points_distal."
            )
        return u.math.concatenate((self.points_proximal[:1], self.points_distal), axis=0)

    @property
    def n_segments(self) -> int:
        """Return the number of segments in this branch.

        Returns
        -------
        int
            Segment count, always ``>= 1``.
        """
        return len(self.lengths)

    def _segment_arrays_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.lengths.to_decimal(u.um), dtype=float),
            np.asarray(self.radii_proximal.to_decimal(u.um), dtype=float),
            np.asarray(self.radii_distal.to_decimal(u.um), dtype=float),
        )

    @staticmethod
    def _quantity_allclose(lhs, rhs) -> bool:
        if lhs.shape != rhs.shape:
            return False
        return bool(u.math.allclose(lhs, rhs))

    @classmethod
    def _optional_quantity_allclose(cls, lhs, rhs) -> bool:
        if lhs is None or rhs is None:
            return lhs is None and rhs is None
        return cls._quantity_allclose(lhs, rhs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Branch):
            return NotImplemented
        return (
            self.type == other.type
            and self._quantity_allclose(self.lengths, other.lengths)
            and self._quantity_allclose(self.radii_proximal, other.radii_proximal)
            and self._quantity_allclose(self.radii_distal, other.radii_distal)
            and self._optional_quantity_allclose(self.points_proximal, other.points_proximal)
            and self._optional_quantity_allclose(self.points_distal, other.points_distal)
        )

    @property
    def length(self) -> u.Quantity[u.um]:
        """Return the total length of this branch.

        Returns
        -------
        Quantity[u.um]
            Sum of all segment lengths.
        """
        lengths_um, _, _ = self._segment_arrays_um()
        return u.Quantity(np.sum(lengths_um), u.um)

    @property
    def mean_radius(self) -> u.Quantity[u.um]:
        """Return the length-weighted mean radius of this branch.

        Each segment contributes its average radius
        ``0.5 * (r_proximal + r_distal)`` weighted by its length.

        Returns
        -------
        Quantity[u.um]
            Length-weighted average radius.
        """
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        total_length_um = float(np.sum(lengths_um))
        values_um = 0.5 * (r0_um + r1_um)
        return u.Quantity(np.sum(lengths_um * values_um) / total_length_um, u.um)

    @property
    def areas(self) -> u.Quantity[u.um ** 2]:
        """Lateral surface area of each segment (frustum formula).

        Returns
        -------
        Quantity[u.um ** 2]
            Area array, shape ``(n_segments,)``.

        Notes
        -----
        Zero-length jump segments inserted at radius discontinuities by
        ``_canonicalize_segments`` contribute a non-zero annular end-cap area
        :math:`\\pi (r_0 + r_1) |r_1 - r_0|`.  This is geometrically correct
        but may be unexpected when iterating segment areas individually.
        """
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        values = np.pi * (r0_um + r1_um) * np.sqrt(lengths_um * lengths_um + (r1_um - r0_um) * (r1_um - r0_um))
        return u.Quantity(values, u.um ** 2)

    @property
    def area(self) -> u.Quantity[u.um ** 2]:
        """Total lateral surface area (sum of segment areas).

        Returns
        -------
        Quantity[u.um ** 2]
            Total surface area.
        """
        return self.areas.sum()

    @property
    def volumes(self) -> u.Quantity[u.um ** 3]:
        """Per-segment volumes (frustum formula).

        Returns
        -------
        Quantity[u.um ** 3]
            Volume array, shape ``(n_segments,)``.
        """
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        values = np.pi * lengths_um * (r0_um * r0_um + r0_um * r1_um + r1_um * r1_um) / 3.0
        return u.Quantity(values, u.um ** 3)

    @property
    def volume(self) -> u.Quantity[u.um ** 3]:
        """Total volume (sum of segment volumes).

        Returns
        -------
        Quantity[u.um ** 3]
            Total volume.
        """
        return self.volumes.sum()

    def vis2d(
        self,
        *,
        mode: str = "frustum",
        backend: str | None = None,
        chooser=None,
        projection_plane: str = "xy",
        return_plotter: bool = False,
        show: bool = True,
    ) -> object:
        """Visualize this branch in 2D.

        Creates a temporary single-branch morphology and renders it using
        the existing 2D visualization pipeline.

        Parameters
        ----------
        mode : str
            Visualization mode: ``"frustum"`` (default, shows segment widths
            as polygons), ``"tree"`` (schematic polyline), or ``"projected"``
            (3-D point projection, requires point geometry).
        backend : str or None
            Rendering backend name (e.g., ``"matplotlib"``).
            Auto-selected when *None*.
        chooser : BackendChooser or None
            Explicit backend chooser; overrides *backend* when given.
        projection_plane : str
            Projection plane for ``"projected"`` mode: ``"xy"``, ``"xz"``,
            or ``"yz"`` (default ``"xy"``).
        return_plotter : bool
            If *True*, return the backend plotter/axes object instead of
            displaying the figure.
        show : bool
            If *True* (default), call ``matplotlib.pyplot.show()`` after
            rendering. Set to *False* to suppress display (e.g., when
            embedding in a larger figure or running in tests).

        Returns
        -------
        object
            The backend plotter/axes when *return_plotter* is True;
            otherwise the backend's default display result.

        Raises
        ------
        ValueError
            If *mode* is ``"projected"`` and the branch has no 3-D point
            geometry.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch
            >>> branch = Branch.from_lengths(
            ...     lengths=[10.0, 15.0, 20.0] * u.um,
            ...     radii=[3.0, 2.5, 2.0, 1.5] * u.um,
            ...     type="dendrite",
            ... )
            >>> branch.vis2d()  # doctest: +SKIP
        """
        from braincell.morpho import Morpho
        from braincell.vis import plot2d

        morpho = Morpho.from_root(self, name="soma")
        result = plot2d(
            morpho,
            mode=mode,
            backend=backend,
            chooser=chooser,
            projection_plane=projection_plane,
            return_plotter=return_plotter,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return result

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(type={self.type!r}, n_segments={self.n_segments!r}, "
            f"length={self.length!r}, area={self.area!r})"
        )


@dataclass(frozen=True, eq=False, repr=False)
class Soma(Branch):
    """A :class:`Branch` with type fixed to ``"soma"``.

    Represents the cell body (soma) of a neuron. All factory methods
    and the constructor automatically set ``type="soma"``; passing
    ``type`` explicitly raises :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "soma"
    type: str = "soma"


@dataclass(frozen=True, eq=False, repr=False)
class Dendrite(Branch):
    """A :class:`Branch` with type fixed to ``"dendrite"``.

    Represents a generic dendrite segment. All factory methods
    and the constructor automatically set ``type="dendrite"``; passing
    ``type`` explicitly raises :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "dendrite"
    type: str = "dendrite"


@dataclass(frozen=True, eq=False, repr=False)
class Axon(Branch):
    """A :class:`Branch` with type fixed to ``"axon"``.

    Represents an axon segment. All factory methods and the constructor
    automatically set ``type="axon"``; passing ``type`` explicitly
    raises :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "axon"
    type: str = "axon"


@dataclass(frozen=True, eq=False, repr=False)
class BasalDendrite(Branch):
    """A :class:`Branch` with type fixed to ``"basal_dendrite"``.

    Represents a basal dendrite segment. All factory methods and the
    constructor automatically set ``type="basal_dendrite"``; passing
    ``type`` explicitly raises :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "basal_dendrite"
    type: str = "basal_dendrite"


@dataclass(frozen=True, eq=False, repr=False)
class ApicalDendrite(Branch):
    """A :class:`Branch` with type fixed to ``"apical_dendrite"``.

    Represents an apical dendrite segment. All factory methods and the
    constructor automatically set ``type="apical_dendrite"``; passing
    ``type`` explicitly raises :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "apical_dendrite"
    type: str = "apical_dendrite"


@dataclass(frozen=True, eq=False, repr=False)
class CustomBranch(Branch):
    """A :class:`Branch` with type fixed to ``"custom"``.

    Represents a branch with unspecified anatomical identity. All
    factory methods and the constructor automatically set
    ``type="custom"``; passing ``type`` explicitly raises
    :class:`TypeError`.

    See Also
    --------
    Branch : Base class with full documentation.
    """

    _BRANCH_TYPE: ClassVar[str] = "custom"
    type: str = "custom"


_BRANCH_TYPE_TO_CLASS: dict[str, type[Branch]] = {
    "soma": Soma,
    "dendrite": Dendrite,
    "axon": Axon,
    "basal_dendrite": BasalDendrite,
    "apical_dendrite": ApicalDendrite,
    "custom": CustomBranch,
}


def branch_class_for_type(branch_type: str) -> type[Branch]:
    """Return the :class:`Branch` subclass for the given type string.

    Parameters
    ----------
    branch_type : str
        One of the allowed branch type strings (e.g. ``"soma"``,
        ``"dendrite"``, ``"axon"``).

    Returns
    -------
    type[Branch]
        The corresponding subclass.

    Raises
    ------
    ValueError
        If *branch_type* is not a recognised branch type.
    """
    cls = _BRANCH_TYPE_TO_CLASS.get(branch_type)
    if cls is None:
        raise ValueError(
            f"Unknown branch type {branch_type!r}. "
            f"Allowed types: {sorted(_BRANCH_TYPE_TO_CLASS)!r}."
        )
    return cls
