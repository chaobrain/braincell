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

import numpy as np

import brainunit as u
from braincell._units import normalize_param

_ALLOWED_BRANCH_TYPES = {
    "soma",
    "dend",
    "axon",
    "basal_dend",
    "basal_dendrite",
    "apical_dend",
    "apical_dendrite",
    "custom",
}


@dataclass(frozen=True, eq=False)
class Branch:
    """An anatomical branch geometry with segment-wise cable properties.

    All geometric quantities are stored as `brainunit` values and normalized to
    `u.um` internally. Callers must supply explicit-unit quantities; bare
    numeric inputs are rejected.
    """

    lengths: u.Quantity[u.um]
    radii_proximal: u.Quantity[u.um]
    radii_distal: u.Quantity[u.um]
    points_proximal: u.Quantity[u.um] | None = None
    points_distal: u.Quantity[u.um] | None = None
    type: str = "custom"

    def __post_init__(self) -> None:
        for name, kwargs in [
            ("lengths", {"unit": u.um, "shape": (None,), "bounds": {"ge": 0 * u.um}}),
            ("radii_proximal", {"unit": u.um, "shape": (None,), "bounds": {"gt": 0 * u.um}}),
            ("radii_distal", {"unit": u.um, "shape": (None,), "bounds": {"gt": 0 * u.um}}),
            ("points_proximal", {"unit": u.um, "shape": (None, 3), "allow_none": True}),
            ("points_distal", {"unit": u.um, "shape": (None, 3), "allow_none": True}),
        ]:
            object.__setattr__(self, name, normalize_param(getattr(self, name), name=name, **kwargs))

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
        type: str = "custom",
    ) -> "Branch":
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
        type: str = "custom",
    ) -> "Branch":
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
        """Return shared radii ``[n_segments + 1]`` when segment boundaries are continuous.

        Raises ``ValueError`` if any adjacent boundary has ``radii_distal[i] != radii_proximal[i+1]``.

        Note: for a single-segment branch the boundary check is vacuously satisfied
        (there are no inter-segment boundaries), so this property always succeeds
        even if the segment is tapered (``radii_proximal != radii_distal``).
        """

        if not u.math.allclose(self.radii_distal[:-1], self.radii_proximal[1:]):
            raise ValueError(
                "Branch radii are discontinuous across segment boundaries; use "
                "radii_proximal and radii_distal."
            )
        return u.math.concatenate((self.radii_proximal[:1], self.radii_distal), axis=0)

    @property
    def points(self) -> u.Quantity[u.um] | None:
        """Return shared points ``[n_segments + 1, 3]`` when point geometry is continuous.

        Returns ``None`` for branches created by ``from_lengths(...)`` (no point geometry).
        Raises ``ValueError`` if any adjacent boundary has ``points_distal[i] != points_proximal[i+1]``.

        Note: for a single-segment branch the boundary check is vacuously satisfied
        (there are no inter-segment boundaries), so this property always succeeds
        regardless of whether the two endpoint coordinates coincide.
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
        lengths_um, _, _ = self._segment_arrays_um()
        return u.Quantity(np.sum(lengths_um), u.um)

    @property
    def mean_radius(self) -> u.Quantity[u.um]:
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        total_length_um = float(np.sum(lengths_um))
        values_um = 0.5 * (r0_um + r1_um)
        return u.Quantity(np.sum(lengths_um * values_um) / total_length_um, u.um)

    @property
    def areas(self) -> u.Quantity[u.um ** 2]:
        """Lateral surface area of each segment (frustum formula).

        Zero-length jump segments inserted at radius discontinuities by
        `_canonicalize_segments` contribute a non-zero annular end-cap area
        ``π(r0 + r1)|r1 - r0|``. This is geometrically correct but may be
        unexpected when iterating segment areas individually.
        """
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        values = np.pi * (r0_um + r1_um) * np.sqrt(lengths_um * lengths_um + (r1_um - r0_um) * (r1_um - r0_um))
        return u.Quantity(values, u.um ** 2)

    @property
    def area(self) -> u.Quantity[u.um ** 2]:
        return self.areas.sum()

    @property
    def volumes(self) -> u.Quantity[u.um ** 3]:
        lengths_um, r0_um, r1_um = self._segment_arrays_um()
        values = np.pi * lengths_um * (r0_um * r0_um + r0_um * r1_um + r1_um * r1_um) / 3.0
        return u.Quantity(values, u.um ** 3)

    @property
    def volume(self) -> u.Quantity[u.um ** 3]:
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
            ...     type="dend",
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
            f"Branch(type={self.type!r}, n_segments={self.n_segments!r}, "
            f"length={self.length!r}, area={self.area!r})"
        )
