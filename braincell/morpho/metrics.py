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



from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainunit as u
import numpy as np
from brainunit import Quantity

if TYPE_CHECKING:
    from .morpho import Morpho


@dataclass(frozen=True)
class MorphMetrics:
    """Compute geometric and topological metrics on a morphology tree.

    ``MorphMetrics`` is a frozen dataclass that wraps a
    :class:`~braincell.morpho.Morpho` instance and exposes
    whole-morphology scalar measurements as properties.  It is typically
    accessed through :attr:`Morpho.metric` rather than constructed
    directly.

    All length-based quantities are returned as ``brainunit`` values in
    micrometres (``u.um``).

    Parameters
    ----------
    morpho : Morpho
        The morphology tree to measure.

    See Also
    --------
    Morpho.metric : Convenience accessor that creates a ``MorphMetrics``.
    Morpho.summary : Dictionary of key metrics for quick inspection.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morpho
        >>> soma = Branch.from_lengths(
        ...     lengths=[20.0] * u.um,
        ...     radii=[10.0, 10.0] * u.um,
        ...     type="soma",
        ... )
        >>> dend = Branch.from_lengths(
        ...     lengths=[50.0, 40.0] * u.um,
        ...     radii=[2.0, 1.5, 1.0] * u.um,
        ...     type="dend",
        ... )
        >>> morpho = Morpho.from_root(soma, name="soma")
        >>> morpho.soma.dend = dend
        >>> m = morpho.metric
        >>> m.n_branches
        2
        >>> m.n_stems
        1
    """

    morpho: 'Morpho'

    def _all_segment_arrays_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = []
        radii_proximal = []
        radii_distal = []
        for branch in self.morpho.branches:
            lengths_um, r0_um, r1_um = branch.branch._segment_arrays_um()
            lengths.append(lengths_um)
            radii_proximal.append(r0_um)
            radii_distal.append(r1_um)
        return (
            np.concatenate(lengths),
            np.concatenate(radii_proximal),
            np.concatenate(radii_distal),
        )

    def _branch_length_um(self, branch_index: int) -> float:
        branch = self.morpho.branch(index=branch_index).branch
        return float(np.sum(np.asarray(branch.lengths.to_decimal(u.um), dtype=float)))

    def _root_branch_attach_x(self) -> float:
        return 0.0

    def _root_point_um(self) -> np.ndarray:
        root = self.morpho.root.branch
        if root.points_proximal is None:
            raise ValueError("Euclidean distance metrics require point geometry on the root branch.")
        return np.asarray(root.points_proximal.to_decimal(u.um), dtype=float)[0]

    def _terminal_branch_indices(self) -> tuple[int, ...]:
        return tuple(branch.index for branch in self.morpho.branches if branch.n_children == 0)

    def _root_attach_distances_um(self) -> dict[int, float]:
        ordered_ids = self.morpho._ordered_node_ids_by("depth")
        distances_um: dict[int, float] = {self.morpho.root._node_id: 0.0}
        attach_x: dict[int, float] = {self.morpho.root._node_id: self._root_branch_attach_x()}

        for node_id in ordered_ids:
            if node_id == self.morpho.root._node_id:
                continue

            node = self.morpho._get_node(node_id)
            parent_id = node.parent_id
            if parent_id is None:
                continue

            parent_attach_x = attach_x[parent_id]
            parent_distance_um = distances_um[parent_id]
            parent_length_um = self._branch_length_um(self.morpho._branch_index(parent_id))

            distances_um[node_id] = parent_distance_um + abs(float(node.parent_x) - parent_attach_x) * parent_length_um
            attach_x[node_id] = float(node.child_x)

        return distances_um

    @property
    def total_length(self) -> Quantity:
        """Total length of all segments across the entire morphology.

        Returns
        -------
        Quantity[u.um]
            Sum of all segment lengths.
        """
        lengths_um, _, _ = self._all_segment_arrays_um()
        return u.Quantity(np.sum(lengths_um), u.um)

    @property
    def mean_radius(self) -> Quantity:
        """Length-weighted mean radius across the entire morphology.

        Returns
        -------
        Quantity[u.um]
            Average radius weighted by segment length.

        Raises
        ------
        ValueError
            If total morphology length is zero.
        """
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        total_length_um = float(np.sum(lengths_um))
        if total_length_um <= 0.0:
            raise ValueError("Morphology total length must be > 0.")
        values_um = 0.5 * (r0_um + r1_um)
        return u.Quantity(np.sum(lengths_um * values_um) / total_length_um, u.um)

    @property
    def total_area(self) -> Quantity:
        """Total lateral surface area across the entire morphology.

        Uses the frustum formula per segment.

        Returns
        -------
        Quantity[u.um ** 2]
            Total surface area.
        """
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        value = np.sum(np.pi * (r0_um + r1_um) * np.sqrt(lengths_um * lengths_um + (r1_um - r0_um) * (r1_um - r0_um)))
        return u.Quantity(value, u.um ** 2)

    @property
    def total_volume(self) -> Quantity:
        """Total volume across the entire morphology.

        Uses the frustum formula per segment.

        Returns
        -------
        Quantity[u.um ** 3]
            Total volume.
        """
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        value = np.sum(np.pi * lengths_um * (r0_um * r0_um + r0_um * r1_um + r1_um * r1_um) / 3.0)
        return u.Quantity(value, u.um ** 3)

    @property
    def n_branches(self) -> int:
        """Total number of branches in the morphology.

        Returns
        -------
        int
            Branch count (including root).
        """
        return len(self.morpho.branches)

    @property
    def n_stems(self) -> int:
        """Number of stem branches (direct children of the root).

        Returns
        -------
        int
            Count of root's children.
        """
        return self.morpho.root.n_children

    @property
    def n_bifurcations(self) -> int:
        """Number of bifurcation points in the morphology.

        A branch counts as a bifurcation if it has two or more children.

        Returns
        -------
        int
            Count of branches with ``n_children >= 2``.
        """
        return sum(branch.n_children >= 2 for branch in self.morpho.branches)

    @property
    def x_range(self) -> Quantity:
        """Extent of the morphology along the x-axis.

        Returns
        -------
        Quantity[u.um]
            ``max(x) - min(x)`` across all branch points.

        Raises
        ------
        ValueError
            If the morphology has no point geometry.
        """
        return self._axis_range(axis=0)

    @property
    def y_range(self) -> Quantity:
        """Extent of the morphology along the y-axis.

        Returns
        -------
        Quantity[u.um]
            ``max(y) - min(y)`` across all branch points.

        Raises
        ------
        ValueError
            If the morphology has no point geometry.
        """
        return self._axis_range(axis=1)

    @property
    def z_range(self) -> Quantity:
        """Extent of the morphology along the z-axis.

        Returns
        -------
        Quantity[u.um]
            ``max(z) - min(z)`` across all branch points.

        Raises
        ------
        ValueError
            If the morphology has no point geometry.
        """
        return self._axis_range(axis=2)

    @property
    def max_branch_order(self) -> int:
        """Maximum branch order (depth) in the morphology tree.

        The branch order of the root is 0; each edge traversed toward a
        terminal adds 1.

        Returns
        -------
        int
            Longest root-to-branch path length in edges.
        """
        return max(len(self.morpho._path_node_ids(branch._node_id)) - 1 for branch in self.morpho.branches)

    @property
    def max_euclidean_distance(self) -> Quantity:
        """Maximum straight-line distance from root to any terminal tip.

        Returns
        -------
        Quantity[u.um]
            Largest Euclidean distance between the root's first proximal
            point and the last distal point of every terminal branch.

        Raises
        ------
        ValueError
            If the root or any terminal branch lacks 3-D point geometry.
        """
        tip_points = []
        for branch_index in self._terminal_branch_indices():
            branch = self.morpho.branch(index=branch_index).branch
            if branch.points_distal is None:
                raise ValueError("Euclidean distance metrics require point geometry on every terminal branch.")
            tip_points.append(np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)[-1])

        if not tip_points:
            return u.Quantity(0.0, u.um)

        root_point_um = self._root_point_um()
        tip_points_um = np.asarray(tip_points, dtype=float)
        distances_um = np.linalg.norm(tip_points_um - root_point_um, axis=1)
        return u.Quantity(float(np.max(distances_um)), u.um)

    @property
    def max_path_distance(self) -> Quantity:
        """Maximum path distance from root to any terminal tip.

        Path distance is the sum of segment lengths traversed along the
        tree from the root attachment point to a terminal branch tip.

        Returns
        -------
        Quantity[u.um]
            Longest cumulative path length to any terminal.
        """
        terminal_branch_indices = self._terminal_branch_indices()
        if not terminal_branch_indices:
            return u.Quantity(0.0, u.um)

        attach_distances_um = self._root_attach_distances_um()
        max_distance_um = 0.0
        for branch_index in terminal_branch_indices:
            node_id = self.morpho._node_id_from_index(branch_index)
            branch = self.morpho.branch(index=branch_index)
            branch_length_um = self._branch_length_um(branch_index)
            attach_x = self._root_branch_attach_x() if branch.parent_id is None else float(branch.child_x)
            distance_um = attach_distances_um[node_id] + abs(1.0 - attach_x) * branch_length_um
            max_distance_um = max(max_distance_um, distance_um)

        return u.Quantity(max_distance_um, u.um)

    def _axis_range(self, *, axis: int) -> Quantity:
        point_sets = [branch.branch.points for branch in self.morpho.branches if branch.branch.points is not None]
        if not point_sets:
            raise ValueError("Morphology has no point geometry.")
        coords = np.concatenate([np.asarray(points.to_decimal(u.um), dtype=float)[:, axis] for points in point_sets])
        return u.Quantity(coords.max() - coords.min(), u.um)

    def path_to_root(self, branch_index: int) -> tuple[int, ...]:
        """Return the ordered path of branch indices from root to a given branch.

        Parameters
        ----------
        branch_index : int
            Index of the target branch in default ordering.

        Returns
        -------
        tuple of int
            Sequence of branch indices starting at the root and ending at
            *branch_index*.
        """
        node_id = self.morpho._node_id_from_index(branch_index)
        return tuple(
            self.morpho._branch_index(path_node_id)
            for path_node_id in self.morpho._path_node_ids(node_id)
        )

    def path_length_to_root(self, branch_index: int) -> Quantity:
        """Return the path length from a branch to the root.

        .. note:: Not yet implemented.

        Parameters
        ----------
        branch_index : int
            Index of the target branch in default ordering.

        Returns
        -------
        Quantity[u.um]
            Cumulative segment length along the path to root.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        raise NotImplementedError

    def shortest_path_length(
        self,
        from_site: tuple[int, float],
        to_site: tuple[int, float],
    ) -> Quantity:
        """Return the shortest path length between two sites on the tree.

        A site is a ``(branch_index, position)`` pair where *position*
        is a fractional location along the branch (0 = proximal, 1 = distal).

        .. note:: Not yet implemented.

        Parameters
        ----------
        from_site : tuple of (int, float)
            Start site as ``(branch_index, position)``.
        to_site : tuple of (int, float)
            End site as ``(branch_index, position)``.

        Returns
        -------
        Quantity[u.um]
            Shortest path length through the tree between the two sites.

        Raises
        ------
        NotImplementedError
            Always, until implemented.
        """
        raise NotImplementedError
