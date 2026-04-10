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

"""Editable morphology model.

User-facing entry points:
- `Morpho`: the whole mutable tree
- `MorphoBranch`: a tree-local branch node view
- `MorphoEdge`: read-only branch-to-branch topology edges

In normal use, users only need `Morpho` and `MorphoBranch`.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Union

import brainunit as u
import numpy as np
from brainunit import Quantity

from .branch import Branch

_MORPHO_METRIC_PROPERTY_NAMES = {
    "max_euclidean_distance",
    "max_euclidean_distance_excluding_soma",
    "max_branch_order",
    "max_path_distance",
    "max_path_distance_excluding_soma",
    "mean_radius",
    "n_bifurcations",
    "n_branches",
    "n_stems",
    "total_area",
    "total_length",
    "total_volume",
    "x_range",
    "y_range",
    "z_range",
}

_MORPHO_RESERVED_NAMES = {
    "attach",
    "branch",
    "branches",
    "branch_by_order",
    "edges",
    "from_asc",
    "from_root",
    "from_swc",
    "has_full_point_geometry",
    "metric",
    "path_to_root",
    "path_length_to_root",
    "root",
    "select",
    "shortest_path_length",
    "vis2d",
    "vis3d",
} | _MORPHO_METRIC_PROPERTY_NAMES
_MORPHO_BRANCH_RESERVED_NAMES = {
    "attach",
    "branch",
    "child_x",
    "children",
    "index",
    "index_by",
    "name",
    "n_children",
    "parent",
    "parent_id",
    "parent_x",
}
_BRANCH_RESERVED_NAMES = set(Branch.__dataclass_fields__) | {name for name in dir(Branch) if not name.startswith("_")}

ParentRef = Union[str, "MorphoBranch"]


@dataclass(frozen=True)
class MorphoEdge:
    """A directed edge between two morphology branches.

    ``MorphoEdge`` is a frozen dataclass representing a parent-child
    connection in a :class:`Morpho` tree.  It records which branches are
    connected and where on each branch the attachment occurs.

    Edges are typically obtained via :attr:`Morpho.edges` rather than
    constructed directly.

    Parameters
    ----------
    parent : MorphoBranch
        The parent branch in the connection.
    child : MorphoBranch
        The child branch in the connection.
    parent_x : float
        Attachment point on the parent branch.  One of ``0`` (proximal),
        ``0.5`` (midpoint, soma only), or ``1`` (distal).
    child_x : float
        Attachment point on the child branch.  One of ``0`` (proximal) or
        ``1`` (distal).  Default is ``0``.

    See Also
    --------
    Morpho.edges : Retrieve all edges in a morphology.
    Morpho.attach : Attach a child branch to a parent.

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
        ...     lengths=[50.0] * u.um,
        ...     radii=[2.0, 1.0] * u.um,
        ...     type="dendrite",
        ... )
        >>> morpho = Morpho.from_root(soma, name="soma")
        >>> morpho.soma.dend = dend
        >>> edges = morpho.edges
        >>> len(edges)
        1
        >>> edges[0].parent.name
        'soma'
        >>> edges[0].child.name
        'dend'
    """

    parent: "MorphoBranch"
    child: "MorphoBranch"
    parent_x: float
    child_x: float = 0.0


@dataclass(frozen=True)
class MorphoMetric:
    """Snapshot of whole-morphology metrics."""

    n_branches: int
    n_stems: int
    n_bifurcations: int
    max_branch_order: int
    total_length: Quantity
    mean_radius: Quantity
    total_area: Quantity
    total_volume: Quantity
    max_path_distance: Quantity
    max_path_distance_excluding_soma: Quantity
    max_euclidean_distance: Optional[Quantity]
    max_euclidean_distance_excluding_soma: Optional[Quantity]
    x_range: Optional[Quantity]
    y_range: Optional[Quantity]
    z_range: Optional[Quantity]
    has_full_point_geometry: bool

    @classmethod
    def from_morpho(cls, morpho: "Morpho") -> "MorphoMetric":
        has_full_point_geometry = morpho.has_full_point_geometry
        return cls(
            n_branches=morpho.n_branches,
            n_stems=morpho.n_stems,
            n_bifurcations=morpho.n_bifurcations,
            max_branch_order=morpho.max_branch_order,
            total_length=morpho.total_length,
            mean_radius=morpho.mean_radius,
            total_area=morpho.total_area,
            total_volume=morpho.total_volume,
            max_path_distance=morpho.max_path_distance,
            max_path_distance_excluding_soma=morpho.max_path_distance_excluding_soma,
            max_euclidean_distance=morpho.max_euclidean_distance if has_full_point_geometry else None,
            max_euclidean_distance_excluding_soma=(
                morpho.max_euclidean_distance_excluding_soma if has_full_point_geometry else None
            ),
            x_range=morpho.x_range if has_full_point_geometry else None,
            y_range=morpho.y_range if has_full_point_geometry else None,
            z_range=morpho.z_range if has_full_point_geometry else None,
            has_full_point_geometry=has_full_point_geometry,
        )

    @staticmethod
    def _format_value(value: object) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, Quantity):
            return f"{value:.2f}"
        return str(value)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"{'-'*35}\n"
            f"{'n_branches':<16} | {self.n_branches}\n"
            f"{'n_stems':<16} | {self.n_stems}\n"
            f"{'n_bifurcations':<16} | {self.n_bifurcations}\n"
            f"{'max_branch_order':<16} | {self.max_branch_order}\n"
            f"{'total_length':<16} | {self._format_value(self.total_length)}\n"
            f"{'mean_radius':<16} | {self._format_value(self.mean_radius)}\n"
            f"{'total_area':<16} | {self._format_value(self.total_area)}\n"
            f"{'total_volume':<16} | {self._format_value(self.total_volume)}\n"
            f"{'max_path_dist':<16} | {self._format_value(self.max_path_distance)}\n"
            f"{'-'*35}\n"
        )


class Morpho:
    """Mutable morphology tree for authoring, querying, and visualization.

    ``Morpho`` is the central entry point for building neuron morphologies.
    It owns a tree of :class:`MorphoBranch` nodes, each wrapping an
    immutable :class:`Branch` geometry.  Children are attached via
    :meth:`attach`, attribute assignment on a :class:`MorphoBranch`, or
    the ``branch[parent_x].child_name = branch`` syntax sugar.

    Prefer the factory classmethods :meth:`from_root`, :meth:`from_swc`,
    and :meth:`from_asc` over the raw constructor.

    Parameters
    ----------
    root_name : str or None
        Name of the root branch.  If ``None``, a name is auto-generated
        from the branch type (e.g., ``"soma_0"``).
    root_branch : Branch
        Geometry of the root branch.

    Raises
    ------
    ValueError
        If *root_name* is a reserved name or already taken, or is not a
        valid Python identifier.

    See Also
    --------
    Branch : Immutable segment geometry.
    MorphoBranch : Tree-local branch node view.

    Notes
    -----
    Branch names must be valid Python identifiers (no leading underscore)
    and must not collide with reserved method or metric names.  Auto-naming
    follows the pattern ``"{type}_{n}"`` (e.g., ``"dend_0"``, ``"axon_1"``).

    Whole-morphology metrics (``total_length``, ``n_branches``, etc.)
    are exposed directly on the ``Morpho`` instance.

    Examples
    --------

    Build a simple morphology by hand:

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
        ...     type="dendrite",
        ... )
        >>> morpho = Morpho.from_root(soma, name="soma")
        >>> morpho.soma.dend = dend
        >>> len(morpho)
        2
        >>> morpho.n_branches
        2
        >>> morpho.total_length
        110.0 * umetre

    Load from an SWC file:

    .. code-block:: python

        >>> morpho = Morpho.from_swc("neuron.swc")  # doctest: +SKIP
    """

    def __init__(self, *, root_name: str | None, root_branch: Branch) -> None:
        self._nodes: dict[int, MorphoBranch] = {}
        self._name_to_id: dict[str, int] = {}
        self._type_name_counters: dict[str, int] = {}
        self._next_id = 0
        self._root_name = self._resolve_node_name(root_branch, explicit_name=root_name)
        self._root_id = self._register_node(
            name=self._root_name,
            branch=root_branch,
            parent_id=None,
            parent_x=1.0,
            child_x=0.0,
        )

    @classmethod
    def from_root(cls, branch: Branch, *, name: str | None = "soma") -> "Morpho":
        """Create a new morphology tree from a single root branch.

        Parameters
        ----------
        branch : Branch
            Geometry of the root branch (typically a soma).
        name : str or None
            Name for the root branch (default ``"soma"``).  If ``None``,
            auto-generated from the branch type.

        Returns
        -------
        Morpho
            New morphology containing only the root.

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
            >>> morpho = Morpho.from_root(soma, name="soma")
            >>> morpho.root.name
            'soma'
        """

        return cls(root_name=name, root_branch=branch)

    @classmethod
    def from_swc(
        cls,
        path,
        *,
        options: 'SwcReadOptions' = None,
        mode: str | None = None,
        return_report: bool = False,
    ):
        """Load a morphology from an SWC file.

        Delegates to :class:`~braincell.io.SwcReader` for parsing and
        validation.

        Parameters
        ----------
        path : str or Path
            Path to the SWC file.
        options : SwcReadOptions or None
            Reader configuration (soma handling, repairs, etc.).
            Uses defaults when ``None``.
        mode : str or None
            Convenience import mode. Supported values are ``"neuron"``
            (NEURON-style soma attachment handling) and ``"neuromorpho"``
            (copy soma attachment points into child branches). When
            provided together with *options*, it must match
            ``options.mode``.
        return_report : bool
            If ``True``, return a ``(Morpho, SwcReport)`` tuple instead
            of just the morphology.

        Returns
        -------
        Morpho or (Morpho, SwcReport)
            The loaded morphology, optionally with a diagnostic report.

        See Also
        --------
        from_asc : Load from Neurolucida ASC format.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morpho
            >>> morpho = Morpho.from_swc("neuron.swc")  # doctest: +SKIP
            >>> morpho, report = Morpho.from_swc(
            ...     "neuron.swc", return_report=True
            ... )  # doctest: +SKIP
        """

        from braincell.io.swc import SwcReadOptions, SwcReader

        if mode is not None:
            validated_options = SwcReadOptions(mode=mode)
            if options is None:
                options = validated_options
            elif options.mode != validated_options.mode:
                raise ValueError(
                    f"Conflicting SWC import mode: mode={mode!r} does not match options.mode={options.mode!r}."
                )

        reader = SwcReader() if options is None else SwcReader(options=options)
        return reader.read(path, return_report=return_report)

    @classmethod
    def from_asc(cls, path, *, return_report: bool = False):
        """Load a morphology from a Neurolucida ASC file.

        Delegates to :class:`~braincell.io.AscReader` for parsing and
        validation.

        Parameters
        ----------
        path : str or Path
            Path to the ASC file.
        return_report : bool
            If ``True``, return a ``(Morpho, AscReport)`` tuple instead
            of just the morphology.

        Returns
        -------
        Morpho or (Morpho, AscReport)
            The loaded morphology, optionally with a diagnostic report.

        See Also
        --------
        from_swc : Load from SWC format.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morpho
            >>> morpho = Morpho.from_asc("neuron.asc")  # doctest: +SKIP
        """

        from braincell.io.asc import AscReader

        return AscReader().read(path, return_report=return_report)

    def save_checkpoint(self, path):
        """Write this morphology to a braincell checkpoint file.

        Checkpoints are a lossless on-disk snapshot that preserves branch
        names, ``parent_x``/``child_x`` attachments (including ``0.5``
        soma midpoints), and the auto-naming counters consulted by
        :meth:`attach`. Unlike SWC/ASC, the round-trip is bit-exact.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path. ``.bcm`` is appended automatically when the
            supplied path has no suffix.

        Returns
        -------
        Path
            The final path the checkpoint was written to.

        See Also
        --------
        Morpho.load_checkpoint : Inverse operation.
        Morpho.from_swc : Load a morphology from an SWC interchange file.
        Morpho.from_asc : Load a morphology from a Neurolucida ASC file.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morpho
            >>> morpho = Morpho.from_swc("neuron.swc")  # doctest: +SKIP
            >>> morpho.save_checkpoint("neuron.bcm")  # doctest: +SKIP
        """
        from braincell.io.checkpoint import save_morpho

        return save_morpho(self, path)

    @classmethod
    def load_checkpoint(cls, path) -> "Morpho":
        """Load a morphology from a braincell checkpoint file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a ``.bcm`` checkpoint produced by
            :meth:`save_checkpoint`.

        Returns
        -------
        Morpho
            The reconstructed morphology, equal to the saved tree under
            :meth:`__eq__`.

        Raises
        ------
        CheckpointError
            If the file is missing, corrupt, or is not a morpho checkpoint.

        See Also
        --------
        Morpho.save_checkpoint : Inverse operation.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morpho
            >>> morpho = Morpho.load_checkpoint("neuron.bcm")  # doctest: +SKIP
        """
        from braincell.io.checkpoint import load_morpho

        return load_morpho(path)

    @property
    def root(self) -> "MorphoBranch":
        """The root branch of this morphology.

        Returns
        -------
        MorphoBranch
            Root branch node.
        """
        return self._get_node(self._root_id)

    @property
    def branches(self) -> tuple["MorphoBranch", ...]:
        """All branches in default order (by node ID).

        Returns
        -------
        tuple of MorphoBranch
            All branches in the tree.

        See Also
        --------
        branch_by_order : Query branches in a specific order.
        """
        return self.branch_by_order(order="default")

    @property
    def edges(self) -> tuple[MorphoEdge, ...]:
        """All directed edges in the morphology tree.

        Returns
        -------
        tuple of MorphoEdge
            Parent-child connections, excluding the root (which has no parent).
        """
        return tuple(
            MorphoEdge(
                parent=self._get_node(node.parent_id),
                child=node,
                parent_x=node.parent_x,
                child_x=node.child_x,
            )
            for node_id in self._ordered_node_ids()
            for node in (self._get_node(node_id),)
            if node.parent_id is not None
        )

    @property
    def metric(self) -> MorphoMetric:
        """Whole-morphology metrics exposed as a dataclass snapshot."""
        return MorphoMetric.from_morpho(self)

    def branch_by_order(self, *, order: str = "default") -> tuple["MorphoBranch", ...]:
        """Query branches in a specific order.

        Parameters
        ----------
        order : str
            Ordering strategy: ``"default"`` (by node ID), ``"type"``
            (by SWC type rank then name), or ``"depth"`` (by depth then index).

        Returns
        -------
        tuple of MorphoBranch
            Ordered branches.

        Raises
        ------
        ValueError
            If *order* is not recognized.
        """
        return tuple(self._get_node(node_id) for node_id in self._ordered_node_ids_by(order))

    @property
    def has_full_point_geometry(self) -> bool:
        return all(
            branch.branch.points_proximal is not None and branch.branch.points_distal is not None
            for branch in self.branches
        )

    def _require_full_point_geometry(self, *, feature: str) -> None:
        if not self.has_full_point_geometry:
            raise ValueError(f"{feature} require full point geometry on every branch.")

    def _all_segment_arrays_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = []
        radii_proximal = []
        radii_distal = []
        for branch in self.branches:
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
        branch = self.branch(index=branch_index).branch
        return float(np.sum(np.asarray(branch.lengths.to_decimal(u.um), dtype=float)))

    def _root_branch_attach_x(self) -> float:
        return 0.0

    def _root_point_um(self) -> np.ndarray:
        self._require_full_point_geometry(feature="Euclidean distance metrics")
        root = self.root.branch
        return np.asarray(root.points_proximal.to_decimal(u.um), dtype=float)[0]

    def _terminal_branch_indices(self) -> tuple[int, ...]:
        return tuple(branch.index for branch in self.branches if branch.n_children == 0)

    def _point_on_branch_at_x_um(self, branch: Branch, x: float) -> np.ndarray:
        self._require_full_point_geometry(feature="Euclidean distance metrics")
        points = branch.points
        if points is None:
            raise ValueError("Euclidean distance metrics require full point geometry on every branch.")

        points_um = np.asarray(points.to_decimal(u.um), dtype=float)
        if np.isclose(float(x), 0.0):
            return points_um[0]
        if np.isclose(float(x), 1.0):
            return points_um[-1]

        lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        total_length_um = float(np.sum(lengths_um))
        if total_length_um <= 0.0:
            return points_um[0]

        target_length_um = float(x) * total_length_um
        prefix_length_um = 0.0
        for index, segment_length_um in enumerate(lengths_um):
            next_prefix_length_um = prefix_length_um + segment_length_um
            if target_length_um <= next_prefix_length_um or index == len(lengths_um) - 1:
                if np.isclose(segment_length_um, 0.0):
                    return points_um[index]
                fraction = (target_length_um - prefix_length_um) / segment_length_um
                return points_um[index] + fraction * (points_um[index + 1] - points_um[index])
            prefix_length_um = next_prefix_length_um

        return points_um[-1]

    def _root_attach_distances_um(self, *, exclude_root_soma: bool = False) -> dict[int, float]:
        ordered_ids = self._ordered_node_ids_by("depth")
        distances_um: dict[int, float] = {self.root._node_id: 0.0}
        attach_x: dict[int, float] = {self.root._node_id: self._root_branch_attach_x()}

        for node_id in ordered_ids:
            if node_id == self.root._node_id:
                continue

            node = self._get_node(node_id)
            parent_id = node.parent_id
            if parent_id is None:
                continue

            parent_attach_x = attach_x[parent_id]
            parent_distance_um = distances_um[parent_id]
            parent_length_um = self._branch_length_um(self._branch_index(parent_id))

            if exclude_root_soma and parent_id == self.root._node_id and self.root.type == "soma":
                distances_um[node_id] = 0.0
            else:
                distances_um[node_id] = (
                    parent_distance_um + abs(float(node.parent_x) - parent_attach_x) * parent_length_um
                )
            attach_x[node_id] = float(node.child_x)

        return distances_um

    def _max_path_distance_um(self, *, exclude_root_soma: bool) -> float:
        terminal_branch_indices = self._terminal_branch_indices()
        if not terminal_branch_indices:
            return 0.0
        if exclude_root_soma and self.root.type == "soma" and self.root.n_children == 0:
            return 0.0

        attach_distances_um = self._root_attach_distances_um(exclude_root_soma=exclude_root_soma)
        max_distance_um = 0.0
        for branch_index in terminal_branch_indices:
            node_id = self._node_id_from_index(branch_index)
            branch = self.branch(index=branch_index)
            branch_length_um = self._branch_length_um(branch_index)
            attach_x = self._root_branch_attach_x() if branch.parent_id is None else float(branch.child_x)
            distance_um = attach_distances_um[node_id] + abs(1.0 - attach_x) * branch_length_um
            max_distance_um = max(max_distance_um, distance_um)
        return max_distance_um

    def _root_subtree_reference_point_um(self, terminal_node_id: int) -> np.ndarray:
        path_node_ids = self._path_node_ids(terminal_node_id)
        if len(path_node_ids) <= 1:
            return self._root_point_um()
        first_child = self._get_node(path_node_ids[1])
        return self._point_on_branch_at_x_um(self.root.branch, float(first_child.parent_x))

    def _max_euclidean_distance_um(self, *, exclude_root_soma: bool) -> float:
        self._require_full_point_geometry(feature="Euclidean distance metrics")
        terminal_branch_indices = self._terminal_branch_indices()
        if not terminal_branch_indices:
            return 0.0
        if exclude_root_soma and self.root.type == "soma" and self.root.n_children == 0:
            return 0.0

        tip_points = []
        start_points = []
        root_point_um = self._root_point_um()
        for branch_index in terminal_branch_indices:
            node_id = self._node_id_from_index(branch_index)
            branch = self.branch(index=branch_index).branch
            tip_points.append(np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)[-1])
            if exclude_root_soma and self.root.type == "soma":
                start_points.append(self._root_subtree_reference_point_um(node_id))
            else:
                start_points.append(root_point_um)

        tip_points_um = np.asarray(tip_points, dtype=float)
        start_points_um = np.asarray(start_points, dtype=float)
        distances_um = np.linalg.norm(tip_points_um - start_points_um, axis=1)
        return float(np.max(distances_um))

    @property
    def total_length(self) -> Quantity:
        lengths_um, _, _ = self._all_segment_arrays_um()
        return u.Quantity(np.sum(lengths_um), u.um)

    @property
    def mean_radius(self) -> Quantity:
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        total_length_um = float(np.sum(lengths_um))
        if total_length_um <= 0.0:
            raise ValueError("Morphology total length must be > 0.")
        values_um = 0.5 * (r0_um + r1_um)
        return u.Quantity(np.sum(lengths_um * values_um) / total_length_um, u.um)

    @property
    def total_area(self) -> Quantity:
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        value = np.sum(np.pi * (r0_um + r1_um) * np.sqrt(lengths_um * lengths_um + (r1_um - r0_um) * (r1_um - r0_um)))
        return u.Quantity(value, u.um ** 2)

    @property
    def total_volume(self) -> Quantity:
        lengths_um, r0_um, r1_um = self._all_segment_arrays_um()
        value = np.sum(np.pi * lengths_um * (r0_um * r0_um + r0_um * r1_um + r1_um * r1_um) / 3.0)
        return u.Quantity(value, u.um ** 3)

    @property
    def n_branches(self) -> int:
        return len(self.branches)

    @property
    def n_stems(self) -> int:
        return self.root.n_children

    @property
    def n_bifurcations(self) -> int:
        return sum(branch.n_children >= 2 for branch in self.branches)

    def _axis_range(self, *, axis: int) -> Quantity:
        self._require_full_point_geometry(feature="Coordinate range metrics")
        point_sets = [branch.branch.points for branch in self.branches]
        coords = np.concatenate([np.asarray(points.to_decimal(u.um), dtype=float)[:, axis] for points in point_sets])
        return u.Quantity(coords.max() - coords.min(), u.um)

    @property
    def x_range(self) -> Quantity:
        return self._axis_range(axis=0)

    @property
    def y_range(self) -> Quantity:
        return self._axis_range(axis=1)

    @property
    def z_range(self) -> Quantity:
        return self._axis_range(axis=2)

    @property
    def max_branch_order(self) -> int:
        return max(len(self._path_node_ids(branch._node_id)) - 1 for branch in self.branches)

    @property
    def max_euclidean_distance(self) -> Quantity:
        return u.Quantity(self._max_euclidean_distance_um(exclude_root_soma=False), u.um)

    @property
    def max_euclidean_distance_excluding_soma(self) -> Quantity:
        if self.root.type != "soma":
            return self.max_euclidean_distance
        return u.Quantity(self._max_euclidean_distance_um(exclude_root_soma=True), u.um)

    @property
    def max_path_distance(self) -> Quantity:
        return u.Quantity(self._max_path_distance_um(exclude_root_soma=False), u.um)

    @property
    def max_path_distance_excluding_soma(self) -> Quantity:
        if self.root.type != "soma":
            return self.max_path_distance
        return u.Quantity(self._max_path_distance_um(exclude_root_soma=True), u.um)

    def branch(
        self,
        *,
        name: str | None = None,
        index: int | None = None,
        order: str | None = None,
    ) -> "MorphoBranch":
        """Retrieve a single branch by name or index.

        Parameters
        ----------
        name : str or None
            Branch name.  Mutually exclusive with *index*.
        index : int or None
            Branch index in the specified *order*.  Mutually exclusive
            with *name*.
        order : str or None
            Ordering strategy when querying by *index*: ``"default"``,
            ``"type"``, or ``"depth"``.  Ignored when querying by *name*.

        Returns
        -------
        MorphoBranch
            The requested branch.

        Raises
        ------
        TypeError
            If neither or both of *name* and *index* are provided, or if
            *order* is given with *name*.
        KeyError
            If *name* does not exist.
        IndexError
            If *index* is out of range.
        """
        if (name is None) == (index is None):
            raise TypeError("exactly one of `name` or `index` must be provided")
        if name is not None:
            if order is not None:
                raise TypeError("`order` cannot be provided when querying by `name`")
            if name not in self._name_to_id:
                raise KeyError(name)
            return self._get_node(self._name_to_id[name])

        ordered_ids = self._ordered_node_ids_by("default" if order is None else order)
        try:
            return self._get_node(ordered_ids[index])  # type: ignore[index]
        except IndexError as exc:
            raise IndexError(f"Branch index {index!r} is out of range.") from exc

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
        node_id = self._node_id_from_index(branch_index)
        return tuple(
            self._branch_index(path_node_id)
            for path_node_id in self._path_node_ids(node_id)
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

    def topo(self) -> str:
        """Return a line-oriented text view of the branch topology.

        Returns
        -------
        str
            ASCII tree representation showing parent-child relationships.

        Examples
        --------

        .. code-block:: python

            >>> print(morpho.topo())  # doctest: +SKIP
            soma
            ├── dend_0
            │   ├── dend_1
            │   └── dend_2
            └── axon_0
        """

        lines = [self.root.name]
        child_ids = tuple(self.root._children.values())
        for index, child_id in enumerate(child_ids):
            lines.extend(self._format_topology(child_id, prefix="", is_last=index == len(child_ids) - 1))
        return "\n".join(lines)

    def vis3d(
        self,
        *,
        mode: str | None = None,
        backend: str | None = None,
        region=None,
        locset=None,
        values=None,
        chooser=None,
        notebook: bool | None = None,
        jupyter_backend: str | None = None,
        return_plotter: bool = False,
        show: bool = True,
    ) -> object:
        """Visualize this morphology in 3D.

        Renders the morphology as 3-D tubes using the PyVista backend.
        All branches must have been created with :meth:`Branch.from_points`
        (i.e. they must carry 3-D point geometry).

        Parameters
        ----------
        mode : str or None
            Visualization mode. When omitted, uses the global 3-D
            default configured via ``braincell.morpho.vis.configure(...)``.
            The initial default is ``"geometry"``.
        backend : str or None
            Rendering backend name (e.g., ``"pyvista"``).
            Auto-selected when *None*.
        region : RegionMask or None
            Evaluated region mask to attach as an overlay.  Obtain one
            via ``expr.evaluate(morpho)`` or ``morpho.select(expr)``.
        locset : LocsetMask or None
            Evaluated location-set mask to attach as an overlay.
        values : array-like or None
            Per-branch or per-segment scalar values for colour-mapping.
        chooser : BackendChooser or None
            Explicit backend chooser; overrides *backend* when given.
        notebook : bool or None
            Enable notebook-specific rendering when *True*.
        jupyter_backend : str or None
            Jupyter backend name for notebook rendering (e.g.,
            ``"trame"``, ``"static"``).
        return_plotter : bool
            If *True*, return the PyVista plotter object instead of
            displaying the figure.
        show : bool
            If *True* (default), call the backend's show method after
            rendering.  Set to *False* to suppress the blocking display
            call.

        Returns
        -------
        object
            The PyVista plotter when *return_plotter* is True;
            otherwise the backend's default display result.

        See Also
        --------
        vis2d : 2-D visualization with the Matplotlib backend.

        Examples
        --------

        **Basic 3-D rendering** of a soma with one dendrite:

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morpho
            >>> soma = Branch.from_points(
            ...     points=[[0, 0, 0], [20, 0, 0]] * u.um,
            ...     radii=[10.0, 10.0] * u.um,
            ...     type="soma",
            ... )
            >>> dend = Branch.from_points(
            ...     points=[[20, 0, 0], [20, 80, 0]] * u.um,
            ...     radii=[2.0, 1.0] * u.um,
            ...     type="apical_dendrite",
            ... )
            >>> morpho = Morpho.from_root(soma, name="soma")
            >>> morpho.soma.dend = dend
            >>> morpho.vis3d()  # doctest: +SKIP

        **Multi-branch morphology** with basal dendrites and an axon:

        .. code-block:: python

            >>> import numpy as np
            >>> basal = Branch.from_points(
            ...     points=[[0, 0, 0], [-30, -50, 10]] * u.um,
            ...     radii=[3.0, 1.5] * u.um,
            ...     type="basal_dendrite",
            ... )
            >>> axon = Branch.from_points(
            ...     points=[[0, 0, 0], [0, 0, -100]] * u.um,
            ...     radii=[1.0, 0.5] * u.um,
            ...     type="axon",
            ... )
            >>> morpho.attach(parent="soma", child_branch=basal,
            ...               child_name="basal", parent_x=0.0)  # doctest: +SKIP
            >>> morpho.attach(parent="soma", child_branch=axon,
            ...               child_name="axon", parent_x=0.5)  # doctest: +SKIP
            >>> morpho.vis3d()  # doctest: +SKIP

        **Highlight a region** — e.g. the full extent of all dendrites:

        .. code-block:: python

            >>> from braincell.filter import branch_in
            >>> region = branch_in("type", ("apical_dendrite", "basal_dendrite")).evaluate(morpho)
            >>> morpho.vis3d(region=region)  # doctest: +SKIP

        **Mark specific locations** — e.g. branch points and terminals:

        .. code-block:: python

            >>> from braincell.filter import BranchPoints, Terminals
            >>> locset = (BranchPoints() | Terminals()).evaluate(morpho)
            >>> morpho.vis3d(locset=locset)  # doctest: +SKIP

        **Retrieve the PyVista plotter** for further customisation:

        .. code-block:: python

            >>> plotter = morpho.vis3d(return_plotter=True, show=False)  # doctest: +SKIP
            >>> plotter.add_text("My neuron", font_size=12)  # doctest: +SKIP
            >>> plotter.show()  # doctest: +SKIP

        **Notebook rendering** with the Trame backend:

        .. code-block:: python

            >>> morpho.vis3d(notebook=True, jupyter_backend="trame")  # doctest: +SKIP
        """
        from braincell.vis.plot3d import plot3d

        result = plot3d(
            self,
            region=region,
            locset=locset,
            values=values,
            mode=mode,
            backend=backend,
            chooser=chooser,
            notebook=notebook,
            jupyter_backend=jupyter_backend,
            return_plotter=return_plotter,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return result

    def vis2d(
        self,
        *,
        layout: str | None = None,
        shape: str | None = None,
        backend: str | None = None,
        region=None,
        locset=None,
        values=None,
        chooser=None,
        ax=None,
        notebook: bool | None = None,
        jupyter_backend: str | None = None,
        return_plotter: bool = False,
        show: bool = True,
        projection_plane: str = "xy",
        min_branch_angle_deg: float | None = 25.0,
        root_layout: str = "type_split",
    ) -> object:
        """Visualize this morphology in 2D.

        Renders the morphology tree using the Matplotlib backend. The 2-D
        visualization is controlled by:

        * ``layout`` — how 2-D coordinates are obtained:
          ``"projected"``, ``"stem"``, ``"balloon"``, or ``"radial_360"``.
        * ``shape`` — how those 2-D coordinates are drawn:
          ``"line"`` or ``"frustum"``.

        Parameters
        ----------
        layout : str or None
            2-D layout choice. ``"projected"`` uses projected 3-D point
            geometry. ``"stem"``, ``"balloon"``, and ``"radial_360"``
            use the schematic branch layout pipeline. When omitted, uses
            the global 2-D default configured via
            ``braincell.morpho.vis.configure(...)``. The initial default
            is ``"stem"``.
        shape : str or None
            2-D drawing shape: ``"line"`` or ``"frustum"``. When omitted,
            uses the global 2-D default configured via
            ``braincell.morpho.vis.configure(...)``. The initial default
            is ``"frustum"``.
        backend : str or None
            Rendering backend name (e.g., ``"matplotlib"``).
            Auto-selected when *None*.
        region : RegionMask or None
            Evaluated region mask to attach as an overlay.  Obtain one
            via ``expr.evaluate(morpho)`` or ``morpho.select(expr)``.
        locset : LocsetMask or None
            Evaluated location-set mask to attach as an overlay.
        values : array-like or None
            Per-branch or per-segment scalar values for colour-mapping.
        chooser : BackendChooser or None
            Explicit backend chooser; overrides *backend* when given.
        projection_plane : str
            Axis pair for ``layout="projected"``: ``"xy"`` (default),
            ``"xz"``, or ``"yz"``. Ignored by schematic layouts.
        min_branch_angle_deg : float or None
            Minimum branch separation angle for schematic layouts.
            Ignored by ``layout="projected"``.
        root_layout : str
            Root branching strategy for schematic layouts. Ignored by
            ``layout="projected"``.
        notebook : bool or None
            Enable notebook-specific rendering when *True*.
        jupyter_backend : str or None
            Jupyter backend name for notebook rendering.
        return_plotter : bool
            If *True*, return the ``matplotlib.axes.Axes`` object instead
            of displaying the figure.
        show : bool
            If *True* (default), call ``matplotlib.pyplot.show()`` after
            rendering.  Set to *False* to suppress the blocking display
            call (useful in scripts, tests, or when embedding the axes
            in a larger figure).

        Returns
        -------
        object
            The ``matplotlib.axes.Axes`` when *return_plotter* is True;
            otherwise the backend's default display result.

        Raises
        ------
        ValueError
            If ``layout="projected"`` and any branch lacks 3-D point
            geometry, or if ``layout="projected"`` is combined with
            ``shape="frustum"``.

        See Also
        --------
        Branch.vis2d : Quick visualization of a single branch.
        vis3d : 3-D visualization with the PyVista backend.

        Examples
        --------

        **Projected layout** with line drawing:

        .. code-block:: python

            >>> morpho.vis2d(layout="projected", shape="line")  # doctest: +SKIP

        **Stem layout** with line drawing:

        .. code-block:: python

            >>> morpho.vis2d(layout="stem", shape="line")  # doctest: +SKIP

        **Stem layout** with frustum drawing:

        .. code-block:: python

            >>> morpho.vis2d(layout="stem", shape="frustum")  # doctest: +SKIP
        """
        from braincell.vis.plot2d import plot2d

        result = plot2d(
            self,
            region=region,
            locset=locset,
            values=values,
            layout=layout,
            shape=shape,
            backend=backend,
            chooser=chooser,
            ax=ax,
            notebook=notebook,
            jupyter_backend=jupyter_backend,
            return_plotter=return_plotter,
            projection_plane=projection_plane,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return result

    def select(self, expr, *, cache=None):
        """Evaluate a region or location-set expression on this morphology.

        Parameters
        ----------
        expr : RegionExpr or LocsetExpr
            Filter expression to evaluate.
        cache : SelectionCache or None
            Optional cache for repeated evaluations.

        Returns
        -------
        RegionMask or LocsetMask
            Evaluated mask result.

        Raises
        ------
        TypeError
            If *expr* is not a ``RegionExpr`` or ``LocsetExpr``.

        See Also
        --------
        braincell.filter.RegionExpr : Region filter expressions.
        braincell.filter.LocsetExpr : Location-set expressions.

        Examples
        --------

        .. code-block:: python

            >>> from braincell.filter import branch_in
            >>> region = morpho.select(branch_in("type", "dendrite"))  # doctest: +SKIP
        """
        from braincell.filter import LocsetExpr, RegionExpr

        if not isinstance(expr, (RegionExpr, LocsetExpr)):
            raise TypeError(
                "Morpho.select(...) expects RegionExpr or LocsetExpr. "
                f"Got {type(expr).__name__!s}."
            )
        return expr.evaluate(self, cache=cache)

    def attach(
        self,
        *,
        parent: ParentRef,
        child_branch: Branch,
        child_name: str | None = None,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        """Attach a child branch to a parent in this morphology.

        Parameters
        ----------
        parent : str or MorphoBranch
            Parent branch, specified by name or node reference.
        child_branch : Branch
            Geometry of the child branch to attach.
        child_name : str or None
            Name for the child branch.  If ``None``, auto-generated from
            the branch type (e.g., ``"dend_0"``).
        parent_x : float
            Attachment point on the parent branch: ``0`` (proximal),
            ``0.5`` (midpoint, soma only), or ``1`` (distal, default).
        child_x : float
            Attachment point on the child branch: ``0`` (proximal,
            default) or ``1`` (distal).

        Returns
        -------
        MorphoBranch
            The newly attached child branch node.

        Raises
        ------
        ValueError
            If *parent_x* or *child_x* is invalid, if *parent_x* is ``0.5``
            on a non-soma parent, or if *child_name* already exists.
        TypeError
            If *parent* is not a string or ``MorphoBranch``.
        KeyError
            If *parent* is a string that does not match any branch name.

        Notes
        -----
        ``parent_x=0`` attaches to the proximal end of the parent branch.
        This is typically used when the parent is itself a child branch and
        you want to attach at its connection point rather than its distal end.

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
            ...     lengths=[50.0] * u.um,
            ...     radii=[2.0, 1.0] * u.um,
            ...     type="dendrite",
            ... )
            >>> morpho = Morpho.from_root(soma, name="soma")
            >>> child = morpho.attach(
            ...     parent="soma",
            ...     child_branch=dend,
            ...     child_name="apical",
            ... )
            >>> child.name
            'apical'
        """

        parent_id = self._resolve_parent(parent)
        return self._insert_child(
            parent_id,
            child_branch,
            child_name=child_name,
            parent_x=parent_x,
            child_x=child_x,
        )

    def __getattr__(self, name: str) -> object:
        # Underscore-prefixed names never name a branch. Failing fast avoids
        # unbounded recursion when pickle / copy.deepcopy probe for
        # ``__setstate__`` etc. on a partially-initialised instance whose
        # ``_name_to_id`` dict has not yet been restored.
        if name.startswith("_"):
            raise AttributeError(name)
        name_to_id = self.__dict__.get("_name_to_id")
        if name_to_id is not None and name in name_to_id:
            return self._get_node(name_to_id[name])
        raise AttributeError(f"{type(self).__name__!s} has no branch named {name!r}")

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._name_to_id))

    def __len__(self) -> int:
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morpho):
            return NotImplemented
        return self._eq_records() == other._eq_records()

    def __repr__(self) -> str:
        geo_status = "complete 3d points" if self.has_full_point_geometry else "incomplete 3d points"
        return f"Morpho(root={self.root.name!r}, n_branches={self.n_branches}, geometry = {geo_status})"
    
    def __str__(self) -> str:
        """Return a formatted summary with key metrics."""
        geo_status = "complete 3d points" if self.has_full_point_geometry else "incomplete 3d points"
        return (
            f"{'-'*35}\n"
            f"{'root':<12} | {self.root.name}\n"
            f"{'n_branches':<12} | {self.n_branches}\n"
            f"{'geometry':<12} | {geo_status}\n"
            f"{'length':<12} | {self.total_length:.2f}\n"
            f"{'area':<12} | {self.total_area:.2f}\n"
            f"{'volume':<12} | {self.total_volume:.2f}\n"
            f"{'-'*35}\n"
        )

    def _ordered_node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def _ordered_node_ids_by(self, order: str) -> tuple[int, ...]:
        ordered_ids = self._ordered_node_ids()
        if order == "default":
            return ordered_ids
        if order == "type":
            return tuple(
                sorted(
                    ordered_ids,
                    key=lambda node_id: (
                        self._branch_type_rank(self._get_node(node_id).type),
                        self._get_node(node_id).name,
                    ),
                )
            )
        if order == "depth":
            return tuple(sorted(ordered_ids,
                                key=lambda node_id: (len(self._path_node_ids(node_id)), self._branch_index(node_id))))
        raise ValueError(f"Unsupported branch order {order!r}.")

    @staticmethod
    def _branch_type_rank(branch_type: str) -> tuple[int, int, str]:
        swc_type_map = Morpho._swc_type_map()
        type_bucket = {
            "soma": "soma",
            "axon": "axon",
            "dend": "basal_dendrite",
            "dendrite": "basal_dendrite",
            "basal_dend": "basal_dendrite",
            "basal_dendrite": "basal_dendrite",
            "apical_dend": "apical_dendrite",
            "apical_dendrite": "apical_dendrite",
        }.get(branch_type, branch_type)
        swc_rank = next((code for code, name in swc_type_map.items() if name == type_bucket and code != 0), None)
        if swc_rank is None:
            return (1, 0, type_bucket)
        return (0, swc_rank, type_bucket)

    @staticmethod
    def _swc_type_map() -> dict[int, str]:
        from ..io.swc.types import SWC_TYPE_MAP

        return SWC_TYPE_MAP

    def _branch_index_map(self, *, order: str = "default") -> dict[int, int]:
        return {node_id: index for index, node_id in enumerate(self._ordered_node_ids_by(order))}

    def _branch_index(self, node_id: int, *, order: str = "default") -> int:
        return self._branch_index_map(order=order)[node_id]

    def _node_id_from_index(self, branch_index: int, *, order: str = "default") -> int:
        ordered_ids = self._ordered_node_ids_by(order)
        try:
            return ordered_ids[branch_index]
        except IndexError as exc:
            raise IndexError(f"Branch index {branch_index!r} is out of range.") from exc

    def _path_node_ids(self, node_id: int) -> tuple[int, ...]:
        path = [node_id]
        seen = {node_id}
        node = self._get_node(node_id)
        while node.parent_id is not None:
            if node.parent_id in seen:
                raise ValueError(f"Cycle detected in morphology tree at node {node_id}")
            seen.add(node.parent_id)
            node = self._get_node(node.parent_id)
            path.append(node._node_id)
        return tuple(reversed(path))

    def _get_node(self, node_id: int) -> "MorphoBranch":
        return self._nodes[node_id]

    def _get_branch(self, node_id: int) -> Branch:
        return self._get_node(node_id).branch

    def _resolve_parent(self, parent: ParentRef) -> int:
        if isinstance(parent, MorphoBranch):
            if parent._owner is not self:
                raise ValueError("Parent MorphoBranch belongs to a different Morpho.")
            return parent._node_id
        if isinstance(parent, str):
            if parent not in self._name_to_id:
                raise KeyError(parent)
            return self._name_to_id[parent]
        raise TypeError("parent must be a branch name or MorphoBranch.")

    def _validate_parent_x(self, parent: "MorphoBranch", parent_x: float) -> None:
        if isinstance(parent_x, bool):
            raise TypeError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")
        if parent_x not in (0, 0.0, 0.5, 1, 1.0):
            raise ValueError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")
        if float(parent_x) == 0.5 and parent.type != "soma":
            raise ValueError("parent_x=0.5 is only allowed when the parent branch type is 'soma'.")

    def _validate_child_x(self, child_x: float) -> None:
        if isinstance(child_x, bool):
            raise TypeError(f"child_x must be 0 or 1, got {child_x!r}.")
        if child_x not in (0, 0.0, 1, 1.0):
            raise ValueError(f"child_x must be 0 or 1, got {child_x!r}.")

    def _validate_public_name(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"Branch name {name!r} must be a valid Python identifier.")
        if name.startswith("_"):
            raise ValueError("Branch names starting with '_' are reserved.")
        if name in (_MORPHO_RESERVED_NAMES | _MORPHO_BRANCH_RESERVED_NAMES | _BRANCH_RESERVED_NAMES):
            raise ValueError(f"Branch name {name!r} is reserved by the Morpho API.")

    def _normalize_child_branch(self, value: object) -> Branch:
        if isinstance(value, MorphoBranch):
            raise ValueError(
                "Cannot reattach a MorphoBranch into a Morpho. Reuse the underlying "
                "Branch geometry or create a new Branch instead."
            )
        if not isinstance(value, Branch):
            raise TypeError("Only Branch values participate in morphology syntax sugar.")
        return value

    def _resolve_node_name(self, branch: Branch, *, explicit_name: str | None) -> str:
        node_name = explicit_name if explicit_name is not None else self._allocate_name_for_type(branch.type)
        self._validate_public_name(node_name)
        if node_name in self._name_to_id:
            raise ValueError(f"Branch name {node_name!r} already exists in this Morpho.")
        return node_name

    def _allocate_name_for_type(self, branch_type: str) -> str:
        suffix = self._type_name_counters.get(branch_type, 0)
        while f"{branch_type}_{suffix}" in self._name_to_id:
            suffix += 1
        self._type_name_counters[branch_type] = suffix + 1
        return f"{branch_type}_{suffix}"

    def _register_node(
        self,
        *,
        name: str,
        branch: Branch,
        parent_id: Optional[int],
        parent_x: float,
        child_x: float,
    ) -> int:
        self._validate_public_name(name)
        if name in self._name_to_id:
            raise ValueError(f"Branch name {name!r} already exists in this Morpho.")
        node_id = self._next_id
        self._next_id += 1
        node = MorphoBranch(
            self,
            node_id,
            name=name,
            branch=branch,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=child_x,
        )
        self._nodes[node_id] = node
        self._name_to_id[name] = node_id
        return node_id

    def _insert_child(
        self,
        parent_id: int,
        value: object,
        *,
        child_name: str | None,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        parent = self._get_node(parent_id)
        branch = self._normalize_child_branch(value)
        resolved_name = self._resolve_node_name(branch, explicit_name=child_name)
        if resolved_name in parent._children:
            raise ValueError(
                f"Parent branch {parent.name!r} already has a child named {resolved_name!r}."
            )
        self._validate_parent_x(parent, parent_x)
        self._validate_child_x(child_x)
        node_id = self._register_node(
            name=resolved_name,
            branch=branch,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=child_x,
        )
        parent._children[resolved_name] = node_id
        return self._get_node(node_id)

    def _eq_records(self) -> tuple[tuple[object, ...], ...]:
        records = []
        for branch in self.branches:
            parent = branch.parent
            records.append(
                (
                    branch.name,
                    branch.branch,
                    None if parent is None else parent.name,
                    None if parent is None else float(branch.parent_x),
                    None if parent is None else float(branch.child_x),
                    tuple(child.name for child in branch.children),
                )
            )
        return tuple(records)

    def _get_child(self, parent_id: int, child_name: str) -> "MorphoBranch":
        parent = self._get_node(parent_id)
        if child_name not in parent._children:
            raise AttributeError(f"Branch {parent.name!r} has no child named {child_name!r}.")
        return self._get_node(parent._children[child_name])

    def _format_topology(self, node_id: int, *, prefix: str, is_last: bool) -> list[str]:
        node = self._get_node(node_id)
        branch_prefix = "└── " if is_last else "├── "
        lines = [f"{prefix}{branch_prefix}{node.name}"]
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        child_ids = tuple(node._children.values())
        for index, child_id in enumerate(child_ids):
            lines.extend(
                self._format_topology(
                    child_id,
                    prefix=child_prefix,
                    is_last=index == len(child_ids) - 1,
                )
            )
        return lines


_MorphoBranchAttrGetter = Callable[["MorphoBranch"], object]
_MORPHO_BRANCH_PUBLIC_ATTRS: dict[str, _MorphoBranchAttrGetter] = {
    "branch": lambda node: node._branch,
    "name": lambda node: node._name,
    "parent_id": lambda node: node._parent_id,
    "parent_x": lambda node: None if node._parent_id is None else node._parent_x,
    "child_x": lambda node: None if node._parent_id is None else node._child_x,
}


class MorphoBranch:
    """A tree-local branch node bound to exactly one :class:`Morpho` owner.

    ``MorphoBranch`` provides transparent access to branch geometry
    (via delegation to :class:`Branch`) and tree navigation (parent,
    children).  It also supports syntax sugar for attaching children:

    * **Attribute assignment**: ``parent.dend = Branch(...)``
    * **Subscript syntax**: ``parent[0.5].dend = Branch(...)``

    ``MorphoBranch`` instances are created internally by :class:`Morpho`
    and should not be constructed directly.

    Parameters
    ----------
    owner : Morpho
        The morphology tree that owns this node.
    node_id : int
        Unique node identifier within the tree.
    name : str
        Branch name.
    branch : Branch
        Underlying geometry object.
    parent_id : int or None
        Node ID of the parent (``None`` for the root).
    parent_x : float
        Attachment point on the parent branch.
    child_x : float
        Attachment point on the child branch.

    See Also
    --------
    Morpho : The tree container that owns ``MorphoBranch`` nodes.
    Branch : The immutable geometry wrapped by this node.

    Notes
    -----
    ``MorphoBranch`` delegates attribute access to the underlying
    :class:`Branch` via ``__getattr__``, so all geometry properties
    (``length``, ``area``, ``n_segments``, ``type``, etc.) are accessible
    directly on the node.  Child branches can also be retrieved by name
    (e.g., ``morpho.soma.dend``).

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
        ...     lengths=[50.0] * u.um,
        ...     radii=[2.0, 1.0] * u.um,
        ...     type="dendrite",
        ... )
        >>> morpho = Morpho.from_root(soma, name="soma")
        >>> morpho.soma.dend = dend
        >>> morpho.soma.n_children
        1
        >>> morpho.soma.dend.length
        50.0 * umetre
    """

    def __init__(
        self,
        owner: Morpho,
        node_id: int,
        *,
        name: str,
        branch: Branch,
        parent_id: int | None,
        parent_x: float,
        child_x: float,
    ) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_branch", branch)
        object.__setattr__(self, "_parent_id", parent_id)
        object.__setattr__(self, "_parent_x", parent_x)
        object.__setattr__(self, "_child_x", child_x)
        object.__setattr__(self, "_children", {})

    @property
    def index(self) -> int:
        """Index of this branch in the default ordering.

        Returns
        -------
        int
            Position in the default (by node ID) ordering.

        See Also
        --------
        index_by : Index in a specific ordering.
        """
        return self._owner._branch_index(self._node_id)

    def index_by(self, *, order: str = "default") -> int:
        """Index of this branch in a specific ordering.

        Parameters
        ----------
        order : str
            Ordering strategy: ``"default"`` (by node ID), ``"type"``
            (by SWC type rank then name), or ``"depth"`` (by depth then index).

        Returns
        -------
        int
            Position in the requested ordering.
        """
        return self._owner._branch_index(self._node_id, order=order)

    @property
    def parent(self) -> Optional["MorphoBranch"]:
        """Parent branch node, or ``None`` for the root.

        Returns
        -------
        MorphoBranch or None
            Parent node.
        """
        if self._parent_id is None:
            return None
        return self._owner._get_node(self._parent_id)

    @property
    def children(self) -> tuple["MorphoBranch", ...]:
        """All direct children of this branch.

        Returns
        -------
        tuple of MorphoBranch
            Child branch nodes.
        """
        return tuple(self._owner._get_node(child_id) for child_id in self._children.values())

    @property
    def n_children(self) -> int:
        """Number of direct children.

        Returns
        -------
        int
            Child count.
        """
        return len(self._children)

    def attach(
        self,
        branch: Branch,
        name: str | None = None,
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        """Attach a child branch to this branch.

        Parameters
        ----------
        branch : Branch
            Geometry of the child branch to attach.
        name : str or None
            Name for the child branch.  If ``None``, auto-generated from
            the branch type.
        parent_x : float
            Attachment point on this branch: ``0`` (proximal), ``0.5``
            (midpoint, soma only), or ``1`` (distal, default).
        child_x : float
            Attachment point on the child branch: ``0`` (proximal,
            default) or ``1`` (distal).

        Returns
        -------
        MorphoBranch
            The newly attached child branch node.

        Raises
        ------
        ValueError
            If *parent_x* or *child_x* is invalid.

        Notes
        -----
        ``parent_x=0`` attaches to the proximal end of this branch.  This
        is typically used when this branch is itself a child and you want
        to attach at its connection point rather than its distal end.

        Examples
        --------

        .. code-block:: python

            >>> child = morpho.soma.attach(dend_branch, name="apical")  # doctest: +SKIP
            >>> child.name  # doctest: +SKIP
            'apical'
        """

        return self._owner.attach(
            parent=self,
            child_branch=branch,
            child_name=name,
            parent_x=parent_x,
            child_x=child_x,
        )

    def __getitem__(self, key: object) -> "_MorphAttachPoint":
        parent_x, child_x = _parse_attachment_key(key)
        return _MorphAttachPoint(self, parent_x=parent_x, child_x=child_x)

    def __getattr__(self, name: str) -> object:
        # Underscore-prefixed names never resolve through the dynamic branch /
        # child lookup path. Bailing out early avoids unbounded recursion when
        # pickle (or copy.deepcopy) probes for ``__setstate__`` etc. on a
        # partially-initialised instance whose ``_branch`` / ``_owner`` slots
        # have not yet been restored.
        if name.startswith("_"):
            raise AttributeError(name)
        getter = _MORPHO_BRANCH_PUBLIC_ATTRS.get(name)
        if getter is not None:
            return getter(self)
        try:
            return getattr(self._branch, name)
        except AttributeError:
            return self._owner._get_child(self._node_id, name)

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self._owner._insert_child(self._node_id, value, child_name=key)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | set(_MORPHO_BRANCH_PUBLIC_ATTRS)
            | set(name for name in dir(self._branch) if not name.startswith("_"))
            | set(self._children)
        )

    def __repr__(self) -> str:
        return (
            f"MorphoBranch(name={self.name!r}, type={self.branch.type!r}, "
            f"index={self.index!r})"
        )


class _MorphAttachPoint:
    """Internal helper for `tree.soma[0.5].dend = Branch(...)` syntax."""

    def __init__(self, parent: MorphoBranch, *, parent_x: float, child_x: float) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_parent_x", parent_x)
        object.__setattr__(self, "_child_x", child_x)

    def attach(self, branch: Branch, name: str | None = None) -> MorphoBranch:
        return self._parent.attach(
            branch,
            name=name,
            parent_x=self._parent_x,
            child_x=self._child_x,
        )

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self.attach(value, name=key)

    def __repr__(self) -> str:
        return (
            f"_MorphAttachPoint(parent={self._parent.name!r}, "
            f"parent_x={self._parent_x!r}, child_x={self._child_x!r})"
        )

def _parse_attachment_key(key: object) -> tuple[float, float]:
    if isinstance(key, bool):
        raise TypeError(f"Attachment keys must be numeric, got {key!r}.")
    if isinstance(key, tuple):
        if len(key) != 2:
            raise ValueError("Attachment keys must be [parent_x] or [parent_x, child_x].")
        if isinstance(key[0], bool) or isinstance(key[1], bool):
            raise TypeError(f"Attachment keys must be numeric, got {key!r}.")
        parent_x = float(key[0])
        child_x = float(key[1])
        if parent_x not in (0, 0.0, 0.5, 1, 1.0):
            raise ValueError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")
        if child_x not in (0, 0.0, 1, 1.0):
            raise ValueError(f"child_x must be 0 or 1, got {child_x!r}.")
        return parent_x, child_x
    parent_x = float(key)
    if parent_x not in (0, 0.0, 0.5, 1, 1.0):
        raise ValueError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")
    return parent_x, 0.0
