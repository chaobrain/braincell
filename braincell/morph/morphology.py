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
- `Morphology`: the whole mutable tree
- `MorphoBranch`: a tree-local branch node view
- `MorphoEdge`: read-only branch-to-branch topology edges
- `PrincipalAxes`: what `Morphology.principal_axes()` returns

In normal use, users only need `Morphology` and `MorphoBranch`.
"""

import collections
import contextlib
import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Callable, NamedTuple, Optional, Union

import brainunit as u
import numpy as np
from brainunit import Quantity

from braincell._misc import normalize_param
from braincell.morph.branch import Branch, frustum_areas_um2, frustum_volumes_um3, length_weighted_mean_radius_um

_BRANCH_RESERVED_NAMES = set(Branch.__dataclass_fields__) | {name for name in dir(Branch) if not name.startswith("_")}

ParentRef = Union[str, "MorphoBranch"]
#: ``rename=`` of :meth:`Morphology.graft`: old name to new name, as a mapping or a function.
RenameSpec = Union[Mapping[str, str], Callable[[str], str], None]


class _BranchRecord(NamedTuple):
    """One branch of a subtree read out of the node tables, in node-id order.

    ``parent_name`` is ``None`` for the subtree's root, whose own edge is not
    part of the subtree; ``parent_x`` and ``child_x`` are then unused.
    """

    name: str
    branch: Branch
    parent_name: Optional[str]
    parent_x: float
    child_x: float


@dataclass(frozen=True)
class MorphoEdge:
    """A directed edge between two morphology branches.

    ``MorphoEdge`` is a frozen dataclass representing a parent-child
    connection in a :class:`Morphology` tree.  It records which branches are
    connected and where on each branch the attachment occurs.

    Edges are typically obtained via :attr:`Morphology.edges` rather than
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
    Morphology.edges : Retrieve all edges in a morphology.
    Morphology.attach : Attach a child branch to a parent.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
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
        >>> morph = Morphology.from_root(soma, name="soma")
        >>> morph.soma.dend = dend
        >>> edges = morph.edges
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
    def from_morpho(cls, morpho: "Morphology") -> "MorphoMetric":
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
            f"{'-' * 35}\n"
            f"{'n_branches':<16} | {self.n_branches}\n"
            f"{'n_stems':<16} | {self.n_stems}\n"
            f"{'n_bifurcations':<16} | {self.n_bifurcations}\n"
            f"{'max_branch_order':<16} | {self.max_branch_order}\n"
            f"{'total_length':<16} | {self._format_value(self.total_length)}\n"
            f"{'mean_radius':<16} | {self._format_value(self.mean_radius)}\n"
            f"{'total_area':<16} | {self._format_value(self.total_area)}\n"
            f"{'total_volume':<16} | {self._format_value(self.total_volume)}\n"
            f"{'max_path_dist':<16} | {self._format_value(self.max_path_distance)}\n"
            f"{'-' * 35}\n"
        )


class PrincipalAxes(NamedTuple):
    """Principal axes of a morphology's 3-D points, from :meth:`Morphology.principal_axes`.

    The axes are those of the length-weighted second moments of all
    segments, each taken as a uniform line, about their length-weighted
    centroid. :meth:`Morphology.align_principal_axes` moves every point
    ``p`` to ``axes @ (p - origin)``.

    Attributes
    ----------
    origin : Quantity
        Length-weighted centroid of the soma (every ``"soma"`` branch
        together; the root branch if there is none), shape ``(3,)``, um.
        Alignment puts it at ``(0, 0, 0)``.
    axes : numpy.ndarray
        ``(3, 3)`` float64; row ``i`` is the unit vector of the ``i``-th
        axis, by decreasing variance. The first two point to the side of the
        soma on which the tree's centroid lies; the third is their cross
        product, so ``det(axes) = +1`` and alignment never mirrors.
    variances : Quantity
        Length-weighted variance of the points along each axis, shape
        ``(3,)``, um**2, decreasing and ``>= 0``.
    """

    origin: Quantity
    axes: np.ndarray
    variances: Quantity


class Morphology:
    """Mutable morphology tree for authoring, querying, and visualization.

    ``Morphology`` is the central entry point for building neuron morphologies.
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
    are exposed directly on the ``Morphology`` instance.

    Edits change the tree in place. :meth:`clone` returns an independent
    copy to edit instead, and :meth:`subtree` a copy of one branch with its
    descendants.

    Examples
    --------

    Build a simple morphology by hand:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
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
        >>> morph = Morphology.from_root(soma, name="soma")
        >>> morph.soma.dend = dend
        >>> len(morph)
        2
        >>> morph.n_branches
        2
        >>> morph.total_length
        Quantity(110., "um")

    Load from an SWC file:

    .. code-block:: python

        >>> morph = Morphology.from_swc("neuron.swc")  # doctest: +SKIP
    """

    # Edit-gate state. Class-level defaults rather than ``__init__``
    # assignments so that a tree unpickled from an older release, whose
    # ``__dict__`` lacks them, still reads as unfrozen and editable.
    #: Set on the read-only snapshot an initialized Cell runs on
    #: (:func:`frozen_snapshot`); never carried into a copy (``__getstate__``).
    _frozen: bool = False
    #: Nesting depth of :meth:`_edit`; only the outermost exit bumps.
    _edit_depth: int = 0
    #: Whether any open :meth:`_edit` asked to move the index watermark.
    _edit_reindex: bool = False
    #: Revision of the last edit that moved an existing node's default index.
    _reindex_revision: int = 0

    def __init__(self, *, root_name: str | None, root_branch: Branch) -> None:
        self._nodes: dict[int, MorphoBranch] = {}
        self._name_to_id: dict[str, int] = {}
        self._type_name_counters: dict[str, int] = {}
        self._next_id = 0
        # Caches derived from the node tables, dropped together by
        # ``_invalidate_derived_caches`` at the end of every edit.
        self._ordered_id_cache: dict[str, tuple[int, ...]] = {}
        self._branch_index_cache: dict[str, dict[int, int]] = {}
        self._branch_tuple_cache: dict[str, tuple["MorphoBranch", ...]] = {}
        self._full_point_geometry: bool | None = None
        self._all_segment_arrays_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._all_points_cache: np.ndarray | None = None
        # (revision, MorphologyData) of the last to_vis_data() call.
        self._vis_data_cache: tuple[int, object] | None = None
        # Bumped once per successful edit so consumers holding derived
        # results (e.g. braincell.filter.SelectionCache) can detect staleness.
        self._revision = 0
        # The root is registered outside the edit gate: the object is not
        # constructed yet, so there is nothing to freeze or to roll back.
        self._root_name, counter = self._resolve_node_name(root_branch, explicit_name=root_name)
        self._commit_name_counter(counter)
        self._root_id = self._register_node(
            name=self._root_name,
            branch=root_branch,
            parent_id=None,
            parent_x=1.0,
            child_x=0.0,
        )
        self._invalidate_derived_caches()

    def __getstate__(self) -> dict:
        # ``copy.deepcopy`` and pickle both go through here. A copy is a new
        # tree the caller owns, so it never inherits the frozen flag of a
        # Cell's runtime snapshot.
        state = self.__dict__.copy()
        for key in ("_frozen", "_edit_depth", "_edit_reindex"):
            state.pop(key, None)
        return state

    @classmethod
    def from_root(cls, branch: Branch, *, name: str | None = "soma") -> "Morphology":
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
        Morphology
            New morphology containing only the root.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(
            ...     lengths=[20.0] * u.um,
            ...     radii=[10.0, 10.0] * u.um,
            ...     type="soma",
            ... )
            >>> morph = Morphology.from_root(soma, name="soma")
            >>> morph.root.name
            'soma'
        """

        return cls(root_name=name, root_branch=branch)

    @classmethod
    def from_soma(
        cls,
        *,
        length: Quantity = 10.0 * u.um,
        radius: Quantity = 5.0 * u.um,
        name: str | None = "soma",
    ) -> "Morphology":
        """Create a morphology made of one cylindrical soma branch.

        The shortest way to give a point neuron its geometry. It is not
        named ``soma`` because every public name of :class:`Morphology` is
        reserved and cannot be used as a branch name. The membrane
        area is the cylinder's lateral area :math:`2 \\pi r L`; the defaults
        give the same area as :class:`braincell.SingleCompartment`'s default
        ``length`` and ``radius``.

        Parameters
        ----------
        length : Quantity[u.um], optional
            Positive scalar cylinder length. Default ``10 um``.
        radius : Quantity[u.um], optional
            Positive scalar cylinder radius. Default ``5 um``.
        name : str or None, optional
            Name of the branch, as in :meth:`from_root`. Default ``"soma"``.

        Returns
        -------
        Morphology
            New morphology whose only branch has ``type="soma"``.

        Raises
        ------
        TypeError
            If ``length`` or ``radius`` is not a length quantity (a bare
            number included).
        ValueError
            If ``length`` or ``radius`` is not a scalar or not positive.

        See Also
        --------
        from_root : Build the tree from any root branch.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Morphology
            >>> morph = Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um)
            >>> morph.root.name, morph.root.branch.type
            ('soma', 'soma')
        """
        length = normalize_param(length, name="length", unit=u.um, shape=(), bounds={"gt": 0 * u.um})
        radius = normalize_param(radius, name="radius", unit=u.um, shape=(), bounds={"gt": 0 * u.um})
        branch = Branch.from_lengths(
            lengths=u.Quantity(np.full((1,), length.to_decimal(u.um)), u.um),
            radii=u.Quantity(np.full((2,), radius.to_decimal(u.um)), u.um),
            type="soma",
        )
        return cls.from_root(branch, name=name)

    @classmethod
    def from_swc(
        cls,
        path,
        *,
        options=None,
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
            If ``True``, return a ``(Morphology, SwcReport)`` tuple instead
            of just the morphology.

        Returns
        -------
        Morphology or (Morphology, SwcReport)
            The loaded morphology, optionally with a diagnostic report.

        See Also
        --------
        from_asc : Load from Neurolucida ASC format.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.from_swc("neuron.swc")  # doctest: +SKIP
            >>> morph, report = Morphology.from_swc(
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

    def to_swc(self, path):
        """Write this morphology to a standard SWC file.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path. ``.swc`` is appended when the supplied path
            has no suffix.

        Returns
        -------
        Path
            The final path written.

        Raises
        ------
        ValueError
            If any branch lacks complete point geometry.

        See Also
        --------
        from_swc : Load a morphology from an SWC file.
        save_checkpoint : Write a lossless BrainCell morphology checkpoint.

        Notes
        -----
        SWC does not preserve branch names or explicit boundaries between
        consecutive branches of the same type. Generic ``dendrite`` branches
        are written as SWC type 3 and import as ``basal_dendrite``.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.from_swc("neuron.swc")  # doctest: +SKIP
            >>> morph.to_swc("copy.swc")  # doctest: +SKIP
        """
        from braincell.io.swc.writer import write_swc

        return write_swc(self, path)

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
            If ``True``, return a ``(Morphology, AscReport)`` tuple instead
            of just the morphology.

        Returns
        -------
        Morphology or (Morphology, AscReport)
            The loaded morphology, optionally with a diagnostic report.

        See Also
        --------
        from_swc : Load from SWC format.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.from_asc("neuron.asc")  # doctest: +SKIP
        """

        from braincell.io.asc import AscReader

        return AscReader().read(path, return_report=return_report)

    @classmethod
    def from_neuromorpho(
        cls,
        neuron_id,
        *,
        cache_dir=None,
        mode='neuromorpho',
        client=None,
        return_report: bool = False,
        overwrite: bool = False,
    ):
        """Fetch a NeuroMorpho.Org neuron and return it as a :class:`Morphology`.

        Downloads the standardized CNG SWC into ``cache_dir``
        (defaulting to ``~/.cache/braincell/neuromorpho``) and parses it
        with :meth:`from_swc` using ``mode="neuromorpho"``. Already-cached
        neurons are not re-downloaded unless ``overwrite=True``.

        Parameters
        ----------
        neuron_id : int
            NeuroMorpho.Org numeric identifier (e.g. ``10047``).
        cache_dir : str or Path or None
            Cache root. ``None`` resolves to
            ``~/.cache/braincell/neuromorpho``.
        mode : str or None
            SWC import mode forwarded to :meth:`from_swc`. Defaults to
            ``"neuromorpho"`` (copies soma attachment points into child
            branches).
        client : NeuroMorphoClient or None
            Optional pre-configured client (custom session, retries,
            etc.). A transient client is created when ``None``.
        return_report : bool
            If ``True``, also return the :class:`SwcReport` from parsing.
        overwrite : bool
            If ``True``, re-download even when the file is already cached.

        Returns
        -------
        Morphology or (Morphology, SwcReport)
            The parsed morphology, optionally with a diagnostic report.

        Raises
        ------
        NeuroMorphoNotFoundError
            If ``neuron_id`` does not exist upstream.
        NeuroMorphoHTTPError
            On non-recoverable HTTP errors.

        See Also
        --------
        from_swc : Load from a local SWC file.
        from_asc : Load from a local Neurolucida ASC file.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.from_neuromorpho(10047)  # doctest: +SKIP
        """

        from braincell.io.neuromorpho import load_neuromorpho

        return load_neuromorpho(
            neuron_id,
            cache_dir=cache_dir,
            mode=mode,
            client=client,
            return_report=return_report,
            overwrite=overwrite,
        )

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
        Morphology.load_checkpoint : Inverse operation.
        Morphology.from_swc : Load a morphology from an SWC interchange file.
        Morphology.from_asc : Load a morphology from a Neurolucida ASC file.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.from_swc("neuron.swc")  # doctest: +SKIP
            >>> morph.save_checkpoint("neuron.bcm")  # doctest: +SKIP
        """
        from braincell.io.checkpoint import save_morpho

        return save_morpho(self, path)

    @classmethod
    def load_checkpoint(cls, path) -> "Morphology":
        """Load a morphology from a braincell checkpoint file.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a ``.bcm`` checkpoint produced by
            :meth:`save_checkpoint`.

        Returns
        -------
        Morphology
            The reconstructed morphology, equal to the saved tree under
            :meth:`__eq__`.

        Raises
        ------
        CheckpointError
            If the file is missing, corrupt, or is not a morph checkpoint.

        See Also
        --------
        Morphology.save_checkpoint : Inverse operation.

        Examples
        --------

        .. code-block:: python

            >>> from braincell import Morphology
            >>> morph = Morphology.load_checkpoint("neuron.bcm")  # doctest: +SKIP
        """
        from braincell.io.checkpoint import load_morpho

        return load_morpho(path)

    @property
    def revision(self) -> int:
        """Counter advanced once by every successful edit of this tree.

        ``Morphology`` is mutable, so any consumer that caches a result
        derived from the tree -- CV geometry, a resolved region mask, a
        discretization -- must be able to tell that the tree it derived
        from has since changed. Object identity cannot say that. Pair the
        two: same object *and* same revision means the derived result is
        still valid.

        Each successful edit call (:meth:`attach`, ``morpho.soma.dend =
        branch``, ``morpho.soma[0.5].dend = branch``, :meth:`remove`,
        :meth:`replace`, :meth:`graft`, :meth:`reattach`, :meth:`translate`,
        :meth:`rotate`, :meth:`scale`, :meth:`align_principal_axes`) advances
        it by exactly one, whether it changes the structure or only the
        geometry. Reads never advance it, and an edit that raises leaves it,
        the tree and :meth:`naming_state` unchanged.

        The absolute value carries no meaning, is not stable across
        processes, and cannot be compared between two trees; only comparison
        against a value previously recorded from the same object is.

        Returns
        -------
        int
            Monotonically increasing revision number.

        See Also
        --------
        braincell.filter.SelectionCache : Caches region/locset evaluations
            against this counter.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um,
            ...                            radii=[10.0, 10.0] * u.um, type="soma")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> before = tree.revision
            >>> tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um,
            ...                                      radii=[2.0, 1.0] * u.um,
            ...                                      type="basal_dendrite")
            >>> tree.revision == before + 1
            True
        """
        return self._revision

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
            for node in self.branches
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
        cached = self._branch_tuple_cache.get(order)
        if cached is None:
            cached = tuple(self._nodes[node_id] for node_id in self._ordered_node_ids_by(order))
            self._branch_tuple_cache[order] = cached
        return cached

    @property
    def has_full_point_geometry(self) -> bool:
        cached = self._full_point_geometry
        if cached is None:
            cached = all(
                branch.branch.points_proximal is not None and branch.branch.points_distal is not None
                for branch in self.branches
            )
            self._full_point_geometry = cached
        return cached

    def naming_state(self) -> dict[str, int]:
        """Return the auto-naming counters, one per branch type.

        Branches attached without an explicit name are called
        ``f"{type}_{n}"``. This returns the next ``n`` each type would use,
        so a round trip through a serialization format can resume the
        sequence instead of restarting it and colliding with saved names.

        Returns
        -------
        dict of str to int
            A copy; mutating it does not affect the morphology.

        See Also
        --------
        restore_naming_state : Apply a previously captured mapping.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import braincell
            >>> soma = braincell.Branch.from_lengths(
            ...     lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"
            ... )
            >>> tree = braincell.Morphology.from_root(soma, name="soma")
            >>> dend = braincell.Branch.from_lengths(
            ...     lengths=[10.0] * u.um, radii=[1.0, 0.5] * u.um, type="dendrite"
            ... )
            >>> tree.attach(parent="soma", child_branch=dend).name
            'dendrite_0'
            >>> tree.naming_state()
            {'dendrite': 1}
        """
        return dict(self._type_name_counters)

    def restore_naming_state(self, mapping: dict[str, int]) -> None:
        """Resume the auto-naming counters from a saved mapping.

        Parameters
        ----------
        mapping : dict of str to int
            Branch type to next auto-name suffix, as produced by
            :meth:`naming_state`. Entries are merged into the current
            counters rather than replacing them wholesale.

        Raises
        ------
        TypeError
            If ``mapping`` is not a mapping, or a value is not an integer.
        ValueError
            If a value is negative.

        See Also
        --------
        naming_state : Capture the counters.

        Notes
        -----
        Restoring a counter never renames an existing branch and never
        makes a collision possible: :meth:`attach` still skips any suffix
        already taken. A too-small counter only costs a few extra probes.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(f"naming state must be a mapping, got {type(mapping).__name__}.")
        for branch_type, count in mapping.items():
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise TypeError(f"naming counter for {branch_type!r} must be an int, got {count!r}.")
            if int(count) < 0:
                raise ValueError(f"naming counter for {branch_type!r} must be non-negative, got {count!r}.")
            self._type_name_counters[str(branch_type)] = int(count)

    def _require_full_point_geometry(self, *, feature: str) -> None:
        if not self.has_full_point_geometry:
            raise ValueError(f"{feature} require full point geometry on every branch.")

    def _all_segment_arrays_um(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Memoised: ``total_length``, ``mean_radius``, ``total_area`` and
        # ``total_volume`` each want the same concatenation, so ``.metric``
        # rebuilt it four times.
        cached = self._all_segment_arrays_cache
        if cached is None:
            lengths = []
            radii_proximal = []
            radii_distal = []
            for branch in self.branches:
                lengths_um, r0_um, r1_um = branch.branch._segment_arrays_um()
                lengths.append(lengths_um)
                radii_proximal.append(r0_um)
                radii_distal.append(r1_um)
            cached = (
                np.concatenate(lengths),
                np.concatenate(radii_proximal),
                np.concatenate(radii_distal),
            )
            self._all_segment_arrays_cache = cached
        return cached

    def _all_points_um(self) -> np.ndarray:
        """Every branch's shared points as one ``(n_points, 3)`` array.

        Memoised because ``x_range`` / ``y_range`` / ``z_range`` each need the
        whole cloud, so building it per axis walked every branch three times.
        """
        cached = self._all_points_cache
        if cached is None:
            cached = np.concatenate(
                [np.asarray(branch.branch.points.to_decimal(u.um), dtype=float) for branch in self.branches]
            )
            self._all_points_cache = cached
        return cached

    def _branch_length_um(self, branch_index: int) -> float:
        branch = self.branch(index=branch_index).branch
        return float(np.sum(np.asarray(branch.lengths.to_decimal(u.um), dtype=float)))

    def _root_point_um(self) -> np.ndarray:
        self._require_full_point_geometry(feature="Euclidean distance metrics")
        root = self.root.branch
        return np.asarray(root.points_proximal.to_decimal(u.um), dtype=float)[0]

    def _terminal_branch_indices(self) -> tuple[int, ...]:
        return tuple(branch.index for branch in self.branches if branch.n_children == 0)

    def _point_on_branch_at_x_um(self, branch: Branch, x: float) -> np.ndarray:
        self._require_full_point_geometry(feature="Euclidean distance metrics")
        points_um = np.asarray(branch.points.to_decimal(u.um), dtype=float)
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
        # No trailing return: the loop condition includes
        # ``index == len(lengths_um) - 1``, so the last iteration always returns.

    def _root_attach_distances_um(self, *, exclude_root_soma: bool = False) -> dict[int, float]:
        ordered_ids = self._ordered_node_ids_by("depth")
        distances_um: dict[int, float] = {self.root._node_id: 0.0}
        # The root branch always attaches at its own proximal end.
        attach_x: dict[int, float] = {self.root._node_id: 0.0}

        for node_id in ordered_ids:
            if node_id == self.root._node_id:
                continue

            node = self._get_node(node_id)
            # Only the root has a null parent, and it was skipped above.
            parent_id = node.parent_id
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
            attach_x = 0.0 if branch.parent_id is None else float(branch.child_x)
            distance_um = attach_distances_um[node_id] + abs(1.0 - attach_x) * branch_length_um
            max_distance_um = max(max_distance_um, distance_um)
        return max_distance_um

    def _root_subtree_reference_point_um(
        self,
        terminal_node_id: int,
        *,
        _memo: dict[float, np.ndarray] | None = None,
    ) -> np.ndarray:
        path_node_ids = self._path_node_ids(terminal_node_id)
        if len(path_node_ids) <= 1:
            return self._root_point_um()
        first_child = self._get_node(path_node_ids[1])
        parent_x = float(first_child.parent_x)
        # ``parent_x`` can only be 0.0, 0.5 or 1.0, so a whole morphology has at
        # most three distinct answers; the caller passes a dict to share them
        # across its terminals instead of re-walking the root branch each time.
        if _memo is None:
            return self._point_on_branch_at_x_um(self.root.branch, parent_x)
        if parent_x not in _memo:
            _memo[parent_x] = self._point_on_branch_at_x_um(self.root.branch, parent_x)
        return _memo[parent_x]

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
        reference_memo: dict[float, np.ndarray] = {}
        for branch_index in terminal_branch_indices:
            node_id = self._node_id_from_index(branch_index)
            branch = self.branch(index=branch_index).branch
            tip_points.append(np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)[-1])
            if exclude_root_soma and self.root.type == "soma":
                start_points.append(self._root_subtree_reference_point_um(node_id, _memo=reference_memo))
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
        return u.Quantity(length_weighted_mean_radius_um(*self._all_segment_arrays_um()), u.um)

    @property
    def total_area(self) -> Quantity:
        return u.Quantity(np.sum(frustum_areas_um2(*self._all_segment_arrays_um())), u.um**2)

    @property
    def total_volume(self) -> Quantity:
        return u.Quantity(np.sum(frustum_volumes_um3(*self._all_segment_arrays_um())), u.um**3)

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
        coords = self._all_points_um()[:, axis]
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
        return max(branch.branch_order for branch in self.branches)

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

        return self._get_node(
            self._node_id_from_index(index, order="default" if order is None else order)  # type: ignore[arg-type]
        )

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
        return tuple(self._branch_index(path_node_id) for path_node_id in self._path_node_ids(node_id))

    def topo(self) -> str:
        """Return a line-oriented text view of the branch topology.

        Returns
        -------
        str
            ASCII tree representation showing parent-child relationships.

        Examples
        --------

        .. code-block:: python

            >>> print(morph.topo())  # doctest: +SKIP
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

    def to_vis_data(self):
        """Return this morphology as plotting data for :mod:`braintools.visualize.morphology`.

        The plots in ``braintools.visualize.morphology`` do not read braincell
        objects; they take a neutral, frozen
        :class:`~braintools.visualize.morphology.MorphologyData`. Every plot
        there also accepts a ``Morphology`` directly, because
        ``__morphology_data__`` is this method.

        Branch ``index`` is the position in :attr:`branches`, so evaluated
        region and locset masks of this morphology refer to the same
        branches. Geometry is copied in micrometres. The result is cached
        until the morphology is edited: calling this twice without an edit
        in between returns the same object.

        Returns
        -------
        braintools.visualize.morphology.MorphologyData
            One :class:`~braintools.visualize.morphology.BranchData` per
            branch, with the children in :attr:`MorphoBranch.children` order.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> tree = Morphology.from_root(
            ...     Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"),
            ... )
            >>> tree.attach(
            ...     parent="soma",
            ...     child_branch=Branch.from_lengths(
            ...         lengths=[50.0, 50.0] * u.um, radii=[1.0, 0.8, 0.5] * u.um, type="basal_dendrite",
            ...     ),
            ...     child_name="dend",
            ... )
            MorphoBranch(name='dend', type='basal_dendrite', index=1)
            >>> data = tree.to_vis_data()
            >>> [branch.name for branch in data.branches]
            ['soma', 'dend']
            >>> data.children
            ((1,), ())
            >>> tree.to_vis_data() is data
            True
        """
        cached = self._vis_data_cache
        if cached is not None and cached[0] == self._revision:
            return cached[1]

        from braintools.visualize.morphology import BranchData, MorphologyData

        views = self.branches
        position = {view.name: index for index, view in enumerate(views)}

        def um(quantity):
            return None if quantity is None else quantity.in_unit(u.um)

        branches = []
        for index, view in enumerate(views):
            parent = view.parent
            geometry = view.branch
            branches.append(
                BranchData(
                    index=index,
                    name=view.name,
                    type=str(view.type),
                    parent_index=None if parent is None else position[parent.name],
                    parent_x=None if view.parent_x is None else float(view.parent_x),
                    child_x=None if view.child_x is None else float(view.child_x),
                    lengths=um(geometry.lengths),
                    radii_proximal=um(geometry.radii_proximal),
                    radii_distal=um(geometry.radii_distal),
                    points_proximal=um(geometry.points_proximal),
                    points_distal=um(geometry.points_distal),
                )
            )
        data = MorphologyData(
            branches=tuple(branches),
            root_index=position[self.root.name],
            children=tuple(tuple(position[child.name] for child in view.children) for view in views),
        )
        self._vis_data_cache = (self._revision, data)
        return data

    __morphology_data__ = to_vis_data

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
            default configured via :func:`braintools.visualize.morphology.configure_defaults`.
            The initial default is ``"geometry"``.
        backend : str or None
            Rendering backend name (e.g., ``"pyvista"``).
            Auto-selected when *None*.
        region : RegionMask or None
            Evaluated region mask to attach as an overlay.  Obtain one
            via ``expr.evaluate(morph)`` or ``morph.select(expr)``.
        locset : LocsetMask or None
            Evaluated location-set mask to attach as an overlay.
        values : array-like or None
            Per-branch or per-segment scalar values for colour-mapping.
        chooser : BackendChooser or None
            Explicit backend chooser; overrides *backend* when given.
        notebook : bool or None
            Enable notebook-specific rendering when *True*. Use this in
            Jupyter notebooks or notebook-based docs so the 3-D scene is
            embedded in the cell output; leave it false/omitted when you
            want the raw PyVista plotter in a script.
        jupyter_backend : str or None
            Jupyter backend name for notebook rendering. Use ``"html"``
            for static HTML docs and headless builds; use interactive
            server/client backends such as ``"trame"`` only in a live
            Jupyter session configured for them.
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
            >>> from braincell import Branch, Morphology
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
            >>> morph = Morphology.from_root(soma, name="soma")
            >>> morph.soma.dend = dend
            >>> morph.vis3d()  # doctest: +SKIP

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
            >>> morph.attach(parent="soma", child_branch=basal,
            ...               child_name="basal", parent_x=0.0)  # doctest: +SKIP
            >>> morph.attach(parent="soma", child_branch=axon,
            ...               child_name="axon", parent_x=0.5)  # doctest: +SKIP
            >>> morph.vis3d()  # doctest: +SKIP

        **Highlight a region** — e.g. the full extent of all dendrites:

        .. code-block:: python

            >>> from braincell.filter import branch_in
            >>> region = branch_in("type", ("apical_dendrite", "basal_dendrite")).evaluate(morph)
            >>> morph.vis3d(region=region)  # doctest: +SKIP

        **Mark specific locations** — e.g. branch points and terminals:

        .. code-block:: python

            >>> from braincell.filter import BranchPoints, Terminals
            >>> locset = (BranchPoints() | Terminals()).evaluate(morph)
            >>> morph.vis3d(locset=locset)  # doctest: +SKIP

        **Retrieve the PyVista plotter** for further customisation:

        .. code-block:: python

            >>> plotter = morph.vis3d(return_plotter=True, show=False)  # doctest: +SKIP
            >>> plotter.add_text("My neuron", font_size=12)  # doctest: +SKIP
            >>> plotter.show()  # doctest: +SKIP

        **Static notebook/doc rendering** for headless builds:

        .. code-block:: python

            >>> morph.vis3d(notebook=True, jupyter_backend="html")  # doctest: +SKIP
        """
        from braintools.visualize import morphology as bvm

        result = bvm.plot3d(
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
        branch_type_colors: dict[str, object] | None = None,
        branch_type_edge_colors_2d: dict[str, object] | None = None,
        frustum_edge_linewidth_2d: float | None = None,
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
          ``"fan"``, ``"projected"``, ``"stem"``, ``"balloon"``, or
          ``"radial_360"``.
        * ``shape`` — how those 2-D coordinates are drawn:
          ``"line"`` or ``"frustum"``.

        Parameters
        ----------
        layout : str or None
            2-D layout choice. ``"projected"`` uses projected 3-D point
            geometry. ``"fan"``, ``"stem"``, ``"balloon"``, and
            ``"radial_360"`` use the schematic branch layout pipeline.
            When omitted, uses the global 2-D default configured via
            :func:`braintools.visualize.morphology.configure_defaults`. The initial default
            is ``"fan"``.
        shape : str or None
            2-D drawing shape: ``"line"`` or ``"frustum"``. When omitted,
            uses the global 2-D default configured via
            :func:`braintools.visualize.morphology.configure_defaults`. The initial default
            is ``"frustum"``.
        branch_type_colors : mapping or None
            Per-branch-type colour overrides for this call only. These
            colours drive 2D line rendering, 2D frustum fills, and the
            shared branch-type theme used by 3D and topology plots.
        branch_type_edge_colors_2d : mapping or None
            Per-branch-type frustum border colour overrides for this
            call only. These borders remain active even when ``values``
            drives the frustum fill colours via a colormap.
        frustum_edge_linewidth_2d : float or None
            Frustum border linewidth override for this call only.
        backend : str or None
            Rendering backend name (e.g., ``"matplotlib"``).
            Auto-selected when *None*.
        region : RegionMask or None
            Evaluated region mask to attach as an overlay.  Obtain one
            via ``expr.evaluate(morph)`` or ``morph.select(expr)``.
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

            >>> morph.vis2d(layout="projected", shape="line")  # doctest: +SKIP

        **Stem layout** with line drawing:

        .. code-block:: python

            >>> morph.vis2d(layout="stem", shape="line")  # doctest: +SKIP

        **Stem layout** with frustum drawing:

        .. code-block:: python

            >>> morph.vis2d(layout="stem", shape="frustum")  # doctest: +SKIP
        """
        from braintools.visualize import morphology as bvm

        result = bvm.plot2d(
            self,
            region=region,
            locset=locset,
            values=values,
            layout=layout,
            shape=shape,
            branch_type_colors=branch_type_colors,
            branch_type_edge_colors_2d=branch_type_edge_colors_2d,
            frustum_edge_linewidth_2d=frustum_edge_linewidth_2d,
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
            >>> region = morph.select(branch_in("type", "dendrite"))  # doctest: +SKIP
        """
        from braincell.filter import LocsetExpr, RegionExpr

        if not isinstance(expr, (RegionExpr, LocsetExpr)):
            raise TypeError(f"Morphology.select(...) expects RegionExpr or LocsetExpr. Got {type(expr).__name__!s}.")
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
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell, or *parent* is a view of a branch that an edit removed.

        Notes
        -----
        ``parent_x=0`` attaches to the proximal end of the parent branch.
        This is typically used when the parent is itself a child branch and
        you want to attach at its connection point rather than its distal end.

        Every argument is checked, and the auto-generated name chosen, before
        anything is written: a call that raises leaves the tree,
        :attr:`revision` and :meth:`naming_state` exactly as they were. A
        successful call advances :attr:`revision` by one.

        Examples
        --------

        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
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
            >>> morph = Morphology.from_root(soma, name="soma")
            >>> child = morph.attach(
            ...     parent="soma",
            ...     child_branch=dend,
            ...     child_name="apical",
            ... )
            >>> child.name
            'apical'
        """

        self._require_editable()
        parent_id = self._resolve_parent(parent)
        return self._insert_child(
            parent_id,
            child_branch,
            child_name=child_name,
            parent_x=parent_x,
            child_x=child_x,
        )

    def clone(self) -> "Morphology":
        """Return an independent copy of this morphology.

        The copy compares equal to this tree, has the same auto-naming
        counters (:meth:`naming_state`) and shares its immutable
        :class:`Branch` geometry objects; its branch views and its
        :attr:`revision` are its own. Editing either tree leaves the other
        unchanged. The copy is never frozen, even when this tree is the
        read-only snapshot of an initialized Cell.

        Returns
        -------
        Morphology
            A new tree equal to this one.

        See Also
        --------
        subtree : Copy one branch and its descendants.

        Notes
        -----
        Reading only: :attr:`revision` does not advance. Sharing the
        geometry makes this much cheaper than ``copy.deepcopy``, which copies
        every array of every branch: on a 681-branch reconstruction a clone
        takes about 1.5 ms and a deep copy about 23 ms.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um,
            ...                                      type="basal_dendrite")
            >>> copy = tree.clone()
            >>> copy == tree
            True
            >>> copy.dend.branch is tree.dend.branch
            True
            >>> copy.dend.tip = Branch.from_lengths(lengths=[10.0] * u.um, radii=[1.0, 1.0] * u.um,
            ...                                     type="basal_dendrite")
            >>> tree.n_branches, copy.n_branches
            (2, 3)
        """
        copy = _morpho_from_records(self._subtree_records(self._root_id))
        copy._type_name_counters = dict(self._type_name_counters)
        return copy

    def subtree(self, target: ParentRef) -> "Morphology":
        """Return a new morphology rooted at ``target``, with all its descendants.

        The new tree's root is ``target``'s geometry and name; its
        descendants follow in default order with their names, geometry and
        attachment points (``parent_x``, ``child_x``) unchanged. ``target``'s
        own attachment to its parent is not kept, and the new tree's
        :meth:`naming_state` is empty. Geometry objects are shared, as in
        :meth:`clone`.

        Parameters
        ----------
        target : str or MorphoBranch
            A branch of this tree, by name or by a view that is still in it.

        Returns
        -------
        Morphology
            An independent tree; editing it leaves this one unchanged.

        Raises
        ------
        KeyError
            If no branch is called ``target``.
        TypeError
            If ``target`` is neither a ``str`` nor a :class:`MorphoBranch`.
        ValueError
            If ``target`` is a view of a branch of another morphology.
        RuntimeError
            If ``target`` is a view of a branch that an edit removed from
            this tree; look the branch up again by name.

        See Also
        --------
        clone : Copy the whole tree, auto-naming counters included.

        Notes
        -----
        Reading only: :attr:`revision` does not advance, and a frozen tree
        accepts the call. ``subtree(root)`` equals :meth:`clone` except for
        the auto-naming counters.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = dend
            >>> tree.dend.tip = dend
            >>> part = tree.subtree("dend")
            >>> [branch.name for branch in part.branches]
            ['dend', 'tip']
            >>> part.root.parent is None
            True
            >>> tree.n_branches
            3
        """
        return _morpho_from_records(self._subtree_records(self._resolve_branch_ref(target, what="subtree()")))

    def remove(self, target: ParentRef) -> "Morphology":
        """Remove a branch and all its descendants; return what was removed.

        Parameters
        ----------
        target : str or MorphoBranch
            A non-root branch of this tree, by name or by a view that is
            still in it.

        Returns
        -------
        Morphology
            The removed subtree as a new tree, equal to what
            :meth:`subtree` returned for ``target`` before the call. It can
            be inspected, or attached elsewhere.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``target`` is a view of a branch that
            an edit already removed.
        KeyError
            If no branch is called ``target``.
        TypeError
            If ``target`` is neither a ``str`` nor a :class:`MorphoBranch`.
        ValueError
            If ``target`` is a view of another morphology's branch, or is the
            root: a morphology always has exactly one root. Copy out the part
            to keep with :meth:`subtree` instead.

        See Also
        --------
        subtree : Copy a branch and its descendants without removing them.

        Notes
        -----
        :attr:`revision` advances by one. Every view of a removed branch is
        detached: its name and geometry stay readable, while its index,
        parent and children raise :class:`RuntimeError`. The remaining
        branches keep their order; the default index of each one created
        after the removed branches moves down. Integer branch indices held
        from before the call -- a resolved ``LocsetMask`` placed on a Cell,
        a ``CVPerBranchList`` policy -- are therefore rejected at the Cell's
        next rebuild, while location expressions such as ``at("axon", 0.5)``
        are evaluated again and follow the edit.

        The removed names are free again: attaching a new branch under one of
        them works, and declarations that name it then follow the new
        branch. The auto-naming counters do not move back. A call that
        raises changes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
            >>> axon = Branch.from_lengths(lengths=[50.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = dend
            >>> tree.dend.tip = dend
            >>> tree.soma.axon = axon
            >>> tree.axon.index
            3
            >>> removed = tree.remove("dend")
            >>> [branch.name for branch in removed.branches]
            ['dend', 'tip']
            >>> [(branch.name, branch.index) for branch in tree.branches]
            [('soma', 0), ('axon', 1)]
        """
        self._require_editable()
        node_id = self._resolve_branch_ref(target, what="remove()")
        node = self._nodes[node_id]
        if node._parent_id is None:
            raise ValueError(
                f"Cannot remove the root branch {node._name!r}: a Morphology always has exactly one root. "
                "To keep part of the tree, copy it out with subtree(name)."
            )
        removed = _morpho_from_records(self._subtree_records(node_id))
        node_ids = self._subtree_node_ids(node_id)
        with self._edit(reindex=True):
            self._unregister_subtree(node_ids)
        return removed

    def replace(self, target: ParentRef, new_branch: Branch) -> "MorphoBranch":
        """Replace the geometry of one branch in place.

        Only the geometry changes: the branch keeps its name, its view, its
        parent and its own attachment point, and every child stays attached
        at the same end (``parent_x`` is a normalized position, ``0``, ``0.5``
        or ``1``). The new geometry may have another length, another number
        of segments or another type.

        Parameters
        ----------
        target : str or MorphoBranch
            A branch of this tree, by name or by a view that is still in it.
        new_branch : Branch
            The new geometry.

        Returns
        -------
        MorphoBranch
            The view of ``target``: the same object as before the call, now
            wrapping ``new_branch``.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``target`` is a view of a branch that
            an edit removed.
        KeyError
            If no branch is called ``target``.
        TypeError
            If ``target`` is neither a ``str`` nor a :class:`MorphoBranch`,
            or ``new_branch`` is not a :class:`Branch`.
        ValueError
            If ``target`` is a view of another morphology's branch; if
            ``new_branch`` is a :class:`MorphoBranch` (pass its ``.branch``);
            or if ``new_branch`` is not a soma while ``target`` has children
            at ``parent_x=0.5``, which only a soma allows.

        See Also
        --------
        attach : Add a branch.
        remove : Remove a branch and its descendants.

        Notes
        -----
        :attr:`revision` advances by one, also when ``new_branch`` is the
        geometry the branch already has. Default indices do not change, so
        integer branch indices held by a Cell stay valid; a new type can
        change the type order. No other branch moves: if the new geometry
        ends somewhere else, the children's 3-D points no longer meet it.
        The cable model only reads lengths, radii and topology and is not
        affected, but :meth:`to_swc` then raises ``ValueError`` naming the
        first child that does not connect; move each child's subtree onto
        the new end with :meth:`translate` (``subtree=``). A call that
        raises changes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = dend
            >>> view = tree.dend
            >>> longer = Branch.from_lengths(lengths=[150.0, 50.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um,
            ...                              type="basal_dendrite")
            >>> tree.replace("dend", longer) is view
            True
            >>> float(tree.total_length.to_decimal(u.um))
            220.0
        """
        self._require_editable()
        node_id = self._resolve_branch_ref(target, what="replace()")
        if isinstance(new_branch, MorphoBranch):
            raise ValueError(
                "replace() takes the new geometry as a Branch, not a MorphoBranch view; "
                f"pass its geometry instead, e.g. view.branch for {new_branch._name!r}."
            )
        if not isinstance(new_branch, Branch):
            raise TypeError(f"replace() expects new_branch to be a Branch, got {type(new_branch).__name__}.")
        view = self._nodes[node_id]
        if new_branch.type != "soma":
            at_midpoint = [
                self._nodes[child_id]._name
                for child_id in view._children.values()
                if float(self._nodes[child_id]._parent_x) == 0.5
            ]
            if at_midpoint:
                raise ValueError(
                    f"Cannot replace {view._name!r} with a branch of type {new_branch.type!r}: its children "
                    f"{at_midpoint} attach at parent_x=0.5, which only a soma parent allows."
                )
        with self._edit():
            object.__setattr__(view, "_branch", new_branch)
        return view

    def graft(
        self,
        other: "Morphology",
        *,
        parent: ParentRef,
        parent_x: float = 1.0,
        child_x: float = 0.0,
        rename: RenameSpec = None,
    ) -> "MorphoBranch":
        """Attach a copy of another tree as a subtree of ``parent``.

        ``other``'s root becomes a child of ``parent``, attached as
        :meth:`attach` would attach it; ``other``'s other branches follow in
        its default order with their own attachment points. This tree keeps
        its root. To graft only part of ``other``, pass ``other.subtree(name)``.

        Parameters
        ----------
        other : Morphology
            The tree to copy. It may be this tree or a frozen tree, and is
            never changed: the new branches are copies that share its
            immutable :class:`Branch` geometry.
        parent : str or MorphoBranch
            The branch of this tree to graft onto, by name or by a view that
            is still in it.
        parent_x : float
            Attachment point on ``parent``: ``0`` (proximal), ``0.5``
            (midpoint, soma only) or ``1`` (distal, default).
        child_x : float
            End of ``other``'s root that attaches: ``0`` (proximal, default)
            or ``1`` (distal).
        rename : mapping, callable or None
            New names for ``other``'s branches: a mapping from old to new
            name (names it omits are kept), or a function applied to every
            name, such as ``lambda name: f"{name}_copy"``.

        Returns
        -------
        MorphoBranch
            The view, in this tree, of the copy of ``other``'s root.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``parent`` is a view of a branch that
            an edit removed.
        TypeError
            If ``other`` is not a :class:`Morphology`, ``parent`` is neither a
            ``str`` nor a :class:`MorphoBranch`, ``rename`` is neither a
            mapping, a callable nor ``None``, or it gives a name that is not
            a ``str``.
        KeyError
            If no branch is called ``parent``.
        ValueError
            If ``parent`` is a view of another morphology's branch;
            ``parent_x`` or ``child_x`` is not an allowed value; a
            ``rename`` key is not a branch of ``other``; or a resulting name
            is not a valid identifier, is reserved, is given to two
            branches, or already names a branch of this tree. A clash lists
            every clashing name; resolve it with ``rename``.

        See Also
        --------
        attach : Attach a single branch.
        subtree : Copy part of a tree, e.g. to graft only that part.

        Notes
        -----
        :attr:`revision` advances by one. Every existing branch keeps its
        default index and the new branches come last, so Cell inputs held by
        branch index stay valid; the type and depth orders can change. The
        auto-naming counters do not change: the grafted names are explicit,
        and later auto-names skip any that are taken. 3-D continuity at the
        joint is not checked; :meth:`to_swc` reports a gap, which
        :meth:`translate` with ``subtree=`` closes. Every argument is
        checked before anything is written, so a call that raises changes
        neither tree.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = dend
            >>> tuft = Morphology.from_root(dend, name="tuft")
            >>> tuft.tuft.tip = dend
            >>> tree.graft(tuft, parent="dend").name
            'tuft'
            >>> tree.graft(tuft, parent="soma", parent_x=0.0, rename=lambda name: name + "_b").name
            'tuft_b'
            >>> [branch.name for branch in tree.branches]
            ['soma', 'dend', 'tuft', 'tip', 'tuft_b', 'tip_b']
        """
        self._require_editable()
        if not isinstance(other, Morphology):
            raise TypeError(f"graft() expects other to be a Morphology, got {type(other).__name__}.")
        parent_id = self._resolve_parent(parent)
        self._validate_parent_x(self._nodes[parent_id], parent_x)
        self._validate_child_x(child_x)
        # Read before anything is written: ``other`` may be this tree.
        records = other._subtree_records(other._root_id)
        names = self._graft_names([record.name for record in records], rename)
        root = records[0]._replace(parent_x=float(parent_x), child_x=float(child_x))
        with self._edit():
            new_ids = self._register_records([root, *records[1:]], names=names, parent_ids={None: parent_id})
        return self._nodes[new_ids[0]]

    def reattach(
        self,
        target: ParentRef,
        *,
        parent: ParentRef,
        parent_x: float = 1.0,
        child_x: float | None = None,
    ) -> "MorphoBranch":
        """Move a branch and its descendants to another parent in this tree.

        ``target`` is cut from its parent and attached to ``parent``, with
        its whole subtree. Names, geometry and every attachment inside the
        moved subtree are kept. To insert a branch between two others, attach
        it first and then move the lower part under it::

            morpho.attach(parent="soma", child_branch=hub, child_name="hub")
            morpho.reattach("dend", parent="hub")

        Parameters
        ----------
        target : str or MorphoBranch
            A non-root branch of this tree, by name or by a view that is
            still in it.
        parent : str or MorphoBranch
            The new parent, by name or by view; not ``target`` itself or one
            of its descendants.
        parent_x : float
            Attachment point on ``parent``: ``0`` (proximal), ``0.5``
            (midpoint, soma only) or ``1`` (distal, default). The old value
            is not kept, because it may not be valid on the new parent.
        child_x : float or None
            End of ``target`` that attaches: ``0`` (proximal) or ``1``
            (distal). ``None`` (default) keeps the end ``target`` attached by
            before the move.

        Returns
        -------
        MorphoBranch
            The view of ``target`` in its new place. It is a new object: the
            views held from before the call are detached.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``target`` or ``parent`` is a view of a
            branch that an edit removed.
        KeyError
            If no branch is called ``target`` or ``parent``.
        TypeError
            If ``target`` or ``parent`` is neither a ``str`` nor a
            :class:`MorphoBranch`.
        ValueError
            If ``target`` or ``parent`` is a view of another morphology's
            branch; ``target`` is the root; ``parent`` is ``target`` or one of
            its descendants, which would make a cycle; or ``parent_x`` or
            ``child_x`` is not an allowed value.

        See Also
        --------
        remove : Remove a subtree and return it.
        graft : Attach a copy of another tree.

        Notes
        -----
        A branch's parent never changes, so a move is a removal plus an
        insertion of a copy, made as one edit: :attr:`revision` advances by
        one and the index watermark moves. The moved branches get new views
        and come last in default order, and the branches after them in the
        old order move down, so Cell inputs held by branch index are rejected
        at the Cell's next rebuild, while location expressions such as
        ``at("dend", 0.5)`` follow the move. Moving a subtree back to where
        it was is an edit too: it still goes to the end of the default
        order. 3-D points do not move; :meth:`to_swc` reports a gap at the
        new joint, which :meth:`translate` with ``subtree=`` closes. A call
        that raises changes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = dend
            >>> tree.soma.apical = dend
            >>> tree.dend.tip = dend
            >>> moved = tree.reattach("tip", parent="apical", parent_x=0.0)
            >>> moved.parent.name, moved.parent_x
            ('apical', 0.0)
            >>> [(branch.name, branch.index) for branch in tree.branches]
            [('soma', 0), ('dend', 1), ('apical', 2), ('tip', 3)]
        """
        self._require_editable()
        node_id = self._resolve_branch_ref(target, what="reattach()")
        node = self._nodes[node_id]
        if node._parent_id is None:
            raise ValueError(f"Cannot reattach the root branch {node._name!r}: it has no parent to move it from.")
        parent_id = self._resolve_parent(parent)
        node_ids = self._subtree_node_ids(node_id)
        if parent_id in set(node_ids):
            where = "itself" if parent_id == node_id else f"its descendant {self._nodes[parent_id]._name!r}"
            raise ValueError(
                f"Cannot reattach {node._name!r} under {where}: the new parent is in the subtree being moved, "
                "so the tree would contain a cycle."
            )
        self._validate_parent_x(self._nodes[parent_id], parent_x)
        if child_x is None:
            child_x = node._child_x
        else:
            self._validate_child_x(child_x)
        records = self._subtree_records(node_id)
        root = records[0]._replace(parent_x=float(parent_x), child_x=float(child_x))
        with self._edit(reindex=True):
            self._unregister_subtree(node_ids)
            new_ids = self._register_records(
                [root, *records[1:]], names=[record.name for record in records], parent_ids={None: parent_id}
            )
        return self._nodes[new_ids[0]]

    def translate(self, offset: Quantity, *, subtree: Optional[ParentRef] = None) -> None:
        """Move every 3-D point of the tree, or of one subtree, by ``offset``.

        Only the points move. Segment lengths and radii are kept as they
        are, so lengths, areas, volumes and path distances do not change,
        not even in the last bit. Names, topology and branch indices do not
        change either.

        Parameters
        ----------
        offset : Quantity
            Displacement in a length unit, shape ``(3,)``.
        subtree : str, MorphoBranch or None
            Move only this branch and its descendants, given by name or by a
            view that is still in this tree. ``None`` (default) moves the
            whole tree.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``subtree`` is a view of a branch that
            an edit removed.
        KeyError
            If no branch is called ``subtree``.
        TypeError
            If ``offset`` is a bare number or not a length, or ``subtree`` is
            neither a ``str``, a :class:`MorphoBranch` nor ``None``.
        ValueError
            If ``offset`` does not have shape ``(3,)`` or is not finite;
            ``subtree`` is a view of another morphology's branch; a branch in
            scope has no 3-D points (built with :meth:`Branch.from_lengths`);
            or a moved point is not finite or has lost so much precision that
            it no longer matches the segment lengths.

        See Also
        --------
        rotate : Rotate the tree or a subtree.
        clone : Copy the tree, e.g. to keep the original.

        Notes
        -----
        The tree changes in place and the call returns ``None``; to keep the
        original, :meth:`clone` it first. :attr:`revision` advances by one,
        and integer branch indices held by a Cell stay valid. Views held
        from before the call stay in the tree and read the moved geometry;
        a :class:`Branch` object read before the call is immutable and keeps
        the old points. A Cell has no position of its own: to place a cell,
        transform its morphology.

        ``subtree=`` closes the 3-D gap that :meth:`replace`, :meth:`graft`
        or :meth:`reattach` can leave at a joint, which :meth:`to_swc`
        reports.
        Move the child's subtree by the parent's attachment point minus the
        child's connecting end: ``points_distal[-1]`` of the parent for
        ``parent_x=1`` (``points_proximal[0]`` for ``0``), and
        ``points_proximal[0]`` of the child for ``child_x=0``
        (``points_distal[-1]`` for ``1``). A call that raises changes
        nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch, Morphology
            >>> def branch(points, radii, type):
            ...     return Branch.from_points(points=np.array(points) * u.um, radii=np.array(radii) * u.um,
            ...                               type=type)
            >>> tree = Morphology.from_root(branch([[0.0, 0, 0], [20, 0, 0]], [5.0, 5.0], "soma"), name="soma")
            >>> tree.soma.dend = branch([[20.0, 0, 0], [120, 0, 0]], [2.0, 1.0], "basal_dendrite")
            >>> tree.translate(np.array([5.0, -3.0, 2.0]) * u.um)
            >>> tree.dend.points_distal[-1].to_decimal(u.um).tolist()
            [125.0, -3.0, 2.0]

        Close the gap left by grafting a branch that starts at the origin:

        .. code-block:: python

            >>> tip = Morphology.from_root(branch([[0.0, 0, 0], [0, 40, 0]], [1.0, 0.5], "basal_dendrite"),
            ...                            name="tip")
            >>> _ = tree.graft(tip, parent="dend")
            >>> gap = tree.dend.points_distal[-1] - tree.tip.points_proximal[0]
            >>> tree.translate(gap, subtree="tip")
            >>> tree.tip.points_distal[-1].to_decimal(u.um).tolist()
            [125.0, 37.0, 2.0]
        """
        self._require_editable()
        node_ids = self._transform_scope(subtree, what="translate()")
        offset_um = _length_um(offset, name="offset")
        _check_point_um(offset_um, name="offset")
        self._require_points_in(node_ids, what="translate()")
        self._write_geometry(node_ids, "translate()", lambda points: points + offset_um)

    def rotate(
        self,
        angle: Optional[Quantity] = None,
        axis: Union[str, np.ndarray, None] = None,
        *,
        matrix: Optional[np.ndarray] = None,
        origin: Optional[Quantity] = None,
        subtree: Optional[ParentRef] = None,
    ) -> None:
        """Rotate every 3-D point of the tree, or of one subtree, about ``origin``.

        Give either ``angle`` and ``axis``, or ``matrix``. Each point ``p``
        becomes ``origin + R (p - origin)``. Only the points move: segment
        lengths and radii are kept as they are, so lengths, areas, volumes
        and path distances do not change, not even in the last bit. Names,
        topology and branch indices do not change either.

        Parameters
        ----------
        angle : Quantity
            Rotation angle with an angle unit, e.g. ``90 * u.degree`` or
            ``np.pi / 2 * u.radian``. Positive angles turn counter-clockwise
            when looking from the tip of ``axis`` towards ``origin``
            (right-hand rule).
        axis : {'x', 'y', 'z'} or array_like
            Rotation axis: a coordinate axis, or a direction of three plain
            numbers, normalized internally.
        matrix : array_like
            A ``(3, 3)`` rotation matrix acting on column vectors, such as
            ``scipy.spatial.transform.Rotation.as_matrix()``. It must be
            orthonormal (``max|R^T R - I| <= 1e-6``) with ``det(R) > 0`` and
            is used as given, without re-orthonormalization.
        origin : Quantity or None
            Fixed point of the rotation in a length unit, shape ``(3,)``.
            ``None`` (default) is ``(0, 0, 0)`` um, as in an external
            registration ``x' = R x + t``; to turn a cell about its soma,
            pass the soma's position.
        subtree : str, MorphoBranch or None
            Rotate only this branch and its descendants, given by name or by
            a view that is still in this tree. ``None`` (default) rotates the
            whole tree.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``subtree`` is a view of a branch that
            an edit removed.
        KeyError
            If no branch is called ``subtree``.
        TypeError
            If both forms or neither are given, or only one of ``angle`` and
            ``axis``; ``angle`` is a bare number or not an angle; ``axis`` or
            ``matrix`` is a quantity or not numeric; ``origin`` is a bare
            number or not a length; or ``subtree`` is neither a ``str``, a
            :class:`MorphoBranch` nor ``None``.
        ValueError
            If ``angle`` is not a finite scalar; ``axis`` is another string,
            not three numbers, zero or not finite; ``matrix`` is not
            ``(3, 3)``, not finite, not orthonormal or a reflection;
            ``origin`` does not have shape ``(3,)`` or is not finite;
            ``subtree`` is a view of another morphology's branch; a branch in
            scope has no 3-D points; or a rotated point is not finite.

        See Also
        --------
        translate : Move the tree or a subtree.

        Notes
        -----
        Euler angles are not accepted: their axis order and conventions make
        one triple mean different rotations. Pass a quaternion as
        ``matrix=Rotation.from_quat(q).as_matrix()``. A mirror image
        (``det(R) < 0``) is not a rotation and is rejected.

        The tree changes in place and the call returns ``None``, as for
        :meth:`translate`; :attr:`revision` advances by one. A call that
        raises changes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch, Morphology
            >>> def branch(points, radii, type):
            ...     return Branch.from_points(points=np.array(points) * u.um, radii=np.array(radii) * u.um,
            ...                               type=type)
            >>> tree = Morphology.from_root(branch([[0.0, 0, 0], [20, 0, 0]], [5.0, 5.0], "soma"), name="soma")
            >>> tree.soma.dend = branch([[20.0, 0, 0], [120, 0, 0]], [2.0, 1.0], "basal_dendrite")
            >>> tree.rotate(90 * u.degree, axis="z")
            >>> np.round(tree.dend.points_distal[-1].to_decimal(u.um), 12).tolist()
            [0.0, 120.0, 0.0]

        Turn only ``dend`` back, about its base, with a matrix:

        .. code-block:: python

            >>> back = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            >>> tree.rotate(matrix=back, origin=tree.dend.points_proximal[0], subtree="dend")
            >>> np.round(tree.dend.points_distal[-1].to_decimal(u.um), 12).tolist()
            [100.0, 20.0, 0.0]
        """
        self._require_editable()
        node_ids = self._transform_scope(subtree, what="rotate()")
        theta, axis, matrix = _rotation_arguments(angle, axis, matrix)
        origin_um = None if origin is None else _length_um(origin, name="origin")
        rotation = _rotation_matrix(theta, axis, matrix)
        origin_um = np.zeros(3) if origin_um is None else _check_point_um(origin_um, name="origin")
        self._require_points_in(node_ids, what="rotate()")
        self._write_geometry(node_ids, "rotate()", lambda points: (points - origin_um) @ rotation.T + origin_um)

    def scale(self, factor: float, *, origin: Optional[Quantity] = None, subtree: Optional[ParentRef] = None) -> None:
        """Scale the tree, or one subtree, by the same factor in every direction.

        Segment lengths and both radii are multiplied by ``factor``, and
        every 3-D point ``p`` becomes ``origin + factor (p - origin)``, so
        lengths, areas and volumes are multiplied by ``factor``,
        ``factor**2`` and ``factor**3``. Branches without 3-D points
        (:meth:`Branch.from_lengths`) are scaled too: scaling is defined on
        the cable geometry. Names, topology and branch indices do not change.

        Parameters
        ----------
        factor : float
            Ratio, a plain positive number (``int``, ``float`` or a 0-d
            numpy number); not a quantity.
        origin : Quantity or None
            Fixed point of the scaling in a length unit, shape ``(3,)``.
            ``None`` (default) is ``(0, 0, 0)`` um.
        subtree : str, MorphoBranch or None
            Scale only this branch and its descendants, given by name or by a
            view that is still in this tree. ``None`` (default) scales the
            whole tree.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first), or ``subtree`` is a view of a branch that
            an edit removed.
        KeyError
            If no branch is called ``subtree``.
        TypeError
            If ``factor`` is a ``bool``, a quantity or not a number;
            ``origin`` is a bare number or not a length; or ``subtree`` is
            neither a ``str``, a :class:`MorphoBranch` nor ``None``.
        ValueError
            If ``factor`` is not a single number (scaling is isotropic), or
            is not finite and positive; ``origin`` does not have shape
            ``(3,)`` or is not finite; ``subtree`` is a view of another
            morphology's branch; or a scaled length, radius or point is not
            finite, or not positive where it must be.

        See Also
        --------
        translate : Move the tree or a subtree.
        rotate : Rotate the tree or a subtree.

        Notes
        -----
        A different factor per axis (for instance to undo the z shrinkage of
        a slice) is not offered: a circular cross-section would become an
        ellipse with no single radius, and a branch without points has no
        direction to stretch along.

        A declared Cell rebuilds its CVs from the scaled lengths, so a
        length-based policy such as ``MaxCVLen`` gives a different number of
        CVs, and a selection by absolute length (``branch_range("length",
        ...)``) can select other branches. The tree changes in place and the
        call returns ``None``, as for :meth:`translate`; :attr:`revision`
        advances by one. A call that raises changes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch, Morphology
            >>> soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um,
            ...                            type="soma")
            >>> tree = Morphology.from_root(soma, name="soma")
            >>> tree.soma.dend = Branch.from_lengths(lengths=np.array([100.0]) * u.um,
            ...                                      radii=np.array([2.0, 1.0]) * u.um, type="basal_dendrite")
            >>> tree.scale(2)
            >>> tree.total_length
            Quantity(240., "um")
            >>> tree.scale(0.5, subtree="dend")
            >>> tree.dend.length
            Quantity(100., "um")
        """
        self._require_editable()
        node_ids = self._transform_scope(subtree, what="scale()")
        factor = _plain_ratio(factor, name="factor")
        origin_um = None if origin is None else _length_um(origin, name="origin")
        factor = _check_scale_factor(factor)
        origin_um = np.zeros(3) if origin_um is None else _check_point_um(origin_um, name="origin")
        self._write_geometry(
            node_ids, "scale()", lambda points: origin_um + factor * (points - origin_um), factor=factor
        )

    def principal_axes(self) -> PrincipalAxes:
        """Return the principal axes of the tree's 3-D points.

        Every segment counts as a uniform line from its proximal to its
        distal point, weighted by its length, so the result does not depend
        on how densely a reconstruction is sampled; a zero-length segment
        (a radius jump) counts for nothing. The covariance is taken about
        the length-weighted centroid of the whole tree; the returned
        ``origin`` is that of the soma.

        Returns
        -------
        PrincipalAxes
            ``origin`` (um), ``axes`` (rows, by decreasing variance) and
            ``variances`` (um**2). See :class:`PrincipalAxes`.

        Raises
        ------
        ValueError
            If a branch has no 3-D points (built with
            :meth:`Branch.from_lengths`); the message names up to five.

        See Also
        --------
        align_principal_axes : Move the tree onto these axes.

        Notes
        -----
        The axes are sorted by decreasing variance. The sign of the first
        two makes the tree's centroid ``c`` lie on their positive side,
        ``(c - origin) . a > 0``; for a pyramidal cell the first axis then
        points to the apical side. If ``|(c - origin) . a|`` is at most
        ``1e-9 * sqrt(variances[0])``, the component of ``a`` with the
        largest magnitude is made positive instead (the first one on a
        tie). The third axis is the cross product of the first two. Where
        two variances differ by at most ``1e-9 * variances[0]`` the axes in
        their plane are not unique; the projections of x, y and z onto that
        plane, orthonormalized in that order, are used. Rotating or moving
        the tree first rotates or moves the result with it, except in that
        case.

        Reading only: :attr:`revision` does not advance, and a frozen tree
        accepts the call. For part of a tree, use
        ``morpho.subtree(name).principal_axes()``.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch, Morphology
            >>> def branch(points, radii, type):
            ...     return Branch.from_points(points=np.array(points) * u.um, radii=np.array(radii) * u.um,
            ...                               type=type)
            >>> tree = Morphology.from_root(branch([[0.0, 0, 0], [0, 20, 0]], [5.0, 5.0], "soma"), name="soma")
            >>> tree.soma.dend = branch([[0.0, 20, 0], [0, 120, 0]], [2.0, 1.0], "basal_dendrite")
            >>> pa = tree.principal_axes()
            >>> pa.origin.to_decimal(u.um).tolist()
            [0.0, 10.0, 0.0]
            >>> pa.axes[0].tolist()
            [0.0, 1.0, 0.0]
        """
        origin_um, axes, variances_um2 = self._principal_axes_um(what="principal_axes()")
        return PrincipalAxes(Quantity(origin_um, u.um), axes, Quantity(variances_um2, u.um**2))

    def align_principal_axes(self) -> None:
        """Move the tree onto its principal axes, the soma at the origin.

        Every 3-D point ``p`` becomes ``axes @ (p - origin)``, with
        ``origin`` and ``axes`` from :meth:`principal_axes` before the
        call: the soma's centroid goes to ``(0, 0, 0)`` and the axes of
        largest, middle and smallest variance to x, y and z. Afterwards
        :meth:`principal_axes` gives the origin ``(0, 0, 0)``, the identity
        matrix and the same variances. Segment lengths and radii are kept as
        they are, as for :meth:`rotate`.

        Raises
        ------
        RuntimeError
            If this morphology is the read-only snapshot of an initialized
            Cell (checked first).
        ValueError
            If a branch has no 3-D points; the message names up to five.

        See Also
        --------
        principal_axes : The origin and axes this applies.
        rotate : Turn the tree afterwards, e.g. the first axis onto y.

        Notes
        -----
        The axis signs (:meth:`principal_axes`) make the result independent
        of how the tree was posed before, so two reconstructions of similar
        cells end up facing the same way. To point the first axis along y
        instead of x, follow with ``rotate(90 * u.degree, axis="z")``.

        The whole tree moves in one edit: the call returns ``None`` and
        :attr:`revision` advances by one. A call that raises changes
        nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import numpy as np
            >>> from braincell import Branch, Morphology
            >>> def branch(points, radii, type):
            ...     return Branch.from_points(points=np.array(points) * u.um, radii=np.array(radii) * u.um,
            ...                               type=type)
            >>> tree = Morphology.from_root(branch([[0.0, 0, 0], [0, 20, 0]], [5.0, 5.0], "soma"), name="soma")
            >>> tree.soma.dend = branch([[0.0, 20, 0], [0, 120, 0]], [2.0, 1.0], "basal_dendrite")
            >>> tree.align_principal_axes()
            >>> tree.dend.points_distal[-1].to_decimal(u.um).tolist()
            [110.0, 0.0, 0.0]
        """
        self._require_editable()
        origin_um, axes, _ = self._principal_axes_um(what="align_principal_axes()")
        self._write_geometry(
            sorted(self._nodes), "align_principal_axes()", lambda points: (points - origin_um) @ axes.T
        )

    def _principal_axes_um(self, *, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`principal_axes` as float64 arrays (um, -, um**2); ``what`` names the caller in errors."""
        self._require_points_in(
            sorted(self._nodes),
            what=what,
            hint="Use the part that has points, e.g. morpho.subtree(name).principal_axes(), or rebuild "
            "these branches with Branch.from_points.",
        )
        views = self.branches
        proximal, distal, lengths = _segment_points_um(views)
        weights = lengths / lengths.sum()
        midpoints = 0.5 * (proximal + distal)
        chords = distal - proximal
        centroid = weights @ midpoints
        centred = midpoints - centroid
        # Second moment of a uniform segment about the centroid: L (m m^T + d d^T / 12).
        covariance = (centred * weights[:, None]).T @ centred + (chords * weights[:, None]).T @ chords / 12.0
        variances, vectors = np.linalg.eigh(covariance)
        order = np.argsort(variances)[::-1]
        variances, vectors = variances[order], _settle_degenerate_axes(variances[order], vectors[:, order])

        soma = [view for view in views if view.type == "soma"] or [self.root]
        soma_proximal, soma_distal, soma_lengths = _segment_points_um(soma)
        origin = (soma_lengths / soma_lengths.sum()) @ (0.5 * (soma_proximal + soma_distal))
        offset = centroid - origin
        tie = _PRINCIPAL_TIE_RTOL * np.sqrt(max(float(variances[0]), 0.0))
        axes = []
        for vector in vectors[:, :2].T:
            projection = float(offset @ vector)
            if abs(projection) > tie:
                axes.append(vector * np.sign(projection))
            else:
                axes.append(vector * np.sign(vector[int(np.argmax(np.abs(vector)))]))
        axes.append(np.cross(axes[0], axes[1]))
        # Adding 0.0 turns the -0.0 entries a sign flip or cross product leaves into 0.0.
        return origin, np.vstack(axes) + 0.0, np.clip(variances, 0.0, None)

    def _transform_scope(self, subtree: Optional[ParentRef], *, what: str) -> list[int]:
        """Node ids a transform acts on, ascending: the whole tree, or ``subtree`` and its descendants."""
        if subtree is None:
            return sorted(self._nodes)
        return self._subtree_node_ids(self._resolve_branch_ref(subtree, what=what, role="subtree"))

    def _require_points_in(
        self,
        node_ids: list[int],
        *,
        what: str,
        hint: str = "Pass subtree= to transform only a part that has points, or rebuild these branches "
        "with Branch.from_points.",
    ) -> None:
        """Raise ``ValueError`` naming up to five branches in ``node_ids`` that have no 3-D points."""
        missing = [
            self._nodes[node_id]._name for node_id in node_ids if self._nodes[node_id]._branch.points_proximal is None
        ]
        if missing:
            shown = ", ".join(repr(name) for name in missing[:5])
            if len(missing) > 5:
                shown += f" and {len(missing) - 5} more"
            raise ValueError(
                f"{what} needs 3-D points, but {len(missing)} branch(es) in its scope have none "
                f"(built from lengths only): {shown}. {hint}"
            )

    def _write_geometry(
        self,
        node_ids: list[int],
        what: str,
        move: Callable[[np.ndarray], np.ndarray],
        *,
        factor: Optional[float] = None,
    ) -> None:
        """Map the points of every branch in ``node_ids`` by ``move`` and write them in one edit.

        ``factor`` (scaling only) also multiplies the lengths and radii.
        Every new :class:`Branch` is built, and so validated, before the
        gate opens; inside it only the views' ``_branch`` slots are
        written, which cannot fail. Without ``factor`` the lengths and radii
        arrays are kept, so a rigid move leaves every length-derived value
        bit for bit as it was.
        """
        moved = []
        for node_id in node_ids:
            view = self._nodes[node_id]
            try:
                moved.append(_moved_branch(view._branch, move, factor=factor))
            except ValueError as exc:
                raise ValueError(f"{what} would give branch {view._name!r} invalid geometry: {exc}") from exc
        with self._edit():
            for node_id, branch in zip(node_ids, moved):
                object.__setattr__(self._nodes[node_id], "_branch", branch)

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
        if not isinstance(other, Morphology):
            return NotImplemented
        return self._eq_records() == other._eq_records()

    def __repr__(self) -> str:
        geo_status = "complete 3d points" if self.has_full_point_geometry else "incomplete 3d points"
        return f"Morphology(root={self.root.name!r}, n_branches={self.n_branches}, geometry = {geo_status})"

    def __str__(self) -> str:
        """Return a formatted summary with key metrics."""
        geo_status = "complete 3d points" if self.has_full_point_geometry else "incomplete 3d points"
        return (
            f"{'-' * 35}\n"
            f"{'root':<12} | {self.root.name}\n"
            f"{'n_branches':<12} | {self.n_branches}\n"
            f"{'geometry':<12} | {geo_status}\n"
            f"{'length':<12} | {self.total_length:.2f}\n"
            f"{'area':<12} | {self.total_area:.2f}\n"
            f"{'volume':<12} | {self.total_volume:.2f}\n"
            f"{'-' * 35}\n"
        )

    def _invalidate_derived_caches(self) -> None:
        """Drop every cache derived from this tree and advance :attr:`revision`.

        ``Morphology`` is mutable, so a stale entry here is a correctness
        bug rather than a missed optimization. The caches cover node order
        and indices, the view tuples, the point-geometry flag, the
        concatenated segment and point arrays, and the ``to_vis_data``
        snapshot; every one of them can depend on any node's name, type,
        parent or geometry, so all are dropped together.

        Only :meth:`_edit` (once, at its outermost exit) and ``__init__``
        call this. An edit that wrote to the node tables without going
        through the gate would leave every consumer keyed on
        (object, revision) serving the pre-edit result.
        """
        self._ordered_id_cache.clear()
        self._branch_index_cache.clear()
        self._branch_tuple_cache.clear()
        self._full_point_geometry = None
        self._all_segment_arrays_cache = None
        self._all_points_cache = None
        self._vis_data_cache = None
        self._revision += 1

    def _require_editable(self) -> None:
        if self._frozen:
            raise RuntimeError(
                "This Morphology is the read-only snapshot of an initialized Cell and cannot be edited. "
                "Edit the morphology the Cell was declared with, then call reset() and init_state() "
                "on the Cell to apply the change."
            )

    @contextlib.contextmanager
    def _edit(self, *, reindex: bool = False) -> Iterator[None]:
        """The one gate every edit of this tree goes through.

        An edit method checks all of its arguments and picks any
        auto-generated name *before* entering, so a rejected call writes
        nothing; inside, it performs only writes that cannot fail. On exit
        every derived cache is dropped and :attr:`revision` advances by
        exactly one. Nested gates advance it once, at the outermost exit.

        Parameters
        ----------
        reindex : bool
            Set by an edit that can change the default index of a node that
            existed before it (a removal does; an append does not). The
            watermark read by :meth:`_reindexed_after` then moves to the new
            revision.

        Raises
        ------
        RuntimeError
            On entry, if the tree is frozen; nothing has been written then.

        Notes
        -----
        If a write inside the gate raises anyway, the caches are still
        dropped, the revision still advances and a requested watermark
        still moves, so that no consumer keeps serving a result derived
        from the tree before the partial write.
        """
        self._require_editable()
        outermost = self._edit_depth == 0
        self._edit_depth += 1
        if reindex:
            self._edit_reindex = True
        try:
            yield
        finally:
            self._edit_depth -= 1
            if outermost:
                self._invalidate_derived_caches()
                if self._edit_reindex:
                    self._reindex_revision = self._revision
                    self._edit_reindex = False

    def _reindexed_after(self, revision: int) -> bool:
        """Whether an edit after ``revision`` moved an existing node's default index.

        A consumer that holds integer branch indices (a ``LocsetMask``
        place rule, a per-branch CV list) records the revision it resolved
        them at. Appending, replacing geometry in place and transforming
        keep every existing default index, so those indices stay valid; a
        removal (:meth:`remove`) or a move (:meth:`reattach`) shifts the
        later ones down and the consumer must re-resolve or fail.
        """
        return self._reindex_revision > revision

    def _ordered_node_ids_by(self, order: str) -> tuple[int, ...]:
        cached = self._ordered_id_cache.get(order)
        if cached is None:
            cached = self._compute_ordered_node_ids(order)
            self._ordered_id_cache[order] = cached
        return cached

    def _compute_ordered_node_ids(self, order: str) -> tuple[int, ...]:
        ordered_ids = tuple(sorted(self._nodes))
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
            return tuple(
                sorted(
                    ordered_ids, key=lambda node_id: (len(self._path_node_ids(node_id)), self._branch_index(node_id))
                )
            )
        raise ValueError(f"Unsupported branch order {order!r}.")

    @staticmethod
    def _branch_type_rank(branch_type: str) -> tuple[int, int, str]:
        swc_type_map = Morphology._swc_type_map()
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
        from braincell.io.swc.types import SWC_TYPE_MAP

        return SWC_TYPE_MAP

    def _branch_index_map(self, *, order: str = "default") -> dict[int, int]:
        cached = self._branch_index_cache.get(order)
        if cached is None:
            cached = {node_id: index for index, node_id in enumerate(self._ordered_node_ids_by(order))}
            self._branch_index_cache[order] = cached
        return cached

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

    def _resolve_parent(self, parent: ParentRef) -> int:
        return self._resolve_branch_ref(parent, what="attaching to it", role="parent")

    def _resolve_branch_ref(self, ref: ParentRef, *, what: str, role: str = "target") -> int:
        """Node id of a branch of this tree given by name or by view; writes nothing.

        Raises ``KeyError`` for an unknown name, ``TypeError`` for anything
        but a ``str`` or :class:`MorphoBranch`, ``ValueError`` for a view of
        another tree and ``RuntimeError`` (naming ``what``) for a view that
        an edit detached from this one.
        """
        if isinstance(ref, MorphoBranch):
            if ref._owner is not self:
                raise ValueError(f"{role.capitalize()} MorphoBranch belongs to a different Morphology.")
            ref._require_attached(what)
            return ref._node_id
        if isinstance(ref, str):
            if ref not in self._name_to_id:
                raise KeyError(ref)
            return self._name_to_id[ref]
        raise TypeError(f"{role} must be a branch name or MorphoBranch, got {type(ref).__name__}.")

    def _subtree_node_ids(self, node_id: int) -> list[int]:
        """Node ids of ``node_id`` and all its descendants, ascending.

        Iterative: a recursive walk overflows on reconstructions a few
        hundred branches deep. Ascending node id is creation order, which
        lists every parent before its children.
        """
        found = []
        stack = [node_id]
        while stack:
            current = stack.pop()
            found.append(current)
            stack.extend(self._nodes[current]._children.values())
        found.sort()
        return found

    def _subtree_records(self, node_id: int) -> list[_BranchRecord]:
        """Read the subtree at ``node_id`` out of the node tables, root first, in node-id order.

        Reads only ``_nodes`` and the nodes' own fields, never a cache, and
        writes nothing, so the records stay valid while the tree is edited.
        """
        records = []
        for current in self._subtree_node_ids(node_id):
            node = self._nodes[current]
            if current == node_id:
                records.append(_BranchRecord(node._name, node._branch, None, 1.0, 0.0))
            else:
                parent_name = self._nodes[node._parent_id]._name
                records.append(
                    _BranchRecord(node._name, node._branch, parent_name, float(node._parent_x), float(node._child_x))
                )
        return records

    def _register_records(
        self,
        records: list[_BranchRecord],
        *,
        names: list[str],
        parent_ids: dict[Optional[str], int],
    ) -> list[int]:
        """Add ``records`` as new nodes; the caller has checked everything and holds the gate.

        ``names[i]`` is the name ``records[i]`` gets in this tree.
        ``parent_ids`` maps a record's ``parent_name`` to the node id of its
        parent here; it is extended with every node added, so a record's
        parent may be an earlier record. Each new node takes the next node id
        and goes last in its parent's children, which keeps every child table
        in node-id order. Returns the new node ids, in the order of
        ``records``.
        """
        new_ids = []
        for record, name in zip(records, names):
            parent_id = parent_ids[record.parent_name]
            node_id = self._register_node(
                name=name,
                branch=record.branch,
                parent_id=parent_id,
                parent_x=record.parent_x,
                child_x=record.child_x,
            )
            self._nodes[parent_id]._children[name] = node_id
            parent_ids[record.name] = node_id
            new_ids.append(node_id)
        return new_ids

    def _graft_names(self, old_names: list[str], rename: RenameSpec) -> list[str]:
        """The names :meth:`graft` gives ``other``'s branches, checked against this tree; writes nothing."""
        if rename is None:
            new_names = list(old_names)
        elif isinstance(rename, Mapping):
            known = set(old_names)
            unknown = [key for key in rename if key not in known]
            if unknown:
                raise ValueError(f"rename names branches that are not in the grafted tree: {unknown}.")
            new_names = [rename.get(name, name) for name in old_names]
        elif callable(rename):
            new_names = [rename(name) for name in old_names]
        else:
            raise TypeError(f"rename must be a mapping, a callable or None, got {type(rename).__name__}.")
        for old_name, new_name in zip(old_names, new_names):
            if not isinstance(new_name, str):
                raise TypeError(f"rename must give a str name, got {new_name!r} for {old_name!r}.")
        for new_name in new_names:
            self._validate_public_name(new_name)
        repeated = sorted(name for name, count in collections.Counter(new_names).items() if count > 1)
        if repeated:
            raise ValueError(f"rename gives several grafted branches the same name: {repeated}.")
        taken = [name for name in new_names if name in self._name_to_id]
        if taken:
            raise ValueError(
                f"Branch names {taken} already exist in this Morphology; pass rename= to graft() "
                "to give the grafted branches other names."
            )
        return new_names

    def _unregister_subtree(self, node_ids: list[int]) -> None:
        """Drop a subtree from the tables; the caller holds the gate.

        ``node_ids`` is :meth:`_subtree_node_ids` of a non-root node, so its
        first entry is the subtree's root. Its views detach: the
        ``_nodes`` entry no longer is them.
        """
        root = self._nodes[node_ids[0]]
        del self._nodes[root._parent_id]._children[root._name]
        for node_id in node_ids:
            del self._name_to_id[self._nodes.pop(node_id)._name]

    def _validate_parent_x(self, parent: "MorphoBranch", parent_x: float) -> None:
        _check_parent_x_value(parent_x)
        if float(parent_x) == 0.5 and parent.type != "soma":
            raise ValueError("parent_x=0.5 is only allowed when the parent branch type is 'soma'.")

    def _validate_child_x(self, child_x: float) -> None:
        _check_child_x_value(child_x)

    def _validate_public_name(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"Branch name {name!r} must be a valid Python identifier.")
        if name.startswith("_"):
            raise ValueError("Branch names starting with '_' are reserved.")
        if name in _ALL_RESERVED_NAMES:
            raise ValueError(f"Branch name {name!r} is reserved by the Morphology API.")

    def _normalize_child_branch(self, value: object) -> Branch:
        if isinstance(value, MorphoBranch):
            raise ValueError(
                "Cannot reattach a MorphoBranch into a Morphology. Reuse the underlying "
                "Branch geometry or create a new Branch instead."
            )
        if not isinstance(value, Branch):
            raise TypeError("Only Branch values participate in morphology syntax sugar.")
        return value

    def _resolve_node_name(self, branch: Branch, *, explicit_name: str | None) -> tuple[str, Optional[tuple[str, int]]]:
        """Validate the name a new node will get, without writing anything.

        Returns the name and, for an auto-generated one, the
        ``(type, next_suffix)`` counter update to apply with
        :meth:`_commit_name_counter` once every other check has passed.
        """
        if explicit_name is None:
            node_name, counter = self._allocate_name_for_type(branch.type)
        else:
            node_name, counter = explicit_name, None
        self._validate_public_name(node_name)
        if node_name in self._name_to_id:
            raise ValueError(f"Branch name {node_name!r} already exists in this Morphology.")
        return node_name, counter

    def _allocate_name_for_type(self, branch_type: str) -> tuple[str, tuple[str, int]]:
        """Pick the next free ``{type}_{n}`` name; the counter is not advanced here."""
        suffix = self._type_name_counters.get(branch_type, 0)
        while f"{branch_type}_{suffix}" in self._name_to_id:
            suffix += 1
        return f"{branch_type}_{suffix}", (branch_type, suffix + 1)

    def _commit_name_counter(self, counter: Optional[tuple[str, int]]) -> None:
        if counter is not None:
            branch_type, next_suffix = counter
            self._type_name_counters[branch_type] = next_suffix

    def _register_node(
        self,
        *,
        name: str,
        branch: Branch,
        parent_id: Optional[int],
        parent_x: float,
        child_x: float,
    ) -> int:
        """Add one node to the tables; the caller has validated everything and holds the gate."""
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
        # Every check, the auto-name included, runs before the gate: a
        # rejected attach must not consume a name suffix.
        self._require_editable()
        parent = self._get_node(parent_id)
        branch = self._normalize_child_branch(value)
        resolved_name, counter = self._resolve_node_name(branch, explicit_name=child_name)
        if resolved_name in parent._children:
            raise ValueError(f"Parent branch {parent.name!r} already has a child named {resolved_name!r}.")
        self._validate_parent_x(parent, parent_x)
        self._validate_child_x(child_x)
        with self._edit():
            self._commit_name_counter(counter)
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
        """Render a subtree as ``tree(1)``-style lines, iteratively.

        Deliberately not recursive: one Python frame per branch overflowed the
        default recursion limit on morphologies deeper than a few hundred
        branches, which is well within the range of real reconstructions.
        The explicit stack carries the per-node prefix bookkeeping that a
        generic depth-first walk cannot supply.
        """
        lines: list[str] = []
        stack = [(node_id, prefix, is_last)]
        while stack:
            current_id, current_prefix, current_is_last = stack.pop()
            node = self._get_node(current_id)
            branch_prefix = "└── " if current_is_last else "├── "
            lines.append(f"{current_prefix}{branch_prefix}{node.name}")
            child_prefix = f"{current_prefix}{'    ' if current_is_last else '│   '}"
            child_ids = tuple(node._children.values())
            # Pushed in reverse so the stack pops children back in order.
            for index in range(len(child_ids) - 1, -1, -1):
                stack.append((child_ids[index], child_prefix, index == len(child_ids) - 1))
        return lines


_MorphoBranchAttrGetter = Callable[["MorphoBranch"], object]
_MORPHO_BRANCH_PUBLIC_ATTRS: dict[str, _MorphoBranchAttrGetter] = {
    "branch": lambda node: node._branch,
    "name": lambda node: node._name,
    "parent_id": lambda node: node._parent_id,
    "parent_x": lambda node: None if node._parent_id is None else node._parent_x,
    "child_x": lambda node: None if node._parent_id is None else node._child_x,
}
#: The dynamic attributes a detached view still answers: they describe the
#: branch itself, not its place in a tree.
_MORPHO_BRANCH_DETACHED_ATTRS = frozenset({"branch", "name"})


class MorphoBranch:
    """A tree-local branch node bound to exactly one :class:`Morphology` owner.

    ``MorphoBranch`` provides transparent access to branch geometry
    (via delegation to :class:`Branch`) and tree navigation (parent,
    children).  It also supports syntax sugar for attaching children:

    * **Attribute assignment**: ``parent.dend = Branch(...)``
    * **Subscript syntax**: ``parent[0.5].dend = Branch(...)``

    ``MorphoBranch`` instances are created internally by :class:`Morphology`
    and should not be constructed directly.

    Parameters
    ----------
    owner : Morphology
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
    Morphology : The tree container that owns ``MorphoBranch`` nodes.
    Branch : The immutable geometry wrapped by this node.

    Notes
    -----
    ``MorphoBranch`` delegates attribute access to the underlying
    :class:`Branch` via ``__getattr__``, so all geometry properties
    (``length``, ``area``, ``n_segments``, ``type``, etc.) are accessible
    directly on the node.  Child branches can also be retrieved by name
    (e.g., ``morph.soma.dend``).

    A view is live: while its branch is in the tree it always reports the
    current geometry, index, parent and children, so an index read before
    an edit may differ from one read after it.  A branch's parent never
    changes.  If an edit removes the branch (:meth:`Morphology.remove`), or
    moves it with a new view (:meth:`Morphology.reattach`), the view is
    *detached*: its ``name``, ``branch`` and the delegated
    geometry stay readable, while every question about its place in the
    tree (``index``, ``branch_id``, ``index_by``, ``branch_order``,
    ``parent``, ``parent_id``, ``parent_x``, ``child_x``, ``children``,
    ``n_children``, child lookup) and every attachment through it raise
    :class:`RuntimeError`; look the branch up again by name on the current
    tree.  Names are the stable way to refer to a branch across edits.

    The view's own attributes are read-only: assigning to a name starting
    with ``_``, or deleting any attribute, raises :class:`AttributeError`,
    because a change made that way would bypass the morphology's
    :attr:`~Morphology.revision` and leave cached results stale.
    ``del morpho.soma.dend`` does not remove ``dend`` either; call
    :meth:`Morphology.remove`.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
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
        >>> morph = Morphology.from_root(soma, name="soma")
        >>> morph.soma.dend = dend
        >>> morph.soma.n_children
        1
        >>> morph.soma.dend.length
        Quantity(50., "um")
    """

    def __init__(
        self,
        owner: Morphology,
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

    def _is_attached(self) -> bool:
        return self._owner._nodes.get(self._node_id) is self

    def _require_attached(self, what: str) -> Morphology:
        """Return the owner, or raise if an edit has removed this branch from it."""
        if not self._is_attached():
            raise RuntimeError(
                f"MorphoBranch {self._name!r} is detached: an edit removed it from its Morphology, "
                f"so {what} is not available. Its name and geometry are still readable; look the "
                f"branch up again by name on the current tree, e.g. morpho.branch(name={self._name!r})."
            )
        return self._owner

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
        return self._require_attached(".index")._branch_index(self._node_id)

    @property
    def branch_id(self) -> int:
        """Alias of :attr:`index`, the branch's position in default ordering.

        Returns
        -------
        int
            Position in the default (by node ID) ordering.

        See Also
        --------
        index : The canonical spelling of the same value.
        """
        return self._require_attached(".branch_id")._branch_index(self._node_id)

    @property
    def branch_order(self) -> int:
        """Number of parent hops between this branch and the root.

        Returns
        -------
        int
            ``0`` for the root branch, ``1`` for its children, and so on.

        See Also
        --------
        Morphology.path_to_root : Full root-to-branch index path.
        """
        return len(self._require_attached(".branch_order")._path_node_ids(self._node_id)) - 1

    @property
    def n_tapers(self) -> int:
        """Number of tapering segments, an alias of ``n_segments``.

        Returns
        -------
        int
            Segment count of the underlying :class:`Branch` geometry.
        """
        return self._branch.n_segments

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
        return self._require_attached(".index_by()")._branch_index(self._node_id, order=order)

    @property
    def parent(self) -> Optional["MorphoBranch"]:
        """Parent branch node, or ``None`` for the root.

        Returns
        -------
        MorphoBranch or None
            Parent node.
        """
        owner = self._require_attached(".parent")
        if self._parent_id is None:
            return None
        return owner._get_node(self._parent_id)

    @property
    def children(self) -> tuple["MorphoBranch", ...]:
        """All direct children of this branch.

        Returns
        -------
        tuple of MorphoBranch
            Child branch nodes.
        """
        owner = self._require_attached(".children")
        return tuple(owner._get_node(child_id) for child_id in self._children.values())

    @property
    def n_children(self) -> int:
        """Number of direct children.

        Returns
        -------
        int
            Child count.
        """
        self._require_attached(".n_children")
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
        RuntimeError
            If an edit has removed this branch from its morphology, or the
            morphology is the read-only snapshot of an initialized Cell.

        Notes
        -----
        ``parent_x=0`` attaches to the proximal end of this branch.  This
        is typically used when this branch is itself a child and you want
        to attach at its connection point rather than its distal end.

        Examples
        --------

        .. code-block:: python

            >>> child = morph.soma.attach(dend_branch, name="apical")  # doctest: +SKIP
            >>> child.name  # doctest: +SKIP
            'apical'
        """

        return self._require_attached(".attach()").attach(
            parent=self,
            child_branch=branch,
            child_name=name,
            parent_x=parent_x,
            child_x=child_x,
        )

    def __getitem__(self, key: object) -> "_MorphAttachPoint":
        self._require_attached(f"[{key!r}] attachment")
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
            if name not in _MORPHO_BRANCH_DETACHED_ATTRS:
                self._require_attached(f".{name}")
            return getter(self)
        try:
            return getattr(self._branch, name)
        except AttributeError:
            pass
        return self._require_attached(f"child lookup .{name}")._get_child(self._node_id, name)

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            # ``__init__`` sets these with ``object.__setattr__``; copy and
            # pickle restore ``__dict__`` without calling this method.
            raise AttributeError(
                f"MorphoBranch attribute {key!r} is read-only: changing it directly would bypass the "
                "Morphology's revision and leave cached results (indices, metrics, selections, "
                "discretizations) stale."
            )
        self._require_attached(f"attaching child {key!r}")._insert_child(self._node_id, value, child_name=key)

    def __delattr__(self, key: str) -> None:
        message = f"MorphoBranch attributes cannot be deleted (tried {key!r})."
        if key in self._children:
            # ``del morpho.soma.dend`` reads like a removal; removing a whole
            # subtree implicitly is too easy to do by accident, so point to
            # the explicit method instead.
            message += f" To remove the branch and its subtree, call Morphology.remove({key!r})."
        raise AttributeError(message)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | set(_MORPHO_BRANCH_PUBLIC_ATTRS)
            | set(name for name in dir(self._branch) if not name.startswith("_"))
            | set(self._children)
        )

    def __repr__(self) -> str:
        if not self._is_attached():
            return f"MorphoBranch(name={self._name!r}, type={self._branch.type!r}, detached)"
        return f"MorphoBranch(name={self._name!r}, type={self._branch.type!r}, index={self.index!r})"


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
            f"_MorphAttachPoint(parent={self._parent.name!r}, parent_x={self._parent_x!r}, child_x={self._child_x!r})"
        )


def _check_parent_x_value(parent_x: object) -> None:
    """Reject a ``parent_x`` outside the three attachment points.

    The soma-only rule for ``0.5`` needs the parent branch and so stays on
    :meth:`Morphology._validate_parent_x`; this is the part that
    ``morph.soma[x]`` and ``attach(parent_x=x)`` must agree on.
    """
    if isinstance(parent_x, bool):
        raise TypeError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")
    if parent_x not in (0, 0.0, 0.5, 1, 1.0):
        raise ValueError(f"parent_x must be 0, 0.5, or 1, got {parent_x!r}.")


def _check_child_x_value(child_x: object) -> None:
    """Reject a ``child_x`` that is neither end of the child branch."""
    if isinstance(child_x, bool):
        raise TypeError(f"child_x must be 0 or 1, got {child_x!r}.")
    if child_x not in (0, 0.0, 1, 1.0):
        raise ValueError(f"child_x must be 0 or 1, got {child_x!r}.")


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
        _check_parent_x_value(parent_x)
        _check_child_x_value(child_x)
        return parent_x, child_x
    parent_x = float(key)
    _check_parent_x_value(parent_x)
    return parent_x, 0.0


#: Unit vectors of the named rotation axes.
_AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
#: Largest ``max|R^T R - I|`` a rotation matrix may have. A 30-degree matrix
#: built in float32 is off by 2.7e-8 and passes; ``2 * I`` is off by 3.
_ROTATION_ORTHONORMAL_ATOL = 1e-6


def _length_um(value: object, *, name: str) -> np.ndarray:
    """``value`` in micrometres as float64; a bare number or another unit raises ``TypeError``.

    Only the unit is checked here, so that every argument's type is checked
    before any argument's value (:meth:`Morphology.rotate`).
    """
    return np.asarray(normalize_param(value, name=name, unit=u.um).to_decimal(u.um), dtype=float)


def _check_point_um(point_um: np.ndarray, *, name: str) -> np.ndarray:
    """Reject a point or displacement that is not three finite numbers."""
    if point_um.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {point_um.shape}.")
    if not np.all(np.isfinite(point_um)):
        raise ValueError(f"{name} must be finite, got {point_um.tolist()} um.")
    return point_um


def _plain_numbers(value: object, *, name: str, what: str) -> np.ndarray:
    """``value`` as a float64 array; a quantity or anything non-numeric raises ``TypeError``."""
    if isinstance(value, Quantity):
        raise TypeError(f"{name} is {what} and takes plain numbers, not a quantity; got {value!r}.")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} is {what} and takes plain numbers, got {value!r}.") from exc


def _rotation_arguments(
    angle: object, axis: object, matrix: object
) -> tuple[Optional[np.ndarray], Union[str, np.ndarray, None], Optional[np.ndarray]]:
    """Type checks of :meth:`Morphology.rotate`: ``(angle in rad, axis, matrix)``, one form set."""
    if matrix is not None:
        if angle is not None or axis is not None:
            raise TypeError("rotate() takes either angle and axis, or matrix, not both.")
        return None, None, _plain_numbers(matrix, name="matrix", what="a (3, 3) rotation matrix")
    if angle is None or axis is None:
        given = "angle" if angle is not None else "axis" if axis is not None else "neither"
        raise TypeError(f"rotate() needs both angle and axis, or matrix; got {given}.")
    theta = np.asarray(normalize_param(angle, name="angle", unit=u.radian).to_decimal(u.radian), dtype=float)
    if not isinstance(axis, str):
        axis = _plain_numbers(axis, name="axis", what="a direction")
    return theta, axis, None


def _rotation_matrix(
    theta: Optional[np.ndarray], axis: Union[str, np.ndarray, None], matrix: Optional[np.ndarray]
) -> np.ndarray:
    """Value checks of :meth:`Morphology.rotate`; the rotation as a (3, 3) float64 matrix."""
    if matrix is not None:
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must be finite.")
        error = float(np.max(np.abs(matrix.T @ matrix - np.eye(3))))
        if error > _ROTATION_ORTHONORMAL_ATOL:
            raise ValueError(
                f"matrix is not a rotation: max|R^T R - I| = {error:.3g} exceeds {_ROTATION_ORTHONORMAL_ATOL:g}."
            )
        if np.linalg.det(matrix) <= 0.0:
            raise ValueError("matrix is a reflection (det < 0); rotate() takes proper rotations only.")
        return matrix
    if theta.shape != ():
        raise ValueError(f"angle must be a scalar, got shape {theta.shape}.")
    if not np.isfinite(theta):
        raise ValueError(f"angle must be finite, got {float(theta)!r} rad.")
    if isinstance(axis, str):
        if axis not in _AXIS_VECTORS:
            raise ValueError(f"axis must be 'x', 'y', 'z' or three numbers, got {axis!r}.")
        direction = np.array(_AXIS_VECTORS[axis])
    else:
        if axis.shape != (3,):
            raise ValueError(f"axis must be 'x', 'y', 'z' or three numbers, got shape {axis.shape}.")
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"axis must be a finite, non-zero direction, got {axis.tolist()}.")
        direction = axis / norm
    # Rodrigues' formula.
    cross = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def _plain_ratio(value: object, *, name: str) -> np.ndarray:
    """Type checks of a ratio: a ``bool``, a quantity or anything non-numeric raises ``TypeError``."""
    if isinstance(value, (bool, np.bool_)) or isinstance(value, Quantity):
        raise TypeError(f"{name} is a plain ratio and takes a number, got {value!r}.")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} is a plain ratio and takes a number, got {value!r}.") from exc
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} is a plain ratio and takes a number, got {value!r}.")
    return array.astype(float)


def _check_scale_factor(factor: np.ndarray) -> float:
    """Value checks of :meth:`Morphology.scale`'s ``factor``: one finite, positive number."""
    if factor.shape != ():
        raise ValueError(
            f"factor must be a single number: scale() is isotropic, the same factor along x, y and z; "
            f"got shape {factor.shape}."
        )
    value = float(factor)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"factor must be finite and > 0, got {value!r}.")
    return value


#: Relative tolerance of :meth:`Morphology.principal_axes` for equal variances and an unbiased sign.
_PRINCIPAL_TIE_RTOL = 1e-9


def _segment_points_um(views) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proximal points, distal points and lengths of every segment of ``views``, in um."""
    proximal = np.concatenate([np.asarray(v._branch.points_proximal.to_decimal(u.um), dtype=float) for v in views])
    distal = np.concatenate([np.asarray(v._branch.points_distal.to_decimal(u.um), dtype=float) for v in views])
    lengths = np.concatenate([np.asarray(v._branch.lengths.to_decimal(u.um), dtype=float) for v in views])
    return proximal, distal, lengths


def _settle_degenerate_axes(variances: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Fix the eigenvectors (columns) of numerically equal variances.

    Within a run of variances that differ by at most
    ``_PRINCIPAL_TIE_RTOL * variances[0]``, any orthonormal basis of the
    eigenspace is valid, and which one ``numpy.linalg.eigh`` returns depends
    on the implementation; aligning twice could then swap axes. The basis
    used is the projections of x, y and z onto the eigenspace,
    orthonormalized in that order (Gram-Schmidt), skipping any that
    vanish.
    """
    settled = vectors.copy()
    tolerance = _PRINCIPAL_TIE_RTOL * max(float(variances[0]), 0.0)
    start = 0
    while start < 3:
        stop = start
        while stop + 1 < 3 and variances[stop] - variances[stop + 1] <= tolerance:
            stop += 1
        if stop > start:
            space = vectors[:, start : stop + 1]
            projector = space @ space.T
            basis = []
            for unit in np.eye(3):
                candidate = projector @ unit
                for chosen in basis:
                    candidate = candidate - (chosen @ candidate) * chosen
                norm = float(np.linalg.norm(candidate))
                if norm > 1e-6:
                    basis.append(candidate / norm)
                if len(basis) == stop - start + 1:
                    break
            settled[:, start : stop + 1] = np.column_stack(basis)
        start = stop + 1
    return settled


def _moved_branch(
    branch: Branch, move: Callable[[np.ndarray], np.ndarray], *, factor: Optional[float] = None
) -> Branch:
    """``branch`` with its points (in um) mapped by ``move``, and its lengths and radii times ``factor``.

    Without ``factor`` the lengths and radii arrays are kept as they are. A
    branch without points has only its lengths and radii scaled. Built
    through the ``Branch`` constructor, so the new points are checked
    against the segment lengths: a move that loses precision (an offset of
    1e308 um) is rejected there.
    """
    changes = {}
    # An overflow is reported below as a ValueError, so numpy need not warn about it.
    with np.errstate(over="ignore", invalid="ignore"):
        if branch.points_proximal is not None:
            proximal = move(np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float))
            distal = move(np.asarray(branch.points_distal.to_decimal(u.um), dtype=float))
            if not (np.all(np.isfinite(proximal)) and np.all(np.isfinite(distal))):
                raise ValueError("the moved points are not finite.")
            changes.update(points_proximal=Quantity(proximal, u.um), points_distal=Quantity(distal, u.um))
        if factor is not None:
            for field in ("lengths", "radii_proximal", "radii_distal"):
                values = np.asarray(getattr(branch, field).to_decimal(u.um), dtype=float) * factor
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"the scaled {field} are not finite.")
                changes[field] = Quantity(values, u.um)
    return dataclasses.replace(branch, **changes)


def _morpho_from_records(records: list[_BranchRecord]) -> "Morphology":
    """Build a new tree from subtree records (root first, node-id order).

    The one copying path: :meth:`Morphology.clone`,
    :meth:`Morphology.subtree` and :func:`frozen_snapshot` all build through
    it. Names are checked against the reserved set (a tree unpickled from
    an older release may hold a name a newer method has since reserved)
    before the branches are added under one edit gate. The new tree's
    auto-naming counters are empty.
    """
    root = records[0]
    tree = Morphology.from_root(root.branch, name=root.name)
    names = [record.name for record in records[1:]]
    for name in names:
        tree._validate_public_name(name)
    with tree._edit():
        tree._register_records(records[1:], names=names, parent_ids={root.name: tree._root_id})
    return tree


def frozen_snapshot(morpho: "Morphology") -> "Morphology":
    """Return a read-only copy of ``morpho`` for an initialized Cell to run on.

    The copy equals ``morpho`` (:meth:`Morphology.__eq__`) and every read
    works on it, but every edit -- :meth:`Morphology.attach`,
    :meth:`MorphoBranch.attach`, attribute and subscript attachment --
    raises :class:`RuntimeError` before writing anything. ``morpho`` itself
    stays editable. Its auto-naming counters are not copied.
    :meth:`Morphology.clone`, ``copy.deepcopy`` or pickling the snapshot
    gives an ordinary, editable tree.
    """
    snapshot = _morpho_from_records(morpho._subtree_records(morpho._root_id))
    snapshot._frozen = True
    return snapshot


# --------------------------------------------------------------------------- #
# Reserved branch names
#
# A branch name that collides with an attribute of ``Morphology`` or
# ``MorphoBranch`` would be accepted into the tree and then be unreachable by
# attribute access, because ``__getattr__`` only fires when normal lookup
# fails. These sets are therefore *derived* rather than hand-listed: a
# hand-maintained copy drifts the moment a public method is added, and it had
# already drifted, leaving two names shadowable.
#
# Defined here, below the classes, because that is the earliest point at which
# ``dir()`` can see them. ``_MORPHO_BRANCH_PUBLIC_ATTRS`` must be unioned in
# explicitly: those five names are served dynamically through
# ``MorphoBranch.__getattr__`` and so never appear in ``dir()``.
# --------------------------------------------------------------------------- #

_ALL_RESERVED_NAMES = frozenset(
    {name for name in dir(Morphology) if not name.startswith("_")}
    | {name for name in dir(MorphoBranch) if not name.startswith("_")}
    | set(_MORPHO_BRANCH_PUBLIC_ATTRS)
    | _BRANCH_RESERVED_NAMES
)
