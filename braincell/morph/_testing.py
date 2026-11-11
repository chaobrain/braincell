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

"""Shared fixture builders for ``braincell.morph`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

Each ``make_*`` branch builder returns the canonical single-segment branch
of one type, with the dimensions the morph tests have always used. They
exist because those five literals were spelled out 41 times across
``morphology_test.py`` alone, which made the dimensions look meaningful
when they are merely arbitrary — a test that genuinely depends on a
specific radius should pass it explicitly.

The second half is the before/after edit table. Every derived result of a
``Morphology`` -- its own order and metric caches, a ``SelectionCache``, a
Cell's discretization -- is keyed on (tree, revision), so an edit that
leaves one of them stale is a silent wrong answer. :data:`EDIT_CASES` lists
one row per kind of edit; each consumer layer warms its caches, applies the
edit and compares what it then reads against the same consumer on
:func:`cold_rebuild`, a copy built from the raw node tables only. The
layers are morph (:func:`assert_morpho_fresh`, the tests in this package),
the selection cache (``braincell.filter``), the Cell before
``init_state`` (the Cell tests, through :func:`assert_layer_fresh`) and the
initialized Cell. A new edit adds a row here and every layer covers it.
:data:`SABOTAGE_CASES` holds edits that skip the gate on purpose; every
layer must report them, which shows its checks can fail.
"""

import dataclasses
from typing import Callable, Mapping

import brainunit as u
import numpy as np

from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology

__all__ = [
    "EDIT_CASES",
    "EDIT_KINDS",
    "EditCase",
    "SABOTAGE_CASES",
    "assert_layer_fresh",
    "assert_morpho_fresh",
    "cold_rebuild",
    "make_apical",
    "make_axon",
    "make_basal",
    "make_deep_chain_tree",
    "make_dendrite",
    "make_soma",
    "make_soma_dend_tree",
    "make_tree_s",
    "make_tree_t",
    "morph_fingerprint",
    "simulate_remove_leaf",
    "simulate_replace_geometry",
    "stretched",
]


def make_soma(*, length: float = 20.0, radius: float = 10.0) -> Branch:
    """A cylindrical soma branch of uniform radius."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=[radius, radius] * u.um, type="soma")


def make_dendrite(*, length: float = 60.0, radii: tuple[float, float] = (2.0, 1.0)) -> Branch:
    """A tapering branch of the untyped ``"dendrite"`` class."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="dendrite")


def make_basal(*, length: float = 60.0, radii: tuple[float, float] = (2.0, 1.0)) -> Branch:
    """A tapering ``"basal_dendrite"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="basal_dendrite")


def make_apical(*, length: float = 30.0, radii: tuple[float, float] = (1.0, 0.6)) -> Branch:
    """A tapering ``"apical_dendrite"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="apical_dendrite")


def make_axon(*, length: float = 40.0, radii: tuple[float, float] = (0.8, 0.4)) -> Branch:
    """A tapering ``"axon"`` branch."""
    return Branch.from_lengths(lengths=[length] * u.um, radii=list(radii) * u.um, type="axon")


def make_soma_dend_tree(
    *,
    soma_length: float = 20.0,
    dend_type: str = "basal_dendrite",
    dend_length: float = 30.0,
) -> Morphology:
    """A soma with one tapering dendrite attached at its distal end.

    The smallest tree with a parent/child relationship, which is what most
    structural assertions need. Three packages built this independently --
    ``filter`` parameterized, ``vis`` and ``_discretization`` with
    byte-identical hard-coded bodies -- and all three produced the same
    two-branch tree. It lives here because the object it builds is a
    morphology with no selection, rendering, or discretization content.

    Parameters
    ----------
    soma_length : float
        Soma length in micrometres. The radius is :func:`make_soma`'s
        default 10 um.
    dend_type : str
        Branch type for the child, e.g. ``"basal_dendrite"`` or
        ``"apical_dendrite"``.
    dend_length : float
        Dendrite length in micrometres. The radii taper 2 um -> 1 um
        regardless of length.

    Returns
    -------
    Morphology
        A two-branch tree named ``soma`` -> ``dend``.
    """
    tree = Morphology.from_root(make_soma(length=soma_length), name="soma")
    child = Branch.from_lengths(
        lengths=[dend_length] * u.um,
        radii=[2.0, 1.0] * u.um,
        type=dend_type,
    )
    tree.attach(parent="soma", child_branch=child, child_name="dend", parent_x=1.0)
    return tree


def make_root_split_tree() -> Morphology:
    """Soma with one apical dendrite and one axon, both length-only."""
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[25.0] * u.um, radii=[2.0, 1.5] * u.um, type="apical_dendrite")
    axon = Branch.from_lengths(lengths=[18.0] * u.um, radii=[1.0, 0.8] * u.um, type="axon")
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def make_projected_node_tree() -> Morphology:
    """Soma and an apical dendrite with explicit 3D points (the dendrite has two segments)."""
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [20.0, 40.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


def make_deep_chain_tree(n_branches: int = 1200) -> Morphology:
    """An unbranched chain of ``n_branches`` branches hanging off a soma.

    Reconstructions of long, thin neurites routinely produce chains far
    deeper than CPython's default recursion limit, so this fixture exists
    to pin the tree walks in :mod:`braincell.morph` and
    :mod:`braintools.visualize.morphology` as iterative. The default depth is comfortably past the ~400-branch
    point where a recursive walk raises ``RecursionError``, while staying
    small enough to build and lay out in well under a second.

    Parameters
    ----------
    n_branches : int
        Total branch count, soma included. Must be at least 1.

    Returns
    -------
    Morphology
        A chain ``soma -> seg_0 -> seg_1 -> ...``, every dendrite branch
        carrying two 15 µm / 10 µm segments.
    """
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches!r}.")
    tree = Morphology.from_root(make_soma(), name="soma")
    parent = "soma"
    for index in range(n_branches - 1):
        child = Branch.from_lengths(
            lengths=[15.0, 10.0] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        name = f"seg_{index}"
        tree.attach(parent=parent, child_branch=child, child_name=name, parent_x=1.0)
        parent = name
    return tree


def simulate_remove_leaf(morpho: Morphology, name: str) -> None:
    """Remove the leaf branch ``name`` through the edit gate, as a delete will.

    There is no public delete yet (MORPH-2). This stand-in lets tests see
    what a removal does to held views (they detach), to the default indices
    of later branches (they shift down) and to the index watermark (it
    moves): it checks first, then edits the private node tables inside
    ``Morphology._edit(reindex=True)``. Replace it with the public delete
    once that lands.

    Parameters
    ----------
    morpho : Morphology
        Tree to edit in place.
    name : str
        Name of a non-root branch with no children.

    Raises
    ------
    KeyError
        If no branch is called ``name``.
    ValueError
        If ``name`` is the root or has children.
    RuntimeError
        If ``morpho`` is frozen.
    """
    node_id = morpho._name_to_id[name]
    node = morpho._nodes[node_id]
    if node._parent_id is None:
        raise ValueError(f"cannot remove the root branch {name!r}.")
    if node._children:
        raise ValueError(f"branch {name!r} has children; only a leaf can be removed.")
    parent = morpho._nodes[node._parent_id]
    with morpho._edit(reindex=True):
        del parent._children[name]
        del morpho._nodes[node_id]
        del morpho._name_to_id[name]


def simulate_replace_geometry(morpho: Morphology, name: str, branch: Branch) -> None:
    """Replace the geometry of branch ``name`` in place through the edit gate.

    There is no public in-place replace yet (MORPH-2). This stand-in swaps
    the :class:`Branch` a view wraps inside ``Morphology._edit()``: names,
    parents and default indices stay, the revision advances by one and
    every derived cache is dropped. Replace it with the public method once
    that lands.

    Parameters
    ----------
    morpho : Morphology
        Tree to edit in place.
    name : str
        Name of the branch whose geometry is replaced.
    branch : Branch
        New geometry.

    Raises
    ------
    KeyError
        If no branch is called ``name``.
    TypeError
        If ``branch`` is not a :class:`Branch`.
    RuntimeError
        If ``morpho`` is frozen.
    """
    view = morpho._nodes[morpho._name_to_id[name]]
    if not isinstance(branch, Branch):
        raise TypeError(f"branch must be a Branch, got {type(branch).__name__}.")
    with morpho._edit():
        object.__setattr__(view, "_branch", branch)


def stretched(branch: Branch, factor: float = 2.0) -> Branch:
    """Return ``branch`` scaled by ``factor`` about its proximal end, radii and type kept.

    Parameters
    ----------
    branch : Branch
        Geometry with 3-D points.
    factor : float
        Scale factor for every segment length.

    Raises
    ------
    ValueError
        If ``branch`` has no 3-D points.
    """
    if branch.points_proximal is None:
        raise ValueError("stretched() needs a branch built from 3-D points.")
    proximal = np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
    distal = np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)
    points = np.vstack([proximal[:1], distal])
    return Branch.from_points(
        points=(points[0] + factor * (points - points[0])) * u.um,
        radii_proximal=branch.radii_proximal,
        radii_distal=branch.radii_distal,
        type=branch.type,
    )


# --------------------------------------------------------------------------- #
# Before/after edit table
# --------------------------------------------------------------------------- #


def _point_branch(points, radii, type: str) -> Branch:
    return Branch.from_points(points=np.asarray(points, dtype=float) * u.um, radii=list(radii) * u.um, type=type)


def make_tree_t() -> Morphology:
    """The default tree of the edit table: five branches, all with 3-D points.

    ``soma`` (0,0,0)-(20,0,0) um, radius 5 um; ``dend`` (20,0,0)-(60,10,0)-
    (120,0,5), radii 2/1.5/1 um; its children ``d1`` (120,0,5)-(150,30,5) and
    ``d2`` (120,0,5)-(150,-30,5), radii 0.8 -> 0.6 um; ``axon`` (0,0,0)-
    (-30,0,0), radius 0.5 um, on the soma's proximal end. Default indices
    are soma 0, dend 1, d1 2, d2 3, axon 4, so removing the leaf ``d2``
    moves ``axon`` down to 3. The fork, the axon (a second type for the
    type order), the proximal attachment and the z offset give every
    fingerprint key something that an edit can change.
    """
    tree = Morphology.from_root(_point_branch([[0, 0, 0], [20, 0, 0]], (5.0, 5.0), "soma"), name="soma")
    tree.attach(
        parent="soma",
        child_branch=_point_branch([[20, 0, 0], [60, 10, 0], [120, 0, 5]], (2.0, 1.5, 1.0), "basal_dendrite"),
        child_name="dend",
    )
    for name, y in (("d1", 30.0), ("d2", -30.0)):
        tree.attach(
            parent="dend",
            child_branch=_point_branch([[120, 0, 5], [150, y, 5]], (0.8, 0.6), "basal_dendrite"),
            child_name=name,
        )
    tree.attach(
        parent="soma",
        child_branch=_point_branch([[0, 0, 0], [-30, 0, 0]], (0.5, 0.5), "axon"),
        child_name="axon",
        parent_x=0.0,
    )
    return tree


def make_tree_s() -> Morphology:
    """Three straight branches on the x axis, for checks with round numbers.

    ``soma`` (0,0,0)-(20,0,0) um, radius 5 um; ``dend`` (20,0,0)-(60,0,0)-
    (120,0,0), radii 2/1.5/1 um; ``axon`` (0,0,0)-(-30,0,0), radius 0.5 um,
    on the soma's proximal end. Total length and x range are both 150 um;
    stretching ``dend`` by 2 makes both 250 um.
    """
    tree = Morphology.from_root(_point_branch([[0, 0, 0], [20, 0, 0]], (5.0, 5.0), "soma"), name="soma")
    tree.attach(
        parent="soma",
        child_branch=_point_branch([[20, 0, 0], [60, 0, 0], [120, 0, 0]], (2.0, 1.5, 1.0), "basal_dendrite"),
        child_name="dend",
    )
    tree.attach(
        parent="soma",
        child_branch=_point_branch([[0, 0, 0], [-30, 0, 0]], (0.5, 0.5), "axon"),
        child_name="axon",
        parent_x=0.0,
    )
    return tree


#: The kinds of edit a row can stand for.
EDIT_KINDS = ("append", "delete", "replace", "splice", "transform")


@dataclasses.dataclass(frozen=True)
class EditCase:
    """One row of the before/after edit table.

    Attributes
    ----------
    name : str
        Label shown by ``subTest`` and in failure messages.
    kind : str
        One of :data:`EDIT_KINDS`.
    edit : callable
        ``edit(morpho)`` makes exactly one successful edit to a tree from
        :func:`make_tree_t`.
    reindex : bool
        Whether the edit can change the default index of a branch that
        existed before it. True for a removal or a whole-subtree
        replacement, which must move the index watermark; False for an
        append, an in-place replacement, a spliced copy or a transform,
        which must keep every existing default index.
    """

    name: str
    kind: str
    edit: Callable[[Morphology], None]
    reindex: bool = False

    def __post_init__(self) -> None:
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"EditCase kind must be one of {EDIT_KINDS}, got {self.kind!r}.")


def _attach_at_terminal(morpho: Morphology) -> None:
    morpho.attach(
        parent="d1",
        child_branch=_point_branch([[150, 30, 5], [170, 50, 5]], (0.5, 0.5), "apical_dendrite"),
        child_name="tip",
    )


def _attach_at_soma_proximal_end(morpho: Morphology) -> None:
    morpho.attach(
        parent="soma",
        child_branch=_point_branch([[0, 0, 0], [0, 40, 0]], (0.5, 0.5), "apical_dendrite"),
        child_name="apical",
        parent_x=0.0,
    )


def _attach_by_attribute(morpho: Morphology) -> None:
    morpho.axon.collateral = _point_branch([[-30, 0, 0], [-40, 10, 0]], (0.5, 0.5), "axon")


def _stretch_dend(morpho: Morphology) -> None:
    simulate_replace_geometry(morpho, "dend", stretched(morpho.dend.branch))


def _remove_d2(morpho: Morphology) -> None:
    simulate_remove_leaf(morpho, "d2")


def _stretch_dend_bypassing_the_gate(morpho: Morphology) -> None:
    object.__setattr__(morpho.dend, "_branch", stretched(morpho.dend.branch))


#: One row per kind of edit; MORPH-2, MORPH-3 and MORPH-9 add theirs here.
EDIT_CASES: tuple[EditCase, ...] = (
    EditCase("attach a leaf at a terminal", "append", _attach_at_terminal),
    EditCase("attach at the soma's proximal end", "append", _attach_at_soma_proximal_end),
    EditCase("attach by attribute", "append", _attach_by_attribute),
    EditCase("replace dend's geometry in place", "replace", _stretch_dend),
    EditCase("remove the leaf d2", "delete", _remove_d2, reindex=True),
)

#: Edits that skip the gate on purpose: every layer's check must fail on them.
SABOTAGE_CASES: tuple[EditCase, ...] = (
    EditCase("replace dend's geometry bypassing the gate", "replace", _stretch_dend_bypassing_the_gate),
)


def cold_rebuild(morpho: Morphology) -> Morphology:
    """Rebuild ``morpho`` from its raw node tables, reading no cache.

    The reference of the edit table. It walks ``_nodes`` and each node's
    ``_children`` from the root and attaches every reachable branch to a
    new tree in node-id order, which is creation order and lists every
    parent before its children (a branch's parent never changes). It never
    reads ``branches``, ``branch_by_order``, ``index``, ``parent`` or any
    other cached view: ``clone_morpho`` iterates ``branches``, so after a
    structural write that skipped the gate it would copy the stale view
    tuple into the reference.

    The new tree shares the immutable :class:`Branch` objects, so every
    value derived from it comes from the same code on the same inputs as
    on ``morpho`` and must match exactly; no tolerance is applied, since a
    tolerance would only hide a stale result that happens to be close.
    """
    nodes = morpho._nodes
    root = nodes[morpho._root_id]
    rebuilt = Morphology.from_root(root._branch, name=root._name)
    reachable = []
    stack = [morpho._root_id]
    while stack:
        node = nodes[stack.pop()]
        reachable.append(node._node_id)
        stack.extend(node._children.values())
    for node_id in sorted(reachable):
        node = nodes[node_id]
        if node._parent_id is None:
            continue
        rebuilt.attach(
            parent=nodes[node._parent_id]._name,
            child_branch=node._branch,
            child_name=node._name,
            parent_x=float(node._parent_x),
            child_x=float(node._child_x),
        )
    return rebuilt


def _plain(value: object) -> object:
    """Turn quantities and arrays into nested tuples that compare with ``==``."""
    if isinstance(value, u.Quantity):
        return _plain(np.asarray(value.mantissa)), str(value.unit)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def morph_fingerprint(morpho: Morphology) -> dict[str, object]:
    """Read every derived value of ``morpho`` that a stale cache could get wrong.

    Reading it also fills those caches, which is how the edit table warms a
    tree before editing it. Keys: ``metric`` (:meth:`MorphoMetric.as_dict`);
    ``names/<order>`` and ``index/<order>`` for the default, type and depth
    orders; ``edges``; ``full_point_geometry``; ``ranges`` (x, y, z; only
    with full 3-D points); ``vis/structure_key`` and ``vis/xyz`` (per-branch
    points) from :meth:`Morphology.to_vis_data`. Together they read every
    cache the edit gate drops.
    """
    fingerprint: dict[str, object] = {"metric": _plain(morpho.metric.as_dict())}
    for order in ("default", "type", "depth"):
        fingerprint[f"names/{order}"] = tuple(view.name for view in morpho.branch_by_order(order=order))
        fingerprint[f"index/{order}"] = tuple(view.index_by(order=order) for view in morpho.branches)
    fingerprint["edges"] = tuple(
        (edge.parent.name, edge.child.name, float(edge.parent_x), float(edge.child_x)) for edge in morpho.edges
    )
    fingerprint["full_point_geometry"] = morpho.has_full_point_geometry
    if morpho.has_full_point_geometry:
        fingerprint["ranges"] = _plain((morpho.x_range, morpho.y_range, morpho.z_range))
    data = morpho.to_vis_data()
    fingerprint["vis/structure_key"] = data.structure_key
    fingerprint["vis/xyz"] = tuple(
        (_plain(branch.points_proximal), _plain(branch.points_distal)) for branch in data.branches
    )
    return fingerprint


_MISSING = object()


def _short(value: object, limit: int = 160) -> str:
    text = "<missing>" if value is _MISSING else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _first_difference(mine: object, theirs: object, path: str) -> tuple[str, object, object]:
    """Descend to the first differing entry, stopping at a flat row such as one xyz point."""
    if isinstance(mine, dict) and isinstance(theirs, dict):
        for key in dict.fromkeys([*mine, *theirs]):
            a, b = mine.get(key, _MISSING), theirs.get(key, _MISSING)
            if a != b:
                return _first_difference(a, b, f"{path}.{key}")
    if (
        isinstance(mine, tuple)
        and isinstance(theirs, tuple)
        and len(mine) == len(theirs)
        and any(isinstance(item, (tuple, dict)) for item in mine)
    ):
        for index, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                return _first_difference(a, b, f"{path}[{index}]")
    return path, mine, theirs


def _fingerprint_diff(edited: Mapping[str, object], reference: Mapping[str, object]) -> list[str]:
    """One line per fingerprint key that differs, naming its first differing entry."""
    problems = []
    for key in dict.fromkeys([*reference, *edited]):
        mine, theirs = edited.get(key, _MISSING), reference.get(key, _MISSING)
        if mine != theirs:
            path, mine, theirs = _first_difference(mine, theirs, key)
            problems.append(f"{path}: edited {_short(mine)} != cold rebuild {_short(theirs)}")
    return problems


def _fail_if_stale(testcase, case: EditCase, layer: str, problems: list[str]) -> None:
    if problems:
        testcase.fail(f"{layer}: {case.kind} edit {case.name!r} left stale results:\n  " + "\n  ".join(problems))


def assert_morpho_fresh(testcase, case: EditCase, build: Callable[[], Morphology] = make_tree_t) -> None:
    """Morph layer: after ``case.edit``, every read of the tree equals a cold rebuild.

    Builds a tree, reads :func:`morph_fingerprint` (warming every derived
    cache), applies the edit and checks that

    - :attr:`Morphology.revision` advanced by exactly one;
    - the index watermark moved if and only if ``case.reindex``;
    - unless ``case.reindex``, every branch that existed keeps its default
      index;
    - the edit changed what the fingerprint reads (otherwise the row could
      not catch a stale cache);
    - every fingerprint key equals that of :func:`cold_rebuild`.

    Parameters
    ----------
    testcase : unittest.TestCase
        Reports the failure.
    case : EditCase
        Row to check.
    build : callable
        Builds the tree to edit.

    Raises
    ------
    AssertionError
        Listing every failed check, one per line.
    """
    morpho = build()
    before = morph_fingerprint(morpho)
    revision = morpho.revision
    indices = {view.name: view.index for view in morpho.branches}
    case.edit(morpho)

    problems = []
    if morpho.revision != revision + 1:
        problems.append(f"revision: {revision} -> {morpho.revision}, expected exactly one step")
    if morpho._reindexed_after(revision) != case.reindex:
        problems.append(
            f"watermark: _reindexed_after(before) is {morpho._reindexed_after(revision)}, "
            f"but the case declares reindex={case.reindex}"
        )
    edited = morph_fingerprint(morpho)
    reference = morph_fingerprint(cold_rebuild(morpho))
    if not case.reindex:
        now = {name: index for index, name in enumerate(reference["names/default"])}
        moved = {name: (index, now[name]) for name, index in indices.items() if now.get(name, index) != index}
        if moved:
            problems.append(f"index: default indices moved {moved}, but the case declares reindex=False")
    if before == reference:
        problems.append("no-op: the cold rebuild equals the tree before the edit, so this row cannot catch a stale cache")
    problems += _fingerprint_diff(edited, reference)
    _fail_if_stale(testcase, case, "morph", problems)


def assert_layer_fresh(
    testcase,
    case: EditCase,
    *,
    layer: str,
    warm: Callable[[Morphology], object],
    observe: Callable[[object], Mapping[str, object]],
    reference: Callable[[Morphology], Mapping[str, object]] | None = None,
    build: Callable[[], Morphology] = make_tree_t,
) -> None:
    """A consumer of the tree, warmed before ``case.edit``, reads the same as one built on a cold rebuild.

    ``warm(morpho)`` builds the consumer on a new tree and ``observe``
    reads it into a fingerprint dict, once before the edit (filling its
    caches) and once after. The expected fingerprint is
    ``reference(cold)`` when given, else ``observe(warm(cold))``: the same
    consumer built from scratch on :func:`cold_rebuild` of the edited tree.

    The Cell tests use this for a Cell before ``init_state`` (``warm``
    declares the Cell, ``observe`` reads its CVs and point placements). A
    selection-cache layer passes ``warm`` returning ``(morpho, cache)`` and
    a ``reference`` that evaluates the same expressions without a cache.
    Unlike :func:`assert_morpho_fresh` this does not require the edit to
    change what ``observe`` reads: a rigid transform keeps the CV grid.

    Parameters
    ----------
    testcase : unittest.TestCase
        Reports the failure.
    case : EditCase
        Row to check.
    layer : str
        Names the layer in the failure message.
    warm, observe, reference : callable
        See above.
    build : callable
        Builds the tree to edit.

    Raises
    ------
    AssertionError
        Listing every fingerprint key that differs.
    """
    morpho = build()
    consumer = warm(morpho)
    observe(consumer)
    case.edit(morpho)
    cold = cold_rebuild(morpho)
    expected = reference(cold) if reference is not None else observe(warm(cold))
    _fail_if_stale(testcase, case, layer, _fingerprint_diff(observe(consumer), expected))
