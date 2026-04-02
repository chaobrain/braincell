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



from dataclasses import dataclass
from typing import Callable, Optional, Union

from .branch import Branch
from .metrics import MorphMetrics

_MORPHO_METRIC_PROPERTY_NAMES = {
    "max_euclidean_distance",
    "max_branch_order",
    "max_path_distance",
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
                             "branches_by",
                             "edges",
                             "from_asc",
                             "from_root",
                             "from_swc",
                             "metric",
                             "path_to_root",
                             "root",
                             "select",
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
    """A directed edge between two morphology branches."""

    parent: "MorphoBranch"
    child: "MorphoBranch"
    parent_x: float
    child_x: float = 0.0


class Morpho:
    """Mutable morphology tree used for authoring, querying, and visualization."""

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
        """Create a new editable tree from one root branch."""

        return cls(root_name=name, root_branch=branch)

    @classmethod
    def from_swc(
        cls,
        path,
        *,
        options: 'SwcReadOptions' = None,
        return_report: bool = False,
    ):
        """Load a morphology from a SWC file through the reader pipeline."""

        from braincell.io import SwcReader

        reader = SwcReader() if options is None else SwcReader(options=options)
        return reader.read(path, return_report=return_report)

    @classmethod
    def from_asc(cls, path, *, return_report: bool = False):
        """Load a morphology from a Neurolucida ASC file through the reader pipeline."""

        from braincell.io import AscReader

        return AscReader().read(path, return_report=return_report)

    @property
    def root(self) -> "MorphoBranch":
        return self._get_node(self._root_id)

    @property
    def branches(self) -> tuple["MorphoBranch", ...]:
        return self.branches_by(order="default")

    @property
    def edges(self) -> tuple[MorphoEdge, ...]:
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
    def metric(self) -> MorphMetrics:
        return MorphMetrics(self)

    def branches_by(self, *, order: str = "default") -> tuple["MorphoBranch", ...]:
        return tuple(self._get_node(node_id) for node_id in self._ordered_node_ids_by(order))

    def branch(
        self,
        *,
        name: str | None = None,
        index: int | None = None,
        order: str | None = None,
    ) -> "MorphoBranch":
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
        return self.metric.path_to_root(branch_index)

    def summary(self) -> dict[str, object]:
        return {
            "root_name": self.root.name,
            "root_type": self.root.type,
            "n_branches": self.n_branches,
            "n_stems": self.n_stems,
            "n_bifurcations": self.n_bifurcations,
            "max_branch_order": self.max_branch_order,
            "total_length": self.total_length,
            "total_area": self.total_area,
            "total_volume": self.total_volume,
            "mean_radius": self.mean_radius,
            "has_point_geometry": any(branch.branch.points is not None for branch in self.branches),
            "has_full_point_geometry_for_distance_metrics": self._has_full_point_geometry_for_distance_metrics(),
        }

    def topo(self) -> str:
        """Return a line-oriented text view of the branch topology."""

        lines = [self.root.name]
        child_ids = tuple(self.root._children.values())
        for index, child_id in enumerate(child_ids):
            lines.extend(self._format_topology(child_id, prefix="", is_last=index == len(child_ids) - 1))
        return "\n".join(lines)

    def vis3d(
        self,
        *,
        mode: str = "geometry",
        backend: str | None = None,
        region=None,
        locset=None,
        values=None,
        chooser=None,
        notebook: bool | None = None,
        jupyter_backend: str | None = None,
        return_plotter: bool = False,
    ) -> object:
        from braincell.vis.plot3d import plot3d

        return plot3d(
            self,
            mode=mode,
            region=region,
            locset=locset,
            values=values,
            backend=backend,
            chooser=chooser,
            notebook=notebook,
            jupyter_backend=jupyter_backend,
            return_plotter=return_plotter,
        )

    def vis2d(
        self,
        *,
        mode: str = "projected",
        backend: str | None = None,
        region=None,
        locset=None,
        values=None,
        chooser=None,
        notebook: bool | None = None,
        jupyter_backend: str | None = None,
        return_plotter: bool = False,
        projection_plane: str = "xy",
    ) -> object:
        from braincell.vis.plot2d import plot2d

        return plot2d(
            self,
            mode=mode,
            backend=backend,
            region=region,
            locset=locset,
            values=values,
            chooser=chooser,
            notebook=notebook,
            jupyter_backend=jupyter_backend,
            return_plotter=return_plotter,
            projection_plane=projection_plane,
        )

    def select(self, expr, *, cache=None):
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
        """Attach a child branch to a named parent or parent branch view.

        Args:
            parent: Parent branch name or MorphoBranch instance
            child_branch: Branch geometry to attach
            child_name: Optional name for the child branch
            parent_x: Attachment point on parent branch (0=proximal, 0.5=midpoint for soma only, 1=distal)
            child_x: Attachment point on child branch (0=proximal, 1=distal)

        Note:
            parent_x=0 attaches to the proximal end of the parent branch. This is typically
            used when the parent is itself a child branch and you want to attach at its
            connection point rather than its distal end.
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
        if name in _MORPHO_METRIC_PROPERTY_NAMES:
            return getattr(self.metric, name)
        if name in self._name_to_id:
            return self._get_node(self._name_to_id[name])
        raise AttributeError(f"{type(self).__name__!s} has no branch or metric named {name!r}")

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._name_to_id) | _MORPHO_METRIC_PROPERTY_NAMES)

    def __len__(self) -> int:
        return len(self._nodes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Morpho):
            return NotImplemented
        return self._eq_records() == other._eq_records()

    def __repr__(self) -> str:
        return f"Morpho(root={self._root_name!r}, branches={len(self)!r})"

    def _ordered_node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def _ordered_node_ids_by(self, order: str) -> tuple[int, ...]:
        ordered_ids = self._ordered_node_ids()
        if order == "default":
            return ordered_ids
        if order == "type":
            return tuple(
                sorted(ordered_ids, key=lambda node_id: (self._get_node(node_id).type, self._get_node(node_id).name)))
        if order == "depth":
            return tuple(sorted(ordered_ids,
                                key=lambda node_id: (len(self._path_node_ids(node_id)), self._branch_index(node_id))))
        raise ValueError(f"Unsupported branch order {order!r}.")

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

    def _has_full_point_geometry_for_distance_metrics(self) -> bool:
        root = self.root.branch
        if root.points_proximal is None:
            return False
        return all(branch.branch.points_distal is not None for branch in self.branches if branch.n_children == 0)

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
    """A tree-local branch node bound to exactly one Morpho owner."""

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
        return self._owner._branch_index(self._node_id)

    def index_by(self, *, order: str = "default") -> int:
        return self._owner._branch_index(self._node_id, order=order)

    @property
    def parent(self) -> Optional["MorphoBranch"]:
        if self._parent_id is None:
            return None
        return self._owner._get_node(self._parent_id)

    @property
    def children(self) -> tuple["MorphoBranch", ...]:
        return tuple(self._owner._get_node(child_id) for child_id in self._children.values())

    @property
    def n_children(self) -> int:
        return len(self._children)

    def attach(
        self,
        branch: Branch,
        name: str | None = None,
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        """Attach a child explicitly from this branch.

        Args:
            branch: Branch geometry to attach
            name: Optional name for the child branch
            parent_x: Attachment point on this branch (0=proximal, 0.5=midpoint for soma only, 1=distal)
            child_x: Attachment point on child branch (0=proximal, 1=distal)

        Note:
            parent_x=0 attaches to the proximal end of this branch. This is typically
            used when this branch is itself a child and you want to attach at its
            connection point rather than its distal end.
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
