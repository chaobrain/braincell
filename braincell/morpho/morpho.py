"""Editable morphology model.

User-facing entry points:
- `Morpho`: the whole mutable tree
- `MorphoBranch`: a tree-local branch view
- `BranchConnection`: read-only branch-to-branch topology edges

Internal structure:
- `_NodeState`: owner-side node storage
- `_MorphAttachPoint`: temporary syntax helper for `tree.soma[0.3].dend = ...`
- `_parse_attachment_key(...)`: parser for `[parent_x]` and `[parent_x, child_x]`

In normal use, users only need `Morpho` and `MorphoBranch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Union

from .branch import Branch

_MORPHO_RESERVED_NAMES = {
    "attach",
    "branch_by_index",
    "branch_by_name",
    "branches",
    "children_of",
    "connections",
    "from_root",
    "from_swc",
    "path_to_root",
    "root",
    "select",
    "total_length",
    "vis2d",
    "vis3d",
}
_MORPHO_BRANCH_RESERVED_NAMES = {
    "attach",
    "branch",
    "child_x",
    "children",
    "index",
    "type",
    "name",
    "parent",
    "parent_x",
}
_BRANCH_RESERVED_NAMES = set(Branch.__dataclass_fields__) | {
    name for name in dir(Branch) if not name.startswith("_")
}

ParentRef = Union[str, "MorphoBranch"]


@dataclass
class _NodeState:
    """Internal mutable node record owned by exactly one Morpho tree."""

    node_id: int
    name: str
    branch: Branch
    parent_id: Optional[int]
    parent_x: float
    child_x: float
    children: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchConnection:
    """A directed edge between two branches.

    `parent_x` preserves the attachment site on the parent branch. `child_x`
    keeps the child-side attachment site explicit using the same normalized
    `[0, 1]` coordinate system as the parent branch.
    """

    parent_branch: int
    child_branch: int
    parent_x: float
    child_x: float = 0.0


class Morpho:
    """Mutable morphology tree used for authoring, querying, and visualization.

    Users mutate this object through attribute assignment such as
    `tree.soma.dend = Branch(...)` or through explicit calls to `attach(...)`.
    Downstream consumers can query it directly; computation-oriented layers are
    responsible for freezing or array-ifying it at their own boundary.
    """

    def __init__(self, *, root_name: str | None, root_branch: Branch) -> None:
        self._nodes: dict[int, _NodeState] = {}
        self._name_to_id: dict[str, int] = {}
        self._type_name_counters: dict[str, int] = {}
        self._next_id = 0
        root_name, root_branch = self._resolve_branch_identity(
            root_branch,
            explicit_name=root_name,
        )
        self._root_name = root_name
        self._root_id = self._register_node(
            name=root_name,
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

        from ..io import SwcReader

        reader = SwcReader() if options is None else SwcReader(options=options)
        return reader.read(path, return_report=return_report)

    # Public top-level access -------------------------------------------------

    @property
    def root(self) -> "MorphoBranch":
        return MorphoBranch(self, self._root_id)

    @property
    def branches(self) -> tuple[Branch, ...]:
        return tuple(self._nodes[node_id].branch for node_id in self._ordered_node_ids())

    @property
    def connections(self) -> tuple[BranchConnection, ...]:
        index_map = self._branch_index_map()
        return tuple(
            BranchConnection(
                parent_branch=index_map[node.parent_id],
                child_branch=index_map[node.node_id],
                parent_x=node.parent_x,
                child_x=node.child_x,
            )
            for node_id in self._ordered_node_ids()
            for node in (self._nodes[node_id],)
            if node.parent_id is not None
        )

    @property
    def total_length(self):
        return sum(branch.total_length for branch in self.branches)

    # Public queries ----------------------------------------------------------

    def branch_by_name(self, name: str) -> "MorphoBranch":
        if name not in self._name_to_id:
            raise KeyError(name)
        return MorphoBranch(self, self._name_to_id[name])

    def branch_by_index(self, branch_index: int) -> "MorphoBranch":
        return MorphoBranch(self, self._node_id_from_index(branch_index))

    def children_of(self, branch_index: int) -> tuple[int, ...]:
        branch = self.branch_by_index(branch_index)
        return tuple(child.index for child in branch.children)

    def path_to_root(self, branch_index: int) -> tuple[int, ...]:
        node = self._get_node(self._node_id_from_index(branch_index))
        path = [self._branch_index(node.node_id)]
        while node.parent_id is not None:
            node = self._get_node(node.parent_id)
            path.append(self._branch_index(node.node_id))
        return tuple(reversed(path))

    def topo(self) -> str:
        """Return a line-oriented text view of the branch topology."""

        lines = [self.root.name]
        child_ids = tuple(self._get_node(self._root_id).children.values())
        for index, child_id in enumerate(child_ids):
            lines.extend(self._format_topology(child_id, prefix="", is_last=index == len(child_ids) - 1))
        return "\n".join(lines)

    def vis3d(
        self,
        *,
        backend: str | None = None,
        region=None,
        locset=None,
        values=None,
        chooser=None,
    ) -> object:
        from ..vis import plot

        return plot(
            self,
            region=region,
            locset=locset,
            values=values,
            dimensionality="3d",
            backend=backend,
            chooser=chooser,
        )

    def vis2d(self, *args, **kwargs) -> object:
        raise NotImplementedError("Morpho.vis2d(...) is not implemented yet.")

    def select(self, expr, *, cache=None):
        from braincell.filter import LocsetExpr, RegionExpr

        if not isinstance(expr, (RegionExpr, LocsetExpr)):
            raise TypeError(
                "Morpho.select(...) expects RegionExpr or LocsetExpr. "
                f"Got {type(expr).__name__!s}."
            )
        return expr.evaluate(self, cache=cache)

    # Public mutation ---------------------------------------------------------

    def attach(
        self,
        *,
        parent: ParentRef,
        child: str,
        branch: Branch,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        """Attach a child branch to a named parent or parent branch view."""

        parent_id = self._resolve_parent(parent)
        return self._insert_child(
            parent_id,
            child,
            branch,
            parent_x=parent_x,
            child_x=child_x,
        )

    def __getattr__(self, name: str) -> "MorphoBranch":
        if name in self._name_to_id:
            return MorphoBranch(self, self._name_to_id[name])
        raise AttributeError(f"{type(self).__name__!s} has no branch named {name!r}")

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._name_to_id))

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"Morpho(root={self._root_name!r}, branches={len(self)!r})"

    # Internal indexing and lookup -------------------------------------------

    def _ordered_node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._nodes))

    def _branch_index_map(self) -> dict[int, int]:
        return {node_id: index for index, node_id in enumerate(self._ordered_node_ids())}

    def _branch_index(self, node_id: int) -> int:
        return self._branch_index_map()[node_id]

    def _node_id_from_index(self, branch_index: int) -> int:
        ordered_ids = self._ordered_node_ids()
        try:
            return ordered_ids[branch_index]
        except IndexError as exc:
            raise IndexError(f"Branch index {branch_index!r} is out of range.") from exc

    def _get_node(self, node_id: int) -> _NodeState:
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

    # Internal validation and insertion --------------------------------------

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
        self._nodes[node_id] = _NodeState(
            node_id=node_id,
            name=name,
            branch=branch,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=child_x,
        )
        self._name_to_id[name] = node_id
        return node_id

    def _insert_child(
        self,
        parent_id: int,
        child_name: str,
        value: object,
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        parent = self._get_node(parent_id)
        self._validate_public_name(child_name)
        if child_name in parent.children:
            raise ValueError(
                f"Parent branch {parent.name!r} already has a child named {child_name!r}."
            )
        self._validate_parent_x(parent_x)
        self._validate_child_x(child_x)
        branch_name, branch = self._normalize_child_branch(value)
        node_id = self._register_node(
            name=branch_name,
            branch=branch,
            parent_id=parent_id,
            parent_x=parent_x,
            child_x=child_x,
        )
        parent.children[child_name] = node_id
        return MorphoBranch(self, node_id)

    def _get_child(self, parent_id: int, child_name: str) -> "MorphoBranch":
        parent = self._get_node(parent_id)
        if child_name not in parent.children:
            raise AttributeError(f"Branch {parent.name!r} has no child named {child_name!r}.")
        return MorphoBranch(self, parent.children[child_name])

    def _child_views(self, node_id: int) -> tuple["MorphoBranch", ...]:
        node = self._get_node(node_id)
        return tuple(MorphoBranch(self, child_id) for child_id in node.children.values())

    def _format_topology(self, node_id: int, *, prefix: str, is_last: bool) -> list[str]:
        node = self._get_node(node_id)
        branch_prefix = "└── " if is_last else "├── "
        lines = [f"{prefix}{branch_prefix}{node.name}"]
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        child_ids = tuple(node.children.values())
        for index, child_id in enumerate(child_ids):
            lines.extend(
                self._format_topology(
                    child_id,
                    prefix=child_prefix,
                    is_last=index == len(child_ids) - 1,
                )
            )
        return lines

    def _normalize_child_branch(self, value: object) -> tuple[str, Branch]:
        if isinstance(value, MorphoBranch):
            raise ValueError(
                "Cannot reattach a MorphoBranch into a Morpho. Reuse the underlying "
                "Branch geometry or create a new Branch instead."
            )
        if not isinstance(value, Branch):
            raise TypeError("Only Branch values participate in morphology syntax sugar.")
        return self._resolve_branch_identity(value)

    def _resolve_branch_identity(
        self,
        branch: Branch,
        *,
        explicit_name: str | None = None,
    ) -> tuple[str, Branch]:
        branch_name = explicit_name if explicit_name is not None else branch.name
        if branch_name is None:
            branch_name = self._allocate_name_for_type(branch.type)
        self._validate_public_name(branch_name)
        if branch_name in self._name_to_id:
            raise ValueError(f"Branch name {branch_name!r} already exists in this Morpho.")
        if branch.name == branch_name:
            return branch_name, branch
        return branch_name, replace(branch, name=branch_name)

    def _allocate_name_for_type(self, branch_type: str) -> str:
        suffix = self._type_name_counters.get(branch_type, 0)
        while f"{branch_type}_{suffix}" in self._name_to_id:
            suffix += 1
        self._type_name_counters[branch_type] = suffix + 1
        return f"{branch_type}_{suffix}"

    def _validate_parent_x(self, parent_x: float) -> None:
        if not 0.0 <= parent_x <= 1.0:
            raise ValueError(f"parent_x must be within [0, 1], got {parent_x!r}.")

    def _validate_child_x(self, child_x: float) -> None:
        if not 0.0 <= child_x <= 1.0:
            raise ValueError(f"child_x must be within [0, 1], got {child_x!r}.")

    def _validate_public_name(self, name: str) -> None:
        if not name.isidentifier():
            raise ValueError(f"Branch name {name!r} must be a valid Python identifier.")
        if name.startswith("_"):
            raise ValueError("Branch names starting with '_' are reserved.")
        if name in (_MORPHO_RESERVED_NAMES | _MORPHO_BRANCH_RESERVED_NAMES | _BRANCH_RESERVED_NAMES):
            raise ValueError(f"Branch name {name!r} is reserved by the Morpho API.")


class MorphoBranch:
    """A tree-local branch view bound to exactly one Morpho owner.

    This is the object returned by `tree.soma` and `tree.soma.dend`. It behaves
    like the branch that lives inside the tree: geometry comes from the wrapped
    `Branch`, while editing and topology-aware navigation come from `Morpho`.
    """

    def __init__(self, owner: Morpho, node_id: int) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_node_id", node_id)

    @property
    def node(self) -> _NodeState:
        return self._owner._get_node(self._node_id)

    @property
    def index(self) -> int:
        return self._owner._branch_index(self._node_id)

    @property
    def branch(self) -> Branch:
        return self._owner._get_branch(self._node_id)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def type(self) -> str:
        return self._branch.type

    @property
    def parent(self) -> Optional["MorphoBranch"]:
        if self.node.parent_id is None:
            return None
        return MorphoBranch(self._owner, self.node.parent_id)

    @property
    def parent_x(self) -> Optional[float]:
        if self.node.parent_id is None:
            return None
        return self.node.parent_x

    @property
    def child_x(self) -> Optional[float]:
        if self.node.parent_id is None:
            return None
        return self.node.child_x

    @property
    def children(self) -> tuple["MorphoBranch", ...]:
        return self._owner._child_views(self._node_id)

    # Editing API -------------------------------------------------------------

    def attach(
        self,
        name: str,
        branch: Branch,
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> "MorphoBranch":
        """Attach a named child explicitly.

        This is the programmatic fallback for dynamic names or code-generated
        morphologies when attribute assignment is not a good fit.
        """

        return self._owner.attach(
            parent=self,
            child=name,
            branch=branch,
            parent_x=parent_x,
            child_x=child_x,
        )

    # Syntax sugar and dynamic access ----------------------------------------

    def __getitem__(self, key: object) -> "_MorphAttachPoint":
        parent_x, child_x = _parse_attachment_key(key)
        return _MorphAttachPoint(self, parent_x=parent_x, child_x=child_x)

    def __getattr__(self, name: str) -> object:
        try:
            return object.__getattribute__(self.branch, name)
        except AttributeError:
            return self._owner._get_child(self._node_id, name)

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self._owner._insert_child(self._node_id, key, value)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | set(name for name in dir(self.branch) if not name.startswith("_"))
            | set(self.node.children)
        )

    def __repr__(self) -> str:
        return (
            f"MorphoBranch(name={self.name!r}, type={self.type!r}, "
            f"index={self.index!r})"
        )


class _MorphAttachPoint:
    """Internal helper for `tree.soma[0.3].dend = Branch(...)` syntax."""

    def __init__(self, parent: MorphoBranch, *, parent_x: float, child_x: float) -> None:
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_parent_x", parent_x)
        object.__setattr__(self, "_child_x", child_x)

    def attach(self, name: str, branch: Branch) -> MorphoBranch:
        return self._parent.attach(
            name,
            branch,
            parent_x=self._parent_x,
            child_x=self._child_x,
        )

    def __setattr__(self, key: str, value: object) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        self.attach(key, value)

    def __repr__(self) -> str:
        return (
            f"_MorphAttachPoint(parent={self._parent.name!r}, "
            f"parent_x={self._parent_x!r}, child_x={self._child_x!r})"
        )


def _parse_attachment_key(key: object) -> tuple[float, float]:
    if isinstance(key, tuple):
        if len(key) != 2:
            raise ValueError("Attachment keys must be [parent_x] or [parent_x, child_x].")
        parent_x = float(key[0])
        child_x = float(key[1])
        return parent_x, child_x
    return float(key), 0.0
