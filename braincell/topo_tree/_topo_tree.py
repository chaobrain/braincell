"""
tree_model.py

A small geometry + topology module.

Core concepts:
- Taper: a single taper defined by (L, r0, r1) and optional endpoints (p0, p1)
- Branch: an ordered list of Taper objects
- Tree: a directed tree of Branch nodes connected by parent->child edges

Printing:
- print(tree) will show topology first
- tree.verbose controls whether branches are expanded after topology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Iterable

import numpy as np
import networkx as nx

__all__ = [
    "Taper",
    "Branch",
    "Tree",
]

# Type aliases (not exported in __all__)
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]


def _fmt_vec3(v: Optional[Vec3], ndigits: int = 3) -> str:
    """Format a Vec3 for display."""
    if v is None:
        return "None"
    return f"({v[0]:.{ndigits}f}, {v[1]:.{ndigits}f}, {v[2]:.{ndigits}f})"


@dataclass
class Taper:
    """
    A single taper defined by length and two radii.

    Attributes
    ----------
    L  : float
        Length of the taper
    r0 : float
        Radius at the start
    r1 : float
        Radius at the end
    p0 : Optional[Vec3]
        Optional start point
    p1 : Optional[Vec3]
        Optional end point

    Example
    -------
    >>> t1 = Taper.from_points((0,0,0), (0,0,10), r0=1.0, r1=0.5)
    >>> t2 = Taper.from_points((0,0,0,1.0), (0,0,10,0.5))
    >>> t3 = Taper.from_length(5.0, 1.0)  # r1 defaults to r0
    """

    L: float
    r0: float
    r1: float
    p0: Optional[Vec3] = None
    p1: Optional[Vec3] = None

    @classmethod
    def from_length(cls, L: float, r0: float, r1: Optional[float] = None) -> "Taper":
        """Create a taper using only length and radii."""
        if r1 is None:
            r1 = r0
        return cls(L=float(L), r0=float(r0), r1=float(r1))

    @classmethod
    def from_points(
        cls,
        p0: Union[Vec3, Vec4],
        p1: Union[Vec3, Vec4],
        r0: Optional[float] = None,
        r1: Optional[float] = None,
    ) -> "Taper":
        """
        Create a taper from two points.

        Overloads
        ---------
        A) Vec3 + Vec3 + explicit radii
           >>> Taper.from_points((x,y,z), (x,y,z), r0=..., r1=...)

        B) Vec4 + Vec4 (radius embedded in point)
           >>> Taper.from_points((x,y,z,r), (x,y,z,r))
        """
        # Vec4 mode: radius embedded
        if len(p0) == 4 and len(p1) == 4:
            x0, y0, z0, rr0 = p0  # type: ignore
            x1, y1, z1, rr1 = p1  # type: ignore
            p0 = (x0, y0, z0)
            p1 = (x1, y1, z1)
            r0 = rr0
            r1 = rr1

        if r0 is None:
            raise ValueError("r0 must be provided for Vec3 input.")
        if r1 is None:
            r1 = r0

        L = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))  # type: ignore
        t = cls.from_length(L=L, r0=r0, r1=r1)
        t.p0 = p0  # type: ignore
        t.p1 = p1  # type: ignore
        return t

    def __repr__(self) -> str:
        return (
            f"Taper(L={self.L:.6g}, r0={self.r0:.6g}, r1={self.r1:.6g}, "
            f"p0={self.p0}, p1={self.p1})"
        )

    def __str__(self) -> str:
        return (
            f"L={self.L:.3f}, r0={self.r0:.3f}->{self.r1:.3f} | "
            f"p0={_fmt_vec3(self.p0)} -> p1={_fmt_vec3(self.p1)}"
        )


@dataclass
class Branch:
    """
    A Branch is an ordered list of Taper objects.

    Example
    -------
    >>> b1 = Branch.from_length([5.0, 4.0], [1.0, 0.8], [0.8, 0.6])
    >>> print(b1)
    Branch(n_tapers=2, total_L=9.000)
      [0] L=5.000, r0=1.000->0.800 | p0=None -> p1=None
      [1] L=4.000, r0=0.800->0.600 | p0=None -> p1=None
    """

    tapers: List[Taper] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Defensive checks to prevent accidental misuse such as Branch("c1").
        """
        if not isinstance(self.tapers, list):
            raise TypeError(f"Branch.tapers must be list[Taper], got {type(self.tapers)}")

        for i, t in enumerate(self.tapers):
            if not isinstance(t, Taper):
                raise TypeError(f"Branch.tapers[{i}] must be a Taper, got {type(t)}: {t}")

    @classmethod
    def empty(cls) -> "Branch":
        """Convenience constructor for an empty branch."""
        return cls(tapers=[])

    @classmethod
    def from_length(
        cls,
        lengths: List[float],
        r0s: List[float],
        r1s: Optional[List[float]] = None,
    ) -> "Branch":
        """
        Build a branch from per-taper length and radii arrays.
        """
        if r1s is None:
            r1s = r0s

        if not (len(lengths) == len(r0s) == len(r1s)):
            raise ValueError(
                f"length mismatch: len(lengths)={len(lengths)}, len(r0s)={len(r0s)}, len(r1s)={len(r1s)}"
            )

        tapers = [Taper.from_length(L, r0, r1) for L, r0, r1 in zip(lengths, r0s, r1s)]
        return cls(tapers=tapers)

    @classmethod
    def from_points(
        cls,
        points: Union[List[Vec3], List[Vec4]],
        radius: Optional[List[float]] = None,
    ) -> "Branch":
        """
        Build a branch from a polyline.

        Overloads
        ---------
        - points: list[Vec3] + radius: list[float]
        - points: list[Vec4] (xyzr), radius ignored
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points to build a branch.")

        # Vec4 mode
        if len(points[0]) == 4:
            xyzr: List[Vec4] = points  # type: ignore
            points3: List[Vec3] = [(x, y, z) for (x, y, z, _) in xyzr]
            radius = [r for (_, _, _, r) in xyzr]
        else:
            # Vec3 mode
            points3 = points  # type: ignore
            if radius is None:
                raise ValueError("Vec3 points require radius list.")
            if len(points3) != len(radius):
                raise ValueError(f"points and radius must match: {len(points3)} vs {len(radius)}")

        tapers = [
            Taper.from_points(points3[i], points3[i + 1], r0=radius[i], r1=radius[i + 1])
            for i in range(len(points3) - 1)
        ]
        return cls(tapers=tapers)

    @classmethod
    def from_tapers(cls, tapers: List[Taper]) -> "Branch":
        """Build a branch from pre-built tapers."""
        return cls(tapers=list(tapers))

    def total_length(self) -> float:
        """Return total length of this branch."""
        return float(sum(t.L for t in self.tapers))

    def __repr__(self) -> str:
        return f"Branch(n_tapers={len(self.tapers)}, total_L={self.total_length():.6g})"

    def __str__(self) -> str:
        lines = [f"Branch(n_tapers={len(self.tapers)}, total_L={self.total_length():.3f})"]
        for i, t in enumerate(self.tapers):
            lines.append(f"  [{i}] {str(t)}")
        return "\n".join(lines)


@dataclass
class Tree:
    """
    A Tree is a collection of Branch nodes connected by directed edges.

    Root rule:
    - root nodes are represented by an edge (-1, root_id)

    Printing
    --------
    - print(tree) always prints topology first
    - tree.verbose controls whether branches are expanded after topology

    Example
    -------
    >>> tree = Tree(verbose=True)
    >>> soma = tree.attach(Branch.empty(), label="soma")
    >>> L = tree.attach(Branch.empty(), parent="soma", label="L")
    >>> print(tree)  # shows topology + branches
    >>> tree.verbose = False
    >>> print(tree)  # shows topology only
    """

    branches: Dict[int, Branch] = field(default_factory=dict)
    label2id: Dict[str, int] = field(default_factory=dict)
    id2label: Dict[int, str] = field(default_factory=dict)

    # edges[(pid, cid)] = (parent_x, child_x)
    edges: Dict[Tuple[int, int], Tuple[float, float]] = field(default_factory=dict)

    _next_id: int = 0
    verbose: bool = True

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _resolve_id(self, x: Union[int, str]) -> int:
        if isinstance(x, int):
            return x
        if x not in self.label2id:
            raise KeyError(f"Unknown label: {x}")
        return self.label2id[x]

    def _node_name(self, nid: int) -> str:
        lab = self.id2label.get(nid)
        return f"{lab}({nid})" if lab else str(nid)

    def attach(
        self,
        branch: Branch,
        parent: Union[int, str] = -1,
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
        label: Optional[str] = None,
    ) -> int:
        """
        Attach a new branch node to the tree.

        If parent == -1, the node becomes a root node.
        """
        cid = self._new_id()
        self.branches[cid] = branch

        if label is not None:
            if label in self.label2id:
                raise ValueError(f"label {label} already exists")
            self.label2id[label] = cid
            self.id2label[cid] = label

        if parent == -1:
            self.edges[(-1, cid)] = (-1.0, -1.0)
        else:
            pid = self._resolve_id(parent)
            if pid not in self.branches:
                raise KeyError(f"Unknown parent id: {pid}")
            self.edges[(pid, cid)] = (float(parent_x), float(child_x))

        return cid
    
    def remap_ids(self, id_map: Dict[int, int], inplace: bool = True) -> "Tree":
        """
        Remap all node ids using id_map.

        Missing ids keep their original value.
        """
        def map_id(i: int) -> int:
            return id_map.get(i, i)

        target = self if inplace else Tree(verbose=self.verbose)

        target.branches = {map_id(old): br for old, br in self.branches.items()}
        target.label2id = {lab: map_id(i) for lab, i in self.label2id.items()}
        target.id2label = {map_id(i): lab for i, lab in self.id2label.items()}
        target.edges = {(map_id(pid), map_id(cid)): w for (pid, cid), w in self.edges.items()}

        all_ids = list(target.branches.keys())
        target._next_id = (max(all_ids) + 1) if all_ids else 0

        return target

    def reorder_ids(self, inplace: bool = True) -> "Tree":
        """
        Reorder ids into a compact range 0..N-1.
        """
        old_ids = sorted(self.branches.keys())
        id_map = {old: new for new, old in enumerate(old_ids)}
        return self.remap_ids(id_map, inplace=inplace)

    def roots(self) -> List[int]:
        """Topology: root node ids (connected from parent=-1)."""
        return sorted([cid for (pid, cid) in self.edges.keys() if pid == -1])

    def _children_map(self) -> Dict[int, List[int]]:
        """Topology: parent -> children list (exclude pid=-1)."""
        ch: Dict[int, List[int]] = {}
        for (pid, cid) in self.edges.keys():
            if pid == -1:
                continue
            ch.setdefault(pid, []).append(cid)
        for pid in ch:
            ch[pid].sort()
        return ch
    
    def topo_str(self) -> str:
        """
        Return an ASCII topology string.
        """
        children = self._children_map()

        if not self.roots():
            return "[topo] No root found: missing edge (-1, root)"

        lines: List[str] = []

        def dfs(nid: int, prefix: str = "") -> None:
            kids = sorted(children.get(nid, []))
            for i, cid in enumerate(kids):
                is_last = (i == len(kids) - 1)
                branch = "└── " if is_last else "├── "
                lines.append(prefix + branch + self._node_name(cid))
                dfs(cid, prefix + ("    " if is_last else "│   "))

        for r in self.roots():
            lines.append(self._node_name(r))
            dfs(r)

        return "\n".join(lines)

    def topo(self) -> None:
        """Print topology directly."""
        print(self.topo_str())
    
    def delete_subtree(
        self,
        node: Union[int, str],
        *,
        inplace: bool = True,
        reorder: bool = True,
    ) -> "Tree":
        """
        Delete node and all descendants.
        """
        tree = self if inplace else self.remap_ids({}, inplace=False)

        root_id = tree._resolve_id(node)
        children = tree._children_map()

        to_delete: set[int] = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            if cur in to_delete:
                continue
            to_delete.add(cur)
            stack.extend(children.get(cur, []))

        for nid in to_delete:
            tree.branches.pop(nid, None)

        tree.edges = {
            (pid, cid): w
            for (pid, cid), w in tree.edges.items()
            if pid not in to_delete and cid not in to_delete
        }

        for nid in list(to_delete):
            if nid in tree.id2label:
                lab = tree.id2label.pop(nid)
                tree.label2id.pop(lab, None)

        if reorder:
            tree.reorder_ids(inplace=True)

        return tree

    def merge_tree(
        self,
        other: "Tree",
        parent: Union[int, str],
        *,
        parent_x: float = 1.0,
        child_x: float = 0.0,
        inplace: bool = True,
        reorder: bool = True,
    ) -> "Tree":
        """
        Merge another tree under a given parent node.
        """
        tree = self if inplace else self.remap_ids({}, inplace=False)

        pid = tree._resolve_id(parent)
        if pid not in tree.branches:
            raise KeyError(f"Unknown parent id: {pid}")

        current_max = max(tree.branches.keys()) if tree.branches else -1
        offset = current_max + 1

        other_ids = sorted(other.branches.keys())
        id_map = {oid: oid + offset for oid in other_ids}
        other2 = other.remap_ids(id_map, inplace=False)

        root_candidates = [cid for (pp, cid) in other2.edges.keys() if pp == -1]
        if len(root_candidates) == 0:
            raise ValueError("other tree has no root-attached edge (-1, root)")
        if len(root_candidates) > 1:
            raise ValueError(f"other tree has multiple roots: {root_candidates}")

        rid = root_candidates[0]

        other2.edges.pop((-1, rid), None)
        other2.edges[(pid, rid)] = (float(parent_x), float(child_x))

        for nid, br in other2.branches.items():
            if nid in tree.branches:
                raise ValueError(f"ID collision after remap: {nid}")
            tree.branches[nid] = br

        for e, w in other2.edges.items():
            if e in tree.edges:
                raise ValueError(f"Edge collision: {e}")
            tree.edges[e] = w

        for lab, nid in other2.label2id.items():
            if lab in tree.label2id:
                raise ValueError(f"Label collision: {lab}")
            tree.label2id[lab] = nid
            tree.id2label[nid] = lab

        tree._next_id = (max(tree.branches.keys()) + 1) if tree.branches else 0

        if reorder:
            tree.reorder_ids(inplace=True)

        return tree

    def __repr__(self) -> str:
        return f"Tree(n_nodes={len(self.branches)}, n_edges={len(self.edges)}, verbose={self.verbose})"

    def __str__(self) -> str:
        out: List[str] = []

        # Always show topology first
        out.append("Tree Topology:")
        out.append(self.topo_str())

        # Optional branch expansion
        if not self.verbose:
            return "\n".join(out)

        out.append("\nBranches:")
        for nid in sorted(self.branches.keys()):
            out.append(f"- Node {self._node_name(nid)}")
            out.append(str(self.branches[nid]))

        return "\n".join(out)
    
    ## ——————add to_gragh——————

    def node_label(self, nid: int) -> str:
        """Return label for nid; fallback to id string."""
        return self.id2label.get(nid, str(nid))
    
    def _node_name(self, nid: int) -> str:
        lab = self.node_label(nid)
        return f"{lab}({nid})" if lab and lab != str(nid) else str(nid)
    
    def iter_nodes(self) -> Iterable[int]:
        """Topology: all node ids."""
        return self.branches.keys()

    def iter_edges(self, *, include_virtual_root: bool = False) -> Iterable[Tuple[int, int]]:
        """Topology: (parent, child). parent=-1 means virtual root."""
        for (pid, cid) in self.edges.keys():
            if pid == -1 and not include_virtual_root:
                continue
            yield pid, cid
    
    def to_graph(self) -> 'nx.DiGraph':
        """Export topology to a NetworkX DiGraph."""
        G = nx.DiGraph()

        # nodes
        for nid in self.iter_nodes():
            G.add_node(nid, label=self.node_label(nid))

        # edges
        for pid, cid in self.iter_edges(include_virtual_root=False):
            G.add_edge(pid, cid)

        return G

