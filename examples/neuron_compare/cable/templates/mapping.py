"""Branch/section mapping helpers for multi-compartment cable comparisons."""



from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np

try:
    from .morphology_io import delete_neuron_sections, load_braincell_morphology, load_neuron_sections
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from morphology_io import delete_neuron_sections, load_braincell_morphology, load_neuron_sections  # type: ignore


@dataclass(frozen=True)
class TreeNodeDescriptor:
    node_index: int
    name: str
    branch_type: str
    length_um: float
    is_root: bool
    parent_index: int | None
    parent_type: str | None
    parent_attach_side: float | None
    child_attach_side: float | None
    child_indices: tuple[int, ...]
    subtree_total_length_um: float
    subtree_leaf_count: int
    subtree_max_depth: int
    child_type_multiset: tuple[str, ...]
    subtree_signature: tuple[Any, ...]


@dataclass(frozen=True)
class MappingResult:
    branch_pairs: tuple[dict[str, Any], ...]
    compartment_pairs: tuple[dict[str, Any], ...]
    stimulus_target_pair: dict[str, Any]


ATTACH_TOL = 1e-3


def build_mapping(
    case,
    *,
    braincell_result: dict[str, Any],
    neuron_result: dict[str, Any],
) -> MappingResult:
    morpho = load_braincell_morphology(case)
    secs = load_neuron_sections(case)
    try:
        braincell_nodes = _braincell_descriptors(morpho)
        neuron_nodes = _neuron_descriptors(secs)
        branch_index_to_section_index = _match_trees(
            braincell_nodes=braincell_nodes,
            neuron_nodes=neuron_nodes,
        )
        braincell_canonical_names, neuron_canonical_names = _build_canonical_names(
            braincell_nodes=braincell_nodes,
            neuron_nodes=neuron_nodes,
            branch_index_to_section_index=branch_index_to_section_index,
        )

        branch_pairs = tuple(
            {
                "braincell_branch_id": int(branch_index),
                "braincell_branch_name": braincell_nodes[branch_index].name,
                "braincell_canonical_name": braincell_canonical_names[branch_index],
                "braincell_branch_type": braincell_nodes[branch_index].branch_type,
                "neuron_section_index": int(section_index),
                "neuron_section_name": neuron_nodes[section_index].name,
                "neuron_canonical_name": neuron_canonical_names[section_index],
                "neuron_section_type": neuron_nodes[section_index].branch_type,
                "match_score": _node_match_score(braincell_nodes[branch_index], neuron_nodes[section_index]),
                "length_diff_um": abs(braincell_nodes[branch_index].length_um - neuron_nodes[section_index].length_um),
                "subtree_total_length_diff_um": abs(
                    braincell_nodes[branch_index].subtree_total_length_um
                    - neuron_nodes[section_index].subtree_total_length_um
                ),
                "parent_attach_diff": _attach_diff(
                    braincell_nodes[branch_index].parent_attach_side,
                    neuron_nodes[section_index].parent_attach_side,
                ),
                "child_attach_diff": _attach_diff(
                    braincell_nodes[branch_index].child_attach_side,
                    neuron_nodes[section_index].child_attach_side,
                ),
                "subtree_signature": list(_serialize_signature(braincell_nodes[branch_index].subtree_signature)),
            }
            for branch_index, section_index in sorted(branch_index_to_section_index.items())
        )

        braincell_labels = braincell_result["compartment_labels"]
        neuron_labels = neuron_result["compartment_labels"]
        compartment_pairs = _build_compartment_pairs(
            braincell_labels=braincell_labels,
            neuron_labels=neuron_labels,
            branch_index_to_section_index=branch_index_to_section_index,
            braincell_canonical_names=braincell_canonical_names,
            neuron_canonical_names=neuron_canonical_names,
        )
        stimulus_target_pair = _build_stimulus_target_pair(
            cv_per_branch=int(case.cv_policy.cv_per_branch),
            braincell_labels=braincell_labels,
            neuron_labels=neuron_labels,
            braincell_nodes=braincell_nodes,
            branch_index_to_section_index=branch_index_to_section_index,
            braincell_canonical_names=braincell_canonical_names,
            neuron_canonical_names=neuron_canonical_names,
        )
        return MappingResult(
            branch_pairs=branch_pairs,
            compartment_pairs=compartment_pairs,
            stimulus_target_pair=stimulus_target_pair,
        )
    finally:
        delete_neuron_sections(secs)


def _braincell_descriptors(morpho) -> dict[int, TreeNodeDescriptor]:
    children_by_index = {
        int(branch.index): tuple(int(child.index) for child in branch.children)
        for branch in morpho.branches
    }

    cache: dict[int, TreeNodeDescriptor] = {}

    def build(branch_index: int) -> TreeNodeDescriptor:
        cached = cache.get(branch_index)
        if cached is not None:
            return cached
        branch = morpho.branch(index=branch_index)
        child_indices = children_by_index[branch_index]
        child_descriptors = tuple(build(child_index) for child_index in child_indices)
        subtree_total_length_um = float(branch.length.to_decimal(u.um)) + sum(
            child.subtree_total_length_um for child in child_descriptors
        )
        subtree_leaf_count = 1 if len(child_descriptors) == 0 else sum(child.subtree_leaf_count for child in child_descriptors)
        subtree_max_depth = 0 if len(child_descriptors) == 0 else 1 + max(child.subtree_max_depth for child in child_descriptors)
        descriptor = TreeNodeDescriptor(
            node_index=int(branch.index),
            name=branch.name,
            branch_type=branch.type,
            length_um=_round_float(float(branch.length.to_decimal(u.um))),
            is_root=branch.parent is None,
            parent_index=None if branch.parent is None else int(branch.parent.index),
            parent_type=None if branch.parent is None else branch.parent.type,
            parent_attach_side=None if branch.parent is None else _normalize_attach_side(float(branch.parent_x)),
            child_attach_side=None if branch.parent is None else _normalize_attach_side(float(branch.child_x)),
            child_indices=child_indices,
            subtree_total_length_um=_round_float(subtree_total_length_um),
            subtree_leaf_count=int(subtree_leaf_count),
            subtree_max_depth=int(subtree_max_depth),
            child_type_multiset=tuple(sorted(_normalized_branch_type(child.branch_type) for child in child_descriptors)),
            subtree_signature=(),
        )
        signature = _node_signature(descriptor, child_descriptors)
        descriptor = TreeNodeDescriptor(
            **{**descriptor.__dict__, "subtree_signature": signature}
        )
        cache[branch_index] = descriptor
        return descriptor

    for branch in morpho.branches:
        build(int(branch.index))
    return cache


def _neuron_descriptors(secs) -> dict[int, TreeNodeDescriptor]:
    from neuron import h

    section_count = len(secs)
    section_by_index = {index: sec for index, sec in enumerate(secs)}
    name_to_index = {sec.name(): index for index, sec in enumerate(secs)}
    parent_by_index: dict[int, int | None] = {}
    children_by_index: dict[int, list[int]] = {index: [] for index in range(section_count)}
    attach_cache: dict[int, tuple[float | None, float | None]] = {}

    for index, sec in section_by_index.items():
        ref = h.SectionRef(sec=sec)
        if ref.has_parent():
            parent_name = ref.parent.name()
            parent_index = name_to_index.get(parent_name)
            parent_by_index[index] = parent_index
            if parent_index is not None:
                children_by_index[parent_index].append(index)
        else:
            parent_by_index[index] = None

    cache: dict[int, TreeNodeDescriptor] = {}

    def build(index: int) -> TreeNodeDescriptor:
        cached = cache.get(index)
        if cached is not None:
            return cached
        sec = section_by_index[index]
        child_indices = tuple(children_by_index[index])
        child_descriptors = tuple(build(child_index) for child_index in child_indices)
        parent_index = parent_by_index[index]
        if index not in attach_cache:
            attach_cache[index] = _infer_neuron_attach_sides(
                sec,
                parent_sec=None if parent_index is None else section_by_index[parent_index],
            )
        parent_attach_side, child_attach_side = attach_cache[index]
        subtree_total_length_um = float(sec.L) + sum(child.subtree_total_length_um for child in child_descriptors)
        subtree_leaf_count = 1 if len(child_descriptors) == 0 else sum(child.subtree_leaf_count for child in child_descriptors)
        subtree_max_depth = 0 if len(child_descriptors) == 0 else 1 + max(child.subtree_max_depth for child in child_descriptors)
        descriptor = TreeNodeDescriptor(
            node_index=index,
            name=sec.name(),
            branch_type=_infer_neuron_branch_type(sec.name()),
            length_um=_round_float(float(sec.L)),
            is_root=parent_index is None,
            parent_index=parent_index,
            parent_type=None if parent_index is None else _infer_neuron_branch_type(section_by_index[parent_index].name()),
            parent_attach_side=parent_attach_side,
            child_attach_side=child_attach_side,
            child_indices=child_indices,
            subtree_total_length_um=_round_float(subtree_total_length_um),
            subtree_leaf_count=int(subtree_leaf_count),
            subtree_max_depth=int(subtree_max_depth),
            child_type_multiset=tuple(sorted(_normalized_branch_type(child.branch_type) for child in child_descriptors)),
            subtree_signature=(),
        )
        signature = _node_signature(descriptor, child_descriptors)
        descriptor = TreeNodeDescriptor(
            **{**descriptor.__dict__, "subtree_signature": signature}
        )
        cache[index] = descriptor
        return descriptor

    for index in range(section_count):
        build(index)
    return cache


def _infer_neuron_branch_type(section_name: str) -> str:
    prefix = section_name.split("[", 1)[0]
    if prefix == "soma":
        return "soma"
    if prefix == "axon":
        return "axon"
    if prefix == "dend":
        return "basal_dendrite"
    return prefix


def _infer_neuron_attach_sides(sec, *, parent_sec) -> tuple[float | None, float | None]:
    if parent_sec is None:
        return None, None
    sec_prox, sec_dist = _section_endpoint_point_um(sec, proximal=True), _section_endpoint_point_um(sec, proximal=False)
    parent_prox, parent_dist = _section_endpoint_point_um(parent_sec, proximal=True), _section_endpoint_point_um(parent_sec, proximal=False)

    d_pp = _distance_um(sec_prox, parent_prox)
    d_pd = _distance_um(sec_prox, parent_dist)
    d_dp = _distance_um(sec_dist, parent_prox)
    d_dd = _distance_um(sec_dist, parent_dist)
    best = min(
        (
            (d_pp, 0.0, 0.0),
            (d_pd, 1.0, 0.0),
            (d_dp, 0.0, 1.0),
            (d_dd, 1.0, 1.0),
        ),
        key=lambda item: item[0],
    )
    best_distance, parent_side, child_side = best
    if best_distance <= 1e-6:
        return parent_side, child_side

    attach_point = sec_prox if min(d_pp, d_pd) <= min(d_dp, d_dd) else sec_dist
    midpoint = (
        0.5 * (parent_prox[0] + parent_dist[0]),
        0.5 * (parent_prox[1] + parent_dist[1]),
        0.5 * (parent_prox[2] + parent_dist[2]),
    )
    d_mid = _distance_um(attach_point, midpoint)
    if d_mid < min(_distance_um(attach_point, parent_prox), _distance_um(attach_point, parent_dist)):
        return 0.5, child_side
    return parent_side, child_side


def _section_endpoint_point_um(sec, *, proximal: bool) -> tuple[float, float, float]:
    from neuron import h

    n3d = int(h.n3d(sec=sec))
    if n3d <= 0:
        return (0.0, 0.0, 0.0)
    index = 0 if proximal else n3d - 1
    return (
        float(h.x3d(index, sec=sec)),
        float(h.y3d(index, sec=sec)),
        float(h.z3d(index, sec=sec)),
    )


def _distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2) + ((a[2] - b[2]) ** 2))


def _node_signature(descriptor: TreeNodeDescriptor, child_descriptors: tuple[TreeNodeDescriptor, ...]) -> tuple[Any, ...]:
    child_sigs = tuple(sorted(child.subtree_signature for child in child_descriptors))
    return (
        _normalized_branch_type(descriptor.branch_type),
        descriptor.length_um,
        descriptor.subtree_total_length_um,
        descriptor.subtree_leaf_count,
        descriptor.subtree_max_depth,
        descriptor.child_type_multiset,
        child_sigs,
    )


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _normalize_attach_side(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value - 0.0) < 1e-6:
        return 0.0
    if abs(value - 0.5) < 1e-6:
        return 0.5
    if abs(value - 1.0) < 1e-6:
        return 1.0
    return _round_float(value)


def _match_trees(
    *,
    braincell_nodes: dict[int, TreeNodeDescriptor],
    neuron_nodes: dict[int, TreeNodeDescriptor],
) -> dict[int, int]:
    braincell_roots = [node for node in braincell_nodes.values() if node.is_root]
    neuron_roots = [node for node in neuron_nodes.values() if node.is_root]
    if len(braincell_roots) != 1 or len(neuron_roots) != 1:
        raise ValueError("Mapping expects exactly one root on each side.")
    mapping: dict[int, int] = {}

    def walk(braincell_index: int, neuron_index: int) -> None:
        if braincell_index in mapping:
            existing = mapping[braincell_index]
            if existing != neuron_index:
                raise ValueError(
                    f"Conflicting mapping for braincell branch {braincell_index!r}: "
                    f"{existing!r} vs {neuron_index!r}."
                )
            return
        b = braincell_nodes[braincell_index]
        n = neuron_nodes[neuron_index]
        if not _nodes_are_compatible(b, n):
            raise ValueError(
                f"Branch signature mismatch between braincell {b.name!r} and NEURON {n.name!r}: "
                f"{_hard_key(b)!r} vs {_hard_key(n)!r}."
            )
        mapping[braincell_index] = neuron_index
        braincell_children = [braincell_nodes[index] for index in b.child_indices]
        neuron_children = [neuron_nodes[index] for index in n.child_indices]
        if len(braincell_children) != len(neuron_children):
            raise ValueError(
                f"Child-count mismatch between braincell {b.name!r} and NEURON {n.name!r}."
            )
        child_pairs = _match_child_group(braincell_children, neuron_children)
        for braincell_child, neuron_child in child_pairs:
            walk(braincell_child.node_index, neuron_child.node_index)

    walk(braincell_roots[0].node_index, neuron_roots[0].node_index)
    if len(mapping) != len(braincell_nodes) or len(mapping) != len(neuron_nodes):
        raise ValueError("Mapping did not cover all branches/sections.")
    return mapping


def _sort_key(node: TreeNodeDescriptor) -> tuple[Any, ...]:
    return (
        _normalized_branch_type(node.branch_type),
        node.length_um,
        node.subtree_total_length_um,
        node.subtree_leaf_count,
        node.subtree_max_depth,
        node.subtree_signature,
    )


def _hard_key(node: TreeNodeDescriptor) -> tuple[Any, ...]:
    return (
        _normalized_branch_type(node.branch_type),
        len(node.child_indices),
        node.subtree_leaf_count,
        node.subtree_max_depth,
        node.child_type_multiset,
    )


def _nodes_are_compatible(left: TreeNodeDescriptor, right: TreeNodeDescriptor) -> bool:
    if _hard_key(left) != _hard_key(right):
        return False
    return True


def _attach_diff(left: float | None, right: float | None) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return 0.5
    return abs(float(left) - float(right))


def _node_match_score(left: TreeNodeDescriptor, right: TreeNodeDescriptor) -> float:
    if not _nodes_are_compatible(left, right):
        return float("inf")
    return (
        10.0 * abs(left.length_um - right.length_um)
        + 2.0 * abs(left.subtree_total_length_um - right.subtree_total_length_um)
        + 0.5 * _attach_diff(left.parent_attach_side, right.parent_attach_side)
        + 0.5 * _attach_diff(left.child_attach_side, right.child_attach_side)
    )


def _match_child_group(
    braincell_children: list[TreeNodeDescriptor],
    neuron_children: list[TreeNodeDescriptor],
) -> list[tuple[TreeNodeDescriptor, TreeNodeDescriptor]]:
    grouped_braincell: dict[tuple[Any, ...], list[TreeNodeDescriptor]] = {}
    grouped_neuron: dict[tuple[Any, ...], list[TreeNodeDescriptor]] = {}
    for child in braincell_children:
        grouped_braincell.setdefault(_hard_key(child), []).append(child)
    for child in neuron_children:
        grouped_neuron.setdefault(_hard_key(child), []).append(child)
    if set(grouped_braincell) != set(grouped_neuron):
        raise ValueError(
            f"Child hard-key mismatch: braincell keys={sorted(grouped_braincell)} neuron keys={sorted(grouped_neuron)}."
        )

    pairs: list[tuple[TreeNodeDescriptor, TreeNodeDescriptor]] = []
    for key in sorted(grouped_braincell):
        bc_group = grouped_braincell[key]
        nrn_group = grouped_neuron[key]
        if len(bc_group) != len(nrn_group):
            raise ValueError(f"Child multiplicity mismatch for key {key!r}.")
        pairs.extend(_optimal_group_matching(bc_group, nrn_group))
    return pairs


def _optimal_group_matching(
    braincell_group: list[TreeNodeDescriptor],
    neuron_group: list[TreeNodeDescriptor],
) -> list[tuple[TreeNodeDescriptor, TreeNodeDescriptor]]:
    size = len(braincell_group)
    if size == 0:
        return []
    cost = np.empty((size, size), dtype=float)
    for i, bc_node in enumerate(braincell_group):
        for j, nrn_node in enumerate(neuron_group):
            cost[i, j] = _node_match_score(bc_node, nrn_node)
    if not np.isfinite(cost).all():
        raise ValueError("No finite tolerant match exists inside one child group.")

    limit = 1 << size
    dp = [float("inf")] * limit
    parent: list[tuple[int, int, int] | None] = [None] * limit
    dp[0] = 0.0
    for mask in range(limit):
        i = int(bin(mask).count("1"))
        if i >= size:
            continue
        for j in range(size):
            if mask & (1 << j):
                continue
            new_mask = mask | (1 << j)
            score = dp[mask] + cost[i, j]
            if score < dp[new_mask]:
                dp[new_mask] = score
                parent[new_mask] = (mask, i, j)

    mask = limit - 1
    matched: list[tuple[TreeNodeDescriptor, TreeNodeDescriptor]] = []
    while mask:
        step = parent[mask]
        if step is None:
            raise ValueError("Failed to reconstruct tolerant child matching.")
        prev_mask, i, j = step
        matched.append((braincell_group[i], neuron_group[j]))
        mask = prev_mask
    matched.reverse()
    return matched


def _build_compartment_pairs(
    *,
    braincell_labels: list[dict[str, Any]],
    neuron_labels: list[dict[str, Any]],
    branch_index_to_section_index: dict[int, int],
    braincell_canonical_names: dict[int, str],
    neuron_canonical_names: dict[int, str],
) -> tuple[dict[str, Any], ...]:
    neuron_by_section_local = {
        (int(label["section_index"]), int(label["local_index"])): int(label["compartment_index"])
        for label in neuron_labels
    }
    pairs: list[dict[str, Any]] = []
    for braincell_label in braincell_labels:
        branch_id = int(braincell_label["branch_id"])
        local_index = int(braincell_label["local_index"])
        section_index = branch_index_to_section_index[branch_id]
        neuron_compartment_index = neuron_by_section_local.get((section_index, local_index))
        if neuron_compartment_index is None:
            raise ValueError(
                f"No NEURON compartment matches branch_id={branch_id!r}, local_index={local_index!r}."
            )
        pairs.append(
            {
                "braincell_compartment_index": int(braincell_label["compartment_index"]),
                "neuron_compartment_index": int(neuron_compartment_index),
                "braincell_branch_id": branch_id,
                "braincell_canonical_name": braincell_canonical_names[branch_id],
                "neuron_section_index": int(section_index),
                "neuron_canonical_name": neuron_canonical_names[section_index],
                "local_index": local_index,
                "braincell_canonical_label": f"{braincell_canonical_names[branch_id]}:cv{local_index}",
                "neuron_canonical_label": f"{neuron_canonical_names[section_index]}:seg{local_index}",
            }
        )
    return tuple(sorted(pairs, key=lambda item: item["braincell_compartment_index"]))


def _build_stimulus_target_pair(
    *,
    cv_per_branch: int,
    braincell_labels: list[dict[str, Any]],
    neuron_labels: list[dict[str, Any]],
    braincell_nodes: dict[int, TreeNodeDescriptor],
    branch_index_to_section_index: dict[int, int],
    braincell_canonical_names: dict[int, str],
    neuron_canonical_names: dict[int, str],
) -> dict[str, Any]:
    root_branches = [node for node in braincell_nodes.values() if node.is_root and node.branch_type == "soma"]
    if len(root_branches) != 1:
        raise ValueError("Expected exactly one root soma branch.")
    local_index = int(cv_per_branch // 2)
    root_branch_id = int(root_branches[0].node_index)
    root_section_index = int(branch_index_to_section_index[root_branch_id])

    braincell_target = next(
        label
        for label in braincell_labels
        if int(label["branch_id"]) == root_branch_id and int(label["local_index"]) == local_index
    )
    neuron_target = next(
        label
        for label in neuron_labels
        if int(label["section_index"]) == root_section_index and int(label["local_index"]) == local_index
    )
    return {
        "braincell_branch_id": root_branch_id,
        "braincell_compartment_index": int(braincell_target["compartment_index"]),
        "braincell_branch_name": braincell_target["branch_name"],
        "braincell_canonical_name": braincell_canonical_names[root_branch_id],
        "neuron_section_index": root_section_index,
        "neuron_compartment_index": int(neuron_target["compartment_index"]),
        "neuron_section_name": neuron_target["section_name"],
        "neuron_canonical_name": neuron_canonical_names[root_section_index],
        "local_index": local_index,
    }


def _build_canonical_names(
    *,
    braincell_nodes: dict[int, TreeNodeDescriptor],
    neuron_nodes: dict[int, TreeNodeDescriptor],
    branch_index_to_section_index: dict[int, int],
) -> tuple[dict[int, str], dict[int, str]]:
    roots = [node for node in braincell_nodes.values() if node.is_root]
    if len(roots) != 1:
        raise ValueError("Expected exactly one root when building canonical names.")
    root_index = int(roots[0].node_index)

    ordered_braincell_indices = sorted(
        braincell_nodes,
        key=lambda node_index: _canonical_sort_key(
            braincell_nodes[node_index],
            branch_index_to_section_index=branch_index_to_section_index,
        ),
    )
    counters: dict[str, int] = {}
    braincell_canonical_names: dict[int, str] = {}
    for node_index in ordered_braincell_indices:
        node = braincell_nodes[node_index]
        prefix = _canonical_type_prefix(node.branch_type)
        if node_index == root_index and prefix == "soma":
            ordinal = 0
            counters[prefix] = 1
        else:
            ordinal = counters.get(prefix, 0)
            counters[prefix] = ordinal + 1
        braincell_canonical_names[node_index] = f"{prefix}[{ordinal}]"

    neuron_canonical_names = {
        int(section_index): braincell_canonical_names[int(branch_index)]
        for branch_index, section_index in branch_index_to_section_index.items()
    }
    if len(neuron_canonical_names) != len(neuron_nodes):
        raise ValueError("Canonical naming did not cover all NEURON sections.")
    return braincell_canonical_names, neuron_canonical_names


def _canonical_sort_key(
    node: TreeNodeDescriptor,
    *,
    branch_index_to_section_index: dict[int, int],
) -> tuple[Any, ...]:
    parent_canonical_anchor = -1 if node.parent_index is None else int(node.parent_index)
    attach_side = 0.5 if node.parent_attach_side is None else float(node.parent_attach_side)
    mapped_section_index = int(branch_index_to_section_index[node.node_index])
    return (
        0 if node.is_root else 1,
        _canonical_type_prefix(node.branch_type),
        mapped_section_index,
        parent_canonical_anchor,
        attach_side,
        node.length_um,
        node.subtree_total_length_um,
        node.subtree_leaf_count,
        node.subtree_max_depth,
        node.node_index,
    )


def _canonical_type_prefix(branch_type: str) -> str:
    if branch_type == "soma":
        return "soma"
    if branch_type == "axon":
        return "axon"
    if branch_type in {"dend", "dendrite", "basal_dendrite", "apical_dendrite"}:
        return "dend"
    return branch_type


def _normalized_branch_type(branch_type: str) -> str:
    if branch_type in {"dend", "dendrite", "basal_dendrite", "apical_dendrite"}:
        return "dend"
    return branch_type


def _serialize_signature(signature: tuple[Any, ...]) -> list[Any]:
    serialized: list[Any] = []
    for item in signature:
        if isinstance(item, tuple):
            serialized.append(_serialize_signature(item))
        else:
            serialized.append(item)
    return serialized
