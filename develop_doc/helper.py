from __future__ import annotations

from dataclasses import asdict

import pandas as pd

# ====== canonical order ======
BRANCH_KEY_ORDER = [
    "lengths",
    "length",
    "areas",
    "area",
    "volumes",
    "volume",
    "mean_radius",
    "radii_proximal",
    "radii_distal",
    "radii",
    "n_segments",
    "points_proximal",
    "points_distal",
    "points",
    "type",
]


def dataclass_full_dict(obj):
    d = asdict(obj)

    props = {
        k: getattr(obj, k)
        for k in obj.__dict__
        if isinstance(getattr(type(obj), k, None), property)
    }

    d.update(props)
    return d


def dict_to_df(data, *, key_name="key", value_name="value"):
    return pd.DataFrame(list(data.items()), columns=[key_name, value_name])


# ====== branch wrappers ======
def branch_to_df(branch, order=BRANCH_KEY_ORDER):
    d = dataclass_full_dict(branch)

    ordered_items = [
        (k, d.get(k, None))
        for k in order
        if k in d
    ]

    extra_items = [
        (k, v)
        for k, v in d.items()
        if k not in order
    ]

    ordered_items.extend(extra_items)
    return pd.DataFrame(ordered_items, columns=["key", "value"])


def _branch_has_points(branch) -> bool:
    return branch.points_proximal is not None and branch.points_distal is not None


def _summary_value(value):
    return value


# ====== morpho wrappers ======
def morpho_summary_dict(morpho):
    return dict(morpho.summary())


def morpho_summary_df(morpho):
    data = {key: _summary_value(value) for key, value in morpho_summary_dict(morpho).items()}
    return dict_to_df(data)


def morpho_branches_df(morpho, order="default"):
    rows = []
    for branch in morpho.branches_by(order=order):
        parent = branch.parent
        rows.append(
            {
                "index": branch.index_by(order=order),
                "name": branch.name,
                "type": branch.type,
                "parent": None if parent is None else parent.name,
                "parent_x": None if parent is None else float(branch.parent_x),
                "child_x": None if parent is None else float(branch.child_x),
                "n_children": branch.n_children,
                "length": branch.length,
                "has_points": _branch_has_points(branch.branch),
            }
        )
    return pd.DataFrame(rows)


def morpho_edges_df(morpho):
    rows = [
        {
            "parent": edge.parent.name,
            "child": edge.child.name,
            "parent_x": float(edge.parent_x),
            "child_x": float(edge.child_x),
        }
        for edge in morpho.edges
    ]
    return pd.DataFrame(rows)


def morpho_report_summary_dict(report):
    return {
        "n_issues": len(report.issues),
        "n_errors": report.error_count,
        "n_warnings": report.warning_count,
        "has_errors": report.has_errors,
        "has_warnings": report.has_warnings,
        "issue_codes": tuple(sorted({issue.code for issue in report.issues})),
    }


def morpho_report_summary_df(report):
    return dict_to_df(morpho_report_summary_dict(report))


def morpho_report_df(report):
    rows = []
    for issue in report.issues:
        rows.append(
            {
                "severity": issue.level,
                "code": issue.code,
                "message": issue.message,
                "line_number": getattr(issue, "line_number", None),
                "node_id": getattr(issue, "node_id", None),
                "fix_message": getattr(issue, "fix_message", None),
                "fix_applied": getattr(issue, "fix_applied", None),
            }
        )
    return pd.DataFrame(rows)


def morpho_compare_df(left, right, *, names=("left", "right")):
    left_name, right_name = names
    rows = [
        {"key": "same_tree", left_name: None, right_name: None, "match": left == right},
        {"key": "same_topology", left_name: None, right_name: None, "match": left.topo() == right.topo()},
        {"key": "n_branches", left_name: left.n_branches, right_name: right.n_branches, "match": left.n_branches == right.n_branches},
        {"key": "max_branch_order", left_name: left.max_branch_order, right_name: right.max_branch_order, "match": left.max_branch_order == right.max_branch_order},
        {"key": "total_length", left_name: left.total_length, right_name: right.total_length, "match": left.total_length == right.total_length},
        {"key": "max_path_distance", left_name: left.max_path_distance, right_name: right.max_path_distance, "match": left.max_path_distance == right.max_path_distance},
    ]
    return pd.DataFrame(rows)


__all__ = [
    "BRANCH_KEY_ORDER",
    "branch_to_df",
    "dict_to_df",
    "dataclass_full_dict",
    "morpho_branches_df",
    "morpho_compare_df",
    "morpho_edges_df",
    "morpho_report_df",
    "morpho_report_summary_df",
    "morpho_report_summary_dict",
    "morpho_summary_df",
    "morpho_summary_dict",
]
