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

"""Write point-backed morphologies as standard SWC files."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import brainunit as u
import numpy as np

from braincell.morph.morphology import MorphoBranch, Morphology

_DEFAULT_SUFFIX = ".swc"
_TYPE_CODES = {
    "custom": 0,
    "soma": 1,
    "axon": 2,
    "dendrite": 3,
    "basal_dendrite": 3,
    "apical_dendrite": 4,
}


@dataclass
class _BranchPath:
    points_um: np.ndarray
    radii_um: np.ndarray
    anchor_indices: dict[float, int]
    sample_ids: list[int | None]


@dataclass(frozen=True)
class _SwcRow:
    sample_id: int
    type_code: int
    point_um: np.ndarray
    radius_um: float
    parent_id: int


def write_swc(morpho: Morphology, path: str | os.PathLike[str]) -> Path:
    """Write a morphology to a standard seven-column SWC file.

    Parameters
    ----------
    morpho : Morphology
        Morphology whose branches all provide point geometry.
    path : str or os.PathLike
        Destination path. ``.swc`` is appended when no suffix is present.

    Returns
    -------
    Path
        Final path written.

    Raises
    ------
    TypeError
        If *morpho* is not a :class:`Morphology`.
    ValueError
        If any branch lacks point geometry or contains non-finite geometry.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"write_swc() expects a Morphology instance, got {type(morpho).__name__}.")

    paths = _prepare_paths(morpho)
    rows = _build_rows(morpho, paths)
    payload = _format_rows(rows)
    return _write_text(path, payload)


def _prepare_paths(morpho: Morphology) -> dict[int, _BranchPath]:
    missing = [
        node.name
        for node in morpho.branches
        if node.branch.points_proximal is None or node.branch.points_distal is None
    ]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise ValueError(f"SWC export requires complete point geometry on every branch; missing points on: {names}.")

    return {node.index: _prepare_branch_path(node) for node in morpho.branches}


def _prepare_branch_path(node: MorphoBranch) -> _BranchPath:
    points_um, radii_um = _segment_samples(node)
    if not np.all(np.isfinite(points_um)) or not np.all(np.isfinite(radii_um)):
        raise ValueError(f"SWC export requires finite geometry; branch {node.name!r} contains non-finite values.")
    if np.any(radii_um <= 0.0):
        raise ValueError(f"SWC export requires positive radii; branch {node.name!r} contains a non-positive radius.")

    midpoint_index = None
    if any(float(child.parent_x) == 0.5 for child in node.children):
        midpoint_index = _soma_midpoint_anchor_index(node, points_um)

    anchor_indices = {0.0: 0, 1.0: len(points_um) - 1}
    if midpoint_index is not None:
        anchor_indices[0.5] = midpoint_index

    if node.parent is not None and float(node.child_x) == 1.0:
        points_um = points_um[::-1].copy()
        radii_um = radii_um[::-1].copy()
        last_index = len(points_um) - 1
        anchor_indices = {x: last_index - index for x, index in anchor_indices.items()}

    return _BranchPath(
        points_um=points_um,
        radii_um=radii_um,
        anchor_indices=anchor_indices,
        sample_ids=[None] * len(points_um),
    )


def _segment_samples(node: MorphoBranch) -> tuple[np.ndarray, np.ndarray]:
    branch = node.branch
    points_proximal = np.asarray(branch.points_proximal.to_decimal(u.um), dtype=np.float64)
    points_distal = np.asarray(branch.points_distal.to_decimal(u.um), dtype=np.float64)
    radii_proximal = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=np.float64)
    radii_distal = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=np.float64)

    points = [points_proximal[0]]
    radii = [float(radii_proximal[0])]
    for index in range(branch.n_segments):
        if not _same_sample(points[-1], radii[-1], points_proximal[index], radii_proximal[index]):
            points.append(points_proximal[index])
            radii.append(float(radii_proximal[index]))
        if not _same_sample(points[-1], radii[-1], points_distal[index], radii_distal[index]):
            points.append(points_distal[index])
            radii.append(float(radii_distal[index]))

    if len(points) < 2:
        raise ValueError(f"SWC export requires at least two distinct samples on branch {node.name!r}.")
    return np.asarray(points, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def _same_sample(point_a, radius_a, point_b, radius_b) -> bool:
    return bool(np.array_equal(np.asarray(point_a), np.asarray(point_b)) and float(radius_a) == float(radius_b))


def _same_point(point_a, point_b) -> bool:
    return bool(np.array_equal(np.asarray(point_a), np.asarray(point_b)))


def _soma_midpoint_anchor_index(node: MorphoBranch, points_um: np.ndarray) -> int:
    if node.type != "soma":
        raise ValueError(f"SWC export only supports parent_x=0.5 on soma branches, got {node.name!r}.")
    if len(points_um) < 3:
        raise ValueError(
            f"SWC export requires an existing internal soma sample for parent_x=0.5; branch {node.name!r} "
            "contains only endpoint samples."
        )

    lengths_um = np.linalg.norm(points_um[1:] - points_um[:-1], axis=1)
    cumulative_um = np.concatenate((np.array([0.0]), np.cumsum(lengths_um)))
    target_um = float(cumulative_um[-1]) * 0.5
    return min(range(1, len(points_um) - 1), key=lambda index: (abs(float(cumulative_um[index]) - target_um), index))


def _build_rows(morpho: Morphology, paths: dict[int, _BranchPath]) -> list[_SwcRow]:
    rows: list[_SwcRow] = []
    next_id = 1

    for node in morpho.branches:
        branch_path = paths[node.index]
        if node.parent is None:
            parent_id = -1
        else:
            parent_path = paths[node.parent.index]
            anchor_index = _resolve_parent_sample_index(node, paths, branch_path)
            anchor_id = parent_path.sample_ids[anchor_index]
            if anchor_id is None:
                raise RuntimeError(f"SWC parent anchor for branch {node.name!r} was not emitted.")
            parent_id = anchor_id

        type_code = _TYPE_CODES[node.type]
        for sample_index in range(len(branch_path.points_um)):
            sample_id = next_id
            next_id += 1
            rows.append(
                _SwcRow(
                    sample_id=sample_id,
                    type_code=type_code,
                    point_um=branch_path.points_um[sample_index],
                    radius_um=float(branch_path.radii_um[sample_index]),
                    parent_id=parent_id,
                )
            )
            branch_path.sample_ids[sample_index] = sample_id
            parent_id = sample_id

    return rows


def _resolve_parent_sample_index(
    node: MorphoBranch,
    paths: dict[int, _BranchPath],
    child_path: _BranchPath,
) -> int:
    parent_node = node.parent
    if parent_node is None:
        raise RuntimeError(f"SWC child branch {node.name!r} is missing its parent.")
    parent_path = paths[parent_node.index]
    child_point = child_path.points_um[0]
    child_radius = float(child_path.radii_um[0])
    declared_anchor = parent_path.anchor_indices[float(node.parent_x)]

    # NEURON treats a multi-point child edge from the soma midpoint as
    # topology-only, so the child's first geometry sample need not coincide.
    if parent_node.type == "soma" and float(node.parent_x) == 0.5:
        return declared_anchor

    declared_point_matches = _same_point(child_point, parent_path.points_um[declared_anchor])
    if parent_node.type == "soma" and declared_point_matches:
        return declared_anchor
    if declared_point_matches and child_radius == float(parent_path.radii_um[declared_anchor]):
        return declared_anchor

    if _is_con2prox_connection(node, paths, child_path):
        return 1

    point_text = ", ".join(_format_float(value) for value in child_point)
    radius_text = _format_float(child_radius)
    raise ValueError(
        f"SWC export cannot connect branch {node.name!r}: its connection endpoint "
        f"({point_text}; radius {radius_text}) does not match the declared anchor on parent {parent_node.name!r}."
    )


def _is_con2prox_connection(
    node: MorphoBranch,
    paths: dict[int, _BranchPath],
    child_path: _BranchPath,
) -> bool:
    parent_node = node.parent
    if parent_node is None or float(node.parent_x) != 0.0:
        return False
    soma_node = parent_node.parent
    if soma_node is None or soma_node.type != "soma":
        return False

    parent_path = paths[parent_node.index]
    if len(parent_path.points_um) < 2:
        return False
    soma_path = paths[soma_node.index]
    soma_anchor = soma_path.anchor_indices[float(parent_node.parent_x)]
    parent_start_collapses = _same_sample(
        parent_path.points_um[0],
        parent_path.radii_um[0],
        soma_path.points_um[soma_anchor],
        soma_path.radii_um[soma_anchor],
    )
    child_matches_second_point = _same_sample(
        child_path.points_um[0],
        child_path.radii_um[0],
        parent_path.points_um[1],
        parent_path.radii_um[1],
    )
    return parent_start_collapses and child_matches_second_point


def _format_rows(rows: list[_SwcRow]) -> str:
    lines = ["# Generated by BrainCell", "# id type x y z radius parent"]
    for row in rows:
        x, y, z = (_format_float(value) for value in row.point_um)
        lines.append(f"{row.sample_id} {row.type_code} {x} {y} {z} {_format_float(row.radius_um)} {row.parent_id}")
    return "\n".join(lines) + "\n"


def _format_float(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(os.fspath(path))
    if resolved.suffix == "":
        resolved = resolved.with_suffix(_DEFAULT_SUFFIX)
    return resolved


def _write_text(path: str | os.PathLike[str], payload: str) -> Path:
    final_path = _resolve_path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=final_path.parent,
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, final_path)
    except BaseException:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return final_path
