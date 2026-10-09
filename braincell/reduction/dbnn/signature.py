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

"""Stable structural signatures for DBNN-compatible Cells."""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

import brainunit as u
import numpy as np

from braincell._multi_compartment.cell import Cell
from braincell.mech import Channel, Ion


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CellStructuralSignature:
    """Store an explicit Cell type or a canonical inferred structure key.

    Parameters
    ----------
    cell_type : str or None
        Preferred caller-provided stable type. ``None`` denotes inference.
    canonical_json : str
        Canonical JSON payload from which the key was derived.
    digest : str
        SHA-256 digest of ``canonical_json``.
    """

    cell_type: str | None
    canonical_json: str
    digest: str

    def __post_init__(self) -> None:
        if self.cell_type is not None and not isinstance(self.cell_type, str):
            raise TypeError("cell_type must be a string or None.")
        if isinstance(self.cell_type, str) and not self.cell_type.strip():
            raise ValueError("cell_type must be None or a non-empty string.")
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string.")
        try:
            parsed = json.loads(self.canonical_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical_json must contain valid JSON.") from exc
        if _canonical_json(parsed) != self.canonical_json:
            raise ValueError("canonical_json must use sorted keys and compact canonical separators.")
        if not isinstance(self.digest, str) or self.digest != _digest(self.canonical_json):
            raise ValueError("digest must be the SHA-256 digest of canonical_json.")

    @property
    def key(self) -> tuple[Literal["explicit", "inferred"], str]:
        """Return the matching key, preferring the explicit Cell type."""
        if self.cell_type is not None:
            return ("explicit", self.cell_type)
        return ("inferred", self.digest)


def build_cell_signature(cell: Cell, *, cell_type: str | None = None) -> CellStructuralSignature:
    """Build an explicit or inferred declaration-time Cell signature.

    Parameters
    ----------
    cell : Cell
        Concrete Cell declaration to classify.
    cell_type : str or None, optional
        Explicit stable type key. When supplied, Cell structure is not used.

    Returns
    -------
    CellStructuralSignature
        Immutable matching signature.

    Raises
    ------
    TypeError
        If ``cell`` is not a Cell or ``cell_type`` is not a string or None.
    ValueError
        If ``cell_type`` is empty.
    RuntimeError
        If structural inference is requested after Cell initialization.
    """
    if not isinstance(cell, Cell):
        raise TypeError(f"cell must be a BrainCell Cell, got {type(cell).__name__!r}.")
    _require_pre_init(cell)
    if cell_type is not None:
        if not isinstance(cell_type, str):
            raise TypeError("cell_type must be a string or None.")
        if not cell_type.strip():
            raise ValueError("cell_type must be a non-empty string.")
        payload = _canonical_json({"cell_type": cell_type})
        return CellStructuralSignature(cell_type, payload, _digest(payload))

    branches = []
    for branch in cell.morpho.branches:
        parent = branch.parent
        geometry = branch.branch
        lengths_um = np.asarray(geometry.lengths.to_decimal(u.um), dtype=float)
        branches.append(
            {
                "children": sorted(child.index for child in branch.children),
                "geometry": {
                    "has_points": geometry.points_proximal is not None,
                    "segments": int(lengths_um.size),
                    "zero_length_segments": int(np.count_nonzero(lengths_um == 0.0)),
                },
                "kind": str(branch.type),
                "local_id": int(branch.index),
                "name": str(branch.name),
                "parent_id": None if parent is None else int(parent.index),
                "parent_x": None if parent is None else float(branch.parent_x),
                "child_x": None if parent is None else float(branch.child_x),
            }
        )

    cvs = []
    for cv in sorted(cell.cvs, key=lambda item: item.id):
        installed = sorted(
            {
                (mechanism.category, mechanism.class_name)
                for mechanism in cv.density_mech
                if isinstance(mechanism, (Channel, Ion))
            }
        )
        cvs.append(
            {
                "branch_id": int(cv.branch_id),
                "children": sorted(int(item) for item in cv.children_cv),
                "id": int(cv.id),
                "installed": [list(item) for item in installed],
                "parent_id": None if cv.parent_cv is None else int(cv.parent_cv),
            }
        )
    payload = _canonical_json(
        {
            "branches": branches,
            "cv_structure": cvs,
            "logical_synapse_count": len(cell.synapses),
            "root_branch_id": int(cell.morpho.root.index),
        }
    )
    return CellStructuralSignature(None, payload, _digest(payload))


def _require_pre_init(cell: Cell) -> None:
    if cell._initialized:
        raise RuntimeError("Cell signatures must be derived before Cell.init_state().")


__all__ = [
    "CellStructuralSignature",
    "build_cell_signature",
]
