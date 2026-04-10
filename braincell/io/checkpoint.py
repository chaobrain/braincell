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

"""Self-contained checkpoint format for :class:`Branch` and :class:`Morpho`.

Checkpoints are a lossless on-disk snapshot of in-memory morphology objects.
Unlike SWC/ASC, they preserve information that interchange formats discard:
branch *names*, ``parent_x``/``child_x`` (including ``parent_x=0.5`` soma
midpoint attachments), the canonical zero-length jump segments inserted at
radius discontinuities, and the auto-naming counters consulted by
``Morpho.attach``.

The wire format is a single ``np.savez`` archive (a plain zip of ``.npy``
files) containing:

* ``manifest`` — a 0-d ``uint8`` array holding UTF-8 JSON with topology and
  metadata.
* ``branches/{i}/lengths`` / ``radii_proximal`` / ``radii_distal`` — required
  ``float64`` arrays in ``u.um`` for branch *i* in the morphology's default
  ordering.
* ``branches/{i}/points_proximal`` / ``points_distal`` — optional
  ``float64`` arrays of shape ``(n_segments, 3)`` in ``u.um``, present only
  when the branch was created with full 3-D point geometry.

The default file extension is ``.bcm`` ("BrainCell Morphology"). The save
helpers append it automatically when the supplied path has no suffix.
"""

from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np

from braincell._version import __version__ as _BC_VERSION
from braincell.morpho import Branch, Morpho, branch_class_for_type

__all__ = [
    "CheckpointError",
    "CheckpointVersionError",
    "load_branch",
    "load_morpho",
    "save_branch",
    "save_morpho",
]

_FORMAT = "braincell.io.checkpoint"
_CURRENT_VERSION = 1
_DEFAULT_SUFFIX = ".bcm"


class CheckpointError(ValueError):
    """Raised when a braincell checkpoint cannot be parsed or validated."""


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint's schema version is not supported by this build."""


# ---------------------------------------------------------------------------
# Branch helpers
# ---------------------------------------------------------------------------


def _branch_arrays(branch: Branch) -> dict[str, np.ndarray]:
    """Extract a branch's geometry as a dict of ``float64`` arrays in ``u.um``."""
    arrays: dict[str, np.ndarray] = {
        "lengths": np.asarray(branch.lengths.to_decimal(u.um), dtype=np.float64),
        "radii_proximal": np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=np.float64),
        "radii_distal": np.asarray(branch.radii_distal.to_decimal(u.um), dtype=np.float64),
    }
    if branch.points_proximal is not None and branch.points_distal is not None:
        arrays["points_proximal"] = np.asarray(branch.points_proximal.to_decimal(u.um), dtype=np.float64)
        arrays["points_distal"] = np.asarray(branch.points_distal.to_decimal(u.um), dtype=np.float64)
    return arrays


def _build_branch(arrays: dict[str, np.ndarray], *, branch_type: str) -> Branch:
    """Reconstruct a typed :class:`Branch` subclass from raw arrays."""
    cls = branch_class_for_type(branch_type)
    kwargs: dict[str, Any] = {
        "lengths": np.asarray(arrays["lengths"], dtype=np.float64) * u.um,
        "radii_proximal": np.asarray(arrays["radii_proximal"], dtype=np.float64) * u.um,
        "radii_distal": np.asarray(arrays["radii_distal"], dtype=np.float64) * u.um,
    }
    if "points_proximal" in arrays:
        kwargs["points_proximal"] = np.asarray(arrays["points_proximal"], dtype=np.float64) * u.um
        kwargs["points_distal"] = np.asarray(arrays["points_distal"], dtype=np.float64) * u.um
    # Typed subclasses use their dataclass-default ``type`` value; do NOT pass
    # ``type=`` here, otherwise the post-init invariant double-checks would
    # still pass but we'd be relying on the (currently relaxed) base-class
    # path that accepts an explicit type. Going through the default keeps
    # subclasses honest.
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_branch(branch: Branch, path: str | os.PathLike) -> Path:
    """Write a :class:`Branch` to a braincell checkpoint file.

    Parameters
    ----------
    branch : Branch
        Branch instance to serialize.
    path : str or os.PathLike
        Destination path. If the path has no suffix, ``.bcm`` is appended
        automatically.

    Returns
    -------
    Path
        The final path the checkpoint was written to.

    Raises
    ------
    TypeError
        If *branch* is not a :class:`Branch` instance.

    See Also
    --------
    load_branch : Inverse operation.
    save_morpho : Persist a whole morphology tree.
    """
    if not isinstance(branch, Branch):
        raise TypeError(
            f"save_branch() expects a Branch instance, got {type(branch).__name__}."
        )
    arrays = _branch_arrays(branch)
    manifest = {
        "format": _FORMAT,
        "version": _CURRENT_VERSION,
        "kind": "branch",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "braincell_version": _BC_VERSION,
        "unit": "um",
        "branch": {
            "type": branch.type,
            "n_segments": int(branch.n_segments),
            "has_points": "points_proximal" in arrays,
        },
    }
    payload = {f"branches/0/{key}": value for key, value in arrays.items()}
    return _write_npz(path, manifest, payload)


def load_branch(path: str | os.PathLike) -> Branch:
    """Load a :class:`Branch` from a braincell checkpoint file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.bcm`` checkpoint produced by :func:`save_branch`.

    Returns
    -------
    Branch
        The reconstructed branch as the appropriate typed subclass
        (e.g. :class:`Soma`, :class:`Dendrite`).

    Raises
    ------
    CheckpointError
        If the file is missing, not a braincell checkpoint, corrupt, or
        is a morphology checkpoint instead of a branch checkpoint.
    CheckpointVersionError
        If the checkpoint version is newer than this build supports.

    See Also
    --------
    save_branch : Inverse operation.
    load_morpho : Load a whole morphology tree.
    """
    manifest, payload = _read_npz(path)
    _check_format(manifest, source=path)
    if manifest.get("kind") != "branch":
        raise CheckpointError(
            f"{os.fspath(path)!s}: this is a {manifest.get('kind')!r} checkpoint; "
            "use load_morpho() instead of load_branch()."
        )
    spec = manifest.get("branch")
    if not isinstance(spec, dict) or "type" not in spec:
        raise CheckpointError(
            f"{os.fspath(path)!s}: branch checkpoint manifest is missing the 'branch' section."
        )
    arrays = _collect_branch_arrays(
        payload,
        index=0,
        has_points=bool(spec.get("has_points", False)),
        source=path,
    )
    return _build_branch(arrays, branch_type=spec["type"])


def save_morpho(morpho: Morpho, path: str | os.PathLike) -> Path:
    """Write a :class:`Morpho` to a braincell checkpoint file.

    Parameters
    ----------
    morpho : Morpho
        Morphology to serialize. The current default-ordering snapshot is
        captured; the in-memory tree is not modified.
    path : str or os.PathLike
        Destination path. If the path has no suffix, ``.bcm`` is appended
        automatically.

    Returns
    -------
    Path
        The final path the checkpoint was written to.

    Raises
    ------
    TypeError
        If *morpho* is not a :class:`Morpho` instance.

    See Also
    --------
    load_morpho : Inverse operation.
    save_branch : Persist a single branch.
    """
    if not isinstance(morpho, Morpho):
        raise TypeError(
            f"save_morpho() expects a Morpho instance, got {type(morpho).__name__}."
        )

    branches_meta: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}

    # Default order is sorted node IDs == insertion order, which is also
    # guaranteed to be parent-before-child.
    for index, node in enumerate(morpho.branches):
        arrays = _branch_arrays(node.branch)
        for key, value in arrays.items():
            payload[f"branches/{index}/{key}"] = value

        parent = node.parent
        branches_meta.append(
            {
                "index": index,
                "name": node.name,
                "type": node.branch.type,
                "parent_name": None if parent is None else parent.name,
                "parent_x": None if parent is None else float(node.parent_x),
                "child_x": None if parent is None else float(node.child_x),
                "n_segments": int(node.branch.n_segments),
                "has_points": "points_proximal" in arrays,
            }
        )

    manifest = {
        "format": _FORMAT,
        "version": _CURRENT_VERSION,
        "kind": "morpho",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "braincell_version": _BC_VERSION,
        "unit": "um",
        "root_name": morpho.root.name,
        "type_name_counters": dict(morpho._type_name_counters),
        "branches": branches_meta,
    }
    return _write_npz(path, manifest, payload)


def load_morpho(path: str | os.PathLike) -> Morpho:
    """Load a :class:`Morpho` from a braincell checkpoint file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.bcm`` checkpoint produced by :func:`save_morpho`.

    Returns
    -------
    Morpho
        The reconstructed morphology, with the same branch names,
        ``parent_x``/``child_x`` attachments, and auto-naming counters
        as the saved tree.

    Raises
    ------
    CheckpointError
        If the file is missing, not a braincell checkpoint, corrupt, or
        is a branch checkpoint instead of a morphology checkpoint, or if
        the manifest references a non-existent or out-of-order parent.
    CheckpointVersionError
        If the checkpoint version is newer than this build supports.

    See Also
    --------
    save_morpho : Inverse operation.
    load_branch : Load a single branch.
    """
    manifest, payload = _read_npz(path)
    _check_format(manifest, source=path)
    if manifest.get("kind") != "morpho":
        raise CheckpointError(
            f"{os.fspath(path)!s}: this is a {manifest.get('kind')!r} checkpoint; "
            "use load_branch() instead of load_morpho()."
        )

    branch_specs = manifest.get("branches")
    if not isinstance(branch_specs, list) or not branch_specs:
        raise CheckpointError(
            f"{os.fspath(path)!s}: morpho checkpoint has no branches."
        )

    by_name: dict[str, dict[str, Any]] = {}
    for spec in branch_specs:
        if not isinstance(spec, dict) or "name" not in spec:
            raise CheckpointError(
                f"{os.fspath(path)!s}: malformed branch entry in manifest."
            )
        if spec["name"] in by_name:
            raise CheckpointError(
                f"{os.fspath(path)!s}: duplicate branch name {spec['name']!r} in manifest."
            )
        by_name[spec["name"]] = spec

    root_specs = [spec for spec in branch_specs if spec.get("parent_name") is None]
    if len(root_specs) != 1:
        raise CheckpointError(
            f"{os.fspath(path)!s}: expected exactly one root branch, "
            f"found {len(root_specs)}."
        )
    root_spec = root_specs[0]
    expected_root_name = manifest.get("root_name")
    if expected_root_name != root_spec["name"]:
        raise CheckpointError(
            f"{os.fspath(path)!s}: root_name={expected_root_name!r} does not match "
            f"the root branch entry {root_spec['name']!r}."
        )

    root_arrays = _collect_branch_arrays(
        payload,
        index=int(root_spec["index"]),
        has_points=bool(root_spec.get("has_points", False)),
        source=path,
    )
    root_branch = _build_branch(root_arrays, branch_type=root_spec["type"])
    morpho = Morpho.from_root(root_branch, name=root_spec["name"])

    inserted: set[str] = {root_spec["name"]}
    for spec in branch_specs:
        if spec["name"] in inserted:
            continue
        parent_name = spec.get("parent_name")
        if parent_name is None:
            raise CheckpointError(
                f"{os.fspath(path)!s}: multiple root branches detected at {spec['name']!r}."
            )
        if parent_name not in inserted:
            raise CheckpointError(
                f"{os.fspath(path)!s}: branch {spec['name']!r} references parent "
                f"{parent_name!r} which has not yet been inserted "
                "(cycle or out-of-order manifest)."
            )
        arrays = _collect_branch_arrays(
            payload,
            index=int(spec["index"]),
            has_points=bool(spec.get("has_points", False)),
            source=path,
        )
        child_branch = _build_branch(arrays, branch_type=spec["type"])
        morpho.attach(
            parent=parent_name,
            child_branch=child_branch,
            child_name=spec["name"],
            parent_x=float(spec["parent_x"]),
            child_x=float(spec["child_x"]),
        )
        inserted.add(spec["name"])

    counters = manifest.get("type_name_counters", {})
    if isinstance(counters, dict):
        for branch_type, count in counters.items():
            try:
                morpho._type_name_counters[str(branch_type)] = int(count)
            except (TypeError, ValueError) as exc:
                raise CheckpointError(
                    f"{os.fspath(path)!s}: invalid type_name_counters entry "
                    f"{branch_type!r}={count!r}."
                ) from exc

    return morpho


# ---------------------------------------------------------------------------
# Low-level NPZ helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: str | os.PathLike) -> Path:
    p = Path(os.fspath(path))
    if p.suffix == "":
        p = p.with_suffix(_DEFAULT_SUFFIX)
    return p


def _write_npz(
    path: str | os.PathLike,
    manifest: dict[str, Any],
    payload: dict[str, np.ndarray],
) -> Path:
    final_path = _resolve_path(path)
    if final_path.parent and not final_path.parent.exists():
        final_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    manifest_array = np.frombuffer(manifest_bytes, dtype=np.uint8)

    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    try:
        # ``np.savez`` always appends ``.npz`` when given a path/string. Pass
        # a writable file object instead so we control the final extension.
        with open(tmp_path, "wb") as fh:
            np.savez(fh, manifest=manifest_array, **payload)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    os.replace(tmp_path, final_path)
    return final_path


def _read_npz(path: str | os.PathLike) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p!s}")
    try:
        with np.load(p, allow_pickle=False) as data:
            keys = list(data.files)
            if "manifest" not in keys:
                raise CheckpointError(
                    f"{p!s}: not a braincell checkpoint (missing 'manifest' entry)."
                )
            manifest_array = np.asarray(data["manifest"])
            payload = {key: np.asarray(data[key]) for key in keys if key != "manifest"}
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"{p!s}: cannot read checkpoint: {exc}") from exc

    try:
        manifest_bytes = bytes(manifest_array.tobytes())
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{p!s}: corrupt manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise CheckpointError(
            f"{p!s}: manifest must be a JSON object, got {type(manifest).__name__}."
        )
    return manifest, payload


def _check_format(manifest: dict[str, Any], *, source: str | os.PathLike) -> None:
    fmt = manifest.get("format")
    if fmt != _FORMAT:
        raise CheckpointError(
            f"{os.fspath(source)!s}: unrecognized checkpoint format {fmt!r}; "
            f"expected {_FORMAT!r}."
        )
    version = manifest.get("version")
    if not isinstance(version, int) or version < 1:
        raise CheckpointError(
            f"{os.fspath(source)!s}: invalid checkpoint version {version!r}."
        )
    if version > _CURRENT_VERSION:
        raise CheckpointVersionError(
            f"{os.fspath(source)!s}: checkpoint version {version} is newer than "
            f"this build of braincell supports (max version {_CURRENT_VERSION}); "
            "please upgrade braincell."
        )


def _collect_branch_arrays(
    payload: dict[str, np.ndarray],
    *,
    index: int,
    has_points: bool,
    source: str | os.PathLike,
) -> dict[str, np.ndarray]:
    base = f"branches/{index}/"
    required = ("lengths", "radii_proximal", "radii_distal")
    arrays: dict[str, np.ndarray] = {}
    for key in required:
        full_key = base + key
        if full_key not in payload:
            raise CheckpointError(
                f"{os.fspath(source)!s}: missing required array {full_key!r}."
            )
        arrays[key] = payload[full_key]
    if has_points:
        for key in ("points_proximal", "points_distal"):
            full_key = base + key
            if full_key not in payload:
                raise CheckpointError(
                    f"{os.fspath(source)!s}: missing required array {full_key!r}."
                )
            arrays[key] = payload[full_key]
    return arrays
