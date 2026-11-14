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

"""Shared repository data and build locations for NEURON workflows."""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CEREBELLUM_ROOT = DATA_ROOT / "cerebellum"
MODELDB_ROOT = DATA_ROOT / "modeldb"
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def model_dir(model: str) -> Path:
    """Return the source bundle for a named cerebellar model."""
    path = CEREBELLUM_ROOT / model
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown cerebellar model: {model}")
    return path


def mechanism_build_dir(model: str, kind: str = "cell") -> Path:
    """Return a workflow-owned build root (the parent of x86_64)."""
    return ARTIFACT_ROOT / "mechanisms" / model / kind


def pinned_source(path: Path, sha256: str) -> Path:
    """Return ``path`` after checking that its bytes have the SHA-256 ``sha256``.

    For third-party sources kept unmodified under ``data/`` (``data/modeldb/``):
    a comparison against them is only meaningful for the recorded file.
    Raises ``FileNotFoundError`` when the file is missing and ``ValueError``
    when its hash differs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; it is a pinned source (SHA-256 {sha256}).")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != sha256:
        raise ValueError(f"{path} has SHA-256 {actual}, not the recorded {sha256}; restore the file from its source.")
    return path


def pinned_mod_files(spec, *, owner: str) -> tuple[Path, ...]:
    """Resolve an ``identity.pinned_mod_files`` entry and check each file's SHA-256.

    ``spec`` is ``{"dir": <path under data/>, "source": <origin>, "sha256":
    {<file name>: <digest>}}``; ``owner`` (the configuration) prefixes any
    error, which is :func:`pinned_source`'s ``FileNotFoundError`` or
    ``ValueError``. Returns the files in name order.
    """
    files = []
    for name, digest in sorted(spec["sha256"].items()):
        try:
            files.append(pinned_source(DATA_ROOT / spec["dir"] / name, digest))
        except (FileNotFoundError, ValueError) as error:
            raise type(error)(f"{owner} ({spec['source']}): {error}") from None
    return tuple(files)
