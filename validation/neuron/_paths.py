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

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CEREBELLUM_ROOT = DATA_ROOT / "cerebellum"
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
