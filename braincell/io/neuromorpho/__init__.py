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

"""NeuroMorpho.Org integration for BrainCell.

The package is organized in three layers, ordered from highest- to
lowest-level:

* Tier 1 — :func:`load_neuromorpho` and :func:`fetch_neuromorpho`:
  one-call entry points for the common case ("give me a Morphology for
  this neuron id"). Also exposed as :meth:`Morphology.from_neuromorpho`.
* Tier 2 — :class:`NeuroMorphoClient`: stateful client supporting
  search, batch downloads, custom HTTP sessions, and retry/backoff.
* Tier 3 — pure helpers in :mod:`braincell.io.neuromorpho.urls` plus
  the :class:`NeuroMorphoCache` / :class:`NeuroMorphoCacheLayout`
  objects: power-user building blocks.

Most users only need Tier 1; Tier 2 covers search and batch downloads;
Tier 3 is the toolkit for custom schedulers, parallel downloaders, or
alternative storage backends.
"""

from .cache import NeuroMorphoCache, NeuroMorphoCacheLayout
from .client import DEFAULT_TIMEOUT, NeuroMorphoClient
from .entry import (
    DEFAULT_USER_CACHE_DIR,
    default_cache_dir,
    fetch_neuromorpho,
    load_neuromorpho,
)
from .errors import (
    NeuroMorphoError,
    NeuroMorphoHTTPError,
    NeuroMorphoNotFoundError,
)
from .models import (
    NeuroMorphoCacheStatus,
    NeuroMorphoDetail,
    NeuroMorphoDownloadItem,
    NeuroMorphoDownloadRecord,
    NeuroMorphoFilePlan,
    NeuroMorphoMeasurement,
    NeuroMorphoNeuron,
    NeuroMorphoSearchPage,
    NeuroMorphoUrls,
)
from .query import NeuroMorphoQuery
from .urls import (
    API_BASE,
    FILE_BASE,
    DownloadMode,
    build_measurement_url,
    build_original_file_url,
    build_standard_swc_url,
    infer_original_extension,
    plan_neuron_files,
    safe_filename,
)

__all__ = [
    "API_BASE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_USER_CACHE_DIR",
    "DownloadMode",
    "FILE_BASE",
    "NeuroMorphoCache",
    "NeuroMorphoCacheLayout",
    "NeuroMorphoCacheStatus",
    "NeuroMorphoClient",
    "NeuroMorphoDetail",
    "NeuroMorphoDownloadItem",
    "NeuroMorphoDownloadRecord",
    "NeuroMorphoError",
    "NeuroMorphoFilePlan",
    "NeuroMorphoHTTPError",
    "NeuroMorphoMeasurement",
    "NeuroMorphoNeuron",
    "NeuroMorphoNotFoundError",
    "NeuroMorphoQuery",
    "NeuroMorphoSearchPage",
    "NeuroMorphoUrls",
    "build_measurement_url",
    "build_original_file_url",
    "build_standard_swc_url",
    "default_cache_dir",
    "fetch_neuromorpho",
    "infer_original_extension",
    "load_neuromorpho",
    "plan_neuron_files",
    "safe_filename",
]
