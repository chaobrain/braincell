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

"""Tier-1 one-liners for downloading and loading NeuroMorpho.Org neurons.

These functions cover the 80% case (``"give me a Morphology for
neuron_id N"``) without requiring callers to instantiate a client or
manage a cache by hand. For batch search, custom HTTP sessions, or
explicit cache control, drop down to :class:`NeuroMorphoClient`.
"""



from pathlib import Path
from typing import TYPE_CHECKING

from .client import NeuroMorphoClient
from .models import NeuroMorphoDownloadRecord
from .urls import DownloadMode

if TYPE_CHECKING:
    from braincell.io.swc.types import SwcReport
    from braincell.morph._morphology import Morphology

__all__ = [
    "DEFAULT_USER_CACHE_DIR",
    "default_cache_dir",
    "fetch_neuromorpho",
    "load_neuromorpho",
]

#: Path used when ``cache_dir`` is omitted from the Tier-1 helpers.
#:
#: Resolves to ``~/.cache/braincell/neuromorpho``. The directory is
#: created on first write, not on import.
DEFAULT_USER_CACHE_DIR = Path.home() / ".cache" / "braincell" / "neuromorpho"


def default_cache_dir() -> Path:
    """Return :data:`DEFAULT_USER_CACHE_DIR` as an absolute path.

    Exposed as a function so callers and tests can monkey-patch the
    default if they need to.

    Returns
    -------
    Path
    """

    return DEFAULT_USER_CACHE_DIR


def _resolve_client(
    client: NeuroMorphoClient | None,
    cache_dir: str | Path | None,
) -> tuple[NeuroMorphoClient, Path]:
    if cache_dir is None:
        cache_root = default_cache_dir()
    else:
        cache_root = Path(cache_dir).expanduser()

    if client is None:
        client = NeuroMorphoClient(cache_dir=cache_root)
        return client, cache_root

    if client.cache is None:
        # Caller passed an existing client without a cache. Honour the
        # explicit cache_dir argument by attaching one without mutating
        # state the caller might rely on elsewhere — we just remember
        # the root locally.
        return client, cache_root
    if cache_dir is not None:
        return client, cache_root
    return client, client.cache.root


def fetch_neuromorpho(
    neuron_id: int,
    dest: str | Path | None = None,
    *,
    mode: DownloadMode = "standard",
    overwrite: bool = False,
    client: NeuroMorphoClient | None = None,
) -> NeuroMorphoDownloadRecord:
    """Download files for a NeuroMorpho.Org neuron without parsing.

    The standardized CNG SWC is fetched by default. Pass
    ``mode="both"`` to additionally fetch the Source-Version file, or
    ``mode="original"`` for only the source file.

    Parameters
    ----------
    neuron_id : int
        NeuroMorpho.Org numeric identifier (e.g. ``10047``).
    dest : str or Path or None
        Cache root. ``None`` resolves to :data:`DEFAULT_USER_CACHE_DIR`
        (``~/.cache/braincell/neuromorpho``). The directory is created
        lazily on first write.
    mode : {"standard", "original", "both"}
        Which file(s) to download.
    overwrite : bool
        If ``True``, re-download files that are already cached.
    client : NeuroMorphoClient or None
        Reuse a pre-configured client (custom session, retries, etc.).
        A transient client is created when ``None``.

    Returns
    -------
    NeuroMorphoDownloadRecord
        Per-file outcomes plus the cached metadata path.

    Raises
    ------
    NeuroMorphoNotFoundError
        If ``neuron_id`` does not exist upstream.
    NeuroMorphoHTTPError
        On non-recoverable HTTP errors.

    See Also
    --------
    load_neuromorpho : Fetch and parse in one call.
    NeuroMorphoClient.download : Lower-level download with more control.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.io import fetch_neuromorpho
        >>> record = fetch_neuromorpho(10047)  # doctest: +SKIP
        >>> record.download_items[0].path  # doctest: +SKIP
        PosixPath('.../10047/TypeA-10.CNG.swc')
    """

    resolved_client, cache_root = _resolve_client(client, dest)
    return resolved_client.download(
        neuron_id,
        output_dir=cache_root,
        mode=mode,
        overwrite=overwrite,
    )


def load_neuromorpho(
    neuron_id: int,
    *,
    cache_dir: str | Path | None = None,
    mode: str | None = "neuromorpho",
    client: NeuroMorphoClient | None = None,
    return_report: bool = False,
    overwrite: bool = False,
) -> "Morphology | tuple[Morphology, SwcReport]":
    """Fetch a NeuroMorpho.Org neuron and return it as a :class:`Morphology`.

    Downloads the standardized CNG SWC into ``cache_dir`` (defaulting
    to ``~/.cache/braincell/neuromorpho``) and parses it with
    :meth:`Morphology.from_swc` using ``mode="neuromorpho"``.

    Already-cached neurons are not re-downloaded; pass ``overwrite=True``
    to force a refresh.

    Parameters
    ----------
    neuron_id : int
        NeuroMorpho.Org numeric identifier (e.g. ``10047``).
    cache_dir : str or Path or None
        Cache root. ``None`` resolves to :data:`DEFAULT_USER_CACHE_DIR`
        (``~/.cache/braincell/neuromorpho``). The directory is created
        lazily.
    mode : str or None
        SWC import mode forwarded to :meth:`Morphology.from_swc`.
        Defaults to ``"neuromorpho"`` (copies soma attachment points
        into child branches).
    client : NeuroMorphoClient or None
        Optional pre-configured client.
    return_report : bool
        If ``True``, also return the :class:`SwcReport` from parsing.
    overwrite : bool
        If ``True``, re-download even when the file is already cached.

    Returns
    -------
    Morphology or (Morphology, SwcReport)
        The parsed morphology, optionally with a diagnostic report.

    Raises
    ------
    NeuroMorphoNotFoundError
        If ``neuron_id`` does not exist upstream.
    NeuroMorphoHTTPError
        On non-recoverable HTTP errors.
    FileNotFoundError
        If the standardized SWC file cannot be located after download
        (should not happen in practice).

    See Also
    --------
    fetch_neuromorpho : Download files without parsing.
    Morphology.from_neuromorpho : Equivalent classmethod entry point.
    Morphology.from_swc : Underlying SWC reader.

    Examples
    --------

    .. code-block:: python

        >>> from braincell import load_neuromorpho
        >>> morph = load_neuromorpho(10047)  # doctest: +SKIP
        >>> morph, report = load_neuromorpho(10047, return_report=True)  # doctest: +SKIP
    """

    from braincell.morph._morphology import Morphology

    resolved_client, cache_root = _resolve_client(client, cache_dir)
    record = resolved_client.download(
        neuron_id,
        output_dir=cache_root,
        mode="standard",
        overwrite=overwrite,
    )

    swc_path: Path | None = None
    for item in record.download_items:
        if item.kind == "standard" and item.path.exists():
            swc_path = item.path
            break

    if swc_path is None:
        # Fall back to scanning the cache folder if the download record
        # didn't surface a usable path.
        from .cache import NeuroMorphoCache

        swc_path = NeuroMorphoCache(cache_root).standard_swc_path(neuron_id)

    if swc_path is None:
        raise FileNotFoundError(
            f"Standardized SWC file for neuron_id={neuron_id} could not be "
            f"located after download (cache_dir={cache_root})."
        )

    return Morphology.from_swc(swc_path, mode=mode, return_report=return_report)
