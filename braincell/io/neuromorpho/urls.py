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

"""Pure URL and filename helpers for NeuroMorpho.Org resources.

This module contains side-effect-free helpers used by every other layer
of the client. Nothing here issues HTTP requests or touches the
filesystem; the helpers just translate ``NeuroMorphoNeuron`` records
into URLs, filenames, and download plans.
"""



import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

if TYPE_CHECKING:
    from .models import NeuroMorphoFilePlan, NeuroMorphoNeuron

__all__ = [
    "API_BASE",
    "FILE_BASE",
    "DownloadMode",
    "build_measurement_url",
    "build_original_file_url",
    "build_standard_swc_url",
    "infer_original_extension",
    "plan_neuron_files",
    "safe_filename",
]

#: Base URL of the NeuroMorpho.Org REST API.
API_BASE = "https://neuromorpho.org/api"

#: Base URL where NeuroMorpho.Org serves morphology files (CNG and Source-Version).
FILE_BASE = "https://neuromorpho.org/dableFiles"

#: Allowed values for the download mode passed to file-plan / download helpers.
DownloadMode = Literal["standard", "original", "both"]

_VALID_DOWNLOAD_MODES = frozenset({"standard", "original", "both"})


def safe_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Replaces every run of characters that are not ASCII letters, digits,
    dot, dash, or underscore with a single underscore, then trims
    leading/trailing punctuation. Returns ``"neuromorpho_neuron"`` if
    the cleaning would otherwise yield an empty string.

    Parameters
    ----------
    name : str
        Arbitrary string (typically a NeuroMorpho neuron name).

    Returns
    -------
    str
        Cleaned, filesystem-safe filename stem.

    Examples
    --------

    .. code-block:: python

        >>> safe_filename("Type A/10 (mouse)")
        'Type_A_10_mouse'
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("._") or "neuromorpho_neuron"


def coerce_https(url: str) -> str:
    """Upgrade a NeuroMorpho.Org URL from ``http://`` to ``https://``.

    Parameters
    ----------
    url : str
        URL string. Returned unchanged if it does not start with
        ``"http://"``.

    Returns
    -------
    str
        ``https://``-prefixed URL when the input was plain HTTP, otherwise
        the original string.
    """

    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def build_standard_swc_url(neuron: "NeuroMorphoNeuron") -> str:
    """Build the URL of the standardized CNG SWC file for a neuron.

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
        Neuron record. Must expose ``archive`` and ``neuron_name``.

    Returns
    -------
    str
        Direct download URL for ``<archive>/CNG version/<name>.CNG.swc``.

    Raises
    ------
    ValueError
        If ``neuron.archive`` is missing or empty.

    Examples
    --------

    .. code-block:: python

        >>> build_standard_swc_url(neuron)  # doctest: +SKIP
        'https://neuromorpho.org/dableFiles/scanziani/CNG%20version/TypeA-10.CNG.swc'
    """

    if not neuron.archive:
        raise ValueError(
            f"neuron_id={neuron.neuron_id} is missing archive metadata."
        )
    archive = quote(neuron.archive.lower(), safe="")
    neuron_name = quote(neuron.neuron_name, safe="")
    return f"{FILE_BASE}/{archive}/CNG%20version/{neuron_name}.CNG.swc"


def infer_original_extension(neuron: "NeuroMorphoNeuron") -> str:
    """Infer the filename extension of the original (Source-Version) file.

    NeuroMorpho.Org records expose the original filename via the
    ``original_format`` field (e.g. ``"TypeA-10.asc"``). This helper
    extracts and returns the extension (``".asc"``).

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
        Neuron record carrying ``original_format``.

    Returns
    -------
    str
        File extension including the leading dot.

    Raises
    ------
    ValueError
        If ``original_format`` is missing or has no extension.
    """

    original_format = neuron.original_format
    if not original_format:
        raise ValueError("This neuron does not expose an original_format field.")
    suffix = Path(original_format).suffix
    if not suffix:
        raise ValueError(
            f"Cannot infer original file extension from "
            f"original_format={original_format!r}."
        )
    return suffix


def build_original_file_url(neuron: "NeuroMorphoNeuron") -> str | None:
    """Build the URL of the original (Source-Version) file for a neuron.

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
        Neuron record. Returns ``None`` (rather than raising) when the
        URL cannot be constructed because ``archive`` or
        ``original_format`` is missing.

    Returns
    -------
    str or None
        Direct download URL, or ``None`` when no original file is available.
    """

    if not neuron.archive:
        return None
    try:
        suffix = infer_original_extension(neuron)
    except ValueError:
        return None
    archive = quote(neuron.archive.lower(), safe="")
    filename = quote(f"{neuron.neuron_name}{suffix}", safe="")
    return f"{FILE_BASE}/{archive}/Source-Version/{filename}"


def build_measurement_url(neuron: "NeuroMorphoNeuron") -> str:
    """Build the morphometry measurement URL for a neuron.

    Prefers the ``_links.measurements.href`` link returned by the API
    (upgraded to HTTPS if necessary), falling back to the conventional
    ``/api/morphometry/id/<id>`` route when no link is present.

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
        Neuron record.

    Returns
    -------
    str
        URL that returns a JSON morphometry payload.
    """

    link = neuron.payload.get("_links", {}).get("measurements", {}).get("href")
    if link:
        return coerce_https(str(link))
    return f"{API_BASE}/morphometry/id/{int(neuron.neuron_id)}"


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_DOWNLOAD_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_VALID_DOWNLOAD_MODES)}, got {mode!r}."
        )


def plan_neuron_files(
    neuron: "NeuroMorphoNeuron",
    *,
    mode: DownloadMode = "both",
) -> tuple["NeuroMorphoFilePlan", ...]:
    """Build the typed download plan for a neuron.

    The plan is a tuple of :class:`NeuroMorphoFilePlan` records, one per
    requested file kind. Plans for files that cannot be constructed
    (missing ``original_format`` for ``mode="original"`` or ``"both"``)
    are returned with ``skip=True`` and a human-readable ``reason``
    instead of raising, so callers can iterate batches without aborting.

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
        Neuron record.
    mode : {"standard", "original", "both"}
        Which file(s) to plan for.

    Returns
    -------
    tuple of NeuroMorphoFilePlan
        Typed plan records.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``{"standard", "original", "both"}``.

    Examples
    --------

    .. code-block:: python

        >>> plans = plan_neuron_files(neuron, mode="standard")  # doctest: +SKIP
        >>> plans[0].kind
        'standard'
    """

    from .models import NeuroMorphoFilePlan  # local to break import cycle

    _validate_mode(mode)
    stem = safe_filename(neuron.neuron_name)
    plans: list[NeuroMorphoFilePlan] = []
    if mode in {"standard", "both"}:
        plans.append(
            NeuroMorphoFilePlan(
                kind="standard",
                url=build_standard_swc_url(neuron),
                filename=f"{stem}.CNG.swc",
                skip=False,
                reason=None,
            )
        )
    if mode in {"original", "both"}:
        try:
            suffix = infer_original_extension(neuron)
        except ValueError as exc:
            plans.append(
                NeuroMorphoFilePlan(
                    kind="original",
                    url="",
                    filename=stem,
                    skip=True,
                    reason=str(exc),
                )
            )
        else:
            url = build_original_file_url(neuron)
            assert url is not None  # neuron.archive presence checked by caller
            plans.append(
                NeuroMorphoFilePlan(
                    kind="original",
                    url=url,
                    filename=f"{stem}{suffix}",
                    skip=False,
                    reason=None,
                )
            )
    return tuple(plans)
