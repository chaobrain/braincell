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

"""Typed data model for NeuroMorpho.Org records.

Every dataclass in this module is :class:`frozen<dataclasses.dataclass>`
and exposes typed attributes instead of leaking ``dict[str, Any]`` to
callers. The classes in this module are pure value objects: none of
them perform I/O or HTTP requests.
"""



import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .urls import DownloadMode

__all__ = [
    "NeuroMorphoCacheStatus",
    "NeuroMorphoDetail",
    "NeuroMorphoDownloadItem",
    "NeuroMorphoDownloadRecord",
    "NeuroMorphoFilePlan",
    "NeuroMorphoMeasurement",
    "NeuroMorphoNeuron",
    "NeuroMorphoSearchPage",
    "NeuroMorphoUrls",
]


@dataclass(frozen=True)
class NeuroMorphoNeuron:
    """A single NeuroMorpho.Org neuron record.

    Common metadata fields are exposed as typed attributes; the full
    HAL JSON payload is also retained on :attr:`payload` for callers
    who need fields not promoted to attributes.

    Parameters
    ----------
    neuron_id : int
        NeuroMorpho.Org numeric identifier.
    neuron_name : str
        Human-readable neuron name (also used as the file stem).
    archive : str or None
        Archive name (lower-cased archives are used in download URLs).
    species : str or None
    brain_region : list of str
        Brain region tags. Always a list, even when the upstream
        payload returns a single string.
    cell_type : list of str
        Cell-type tags. Always a list.
    original_format : str or None
        Original (Source-Version) filename, e.g. ``"TypeA-10.asc"``.
    png_url : str or None
        Thumbnail PNG URL (``http://`` is preserved verbatim from the API).
    payload : dict
        Full HAL JSON payload as returned by NeuroMorpho.Org.
    """

    neuron_id: int
    neuron_name: str
    archive: str | None
    species: str | None
    brain_region: tuple[str, ...]
    cell_type: tuple[str, ...]
    original_format: str | None
    png_url: str | None
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeuroMorphoNeuron":
        """Build a :class:`NeuroMorphoNeuron` from an upstream JSON payload.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON object as returned by ``/api/neuron/...`` endpoints.

        Returns
        -------
        NeuroMorphoNeuron
        """

        brain_region = payload.get("brain_region") or []
        if isinstance(brain_region, str):
            brain_region = [brain_region]
        cell_type = payload.get("cell_type") or []
        if isinstance(cell_type, str):
            cell_type = [cell_type]
        return cls(
            neuron_id=int(payload["neuron_id"]),
            neuron_name=str(payload["neuron_name"]),
            archive=payload.get("archive"),
            species=payload.get("species"),
            brain_region=tuple(brain_region),
            cell_type=tuple(cell_type),
            original_format=payload.get("original_format"),
            png_url=payload.get("png_url"),
            payload=dict(payload),
        )


@dataclass(frozen=True)
class NeuroMorphoSearchPage:
    """One page of search results from ``/api/neuron/select``.

    Parameters
    ----------
    items : tuple of NeuroMorphoNeuron
        Neurons returned for this page.
    page : int
        Zero-based page index.
    size : int
        Page size requested by the caller.
    total_pages : int
        Total number of pages reported by the upstream API.
    total_elements : int
        Total number of matching neurons.
    query_url : str
        Effective URL of the request that produced this page.
    """

    items: tuple[NeuroMorphoNeuron, ...]
    page: int
    size: int
    total_pages: int
    total_elements: int
    query_url: str


@dataclass(frozen=True)
class NeuroMorphoFilePlan:
    """A planned file download for a single neuron.

    Plan records are returned by :func:`plan_neuron_files` and
    :meth:`NeuroMorphoClient.file_plan`. A plan with ``skip=True`` is
    a deliberate non-download (e.g. the neuron has no original file)
    and ``reason`` describes why.

    Parameters
    ----------
    kind : {"standard", "original"}
        Which file the plan refers to.
    url : str
        Direct download URL. Empty when ``skip`` is ``True``.
    filename : str
        Suggested filename inside the cache folder.
    skip : bool
        ``True`` if the file cannot be downloaded.
    reason : str or None
        Explanation when ``skip`` is ``True``, otherwise ``None``.
    """

    kind: str
    url: str
    filename: str
    skip: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class NeuroMorphoDownloadItem:
    """Result of a single file download for a neuron.

    Parameters
    ----------
    kind : {"standard", "original"}
    url : str
        URL the file was fetched from. Empty when ``downloaded_now`` is
        ``False`` because the plan was skipped.
    filename : str
        Final filename inside the cache folder.
    path : Path
        Absolute path of the downloaded (or expected) file.
    downloaded_now : bool
        ``True`` when this call wrote the file to disk; ``False`` when
        the file was already cached, the plan was skipped, or the call
        was a dry run.
    reason : str or None
        Set when the file was skipped or the call was a dry run.
    """

    kind: str
    url: str
    filename: str
    path: Path
    downloaded_now: bool
    reason: str | None = None


@dataclass(frozen=True)
class NeuroMorphoUrls:
    """Bundle of URLs related to a single neuron.

    Parameters
    ----------
    standard_swc : str
        URL of the standardized CNG SWC file.
    original_file : str or None
        URL of the original (Source-Version) file, when available.
    measurement : str
        URL of the morphometry JSON resource.
    thumbnail : str or None
        Thumbnail PNG URL, normalized to HTTPS.
    """

    standard_swc: str
    original_file: str | None
    measurement: str
    thumbnail: str | None


@dataclass(frozen=True)
class NeuroMorphoCacheStatus:
    """Snapshot of what is cached on disk for a single neuron.

    Parameters
    ----------
    configured : bool
        Whether the parent client / cache has a known root directory.
    folder : Path or None
        Absolute path of the per-neuron cache folder, when configured.
    exists : bool
        ``True`` if the per-neuron folder exists on disk.
    metadata_exists : bool
        ``True`` if ``metadata.json`` exists in the folder.
    standard_exists : bool
        ``True`` if the standardized SWC file is present.
    original_exists : bool
        ``True`` if the original (Source-Version) file is present.
    neuron_id : int or None
        The neuron id this status describes.
    """

    configured: bool
    folder: Path | None
    exists: bool
    metadata_exists: bool
    standard_exists: bool
    original_exists: bool
    neuron_id: int | None


@dataclass(frozen=True)
class NeuroMorphoDetail:
    """Aggregate description of a neuron, including measurement and URLs.

    Returned by :meth:`NeuroMorphoClient.describe`.

    Parameters
    ----------
    neuron : NeuroMorphoNeuron
    measurement : NeuroMorphoMeasurement or None
        Typed morphometry record. ``None`` if the caller asked to skip
        measurement fetching.
    urls : NeuroMorphoUrls
        Resolved URLs for SWC, original file, measurement, and thumbnail.
    cache_status : NeuroMorphoCacheStatus
        On-disk cache status; always present even when no cache is
        configured (``configured=False``).
    """

    neuron: NeuroMorphoNeuron
    measurement: "NeuroMorphoMeasurement | None"
    urls: NeuroMorphoUrls
    cache_status: NeuroMorphoCacheStatus

    @property
    def thumbnail_url(self) -> str | None:
        """Convenience accessor for ``self.urls.thumbnail``."""

        return self.urls.thumbnail

    @property
    def standard_swc_url(self) -> str:
        """Convenience accessor for ``self.urls.standard_swc``."""

        return self.urls.standard_swc

    @property
    def original_file_url(self) -> str | None:
        """Convenience accessor for ``self.urls.original_file``."""

        return self.urls.original_file


@dataclass(frozen=True)
class NeuroMorphoDownloadRecord:
    """Result of a multi-file download for a neuron.

    Returned by :meth:`NeuroMorphoClient.download` and the
    :func:`fetch_neuromorpho` Tier-1 helper.

    Parameters
    ----------
    folder : Path
        Per-neuron cache folder.
    metadata_path : Path
        Path of the ``metadata.json`` written alongside the downloads.
    download_items : tuple of NeuroMorphoDownloadItem
        Per-file download outcomes.
    measurement : NeuroMorphoMeasurement or None
        Typed morphometry record fetched alongside the files. ``None``
        when the call was a dry run.
    download_mode : str
        The download mode used; one of ``"standard"``, ``"original"``,
        ``"both"``.
    dry_run : bool
        ``True`` if the record was produced without touching the network
        or filesystem.
    """

    folder: Path
    metadata_path: Path
    download_items: tuple[NeuroMorphoDownloadItem, ...]
    measurement: "NeuroMorphoMeasurement | None"
    download_mode: DownloadMode
    dry_run: bool = False


# ---------------------------------------------------------------------------
# NeuroMorphoMeasurement
# ---------------------------------------------------------------------------

#: Mapping from snake_case attribute name to upstream NeuroMorpho.Org key(s).
#:
#: The first matching key wins. Both snake_case and camelCase upstream
#: spellings are listed because NeuroMorpho.Org returns a mixture.
_MEASUREMENT_FIELD_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("surface", ("surface", "Surface")),
    ("volume", ("volume", "Volume")),
    ("soma_surface", ("soma_Surface", "soma_surface", "somaSurface")),
    ("n_stems", ("n_stems", "N_stems")),
    ("n_bifs", ("n_bifs", "N_bifs")),
    ("n_branch", ("n_branch", "N_branch")),
    ("width", ("width", "Width")),
    ("height", ("height", "Height")),
    ("depth", ("depth", "Depth")),
    ("diameter", ("diameter", "Diameter")),
    ("length", ("length", "Length")),
    ("euclidean_distance", ("eucDistance", "euc_distance")),
    ("path_distance", ("pathDistance", "path_distance")),
    ("branch_order", ("branch_Order", "branch_order")),
    ("contraction", ("contraction", "Contraction")),
    ("fragmentation", ("fragmentation", "Fragmentation")),
    ("partition_asymmetry", ("partition_asymmetry", "partitionAsymmetry")),
    ("pk_classic", ("pk_classic", "pkClassic")),
    ("bif_ampl_local", ("bif_ampl_local", "bifAmplLocal")),
    ("bif_ampl_remote", ("bif_ampl_remote", "bifAmplRemote")),
)

_INT_FIELDS = frozenset({"n_stems", "n_bifs", "n_branch", "branch_order", "fragmentation"})


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NeuroMorphoMeasurement:
    """Typed view over a NeuroMorpho.Org morphometry record.

    All quantitative fields are bare floats using the units documented
    by NeuroMorpho.Org: lengths in micrometres (μm), surfaces in square
    micrometres (μm²), volumes in cubic micrometres (μm³), angles in
    degrees. Missing fields are ``None`` rather than ``NaN``.

    The dataclass only promotes the fields that NeuroMorpho.Org returns
    most consistently. Less common fields remain available through
    :attr:`extras` and the :meth:`get` accessor.

    Parameters
    ----------
    neuron_id : int
        NeuroMorpho.Org numeric id this measurement belongs to.
    surface : float or None
        Total neuritic surface area (μm²).
    volume : float or None
        Total neuritic volume (μm³).
    soma_surface : float or None
        Soma surface area (μm²).
    n_stems : int or None
        Number of stems leaving the soma.
    n_bifs : int or None
        Number of bifurcations.
    n_branch : int or None
        Number of branches (excluding soma).
    width : float or None
        Bounding-box width (μm).
    height : float or None
        Bounding-box height (μm).
    depth : float or None
        Bounding-box depth (μm).
    diameter : float or None
        Average branch diameter (μm).
    length : float or None
        Total neuritic length (μm).
    euclidean_distance : float or None
        Maximum Euclidean distance from soma (μm).
    path_distance : float or None
        Maximum path distance from soma (μm).
    branch_order : int or None
        Maximum branch order.
    contraction : float or None
        Average branch contraction.
    fragmentation : int or None
        Average compartment count per branch.
    partition_asymmetry : float or None
    pk_classic : float or None
    bif_ampl_local : float or None
    bif_ampl_remote : float or None
    extras : Mapping[str, Any]
        Long-tail measurement keys not promoted to dataclass fields.
    raw : Mapping[str, Any]
        Read-only view of the upstream JSON payload (with the original
        upstream key spellings preserved). Useful when callers want to
        forward the unmodified upstream record or compare against
        documentation.

    Examples
    --------

    .. code-block:: python

        >>> meas = NeuroMorphoMeasurement.from_payload(
        ...     {"neuron_id": 10047, "n_stems": 1.0, "length": 123.4}
        ... )
        >>> meas.n_stems, meas.length
        (1, 123.4)
        >>> meas.get("custom_field", "missing")
        'missing'
    """

    neuron_id: int
    surface: float | None = None
    volume: float | None = None
    soma_surface: float | None = None
    n_stems: int | None = None
    n_bifs: int | None = None
    n_branch: int | None = None
    width: float | None = None
    height: float | None = None
    depth: float | None = None
    diameter: float | None = None
    length: float | None = None
    euclidean_distance: float | None = None
    path_distance: float | None = None
    branch_order: int | None = None
    contraction: float | None = None
    fragmentation: int | None = None
    partition_asymmetry: float | None = None
    pk_classic: float | None = None
    bif_ampl_local: float | None = None
    bif_ampl_remote: float | None = None
    extras: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NeuroMorphoMeasurement":
        """Build a :class:`NeuroMorphoMeasurement` from a JSON payload.

        Missing keys map to ``None``. Unrecognized keys are stashed in
        :attr:`extras`.

        Parameters
        ----------
        payload : Mapping[str, Any]
            JSON object returned by ``/api/morphometry/id/<id>``.

        Returns
        -------
        NeuroMorphoMeasurement
        """

        consumed: set[str] = set()
        kwargs: dict[str, Any] = {}
        for attr, candidates in _MEASUREMENT_FIELD_MAP:
            value: Any = None
            for key in candidates:
                if key in payload and payload[key] is not None:
                    value = payload[key]
                    consumed.add(key)
                    break
            if attr in _INT_FIELDS:
                kwargs[attr] = _coerce_int(value)
            else:
                kwargs[attr] = _coerce_float(value)

        neuron_id_raw = payload.get("neuron_id")
        if neuron_id_raw is None:
            raise ValueError("Measurement payload is missing 'neuron_id'.")
        consumed.add("neuron_id")

        extras = {
            key: value
            for key, value in payload.items()
            if key not in consumed
        }
        return cls(
            neuron_id=int(neuron_id_raw),
            extras=MappingProxyType(extras),
            raw=MappingProxyType(dict(payload)),
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a flat ``dict`` view including ``extras`` keys.

        Returns
        -------
        dict
            All promoted attributes plus the contents of ``extras``,
            with attribute values taking precedence on key collisions.
        """

        out: dict[str, Any] = dict(self.extras)
        for attr, _ in _MEASUREMENT_FIELD_MAP:
            out[attr] = getattr(self, attr)
        out["neuron_id"] = self.neuron_id
        return out

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a measurement value by attribute or extras key.

        Promoted attributes are tried first; otherwise the lookup falls
        through to :attr:`extras`. Returns ``default`` when neither
        contains the key.

        Parameters
        ----------
        key : str
        default : Any

        Returns
        -------
        Any
        """

        _field_names = frozenset(f.name for f in dataclasses.fields(self))
        if key in _field_names:
            return getattr(self, key)
        return self.extras.get(key, default)
