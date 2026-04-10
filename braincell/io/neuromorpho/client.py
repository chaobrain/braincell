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

"""Stateful HTTP client for the NeuroMorpho.Org REST API.

The client is the Tier-2 entry point for the NeuroMorpho.Org integration.
For one-shot use cases, prefer the Tier-1 helpers in
:mod:`braincell.io.neuromorpho.entry`.
"""



import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .cache import NeuroMorphoCache
from .errors import NeuroMorphoError, NeuroMorphoHTTPError
from .http import request_with_retry
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
    DownloadMode,
    build_measurement_url,
    build_original_file_url,
    build_standard_swc_url,
    coerce_https,
    plan_neuron_files,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "NeuroMorphoClient",
]

#: Default per-request timeout in seconds.
DEFAULT_TIMEOUT = 30.0


class NeuroMorphoClient:
    """Stateful client for the NeuroMorpho.Org REST API.

    The client wraps a ``requests.Session`` (lazily imported on first
    construction so the package can be imported without ``requests``
    installed) and adds retry/backoff for transient failures, an
    optional :class:`NeuroMorphoCache`, and typed return values.

    Parameters
    ----------
    session : object or None
        Session-like object exposing ``get(url, params=..., timeout=...,
        stream=...)``. Pass a configured ``requests.Session`` to control
        proxies, headers, or HTTPS verification. ``None`` constructs a
        fresh ``requests.Session`` (importing ``requests`` lazily).
    timeout : float
        Per-request timeout in seconds. Defaults to
        :data:`DEFAULT_TIMEOUT`.
    cache_dir : str or Path or None
        Root directory for cached downloads. When provided, the client
        creates a :class:`NeuroMorphoCache` accessible via :attr:`cache`.
    retries : int
        Number of attempts for JSON API calls. Must be ``>= 1``.
        Streaming file downloads are not retried.
    backoff_base : float
        Base delay (seconds) for exponential backoff between retries.

    Attributes
    ----------
    session : object
    timeout : float
    cache : NeuroMorphoCache or None
        Set when ``cache_dir`` was provided.
    retries : int
    backoff_base : float

    Examples
    --------

    .. code-block:: python

        >>> client = NeuroMorphoClient(cache_dir="~/data/neuromorpho")
        >>> page = client.search("species:mouse", size=5)  # doctest: +SKIP
        >>> for neuron in client.iter_search("species:mouse", limit=20):
        ...     client.download(neuron)  # doctest: +SKIP
    """

    def __init__(
        self,
        session: Any = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        cache_dir: str | Path | None = None,
        retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        if session is None:
            try:
                import requests
            except Exception as exc:  # pragma: no cover - depends on env
                raise ImportError(
                    "The 'requests' library is required to use NeuroMorphoClient. "
                    "Install it with 'pip install requests'."
                ) from exc
            session = requests.Session()

        if retries < 1:
            raise ValueError(f"retries must be >= 1, got {retries!r}")

        self.session = session
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff_base = float(backoff_base)
        self._cache: NeuroMorphoCache | None = (
            NeuroMorphoCache(cache_dir) if cache_dir is not None else None
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @property
    def cache(self) -> NeuroMorphoCache | None:
        """Return the attached :class:`NeuroMorphoCache`, if configured."""

        return self._cache

    @property
    def cache_dir(self) -> Path | None:
        """Convenience accessor for the cache root directory."""

        return self._cache.root if self._cache is not None else None

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        response = request_with_retry(
            self.session,
            url,
            timeout=self.timeout,
            attempts=self.retries,
            backoff_base=self.backoff_base,
            params=params,
        )
        return response.json(), str(getattr(response, "url", url))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str | NeuroMorphoQuery,
        *,
        fq: list[str] | None = None,
        size: int = 20,
        page: int = 0,
        sort: str = "neuron_id,asc",
    ) -> NeuroMorphoSearchPage:
        """Run a single search request against ``/api/neuron/select``.

        Parameters
        ----------
        query : str or NeuroMorphoQuery
            Either a raw Solr ``q`` string or a typed
            :class:`NeuroMorphoQuery`. When a query object is passed,
            its ``raw_fq`` is merged with the explicit ``fq`` argument.
        fq : list of str or None
            Extra Solr filter strings to append to ``fq``.
        size : int
            Page size requested from the API.
        page : int
            Zero-based page index.
        sort : str
            Sort string forwarded to the API.

        Returns
        -------
        NeuroMorphoSearchPage
        """

        params: dict[str, Any] = {"size": size, "page": page, "sort": sort}
        merged_fq: list[str] = []
        if isinstance(query, NeuroMorphoQuery):
            params["q"] = query.to_q()
            merged_fq.extend(query.to_fq())
        else:
            params["q"] = str(query)
        if fq:
            merged_fq.extend(fq)
        if merged_fq:
            params["fq"] = merged_fq

        payload, url = self._get_json(f"{API_BASE}/neuron/select/", params=params)
        items = tuple(
            NeuroMorphoNeuron.from_payload(item)
            for item in payload.get("_embedded", {}).get("neuronResources", [])
        )
        page_info = payload.get("page", {})
        return NeuroMorphoSearchPage(
            items=items,
            page=int(page_info.get("number", page)),
            size=int(page_info.get("size", size)),
            total_pages=int(page_info.get("totalPages", 0)),
            total_elements=int(page_info.get("totalElements", len(items))),
            query_url=url,
        )

    def iter_search(
        self,
        query: str | NeuroMorphoQuery,
        *,
        fq: list[str] | None = None,
        size: int = 20,
        limit: int | None = None,
        start_page: int = 0,
        sort: str = "neuron_id,asc",
    ) -> Iterator[NeuroMorphoNeuron]:
        """Yield every neuron matching *query*, paging lazily.

        Stops when the upstream reports the last page, when an empty
        page is returned, or when ``limit`` neurons have been yielded.
        Duplicates are filtered out across pages.

        Parameters
        ----------
        query : str or NeuroMorphoQuery
        fq : list of str or None
        size : int
            Page size for each request.
        limit : int or None
            Maximum number of neurons to yield. ``None`` means
            unlimited.
        start_page : int
            Zero-based page index to start from.
        sort : str

        Yields
        ------
        NeuroMorphoNeuron
        """

        seen: set[int] = set()
        yielded = 0
        page = start_page
        while True:
            result = self.search(
                query,
                fq=fq,
                size=size,
                page=page,
                sort=sort,
            )
            if not result.items:
                return
            for neuron in result.items:
                if neuron.neuron_id in seen:
                    continue
                seen.add(neuron.neuron_id)
                yield neuron
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            if result.total_pages and page + 1 >= result.total_pages:
                return
            page += 1

    # ------------------------------------------------------------------
    # Single neuron
    # ------------------------------------------------------------------

    def get_neuron(self, neuron_id: int) -> NeuroMorphoNeuron:
        """Fetch the neuron record for *neuron_id*.

        Parameters
        ----------
        neuron_id : int

        Returns
        -------
        NeuroMorphoNeuron

        Raises
        ------
        NeuroMorphoNotFoundError
            If the upstream returns 404 for the id.
        """

        payload, _ = self._get_json(f"{API_BASE}/neuron/id/{int(neuron_id)}")
        return NeuroMorphoNeuron.from_payload(payload)

    def get_measurement(
        self,
        neuron: NeuroMorphoNeuron | int,
    ) -> NeuroMorphoMeasurement:
        """Fetch the morphometry record for a neuron.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron or int
            Either a previously-fetched neuron or its id.

        Returns
        -------
        NeuroMorphoMeasurement
        """

        if isinstance(neuron, NeuroMorphoNeuron):
            url = build_measurement_url(neuron)
            neuron_id = neuron.neuron_id
        else:
            neuron_id = int(neuron)
            url = f"{API_BASE}/morphometry/id/{neuron_id}"
        payload, _ = self._get_json(url)
        if not isinstance(payload, Mapping):
            raise NeuroMorphoError(
                f"Measurement endpoint {url} did not return a JSON object."
            )
        if "neuron_id" not in payload:
            payload = dict(payload)
            payload["neuron_id"] = neuron_id
        return NeuroMorphoMeasurement.from_payload(payload)

    def get_urls(self, neuron: NeuroMorphoNeuron) -> NeuroMorphoUrls:
        """Return the resolved URLs for *neuron*.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron

        Returns
        -------
        NeuroMorphoUrls
        """

        thumbnail = neuron.png_url
        if thumbnail is not None:
            thumbnail = coerce_https(thumbnail)
        return NeuroMorphoUrls(
            standard_swc=build_standard_swc_url(neuron),
            original_file=build_original_file_url(neuron),
            measurement=build_measurement_url(neuron),
            thumbnail=thumbnail,
        )

    def get_cache_status(
        self,
        neuron: NeuroMorphoNeuron | int,
    ) -> NeuroMorphoCacheStatus:
        """Return on-disk cache status for *neuron*.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron or int

        Returns
        -------
        NeuroMorphoCacheStatus
            ``configured=False`` when this client has no cache attached.
        """

        if self._cache is None:
            return NeuroMorphoCacheStatus(
                configured=False,
                folder=None,
                exists=False,
                metadata_exists=False,
                standard_exists=False,
                original_exists=False,
                neuron_id=(
                    neuron.neuron_id
                    if isinstance(neuron, NeuroMorphoNeuron)
                    else int(neuron)
                ),
            )
        neuron_id = (
            neuron.neuron_id if isinstance(neuron, NeuroMorphoNeuron) else int(neuron)
        )
        status = self._cache.status(neuron_id)
        if isinstance(neuron, NeuroMorphoNeuron):
            # Refine standard/original presence using the actual neuron name.
            folder = self._cache.layout.neuron_dir(neuron_id)
            standard = self._cache.layout.standard_swc_path(neuron_id, neuron.neuron_name)
            standard_exists = standard.exists()
            original_exists = False
            if neuron.original_format:
                suffix = Path(neuron.original_format).suffix
                if suffix:
                    original_exists = self._cache.layout.original_file_path(
                        neuron_id, neuron.neuron_name, suffix
                    ).exists()
            return NeuroMorphoCacheStatus(
                configured=True,
                folder=folder,
                exists=folder.exists(),
                metadata_exists=self._cache.layout.metadata_path(neuron_id).exists(),
                standard_exists=standard_exists,
                original_exists=original_exists,
                neuron_id=neuron_id,
            )
        return status

    def describe(
        self,
        neuron: NeuroMorphoNeuron | int,
        *,
        include_measurement: bool = True,
    ) -> NeuroMorphoDetail:
        """Return an aggregate :class:`NeuroMorphoDetail` for a neuron.

        Combines :meth:`get_neuron` (when an id is passed),
        :meth:`get_measurement`, :meth:`get_urls`, and
        :meth:`get_cache_status` in one call.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron or int
        include_measurement : bool
            If ``False``, the ``measurement`` field of the result is
            ``None`` and no measurement request is issued.

        Returns
        -------
        NeuroMorphoDetail
        """

        resolved = (
            neuron
            if isinstance(neuron, NeuroMorphoNeuron)
            else self.get_neuron(neuron)
        )
        measurement = self.get_measurement(resolved) if include_measurement else None
        urls = self.get_urls(resolved)
        cache_status = self.get_cache_status(resolved)
        return NeuroMorphoDetail(
            neuron=resolved,
            measurement=measurement,
            urls=urls,
            cache_status=cache_status,
        )

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def file_plan(
        self,
        neuron: NeuroMorphoNeuron,
        *,
        mode: DownloadMode = "both",
    ) -> tuple[NeuroMorphoFilePlan, ...]:
        """Build the typed download plan for *neuron*.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron
        mode : {"standard", "original", "both"}

        Returns
        -------
        tuple of NeuroMorphoFilePlan
        """

        return plan_neuron_files(neuron, mode=mode)

    def download(
        self,
        neuron: NeuroMorphoNeuron | int,
        output_dir: str | Path | None = None,
        *,
        mode: DownloadMode = "both",
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> NeuroMorphoDownloadRecord:
        """Download files for *neuron* into the cache layout.

        When ``output_dir`` is ``None`` the client falls back to
        :attr:`cache_dir`. Either ``output_dir`` or a configured cache
        is required.

        Parameters
        ----------
        neuron : NeuroMorphoNeuron or int
        output_dir : str or Path or None
            Cache root. ``None`` uses the client's configured cache.
        mode : {"standard", "original", "both"}
        overwrite : bool
            Re-download files that are already on disk.
        dry_run : bool
            If ``True``, do not touch the network or filesystem. Returns
            a populated :class:`NeuroMorphoDownloadRecord` whose items
            have ``downloaded_now=False`` and ``reason="dry_run"``.

        Returns
        -------
        NeuroMorphoDownloadRecord

        Raises
        ------
        ValueError
            If neither ``output_dir`` nor a client cache is provided.
        """

        if output_dir is None:
            if self._cache is None:
                raise ValueError(
                    "download() requires either an output_dir argument or a "
                    "client constructed with cache_dir=..."
                )
            cache_root = self._cache.root
        else:
            cache_root = Path(output_dir).expanduser()

        resolved = (
            neuron
            if isinstance(neuron, NeuroMorphoNeuron)
            else self.get_neuron(neuron)
        )

        # Fetch measurement first so we fail fast on a network error before
        # touching the filesystem.
        measurement: NeuroMorphoMeasurement | None
        if dry_run:
            measurement = None
        else:
            measurement = self.get_measurement(resolved)

        folder = cache_root / str(resolved.neuron_id)
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)

        plans = plan_neuron_files(resolved, mode=mode)
        items: list[NeuroMorphoDownloadItem] = []
        for plan in plans:
            target = folder / plan.filename
            if plan.skip:
                items.append(
                    NeuroMorphoDownloadItem(
                        kind=plan.kind,
                        url=plan.url,
                        filename=plan.filename,
                        path=target,
                        downloaded_now=False,
                        reason=plan.reason,
                    )
                )
                continue
            if dry_run:
                items.append(
                    NeuroMorphoDownloadItem(
                        kind=plan.kind,
                        url=plan.url,
                        filename=plan.filename,
                        path=target,
                        downloaded_now=False,
                        reason="dry_run",
                    )
                )
                continue
            downloaded_now = self._download_file(plan.url, target, overwrite=overwrite)
            items.append(
                NeuroMorphoDownloadItem(
                    kind=plan.kind,
                    url=plan.url,
                    filename=target.name,
                    path=target,
                    downloaded_now=downloaded_now,
                )
            )

        metadata_path = folder / "metadata.json"
        if not dry_run:
            metadata = self._build_metadata(resolved, items, measurement, mode)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return NeuroMorphoDownloadRecord(
            folder=folder,
            metadata_path=metadata_path,
            download_items=tuple(items),
            measurement=measurement,
            download_mode=mode,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download_file(self, url: str, path: Path, *, overwrite: bool) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return False
        try:
            response = self.session.get(
                url,
                stream=True,
                timeout=max(self.timeout, 60.0),
            )
        except Exception as exc:  # noqa: BLE001
            raise NeuroMorphoHTTPError(
                f"GET {url} failed: {exc}", status=0, url=url
            ) from exc
        with response:
            status = int(getattr(response, "status_code", 0) or 0)
            if status >= 400:
                raise NeuroMorphoHTTPError(
                    f"GET {url} returned HTTP {status}", status=status, url=url
                )
            with path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1 << 15):
                    if chunk:
                        file_obj.write(chunk)
        return True

    @staticmethod
    def _build_metadata(
        neuron: NeuroMorphoNeuron,
        items: list[NeuroMorphoDownloadItem],
        measurement: NeuroMorphoMeasurement | None,
        mode: DownloadMode,
    ) -> dict[str, Any]:
        thumbnail = coerce_https(neuron.png_url) if neuron.png_url else None
        measurement_url = build_measurement_url(neuron)
        return {
            "neuron_id": neuron.neuron_id,
            "neuron_name": neuron.neuron_name,
            "archive": neuron.archive,
            "species": neuron.species,
            "brain_region": neuron.brain_region,
            "cell_type": neuron.cell_type,
            "original_format": neuron.original_format,
            "thumbnail_url": thumbnail,
            "standard_swc_url": build_standard_swc_url(neuron),
            "original_file_url": build_original_file_url(neuron),
            "measurement_url": measurement_url,
            "links": neuron.payload.get("_links", {}),
            "neuron": neuron.payload,
            "measurement": dict(measurement.raw) if measurement is not None else None,
            "download_mode": mode,
            "download_items": [
                {
                    "kind": item.kind,
                    "url": item.url,
                    "filename": item.filename,
                    "path": str(item.path),
                    "downloaded_now": item.downloaded_now,
                    "reason": item.reason,
                }
                for item in items
            ],
        }
