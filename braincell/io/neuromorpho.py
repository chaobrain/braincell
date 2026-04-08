from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import requests

API_BASE = "https://neuromorpho.org/api"
FILE_BASE = "https://neuromorpho.org/dableFiles"
DEFAULT_TIMEOUT = 30.0
DownloadMode = Literal["standard", "original", "both"]


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return cleaned.strip("._") or "neuromorpho_neuron"


def _coerce_url(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def load_cached_metadata(folder: str | Path) -> dict[str, Any]:
    return json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8"))


def find_standard_swc(folder: str | Path, metadata: dict[str, Any]) -> Path | None:
    folder_path = Path(folder)
    for item in metadata.get("download_items", []):
        if item.get("kind") != "standard":
            continue
        raw_path = item.get("path")
        if raw_path:
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate
            candidate = folder_path / candidate.name
            if candidate.exists():
                return candidate
        filename = item.get("filename")
        if filename:
            candidate = folder_path / str(filename)
            if candidate.exists():
                return candidate
    swc_paths = sorted(folder_path.glob("*.swc"))
    return swc_paths[0] if swc_paths else None


@dataclass(frozen=True)
class NeuroMorphoNeuron:
    neuron_id: int
    neuron_name: str
    archive: str | None
    species: str | None
    brain_region: list[str]
    cell_type: list[str]
    original_format: str | None
    png_url: str | None
    payload: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NeuroMorphoNeuron":
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
            brain_region=list(brain_region),
            cell_type=list(cell_type),
            original_format=payload.get("original_format"),
            png_url=payload.get("png_url"),
            payload=dict(payload),
        )


@dataclass(frozen=True)
class NeuroMorphoSearchPage:
    items: tuple[NeuroMorphoNeuron, ...]
    page: int
    size: int
    total_pages: int
    total_elements: int
    query_url: str


@dataclass(frozen=True)
class NeuroMorphoDetail:
    neuron: NeuroMorphoNeuron
    measurement: dict[str, Any] | None
    thumbnail_url: str | None
    standard_swc_url: str
    original_file_url: str | None
    cache_status: dict[str, Any]


@dataclass(frozen=True)
class NeuroMorphoDownloadItem:
    kind: str
    url: str
    filename: str
    path: Path
    downloaded_now: bool
    reason: str | None = None


@dataclass(frozen=True)
class NeuroMorphoDownloadRecord:
    folder: Path
    metadata_path: Path
    download_items: tuple[NeuroMorphoDownloadItem, ...]
    measurement: dict[str, Any] | None
    download_mode: DownloadMode


class NeuroMorphoClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def search(
        self,
        q: str,
        fq: list[str] | None = None,
        size: int = 20,
        page: int = 0,
        sort: str = "neuron_id,asc",
    ) -> NeuroMorphoSearchPage:
        params: dict[str, Any] = {"q": q, "size": size, "page": page, "sort": sort}
        if fq:
            params["fq"] = list(fq)
        response = self.session.get(
            f"{API_BASE}/neuron/select/",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
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
            query_url=str(response.url),
        )

    def search_batch(
        self,
        q: str,
        fq: list[str] | None = None,
        size: int = 20,
        page_start: int = 0,
        max_pages: int = 1,
        sort: str = "neuron_id,asc",
    ) -> tuple[tuple[NeuroMorphoNeuron, ...], tuple[str, ...]]:
        items: list[NeuroMorphoNeuron] = []
        query_urls: list[str] = []
        seen_ids: set[int] = set()
        for page in range(page_start, page_start + max_pages):
            result = self.search(q=q, fq=fq, size=size, page=page, sort=sort)
            query_urls.append(result.query_url)
            if not result.items:
                break
            for neuron in result.items:
                if neuron.neuron_id in seen_ids:
                    continue
                seen_ids.add(neuron.neuron_id)
                items.append(neuron)
            if result.total_pages and page + 1 >= result.total_pages:
                break
        return tuple(items), tuple(query_urls)

    def get_neuron(self, neuron_id: int) -> NeuroMorphoNeuron:
        response = self.session.get(
            f"{API_BASE}/neuron/id/{int(neuron_id)}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return NeuroMorphoNeuron.from_payload(response.json())

    def get_measurement(self, neuron: NeuroMorphoNeuron | int) -> dict[str, Any]:
        if isinstance(neuron, NeuroMorphoNeuron):
            link = neuron.payload.get("_links", {}).get("measurements", {}).get("href")
            if link:
                url = _coerce_url(str(link))
            else:
                url = f"{API_BASE}/morphometry/id/{neuron.neuron_id}"
        else:
            url = f"{API_BASE}/morphometry/id/{int(neuron)}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())

    def describe(
        self,
        neuron: NeuroMorphoNeuron | int,
        include_measurement: bool = True,
    ) -> NeuroMorphoDetail:
        resolved = neuron if isinstance(neuron, NeuroMorphoNeuron) else self.get_neuron(neuron)
        measurement = self.get_measurement(resolved) if include_measurement else None
        standard_url = self.standardized_swc_url(resolved)
        original_url = self.original_file_url(resolved)
        folder = self.neuron_cache_dir(resolved) if self.cache_dir is not None else None
        cache_status = self._cache_status(folder, resolved)
        thumbnail_url = resolved.png_url
        if thumbnail_url is not None:
            thumbnail_url = _coerce_url(thumbnail_url)
        return NeuroMorphoDetail(
            neuron=resolved,
            measurement=measurement,
            thumbnail_url=thumbnail_url,
            standard_swc_url=standard_url,
            original_file_url=original_url,
            cache_status=cache_status,
        )

    def download(
        self,
        neuron: NeuroMorphoNeuron | int,
        output_dir: str | Path,
        mode: DownloadMode = "both",
        overwrite: bool = False,
    ) -> NeuroMorphoDownloadRecord:
        resolved = neuron if isinstance(neuron, NeuroMorphoNeuron) else self.get_neuron(neuron)
        measurement = self.get_measurement(resolved)
        folder = Path(output_dir) / str(resolved.neuron_id)
        folder.mkdir(parents=True, exist_ok=True)

        items: list[NeuroMorphoDownloadItem] = []
        for plan in self.file_plan(resolved, mode=mode):
            if plan.get("skip"):
                items.append(
                    NeuroMorphoDownloadItem(
                        kind=str(plan["kind"]),
                        url=str(plan["url"]),
                        filename=str(plan["filename"]),
                        path=folder / str(plan["filename"]),
                        downloaded_now=False,
                        reason=str(plan["reason"]),
                    )
                )
                continue
            target = folder / str(plan["filename"])
            downloaded_now = self._download_file(str(plan["url"]), target, overwrite=overwrite)
            items.append(
                NeuroMorphoDownloadItem(
                    kind=str(plan["kind"]),
                    url=str(plan["url"]),
                    filename=target.name,
                    path=target,
                    downloaded_now=downloaded_now,
                )
            )

        metadata = {
            "neuron_id": resolved.neuron_id,
            "neuron_name": resolved.neuron_name,
            "archive": resolved.archive,
            "species": resolved.species,
            "brain_region": resolved.brain_region,
            "cell_type": resolved.cell_type,
            "original_format": resolved.original_format,
            "thumbnail_url": _coerce_url(resolved.png_url) if resolved.png_url else None,
            "standard_swc_url": self.standardized_swc_url(resolved),
            "original_file_url": self.original_file_url(resolved),
            "measurement_url": _coerce_url(
                str(resolved.payload.get("_links", {}).get("measurements", {}).get("href", ""))
            )
            or None,
            "links": resolved.payload.get("_links", {}),
            "neuron": resolved.payload,
            "measurement": measurement,
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
        metadata_path = folder / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        return NeuroMorphoDownloadRecord(
            folder=folder,
            metadata_path=metadata_path,
            download_items=tuple(items),
            measurement=measurement,
            download_mode=mode,
        )

    def neuron_cache_dir(self, neuron: NeuroMorphoNeuron | int) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured for this NeuroMorphoClient.")
        neuron_id = neuron.neuron_id if isinstance(neuron, NeuroMorphoNeuron) else int(neuron)
        return self.cache_dir / str(neuron_id)

    def standardized_swc_url(self, neuron: NeuroMorphoNeuron) -> str:
        if not neuron.archive:
            raise ValueError(f"neuron_id={neuron.neuron_id} is missing archive metadata.")
        archive = quote(neuron.archive.lower(), safe="")
        neuron_name = quote(neuron.neuron_name, safe="")
        return f"{FILE_BASE}/{archive}/CNG%20version/{neuron_name}.CNG.swc"

    def original_file_extension(self, neuron: NeuroMorphoNeuron) -> str:
        original_format = neuron.original_format
        if not original_format:
            raise ValueError("This neuron does not expose an original_format field.")
        suffix = Path(original_format).suffix
        if not suffix:
            raise ValueError(
                f"Cannot infer original file extension from original_format={original_format!r}."
            )
        return suffix

    def original_file_url(self, neuron: NeuroMorphoNeuron) -> str | None:
        if not neuron.archive:
            return None
        try:
            suffix = self.original_file_extension(neuron)
        except ValueError:
            return None
        archive = quote(neuron.archive.lower(), safe="")
        filename = quote(f"{neuron.neuron_name}{suffix}", safe="")
        return f"{FILE_BASE}/{archive}/Source-Version/{filename}"

    def file_plan(self, neuron: NeuroMorphoNeuron, mode: DownloadMode = "both") -> list[dict[str, Any]]:
        if mode not in {"standard", "original", "both"}:
            raise ValueError("mode must be one of: 'standard', 'original', 'both'.")
        stem = _safe_filename(neuron.neuron_name)
        plans: list[dict[str, Any]] = []
        if mode in {"standard", "both"}:
            plans.append(
                {
                    "kind": "standard",
                    "url": self.standardized_swc_url(neuron),
                    "filename": f"{stem}.CNG.swc",
                }
            )
        if mode in {"original", "both"}:
            try:
                suffix = self.original_file_extension(neuron)
            except ValueError as exc:
                plans.append(
                    {
                        "kind": "original",
                        "url": "",
                        "filename": stem,
                        "skip": True,
                        "reason": str(exc),
                    }
                )
            else:
                url = self.original_file_url(neuron)
                assert url is not None
                plans.append(
                    {
                        "kind": "original",
                        "url": url,
                        "filename": f"{stem}{suffix}",
                    }
                )
        return plans

    def _download_file(self, url: str, path: Path, overwrite: bool = False) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            return False
        with self.session.get(url, stream=True, timeout=max(self.timeout, 60.0)) as response:
            response.raise_for_status()
            with path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1 << 15):
                    if chunk:
                        file_obj.write(chunk)
        return True

    def _cache_status(self, folder: Path | None, neuron: NeuroMorphoNeuron) -> dict[str, Any]:
        if folder is None:
            return {"configured": False}
        metadata_path = folder / "metadata.json"
        status = {
            "configured": True,
            "folder": str(folder),
            "exists": folder.exists(),
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata_path.exists(),
            "standard_exists": False,
            "original_exists": False,
        }
        if not folder.exists():
            return status
        standard_name = f"{_safe_filename(neuron.neuron_name)}.CNG.swc"
        status["standard_exists"] = (folder / standard_name).exists()
        try:
            suffix = self.original_file_extension(neuron)
        except ValueError:
            return status
        status["original_exists"] = (folder / f"{_safe_filename(neuron.neuron_name)}{suffix}").exists()
        return status


def _print_search(page: NeuroMorphoSearchPage) -> None:
    print(
        f"page={page.page} size={page.size} total_pages={page.total_pages} total_elements={page.total_elements}"
    )
    print(f"query_url={page.query_url}")
    for index, item in enumerate(page.items, start=1):
        brain_region = ",".join(item.brain_region) if item.brain_region else "-"
        print(
            f"[{index}] id={item.neuron_id} name={item.neuron_name} "
            f"archive={item.archive or '-'} brain_region={brain_region} "
            f"original_format={item.original_format or '-'}"
        )


def _print_detail(detail: NeuroMorphoDetail) -> None:
    neuron = detail.neuron
    print(f"id={neuron.neuron_id}")
    print(f"name={neuron.neuron_name}")
    print(f"archive={neuron.archive or '-'}")
    print(f"species={neuron.species or '-'}")
    print(f"brain_region={','.join(neuron.brain_region) if neuron.brain_region else '-'}")
    print(f"cell_type={','.join(neuron.cell_type) if neuron.cell_type else '-'}")
    print(f"original_format={neuron.original_format or '-'}")
    print(f"thumbnail_url={detail.thumbnail_url or '-'}")
    print(f"standard_swc_url={detail.standard_swc_url}")
    print(f"original_file_url={detail.original_file_url or '-'}")
    print(f"cache_status={json.dumps(detail.cache_status, sort_keys=True)}")
    if detail.measurement is not None:
        print("measurement=")
        print(json.dumps(detail.measurement, indent=2, sort_keys=True))


def _print_download(record: NeuroMorphoDownloadRecord) -> None:
    print(f"folder={record.folder}")
    print(f"metadata_path={record.metadata_path}")
    print(f"download_mode={record.download_mode}")
    for item in record.download_items:
        print(
            f"{item.kind}: filename={item.filename} path={item.path} "
            f"downloaded_now={item.downloaded_now} url={item.url or '-'} "
            f"reason={item.reason or '-'}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braincell-neuromorpho")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--q", required=True)
    search_parser.add_argument("--fq", action="append", default=None)
    search_parser.add_argument("--size", type=int, default=20)
    search_parser.add_argument("--page", type=int, default=0)
    search_parser.add_argument("--sort", default="neuron_id,asc")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--id", type=int, required=True)
    show_parser.add_argument("--no-measurement", action="store_true")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--id", type=int, required=True)
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.add_argument("--mode", choices=("standard", "original", "both"), default="both")
    download_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    client = NeuroMorphoClient(timeout=args.timeout, cache_dir=args.cache_dir)
    if args.command == "search":
        page = client.search(q=args.q, fq=args.fq, size=args.size, page=args.page, sort=args.sort)
        _print_search(page)
        return 0
    if args.command == "show":
        detail = client.describe(args.id, include_measurement=not args.no_measurement)
        _print_detail(detail)
        return 0
    if args.command == "download":
        record = client.download(args.id, output_dir=args.output_dir, mode=args.mode, overwrite=args.overwrite)
        _print_download(record)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
