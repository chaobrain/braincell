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

"""``braincell-neuromorpho`` command-line interface.

Subcommands:

``search``
    Search the NeuroMorpho.Org database. Accepts the same typed flags
    as :class:`NeuroMorphoQuery` (``--species``, ``--brain-region``,
    ``--cell-type``, ``--archive``), or the legacy raw flags
    ``--q`` / ``--fq``.
``show``
    Print metadata, URLs, measurement, and cache status for one neuron.
``fetch``
    Download files for one neuron, optionally parsing the result with
    :func:`load_neuromorpho` (``--load``).
``urls``
    Print resolved URLs for one neuron without downloading anything.
``cache list``
    List every neuron currently cached on disk.
``cache info``
    Print the cache status of one neuron.
``cache rm``
    Remove the cache folder of one neuron.
``cache clear``
    Remove every per-neuron folder under the cache root.

Pass ``--json`` to switch any subcommand to JSON output.
"""



import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .cache import NeuroMorphoCache
from .client import DEFAULT_TIMEOUT, NeuroMorphoClient
from .entry import default_cache_dir, load_neuromorpho
from .errors import NeuroMorphoError
from .models import (
    NeuroMorphoCacheStatus,
    NeuroMorphoDetail,
    NeuroMorphoDownloadRecord,
    NeuroMorphoMeasurement,
    NeuroMorphoSearchPage,
)
from .query import NeuroMorphoQuery
from .urls import build_measurement_url

__all__ = ["build_arg_parser", "main"]


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, NeuroMorphoMeasurement):
        return obj.as_dict()
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=_json_default))


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _format_search(page: NeuroMorphoSearchPage) -> str:
    lines: list[str] = []
    lines.append(
        f"page={page.page} size={page.size} total_pages={page.total_pages} "
        f"total_elements={page.total_elements}"
    )
    lines.append(f"query_url={page.query_url}")
    for index, item in enumerate(page.items, start=1):
        brain_region = ",".join(item.brain_region) if item.brain_region else "-"
        lines.append(
            f"[{index}] id={item.neuron_id} name={item.neuron_name} "
            f"archive={item.archive or '-'} brain_region={brain_region} "
            f"original_format={item.original_format or '-'}"
        )
    return "\n".join(lines)


def _format_detail(detail: NeuroMorphoDetail) -> str:
    neuron = detail.neuron
    lines: list[str] = [
        f"id={neuron.neuron_id}",
        f"name={neuron.neuron_name}",
        f"archive={neuron.archive or '-'}",
        f"species={neuron.species or '-'}",
        f"brain_region={','.join(neuron.brain_region) if neuron.brain_region else '-'}",
        f"cell_type={','.join(neuron.cell_type) if neuron.cell_type else '-'}",
        f"original_format={neuron.original_format or '-'}",
        f"thumbnail_url={detail.urls.thumbnail or '-'}",
        f"standard_swc_url={detail.urls.standard_swc}",
        f"original_file_url={detail.urls.original_file or '-'}",
        f"measurement_url={detail.urls.measurement}",
        f"cache_status={json.dumps(asdict(detail.cache_status), default=_json_default, sort_keys=True)}",
    ]
    if detail.measurement is not None:
        lines.append("measurement=")
        lines.append(
            json.dumps(detail.measurement.as_dict(), indent=2, sort_keys=True, default=_json_default)
        )
    return "\n".join(lines)


def _format_download(record: NeuroMorphoDownloadRecord) -> str:
    lines: list[str] = [
        f"folder={record.folder}",
        f"metadata_path={record.metadata_path}",
        f"download_mode={record.download_mode}",
        f"dry_run={record.dry_run}",
    ]
    for item in record.download_items:
        lines.append(
            f"{item.kind}: filename={item.filename} path={item.path} "
            f"downloaded_now={item.downloaded_now} url={item.url or '-'} "
            f"reason={item.reason or '-'}"
        )
    return "\n".join(lines)


def _format_cache_status(status: NeuroMorphoCacheStatus) -> str:
    return (
        f"neuron_id={status.neuron_id} configured={status.configured} "
        f"folder={status.folder or '-'} exists={status.exists} "
        f"metadata_exists={status.metadata_exists} "
        f"standard_exists={status.standard_exists} "
        f"original_exists={status.original_exists}"
    )


def _format_cache_list(cache: NeuroMorphoCache) -> str:
    ids = cache.list_neurons()
    lines = [f"root={cache.root}", f"count={len(ids)}"]
    for neuron_id in ids:
        lines.append(str(neuron_id))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_query_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--species", action="append", default=None)
    parser.add_argument("--brain-region", dest="brain_region", action="append", default=None)
    parser.add_argument("--cell-type", dest="cell_type", action="append", default=None)
    parser.add_argument("--archive", action="append", default=None)
    parser.add_argument("--original-format", dest="original_format", action="append", default=None)
    parser.add_argument("--stain", action="append", default=None)
    parser.add_argument("--age-classification", dest="age_classification", action="append", default=None)
    parser.add_argument("--gender", action="append", default=None)
    parser.add_argument("--q", default=None, help="Raw Solr q string (legacy).")
    parser.add_argument("--fq", action="append", default=None, help="Raw Solr fq string(s).")


def _query_from_args(args: argparse.Namespace) -> tuple[str | NeuroMorphoQuery, list[str] | None]:
    typed_fields = (
        args.species,
        args.brain_region,
        args.cell_type,
        args.archive,
        args.original_format,
        args.stain,
        args.age_classification,
        args.gender,
    )
    typed_set = any(field is not None for field in typed_fields)
    if typed_set:
        query = NeuroMorphoQuery(
            species=tuple(args.species) if args.species else None,
            brain_region=tuple(args.brain_region) if args.brain_region else None,
            cell_type=tuple(args.cell_type) if args.cell_type else None,
            archive=tuple(args.archive) if args.archive else None,
            original_format=tuple(args.original_format) if args.original_format else None,
            stain=tuple(args.stain) if args.stain else None,
            age_classification=tuple(args.age_classification) if args.age_classification else None,
            gender=tuple(args.gender) if args.gender else None,
            raw_q=(args.q,) if args.q else (),
            raw_fq=tuple(args.fq) if args.fq else (),
        )
        return query, None
    raw_q = args.q if args.q is not None else "*:*"
    return raw_q, list(args.fq) if args.fq else None


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the ``braincell-neuromorpho`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(prog="braincell-neuromorpho")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="as_json")

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the NeuroMorpho.Org database.")
    _add_query_flags(search_parser)
    search_parser.add_argument("--size", type=int, default=20)
    search_parser.add_argument("--page", type=int, default=0)
    search_parser.add_argument("--limit", type=int, default=None)
    search_parser.add_argument("--sort", default="neuron_id,asc")

    show_parser = subparsers.add_parser("show", help="Show metadata for one neuron.")
    show_parser.add_argument("--id", type=int, required=True, dest="neuron_id")
    show_parser.add_argument("--no-measurement", action="store_true")

    download_parser = subparsers.add_parser("download", help="Download files for one neuron.")
    download_parser.add_argument("--id", type=int, required=True, dest="neuron_id")
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.add_argument(
        "--mode", choices=("standard", "original", "both"), default="both"
    )
    download_parser.add_argument("--overwrite", action="store_true")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch one neuron into the cache (and optionally parse it).",
    )
    fetch_parser.add_argument("neuron_id", type=int)
    fetch_parser.add_argument(
        "--mode", choices=("standard", "original", "both"), default="standard"
    )
    fetch_parser.add_argument("--overwrite", action="store_true")
    fetch_parser.add_argument(
        "--load",
        action="store_true",
        help="Also parse the standardized SWC and print a one-line summary.",
    )

    urls_parser = subparsers.add_parser("urls", help="Print resolved URLs for one neuron.")
    urls_parser.add_argument("neuron_id", type=int)

    cache_parser = subparsers.add_parser("cache", help="Inspect the on-disk cache.")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)

    cache_sub.add_parser("list", help="List every cached neuron id.")

    cache_info_parser = cache_sub.add_parser(
        "info", help="Print the cache status of one neuron."
    )
    cache_info_parser.add_argument("neuron_id", type=int)

    cache_rm_parser = cache_sub.add_parser(
        "rm", help="Remove the cache folder of one neuron."
    )
    cache_rm_parser.add_argument("neuron_id", type=int)

    cache_clear_parser = cache_sub.add_parser(
        "clear", help="Remove every per-neuron folder under the cache root."
    )
    cache_clear_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the destructive operation.",
    )

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _resolved_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir is not None:
        return args.cache_dir
    return default_cache_dir()


def _make_client(args: argparse.Namespace) -> NeuroMorphoClient:
    return NeuroMorphoClient(
        timeout=args.timeout,
        cache_dir=args.cache_dir,
        retries=args.retries,
    )


def _cmd_search(args: argparse.Namespace) -> int:
    client = _make_client(args)
    query, fq = _query_from_args(args)
    if args.limit is not None:
        neurons = list(
            client.iter_search(
                query,
                fq=fq,
                size=args.size,
                limit=args.limit,
                start_page=args.page,
                sort=args.sort,
            )
        )
        if args.as_json:
            _print_json([asdict(n) for n in neurons])
        else:
            print(f"matched={len(neurons)}")
            for index, neuron in enumerate(neurons, start=1):
                brain_region = ",".join(neuron.brain_region) or "-"
                print(
                    f"[{index}] id={neuron.neuron_id} name={neuron.neuron_name} "
                    f"archive={neuron.archive or '-'} brain_region={brain_region} "
                    f"original_format={neuron.original_format or '-'}"
                )
        return 0
    page = client.search(query, fq=fq, size=args.size, page=args.page, sort=args.sort)
    if args.as_json:
        _print_json(asdict(page))
    else:
        print(_format_search(page))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    client = _make_client(args)
    detail = client.describe(args.neuron_id, include_measurement=not args.no_measurement)
    if args.as_json:
        _print_json(detail)
    else:
        print(_format_detail(detail))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    client = _make_client(args)
    record = client.download(
        args.neuron_id,
        output_dir=args.output_dir,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    if args.as_json:
        _print_json(record)
    else:
        print(_format_download(record))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    cache_root = _resolved_cache_dir(args)
    client = NeuroMorphoClient(
        timeout=args.timeout,
        cache_dir=cache_root,
        retries=args.retries,
    )
    record = client.download(
        args.neuron_id,
        output_dir=cache_root,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    summary: dict[str, Any] = {"record": record}
    if args.load:
        morph = load_neuromorpho(
            args.neuron_id,
            cache_dir=cache_root,
            client=client,
            overwrite=False,
        )
        summary["loaded"] = {
            "n_branches": len(morph.branches),
            "n_points": int(sum(b.n_points for b in morph.branches)),
        }
    if args.as_json:
        _print_json({
            "record": record,
            "loaded": summary.get("loaded"),
        })
    else:
        print(_format_download(record))
        if "loaded" in summary:
            loaded = summary["loaded"]
            print(
                f"loaded OK: {loaded['n_branches']} branches, "
                f"{loaded['n_points']} points"
            )
    return 0


def _cmd_urls(args: argparse.Namespace) -> int:
    client = _make_client(args)
    neuron = client.get_neuron(args.neuron_id)
    urls = client.get_urls(neuron)
    if args.as_json:
        _print_json({
            "neuron_id": neuron.neuron_id,
            "urls": asdict(urls),
            "measurement": build_measurement_url(neuron),
        })
    else:
        print(f"id={neuron.neuron_id}")
        print(f"standard_swc_url={urls.standard_swc}")
        print(f"original_file_url={urls.original_file or '-'}")
        print(f"measurement_url={urls.measurement}")
        print(f"thumbnail_url={urls.thumbnail or '-'}")
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    cache_root = _resolved_cache_dir(args)
    cache = NeuroMorphoCache(cache_root)
    if args.cache_command == "list":
        if args.as_json:
            _print_json({"root": str(cache.root), "neuron_ids": list(cache.list_neurons())})
        else:
            print(_format_cache_list(cache))
        return 0
    if args.cache_command == "info":
        status = cache.status(args.neuron_id)
        if args.as_json:
            _print_json(status)
        else:
            print(_format_cache_status(status))
        return 0
    if args.cache_command == "rm":
        removed = cache.remove(args.neuron_id)
        if args.as_json:
            _print_json({"neuron_id": args.neuron_id, "removed": removed})
        else:
            print(f"neuron_id={args.neuron_id} removed={removed}")
        return 0
    if args.cache_command == "clear":
        if not args.yes:
            print("refusing to clear without --yes")
            return 2
        count = cache.clear()
        if args.as_json:
            _print_json({"root": str(cache.root), "removed": count})
        else:
            print(f"removed {count} neuron folder(s) from {cache.root}")
        return 0
    print(f"unknown cache command: {args.cache_command}")
    return 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_COMMAND_DISPATCH = {
    "search": _cmd_search,
    "show": _cmd_show,
    "download": _cmd_download,
    "fetch": _cmd_fetch,
    "urls": _cmd_urls,
    "cache": _cmd_cache,
}


def main(argv: list[str] | None = None) -> int:
    """Run the ``braincell-neuromorpho`` CLI.

    Parameters
    ----------
    argv : list of str or None
        Arguments to parse; defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    handler = _COMMAND_DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
    try:
        return handler(args)
    except NeuroMorphoError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
