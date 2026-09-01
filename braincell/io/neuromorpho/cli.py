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
    Download files for one neuron, optionally parsing the file it just
    wrote with :meth:`Morphology.from_swc` (``--load``).
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

from braincell.morph.morphology import Morphology
from .cache import NeuroMorphoCache
from .client import DEFAULT_TIMEOUT, NeuroMorphoClient
from .entry import default_cache_dir
from .errors import NeuroMorphoError
from .models import (
    NeuroMorphoCacheStatus,
    NeuroMorphoDetail,
    NeuroMorphoDownloadRecord,
    NeuroMorphoMeasurement,
    NeuroMorphoNeuron,
    NeuroMorphoSearchPage,
)
from .query import QUERY_FIELDS, NeuroMorphoQuery

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


def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    """Print *payload* as JSON under ``--json``, otherwise print *text*."""

    if args.as_json:
        _print_json(payload)
    else:
        print(text)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _format_neuron_line(index: int, neuron: NeuroMorphoNeuron) -> str:
    brain_region = ",".join(neuron.brain_region) if neuron.brain_region else "-"
    return (
        f"[{index}] id={neuron.neuron_id} name={neuron.neuron_name} "
        f"archive={neuron.archive or '-'} brain_region={brain_region} "
        f"original_format={neuron.original_format or '-'}"
    )


def _format_search(page: NeuroMorphoSearchPage) -> str:
    lines: list[str] = [
        f"page={page.page} size={page.size} total_pages={page.total_pages} total_elements={page.total_elements}",
        f"query_url={page.query_url}",
    ]
    lines.extend(_format_neuron_line(index, item) for index, item in enumerate(page.items, start=1))
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
        lines.append(json.dumps(detail.measurement.as_dict(), indent=2, sort_keys=True, default=_json_default))
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
    for name in QUERY_FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, action="append", default=None)
    parser.add_argument("--q", default=None, help="Raw Solr q string (legacy).")
    parser.add_argument("--fq", action="append", default=None, help="Raw Solr fq string(s).")


def _query_from_args(args: argparse.Namespace) -> tuple[str | NeuroMorphoQuery, list[str] | None]:
    typed = {name: getattr(args, name) for name in QUERY_FIELDS}
    if any(value is not None for value in typed.values()):
        query = NeuroMorphoQuery(
            **{name: tuple(value) if value else None for name, value in typed.items()},
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
    download_parser.add_argument("--mode", choices=("standard", "original", "both"), default="both")
    download_parser.add_argument("--overwrite", action="store_true")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch one neuron into the cache (and optionally parse it).",
    )
    fetch_parser.add_argument("neuron_id", type=int)
    fetch_parser.add_argument("--mode", choices=("standard", "original", "both"), default="standard")
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

    cache_info_parser = cache_sub.add_parser("info", help="Print the cache status of one neuron.")
    cache_info_parser.add_argument("neuron_id", type=int)

    cache_rm_parser = cache_sub.add_parser("rm", help="Remove the cache folder of one neuron.")
    cache_rm_parser.add_argument("neuron_id", type=int)

    cache_clear_parser = cache_sub.add_parser("clear", help="Remove every per-neuron folder under the cache root.")
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
    return args.cache_dir if args.cache_dir is not None else default_cache_dir()


def _make_client(args: argparse.Namespace) -> NeuroMorphoClient:
    # Resolve the default here too, so ``show``/``urls`` report cache status
    # against the same directory ``fetch``/``cache`` read and write.
    return NeuroMorphoClient(
        timeout=args.timeout,
        cache_dir=_resolved_cache_dir(args),
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
        lines = [f"matched={len(neurons)}"]
        lines.extend(_format_neuron_line(index, neuron) for index, neuron in enumerate(neurons, start=1))
        _emit(args, [asdict(n) for n in neurons], "\n".join(lines))
        return 0
    page = client.search(query, fq=fq, size=args.size, page=args.page, sort=args.sort)
    _emit(args, asdict(page), _format_search(page))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    client = _make_client(args)
    detail = client.describe(args.neuron_id, include_measurement=not args.no_measurement)
    _emit(args, detail, _format_detail(detail))
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    client = _make_client(args)
    record = client.download(
        args.neuron_id,
        output_dir=args.output_dir,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    _emit(args, record, _format_download(record))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    cache_root = _resolved_cache_dir(args)
    client = _make_client(args)
    record = client.download(
        args.neuron_id,
        output_dir=cache_root,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    loaded: dict[str, int] | None = None
    text = _format_download(record)
    if args.load:
        # Parse the file this command just downloaded. Calling
        # load_neuromorpho() here would run the whole download path a second
        # time for the same neuron.
        swc_path = next(
            (item.path for item in record.download_items if item.kind == "standard" and item.path.exists()),
            None,
        )
        if swc_path is None:
            raise FileNotFoundError(
                f"Standardized SWC file for neuron_id={args.neuron_id} could not be "
                f"located after download (cache_dir={cache_root})."
            )
        morph = Morphology.from_swc(swc_path, mode="neuromorpho")
        loaded = {
            "n_branches": len(morph.branches),
            # ``Branch`` has no ``n_points``; its point array is one longer
            # than its segment count. Reading the missing attribute made
            # ``fetch --load`` raise AttributeError on every run.
            "n_points": int(sum(b.n_segments + 1 for b in morph.branches)),
        }
        text += f"\nloaded OK: {loaded['n_branches']} branches, {loaded['n_points']} points"
    _emit(args, {"record": record, "loaded": loaded}, text)
    return 0


def _cmd_urls(args: argparse.Namespace) -> int:
    client = _make_client(args)
    neuron = client.get_neuron(args.neuron_id)
    urls = client.get_urls(neuron)
    _emit(
        args,
        {"neuron_id": neuron.neuron_id, "urls": asdict(urls), "measurement": urls.measurement},
        "\n".join(
            (
                f"id={neuron.neuron_id}",
                f"standard_swc_url={urls.standard_swc}",
                f"original_file_url={urls.original_file or '-'}",
                f"measurement_url={urls.measurement}",
                f"thumbnail_url={urls.thumbnail or '-'}",
            )
        ),
    )
    return 0


def _cache_list(args: argparse.Namespace, cache: NeuroMorphoCache) -> int:
    _emit(
        args,
        {"root": str(cache.root), "neuron_ids": list(cache.list_neurons())},
        _format_cache_list(cache),
    )
    return 0


def _cache_info(args: argparse.Namespace, cache: NeuroMorphoCache) -> int:
    status = cache.status(args.neuron_id)
    _emit(args, status, _format_cache_status(status))
    return 0


def _cache_rm(args: argparse.Namespace, cache: NeuroMorphoCache) -> int:
    removed = cache.remove(args.neuron_id)
    _emit(
        args,
        {"neuron_id": args.neuron_id, "removed": removed},
        f"neuron_id={args.neuron_id} removed={removed}",
    )
    return 0


def _cache_clear(args: argparse.Namespace, cache: NeuroMorphoCache) -> int:
    if not args.yes:
        print("refusing to clear without --yes")
        return 2
    count = cache.clear()
    _emit(
        args,
        {"root": str(cache.root), "removed": count},
        f"removed {count} neuron folder(s) from {cache.root}",
    )
    return 0


_CACHE_DISPATCH = {
    "list": _cache_list,
    "info": _cache_info,
    "rm": _cache_rm,
    "clear": _cache_clear,
}


def _cmd_cache(args: argparse.Namespace) -> int:
    cache = NeuroMorphoCache(_resolved_cache_dir(args))
    return _CACHE_DISPATCH[args.cache_command](args, cache)


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
