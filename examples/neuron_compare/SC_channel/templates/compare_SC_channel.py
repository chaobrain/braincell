#!/usr/bin/env python3
"""Dispatcher for single-compartment channel compare inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .hh_fixed_ion.compare_single_case import (
        compare_case as compare_hh_fixed_ion_case,
        load_case as load_hh_fixed_ion_case,
    )
    from .hh_fixed_ion.sweep_config import load_config as load_hh_fixed_ion_config
    from .hh_fixed_ion.sweep_driver import run_config as run_hh_fixed_ion_sweep
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from hh_fixed_ion.compare_single_case import (  # type: ignore
        compare_case as compare_hh_fixed_ion_case,
        load_case as load_hh_fixed_ion_case,
    )
    from hh_fixed_ion.sweep_config import load_config as load_hh_fixed_ion_config  # type: ignore
    from hh_fixed_ion.sweep_driver import run_config as run_hh_fixed_ion_sweep  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dispatch single-compartment channel compare inputs.",
    )
    parser.add_argument("input_path", help="Path to a single-case JSON or sweep-config JSON.")
    parser.add_argument("--output", help="Optional JSON output path for single-case compare.")
    parser.add_argument("--out-dir", help="Optional output directory for sweep runs.")
    parser.add_argument("--expand-only", action="store_true", help="Only expand a sweep config without running compare.")
    parser.set_defaults(plot=None)
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-case comparison plots for sweeps.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation for sweeps.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    input_path = Path(args.input_path)
    payload = json.loads(input_path.read_text())

    if "case_groups" in payload:
        if args.output:
            raise ValueError("--output is only valid for single-case inputs; use --out-dir for sweeps.")
        config = load_hh_fixed_ion_config(input_path)
        return run_hh_fixed_ion_sweep(
            config,
            out_dir=args.out_dir,
            expand_only=bool(args.expand_only),
            plot=args.plot,
        )

    if args.expand_only:
        raise ValueError("--expand-only requires a sweep config input.")
    if payload.get("template_variant", "hh_fixed_ion") != "hh_fixed_ion":
        raise NotImplementedError(
            f"Unsupported single_compartment_channel template_variant {payload.get('template_variant')!r}."
        )
    case = load_hh_fixed_ion_case(input_path)
    result = compare_hh_fixed_ion_case(case)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
