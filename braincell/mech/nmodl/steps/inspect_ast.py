#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import DEFAULT_PIPELINE_NAME
from pipeline import get_step_choices
from pipeline import resolve_pipeline_spec
from pipeline import resolve_mod_file
from pipeline.inspect_ast import run as run_inspect_ast


def parse_args(argv: list[str]) -> argparse.Namespace:
    choices = get_step_choices()
    parser = argparse.ArgumentParser(
        description="Inspect a .mod file as AST, RawBlocks, and CanonicalBlocks."
    )
    parser.add_argument("input", nargs="?", help="Path to the input .mod file")
    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE_NAME,
        choices=choices["pipeline"],
        help="Named pipeline combination to use.",
    )
    parser.add_argument(
        "--step1",
        choices=choices["step1"],
        help="Override the Step 1 implementation. Must match the selected pipeline.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    mod_file = resolve_mod_file(args.input)
    spec = resolve_pipeline_spec(pipeline_name=args.pipeline, step1=args.step1)
    step1_result = run_inspect_ast(mod_file, variant=spec.step1)
    payload = {
        "pipeline": spec.pipeline_name,
        "step1_variant": spec.step1,
        "source_file": step1_result["source_file"],
        "ast_root_type": step1_result["ast_root_type"],
        "block_counts": step1_result["block_counts"],
        "reconstructed_nmodl": step1_result["reconstructed_nmodl"],
        "raw_blocks": step1_result["raw_blocks"],
        "canonical_blocks": step1_result["canonical_blocks"],
        "ast_json": step1_result["ast_json"],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
