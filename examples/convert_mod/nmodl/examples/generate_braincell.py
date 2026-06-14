#!/usr/bin/env python3


import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steps import DEFAULT_PIPELINE_NAME
from steps import get_step_choices
from steps import resolve_mod_file
from steps import run_pipeline


def default_output_file(mod_file: Path) -> Path:
    return Path(__file__).resolve().with_name(f"generated_{mod_file.stem}_one_ion_hh_ohmic.py")


def parse_args(argv: list[str]) -> argparse.Namespace:
    choices = get_step_choices()
    parser = argparse.ArgumentParser(
        description="Generate a BrainCell-style Python channel from a NMODL file."
    )
    parser.add_argument("input", nargs="?", help="Path to the input .mod file")
    parser.add_argument(
        "--pipeline",
        default=DEFAULT_PIPELINE_NAME,
        choices=choices["pipeline"],
        help="Named pipeline combination to use.",
    )
    parser.add_argument("--step1", choices=choices["step1"], help="Override the Step 1 implementation.")
    parser.add_argument("--step2", choices=choices["step2"], help="Override the Step 2 implementation.")
    parser.add_argument("--step3", choices=choices["step3"], help="Override the Step 3 implementation.")
    parser.add_argument("-o", "--output", help="Path to the output .py file")
    parser.add_argument(
        "--print-spec",
        action="store_true",
        help="Print the Step 2 IR JSON after generation.",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=60,
        help="Number of rendered lines to include in the JSON preview.",
    )
    return parser.parse_args(argv[1:])


def resolve_output_file(mod_file: Path, output_path: str | None) -> Path:
    output = Path(output_path).expanduser().resolve() if output_path else default_output_file(mod_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    mod_file = resolve_mod_file(args.input)
    output_file = resolve_output_file(mod_file, args.output)
    result = run_pipeline(
        mod_file,
        pipeline_name=args.pipeline,
        step1=args.step1,
        step2=args.step2,
        step3=args.step3,
    )
    spec = result["spec"]
    step2_result = result["step2_result"]
    step3_result = result["step3_result"]
    rendered = step3_result["rendered_text"]
    output_file.write_text(rendered, encoding="utf-8")

    payload = {
        "pipeline": spec.pipeline_name,
        "step1_variant": spec.step1,
        "step2_variant": spec.step2,
        "step3_variant": spec.step3,
        "source_file": step3_result["source_file"],
        "output_file": str(output_file),
        "summary": step3_result["summary"],
        "preview": rendered.splitlines()[: args.preview_lines],
    }
    print(json.dumps(payload, indent=2))
    if args.print_spec:
        print(json.dumps(step2_result["ir"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
