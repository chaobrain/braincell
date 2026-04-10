#!/usr/bin/env python3


import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import DEFAULT_PIPELINE_NAME
from pipeline import get_step_choices
from pipeline import run_pipeline_from_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    choices = get_step_choices()
    parser = argparse.ArgumentParser(
        description="Render a BrainCell preview for a .mod file; optionally write it to disk."
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
    parser.add_argument("-o", "--output", help="Optional output .py path")
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=80,
        help="Number of rendered lines to include in the JSON preview.",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = run_pipeline_from_path(
        args.input,
        pipeline_name=args.pipeline,
        step1=args.step1,
        step2=args.step2,
        step3=args.step3,
    )
    spec = result["spec"]
    step3_result = result["step3_result"]
    rendered = step3_result["rendered_text"]

    payload = {
        "pipeline": spec.pipeline_name,
        "step1_variant": spec.step1,
        "step2_variant": spec.step2,
        "step3_variant": spec.step3,
        "source_file": step3_result["source_file"],
        "summary": step3_result["summary"],
        "render_preview": rendered.splitlines()[: args.preview_lines],
    }

    if args.output:
        output_file = Path(args.output).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered, encoding="utf-8")
        payload["output_file"] = str(output_file)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
