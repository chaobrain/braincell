from __future__ import annotations

from pathlib import Path

from . import inspect_ast
from . import inspect_ir
from . import render
from .registry import DEFAULT_PIPELINE_NAME
from .registry import PipelineSpec
from .registry import resolve_pipeline_spec


def run_pipeline(
    mod_file: Path,
    *,
    pipeline_name: str | None = DEFAULT_PIPELINE_NAME,
    step1: str | None = None,
    step2: str | None = None,
    step3: str | None = None,
) -> dict:
    spec = resolve_pipeline_spec(
        pipeline_name=pipeline_name,
        step1=step1,
        step2=step2,
        step3=step3,
    )
    step1_result = inspect_ast.run(mod_file, variant=spec.step1)
    step2_result = inspect_ir.run(step1_result, variant=spec.step2)
    step3_result = render.run(step2_result, variant=spec.step3)
    return {
        "spec": spec,
        "step1_result": step1_result,
        "step2_result": step2_result,
        "step3_result": step3_result,
    }


def run_pipeline_from_path(
    input_path: str | None,
    *,
    pipeline_name: str | None = DEFAULT_PIPELINE_NAME,
    step1: str | None = None,
    step2: str | None = None,
    step3: str | None = None,
) -> dict:
    mod_file = inspect_ast.resolve_mod_file(input_path)
    return run_pipeline(
        mod_file,
        pipeline_name=pipeline_name,
        step1=step1,
        step2=step2,
        step3=step3,
    )
