#!/usr/bin/env python3
"""CLI helper for the ion_test NEURON vs braincell comparison workflow.

This wrapper keeps the tutorial-facing entrypoint separate from the original
comparison implementation while re-exporting the same core helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_core_module():
    core_path = Path(__file__).with_name("single_ion_compare_template.py")
    spec = importlib.util.spec_from_file_location("single_ion_compare_template", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load comparison core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = _load_core_module()

align_pair = _core.align_pair
DEFAULT_CONFIG = _core.DEFAULT_CONFIG
aggregate_case_metrics = _core.aggregate_case_metrics
build_braincell_model = _core.build_braincell_model
build_neuron_model = _core.build_neuron_model
compare_case = _core.compare_case
compute_error_metrics = _core.compute_error_metrics
expand_cases = _core.expand_cases
load_config = _core.load_config
main = _core.main
parse_args = _core.parse_args
run_case_braincell = _core.run_case_braincell
run_case_neuron = _core.run_case_neuron
run = _core.run


if __name__ == "__main__":
    raise SystemExit(main())
