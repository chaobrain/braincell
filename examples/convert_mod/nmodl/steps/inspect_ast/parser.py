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

from pathlib import Path

from .backend import _bindings
from .backend import node_text
from .backend import lookup


def parse_program(mod_file: Path):
    NmodlDriver, _, _, _, _ = _bindings()
    driver = NmodlDriver()
    try:
        return driver.parse_file(str(mod_file))
    except Exception as exc:
        raise SystemExit(f"Failed to parse {mod_file}: {exc}") from exc


def reconstruct_nmodl(program) -> str:
    return node_text(program)


def collect_block_counts(program) -> list[tuple[str, int]]:
    count_map = [
        ("NEURON_BLOCK", "NEURON_BLOCK"),
        ("PARAM_BLOCK", "PARAM_BLOCK"),
        ("STATE_BLOCK", "STATE_BLOCK"),
        ("ASSIGNED_BLOCK", "ASSIGNED_BLOCK"),
        ("INITIAL_BLOCK", "INITIAL_BLOCK"),
        ("BREAKPOINT_BLOCK", "BREAKPOINT_BLOCK"),
        ("DERIVATIVE_BLOCK", "DERIVATIVE_BLOCK"),
        ("FUNCTION_BLOCK", "FUNCTION_BLOCK"),
        ("PROCEDURE_BLOCK", "PROCEDURE_BLOCK"),
    ]
    return [(label, len(lookup(program, node_type))) for label, node_type in count_map]
