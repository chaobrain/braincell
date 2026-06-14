

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
