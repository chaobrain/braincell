from __future__ import annotations

from .backend import lookup
from .backend import node_text

BLOCK_ORDER = [
    "TITLE",
    "COMMENT",
    "NEURON",
    "UNITS",
    "PARAMETER",
    "ASSIGNED",
    "STATE",
    "INITIAL",
    "BREAKPOINT",
    "DERIVATIVE",
    "FUNCTION",
    "PROCEDURE",
    "KINETIC",
    "DISCRETE",
    "NET_RECEIVE",
    "BEFORE",
    "AFTER",
    "INDEPENDENT",
    "CONSTANT",
    "LINEAR",
    "NONLINEAR",
    "FUNCTION_TABLE",
    "CVODE",
    "LONGITUDINAL_DIFFUSION",
]

RAW_BLOCK_NODE_MAP = {
    "TITLE": ["MODEL"],
    "COMMENT": ["BLOCK_COMMENT", "LINE_COMMENT"],
    "NEURON": ["NEURON_BLOCK"],
    "UNITS": ["UNIT_BLOCK"],
    "PARAMETER": ["PARAM_BLOCK"],
    "ASSIGNED": ["ASSIGNED_BLOCK"],
    "STATE": ["STATE_BLOCK"],
    "INITIAL": ["INITIAL_BLOCK"],
    "BREAKPOINT": ["BREAKPOINT_BLOCK"],
    "DERIVATIVE": ["DERIVATIVE_BLOCK"],
    "FUNCTION": ["FUNCTION_BLOCK"],
    "PROCEDURE": ["PROCEDURE_BLOCK"],
    "KINETIC": ["KINETIC_BLOCK"],
    "DISCRETE": ["DISCRETE_BLOCK"],
    "NET_RECEIVE": ["NET_RECEIVE_BLOCK"],
    "BEFORE": ["BEFORE_BLOCK"],
    "AFTER": ["AFTER_BLOCK"],
    "INDEPENDENT": ["INDEPENDENT_BLOCK"],
    "CONSTANT": ["CONSTANT_BLOCK"],
    "LINEAR": ["LINEAR_BLOCK"],
    "NONLINEAR": ["NON_LINEAR_BLOCK"],
    "FUNCTION_TABLE": ["FUNCTION_TABLE_BLOCK"],
    "CVODE": ["CVODE_BLOCK"],
    "LONGITUDINAL_DIFFUSION": ["LONGITUDINAL_DIFFUSION_BLOCK"],
}


def extract_raw_blocks(program) -> dict[str, list[str]]:
    raw_blocks = {block_name: [] for block_name in BLOCK_ORDER}
    for block_name, node_type_names in RAW_BLOCK_NODE_MAP.items():
        block_texts: list[str] = []
        for node_type_name in node_type_names:
            block_texts.extend(node_text(node) for node in lookup(program, node_type_name))
        raw_blocks[block_name] = block_texts
    return raw_blocks
