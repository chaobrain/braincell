from __future__ import annotations

import re
from typing import Any

from .backend import lookup
from .backend import node_text
from .raw_blocks import BLOCK_ORDER

HANDLED_CANONICAL_BLOCKS = {
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
}


def strip_unit(text: str | None) -> str | None:
    if text is None:
        return None
    value = text.strip()
    if value.startswith("(") and value.endswith(")"):
        return value[1:-1]
    return value


def maybe_number(text: str | None) -> int | float | str | None:
    if text is None:
        return None
    value = text.strip()
    if not value:
        return value
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)
    if re.fullmatch(r"[+-]?\d+(?:[eE][+-]?\d+)", value):
        return float(value)
    return value


def pascal_case(name: str) -> str:
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", name) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Mechanism"


def _normalize_params(parameters) -> list[dict[str, Any]]:
    normalized = []
    for parameter in parameters:
        normalized.append(
            {
                "name": parameter.get_node_name(),
                "unit": strip_unit(node_text(parameter.unit)) if getattr(parameter, "unit", None) else None,
            }
        )
    return normalized


def _normalize_expression_item(expression) -> dict[str, Any]:
    if expression.is_solve_block():
        return {
            "kind": "solve",
            "target": str(expression.block_name),
            "method": str(expression.method) if getattr(expression, "method", None) else None,
            "code": node_text(expression),
        }

    if expression.is_wrapped_expression():
        inner = expression.expression
        if inner.is_function_call():
            return {
                "kind": "call",
                "name": inner.get_node_name(),
                "args": [node_text(argument) for argument in inner.arguments],
                "code": node_text(expression),
            }
        return {"kind": "raw_expression", "code": node_text(expression)}

    if expression.is_binary_expression():
        if str(expression.op) == "=":
            return {
                "kind": "assignment",
                "assigned_var": node_text(expression.lhs),
                "expression": node_text(expression.rhs),
                "code": node_text(expression),
            }
        return {"kind": "binary_expression", "code": node_text(expression)}

    if expression.is_diff_eq_expression():
        inner = expression.expression
        if inner.is_binary_expression() and str(inner.op) == "=":
            prime_var = node_text(inner.lhs)
            return {
                "kind": "derivative_assignment",
                "assigned_var": prime_var.removesuffix("'"),
                "prime_var": prime_var,
                "expression": node_text(inner.rhs),
                "code": node_text(expression),
            }
        return {"kind": "derivative_expression", "code": node_text(expression)}

    if expression.is_function_call():
        return {
            "kind": "call",
            "name": expression.get_node_name(),
            "args": [node_text(argument) for argument in expression.arguments],
            "code": node_text(expression),
        }

    return {"kind": "raw_expression", "code": node_text(expression)}


def _normalize_statement_item(statement) -> dict[str, Any]:
    if statement.is_local_list_statement():
        return {
            "kind": "local",
            "names": [variable.get_node_name() for variable in statement.variables],
            "code": node_text(statement),
        }

    if statement.is_solve_block():
        return {
            "kind": "solve",
            "target": str(statement.block_name),
            "method": str(statement.method) if getattr(statement, "method", None) else None,
            "code": node_text(statement),
        }

    if statement.is_expression_statement():
        return _normalize_expression_item(statement.expression)

    return {"kind": "raw_statement", "code": node_text(statement)}


def _normalize_statement_block(statement_block) -> list[dict[str, Any]]:
    if statement_block is None or not hasattr(statement_block, "statements"):
        return []
    return [_normalize_statement_item(statement) for statement in statement_block.statements]


def _assignment_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "assigned_var": item["assigned_var"],
        "expression": item["expression"],
    }


def _call_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "args": item["args"],
    }


def build_canonical_blocks(program) -> dict[str, Any]:
    canonical: dict[str, Any] = {block_name: [] for block_name in BLOCK_ORDER}

    title_nodes = lookup(program, "MODEL")
    canonical["TITLE"] = {"text": node_text(title_nodes[0]).removeprefix("TITLE").strip()} if title_nodes else {}

    canonical["COMMENT"] = [node_text(node) for node in lookup(program, "BLOCK_COMMENT")]
    canonical["COMMENT"].extend(node_text(node) for node in lookup(program, "LINE_COMMENT"))

    neuron_nodes = lookup(program, "NEURON_BLOCK")
    if neuron_nodes:
        neuron_block = neuron_nodes[0]
        canonical["NEURON"] = {
            "suffix": next((node.get_node_name() for node in lookup(neuron_block, "SUFFIX")), None),
            "useion": [
                {
                    "ion": useion.get_node_name(),
                    "read": [node.get_node_name() for node in lookup(useion, "READ_ION_VAR")],
                    "write": [node.get_node_name() for node in lookup(useion, "WRITE_ION_VAR")],
                }
                for useion in lookup(neuron_block, "USEION")
            ],
            "range": [node.get_node_name() for node in lookup(neuron_block, "RANGE_VAR")],
            "global": [node.get_node_name() for node in lookup(neuron_block, "GLOBAL_VAR")],
            "nonspecific_current": [
                node.get_node_name() for node in lookup(neuron_block, "NONSPECIFIC_CUR_VAR")
            ],
        }
    else:
        canonical["NEURON"] = {}

    canonical["UNITS"] = {
        strip_unit(node_text(unit_def.unit1)): strip_unit(node_text(unit_def.unit2))
        for unit_def in lookup(program, "UNIT_DEF")
    }

    param_blocks = lookup(program, "PARAM_BLOCK")
    canonical["PARAMETER"] = []
    if param_blocks:
        canonical["PARAMETER"] = [
            {
                "name": statement.get_node_name(),
                "value": maybe_number(node_text(statement.value)) if getattr(statement, "value", None) else None,
                "unit": strip_unit(node_text(statement.unit)) if getattr(statement, "unit", None) else None,
            }
            for statement in param_blocks[0].statements
        ]

    assigned_blocks = lookup(program, "ASSIGNED_BLOCK")
    canonical["ASSIGNED"] = []
    if assigned_blocks:
        canonical["ASSIGNED"] = [
            {
                "name": definition.get_node_name(),
                "unit": strip_unit(node_text(definition.unit)) if getattr(definition, "unit", None) else None,
            }
            for definition in assigned_blocks[0].definitions
        ]

    state_blocks = lookup(program, "STATE_BLOCK")
    if state_blocks:
        canonical["STATE"] = {
            "variables": [{"name": definition.get_node_name(), "power": 1} for definition in state_blocks[0].definitions]
        }
    else:
        canonical["STATE"] = {"variables": []}

    initial_blocks = lookup(program, "INITIAL_BLOCK")
    if initial_blocks:
        normalized_items = _normalize_statement_block(initial_blocks[0].statement_block)
        canonical["INITIAL"] = {
            "func_calls": [_call_payload(item) for item in normalized_items if item["kind"] == "call"],
            "statements": [_assignment_payload(item) for item in normalized_items if item["kind"] == "assignment"],
            "other_statements": [item["code"] for item in normalized_items if item["kind"] not in {"call", "assignment"}],
        }
    else:
        canonical["INITIAL"] = {"func_calls": [], "statements": [], "other_statements": []}

    breakpoint_blocks = lookup(program, "BREAKPOINT_BLOCK")
    if breakpoint_blocks:
        normalized_items = _normalize_statement_block(breakpoint_blocks[0].statement_block)
        solve_stmt = next((item for item in normalized_items if item["kind"] == "solve"), None)
        canonical["BREAKPOINT"] = {
            "solve_stmt": {
                "target": solve_stmt["target"],
                "method": solve_stmt["method"],
            }
            if solve_stmt
            else None,
            "func_calls": [_call_payload(item) for item in normalized_items if item["kind"] == "call"],
            "statements": [_assignment_payload(item) for item in normalized_items if item["kind"] == "assignment"],
            "other_statements": [
                item["code"] for item in normalized_items if item["kind"] not in {"solve", "call", "assignment"}
            ],
        }
    else:
        canonical["BREAKPOINT"] = {
            "solve_stmt": None,
            "func_calls": [],
            "statements": [],
            "other_statements": [],
        }

    canonical["DERIVATIVE"] = []
    for derivative_block in lookup(program, "DERIVATIVE_BLOCK"):
        normalized_items = _normalize_statement_block(derivative_block.statement_block)
        canonical["DERIVATIVE"].append(
            {
                "name": derivative_block.name.get_node_name(),
                "func_calls": [_call_payload(item) for item in normalized_items if item["kind"] == "call"],
                "statements": [
                    {
                        "assigned_var": item["assigned_var"],
                        "prime_var": item["prime_var"],
                        "expression": item["expression"],
                    }
                    for item in normalized_items
                    if item["kind"] == "derivative_assignment"
                ],
                "other_statements": [
                    item["code"]
                    for item in normalized_items
                    if item["kind"] not in {"call", "derivative_assignment"}
                ],
            }
        )

    canonical["FUNCTION"] = []
    for function_block in lookup(program, "FUNCTION_BLOCK"):
        normalized_items = _normalize_statement_block(function_block.statement_block)
        canonical["FUNCTION"].append(
            {
                "signature": {
                    "name": function_block.name.get_node_name(),
                    "params": _normalize_params(function_block.parameters),
                    "returned_unit": strip_unit(node_text(function_block.unit))
                    if getattr(function_block, "unit", None)
                    else None,
                },
                "locals": [
                    name for item in normalized_items if item["kind"] == "local" for name in item["names"]
                ],
                "statements": [
                    _assignment_payload(item)
                    if item["kind"] == "assignment"
                    else _call_payload(item)
                    if item["kind"] == "call"
                    else {"kind": item["kind"], "code": item["code"]}
                    for item in normalized_items
                    if item["kind"] != "local"
                ],
            }
        )

    canonical["PROCEDURE"] = []
    for procedure_block in lookup(program, "PROCEDURE_BLOCK"):
        normalized_items = _normalize_statement_block(procedure_block.statement_block)
        canonical["PROCEDURE"].append(
            {
                "signature": {
                    "name": procedure_block.name.get_node_name(),
                    "params": _normalize_params(procedure_block.parameters),
                },
                "locals": [
                    name for item in normalized_items if item["kind"] == "local" for name in item["names"]
                ],
                "statements": [
                    _assignment_payload(item)
                    if item["kind"] == "assignment"
                    else _call_payload(item)
                    if item["kind"] == "call"
                    else {"kind": item["kind"], "code": item["code"]}
                    for item in normalized_items
                    if item["kind"] != "local"
                ],
            }
        )

    return canonical


def collect_unhandled_raw_blocks(
    raw_blocks: dict[str, list[str]],
    handled_blocks: set[str] | None = None,
) -> dict[str, list[str]]:
    handled = handled_blocks or HANDLED_CANONICAL_BLOCKS
    return {
        block_name: block_values
        for block_name, block_values in raw_blocks.items()
        if block_name not in handled and block_values
    }
