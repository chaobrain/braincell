from __future__ import annotations

import ast as pyast
from typing import Any

from ..model import CallableAst
from ..model import ExprAst
from ..model import ModuleAst
from ..model import ParameterAst
from ..model import SolveAst
from ..model import SourceSpan
from ..model import StatementAst
from ..model import UseIonAst
from ..model import VariableAst
from .canonical import collect_unhandled_raw_blocks


def _rewrite_expression(text: str) -> str:
    return text.replace("^", "**")


def _find_source_span(source_text: str, snippet: str | None) -> SourceSpan | None:
    if not snippet:
        return None
    index = source_text.find(snippet)
    if index < 0:
        return SourceSpan(snippet=snippet)

    start_line = source_text.count("\n", 0, index) + 1
    line_start = source_text.rfind("\n", 0, index)
    start_column = index + 1 if line_start < 0 else index - line_start
    end_index = index + len(snippet)
    end_line = source_text.count("\n", 0, end_index) + 1
    end_line_start = source_text.rfind("\n", 0, end_index)
    end_column = end_index + 1 if end_line_start < 0 else end_index - end_line_start
    return SourceSpan(
        start_line=start_line,
        start_column=start_column,
        end_line=end_line,
        end_column=end_column,
        snippet=snippet,
    )


def _expr_from_python_node(node: pyast.AST, text: str, source_text: str) -> ExprAst:
    if isinstance(node, pyast.Name):
        return ExprAst(kind="name", text=text, name=node.id, source_span=_find_source_span(source_text, text))
    if isinstance(node, pyast.Constant):
        return ExprAst(kind="literal", text=text, value=node.value, source_span=_find_source_span(source_text, text))
    if isinstance(node, pyast.UnaryOp):
        return ExprAst(
            kind="unary",
            text=text,
            operator=type(node.op).__name__,
            operand=_expr_from_python_node(node.operand, pyast.unparse(node.operand), source_text),
            source_span=_find_source_span(source_text, text),
        )
    if isinstance(node, pyast.BinOp):
        return ExprAst(
            kind="binary",
            text=text,
            operator=type(node.op).__name__,
            left=_expr_from_python_node(node.left, pyast.unparse(node.left), source_text),
            right=_expr_from_python_node(node.right, pyast.unparse(node.right), source_text),
            source_span=_find_source_span(source_text, text),
        )
    if isinstance(node, pyast.Call):
        func_text = pyast.unparse(node.func)
        return ExprAst(
            kind="call",
            text=text,
            name=func_text,
            args=tuple(
                _expr_from_python_node(argument, pyast.unparse(argument), source_text)
                for argument in node.args
            ),
            source_span=_find_source_span(source_text, text),
        )
    return ExprAst(kind="raw", text=text, source_span=_find_source_span(source_text, text))


def parse_expression_ast(text: str | None, source_text: str) -> ExprAst | None:
    if text is None:
        return None
    try:
        node = pyast.parse(_rewrite_expression(text), mode="eval").body
    except SyntaxError:
        return ExprAst(kind="raw", text=text, source_span=_find_source_span(source_text, text))
    return _expr_from_python_node(node, text, source_text)


def _statement_from_canonical(item: dict[str, Any], source_text: str) -> StatementAst:
    kind = item["kind"]
    code = item.get("code") or item.get("expression") or item.get("name") or ""
    if kind in {"assignment", "derivative_assignment"}:
        expression = parse_expression_ast(item.get("expression"), source_text)
        return StatementAst(
            kind=kind,
            code=code,
            assigned_var=item.get("assigned_var"),
            prime_var=item.get("prime_var"),
            expression=expression,
            source_span=_find_source_span(source_text, code),
        )
    if kind == "call":
        return StatementAst(
            kind=kind,
            code=code,
            name=item.get("name"),
            args=tuple(parse_expression_ast(argument, source_text) for argument in item.get("args", [])),
            source_span=_find_source_span(source_text, code),
        )
    if kind == "solve":
        return StatementAst(
            kind=kind,
            code=code,
            target=item.get("target"),
            method=item.get("method"),
            source_span=_find_source_span(source_text, code),
        )
    if kind == "local":
        return StatementAst(
            kind=kind,
            code=code,
            names=tuple(item.get("names", [])),
            source_span=_find_source_span(source_text, code),
        )
    return StatementAst(kind=kind, code=code, source_span=_find_source_span(source_text, code))


def _parameter_ast(item: dict[str, Any], source_text: str) -> ParameterAst:
    snippet = item["name"]
    return ParameterAst(
        name=item["name"],
        unit=item.get("unit"),
        value=item.get("value"),
        source_span=_find_source_span(source_text, snippet),
    )


def _variable_ast(item: dict[str, Any], source_text: str) -> VariableAst:
    snippet = item["name"]
    return VariableAst(
        name=item["name"],
        unit=item.get("unit"),
        power=item.get("power"),
        source_span=_find_source_span(source_text, snippet),
    )


def _callable_ast(item: dict[str, Any], source_text: str, *, returned_unit: str | None = None) -> CallableAst:
    statements: list[StatementAst] = []
    for statement in item.get("statements", []):
        if isinstance(statement, dict) and "kind" not in statement:
            statement = {
                "kind": "assignment" if "assigned_var" in statement else "call",
                "code": statement.get("code")
                or statement.get("expression")
                or statement.get("name")
                or item["signature"]["name"],
                **statement,
            }
        statements.append(_statement_from_canonical(statement, source_text))
    return CallableAst(
        name=item["signature"]["name"],
        params=tuple(
            ParameterAst(
                name=parameter["name"],
                unit=parameter.get("unit"),
                source_span=_find_source_span(source_text, parameter["name"]),
            )
            for parameter in item["signature"].get("params", [])
        ),
        locals=tuple(item.get("locals", [])),
        statements=tuple(statements),
        returned_unit=returned_unit,
        source_span=_find_source_span(source_text, item["signature"]["name"]),
    )


def build_typed_ast(
    *,
    mod_file: str,
    source_text: str,
    canonical_blocks: dict[str, Any],
    raw_blocks: dict[str, list[str]],
) -> ModuleAst:
    neuron_block = canonical_blocks.get("NEURON") or {}
    derivative_blocks = {}
    for derivative in canonical_blocks.get("DERIVATIVE", []):
        statements = []
        for statement in derivative.get("statements", []):
            statement = {
                "kind": "derivative_assignment",
                "code": statement.get("code") or statement["expression"],
                **statement,
            }
            statements.append(_statement_from_canonical(statement, source_text))
        derivative_blocks[derivative["name"]] = tuple(statements)

    breakpoint_statements: list[StatementAst] = []
    breakpoint_block = canonical_blocks.get("BREAKPOINT") or {}
    if breakpoint_block.get("solve_stmt"):
        solve_stmt = breakpoint_block["solve_stmt"]
        breakpoint_statements.append(
            StatementAst(
                kind="solve",
                code=f"SOLVE {solve_stmt['target']} METHOD {solve_stmt['method'] or ''}".strip(),
                target=solve_stmt["target"],
                method=solve_stmt.get("method"),
                source_span=_find_source_span(source_text, solve_stmt["target"]),
            )
        )
    for call in breakpoint_block.get("func_calls", []):
        breakpoint_statements.append(
            _statement_from_canonical({"kind": "call", "code": call["name"], **call}, source_text)
        )
    for statement in breakpoint_block.get("statements", []):
        breakpoint_statements.append(
            _statement_from_canonical(
                {"kind": "assignment", "code": statement["expression"], **statement},
                source_text,
            )
        )
    for code in breakpoint_block.get("other_statements", []):
        breakpoint_statements.append(
            StatementAst(kind="raw_statement", code=code, source_span=_find_source_span(source_text, code))
        )

    initial_statements: list[StatementAst] = []
    initial_block = canonical_blocks.get("INITIAL") or {}
    for call in initial_block.get("func_calls", []):
        initial_statements.append(
            _statement_from_canonical({"kind": "call", "code": call["name"], **call}, source_text)
        )
    for statement in initial_block.get("statements", []):
        initial_statements.append(
            _statement_from_canonical(
                {"kind": "assignment", "code": statement["expression"], **statement},
                source_text,
            )
        )
    for code in initial_block.get("other_statements", []):
        initial_statements.append(
            StatementAst(kind="raw_statement", code=code, source_span=_find_source_span(source_text, code))
        )

    solve_blocks = tuple(
        SolveAst(
            target=statement.target or "",
            method=statement.method,
            source_span=statement.source_span,
        )
        for statement in breakpoint_statements
        if statement.kind == "solve" and statement.target
    )

    unsupported_blocks = collect_unhandled_raw_blocks(raw_blocks)

    return ModuleAst(
        source_file=mod_file,
        title=(canonical_blocks.get("TITLE") or {}).get("text"),
        mechanism_name=neuron_block.get("suffix") or mod_file.rsplit("/", 1)[-1].removesuffix(".mod"),
        raw_blocks=raw_blocks,
        canonical_blocks=canonical_blocks,
        comments=tuple(canonical_blocks.get("COMMENT", [])),
        useions=tuple(
            UseIonAst(
                ion=item["ion"],
                read=tuple(item.get("read", [])),
                write=tuple(item.get("write", [])),
                source_span=_find_source_span(source_text, item["ion"]),
            )
            for item in neuron_block.get("useion", [])
        ),
        ranges=tuple(neuron_block.get("range", [])),
        globals=tuple(neuron_block.get("global", [])),
        nonspecific_currents=tuple(neuron_block.get("nonspecific_current", [])),
        parameters=tuple(_parameter_ast(item, source_text) for item in canonical_blocks.get("PARAMETER", [])),
        assigned=tuple(_variable_ast(item, source_text) for item in canonical_blocks.get("ASSIGNED", [])),
        states=tuple(
            _variable_ast(item, source_text)
            for item in (canonical_blocks.get("STATE") or {}).get("variables", [])
        ),
        initial=tuple(initial_statements),
        breakpoint=tuple(breakpoint_statements),
        solve_blocks=solve_blocks,
        derivative_blocks=derivative_blocks,
        functions=tuple(
            _callable_ast(item, source_text, returned_unit=item["signature"].get("returned_unit"))
            for item in canonical_blocks.get("FUNCTION", [])
        ),
        procedures=tuple(_callable_ast(item, source_text) for item in canonical_blocks.get("PROCEDURE", [])),
        unsupported_blocks=unsupported_blocks,
    )
