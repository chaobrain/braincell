from __future__ import annotations

import ast as pyast
import re
from typing import Any

from .model import CallableAst
from .model import CurrentInfo
from .model import GateKinetics
from .model import ModuleAst
from .model import ParameterAst
from .model import SemanticModuleIR
from .model import SolveInfo
from .model import SymbolInfo

TOKEN_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _expression_tokens(expression: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(expression))


def _replace_identifiers(expression: str, replacements: dict[str, str]) -> str:
    rewritten = expression
    for name, target in sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0])):
        rewritten = re.sub(rf"(?<!\.)\b{re.escape(name)}\b", target, rewritten)
    return rewritten


def _strip_outer_parens(expression: str) -> str:
    value = expression.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(value):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    balanced = False
                    break
        if balanced:
            value = value[1:-1].strip()
        else:
            break
    return value


def _substitute_env(expression: str, env: dict[str, str]) -> str:
    rewritten = expression
    for name, value in sorted(env.items(), key=lambda item: (-len(item[0]), item[0])):
        rewritten = re.sub(rf"\b{re.escape(name)}\b", f"({value})", rewritten)
    return rewritten


def _find_matching_paren(text: str, start_index: int) -> int:
    depth = 0
    for index in range(start_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced parentheses in expression: {text}")


def _split_top_level(text: str, delimiter: str) -> list[str]:
    items: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == delimiter and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def _inline_function_calls(
    expression: str,
    function_map: dict[str, dict[str, Any]],
    env: dict[str, str],
    *,
    depth: int = 0,
) -> str:
    if depth > 8:
        return expression

    result: list[str] = []
    index = 0
    changed = False
    while index < len(expression):
        if expression[index].isalpha() or expression[index] == "_":
            start = index
            index += 1
            while index < len(expression) and (expression[index].isalnum() or expression[index] == "_"):
                index += 1
            name = expression[start:index]
            next_index = index
            while next_index < len(expression) and expression[next_index].isspace():
                next_index += 1
            if next_index < len(expression) and expression[next_index] == "(" and name in function_map:
                end_index = _find_matching_paren(expression, next_index)
                arg_text = expression[next_index + 1:end_index]
                args = [
                    _expand_expression(argument, env, function_map, depth=depth + 1)
                    for argument in _split_top_level(arg_text, ",")
                ]
                function_info = function_map[name]
                replacements = {
                    parameter_name: f"({argument})"
                    for parameter_name, argument in zip(function_info["params"], args)
                }
                body = _replace_identifiers(function_info["expression"], replacements)
                result.append(f"({body})")
                changed = True
                index = end_index + 1
                continue
            result.append(name)
            continue
        result.append(expression[index])
        index += 1

    rewritten = "".join(result)
    if changed:
        rewritten = _expand_expression(rewritten, env, function_map, depth=depth + 1)
    return rewritten


def _expand_expression(
    expression: str,
    env: dict[str, str],
    function_map: dict[str, dict[str, Any]],
    *,
    depth: int = 0,
) -> str:
    if depth > 8:
        return expression
    rewritten = _strip_outer_parens(expression)
    rewritten = _substitute_env(rewritten, env)
    rewritten = _inline_function_calls(rewritten, function_map, env, depth=depth)
    return _strip_outer_parens(_substitute_env(rewritten, env))


def _build_inline_function_map(functions: tuple[CallableAst, ...]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    function_map: dict[str, dict[str, Any]] = {}
    diagnostics: list[str] = []

    for function_block in functions:
        local_env: dict[str, str] = {}
        return_expression: str | None = None
        unsupported = False
        for statement in function_block.statements:
            if statement.kind == "assignment" and statement.assigned_var and statement.expression is not None:
                expanded = _expand_expression(statement.expression.text, local_env, function_map)
                if statement.assigned_var == function_block.name:
                    return_expression = expanded
                else:
                    local_env[statement.assigned_var] = expanded
            else:
                unsupported = True
                diagnostics.append(
                    f"FUNCTION {function_block.name} contains unsupported statements and cannot be fully inlined."
                )
                break
        if unsupported or return_expression is None:
            continue
        function_map[function_block.name] = {
            "params": [parameter.name for parameter in function_block.params],
            "expression": return_expression,
        }
    return function_map, diagnostics


def _build_procedure_env(
    procedures: tuple[CallableAst, ...],
    requested_names: list[str],
    function_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    procedure_lookup = {procedure.name: procedure for procedure in procedures}
    diagnostics: list[str] = []
    env: dict[str, str] = {}

    names = list(dict.fromkeys(requested_names))
    if not names and len(procedures) == 1:
        names = [procedures[0].name]

    for name in names:
        procedure = procedure_lookup.get(name)
        if procedure is None:
            diagnostics.append(f"Referenced PROCEDURE {name} is missing.")
            continue
        for statement in procedure.statements:
            if statement.kind == "assignment" and statement.assigned_var and statement.expression is not None:
                env[statement.assigned_var] = _expand_expression(statement.expression.text, env, function_map)
            elif statement.kind != "call":
                diagnostics.append(
                    f"PROCEDURE {procedure.name} contains unsupported statement kind {statement.kind} for semantic IR."
                )
    return env, diagnostics


def _parse_python_ast(expression: str):
    return pyast.parse(expression.replace("^", "**"), mode="eval").body


def _ast_to_source(node) -> str:
    return pyast.unparse(node)


def _factor_fraction(node) -> tuple[list[Any], list[Any]]:
    if isinstance(node, pyast.BinOp) and isinstance(node.op, pyast.Mult):
        left_num, left_den = _factor_fraction(node.left)
        right_num, right_den = _factor_fraction(node.right)
        return left_num + right_num, left_den + right_den
    if isinstance(node, pyast.BinOp) and isinstance(node.op, pyast.Div):
        left_num, left_den = _factor_fraction(node.left)
        right_num, right_den = _factor_fraction(node.right)
        return left_num + right_den, left_den + right_num
    return [node], []


def _maybe_extract_state_power(node, state_names: set[str]) -> tuple[str, int] | None:
    if isinstance(node, pyast.Name) and node.id in state_names:
        return node.id, 1
    if isinstance(node, pyast.BinOp) and isinstance(node.op, pyast.Pow):
        if isinstance(node.left, pyast.Name) and node.left.id in state_names:
            if isinstance(node.right, pyast.Constant) and isinstance(node.right.value, int):
                return node.left.id, int(node.right.value)
    return None


def _maybe_extract_driving_force(node, reversal_name: str) -> str | None:
    if not isinstance(node, pyast.BinOp) or not isinstance(node.op, pyast.Sub):
        return None
    left_names = _expression_tokens(_ast_to_source(node.left))
    right_names = _expression_tokens(_ast_to_source(node.right))
    if "v" in left_names and reversal_name in right_names:
        return "voltage_minus_reversal"
    if reversal_name in left_names and "v" in right_names:
        return "reversal_minus_voltage"
    return None


def _product_nodes_to_source(nodes: list[Any]) -> str:
    if not nodes:
        return "1"
    if len(nodes) == 1:
        return _ast_to_source(nodes[0])
    return " * ".join(f"({_ast_to_source(node)})" for node in nodes)


def _extract_gate_kinetics(
    state_name: str,
    derivative_expression: str,
    function_map: dict[str, dict[str, Any]],
    procedure_env: dict[str, str],
    parameter_lookup: dict[str, ParameterAst],
) -> tuple[GateKinetics | None, list[str]]:
    diagnostics: list[str] = []
    try:
        root = _parse_python_ast(derivative_expression)
    except SyntaxError:
        diagnostics.append(f"Derivative expression for state {state_name} is not parseable: {derivative_expression}")
        return None, diagnostics

    numerator_factors, denominator_factors = _factor_fraction(root)

    diff_factor = None
    phi_factors: list[Any] = []
    for factor in numerator_factors:
        if (
            isinstance(factor, pyast.BinOp)
            and isinstance(factor.op, pyast.Sub)
            and isinstance(factor.right, pyast.Name)
            and factor.right.id == state_name
        ):
            diff_factor = factor
            continue
        phi_factors.append(factor)

    if diff_factor is not None and denominator_factors:
        inf_expression = _expand_expression(_ast_to_source(diff_factor.left), procedure_env, function_map)
        tau_expression = _expand_expression(_product_nodes_to_source(denominator_factors), procedure_env, function_map)
        phi_expression = _product_nodes_to_source(phi_factors) if phi_factors else None
        return GateKinetics(
            state=state_name,
            derivative_expression=derivative_expression,
            source_form="inf_tau",
            inf_expression=inf_expression,
            tau_expression=tau_expression,
            phi_expression=phi_expression,
        ), diagnostics

    if denominator_factors:
        diagnostics.append(
            f"State {state_name} derivative includes division but is not recognizable inf/tau form: {derivative_expression}"
        )
        return None, diagnostics

    core_node = None
    residual_phi: list[Any] = []
    for factor in numerator_factors:
        if isinstance(factor, pyast.BinOp) and isinstance(factor.op, pyast.Sub):
            core_node = factor
            continue
        residual_phi.append(factor)

    if core_node is None:
        diagnostics.append(f"State {state_name} derivative is not recognizable HH form: {derivative_expression}")
        return None, diagnostics

    left_num, left_den = _factor_fraction(core_node.left)
    right_num, right_den = _factor_fraction(core_node.right)
    if left_den or right_den:
        diagnostics.append(f"State {state_name} derivative has unsupported alpha/beta fractions: {derivative_expression}")
        return None, diagnostics

    alpha_factors: list[Any] = []
    left_has_one_minus = False
    for factor in left_num:
        if (
            isinstance(factor, pyast.BinOp)
            and isinstance(factor.op, pyast.Sub)
            and isinstance(factor.left, pyast.Constant)
            and factor.left.value in (1, 1.0)
            and isinstance(factor.right, pyast.Name)
            and factor.right.id == state_name
        ):
            left_has_one_minus = True
            continue
        alpha_factors.append(factor)

    beta_factors: list[Any] = []
    right_has_state = False
    for factor in right_num:
        if isinstance(factor, pyast.Name) and factor.id == state_name:
            right_has_state = True
            continue
        beta_factors.append(factor)

    if not left_has_one_minus or not right_has_state:
        diagnostics.append(f"State {state_name} derivative is not recognizable alpha/beta form: {derivative_expression}")
        return None, diagnostics

    alpha_expression = _expand_expression(_product_nodes_to_source(alpha_factors), procedure_env, function_map)
    beta_expression = _expand_expression(_product_nodes_to_source(beta_factors), procedure_env, function_map)
    q10_expression = "1.0"
    q10_source = "default:1.0"
    if "q10" in parameter_lookup:
        q10_param = parameter_lookup["q10"]
        if q10_param.value is not None:
            q10_expression = repr(q10_param.value)
            q10_source = "parameter:q10"
    return GateKinetics(
        state=state_name,
        derivative_expression=derivative_expression,
        source_form="alpha_beta",
        alpha_expression=alpha_expression,
        beta_expression=beta_expression,
        phi_expression=_product_nodes_to_source(residual_phi) if residual_phi else None,
        q10_expression=q10_expression,
        q10_source=q10_source,
    ), diagnostics


def build_semantic_ir(module_ast: ModuleAst) -> SemanticModuleIR:
    diagnostics: list[str] = []
    function_map, function_diagnostics = _build_inline_function_map(module_ast.functions)
    diagnostics.extend(function_diagnostics)

    initial_call_names = [statement.name for statement in module_ast.initial if statement.kind == "call" and statement.name]
    derivative_call_names: list[str] = []
    derivative_lookup: dict[str, str] = {}
    for statements in module_ast.derivative_blocks.values():
        for statement in statements:
            if statement.kind == "call" and statement.name:
                derivative_call_names.append(statement.name)
            if statement.kind == "derivative_assignment" and statement.assigned_var and statement.expression:
                derivative_lookup[statement.assigned_var] = statement.expression.text
    procedure_env, procedure_diagnostics = _build_procedure_env(
        module_ast.procedures,
        initial_call_names + derivative_call_names,
        function_map,
    )
    diagnostics.extend(procedure_diagnostics)

    breakpoint_assignments: dict[str, str] = {}
    for statement in module_ast.breakpoint:
        if statement.kind == "assignment" and statement.assigned_var and statement.expression is not None:
            breakpoint_assignments[statement.assigned_var] = _expand_expression(
                statement.expression.text,
                breakpoint_assignments | procedure_env,
                function_map,
            )

    initial_assignments = {
        statement.assigned_var: statement.expression.text
        for statement in module_ast.initial
        if statement.kind == "assignment" and statement.assigned_var and statement.expression is not None
    }

    parameter_lookup = {parameter.name: parameter for parameter in module_ast.parameters}
    state_names = {state.name for state in module_ast.states}

    call_graph = {
        function.name: tuple(
            sorted(
                target
                for statement in function.statements
                if statement.kind == "call" and statement.name
                for target in [statement.name]
            )
        )
        for function in (*module_ast.functions, *module_ast.procedures)
    }

    ion_dependencies: dict[str, dict[str, tuple[str, ...] | str | None]] = {}
    currents: list[CurrentInfo] = []
    for useion in module_ast.useions:
        ion_dependencies[useion.ion] = {
            "read": useion.read,
            "write": useion.write,
            "root_kind": "density_channel",
        }
        for write_var in useion.write:
            expression = breakpoint_assignments.get(write_var)
            if expression is None:
                diagnostics.append(f"BREAKPOINT does not assign ion current variable {write_var}.")
                continue
            try:
                root = _parse_python_ast(expression)
            except SyntaxError:
                diagnostics.append(f"Could not parse ion current expression for {write_var}: {expression}")
                continue
            numerator_factors, denominator_factors = _factor_fraction(root)
            if denominator_factors:
                diagnostics.append(f"Current {write_var} contains division and is not treated as pure ohmic form.")
            conductance_candidates: list[str] = []
            state_factors: dict[str, int] = {}
            driving_force = None
            for factor in numerator_factors:
                gate_info = _maybe_extract_state_power(factor, state_names)
                if gate_info is not None:
                    state_name, power = gate_info
                    state_factors[state_name] = state_factors.get(state_name, 0) + power
                    continue
                orientation = _maybe_extract_driving_force(factor, useion.read[0] if useion.read else "")
                if orientation is not None:
                    driving_force = orientation
                    continue
                if isinstance(factor, pyast.Name) and factor.id in parameter_lookup:
                    conductance_candidates.append(factor.id)
            currents.append(
                CurrentInfo(
                    name=write_var,
                    expression=expression,
                    ion=useion.ion,
                    read_var=useion.read[0] if useion.read else None,
                    conductance_candidates=tuple(conductance_candidates),
                    state_factors=state_factors,
                    driving_force=driving_force,
                )
            )

    gate_kinetics: list[GateKinetics] = []
    for state_name, derivative_expression in sorted(derivative_lookup.items()):
        kinetics, gate_diagnostics = _extract_gate_kinetics(
            state_name,
            derivative_expression,
            function_map,
            procedure_env,
            parameter_lookup,
        )
        diagnostics.extend(gate_diagnostics)
        if kinetics is not None:
            gate_kinetics.append(kinetics)

    unsupported_features = sorted(module_ast.unsupported_blocks)
    if len(module_ast.useions) > 1:
        unsupported_features.append("multi_useion")
    if module_ast.nonspecific_currents:
        unsupported_features.append("nonspecific_current")
    unsupported_features = sorted(dict.fromkeys(unsupported_features))

    symbols: list[SymbolInfo] = []
    for parameter in module_ast.parameters:
        symbols.append(
            SymbolInfo(
                name=parameter.name,
                kind="parameter",
                unit=parameter.unit,
                default_value=parameter.value,
                declared_in="PARAMETER",
            )
        )
    for variable in module_ast.assigned:
        symbols.append(SymbolInfo(name=variable.name, kind="assigned", unit=variable.unit, declared_in="ASSIGNED"))
    for variable in module_ast.states:
        symbols.append(SymbolInfo(name=variable.name, kind="state", unit=variable.unit, declared_in="STATE"))
    for name in module_ast.ranges:
        symbols.append(SymbolInfo(name=name, kind="range", declared_in="NEURON"))
    for name in module_ast.globals:
        symbols.append(SymbolInfo(name=name, kind="global", declared_in="NEURON"))

    source_summary = {
        "function_names": sorted(function_map),
        "procedure_assignments": procedure_env,
        "breakpoint_assignments": breakpoint_assignments,
        "initial_assignments": initial_assignments,
        "currents": {current.name: current.expression for current in currents},
    }

    return SemanticModuleIR(
        source_file=module_ast.source_file,
        mechanism_name=module_ast.mechanism_name,
        module_kind="density_channel",
        title=module_ast.title,
        useions=module_ast.useions,
        symbols=tuple(symbols),
        states=module_ast.states,
        parameters=module_ast.parameters,
        assigned=module_ast.assigned,
        ranges=module_ast.ranges,
        globals=module_ast.globals,
        functions=module_ast.functions,
        procedures=module_ast.procedures,
        call_graph=call_graph,
        procedure_env=procedure_env,
        breakpoint_assignments=breakpoint_assignments,
        initial_assignments=initial_assignments,
        solve_specs=tuple(
            SolveInfo(block="BREAKPOINT", target=solve.target, method=solve.method)
            for solve in module_ast.solve_blocks
        ),
        currents=tuple(currents),
        gate_kinetics=tuple(gate_kinetics),
        ion_dependencies=ion_dependencies,
        diagnostics=tuple(diagnostics),
        unsupported_features=tuple(unsupported_features),
        source_summary=source_summary,
    )
