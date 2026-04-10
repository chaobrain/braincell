

import ast as pyast
import keyword
import re
from pathlib import Path
from typing import Any

from ...inspect_ast import collect_unhandled_raw_blocks
from ...inspect_ast import pascal_case

VARIANT_NAME = "one_ion_hh_ohmic"

ION_INFO_MAP = {
    "k": {
        "base_class_name": "PotassiumChannel",
        "ion_arg_name": "K",
        "current_prefix": "IK",
    },
    "na": {
        "base_class_name": "SodiumChannel",
        "ion_arg_name": "Na",
        "current_prefix": "INa",
    },
    "ca": {
        "base_class_name": "CalciumChannel",
        "ion_arg_name": "Ca",
        "current_prefix": "ICa",
    },
}

MATH_FUNCTION_MAP = {
    "exp": "u.math.exp",
    "sin": "u.math.sin",
    "cos": "u.math.cos",
    "tanh": "u.math.tanh",
    "log": "u.math.log",
    "sqrt": "u.math.sqrt",
    "where": "u.math.where",
    "exprel": "u.math.exprel",
    "power": "u.math.power",
}

UNIT_NAME_MAP = {
    "1": "1",
    "S": "u.S",
    "mS": "u.mS",
    "mV": "u.mV",
    "mA": "u.mA",
    "uA": "u.uA",
    "M": "u.M",
    "mM": "u.mM",
    "ms": "u.ms",
    "s": "u.s",
    "cm": "u.cm",
    "um": "u.um",
    "degC": "u.degC",
}

TOKEN_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _safe_identifier(name: str) -> str:
    candidate = re.sub(r"[^0-9A-Za-z_]", "_", name.strip())
    if not candidate:
        return "value"
    if candidate[0].isdigit():
        candidate = f"value_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    return candidate


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


def _unit_to_python_expression(unit: str | None) -> str | None:
    if not unit:
        return None
    compact = unit.replace(" ", "")
    if compact == "1":
        return "1"

    tokens = re.findall(r"[*/]|[A-Za-z]+(?:-?\d+)?|1", compact)
    if not tokens:
        return None

    rendered: list[str] = []
    for token in tokens:
        if token in {"*", "/"}:
            rendered.append(token)
            continue
        match = re.fullmatch(r"([A-Za-z]+)(-?\d+)?", token)
        if not match:
            return None
        base_name, power = match.groups()
        mapped = UNIT_NAME_MAP.get(base_name)
        if mapped is None:
            return None
        term = mapped
        if power and power != "1":
            term = f"({mapped} ** {int(power)})"
        rendered.append(term)

    expression = " ".join(rendered)
    if expression.startswith("/ "):
        expression = f"1 {expression}"
    return expression


def _default_value_expression(value: Any, unit: str | None) -> str:
    if value is None:
        return "None"
    unit_expr = _unit_to_python_expression(unit)
    if isinstance(value, (int, float)):
        if unit_expr and unit_expr != "1":
            return f"{value!r} * ({unit_expr})"
        return repr(value)
    return repr(value)


def _rewrite_math_functions(expression: str) -> str:
    rewritten = expression.replace("^", "**")
    for source_name, target_name in MATH_FUNCTION_MAP.items():
        rewritten = re.sub(rf"\b{source_name}\s*\(", f"{target_name}(", rewritten)
    return rewritten


def _replace_identifiers(expression: str, replacements: dict[str, str]) -> str:
    rewritten = expression
    for name, target in sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0])):
        rewritten = re.sub(rf"(?<!\.)\b{re.escape(name)}\b", target, rewritten)
    return rewritten


def _expression_tokens(expression: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(expression))


def _prepare_python_expression(expression: str, replacements: dict[str, str]) -> str:
    return _replace_identifiers(_rewrite_math_functions(expression), replacements)


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


def _substitute_env(expression: str, env: dict[str, str]) -> str:
    rewritten = expression
    for name, value in sorted(env.items(), key=lambda item: (-len(item[0]), item[0])):
        rewritten = re.sub(rf"\b{re.escape(name)}\b", f"({value})", rewritten)
    return rewritten


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
                body = function_info["expression"]
                replacements = {
                    parameter_name: f"({argument})"
                    for parameter_name, argument in zip(function_info["params"], args)
                }
                inlined = _replace_identifiers(body, replacements)
                result.append(f"({inlined})")
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
    second_pass = _substitute_env(rewritten, env)
    return _strip_outer_parens(second_pass)


def _build_inline_function_map(canonical_blocks: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    function_map: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for function_block in canonical_blocks.get("FUNCTION", []):
        function_name = function_block["signature"]["name"]
        local_env: dict[str, str] = {}
        return_expression: str | None = None
        unsupported = False

        for statement in function_block.get("statements", []):
            if "assigned_var" in statement and "expression" in statement:
                expanded = _expand_expression(statement["expression"], local_env, function_map)
                if statement["assigned_var"] == function_name:
                    return_expression = expanded
                else:
                    local_env[statement["assigned_var"]] = expanded
            else:
                unsupported = True
                issues.append(
                    f"FUNCTION {function_name} contains unsupported statements and cannot be fully inlined."
                )
                break

        if unsupported or return_expression is None:
            continue

        function_map[function_name] = {
            "params": [parameter["name"] for parameter in function_block["signature"].get("params", [])],
            "expression": return_expression,
        }

    return function_map, issues


def _select_driver_procedures(canonical_blocks: dict[str, Any]) -> list[dict[str, Any]]:
    requested_names: list[str] = []
    initial_block = canonical_blocks.get("INITIAL") or {}
    for call in initial_block.get("func_calls", []):
        requested_names.append(call["name"])
    for derivative_block in canonical_blocks.get("DERIVATIVE", []):
        for call in derivative_block.get("func_calls", []):
            requested_names.append(call["name"])

    unique_names = list(dict.fromkeys(requested_names))
    procedures = canonical_blocks.get("PROCEDURE", [])
    if not unique_names and len(procedures) == 1:
        return list(procedures)

    selected = []
    for name in unique_names:
        match = next((procedure for procedure in procedures if procedure["signature"]["name"] == name), None)
        if match is not None:
            selected.append(match)
    return selected


def _build_procedure_env(
    canonical_blocks: dict[str, Any],
    function_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    issues: list[str] = []

    for procedure in _select_driver_procedures(canonical_blocks):
        for statement in procedure.get("statements", []):
            if "assigned_var" in statement and "expression" in statement:
                env[statement["assigned_var"]] = _expand_expression(statement["expression"], env, function_map)
            else:
                issues.append(
                    f"PROCEDURE {procedure['signature']['name']} contains unsupported statements for IR extraction."
                )

    return env, issues


def _resolve_breakpoint_assignments(
    canonical_blocks: dict[str, Any],
    function_map: dict[str, dict[str, Any]],
) -> dict[str, str]:
    env: dict[str, str] = {}
    breakpoint_block = canonical_blocks.get("BREAKPOINT") or {}
    for statement in breakpoint_block.get("statements", []):
        if "assigned_var" in statement and "expression" in statement:
            env[statement["assigned_var"]] = _expand_expression(statement["expression"], env, function_map)
    return env


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


def _is_name(node, name: str) -> bool:
    return isinstance(node, pyast.Name) and node.id == name


def _node_is_constant_one(node) -> bool:
    return isinstance(node, pyast.Constant) and node.value in (1, 1.0)


def _maybe_extract_state_power(node, state_names: set[str]) -> tuple[str, int] | None:
    if isinstance(node, pyast.Name) and node.id in state_names:
        return node.id, 1
    if isinstance(node, pyast.BinOp) and isinstance(node.op, pyast.Pow):
        if isinstance(node.left, pyast.Name) and node.left.id in state_names:
            if isinstance(node.right, pyast.Constant) and isinstance(node.right.value, int):
                return node.left.id, int(node.right.value)
    return None


def _maybe_extract_driving_force(node, voltage_name: str, reversal_name: str) -> str | None:
    if isinstance(node, pyast.BinOp) and isinstance(node.op, pyast.Sub):
        left_names = _expression_tokens(_ast_to_source(node.left))
        right_names = _expression_tokens(_ast_to_source(node.right))
        if voltage_name in left_names and reversal_name in right_names:
            return "voltage_minus_reversal"
        if reversal_name in left_names and voltage_name in right_names:
            return "reversal_minus_voltage"
    return None


def _product_nodes_to_source(nodes: list[Any]) -> str:
    if not nodes:
        return "1"
    if len(nodes) == 1:
        return _ast_to_source(nodes[0])
    return " * ".join(f"({_ast_to_source(node)})" for node in nodes)


def _extract_ohmic_current_model(
    canonical_blocks: dict[str, Any],
    breakpoint_env: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    neuron_block = canonical_blocks.get("NEURON") or {}
    useions = list(neuron_block.get("useion", []))
    if len(useions) != 1:
        issues.append("one_ion_hh_ohmic requires exactly one USEION declaration.")
        return None, issues

    useion = useions[0]
    write_vars = list(useion.get("write", []))
    read_vars = list(useion.get("read", []))
    if len(write_vars) != 1 or len(read_vars) != 1:
        issues.append("one_ion_hh_ohmic requires exactly one READ ion var and one WRITE ion var.")
        return None, issues

    current_var = write_vars[0]
    reversal_var = read_vars[0]
    current_expression = breakpoint_env.get(current_var)
    if current_expression is None:
        issues.append(f"BREAKPOINT does not assign the ion current variable {current_var}.")
        return None, issues

    state_names = {item["name"] for item in canonical_blocks.get("STATE", {}).get("variables", [])}
    parameter_lookup = {item["name"]: item for item in canonical_blocks.get("PARAMETER", [])}

    try:
        root = _parse_python_ast(current_expression)
    except SyntaxError:
        issues.append(f"Current expression could not be parsed as ohmic HH syntax: {current_expression}")
        return None, issues

    numerator_factors, denominator_factors = _factor_fraction(root)
    if denominator_factors:
        issues.append("Current expression contains division and is not treated as pure ohmic form.")
        return None, issues

    driving_force = None
    gate_powers: dict[str, int] = {}
    conductance_name: str | None = None
    extra_factors: list[str] = []

    for factor in numerator_factors:
        orientation = _maybe_extract_driving_force(factor, "v", reversal_var)
        if orientation is not None:
            if driving_force is not None:
                issues.append("Current expression contains more than one driving-force factor.")
                return None, issues
            driving_force = orientation
            continue

        gate_info = _maybe_extract_state_power(factor, state_names)
        if gate_info is not None:
            gate_name, power = gate_info
            gate_powers[gate_name] = gate_powers.get(gate_name, 0) + power
            continue

        factor_source = _ast_to_source(factor)
        if isinstance(factor, pyast.Name):
            factor_name = factor.id
            parameter = parameter_lookup.get(factor_name)
            if parameter and parameter.get("unit") in {"S/cm2", "mS/cm2"} and conductance_name is None:
                conductance_name = factor_name
                continue

        if isinstance(factor, pyast.Constant) and factor.value in (1, 1.0):
            continue
        extra_factors.append(factor_source)

    if driving_force is None:
        issues.append("Could not identify an ohmic driving-force factor `(v - eion)` or `(eion - v)`.")
    if conductance_name is None:
        issues.append("Could not identify a single conductance parameter in the current expression.")
    if not gate_powers:
        issues.append("Could not identify any gate states in the current expression.")
    if extra_factors:
        issues.append(
            "Current expression contains extra multiplicative factors outside conductance and gate powers: "
            + ", ".join(extra_factors)
        )

    if issues:
        return None, issues

    gate_product_parts = []
    for gate_name, power in sorted(gate_powers.items()):
        if power == 1:
            gate_product_parts.append(f"self.{_safe_identifier(gate_name)}.value")
        else:
            gate_product_parts.append(f"self.{_safe_identifier(gate_name)}.value ** {power}")

    current_expression_python = (
        f"self.g_max * {' * '.join(gate_product_parts)} * "
        f"({ION_INFO_MAP[useion['ion']]['ion_arg_name']}.E - V)"
    )

    return {
        "ion_name": useion["ion"],
        "ion_arg_name": ION_INFO_MAP[useion["ion"]]["ion_arg_name"],
        "base_class_name": ION_INFO_MAP[useion["ion"]]["base_class_name"],
        "write_var": current_var,
        "read_var": reversal_var,
        "conductance_source_name": conductance_name,
        "driving_force": driving_force,
        "gate_powers": gate_powers,
        "gate_product_expression": " * ".join(gate_product_parts),
        "current_expression_python": current_expression_python,
    }, issues


def _resolve_name_or_expression(
    candidate: str,
    procedure_env: dict[str, str],
    function_map: dict[str, dict[str, Any]],
) -> str:
    if candidate in procedure_env:
        return procedure_env[candidate]
    return _expand_expression(candidate, procedure_env, function_map)


def _collapse_product(nodes: list[Any]) -> Any:
    if not nodes:
        return pyast.Constant(value=1)
    node = nodes[0]
    for next_node in nodes[1:]:
        node = pyast.BinOp(left=node, op=pyast.Mult(), right=next_node)
    return node


def _extract_inf_tau_pattern(
    state_name: str,
    expression: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        root = _parse_python_ast(expression)
    except SyntaxError:
        return None, "syntax"

    numerator_factors, denominator_factors = _factor_fraction(root)
    if not denominator_factors:
        return None, None

    diff_factor = None
    phi_factors: list[Any] = []
    for factor in numerator_factors:
        if (
            isinstance(factor, pyast.BinOp)
            and isinstance(factor.op, pyast.Sub)
            and _is_name(factor.right, state_name)
        ):
            if diff_factor is None:
                diff_factor = factor
                continue
        phi_factors.append(factor)

    if diff_factor is None:
        return None, None

    inf_source = _ast_to_source(diff_factor.left)
    tau_source = _product_nodes_to_source(denominator_factors)
    phi_source = _product_nodes_to_source(phi_factors) if phi_factors else None
    return {
        "source_form": "inf_tau",
        "inf_source_expr": inf_source,
        "tau_source_expr": tau_source,
        "phi_source_expr": phi_source,
        "inf_source_name": inf_source if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", inf_source) else None,
        "tau_source_name": tau_source if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tau_source) else None,
        "alpha_source_name": None,
        "beta_source_name": None,
    }, None


def _extract_alpha_beta_pattern(
    state_name: str,
    expression: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        root = _parse_python_ast(expression)
    except SyntaxError:
        return None, "syntax"

    numerator_factors, denominator_factors = _factor_fraction(root)
    if denominator_factors:
        return None, None

    phi_factors: list[Any] = []
    core_node = None
    for factor in numerator_factors:
        if isinstance(factor, pyast.BinOp) and isinstance(factor.op, pyast.Sub):
            left_source = _ast_to_source(factor.left)
            right_source = _ast_to_source(factor.right)
            if state_name in _expression_tokens(left_source) and state_name in _expression_tokens(right_source):
                core_node = factor
                continue
        phi_factors.append(factor)

    if core_node is None:
        return None, None

    left_num, left_den = _factor_fraction(core_node.left)
    right_num, right_den = _factor_fraction(core_node.right)
    if left_den or right_den:
        return None, None

    alpha_factors: list[Any] = []
    left_has_one_minus = False
    for factor in left_num:
        if (
            isinstance(factor, pyast.BinOp)
            and isinstance(factor.op, pyast.Sub)
            and _node_is_constant_one(factor.left)
            and _is_name(factor.right, state_name)
        ):
            left_has_one_minus = True
            continue
        alpha_factors.append(factor)

    beta_factors: list[Any] = []
    right_has_state = False
    for factor in right_num:
        if _is_name(factor, state_name):
            right_has_state = True
            continue
        beta_factors.append(factor)

    if not left_has_one_minus or not right_has_state:
        return None, None

    alpha_source = _product_nodes_to_source(alpha_factors)
    beta_source = _product_nodes_to_source(beta_factors)
    phi_source = _product_nodes_to_source(phi_factors) if phi_factors else None
    return {
        "source_form": "alpha_beta",
        "inf_source_expr": None,
        "tau_source_expr": None,
        "phi_source_expr": phi_source,
        "inf_source_name": None,
        "tau_source_name": None,
        "alpha_source_name": alpha_source if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alpha_source) else None,
        "beta_source_name": beta_source if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", beta_source) else None,
        "alpha_source_expr": alpha_source,
        "beta_source_expr": beta_source,
    }, None


def _infer_q10_from_phi(
    phi_expression: str | None,
    parameter_lookup: dict[str, dict[str, Any]],
) -> tuple[str, str | None]:
    if not phi_expression:
        return "1.0", None

    compact = phi_expression.replace(" ", "").replace("**", "^")
    if "q10" in _expression_tokens(compact) and "celsius" in compact and "temp" in compact:
        q10_parameter = parameter_lookup.get("q10")
        if q10_parameter and q10_parameter.get("value") is not None:
            return repr(q10_parameter["value"]), "parameter:q10"
    return "1.0", None


def _combine_tau_with_phi(tau_expression: str, phi_expression: str | None) -> str:
    if not phi_expression or phi_expression.strip() in {"1", "1.0"}:
        return tau_expression
    return f"({tau_expression}) / ({phi_expression})"


def _extract_gate_kinetics(
    gate_name: str,
    derivative_expression: str,
    procedure_env: dict[str, str],
    function_map: dict[str, dict[str, Any]],
    parameter_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []

    inf_tau_match, parse_issue = _extract_inf_tau_pattern(gate_name, derivative_expression)
    if parse_issue == "syntax":
        issues.append(f"Derivative expression for gate {gate_name} is not parseable: {derivative_expression}")
        return None, issues

    alpha_beta_match, _ = _extract_alpha_beta_pattern(gate_name, derivative_expression)
    match = inf_tau_match or alpha_beta_match
    if match is None:
        issues.append(
            f"Gate {gate_name} is neither direct inf/tau nor recognizable alpha/beta form: {derivative_expression}"
        )
        return None, issues

    q10_expression, q10_source = _infer_q10_from_phi(match.get("phi_source_expr"), parameter_lookup)

    if match["source_form"] == "inf_tau":
        inf_expression = _resolve_name_or_expression(match["inf_source_expr"], procedure_env, function_map)
        tau_expression = _resolve_name_or_expression(match["tau_source_expr"], procedure_env, function_map)
        if q10_source is None:
            tau_expression = _combine_tau_with_phi(
                tau_expression,
                _resolve_name_or_expression(match["phi_source_expr"], procedure_env, function_map)
                if match.get("phi_source_expr")
                else None,
            )
    else:
        alpha_expression = _resolve_name_or_expression(match["alpha_source_expr"], procedure_env, function_map)
        beta_expression = _resolve_name_or_expression(match["beta_source_expr"], procedure_env, function_map)
        alpha_plus_beta = f"(({alpha_expression}) + ({beta_expression}))"
        inf_expression = f"({alpha_expression}) / {alpha_plus_beta}"
        tau_expression = f"1 / {alpha_plus_beta}"
        if q10_source is None:
            tau_expression = _combine_tau_with_phi(
                tau_expression,
                _resolve_name_or_expression(match["phi_source_expr"], procedure_env, function_map)
                if match.get("phi_source_expr")
                else None,
            )

    return {
        "name": gate_name,
        "safe_name": _safe_identifier(gate_name),
        "q10_expression": q10_expression,
        "q10_source": q10_source or "default:1.0",
        "source_form": match["source_form"],
        "inf_expression_source": inf_expression,
        "tau_expression_source": tau_expression,
        "inf_source_name": match.get("inf_source_name"),
        "tau_source_name": match.get("tau_source_name"),
        "alpha_source_name": match.get("alpha_source_name"),
        "beta_source_name": match.get("beta_source_name"),
        "derivative_expression": derivative_expression,
    }, issues


def _build_gate_helper_payload(
    gate_info: dict[str, Any],
    parameter_lookup: dict[str, dict[str, Any]],
    *,
    voltage_name: str = "v",
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    combined_expression = f"{gate_info['inf_expression_source']} + {gate_info['tau_expression_source']}"
    used_names = sorted(
        name
        for name in _expression_tokens(combined_expression)
        if name not in set(MATH_FUNCTION_MAP) | {"u", voltage_name}
    )

    alias_lines: list[str] = []
    replacements = {voltage_name: "V"}

    for name in used_names:
        parameter = parameter_lookup.get(name)
        if parameter is None:
            issues.append(f"Gate {gate_info['name']} depends on non-parameter symbol `{name}` after normalization.")
            continue
        safe_name = _safe_identifier(name)
        replacements[name] = safe_name
        unit = parameter.get("unit")
        unit_expr = _unit_to_python_expression(unit)
        if unit_expr and unit_expr != "1":
            alias_lines.append(f"{safe_name} = self.{safe_name} / ({unit_expr})")
        else:
            alias_lines.append(f"{safe_name} = self.{safe_name}")

    inf_expr_python = _prepare_python_expression(gate_info["inf_expression_source"], replacements)
    tau_expr_python = _prepare_python_expression(gate_info["tau_expression_source"], replacements)

    payload = dict(gate_info)
    payload["helper_alias_lines"] = alias_lines
    payload["inf_expr_python"] = inf_expr_python
    payload["tau_expr_python"] = tau_expr_python
    payload["power"] = gate_info.get("power", 1)
    return payload, issues


def build_one_ion_hh_ohmic_ir(
    canonical_blocks: dict[str, Any],
    mod_file: Path,
    *,
    raw_blocks: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    title = canonical_blocks.get("TITLE", {}).get("text")
    mechanism_name = (canonical_blocks.get("NEURON") or {}).get("suffix") or mod_file.stem
    parameter_lookup = {item["name"]: item for item in canonical_blocks.get("PARAMETER", [])}

    issues: list[str] = []
    unsupported_blocks = collect_unhandled_raw_blocks(raw_blocks or {})

    function_map, function_issues = _build_inline_function_map(canonical_blocks)
    issues.extend(function_issues)
    procedure_env, procedure_issues = _build_procedure_env(canonical_blocks, function_map)
    issues.extend(procedure_issues)
    breakpoint_env = _resolve_breakpoint_assignments(canonical_blocks, function_map)
    current_model, current_issues = _extract_ohmic_current_model(canonical_blocks, breakpoint_env)
    issues.extend(current_issues)

    supported = current_model is not None
    gates: list[dict[str, Any]] = []
    extra_parameters: list[dict[str, Any]] = []
    g_max_param = {
        "target_name": "g_max",
        "safe_name": "g_max",
        "source_name": None,
        "default_expression": "10. * (u.mS / (u.cm ** -2))",
        "unit": "mS/cm2",
    }
    base_class_name = "Channel"
    ion_arg_name = "Ion"
    ion_name = None

    if current_model is not None:
        ion_name = current_model["ion_name"]
        base_class_name = current_model["base_class_name"]
        ion_arg_name = current_model["ion_arg_name"]

        conductance_source = current_model["conductance_source_name"]
        conductance_parameter = parameter_lookup.get(conductance_source, {})
        g_max_param = {
            "target_name": "g_max",
            "safe_name": "g_max",
            "source_name": conductance_source,
            "default_expression": _default_value_expression(
                conductance_parameter.get("value"),
                conductance_parameter.get("unit"),
            ),
            "unit": conductance_parameter.get("unit"),
        }

        derivative_lookup = {}
        for derivative_block in canonical_blocks.get("DERIVATIVE", []):
            for statement in derivative_block.get("statements", []):
                derivative_lookup[statement["assigned_var"]] = statement["expression"]

        for gate_name, power in sorted(current_model["gate_powers"].items()):
            derivative_expression = derivative_lookup.get(gate_name)
            if derivative_expression is None:
                issues.append(f"Gate {gate_name} appears in current but has no DERIVATIVE equation.")
                supported = False
                continue

            gate_kinetics, gate_issues = _extract_gate_kinetics(
                gate_name,
                derivative_expression,
                procedure_env,
                function_map,
                parameter_lookup,
            )
            issues.extend(gate_issues)
            if gate_kinetics is None:
                supported = False
                continue

            gate_kinetics["power"] = power
            gate_payload, payload_issues = _build_gate_helper_payload(gate_kinetics, parameter_lookup)
            issues.extend(payload_issues)
            if payload_issues:
                supported = False
            gates.append(gate_payload)

        gate_param_names: set[str] = set()
        for gate in gates:
            gate_param_names.update(
                name
                for name in _expression_tokens(
                    f"{gate['inf_expression_source']} + {gate['tau_expression_source']}"
                )
                if name in parameter_lookup
            )

        q10_source_name = "q10" if "q10" in parameter_lookup else None
        tref_source_name = "temp" if "temp" in parameter_lookup else None
        v_shift_source_name = next(
            (name for name in parameter_lookup if name.lower() in {"v_sh", "vshift", "v_shift"}),
            None,
        )

        hidden_parameters = {
            conductance_source,
            q10_source_name,
            tref_source_name,
            v_shift_source_name,
        }

        for parameter_name in sorted(gate_param_names):
            if parameter_name in hidden_parameters:
                continue
            parameter = parameter_lookup[parameter_name]
            extra_parameters.append(
                {
                    "source_name": parameter_name,
                    "safe_name": _safe_identifier(parameter_name),
                    "default_expression": _default_value_expression(parameter.get("value"), parameter.get("unit")),
                    "unit": parameter.get("unit"),
                }
            )

        for gate in gates:
            if gate["q10_source"] == "parameter:q10" and q10_source_name is not None:
                q10_parameter = parameter_lookup[q10_source_name]
                gate["q10_expression"] = repr(q10_parameter.get("value"))

    class_prefix = ION_INFO_MAP.get(ion_name or "", {}).get("current_prefix", "I")
    class_name = f"{class_prefix}_{pascal_case(mechanism_name)}"
    supported = supported and not unsupported_blocks and not any(
        issue.startswith("FUNCTION") or issue.startswith("PROCEDURE") or "unsupported" in issue.lower()
        for issue in issues
    )

    if raw_blocks and unsupported_blocks:
        issues.append(
            "Unsupported NMODL blocks are present for one_ion_hh_ohmic: "
            + ", ".join(sorted(unsupported_blocks))
        )

    return {
        "source_file": str(mod_file),
        "title": title,
        "mechanism_name": mechanism_name,
        "class_name": class_name,
        "template_name": "one_ion_hh_ohmic.py",
        "supported": supported,
        "rejection_reasons": [issue for issue in issues if issue not in set(function_issues + procedure_issues)],
        "manual_fix_required": issues,
        "unsupported_blocks": unsupported_blocks,
        "base_class_name": base_class_name,
        "ion_name": ion_name,
        "ion_arg_name": ion_arg_name,
        "g_max_param": g_max_param,
        "v_shift_param": {
            "target_name": "V_sh",
            "safe_name": "V_sh",
            "default_expression": "0. * u.mV",
        },
        "temperature_param": {
            "target_name": "temp",
            "safe_name": "temp",
            "default_expression": "u.celsius2kelvin(23)",
        },
        "tref_expression": "u.celsius2kelvin(23)",
        "extra_parameters": extra_parameters,
        "gates": gates,
        "current_model": current_model,
        "source_summary": {
            "function_names": sorted(function_map),
            "procedure_assignments": procedure_env,
            "breakpoint_assignments": breakpoint_env,
        },
    }


def summarize_one_ion_hh_ohmic_ir(ir: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": ir["source_file"],
        "supported": ir["supported"],
        "class_name": ir["class_name"],
        "base_class_name": ir["base_class_name"],
        "ion_name": ir["ion_name"],
        "g_max_source_name": ir["g_max_param"]["source_name"],
        "gate_summary": [
            {
                "name": gate["name"],
                "power": gate["power"],
                "q10": gate["q10_expression"],
                "source_form": gate["source_form"],
                "inf_source_name": gate.get("inf_source_name"),
                "tau_source_name": gate.get("tau_source_name"),
                "alpha_source_name": gate.get("alpha_source_name"),
                "beta_source_name": gate.get("beta_source_name"),
            }
            for gate in ir.get("gates", [])
        ],
        "current_model": ir.get("current_model"),
        "rejection_reasons": ir.get("rejection_reasons", []),
    }


def run(step1_result: dict[str, Any]) -> dict[str, Any]:
    ir = build_one_ion_hh_ohmic_ir(
        step1_result["canonical_blocks"],
        step1_result["mod_file"],
        raw_blocks=step1_result["raw_blocks"],
    )
    return {
        "variant": VARIANT_NAME,
        "family": VARIANT_NAME,
        "source_file": step1_result["source_file"],
        "supported": ir["supported"],
        "rejection_reasons": ir.get("rejection_reasons", []),
        "summary": summarize_one_ion_hh_ohmic_ir(ir),
        "ir": ir,
    }
