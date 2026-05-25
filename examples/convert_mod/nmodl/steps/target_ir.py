

import keyword
import re
from typing import Any

from .model import DensityChannelIR
from .model import GateKinetics
from .model import ParameterAst
from .model import RenderValidation
from .model import SemanticModuleIR

ION_INFO_MAP = {
    "k": {
        "base_class_name": "PotassiumChannel",
        "root_type": "braincell.ion.Potassium",
        "ion_arg_name": "K",
        "current_prefix": "IK",
    },
    "na": {
        "base_class_name": "SodiumChannel",
        "root_type": "braincell.ion.Sodium",
        "ion_arg_name": "Na",
        "current_prefix": "INa",
    },
    "ca": {
        "base_class_name": "CalciumChannel",
        "root_type": "braincell.ion.Calcium",
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
    "S": "u.siemens",
    "mS": "u.mS",
    "mV": "u.mV",
    "mA": "u.mA",
    "uA": "u.uA",
    "M": "u.mole",
    "mM": "u.mmole",
    "ms": "u.ms",
    "s": "u.second",
    "cm": "u.cm",
    "um": "u.um",
    "degC": "u._celsius",
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


def _expression_tokens(expression: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(expression))


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


def _prepare_python_expression(expression: str, replacements: dict[str, str]) -> str:
    return _replace_identifiers(_rewrite_math_functions(expression), replacements)


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


def _resolve_gate_payload(
    gate: GateKinetics,
    parameter_lookup: dict[str, ParameterAst],
    power: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    diagnostics: list[str] = []
    safe_name = _safe_identifier(gate.state)

    if gate.source_form == "inf_tau":
        inf_expression = gate.inf_expression
        tau_expression = gate.tau_expression
    else:
        if gate.alpha_expression is None or gate.beta_expression is None:
            diagnostics.append(f"Gate {gate.state} is missing alpha/beta expressions.")
            return None, diagnostics
        alpha_plus_beta = f"(({gate.alpha_expression}) + ({gate.beta_expression}))"
        inf_expression = f"({gate.alpha_expression}) / {alpha_plus_beta}"
        tau_expression = f"1 / {alpha_plus_beta}"

    if inf_expression is None or tau_expression is None:
        diagnostics.append(f"Gate {gate.state} is missing inf/tau expressions after normalization.")
        return None, diagnostics

    if gate.phi_expression and gate.phi_expression.strip() not in {"1", "1.0"} and gate.q10_source == "default:1.0":
        tau_expression = f"({tau_expression}) / ({gate.phi_expression})"

    combined_expression = f"{inf_expression} + {tau_expression}"
    used_names = sorted(
        name for name in _expression_tokens(combined_expression) if name not in set(MATH_FUNCTION_MAP) | {"u", "v", "V"}
    )

    replacements = {"v": "V", "V": "V"}
    alias_lines: list[str] = []
    for name in used_names:
        parameter = parameter_lookup.get(name)
        if parameter is None:
            diagnostics.append(f"Gate {gate.state} depends on non-parameter symbol `{name}` after normalization.")
            continue
        safe_param_name = _safe_identifier(name)
        replacements[name] = safe_param_name
        unit_expr = _unit_to_python_expression(parameter.unit)
        if unit_expr and unit_expr != "1":
            alias_lines.append(f"{safe_param_name} = self.{safe_param_name} / ({unit_expr})")
        else:
            alias_lines.append(f"{safe_param_name} = self.{safe_param_name}")

    return {
        "name": gate.state,
        "safe_name": safe_name,
        "power": power,
        "q10_expression": gate.q10_expression,
        "q10_source": gate.q10_source,
        "source_form": gate.source_form,
        "derivative_expression": gate.derivative_expression,
        "helper_alias_lines": alias_lines,
        "inf_expression_source": inf_expression,
        "tau_expression_source": tau_expression,
        "alpha_expression_source": gate.alpha_expression,
        "beta_expression_source": gate.beta_expression,
        "inf_expr_python": _prepare_python_expression(inf_expression, replacements),
        "tau_expr_python": _prepare_python_expression(tau_expression, replacements),
    }, diagnostics


def lower_density_channel_ir(semantic_ir: SemanticModuleIR) -> DensityChannelIR:
    diagnostics = list(semantic_ir.diagnostics)
    rejection_reasons: list[str] = []
    unsupported_features = list(semantic_ir.unsupported_features)

    useions = list(semantic_ir.useions)
    if len(useions) != 1:
        rejection_reasons.append("density channel lowering currently requires exactly one USEION declaration.")
    useion = useions[0] if len(useions) == 1 else None

    current_model = None
    gate_powers: dict[str, int] = {}
    g_max_source_name = None
    ion_name = useion.ion if useion else None
    base_class_name = ION_INFO_MAP.get(ion_name or "", {}).get("base_class_name", "Channel")
    root_type = ION_INFO_MAP.get(ion_name or "", {}).get("root_type")
    ion_arg_name = ION_INFO_MAP.get(ion_name or "", {}).get("ion_arg_name", "Ion")

    if useion is not None:
        if len(useion.read) != 1 or len(useion.write) != 1:
            rejection_reasons.append("density channel lowering requires exactly one READ ion var and one WRITE ion var.")
        else:
            current = next((item for item in semantic_ir.currents if item.name == useion.write[0]), None)
            if current is None:
                rejection_reasons.append(f"No current expression found for ion write var `{useion.write[0]}`.")
            else:
                gate_powers = dict(current.state_factors)
                conductance_candidates = [
                    name
                    for name in current.conductance_candidates
                    if any(parameter.name == name and (parameter.unit or "").endswith("S/cm2") for parameter in semantic_ir.parameters)
                    or any(parameter.name == name and (parameter.unit or "").endswith("mS/cm2") for parameter in semantic_ir.parameters)
                ]
                if len(conductance_candidates) != 1:
                    rejection_reasons.append(
                        f"Expected exactly one conductance parameter candidate, got {conductance_candidates or 'none'}."
                    )
                else:
                    g_max_source_name = conductance_candidates[0]
                if not gate_powers:
                    rejection_reasons.append("Could not identify any gate states in the ion current expression.")
                if current.driving_force is None:
                    rejection_reasons.append("Could not identify an ohmic driving-force factor in the ion current expression.")
                if not rejection_reasons and g_max_source_name is not None:
                    gate_product_parts = []
                    for gate_name, power in sorted(gate_powers.items()):
                        safe_gate_name = _safe_identifier(gate_name)
                        if power == 1:
                            gate_product_parts.append(f"self.{safe_gate_name}.value")
                        else:
                            gate_product_parts.append(f"self.{safe_gate_name}.value ** {power}")
                    current_model = {
                        "ion_name": useion.ion,
                        "ion_arg_name": ion_arg_name,
                        "base_class_name": base_class_name,
                        "write_var": useion.write[0],
                        "read_var": useion.read[0],
                        "conductance_source_name": g_max_source_name,
                        "driving_force": current.driving_force,
                        "gate_powers": gate_powers,
                        "gate_product_expression": " * ".join(gate_product_parts),
                        "current_expression_python": (
                            f"self.g_max * {' * '.join(gate_product_parts)} * ({ion_arg_name}.E - V)"
                        ),
                    }

    parameter_lookup = {parameter.name: parameter for parameter in semantic_ir.parameters}
    gate_lookup = {gate.state: gate for gate in semantic_ir.gate_kinetics}
    gates: list[dict[str, Any]] = []
    for state_name, power in sorted(gate_powers.items()):
        gate = gate_lookup.get(state_name)
        if gate is None:
            rejection_reasons.append(f"State `{state_name}` appears in current but has no recognized HH kinetics.")
            continue
        gate_payload, gate_diagnostics = _resolve_gate_payload(gate, parameter_lookup, power)
        diagnostics.extend(gate_diagnostics)
        if gate_payload is None:
            rejection_reasons.extend(gate_diagnostics)
            continue
        gates.append(gate_payload)

    extra_parameter_names: set[str] = set()
    for gate in gates:
        extra_parameter_names.update(
            name
            for name in _expression_tokens(f"{gate['inf_expression_source']} + {gate['tau_expression_source']}")
            if name in parameter_lookup
        )

    hidden_parameters = {g_max_source_name, "temp", "q10", "v_sh", "vshift", "v_shift"}
    extra_parameters = tuple(
        {
            "source_name": name,
            "safe_name": _safe_identifier(name),
            "default_expression": _default_value_expression(parameter_lookup[name].value, parameter_lookup[name].unit),
            "unit": parameter_lookup[name].unit,
        }
        for name in sorted(extra_parameter_names)
        if name not in hidden_parameters
    )

    g_max_parameter = parameter_lookup.get(g_max_source_name) if g_max_source_name else None
    g_max_param = {
        "target_name": "g_max",
        "safe_name": "g_max",
        "source_name": g_max_source_name,
        "default_expression": _default_value_expression(
            g_max_parameter.value if g_max_parameter else 0.0,
            g_max_parameter.unit if g_max_parameter else "mS/cm2",
        ),
        "unit": g_max_parameter.unit if g_max_parameter else "mS/cm2",
    }

    target_family = "hh_ohmic_inf_tau"
    if any(gate["source_form"] == "alpha_beta" for gate in gates):
        target_family = "hh_ohmic_alpha_beta"

    class_prefix = ION_INFO_MAP.get(ion_name or "", {}).get("current_prefix", "I")
    mechanism_suffix = re.sub(r"[^0-9A-Za-z]+", " ", semantic_ir.mechanism_name).title().replace(" ", "") or "Mechanism"
    class_name = f"{class_prefix}_{mechanism_suffix}"
    supported = not rejection_reasons and not unsupported_features
    if unsupported_features:
        rejection_reasons.append(
            "Unsupported features for density channel lowering: " + ", ".join(sorted(dict.fromkeys(unsupported_features)))
        )

    return DensityChannelIR(
        source_file=semantic_ir.source_file,
        mechanism_name=semantic_ir.mechanism_name,
        title=semantic_ir.title,
        class_name=class_name,
        registry_name=class_name,
        target_family=target_family,
        base_class_name=base_class_name,
        root_type=root_type,
        ion_name=ion_name,
        ion_arg_name=ion_arg_name,
        supported=supported,
        rejection_reasons=tuple(rejection_reasons),
        unsupported_features=tuple(sorted(dict.fromkeys(unsupported_features))),
        diagnostics=tuple(diagnostics),
        g_max_param=g_max_param,
        v_shift_param={
            "target_name": "V_sh",
            "safe_name": "V_sh",
            "default_expression": "0. * u.mV",
        },
        temperature_param={
            "target_name": "temp",
            "safe_name": "temp",
            "default_expression": "u.celsius2kelvin(23)",
        },
        tref_expression="u.celsius2kelvin(23)",
        extra_parameters=extra_parameters,
        gates=tuple(gates),
        current_model=current_model,
        helper_methods=("_q10", "_to_decimal_if_possible", "_register_generated_channel"),
        render_metadata={
            "template_name": "density_channel.py",
            "target_family": target_family,
        },
        semantic_summary=semantic_ir.source_summary,
        validation=None,
    )


def summarize_density_channel_ir(ir: DensityChannelIR) -> dict[str, Any]:
    return {
        "source_file": ir.source_file,
        "supported": ir.supported,
        "class_name": ir.class_name,
        "registry_name": ir.registry_name,
        "base_class_name": ir.base_class_name,
        "ion_name": ir.ion_name,
        "target_family": ir.target_family,
        "g_max_source_name": ir.g_max_param["source_name"],
        "gate_summary": [
            {
                "name": gate["name"],
                "power": gate["power"],
                "q10": gate["q10_expression"],
                "source_form": gate["source_form"],
            }
            for gate in ir.gates
        ],
        "current_model": ir.current_model,
        "rejection_reasons": list(ir.rejection_reasons),
    }


def attach_validation(ir: DensityChannelIR, validation: RenderValidation) -> DensityChannelIR:
    return DensityChannelIR(**{**ir.__dict__, "validation": validation})
