

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    snippet: str | None = None


@dataclass(frozen=True)
class ExprAst:
    kind: str
    text: str
    operator: str | None = None
    name: str | None = None
    value: int | float | str | None = None
    args: tuple["ExprAst", ...] = ()
    left: "ExprAst | None" = None
    right: "ExprAst | None" = None
    operand: "ExprAst | None" = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class StatementAst:
    kind: str
    code: str
    assigned_var: str | None = None
    prime_var: str | None = None
    expression: ExprAst | None = None
    name: str | None = None
    args: tuple[ExprAst, ...] = ()
    names: tuple[str, ...] = ()
    target: str | None = None
    method: str | None = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class ParameterAst:
    name: str
    unit: str | None = None
    value: int | float | str | None = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class VariableAst:
    name: str
    unit: str | None = None
    power: int | None = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class UseIonAst:
    ion: str
    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class CallableAst:
    name: str
    params: tuple[ParameterAst, ...] = ()
    locals: tuple[str, ...] = ()
    statements: tuple[StatementAst, ...] = ()
    returned_unit: str | None = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class SolveAst:
    target: str
    method: str | None = None
    source_span: SourceSpan | None = None


@dataclass(frozen=True)
class ModuleAst:
    source_file: str
    title: str | None
    mechanism_name: str
    raw_blocks: dict[str, list[str]]
    canonical_blocks: dict[str, Any]
    comments: tuple[str, ...] = ()
    useions: tuple[UseIonAst, ...] = ()
    ranges: tuple[str, ...] = ()
    globals: tuple[str, ...] = ()
    nonspecific_currents: tuple[str, ...] = ()
    parameters: tuple[ParameterAst, ...] = ()
    assigned: tuple[VariableAst, ...] = ()
    states: tuple[VariableAst, ...] = ()
    initial: tuple[StatementAst, ...] = ()
    breakpoint: tuple[StatementAst, ...] = ()
    solve_blocks: tuple[SolveAst, ...] = ()
    derivative_blocks: dict[str, tuple[StatementAst, ...]] = field(default_factory=dict)
    functions: tuple[CallableAst, ...] = ()
    procedures: tuple[CallableAst, ...] = ()
    unsupported_blocks: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    kind: str
    unit: str | None = None
    default_value: int | float | str | None = None
    declared_in: str | None = None


@dataclass(frozen=True)
class SolveInfo:
    block: str
    target: str
    method: str | None = None


@dataclass(frozen=True)
class CurrentInfo:
    name: str
    expression: str
    ion: str | None = None
    read_var: str | None = None
    conductance_candidates: tuple[str, ...] = ()
    state_factors: dict[str, int] = field(default_factory=dict)
    driving_force: str | None = None


@dataclass(frozen=True)
class GateKinetics:
    state: str
    derivative_expression: str
    source_form: str
    inf_expression: str | None = None
    tau_expression: str | None = None
    alpha_expression: str | None = None
    beta_expression: str | None = None
    phi_expression: str | None = None
    q10_expression: str = "1.0"
    q10_source: str = "default:1.0"


@dataclass(frozen=True)
class SemanticModuleIR:
    source_file: str
    mechanism_name: str
    module_kind: str
    title: str | None
    useions: tuple[UseIonAst, ...]
    symbols: tuple[SymbolInfo, ...]
    states: tuple[VariableAst, ...]
    parameters: tuple[ParameterAst, ...]
    assigned: tuple[VariableAst, ...]
    ranges: tuple[str, ...]
    globals: tuple[str, ...]
    functions: tuple[CallableAst, ...]
    procedures: tuple[CallableAst, ...]
    call_graph: dict[str, tuple[str, ...]]
    procedure_env: dict[str, str]
    breakpoint_assignments: dict[str, str]
    initial_assignments: dict[str, str]
    solve_specs: tuple[SolveInfo, ...]
    currents: tuple[CurrentInfo, ...]
    gate_kinetics: tuple[GateKinetics, ...]
    ion_dependencies: dict[str, dict[str, tuple[str, ...] | str | None]]
    diagnostics: tuple[str, ...]
    unsupported_features: tuple[str, ...]
    source_summary: dict[str, Any]


@dataclass(frozen=True)
class RenderValidation:
    compiled: bool
    imported: bool
    class_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DensityChannelIR:
    source_file: str
    mechanism_name: str
    title: str | None
    class_name: str
    registry_name: str
    target_family: str
    base_class_name: str
    root_type: str | None
    ion_name: str | None
    ion_arg_name: str
    supported: bool
    rejection_reasons: tuple[str, ...]
    unsupported_features: tuple[str, ...]
    diagnostics: tuple[str, ...]
    g_max_param: dict[str, Any]
    v_shift_param: dict[str, Any]
    temperature_param: dict[str, Any]
    tref_expression: str
    extra_parameters: tuple[dict[str, Any], ...]
    gates: tuple[dict[str, Any], ...]
    current_model: dict[str, Any] | None
    helper_methods: tuple[str, ...]
    render_metadata: dict[str, Any]
    semantic_summary: dict[str, Any]
    validation: RenderValidation | None = None


def to_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_payload(item) for item in value]
    return value
