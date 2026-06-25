"""Profiling adapter for the cerebellar probability network notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CellSpec:
    name: str
    class_path: str
    params_loader_path: str
    temperature_celsius: float
    morph_path_attr: str | None = None
    params_need_temperature: bool = False


CELL_SPECS: dict[str, CellSpec] = {
    "GrC": CellSpec(
        "GrC",
        "examples.neuron_compare.cell.grc_ma2020.grc_full_braincell:GrCFull",
        "examples.neuron_compare.cell.grc_ma2020.grc_full_parameters:load_grc20_full_params",
        25.0,
        "examples.neuron_compare.cell.grc_ma2020.parameters:DEFAULT_MORPH_PATH",
    ),
    "GoC": CellSpec(
        "GoC",
        "examples.neuron_compare.cell.goc_ma2020.goc_braincell:GoC",
        "examples.neuron_compare.cell.goc_ma2020.parameters:load_goc20_params",
        34.0,
        "examples.neuron_compare.cell.goc_ma2020.parameters:DEFAULT_MORPH_PATH",
    ),
    "PC": CellSpec(
        "PC",
        "examples.neuron_compare.cell.pc_ma2024.pc_braincell:PC",
        "examples.neuron_compare.cell.pc_ma2024.parameters:load_pc24_params",
        36.0,
        "examples.neuron_compare.cell.pc_ma2024.parameters:DEFAULT_MORPH_PATH",
    ),
    "SC": CellSpec(
        "SC",
        "examples.neuron_compare.cell.sc_ma2021.sc_braincell:SC",
        "examples.neuron_compare.cell.sc_ma2021.parameters:load_sc21_params",
        32.0,
        "examples.neuron_compare.cell.sc_ma2021.parameters:DEFAULT_MORPH_PATH",
    ),
    "BC": CellSpec(
        "BC",
        "examples.neuron_compare.cell.bc_ma2025.bc_braincell:BC",
        "examples.neuron_compare.cell.bc_ma2025.parameters:load_bc25_params",
        36.0,
        "examples.neuron_compare.cell.bc_ma2025.parameters:DEFAULT_MORPH_PATH",
    ),
    "DCN": CellSpec(
        "DCN",
        "examples.neuron_compare.cell.dcn_su2015.dcn_braincell:DCN",
        "examples.neuron_compare.cell.dcn_su2015.parameters:load_dcn15_params",
        32.0,
        None,
        True,
    ),
    "IO": CellSpec(
        "IO",
        "examples.neuron_compare.cell.io_zh2019.io_braincell:IO",
        "examples.neuron_compare.cell.io_zh2019.parameters:load_io19_params",
        36.0,
    ),
}


SCALE_SIZES: dict[str, dict[str, int]] = {
    "tiny": {"GrC": 2, "GoC": 1, "PC": 1, "SC": 1, "BC": 1, "DCN": 1, "IO": 1},
    "small": {"GrC": 64, "GoC": 8, "PC": 4, "SC": 4, "BC": 4, "DCN": 2, "IO": 4},
    "notebook": {
        "GrC": 1024 * 8,
        "GoC": 16 * 8,
        "PC": 36 * 8,
        "SC": 24 * 8,
        "BC": 24 * 8,
        "DCN": 8 * 8,
        "IO": 24 * 8,
    },
}


PROJECTION_CONFIGS = [
    {"pre": "GrC", "post": "GoC", "p": 0.2, "seed": 101, "weight_uS": 0.002, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
    {"pre": "GoC", "post": "GrC", "p": 0.01, "seed": 102, "weight_uS": 0.00014, "delay_ms": 0.5, "e_mV": -75.0, "tau_ms": 2.0},
    {"pre": "GrC", "post": "PC", "p": 0.01, "seed": 103, "weight_uS": 0.006, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
    {"pre": "GrC", "post": "SC", "p": 0.2, "seed": 104, "weight_uS": 0.00045, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
    {"pre": "GrC", "post": "BC", "p": 0.2, "seed": 105, "weight_uS": 0.00035, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
    {"pre": "SC", "post": "PC", "p": 0.01, "seed": 106, "weight_uS": 0.025, "delay_ms": 0.5, "e_mV": -75.0, "tau_ms": 2.0},
    {"pre": "BC", "post": "PC", "p": 0.01, "seed": 107, "weight_uS": 0.02, "delay_ms": 0.5, "e_mV": -75.0, "tau_ms": 2.0},
    {"pre": "PC", "post": "DCN", "p": 0.01, "seed": 108, "weight_uS": 0.006, "delay_ms": 0.5, "e_mV": -75.0, "tau_ms": 2.0},
    {"pre": "IO", "post": "PC", "p": 0.05, "seed": 109, "weight_uS": 0.025, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
    {"pre": "DCN", "post": "IO", "p": 0.05, "seed": 110, "weight_uS": 0.0003, "delay_ms": 0.5, "e_mV": -75.0, "tau_ms": 2.0},
    {"pre": "IO", "post": "DCN", "p": 0.05, "seed": 111, "weight_uS": 0.005, "delay_ms": 0.5, "e_mV": 0.0, "tau_ms": 2.0},
]


def add_case_args(parser) -> None:
    """Add cerebellar probability network arguments to ``parser``."""
    parser.add_argument("--scale", choices=sorted(SCALE_SIZES), default="tiny")
    parser.add_argument(
        "--populations",
        default="all",
        help="Comma-separated population subset, or 'all'. Example: GrC,GoC",
    )
    parser.add_argument("--precision", type=int, choices=(32, 64), default=32)
    parser.add_argument("--event-backend", default="auto")
    parser.add_argument("--brainevent-backend", default="jax_raw")
    parser.add_argument(
        "--spike-recording",
        choices=("none", "population", "full"),
        default="population",
    )
    parser.add_argument("--grc-size", type=int, default=None)
    parser.add_argument("--goc-size", type=int, default=None)
    parser.add_argument("--pc-size", type=int, default=None)
    parser.add_argument("--sc-size", type=int, default=None)
    parser.add_argument("--bc-size", type=int, default=None)
    parser.add_argument("--dcn-size", type=int, default=None)
    parser.add_argument("--io-size", type=int, default=None)


def create_workload(args):
    """Create the cerebellar probability network profiling workload."""
    return CerebellarProbabilityNetworkWorkload(args)


class CerebellarProbabilityNetworkWorkload:
    """Network workload extracted from the cerebellar probability notebook."""

    def __init__(self, args):
        self.args = args
        self.dt_ms = 0.01 if args.dt_ms is None else float(args.dt_ms)
        self.duration_ms = 10.0 if args.duration_ms is None else float(args.duration_ms)
        self.sizes = _sizes_from_args(args)
        self.population_names = _population_names_from_args(args)
        self.populations = {}
        self.net = None
        self.projection_rows: list[dict[str, Any]] = []

    def build(self) -> None:
        import braincell
        import brainstate

        brainstate.environ.set(precision=int(self.args.precision))
        for pop_name in self.population_names:
            spec = CELL_SPECS[pop_name]
            size = int(self.sizes[pop_name])
            self.populations[pop_name] = _build_population(
                spec,
                size=size,
                incoming_configs=_incoming_projection_configs(pop_name, self.population_names),
            )

        self.net = braincell.Network(name="profile_cerebellar_probability_network")
        for pop_name, cell in self.populations.items():
            self.net.add_population(pop_name, cell)
        self.projection_rows = _add_projections(self.net, self.population_names)

    def init_reset(self) -> None:
        self._require_net()
        self.net.reset_state()

    def reset_for_run(self) -> None:
        self._require_net()
        self.net.reset_state()

    def run(self):
        import brainunit as u

        self._require_net()
        return self.net.run(
            dt=self.dt_ms * u.ms,
            duration=self.duration_ms * u.ms,
            event_backend=self.args.event_backend,
            brainevent_backend=self.args.brainevent_backend,
            spike_recording=self.args.spike_recording,
        )

    def block(self, result) -> None:
        _block_until_ready_tree(result)

    def materialize(self, result) -> dict[str, Any]:
        import brainunit as u
        import numpy as np

        time_ms = np.asarray(result.time.to_decimal(u.ms), dtype=float)
        trace_shapes = {
            pop_name: {
                probe_name: list(np.asarray(value.to_decimal(u.mV), dtype=float).shape)
                for probe_name, value in probes.items()
            }
            for pop_name, probes in result.traces.items()
        }
        spike_counts = {
            pop_name: float(np.asarray(spikes, dtype=float).sum())
            for pop_name, spikes in result.spikes.items()
        }
        return {
            "time_shape": list(time_ms.shape),
            "trace_shapes": trace_shapes,
            "spike_counts": spike_counts,
            "n_projection_edges": {
                row["projection"]: row["edges"] for row in self.projection_rows
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "scale": self.args.scale,
            "populations": self.population_names,
            "sizes": self.sizes,
            "precision": int(self.args.precision),
            "dt_ms": self.dt_ms,
            "duration_ms": self.duration_ms,
            "event_backend": self.args.event_backend,
            "brainevent_backend": self.args.brainevent_backend,
            "spike_recording": self.args.spike_recording,
        }

    def _require_net(self) -> None:
        if self.net is None:
            raise RuntimeError("build() must be called before this phase.")


def _build_population(spec: CellSpec, *, size: int, incoming_configs):
    import brainunit as u
    import numpy as np
    from braincell import mech
    from braincell.filter import at

    cls = _load_attr(spec.class_path)
    params_loader = _load_attr(spec.params_loader_path)
    if spec.params_need_temperature:
        params = params_loader(temperature_celsius=spec.temperature_celsius)
    else:
        params = params_loader()

    kwargs = {
        "params": params,
        "temperature_celsius": spec.temperature_celsius,
        "v_init_mV": -65.0,
        "pop_size": (size,),
        "name": f"profile_{spec.name.lower()}_{size}",
    }
    morph_path = _load_attr(spec.morph_path_attr) if spec.morph_path_attr else None
    assembly = cls(morph_path, **kwargs).build() if morph_path is not None else cls(**kwargs).build()
    cell = assembly.cell
    cell.V_th = 0.0 * u.mV
    cell.place(at("soma", 0.5), mech.StateProbe(name="v", field="v"))

    drive = _drive_protocol(spec.name, size)
    if drive is not None:
        cell.place(at("soma", 0.5), drive)

    for cfg in incoming_configs:
        cell.place(
            at("soma", 0.5),
            mech.Synapse(
                "ExpSyn",
                tau=cfg["tau_ms"] * u.ms,
                e=cfg["e_mV"] * u.mV,
                weight=1.0 * u.uS,
                name=_synapse_name(cfg["pre"], cfg["post"]),
            ),
        )
    return cell


def _add_projections(net, population_names: tuple[str, ...]) -> list[dict[str, Any]]:
    import braincell
    import brainunit as u
    import numpy as np

    rows = []
    active = set(population_names)
    for cfg in PROJECTION_CONFIGS:
        pre = cfg["pre"]
        post = cfg["post"]
        if pre not in active or post not in active:
            continue
        edge_name = f"edges_{pre}_to_{post}"
        projection_name = f"proj_{pre}_to_{post}"
        edges = net.add_edges(
            name=edge_name,
            pre=pre,
            post=post,
            method=braincell.network.probability(
                p=float(cfg["p"]), seed=int(cfg["seed"]), allow_self=True
            ),
        )
        if edges.n_edge > 0:
            weights = np.full(edges.n_edge, float(cfg["weight_uS"]), dtype=float) * u.uS
            net.add_projection(
                name=projection_name,
                edges=edge_name,
                synapse=_synapse_name(pre, post),
                weight=weights,
                delay=cfg["delay_ms"] * u.ms,
            )
        rows.append({"projection": f"{pre}->{post}", "edges": int(edges.n_edge)})
    return rows


def _drive_protocol(pop_name: str, size: int):
    import brainunit as u
    import numpy as np
    from braincell import mech

    if pop_name not in {"GrC", "IO"}:
        return None
    amplitudes = u.Quantity(np.full(size, 0.1, dtype=float), u.nA)
    delays = u.Quantity(np.full(size, 0.1, dtype=float), u.ms)
    return mech.CurrentClamp(delay=delays, durations=50.0 * u.ms, amplitudes=amplitudes)


def _incoming_projection_configs(pop_name: str, population_names: tuple[str, ...]):
    active = set(population_names)
    return [
        cfg
        for cfg in PROJECTION_CONFIGS
        if cfg["post"] == pop_name and cfg["pre"] in active
    ]


def _synapse_name(pre: str, post: str) -> str:
    return f"syn_{pre}_to_{post}"


def _sizes_from_args(args) -> dict[str, int]:
    sizes = dict(SCALE_SIZES[args.scale])
    overrides = {
        "GrC": args.grc_size,
        "GoC": args.goc_size,
        "PC": args.pc_size,
        "SC": args.sc_size,
        "BC": args.bc_size,
        "DCN": args.dcn_size,
        "IO": args.io_size,
    }
    for name, value in overrides.items():
        if value is not None:
            sizes[name] = int(value)
    for name, value in sizes.items():
        if value <= 0:
            raise ValueError(f"{name} size must be positive, got {value}.")
    return sizes


def _population_names_from_args(args) -> tuple[str, ...]:
    if args.populations == "all":
        return tuple(CELL_SPECS)
    names = tuple(name.strip() for name in args.populations.split(",") if name.strip())
    if not names:
        raise ValueError("--populations must not be empty.")
    unknown = [name for name in names if name not in CELL_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown population(s): {unknown!r}. Choices are {sorted(CELL_SPECS)!r}."
        )
    return names


def _load_attr(path: str):
    from importlib import import_module

    module_name, attr_name = path.split(":", 1)
    return getattr(import_module(module_name), attr_name)


def _block_until_ready_tree(value) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif hasattr(value, "mantissa"):
        _block_until_ready_tree(value.mantissa)
    elif isinstance(value, dict):
        for item in value.values():
            _block_until_ready_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready_tree(item)
    elif hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            _block_until_ready_tree(getattr(value, field_name))
