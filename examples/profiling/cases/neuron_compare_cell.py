"""Profiling adapter for ``examples/neuron_compare/cell`` BrainCell models."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class CellExampleSpec:
    module: str
    class_name: str
    params_module: str
    params_loader: str
    morph_attr: str | None
    temperature_celsius: float
    precision: int
    dt_ms: float
    duration_ms: float
    delay_ms: float
    stim_dur_ms: float
    amp_nA: float
    params_need_temperature: bool = False


CELL_SPECS: dict[str, CellExampleSpec] = {
    "bc_ma2025": CellExampleSpec(
        module="examples.neuron_compare.cell.bc_ma2025.bc_braincell",
        class_name="BC",
        params_module="examples.neuron_compare.cell.bc_ma2025.parameters",
        params_loader="load_bc25_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=36.0,
        precision=64,
        dt_ms=0.05,
        duration_ms=50.0,
        delay_ms=10.0,
        stim_dur_ms=80.0,
        amp_nA=0.05,
    ),
    "dcn_su2015": CellExampleSpec(
        module="examples.neuron_compare.cell.dcn_su2015.dcn_braincell",
        class_name="DCN",
        params_module="examples.neuron_compare.cell.dcn_su2015.parameters",
        params_loader="load_dcn15_params",
        morph_attr=None,
        temperature_celsius=32.0,
        precision=64,
        dt_ms=0.1,
        duration_ms=50.0,
        delay_ms=10.0,
        stim_dur_ms=30.0,
        amp_nA=0.1,
        params_need_temperature=True,
    ),
    "goc_ma2020": CellExampleSpec(
        module="examples.neuron_compare.cell.goc_ma2020.goc_braincell",
        class_name="GoC",
        params_module="examples.neuron_compare.cell.goc_ma2020.parameters",
        params_loader="load_goc20_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=34.0,
        precision=32,
        dt_ms=0.01,
        duration_ms=10.0,
        delay_ms=0.0,
        stim_dur_ms=80.0,
        amp_nA=0.2,
    ),
    "grc_ma2020": CellExampleSpec(
        module="examples.neuron_compare.cell.grc_ma2020.grc_braincell",
        class_name="GrC",
        params_module="examples.neuron_compare.cell.grc_ma2020.parameters",
        params_loader="load_grc20_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=25.0,
        precision=64,
        dt_ms=0.1,
        duration_ms=50.0,
        delay_ms=10.0,
        stim_dur_ms=30.0,
        amp_nA=0.01,
    ),
    "grc_ma2020_full": CellExampleSpec(
        module="examples.neuron_compare.cell.grc_ma2020.grc_full_braincell",
        class_name="GrCFull",
        params_module="examples.neuron_compare.cell.grc_ma2020.grc_full_parameters",
        params_loader="load_grc20_full_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=25.0,
        precision=64,
        dt_ms=0.01,
        duration_ms=20.0,
        delay_ms=0.0,
        stim_dur_ms=30.0,
        amp_nA=0.01,
    ),
    "io_zh2019": CellExampleSpec(
        module="examples.neuron_compare.cell.io_zh2019.io_braincell",
        class_name="IO",
        params_module="examples.neuron_compare.cell.io_zh2019.parameters",
        params_loader="load_io19_params",
        morph_attr=None,
        temperature_celsius=36.0,
        precision=64,
        dt_ms=0.1,
        duration_ms=100.0,
        delay_ms=10.0,
        stim_dur_ms=80.0,
        amp_nA=0.05,
    ),
    "pc_ma2024": CellExampleSpec(
        module="examples.neuron_compare.cell.pc_ma2024.pc_braincell",
        class_name="PC",
        params_module="examples.neuron_compare.cell.pc_ma2024.parameters",
        params_loader="load_pc24_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=36.0,
        precision=32,
        dt_ms=0.01,
        duration_ms=10.0,
        delay_ms=5.0,
        stim_dur_ms=10.0,
        amp_nA=0.5,
    ),
    "sc_ma2021": CellExampleSpec(
        module="examples.neuron_compare.cell.sc_ma2021.sc_braincell",
        class_name="SC",
        params_module="examples.neuron_compare.cell.sc_ma2021.parameters",
        params_loader="load_sc21_params",
        morph_attr="DEFAULT_MORPH_PATH",
        temperature_celsius=32.0,
        precision=64,
        dt_ms=0.1,
        duration_ms=100.0,
        delay_ms=10.0,
        stim_dur_ms=80.0,
        amp_nA=0.05,
    ),
}


def add_case_args(parser) -> None:
    """Add ``neuron_compare_cell`` arguments to ``parser``."""
    parser.add_argument("--cell", choices=sorted(CELL_SPECS), default="pc_ma2024")
    parser.add_argument("--precision", type=int, choices=(32, 64), default=None)
    parser.add_argument("--temperature-celsius", type=float, default=None)
    parser.add_argument("--v-init-mv", type=float, default=-65.0)
    parser.add_argument("--population-size", type=int, default=1)
    parser.add_argument("--delay-ms", type=float, default=None)
    parser.add_argument("--stim-dur-ms", type=float, default=None)
    parser.add_argument("--amp-na", type=float, default=None)


def create_workload(args):
    """Create a profiling workload for a BrainCell cell example."""
    return NeuronCompareCellWorkload(args)


class NeuronCompareCellWorkload:
    """BrainCell-only workload matching the notebook cell examples."""

    def __init__(self, args):
        self.args = args
        self.spec = CELL_SPECS[args.cell]
        self.precision = self.spec.precision if args.precision is None else int(args.precision)
        self.temperature_celsius = (
            self.spec.temperature_celsius
            if args.temperature_celsius is None
            else float(args.temperature_celsius)
        )
        self.dt_ms = self.spec.dt_ms if args.dt_ms is None else float(args.dt_ms)
        self.duration_ms = (
            self.spec.duration_ms if args.duration_ms is None else float(args.duration_ms)
        )
        self.delay_ms = self.spec.delay_ms if args.delay_ms is None else float(args.delay_ms)
        self.stim_dur_ms = (
            self.spec.stim_dur_ms if args.stim_dur_ms is None else float(args.stim_dur_ms)
        )
        self.amp_nA = self.spec.amp_nA if args.amp_na is None else float(args.amp_na)
        self.assembly = None
        self.cell = None
        self._u = None
        self._mech = None
        self._at = None
        self._params_mod = None
        self._params = None
        self._model_cls = None

    def build_phases(self):
        """Return named build phases for attribution in the harness."""
        return (
            ("import_case", self.import_case),
            ("load_params", self.load_params),
            ("import_model", self.import_model),
            ("model_build", self.model_build),
            ("place_probes", self.place_probes),
        )

    def build(self) -> None:
        """Build the workload in one call for compatibility."""
        for _, phase in self.build_phases():
            phase()

    def import_case(self) -> None:
        import brainstate
        import brainunit as u
        from braincell import mech
        from braincell.filter import at

        brainstate.environ.set(precision=self.precision)
        self._u = u
        self._mech = mech
        self._at = at
        self._params_mod = import_module(self.spec.params_module)

    def load_params(self) -> None:
        self._require_imports()
        params_loader = getattr(self._params_mod, self.spec.params_loader)
        if self.spec.params_need_temperature:
            self._params = params_loader(temperature_celsius=self.temperature_celsius)
        else:
            self._params = params_loader()

    def import_model(self) -> None:
        model_mod = import_module(self.spec.module)
        self._model_cls = getattr(model_mod, self.spec.class_name)

    def model_build(self) -> None:
        self._require_imports()
        if self._params is None:
            raise RuntimeError("load_params() must be called before model_build().")
        if self._model_cls is None:
            raise RuntimeError("import_model() must be called before model_build().")
        pop_size = () if int(self.args.population_size) == 1 else (int(self.args.population_size),)
        kwargs = {
            "params": self._params,
            "temperature_celsius": self.temperature_celsius,
            "v_init_mV": float(self.args.v_init_mv),
            "pop_size": pop_size,
            "name": f"profile_{self.args.cell}",
        }
        if self.spec.morph_attr is None:
            self.assembly = self._model_cls(**kwargs).build()
        else:
            morph_path = getattr(self._params_mod, self.spec.morph_attr)
            self.assembly = self._model_cls(morph_path, **kwargs).build()

        self.cell = self.assembly.cell

    def place_probes(self) -> None:
        self._require_cell()
        self._require_imports()
        u = self._u
        mech = self._mech
        at = self._at
        self.cell.place(at("soma", 0.5), mech.StateProbe(name="v_soma"))
        self.cell.place(
            at("soma", 0.5),
            mech.CurrentClamp(
                delay=self.delay_ms * u.ms,
                durations=self.stim_dur_ms * u.ms,
                amplitudes=self.amp_nA * u.nA,
            ),
        )

    def init_reset(self) -> None:
        self._require_cell()
        self.cell.init_state()
        self.cell.reset_state()

    def reset_for_run(self) -> None:
        self._require_cell()
        self.cell.reset_state()

    def run(self):
        import brainunit as u

        self._require_cell()
        return self.cell.run(dt=self.dt_ms * u.ms, duration=self.duration_ms * u.ms)

    def block(self, result) -> None:
        _block_until_ready_tree(result)

    def materialize(self, result) -> dict[str, Any]:
        import brainunit as u
        import numpy as np

        time_ms = np.asarray(result.time.to_decimal(u.ms), dtype=float)
        voltage = np.asarray(result.traces["v_soma"].to_decimal(u.mV), dtype=float)
        return {
            "time_shape": list(time_ms.shape),
            "voltage_shape": list(voltage.shape),
            "voltage_mean_mV": float(np.mean(voltage)),
            "voltage_min_mV": float(np.min(voltage)),
            "voltage_max_mV": float(np.max(voltage)),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "cell": self.args.cell,
            "precision": self.precision,
            "dt_ms": self.dt_ms,
            "duration_ms": self.duration_ms,
            "temperature_celsius": self.temperature_celsius,
            "population_size": int(self.args.population_size),
            "delay_ms": self.delay_ms,
            "stim_dur_ms": self.stim_dur_ms,
            "amp_nA": self.amp_nA,
        }

    def _require_cell(self) -> None:
        if self.cell is None:
            raise RuntimeError("build() must be called before this phase.")

    def _require_imports(self) -> None:
        if self._u is None or self._mech is None or self._at is None or self._params_mod is None:
            raise RuntimeError("import_case() must be called before this phase.")


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
