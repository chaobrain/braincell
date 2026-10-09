# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DIF Cell integration and calibrated event-count routes."""

from __future__ import annotations

import weakref

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import validate_time_quantity
from braincell.reduction.core import ReductionModel, ReductionOutput
from braincell.reduction.dif.model_utils import DIFInputRuntime, VoltageRecorder
from braincell.reduction.dif.tables import DIFTable, load_table
from braincell.reduction.runtime_utils import _build_reduction_context


class DIFReduction(ReductionModel):
    """Run a calibrated DIF model through a BrainCell Cell.

    Parameters
    ----------
    table : str, pathlib.Path or DIFTable, optional
        Existing calibration to reuse. When omitted, Network initialization
        calibrates the completed declarations and retains the table in memory.
    high_order : bool, optional
        Enable the backend's adopted high-order compensation.
    block_threads : int, optional
        CUDA block size, a positive multiple of 32 up to 1024.
    gpu_heap_bytes : int, optional
        Virtual event-arena capacity in bytes; pages are committed on demand.

    Notes
    -----
    ``Cell.use_model("dif")`` selects automatic calibration with defaults.
    Register an instance with ``Cell.add_reduction`` only to customize settings
    or reuse a table. Inputs are declared Connections with calibrated weights.
    Dynamic bound synapse drives and Cell batch execution are unsupported.
    FP64 must be enabled before init.

    The default Network event backend uses DIF's private input queues.
    Connections to detailed Cells retain their ordinary delivery path.

    ``Cell.outputs['voltage_online']`` and ``Cell.outputs['voltage']`` contain
    the current provisional voltage. ``observe.output('voltage')`` recordings
    return finalized trajectories after candidate cancellation; an unresolved
    tail is held until the next run. ``observe.output('voltage_online')`` keeps
    the original per-step samples. Spike events are emitted only on commitment.
    """

    def __init__(self, table=None, *, high_order=True, block_threads=256, gpu_heap_bytes=16 * 1024**3):
        if (
            isinstance(block_threads, bool)
            or not isinstance(block_threads, int)
            or not 32 <= block_threads <= 1024
            or block_threads % 32
        ):
            raise ValueError("block_threads must be a multiple of 32 between 32 and 1024.")
        if isinstance(gpu_heap_bytes, bool) or not isinstance(gpu_heap_bytes, int) or gpu_heap_bytes <= 0:
            raise ValueError("gpu_heap_bytes must be a positive integer.")
        self.table = table if table is None or isinstance(table, DIFTable) else load_table(table)
        self.high_order = bool(high_order)
        self.block_threads = block_threads
        self.gpu_heap_bytes = gpu_heap_bytes
        self._owner_ref = None
        self._runtime = None
        self._context = None
        self._inputs = None
        self._recorder = None

    def prepare(self, cell, *, dt):
        """Calibrate once from the complete network, then restore its declarations."""
        if self.table is not None:
            return
        from braincell.reduction.dif.calibration import calibrate

        # 1. Resolve the completed network and calibration timestep.
        network = cell.network_owner
        if network is None:
            raise RuntimeError("Automatic DIF calibration requires adding the Cell to a Network before initialization.")
        if dt is None:
            raise ValueError(
                "Automatic DIF calibration needs dt; use Network.run(dt=..., duration=...) or init_state(dt=...)."
            )
        validate_time_quantity(dt, name="dt", prefix="DIF preparation")
        if not jax.config.x64_enabled:
            raise ValueError(
                "DIF calibration requires FP64; set brainstate.environ precision=64 before initialization."
            )
        populations = network._cell_populations()
        if any(owner.cell._initialized for owner in populations.values()):
            raise RuntimeError("Automatic DIF calibration must start before any population Cell is initialized.")
        population = next(name for name, owner in populations.items() if owner.cell is cell)
        # 2. Preserve selections, runner caches and the random state.
        selections = tuple((owner.cell, owner.cell._selected_model_name) for owner in populations.values())
        cache_names = ("_run_setup_cache", "_network_run_loop_cache", "_delivery_state_cache")
        saved = {
            name: getattr(network, name)
            for name in (*cache_names, "_initialized", "_runtime_config", "_scheduled_dt_ms", "_source_current_time")
        }
        random_key = brainstate.random.get_key()
        # 3. Run the declared network with detailed dynamics and build its DIF table.
        try:
            for member, _ in selections:
                member.use_model("detailed")
            for name in cache_names:
                setattr(network, name, {})
            with brainstate.environ.context(dt=dt):
                table = calibrate(network, population=population, dt=dt)
        finally:
            # 4. Drop temporary detailed runtimes and restore the selected models.
            try:
                with network._cell_lifecycle():
                    for member, _ in selections:
                        if member._initialized:
                            member.reset()
            finally:
                for name, value in saved.items():
                    setattr(network, name, value)
                for member, selection in selections:
                    member.use_model(selection)
                brainstate.random.seed(random_key)
        self.table = table

    def build_input_runtime(self, cell):
        """Bind the Cell's declared connections to calibrated count slots."""
        owner = None if self._owner_ref is None else self._owner_ref()
        if owner is not None and owner is not cell:
            raise ValueError("One reduction instance belongs to one Cell; create a separate instance for another Cell.")
        if cell._synapse_input_bindings:
            raise NotImplementedError(
                "Calibrated reductions require Connection inputs with explicit calibrated weights."
            )
        if not jax.config.x64_enabled:
            raise ValueError(
                "Calibrated reductions require FP64; set brainstate.environ precision=64 before initialization."
            )
        self._owner_ref = weakref.ref(cell)
        self._inputs = DIFInputRuntime(_build_reduction_context(cell), self.table)
        return self._inputs

    def init_state(self, context, batch_size=None):
        """Allocate one CUDA population and publish initial voltage and spikes."""
        if batch_size is not None:
            raise NotImplementedError("Calibrated reductions do not support Cell batch execution.")
        from braincell.reduction.dif.runtime import CudaPopulationRuntime

        self._context = context
        self._runtime = CudaPopulationRuntime(
            self.table,
            context.pop_size,
            schedules=self._inputs.schedules,
            block_threads=self.block_threads,
            gpu_heap_bytes=self.gpu_heap_bytes,
            high_order=self.high_order,
        )
        self._inputs.runtime = self._runtime
        self._recorder = VoltageRecorder(self._runtime, self.table.dt_ms)
        shape = self._context.pop_size
        voltage = jnp.full(shape, self.table.initial_voltage, dtype=jnp.float64) * u.mV
        return ReductionOutput({"voltage": voltage, "voltage_online": voltage}, jnp.zeros(shape, dtype=jnp.int32))

    def update(self, inputs):
        """Consume calibrated counts and advance exactly one stored timestep."""
        dt = brainstate.environ.get_dt()
        if not np.isclose(float(dt.to_decimal(u.ms)), self.table.dt_ms, rtol=0.0, atol=1e-12):
            raise ValueError("The simulation dt must equal the calibration timestep.")
        counts = inputs.groups[0].payload
        voltage, spike = self._runtime.advance(jnp.asarray(counts, dtype=jnp.int64))
        voltage = voltage.reshape(self._context.pop_size) * u.mV
        return ReductionOutput(
            {"voltage": voltage, "voltage_online": voltage}, spike.reshape(self._context.pop_size).astype(jnp.int32)
        )

    def prepare_recording(self, schema, *, dt):
        """Prepare finalized voltage sampling for the public recording schema."""
        return self._recorder.prepare(schema)

    def reset_state(self, batch_size=None):
        """Reset dynamic GPU state while retaining the bound input layout."""
        if batch_size is not None:
            raise NotImplementedError("Calibrated reductions do not support Cell batch execution.")
        self._runtime.reset()
        self._recorder.reset()
        shape = self._context.pop_size
        voltage = jnp.full(shape, self.table.initial_voltage, dtype=jnp.float64) * u.mV
        return ReductionOutput({"voltage": voltage, "voltage_online": voltage}, jnp.zeros(shape, dtype=jnp.int32))

    def reset(self):
        """Release dynamic resources and retain the calibration and settings."""
        if self._runtime is not None:
            self._runtime._state.close()
        self._runtime = None
        self._context = None
        self._inputs = None
        self._recorder = None
