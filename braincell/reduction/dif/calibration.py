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

"""Capture a detailed workload and calibrate a DIF table in one flow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import brainstate
import brainunit as u
import jax
import numpy as np

from braincell.network.recording import RecordingSpec, observe
from braincell.reduction.dif.calibration_utils import (
    CLAMP_PREPARATION_MS,
    SINGLE_PROBE_MS,
    CalibrationExecution,
    CalibrationLayout,
    CalibrationPlan,
    PopulationCalibrationPlan,
    _arrival_steps,
    _root_cv,
    anchors,
    collect_connections,
    connected_synapses,
    integration_conductance,
    interpolate_single,
    invert_single,
    normalize_calibration_timing,
    normalize_timing,
    pack,
    replay,
    segment_nodes,
    waveform,
)
from braincell.reduction.dif.parameters import extract_parameters
from braincell.reduction.dif.responses import measure_responses
from braincell.reduction.dif.responses_utils import _source_instance_value
from braincell.reduction.dif.tables import ResponseSlot
from braincell.reduction.dif.tables_utils import build_table, synapse_identities
from braincell.reduction.runtime_utils import synapse_records

__all__ = ["CalibrationExecution", "calibrate"]


def calibrate(
    network,
    *,
    population,
    dt,
    output=None,
    duration=None,
    voltage=None,
    arrival_steps=None,
    execution=None,
    time_grid=None,
    canonical_current_delay=20.0 * u.ms,
    canonical_current_duration=0.3 * u.ms,
    canonical_current_amplitude=6.125 * u.nA,
):
    """Run detailed preparation and generate one DIF table.

    Parameters
    ----------
    network : braincell.Network
        Declared network with the detailed model selected. It is initialized
        and reset here so preparation starts from the declared state.
    population : str
        Representative Cell population in the network.
    duration : brainunit.Quantity, optional
        Duration of the declared detailed workload, captured automatically.
        Defaults to 30 seconds for automatic capture. When reusing voltage
        and arrival_steps, the trace length determines the duration.
    voltage : brainunit.Quantity, optional
        One representative neuron's soma voltage after each workload step,
        with voltage units. Use the same declared detailed dynamics and inputs
        as the network. No captured Python objects or analysis pickle is needed.
    arrival_steps : array_like of int, optional
        Zero-based input arrival steps for that representative neuron, including
        connection delays. Repeated arrivals are allowed.
    dt : brainunit.Quantity
        Workload and calibration timestep.
    output : str or pathlib.Path, optional
        New output directory outside the source tree for exporting table.npz.
        When omitted, return the calibrated table directly in memory.
    execution : CalibrationExecution, optional
        Logical GPUs and condition batch size. Enable BrainState FP64 first.
    time_grid : dict, optional
        Explicit ``grid_ms`` and ``query_cutoff_ms`` for a non-Exp2 waveform.
        Exp2 grids are derived from the declared rise and decay constants.
    canonical_current_delay, canonical_current_duration : brainunit.Quantity
        Timing of the canonical somatic pulse; defaults are 20 and 0.3 ms.
    canonical_current_amplitude : brainunit.Quantity
        Canonical pulse amplitude; default is 6.125 nA.

    Returns
    -------
    DIFTable or pathlib.Path
        In-memory calibration, or the exported table path when output is given.

    Notes
    -----
    One call runs the workload, triggers a canonical AP, captures complete
    post-spike states, and fits the REST/POST/rebase conductance banks directly
    from detailed response measurements. Supplying an existing workload trace
    is optional and skips only the first simulation, not canonical state
    capture. No Voltage model table or separate conversion step is required.
    """
    # 1. Capture the declared workload, or reuse its post-step voltage and arrivals.
    execution = CalibrationExecution() if execution is None else execution
    if (voltage is None) != (arrival_steps is None):
        raise ValueError("Supply both voltage and arrival_steps, or neither for automatic capture.")
    if voltage is None and duration is None:
        duration = 30.0 * u.second
    bundle = None if output is None else Path(output)
    if bundle is not None:
        bundle.mkdir(parents=True, exist_ok=False)
    if voltage is None:
        print("[dif-calibration] capturing detailed workload", flush=True)
        with jax.default_device(execution.devices[0]):
            voltage, arrival_steps = capture_workload(
                network,
                population=population,
                dt=dt,
                duration=duration,
            )
    voltage = np.asarray(voltage.to_decimal(u.mV), dtype=np.float64)
    if voltage.ndim not in (1, 2) or (voltage.ndim == 2 and voltage.shape[1] != 1):
        raise ValueError("Provide one representative neuron's post-step soma trace.")
    voltage = voltage.reshape(-1)
    trace_duration = voltage.size * dt
    if duration is not None and not np.isclose(
        float(duration.to_decimal(u.ms)),
        float(trace_duration.to_decimal(u.ms)),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Workload duration must match the captured number of steps times dt.")
    duration = trace_duration
    dt = float(dt.to_decimal(u.ms))
    # 2. Resolve physical input slots and their shared interpolation time grid.
    network.init_state()
    network.reset_state()
    cell = network.populations[population].cell
    plan, calplan = analyze_network(
        network,
        dt=dt * u.ms,
        duration=duration,
        clamp_duration=CLAMP_PREPARATION_MS * u.ms,
        population_names=(population,),
    )
    inputs = calplan.populations[0]
    reversals, cutoffs, node_sets = [], [], []
    groups = {}
    supplied = time_grid
    for slot, (owner, layout_id, index) in enumerate(
        zip(inputs.input_cells, inputs.input_layout_id, inputs.input_synapse_index)
    ):
        layout = next(layout for layout, _ in owner.runtime.iter_synapse_layouts() if layout.id == int(layout_id))
        synapse = owner.runtime.get_runtime_node(int(layout_id))

        def parameter(name, unit):
            if not hasattr(synapse, name):
                raise ValueError(
                    f'Slot {slot} requires parameter {name!r}; provide an explicit time grid for non-Exp2 waveforms.'
                )
            value = _source_instance_value(
                getattr(synapse, name), n_active=int(layout.n_active), synapse_index=int(index)
            )
            return float(np.asarray(value.to_decimal(unit)))

        reversals.append(parameter('e', u.mV))
        if supplied is None:
            key = (parameter('tau1', u.ms), parameter('tau2', u.ms))
            if key not in groups:
                groups[key] = select(*key, dt, [])
            curve = groups[key]
            curve['roles'].append(f'slot{slot}')
            if curve['T_ms'] > SINGLE_PROBE_MS:
                raise ValueError('The synapse query horizon exceeds the calibrated response horizon.')
            cutoffs.append(int(round(curve['T_ms'] / dt)))
            node_sets.extend(curve['grid_ms'])
        else:
            node_sets.extend(supplied['grid_ms'])
            cutoffs.append(int(round(supplied['query_cutoff_ms'] / dt)))
    # A single union grid makes one response measurement sufficient for every
    # waveform type. Interval, event age and release age use that same grid.
    grid = np.unique(np.r_[node_sets, SINGLE_PROBE_MS])
    timing = normalize_timing(dt=dt * u.ms, duration=duration, calibration_time_grid=grid * u.ms)
    plan = replace(plan, timing=timing)
    calplan = replace(
        calplan,
        timing=normalize_calibration_timing(
            clamp_duration=CLAMP_PREPARATION_MS * u.ms,
            timing=timing,
        ),
    )
    # 3. Fit the candidate threshold and trigger the canonical AP/POST trajectory.
    print("[dif-calibration] extracting parameters and capturing canonical POST states", flush=True)
    analysis, template = extract_parameters(
        cell,
        voltage,
        arrival_steps,
        plan,
        device=execution.devices[0],
        canonical_current_delay=canonical_current_delay,
        canonical_current_duration=canonical_current_duration,
        canonical_current_amplitude=canonical_current_amplitude,
    )
    # 4. Measure single/pair responses from REST, POST and rebase states.
    print("[dif-calibration] building REST, POST and rebase response banks", flush=True)
    measurements, preparation = measure_responses(
        network,
        plan,
        calplan,
        analysis=analysis,
        execution=execution,
    )
    measurements = template.apply(measurements)
    # 5. Fit the effective leak from REST relaxation, then invert the response banks.
    baseline = np.asarray(measurements.baseline_voltage_mV)[0]
    rest = float(np.asarray(measurements.V_rest_mV)[0])
    rates = []
    steps = min(int(round(10.0 / dt)), baseline.shape[-1] - 1)
    t = np.arange(steps + 1) * dt
    for index, voltage in enumerate(np.asarray(measurements.rest_voltage_grid_mV)):
        if 0.5 <= abs(voltage - rest) <= 1.5:
            y = baseline[index, : steps + 1] - rest
            mask = np.abs(y) > 1e-3
            if np.count_nonzero(mask) >= 2:
                rates.append(float(-np.polyfit(t[mask], np.log(np.abs(y[mask])), 1)[0]))
    if not rates or not np.all(np.isfinite(rates)) or np.mean(rates) <= 0:
        raise ValueError('REST preparation does not determine a finite positive effective leak.')
    print("[dif-calibration] fitting conductance and current response banks", flush=True)
    arrays = fit_conductance(
        measurements,
        preparation,
        reversals=reversals,
        leak=np.mean(rates),
        query_support_steps=cutoffs,
    )
    # 6. Retain the table in memory for automatic preparation, or export it.
    if bundle is None:
        return build_table(arrays)
    table_path = bundle / 'table.npz'
    temporary = table_path.with_suffix('.npz.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez(stream, **arrays)
        temporary.replace(table_path)
    finally:
        temporary.unlink(missing_ok=True)
    return table_path


def capture_workload(network, *, population, dt, duration):
    """Run the declared workload and return post-update soma voltage and arrivals.

    The representative is the root CV of member zero of the selected detailed
    population, matching canonical and response-bank calibration.
    Observer declarations and compiled observer caches are restored afterward;
    dynamic network state is reset before and after this preparation run.
    """
    # 1. Start the detailed network from its declared state.
    cell = network.populations[population].cell
    if cell._uses_reduction:
        raise ValueError("Calibration workload capture requires the representative Cell's detailed model.")
    network.init_state()
    network.reset_state()
    root_cv = _root_cv(cell)
    cells = tuple(owner.cell for owner in network._cell_populations().values())
    # 2. Preserve user observers and install one representative soma recording.
    saved_recordings = [(owner, owner._recording_specs, owner._compiled_recording_cache) for owner in cells]
    cache_names = ("_run_setup_cache", "_network_run_loop_cache", "_delivery_state_cache")
    saved_caches = {name: getattr(network, name) for name in cache_names}
    saved_config = network._runtime_config, network._scheduled_dt_ms
    random_key = brainstate.random.get_key()
    name = "response_calibration_voltage"
    try:
        for owner in cells:
            owner._recording_specs = {}
            owner._compiled_recording_cache = {}
        # Temporary observers also work for an already initialized network.
        # They do not alter the declared dynamics or user's recording objects.
        cell._recording_specs[name] = RecordingSpec(
            name=name,
            scope=cell[0].cv.by_id([root_cv])._scope,
            observable=observe.state("v"),
            period=dt,
        )
        for key in cache_names:
            setattr(network, key, {})
        network._runtime_config = None
        # 3. Run the workload and align voltage samples with actual input arrivals.
        result = network.run(dt=dt, duration=duration)
        pre = np.asarray(result.samples[population][name].values.to_decimal(u.mV)).reshape(-1)
        final = np.asarray(cell.V.value[..., root_cv].to_decimal(u.mV)).reshape(-1)[0]
        # State observers sample before an update. Parameter extraction needs
        # the voltage after each update, including the final unrecorded state.
        voltage = np.concatenate((pre[1:], [final])) * u.mV
        arrivals = _arrival_steps(network, population, result, dt)
        return voltage, arrivals
    finally:
        # 4. Restore declarations, caches and the random state even on failure.
        try:
            network.reset_state()
        finally:
            for owner, specs, cache in saved_recordings:
                owner._recording_specs = specs
                owner._compiled_recording_cache = cache
            for key, value in saved_caches.items():
                setattr(network, key, value)
            network._runtime_config, network._scheduled_dt_ms = saved_config
            brainstate.random.seed(random_key)


def analyze_network(
    network, *, dt, duration, clamp_duration, calibration_time_grid=None, population_names=None, connections=None
):
    """Build one homogeneous class table from the union of actual member inputs."""
    # 1. Read the completed connections and resolve the calibration clock.
    timing = normalize_timing(dt=dt, duration=duration, calibration_time_grid=calibration_time_grid)
    calibration_timing = normalize_calibration_timing(clamp_duration=clamp_duration, timing=timing)
    if connections is None:
        connections = collect_connections(network.populations)
    populations = network._cell_populations()
    if population_names is not None:
        populations = {name: populations[name] for name in population_names}
    active = connected_synapses(connections)

    # 2. Collect physical synapse/weight slots and the pairs used by each member.
    endpoints, members = {}, {}
    for name, owner in populations.items():
        cell = owner.cell
        records = {item.id: item for item in synapse_records(cell)}
        identities = synapse_identities(item for item in records.values() if item.id in active.get(name, ()))
        for connection in connections:
            if connection.post_population != name:
                continue
            for ordinal, logical_id in enumerate(connection.synapse_id):
                weight = None if connection.weight is None else connection.weight[ordinal]
                logical_id = int(logical_id)
                if logical_id not in identities:
                    continue
                slot = ResponseSlot.from_input(identities[logical_id], weight)
                if slot.magnitude == 0:
                    continue
                endpoints.setdefault(slot, (cell, logical_id))
                members.setdefault((name, records[logical_id].population_index), set()).add(slot)
    slots = tuple(sorted(endpoints))
    if not slots:
        raise ValueError("A calibration class must have nonzero connected inputs.")
    slot_ids = {slot: index for index, slot in enumerate(slots)}
    instances = {
        identity: index for index, identity in enumerate(sorted({(slot.signature, slot.instance) for slot in slots}))
    }
    member_inputs = {frozenset(slot_ids[slot] for slot in member) for member in members.values()}
    pairs = sorted({(old, new) for member in member_inputs for old in member for new in member})

    # 3. Bind every slot to its detailed input and pack the measurement plan.
    input_cells, layouts, rows = [], [], []
    for slot in slots:
        cell, logical_id = endpoints[slot]
        store = cell._get_synapse_store()
        row = store.row_indices([logical_id])[0]
        input_cells.append(cell)
        layouts.append(store.layout_id(str(store.synapse_type[row])))
        rows.append(store.runtime_rows([logical_id])[0])
    plan = PopulationCalibrationPlan(
        input_layout_id=np.asarray(layouts, dtype=np.int32),
        input_synapse_index=np.asarray(rows, dtype=np.int32),
        input_instance_index=np.asarray([instances[(slot.signature, slot.instance)] for slot in slots], dtype=np.int32),
        input_cells=tuple(input_cells),
        input_event_weights=tuple(slot.payload() for slot in slots),
    )
    representative = next(iter(populations))
    layout = CalibrationLayout(
        (representative,),
        np.asarray([0, len(slots)], dtype=np.int32),
        np.asarray([0, len(pairs)], dtype=np.int32),
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        slots,
        timing,
        float(network._common_start_time(tuple(populations)).to_decimal(u.ms)),
    )
    return layout, CalibrationPlan(calibration_timing, (plan,))


INTERPOLATION_FRACTION = 0.03


def select(rise, decay, dt, roles):
    if dt <= 0:
        raise ValueError('The timestep must be positive.')
    evaluate = waveform(rise, decay)
    peak_node, end, peak_time, exact_end = anchors(rise, decay, dt)
    samples = evaluate(np.arange(int(round(end / dt)) + 1) * dt)
    for size in range(3, 41):
        row = None
        for rise_intervals in range(1, size - 1):
            decay_intervals = size - 1 - rise_intervals
            rising = segment_nodes(0.0, peak_node, rise_intervals, evaluate, dt)
            falling = segment_nodes(peak_node, end, decay_intervals, evaluate, dt)
            if rising is None or falling is None:
                continue
            ticks = np.r_[rising, falling[1:]]
            predicted = np.interp(np.arange(len(samples)), ticks, samples[ticks])
            error = float(np.max(abs(predicted - samples)))
            if row is None or error < row['error_peak_fraction']:
                row = dict(
                    node_count=size,
                    grid_ms=np.round(ticks * dt, 9).tolist(),
                    error_peak_fraction=error,
                    rise_intervals=rise_intervals,
                    decay_intervals=decay_intervals,
                )
        if row is not None and row['error_peak_fraction'] <= INTERPOLATION_FRACTION:
            return dict(
                roles=roles,
                tau_rise_ms=rise,
                tau_decay_ms=decay,
                dt_ms=dt,
                T_ms=end,
                T_exact_ms=exact_end,
                peak_node_ms=peak_node,
                peak_exact_ms=peak_time,
                terminal_peak_fraction=float(samples[-1]),
                **row,
            )
    raise ValueError('No curvature grid with at most 40 nodes meets the waveform error limit.')


def fit_conductance(measurements, preparation, *, reversals, leak, query_support_steps):
    """Fit DIF arrays from in-memory detailed measurements."""
    # 1. Resolve the measured banks, physical reversals and shared time axes.
    raw = measurements.as_arrays()
    rest = preparation['rest']
    post = preparation['post']
    reversals = np.asarray(reversals, dtype=np.float64)
    lam = float(leak)
    dt = float(raw['dt_ms'])
    nloc = len(reversals)
    if reversals.shape != (len(raw['slot_signature']),) or not np.all(np.isfinite(reversals)):
        raise ValueError('Reversal potentials must match every calibrated slot.')
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('The calibrated leak rate must be finite and positive.')
    pairs = raw['pair_slots']
    count = len(raw['tau_steps'])
    for axis in ('event_age_steps', 'release_age_steps'):
        np.testing.assert_array_equal(raw[axis], raw['tau_steps'])
    T = rest['baseline_voltage_mV'].shape[-1]
    cutoffs = np.asarray(query_support_steps, dtype=np.int32)
    if cutoffs.shape != (nloc,) or np.any(cutoffs <= 0) or np.any(cutoffs > raw['tau_steps'][-1]):
        raise ValueError('Every slot requires a valid query-age cutoff.')
    t = dict(raw)
    t['dif_axes_steps'] = np.tile(raw['tau_steps'], (nloc, 1))
    t['dif_query_support_steps'] = cutoffs

    # 2. Invert fresh REST/POST singles and replay each fit against its voltage.
    # Fresh singles retain their original 200-240 ms response supports.
    for phase, key, ptrkey, supportkey, bases in (
        (
            'rest',
            'single_voltage_mV',
            'single_voltage_ptr',
            'rest_single_support_steps',
            rest['baseline_voltage_mV'][0],
        ),
        (
            'post',
            'post_single_voltage_mV',
            'post_single_voltage_ptr',
            'post_single_support_steps',
            post['post_baseline_voltage_mV'],
        ),
    ):
        G = np.zeros_like(t[key])
        worst = 0.0
        for loc in range(nloc):
            ptr = raw[ptrkey]
            a, b = map(int, ptr[loc : loc + 2])
            L = int(t[supportkey][loc]) + 1
            values = raw[key][a:b].reshape(len(bases), L)
            g = invert_single(values, bases[:, :L], reversals[loc], lam, dt)
            source = g * (reversals[loc] - bases[:, :L])
            simulated = replay(g, source, values[:, 0], lam, dt)
            error = np.abs(simulated - values)
            if not np.isfinite(error).all():
                raise ValueError(f'{phase} single replay has non-finite samples.')
            worst = max(worst, float(np.max(error)))
            G[a:b] = g.ravel()
        assert worst < 1e-8, (phase, worst)
        t[key] = G

    # 3. Fit REST/POST pair corrections against both aged single responses.
    for phase, key, ptrkey, bases in (
        ('rest', 'pair_voltage_mV', 'pair_voltage_ptr', rest['baseline_voltage_mV'][0]),
        ('post', 'post_pair_voltage_mV', 'post_pair_voltage_ptr', post['post_baseline_voltage_mV']),
    ):
        lengths = []
        for old, new in pairs:
            row = len(lengths) // count
            p = raw[ptrkey]
            lengths.extend(
                int(p[row * count + ti + 1] - p[row * count + ti]) if raw['tau_steps'][ti] <= cutoffs[old] else 0
                for ti in range(count)
            )
        ptr = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        stride = int(ptr[-1])
        values = np.zeros(len(bases) * stride + T)
        worst = 0.0
        for v in range(len(bases)):
            for row, (old, new) in enumerate(pairs):
                source_ptr = raw[ptrkey]
                source_stride = int(source_ptr[-1])
                pending = []
                for ti in range(count):
                    slot = row * count + ti
                    a, b = map(int, ptr[slot : slot + 2])
                    L = b - a
                    if not L:
                        continue
                    lo = int(source_ptr[slot])
                    r = np.zeros(T)
                    r[:L] = raw[key][v * source_stride + lo : v * source_stride + lo + L]
                    if phase == 'rest':
                        bo = rest['aged_single_voltage_mV'][old, v, ti]
                        bn = rest['aged_single_voltage_mV'][new, v, 0]
                    else:
                        bo = post['post_aged_single_voltage_mV'][v, old, ti]
                        bn = post['post_aged_single_voltage_mV'][v, new, 0]
                    pending.append((v * stride + a, L, bo, bn, r))
                if not pending:
                    continue
                bo, bn, r = (np.asarray([x[i] for x in pending]) for i in (2, 3, 4))
                beta = np.broadcast_to(bases[v], bo.shape)
                go = invert_single(bo, beta, reversals[old], lam, dt)
                gn = invert_single(bn, beta, reversals[new], lam, dt)
                h, err = integration_conductance(
                    bo + bn + r,
                    go + gn,
                    go * (reversals[old] - beta) + gn * (reversals[new] - beta),
                    beta,
                    lam,
                    dt,
                    max(reversals[old], reversals[new]),
                )
                worst = max(worst, err)
                for item, curve in zip(pending, h):
                    values[item[0] : item[0] + item[1]] = curve[: item[1]]
        assert worst < 1e-9, (phase, worst)
        t[key] = values
        t[ptrkey] = ptr

    # 4. Re-express historical singles at the fixed rebase state.
    base = post['rebase_baseline_voltage_mV'][0]
    singles = []
    single_initial = []
    for loc in range(nloc):
        ptr = raw['post_rebase_single_voltage_ptr']
        for ai in range(count):
            lo, hi = map(int, ptr[loc * count + ai : loc * count + ai + 2])
            if raw['event_age_steps'][ai] > cutoffs[loc] or hi == lo:
                singles.append(np.empty(0))
                single_initial.append(np.empty(0))
                continue
            curve = raw['post_rebase_single_voltage_mV'][lo:hi][None]
            beta = base[None, : hi - lo]
            g = invert_single(curve, beta, reversals[loc], lam, dt)[0]
            check = replay(g[None], g[None] * (reversals[loc] - beta), curve[:, 0], lam, dt)
            error = np.abs(check - curve)
            if not np.isfinite(error).all() or np.max(error) >= 1e-8:
                raise ValueError('Rebase single replay failed.')
            singles.append(g)
            single_initial.append(curve[0, :1])
    t['post_rebase_single_voltage_mV'], t['post_rebase_single_voltage_ptr'] = pack(singles, T)
    t['dif_rebase_single_initial_mV'], t['dif_rebase_single_initial_ptr'] = pack(single_initial, T)
    # 5. Fit rebase pairs using each input's own age, in bounded batches.
    pair_curves = [np.empty(0) for _ in range(len(pairs) * count * count)]
    pair_initial = [np.empty(0) for _ in pair_curves]
    pending = []
    worst = 0.0

    def flush():
        nonlocal worst, pending
        if not pending:
            return
        bo, bn, target = (np.asarray([x[i] for x in pending]) for i in (2, 3, 4))
        eo = np.asarray([reversals[x[5]] for x in pending])[:, None]
        en = np.asarray([reversals[x[6]] for x in pending])[:, None]
        beta = np.broadcast_to(base, bo.shape)
        go = invert_single(bo, beta, eo, lam, dt)
        gn = invert_single(bn, beta, en, lam, dt)
        h, err = integration_conductance(
            target, go + gn, go * (eo - beta) + gn * (en - beta), beta, lam, dt, np.maximum(eo, en)
        )
        worst = max(worst, err)
        for item, curve in zip(pending, h):
            pair_curves[item[0]] = curve[: item[1]].copy()
        pending = []

    for row, (old, new) in enumerate(pairs):
        ptr = raw['post_rebase_pair_voltage_ptr']
        for ti, tau in enumerate(raw['tau_steps']):
            for ri, age in enumerate(raw['release_age_steps']):
                slot = (row * count + ti) * count + ri
                lo, hi = map(int, ptr[slot : slot + 2])
                L = hi - lo
                if not L or tau > cutoffs[old] or age > cutoffs[old]:
                    continue
                # The new single must be represented at its own age nodes.
                new_age = int(age - tau)
                bo = interpolate_single(raw, old, int(age), T)
                bn = interpolate_single(raw, new, new_age, T)
                r = np.zeros(T)
                r[:L] = raw['post_rebase_pair_voltage_mV'][lo:hi]
                target = bo + bn + r
                corrected = target - bo - bn
                pair_initial[slot] = corrected[:1].copy()
                pending.append((slot, L, bo, bn, target, old, new))
                if len(pending) >= 64:
                    flush()
    flush()
    assert worst < 1e-9, worst
    t['post_rebase_pair_voltage_mV'], t['post_rebase_pair_voltage_ptr'] = pack(pair_curves, T)
    t['dif_rebase_pair_initial_mV'], t['dif_rebase_pair_initial_ptr'] = pack(pair_initial, T)
    # 6. Attach model parameters and check the completed banks before export.
    t['dif_version'] = np.asarray('dif-native-location-axes-1')
    t['dif_reversal_mV'] = reversals
    t['dif_leak_per_ms'] = np.asarray(lam)
    for name, value in t.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == 'f' and not np.isfinite(value).all():
            raise ValueError(f'DIF calibration produced non-finite values in {name}.')
    return t
