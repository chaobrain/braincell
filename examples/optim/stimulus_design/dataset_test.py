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

from __future__ import annotations

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.optim.stimulus_design.dataset import (
    BASELINE_STOP,
    DEND_CONDUCTANCES,
    N_STEPS,
    PARAMETER_NAMES,
    PRMLS_LEVELS,
    SOMA_CONDUCTANCES,
    STIMULUS_STOP,
    BaseWaveform,
    StimulusCalibration,
    StimulusDataset,
    _balanced_prmls,
    _normalized_prmls_waveforms,
    _validate_dataset,
    build_cell,
    cross_locations,
    simulate_voltage,
)


def test_seven_cv_cell_has_soma_and_shared_dend_parameter_roots() -> None:
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        cell = build_cell(np.zeros((1, N_STEPS, 3)), trainable=True)

    assert cell.n_cv == 7
    assert tuple(cell.trainables.parameters().states()) == PARAMETER_NAMES
    np.testing.assert_allclose(cell.channels["soma_leak"].get("g_max").to_decimal(u.mS / u.cm**2), SOMA_CONDUCTANCES[0])
    np.testing.assert_allclose(cell.channels["dend_leak"].get("g_max").to_decimal(u.mS / u.cm**2), DEND_CONDUCTANCES[0])
    assert cell.channels["dend_leak"].get("g_max").shape == (6,)


def test_prmls_is_windowed_reproducible_and_level_balanced() -> None:
    first = _balanced_prmls(seed=7, clock_ms=2.0)
    second = _balanced_prmls(seed=7, clock_ms=2.0)

    np.testing.assert_array_equal(first, second)
    assert np.all(first[:BASELINE_STOP] == 0.0)
    assert np.all(first[STIMULUS_STOP:] == 0.0)
    symbols = first[BASELINE_STOP:STIMULUS_STOP:80]
    assert set(np.unique(symbols)) == set(PRMLS_LEVELS)
    counts = np.asarray([np.sum(symbols == level) for level in PRMLS_LEVELS])
    assert counts.max() - counts.min() <= 1


def test_cross_locations_preserves_waveform_and_builds_expected_split_counts() -> None:
    steps = []
    for index, split in enumerate(("train",) * 5 + ("validation",) * 2 + ("test",)):
        values = np.zeros((N_STEPS,))
        values[BASELINE_STOP:STIMULUS_STOP] = 0.01 * (index + 1)
        steps.append(BaseWaveform(f"step_{index}", "step", f"feature_{index}", split, values))
    waveforms = tuple(steps) + tuple(
        BaseWaveform(
            item.waveform_id,
            item.family,
            item.feature,
            item.split,
            item.current_na * 0.02,
            item.clock_ms,
            item.seed,
            0.02,
        )
        for item in _normalized_prmls_waveforms()
    )
    protocols, currents = cross_locations(waveforms)

    assert len(protocols) == 60
    assert currents.shape == (60, N_STEPS, 3)
    assert {split: sum(item.split == split for item in protocols) for split in ("train", "validation", "test")} == {
        "train": 33,
        "validation": 15,
        "test": 12,
    }
    for offset in range(0, len(protocols), 3):
        waveforms_at_sites = [currents[offset + site, :, site] for site in range(3)]
        np.testing.assert_array_equal(waveforms_at_sites[0], waveforms_at_sites[1])
        np.testing.assert_array_equal(waveforms_at_sites[0], waveforms_at_sites[2])

    calibration = StimulusCalibration(-0.1, -0.2, 0.01, 0.1, {i: (0.1, 0.2) for i in range(1, 5)}, 0.02, 0)
    dataset = StimulusDataset(
        protocols,
        np.arange(N_STEPS) * 0.025,
        currents,
        np.full((60, N_STEPS, 7), -65.0),
        np.zeros((60,), dtype=np.int16),
        calibration,
    )
    _validate_dataset(dataset)


def test_current_playback_drives_each_protocol_at_its_own_site() -> None:
    # Two protocols: none, and 0.05 nA into the soma. Each protocol is one
    # population member, so the playback divides by per-member point areas;
    # indexing those as one flat array raised a broadcasting TypeError.
    steps = 40
    currents = np.zeros((2, N_STEPS, 3))
    currents[1, :, 0] = 0.05
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        voltage = np.asarray(simulate_voltage(build_cell(currents, trainable=False), num_steps=steps))

    assert voltage.shape == (2, steps, 7)
    assert np.all(np.isfinite(voltage))
    rise = voltage[1, -1] - voltage[0, -1]
    soma, dend_a_tip, dend_b_tip = 0, 3, 6
    assert rise[soma] > 0.1  # mV after 1 ms
    assert rise[soma] > rise[dend_a_tip] > 0.0
    assert rise[soma] > rise[dend_b_tip] > 0.0
