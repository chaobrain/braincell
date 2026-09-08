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

import numpy as np

from examples.experimental.optim_initialization.dc_protocol_dataset import (
    N_STEPS,
    SITES,
    DcCalibration,
    DcDataset,
    build_catalog,
    load_dataset,
    parameter_names,
    upward_crossing_counts,
    write_dataset,
)


def _calibration() -> DcCalibration:
    return DcCalibration(
        negative_amplitudes_na={site: (-0.1, -0.2, -0.3, -0.4) for site in SITES},
        negative_minima_mv={site: (-80.0, -90.0, -100.0, -110.0) for site in SITES},
        positive_intervals_na={
            site: {count: (0.01 * count, 0.01 * count + 0.005) for count in range(5)} for site in SITES
        },
    )


def test_catalog_has_fixed_site_response_and_split_coverage() -> None:
    catalog = build_catalog(_calibration())

    assert len(catalog) == 27
    assert {split: sum(item.split == split for item in catalog) for split in ("train", "validation", "test")} == {
        "train": 15,
        "validation": 6,
        "test": 6,
    }
    assert {site: sum(item.injection_site == site for item in catalog) for site in SITES} == {site: 9 for site in SITES}
    assert len(parameter_names()) == 9
    assert len(set(parameter_names())) == 9


def test_dataset_round_trip_preserves_protocol_metadata(tmp_path) -> None:
    protocols = build_catalog(_calibration())
    dataset = DcDataset(
        protocols=protocols,
        time_ms=np.arange(N_STEPS) * 0.025,
        voltage_mv=np.full((27, N_STEPS, 3), -65.0),
        target_spike_counts=np.zeros((27,), dtype=np.int16),
    )

    write_dataset(tmp_path, dataset, _calibration())
    restored = load_dataset(tmp_path)

    assert restored.protocols == protocols
    np.testing.assert_array_equal(restored.voltage_mv, dataset.voltage_mv)
    np.testing.assert_array_equal(restored.indices("train"), dataset.indices("train"))


def test_upward_crossings_preserve_leading_axes() -> None:
    voltage = np.asarray([[[-1.0, 1.0, -1.0, 1.0], [-1.0, -0.5, 0.5, 1.0]]])

    counts = upward_crossing_counts(voltage)

    np.testing.assert_array_equal(counts, [[2, 1]])
