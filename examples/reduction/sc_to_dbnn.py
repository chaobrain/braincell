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

"""Fit a DBNN-GIF reduction from the detailed SC2021 Cell model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import warnings

import brainunit as u
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import braincell
from braincell.filter import at
from braincell.reduction.dbnn import ChannelLayout, DBNNGIF, fit_dbnn_gif
from braincell.reduction.dbnn.layout import build_channel_layout
from examples.neuron_compare.cell.sc_ma2021.parameters import load_sc21_params
from examples.neuron_compare.cell.sc_ma2021.sc_braincell import SC


EXCITATORY = (8, 12, 14, 21, 22, 23, 26, 27, 28, 32, 43, 90, 100)
INHIBITORY = (
    8,
    12,
    14,
    21,
    22,
    23,
    26,
    27,
    28,
    31,
    32,
    34,
    35,
    43,
    50,
    57,
    58,
    66,
    67,
    68,
    71,
    73,
    81,
    82,
    84,
    86,
    89,
    90,
    98,
    100,
    40,
    69,
)
ALL_DENDRITES = EXCITATORY + INHIBITORY
ALL_KINDS = ("E",) * len(EXCITATORY) + ("I",) * len(INHIBITORY)
ALL_RATES_HZ = tuple(
    (38.0, 39.0, 40.0, 41.0)[index % 4] for index in range(len(ALL_DENDRITES))
)
WORKFLOW_VERSION = 4


@dataclass(frozen=True)
class Config:
    """Define one reproducible SC-to-DBNN run profile."""

    channels: tuple[int, ...]
    train_traces: int
    validation_traces: int
    test_traces: int
    duration_ms: float
    epochs: int
    train_batch_size: int
    teacher_batch_size: int
    data_seed: int
    training_seed: int
    init_maxiter: int
    init_popsize: int
    init_max_pairs: int
    min_spikes: int
    max_spikes: int
    max_teacher_batches: int


PROFILES = {
    "quick": Config(
        channels=(0, 1, 2, 3, 13, 14, 15, 16),
        train_traces=32,
        validation_traces=8,
        test_traces=8,
        duration_ms=1000.0,
        epochs=200,
        train_batch_size=8,
        teacher_batch_size=16,
        data_seed=20260828,
        training_seed=42,
        init_maxiter=4,
        init_popsize=4,
        init_max_pairs=64,
        min_spikes=0,
        max_spikes=1000,
        max_teacher_batches=4,
    ),
    "full": Config(
        channels=tuple(range(len(ALL_DENDRITES))),
        train_traces=900,
        validation_traces=50,
        test_traces=50,
        duration_ms=6000.0,
        epochs=500,
        train_batch_size=16,
        teacher_batch_size=512,
        data_seed=20260828,
        training_seed=42,
        init_maxiter=8,
        init_popsize=8,
        init_max_pairs=512,
        min_spikes=1,
        max_spikes=7,
        max_teacher_batches=100,
    ),
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse one SC-to-DBNN run configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reduction" / "sc_to_dbnn" / f"v{WORKFLOW_VERSION}",
    )
    return parser.parse_args(argv)


def _source_fingerprint(profile: str, channels, dendrites, kinds) -> str:
    payload = {
        "version": WORKFLOW_VERSION,
        "profile": profile,
        "channels": list(channels),
        "sites": list(zip(dendrites, kinds)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def migrate_v3_gif_model(source, destination, expected_layout: ChannelLayout) -> DBNNGIF:
    """Migrate one legacy SC DBNN-GIF asset onto the current channel layout."""
    source = Path(source)
    with np.load(source, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"]))
        parameters = {
            name.removeprefix("param_"): np.array(data[name], copy=True)
            for name in data.files
            if name.startswith("param_")
        }
    if manifest.get("format_version") != 6 or manifest.get("model_type") != "DBNNGIF":
        raise ValueError("Expected a legacy version-6 DBNNGIF asset.")
    if manifest.get("reset_enabled", False):
        raise ValueError("Legacy DBNNGIF assets with voltage reset are not supported by this migration.")
    legacy_data = dict(manifest["layout"])
    legacy_data["branch_ids"] = ()
    legacy_data["branch_xs"] = ()
    legacy_layout = ChannelLayout.from_dict(legacy_data)
    for field in (
        "specs",
        "synapse_ids",
        "placement_ids",
        "point_ids",
        "spec_indices",
        "reference_weights_us",
        "training_ranges",
    ):
        if legacy_layout.to_dict()[field] != expected_layout.to_dict()[field]:
            raise ValueError(f"Legacy asset layout is incompatible with expected layout field {field!r}.")
    model = DBNNGIF(
        expected_layout,
        mode="r",
        input_sign_mode=manifest["input_sign_mode"],
        dt=float(manifest["dt_ms"]) * u.ms,
    )
    expected_parameters = set(model.get_params()).difference({"dt_ms"})
    if set(parameters) != expected_parameters:
        raise ValueError("Legacy asset parameter names are incompatible with the current DBNNGIF model.")
    parameters["dt_ms"] = manifest["dt_ms"]
    model.set_params(parameters)
    metadata = dict(manifest.get("metadata", {}))
    if not metadata.get("source_fingerprint"):
        raise ValueError("Legacy asset metadata is missing source_fingerprint.")
    model.source_fingerprint = metadata["source_fingerprint"]
    model.input_alignment = manifest.get("input_alignment")
    model.save(destination, metadata=metadata)
    return model


def build_sc_layout(profile: str):
    """Build the SC factory and DBNN channel layout for one run profile."""
    config = PROFILES[profile]
    selected = config.channels
    dendrite_indices = tuple(ALL_DENDRITES[index] for index in selected)
    kinds = tuple(ALL_KINDS[index] for index in selected)
    rates_hz = np.asarray([ALL_RATES_HZ[index] for index in selected], dtype=np.float32)
    specs = tuple(
        braincell.mech.Synapse(
            "Exp2Syn",
            name=f"input_{source_index:02d}",
            tau1=0.5 * u.ms,
            tau2=(2.0 if kind == "E" else 5.0) * u.ms,
            e=(0.0 if kind == "E" else -80.0) * u.mV,
        )
        for source_index, kind in zip(selected, kinds)
    )
    reference_weights_us = tuple(0.0002 if kind == "E" else 0.0005 for kind in kinds)

    def build_sc(pop_size):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"from_points\(\) produced .* zero-length segment")
            assembly = SC(
                params=load_sc21_params(),
                pop_size=pop_size,
                name="sc_dbnn_teacher",
            ).build()
        dendrite_branch_ids = tuple(
            branch.index for branch in assembly.morpho.branches if branch.type == "dendrite"
        )
        for dendrite_index, spec in zip(dendrite_indices, specs):
            assembly.cell.place(at(dendrite_branch_ids[dendrite_index], 0.5), spec)
        return assembly.cell

    template = build_sc((1,))
    layout = build_channel_layout(
        template,
        specs,
        reference_weights_us=reference_weights_us,
        training_ranges=tuple((0.0, 3.0) for _ in specs),
    )
    source_fingerprint = _source_fingerprint(profile, selected, dendrite_indices, kinds)
    return config, layout, build_sc, rates_hz, source_fingerprint


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    """Generate detailed SC data and fit one DBNN-GIF model."""
    config, layout, build_sc, rates_hz, source_fingerprint = build_sc_layout(args.profile)

    def accept(batch):
        voltage = np.asarray(batch.voltage_mv)
        above = voltage >= -20.0
        spikes = np.count_nonzero(above[:, 1:] & ~above[:, :-1], axis=1)
        return np.flatnonzero(
            np.isfinite(voltage).all(axis=1)
            & (voltage.min(axis=1) >= -95.0)
            & (voltage.max(axis=1) <= 80.0)
            & (spikes >= config.min_spikes)
            & (spikes <= config.max_spikes)
        )

    destination = Path(args.output_dir) / args.profile
    model, mse, variance_explained = fit_dbnn_gif(
        layout,
        cell_factory=build_sc,
        rate_hz=rates_hz,
        output_location=at("soma", 0.5),
        acceptance=accept,
        train_traces=config.train_traces,
        validation_traces=config.validation_traces,
        test_traces=config.test_traces,
        duration_ms=config.duration_ms,
        dt=1.0 * u.ms,
        teacher_batch_size=config.teacher_batch_size,
        max_teacher_batches=config.max_teacher_batches,
        data_seed=config.data_seed,
        split_seed=config.training_seed,
        training_seed=config.training_seed,
        source_fingerprint=source_fingerprint,
        spike_threshold_mv=-20.0,
        epochs=config.epochs,
        batch_size=config.train_batch_size,
        initialization_options={
            "maxiter": config.init_maxiter,
            "popsize": config.init_popsize,
            "max_pairs": config.init_max_pairs,
        },
        output_dir=destination,
    )
    return {
        "profile": args.profile,
        "channels": layout.n_channels,
        "test_masked_mse": mse,
        "test_masked_variance_explained": variance_explained,
        "model": str(destination / "dbnn_gif_model.npz"),
        "input_alignment": model.input_alignment,
    }


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
