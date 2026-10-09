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

"""Versioned DBNN training checkpoints and deployment assets."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import pickle
from typing import Any

import brainstate
import brainunit as u
import jax
import numpy as np

from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.reduction.dbnn.model import SpikeAlignmentEvidence


MODEL_FORMAT_VERSION = 9
TRAINING_FORMAT_VERSION = 10
SPIKE_ALIGNMENT_FORMAT_VERSION = 1
_UNSET = object()


def _npz_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else Path(f"{path}.npz")


def host_params(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Copy a JAX parameter mapping to host NumPy arrays."""
    return {name: np.asarray(value) for name, value in jax.device_get(params).items()}


def _json_compatible_initialization_options(options: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(options)
    result["bounds"] = [list(map(float, bound)) for bound in result["bounds"]]
    return result


def save_model(
    path: str | Path,
    model: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    spike_alignment_path: str | Path | None = None,
    spike_alignment_evidence: SpikeAlignmentEvidence | None | object = _UNSET,
) -> None:
    """Save a standalone DBNN deployment asset without training dependencies."""
    if type(model).__name__ not in {"DBNN", "DBNNGIF"}:
        raise TypeError(f"Unsupported DBNN deployment model type: {type(model).__name__}.")
    layout = model.layout
    metadata = dict(metadata or {})
    model_fingerprint = getattr(model, "source_fingerprint", None)
    metadata_fingerprint = metadata.get("source_fingerprint")
    if model_fingerprint and metadata_fingerprint and model_fingerprint != metadata_fingerprint:
        raise ValueError("Model and metadata source_fingerprint values must match.")
    if metadata_fingerprint is None and model_fingerprint:
        metadata["source_fingerprint"] = model_fingerprint
    if not metadata.get("source_fingerprint"):
        raise ValueError("A standalone DBNN model asset requires metadata['source_fingerprint'].")
    path = _npz_path(path)
    evidence = (
        getattr(model, "spike_alignment_evidence", None)
        if spike_alignment_evidence is _UNSET
        else spike_alignment_evidence
    )
    evidence_path = None
    if evidence is not None:
        if not isinstance(evidence, SpikeAlignmentEvidence):
            raise TypeError("model.spike_alignment_evidence must be a SpikeAlignmentEvidence or None.")
        evidence_path = (
            path.with_name(f"{path.stem}_spike_alignment.npz")
            if spike_alignment_path is None
            else Path(spike_alignment_path)
        )
        save_spike_alignment_evidence(evidence_path, evidence)
    elif spike_alignment_path is not None:
        raise ValueError("spike_alignment_path requires model.spike_alignment_evidence.")
    manifest = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": type(model).__name__,
        "n_channels": model.n_channels,
        "dt_ms": model.dt_ms,
        "mode": model.mode,
        "input_sign_mode": model.input_sign_mode,
        "input_alignment": getattr(model, "input_alignment", None),
        "reset_enabled": bool(getattr(model, "reset_enabled", False)),
        "spike_time_offset_ms": (
            float(model.spike_time_offset_ms) if type(model).__name__ == "DBNNGIF" else None
        ),
        "layout": layout.to_dict(),
        "layout_fingerprint": layout.fingerprint,
        "spike_alignment_evidence_path": (
            None if evidence_path is None else os.path.relpath(evidence_path, start=path.parent)
        ),
        "metadata": metadata,
    }
    values = {f"param_{name}": value for name, value in host_params(model.get_params()).items() if name != "dt_ms"}
    values["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False))
    np.savez_compressed(path, **values)


def load_model(
    path: str | Path,
    *,
    mode: str | None = None,
    expected_layout: ChannelLayout | None = None,
    expected_source_fingerprint: str | None = None,
):
    """Load and validate a standalone DBNN deployment asset."""
    from braincell.reduction.dbnn.model import DBNN, DBNNGIF

    path = _npz_path(path)
    with np.load(path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise KeyError("DBNN model asset is missing manifest_json.")
        manifest = json.loads(str(data["manifest_json"]))
        model_types = {"DBNN": DBNN, "DBNNGIF": DBNNGIF}
        format_version = manifest.get("format_version")
        if format_version not in {8, MODEL_FORMAT_VERSION} or manifest.get("model_type") not in model_types:
            raise ValueError("Unsupported DBNN model asset format or model type.")
        if manifest.get("layout") is None:
            raise ValueError("DBNN model asset does not contain required channel layout semantics.")
        asset_layout = ChannelLayout.from_dict(manifest["layout"])
        if asset_layout.fingerprint != manifest.get("layout_fingerprint"):
            raise ValueError("DBNN model asset layout fingerprint is corrupt.")
        if int(manifest["n_channels"]) != asset_layout.n_channels:
            raise ValueError("DBNN model asset channel count does not match its layout.")
        if expected_layout is not None:
            if expected_layout.fingerprint != asset_layout.fingerprint:
                raise ValueError("DBNN model asset layout is incompatible with the expected layout.")
        asset_metadata = dict(manifest.get("metadata", {}))
        if (
            expected_source_fingerprint is not None
            and asset_metadata.get("source_fingerprint") != expected_source_fingerprint
        ):
            raise ValueError("DBNN model asset source fingerprint is incompatible with the expected teacher.")
        selected_mode = manifest["mode"] if mode is None else mode
        model = model_types[manifest["model_type"]](
            asset_layout,
            mode=selected_mode,
            input_sign_mode=manifest.get("input_sign_mode", "none"),
            dt=float(manifest["dt_ms"]) * u.ms,
        )
        if manifest.get("reset_enabled", False):
            if not isinstance(model, DBNNGIF):
                raise ValueError("Only DBNNGIF assets may enable voltage reset.")
            model.enable_reset()
        spike_time_offset_ms = manifest.get("spike_time_offset_ms")
        if isinstance(model, DBNNGIF):
            if spike_time_offset_ms is None:
                raise ValueError("DBNNGIF asset is missing spike_time_offset_ms.")
            model.set_spike_time_offset(spike_time_offset_ms)
        elif spike_time_offset_ms is not None:
            raise ValueError("Base DBNN assets cannot carry a spike time offset.")
        params = {
            name.removeprefix("param_"): np.array(data[name], copy=True)
            for name in data.files
            if name.startswith("param_")
        }
        params["dt_ms"] = manifest["dt_ms"]
        model.set_params(params)
        model.asset_metadata = asset_metadata
        model.source_fingerprint = asset_metadata.get("source_fingerprint")
        model.input_alignment = manifest.get("input_alignment")
        evidence_reference = manifest.get("spike_alignment_evidence_path")
        if evidence_reference is not None:
            reference_path = Path(evidence_reference)
            if reference_path.is_absolute():
                raise ValueError("Spike alignment evidence path must be relative to the model asset.")
            model.spike_alignment_evidence = load_spike_alignment_evidence(Path(path).parent / reference_path)
            evidence = model.spike_alignment_evidence
            if evidence.layout_fingerprint != asset_layout.fingerprint or evidence.dt_ms != model.dt_ms:
                raise ValueError("Spike alignment evidence is incompatible with the model asset.")
        return model


def save_spike_alignment_evidence(path: str | Path, evidence: SpikeAlignmentEvidence) -> None:
    """Save retained spike-alignment validation data as a standalone asset."""
    if not isinstance(evidence, SpikeAlignmentEvidence):
        raise TypeError("evidence must be a SpikeAlignmentEvidence.")
    spike_trace = np.concatenate(
        [np.full(row.size, trace, dtype=np.int64) for trace, row in enumerate(evidence.raw_spike_times_ms)]
    )
    spike_time = np.concatenate(evidence.raw_spike_times_ms).astype(np.float32, copy=False)
    manifest = {
        "format_version": SPIKE_ALIGNMENT_FORMAT_VERSION,
        "dt_ms": evidence.dt_ms,
        "match_window_ms": evidence.match_window_ms,
        "layout_fingerprint": evidence.layout_fingerprint,
        "dynamics_fingerprint": evidence.dynamics_fingerprint,
        "validation_seeds": evidence.validation_seeds,
        "n_traces": evidence.teacher_voltage_mv.shape[0],
    }
    np.savez_compressed(
        _npz_path(path),
        teacher_voltage_mv=evidence.teacher_voltage_mv,
        raw_spike_trace_id=spike_trace,
        raw_spike_time_ms=spike_time,
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False)),
    )


def load_spike_alignment_evidence(path: str | Path) -> SpikeAlignmentEvidence:
    """Load and validate a standalone spike-alignment validation asset."""
    with np.load(_npz_path(path), allow_pickle=False) as data:
        required = {"teacher_voltage_mv", "raw_spike_trace_id", "raw_spike_time_ms", "manifest_json"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Spike alignment evidence is missing fields: {sorted(missing)}.")
        manifest = json.loads(str(data["manifest_json"]))
        if manifest.get("format_version") != SPIKE_ALIGNMENT_FORMAT_VERSION:
            raise ValueError("Unsupported spike alignment evidence format version.")
        trace_ids = np.asarray(data["raw_spike_trace_id"])
        spike_times = np.asarray(data["raw_spike_time_ms"])
        n_traces = int(manifest["n_traces"])
        if trace_ids.shape != spike_times.shape or trace_ids.ndim != 1:
            raise ValueError("Spike alignment evidence sparse spike arrays are incompatible.")
        if trace_ids.size and (np.any(trace_ids < 0) or np.any(trace_ids >= n_traces)):
            raise ValueError("Spike alignment evidence contains an invalid trace index.")
        rows = tuple(np.array(spike_times[trace_ids == trace], copy=True) for trace in range(n_traces))
        return SpikeAlignmentEvidence(
            teacher_voltage_mv=np.array(data["teacher_voltage_mv"], copy=True),
            raw_spike_times_ms=rows,
            dt_ms=manifest["dt_ms"],
            match_window_ms=manifest["match_window_ms"],
            layout_fingerprint=manifest["layout_fingerprint"],
            dynamics_fingerprint=manifest["dynamics_fingerprint"],
            validation_seeds=tuple(manifest["validation_seeds"]),
        )


def save_training_checkpoint(
    path: str | Path,
    trainer: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save model, optimizer, and progress for trusted training resumption."""
    manifest = {
        "format_version": TRAINING_FORMAT_VERSION,
        "epoch": trainer.epoch,
        "best_validation_loss": (trainer.best_validation_loss if np.isfinite(trainer.best_validation_loss) else None),
        "learning_rate": trainer.learning_rate,
        "lr_step_size": trainer.lr_step_size,
        "lr_gamma": trainer.lr_gamma,
        "loss": trainer.loss,
        "gradient_clip": trainer.gradient_clip,
        "seed": trainer.seed,
        "n_channels": trainer.model.n_channels,
        "dt_ms": trainer.model.dt_ms,
        "mode": trainer.model.mode,
        "input_sign_mode": trainer.model.input_sign_mode,
        "source_fingerprint": trainer.source_fingerprint,
        "input_alignment": trainer.input_alignment,
        "reset_enabled": bool(getattr(trainer.model, "reset_enabled", False)),
        "has_best_snapshot": trainer._best_params is not None,
        "initialization_search": trainer.initialization_search,
        "initialization_options": _json_compatible_initialization_options(trainer.initialization_options),
        "initialization_completed": trainer.initialization_completed,
        "initialization_report": trainer.initialization_report,
        "layout": trainer.model.layout.to_dict(),
        "layout_fingerprint": trainer.model.layout.fingerprint,
        "metadata": dict(metadata or {}),
    }
    optimizer_states = {
        path: state.value
        for path, state in brainstate.graph.states(trainer.optimizer).items()
        if not isinstance(state, brainstate.ParamState)
    }
    optimizer_bytes = np.frombuffer(pickle.dumps(optimizer_states), dtype=np.uint8)
    values = {
        f"param_{name}": value for name, value in host_params(trainer.model.get_params()).items() if name != "dt_ms"
    }
    values.update(
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False)),
        optimizer_state=optimizer_bytes,
    )
    if trainer._best_params is not None:
        values.update(
            {
                f"best_param_{name}": value
                for name, value in host_params(trainer._best_params).items()
                if name != "dt_ms"
            }
        )
        values["best_optimizer_state"] = np.frombuffer(pickle.dumps(trainer._best_optimizer_state), dtype=np.uint8)
    np.savez_compressed(path, **values)


def load_training_checkpoint(path: str | Path, trainer: Any) -> None:
    """Restore a trusted DBNN training checkpoint into an existing trainer."""
    with np.load(path, allow_pickle=False) as data:
        required = {"manifest_json", "optimizer_state"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Training checkpoint is missing fields: {sorted(missing)}.")
        manifest = json.loads(str(data["manifest_json"]))
        if manifest.get("format_version") != TRAINING_FORMAT_VERSION:
            raise ValueError(f"Unsupported DBNN training checkpoint version: {manifest.get('format_version')!r}.")
        checkpoint_layout = ChannelLayout.from_dict(manifest["layout"])
        if checkpoint_layout.fingerprint != manifest.get("layout_fingerprint"):
            raise ValueError("Training checkpoint layout fingerprint is corrupt.")
        if checkpoint_layout.n_channels != int(manifest["n_channels"]):
            raise ValueError("Training checkpoint channel count does not match its layout.")
        if checkpoint_layout.fingerprint != trainer.model.layout.fingerprint:
            raise ValueError("Training checkpoint channel layout is incompatible with the model.")
        if bool(manifest.get("reset_enabled", False)) != bool(getattr(trainer.model, "reset_enabled", False)):
            raise ValueError("Training checkpoint reset configuration is incompatible with the model.")
        if not np.isclose(float(manifest["dt_ms"]), trainer.model.dt_ms):
            raise ValueError("Training checkpoint dt is incompatible with the model.")
        config = {
            "learning_rate": trainer.learning_rate,
            "lr_step_size": trainer.lr_step_size,
            "lr_gamma": trainer.lr_gamma,
            "loss": trainer.loss,
            "gradient_clip": trainer.gradient_clip,
            "seed": trainer.seed,
            "mode": trainer.model.mode,
            "input_sign_mode": trainer.model.input_sign_mode,
            "initialization_search": trainer.initialization_search,
            "initialization_options": _json_compatible_initialization_options(trainer.initialization_options),
        }
        for field, expected in config.items():
            if manifest.get(field, "none" if field == "input_sign_mode" else None) != expected:
                raise ValueError(f"Training checkpoint {field} is incompatible with the trainer.")
        params = {
            name.removeprefix("param_"): np.array(data[name], copy=True)
            for name in data.files
            if name.startswith("param_")
        }
        params["dt_ms"] = manifest["dt_ms"]
        optimizer_state = pickle.loads(np.asarray(data["optimizer_state"], dtype=np.uint8).tobytes())
        best_params = None
        best_optimizer_state = None
        if manifest.get("has_best_snapshot"):
            if "best_optimizer_state" not in data.files:
                raise KeyError("Training checkpoint is missing best_optimizer_state.")
            best_params = {
                name.removeprefix("best_param_"): np.array(data[name], copy=True)
                for name in data.files
                if name.startswith("best_param_")
            }
            best_params["dt_ms"] = manifest["dt_ms"]
            best_optimizer_state = pickle.loads(np.asarray(data["best_optimizer_state"], dtype=np.uint8).tobytes())
    current_states = brainstate.graph.states(trainer.optimizer)
    current_optimizer_paths = {
        path for path, state in current_states.items() if not isinstance(state, brainstate.ParamState)
    }
    if set(optimizer_state) != current_optimizer_paths:
        raise ValueError("Training checkpoint optimizer state paths are incompatible with the trainer.")
    if best_optimizer_state is not None and set(best_optimizer_state) != current_optimizer_paths:
        raise ValueError("Training checkpoint best optimizer state paths are incompatible with the trainer.")
    checkpoint_source = manifest.get("source_fingerprint")
    if trainer.source_fingerprint is not None and checkpoint_source != trainer.source_fingerprint:
        raise ValueError("Training checkpoint source fingerprint is incompatible with the trainer.")
    checkpoint_alignment = manifest.get("input_alignment")
    if trainer.input_alignment is not None and checkpoint_alignment != trainer.input_alignment:
        raise ValueError("Training checkpoint input alignment is incompatible with the trainer.")
    trainer.model.set_params(params)
    for state_path, value in optimizer_state.items():
        current_states[state_path].value = value
    trainer.epoch = int(manifest["epoch"])
    trainer.best_validation_loss = (
        float("inf") if manifest["best_validation_loss"] is None else float(manifest["best_validation_loss"])
    )
    trainer.source_fingerprint = manifest.get("source_fingerprint")
    trainer.input_alignment = manifest.get("input_alignment")
    trainer.model.source_fingerprint = trainer.source_fingerprint
    trainer.model.input_alignment = trainer.input_alignment
    trainer._best_params = best_params
    trainer._best_optimizer_state = best_optimizer_state
    trainer.initialization_completed = bool(manifest["initialization_completed"])
    trainer.initialization_report = manifest.get("initialization_report")
    if trainer._best_params is None and np.isfinite(trainer.best_validation_loss):
        trainer._best_params = {name: np.array(value) for name, value in params.items()}
        trainer._best_optimizer_state = trainer._snapshot_optimizer_state()


__all__ = [
    "MODEL_FORMAT_VERSION",
    "SPIKE_ALIGNMENT_FORMAT_VERSION",
    "TRAINING_FORMAT_VERSION",
    "host_params",
    "load_model",
    "load_spike_alignment_evidence",
    "load_training_checkpoint",
    "save_model",
    "save_spike_alignment_evidence",
    "save_training_checkpoint",
]
