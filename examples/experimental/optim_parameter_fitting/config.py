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

"""Typed Python configuration for parameter-learning experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import types


@dataclass(frozen=True)
class InitializationConfig:
    """Describe one reproducible physical-parameter population."""

    seed: int = 0
    num_candidates: int = 64
    distribution: str = "uniform_physical"
    target_relative_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("initialization seed must be an integer.")
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be positive.")
        if self.distribution != "uniform_physical":
            raise ValueError("The baseline supports only uniform_physical initialization.")
        if self.target_relative_range is not None:
            lower, upper = self.target_relative_range
            if not 0.0 < lower < upper:
                raise ValueError("target_relative_range must be positive and increasing.")

    def describe(self) -> dict[str, object]:
        """Return serializable initialization metadata."""
        result = {
            "seed": self.seed,
            "num_candidates": self.num_candidates,
            "distribution": self.distribution,
        }
        if self.target_relative_range is not None:
            result["target_relative_range"] = list(self.target_relative_range)
        return result


@dataclass(frozen=True)
class ReportingConfig:
    """Define endpoint success thresholds without changing training."""

    validation_rmse_threshold_mv: float = 5.0
    parameter_relative_rms_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.validation_rmse_threshold_mv <= 0.0 or self.parameter_relative_rms_threshold <= 0.0:
            raise ValueError("Reporting thresholds must be positive.")

    def describe(self) -> dict[str, object]:
        """Return serializable reporting metadata."""
        return {
            "validation_rmse_threshold_mv": self.validation_rmse_threshold_mv,
            "parameter_relative_rms_threshold": self.parameter_relative_rms_threshold,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Compose one model, dataset, objective, initialization, and stage pipeline."""

    name: str
    model: object
    dataset: object
    loss: object
    initialization: InitializationConfig
    stages: tuple[object, ...]
    reporting: ReportingConfig = ReportingConfig()
    schema_version: int = 1
    artifact_label: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Experiment name must contain only letters, digits, '-' and '_'.")
        if self.schema_version != 1:
            raise ValueError("Only experiment schema_version=1 is supported.")
        if self.artifact_label is not None and (
            not self.artifact_label or not self.artifact_label.replace("_", "").replace("-", "").isalnum()
        ):
            raise ValueError("artifact_label must contain only letters, digits, '-' and '_'.")
        if not self.stages:
            raise ValueError("An experiment requires at least one optimization stage.")
        for label, component in (
            ("model", self.model),
            ("dataset", self.dataset),
            ("loss", self.loss),
            ("initialization", self.initialization),
            ("reporting", self.reporting),
        ):
            if not callable(getattr(component, "describe", None)):
                raise TypeError(f"{label} component must implement describe().")
        for index, stage in enumerate(self.stages):
            if not callable(getattr(stage, "describe", None)):
                raise TypeError(f"stage {index} must implement describe().")

    def describe(self) -> dict[str, object]:
        """Resolve the Python composition to stable JSON metadata."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model": self.model.describe(),
            "dataset": self.dataset.describe(),
            "loss": self.loss.describe(),
            "initialization": self.initialization.describe(),
            "stages": [stage.describe() for stage in self.stages],
            "reporting": self.reporting.describe(),
        }

    def digest(self) -> str:
        """Return a short stable hash of resolved behavior metadata."""
        payload = json.dumps(self.describe(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def load_config(path: Path) -> tuple[ExperimentConfig, types.ModuleType]:
    """Load a trusted Python preset exporting ``CONFIG``."""
    source = Path(path).resolve()
    if not source.is_file() or source.suffix != ".py":
        raise ValueError(f"Config must be an existing Python file, got {source}.")
    module_name = f"braincell_experiment_config_{hashlib.sha256(str(source).encode()).hexdigest()[:12]}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to load experiment config {source}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"{source} must export CONFIG: ExperimentConfig.")
    return config, module
