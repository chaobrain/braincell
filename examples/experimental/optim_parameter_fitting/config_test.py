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

from pathlib import Path

from examples.experimental.optim_parameter_fitting.config import ExperimentConfig, load_config


def test_python_preset_loads_and_resolves_without_string_registry() -> None:
    path = Path(__file__).resolve().parent / "configs" / "basic_1cv_bounded_direct_adam.py"

    config, _module = load_config(path)
    resolved = config.describe()

    assert isinstance(config, ExperimentConfig)
    assert resolved["model"]["n_cv"] == 1
    assert resolved["model"]["parameter_space"]["runtime_binding"] == "direct_parameter"
    assert resolved["initialization"]["num_candidates"] == 64
    assert len(config.digest()) == 12


def test_e300_preset_changes_only_adam_epochs() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    baseline, _ = load_config(directory / "basic_1cv_bounded_direct_adam.py")
    extended, _ = load_config(directory / "basic_1cv_bounded_direct_adam_e300.py")
    baseline_values = baseline.describe()
    extended_values = extended.describe()

    baseline_stage = baseline_values.pop("stages")
    extended_stage = extended_values.pop("stages")
    assert extended_values == baseline_values
    assert baseline_stage[0] | {"epochs": 300} == extended_stage[0]
    assert extended.digest() == "c00763f960b0"


def test_wide_bound_preset_keeps_physical_initial_support_and_changes_lr() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    baseline, _ = load_config(directory / "basic_1cv_bounded_direct_adam_e300.py")
    changed, _ = load_config(directory / "basic_1cv_bounded_direct_adam_e300_lr02_wide.py")

    assert changed.stages[0].epochs == baseline.stages[0].epochs == 300
    assert changed.stages[0].learning_rate == 0.02
    assert changed.model.parameter_space.lower == (0.03, 12.0, 3.6)
    assert changed.model.parameter_space.upper == (0.6, 240.0, 72.0)
    assert changed.initialization.target_relative_range == (0.5, 1.5)


def test_rprop_preset_changes_only_optimizer_stage() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    baseline, _ = load_config(directory / "basic_1cv_bounded_direct_adam_e300_wide.py")
    changed, _ = load_config(directory / "basic_1cv_bounded_direct_rprop_e300_wide.py")
    baseline_values = baseline.describe()
    changed_values = changed.describe()

    baseline_values.pop("stages")
    changed_values.pop("stages")
    assert changed_values == baseline_values
    assert changed.stages[0].describe()["optimizer"] == "Rprop"


def test_huber_preset_has_short_artifact_label_and_changes_only_loss() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    baseline, _ = load_config(directory / "basic_1cv_bounded_direct_optax_rprop_e300_balanced.py")
    changed, _ = load_config(directory / "basic_1cv_bounded_direct_optax_rprop_e300_balanced_huber_d5.py")
    baseline_values = baseline.describe()
    changed_values = changed.describe()

    baseline_values.pop("loss")
    changed_values.pop("loss")
    assert changed_values == baseline_values
    assert changed.artifact_label == "balanced-huber-d5"
    assert changed.loss.delta_mv == 5.0


def test_sgd_preset_changes_only_optimizer_stage() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    baseline, _ = load_config(directory / "basic_1cv_bounded_direct_optax_rprop_e300_balanced_huber_d5.py")
    changed, _ = load_config(directory / "basic_1cv_bounded_direct_sgd_e300_balanced_huber_d5.py")
    baseline_values = baseline.describe()
    changed_values = changed.describe()

    baseline_values.pop("stages")
    changed_values.pop("stages")
    assert changed_values == baseline_values
    assert changed.stages[0].describe()["optimizer"] == "SGD"
    assert changed.stages[0].learning_rate == 1e-4


def test_momentum_presets_differ_only_in_nesterov_flag() -> None:
    directory = Path(__file__).resolve().parent / "configs"
    momentum, _ = load_config(directory / "basic_1cv_bounded_direct_momentum_e300_balanced_huber_d5.py")
    nesterov, _ = load_config(directory / "basic_1cv_bounded_direct_nesterov_e300_balanced_huber_d5.py")
    momentum_values = momentum.describe()
    nesterov_values = nesterov.describe()

    momentum_stage = momentum_values.pop("stages")[0]
    nesterov_stage = nesterov_values.pop("stages")[0]
    assert momentum_values == nesterov_values
    momentum_stage["nesterov"] = True
    assert momentum_stage == nesterov_stage
