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

from datetime import datetime

from examples.experimental.optim_parameter_fitting.config import load_config
from examples.experimental.optim_parameter_fitting.run import DEFAULT_CONFIG, _parser, resolve_output_dir


def test_cli_uses_python_preset_and_explicit_gpu_without_batch_options() -> None:
    args = _parser().parse_args(["run", "--gpu", "1"])

    assert args.config == DEFAULT_CONFIG
    assert args.gpu == 1
    assert not hasattr(args, "start_batch_size")


def test_compare_cli_accepts_two_completed_result_directories() -> None:
    args = _parser().parse_args(["compare", "baseline", "extended"])

    assert str(args.baseline_dir) == "baseline"
    assert str(args.extended_dir) == "extended"


def test_default_result_directory_contains_time_label_and_short_digest() -> None:
    config_path = DEFAULT_CONFIG.parent / "basic_1cv_bounded_direct_optax_rprop_e300_balanced_huber_d5.py"
    config, _module = load_config(config_path)

    path = resolve_output_dir(
        config,
        config_path,
        requested=None,
        resume=False,
        now=datetime(2026, 9, 1, 12, 34, 56),
    )

    assert path.name == f"20260901-123456_balanced-huber-d5_{config.digest()[:8]}"
