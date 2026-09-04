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

import numpy as np

from examples.experimental.optim_parameter_fitting.datasets import (
    DatasetBundle,
    Protocol,
    validate_bundle,
)


def _bundle() -> DatasetBundle:
    splits = ("train",) * 5 + ("validation",) * 2 + ("test",)
    protocols = tuple(
        Protocol(f"step_{index}", split, f"feature_{index}", float(index), index % 5)
        for index, split in enumerate(splits)
    )
    current = np.zeros((8, 4))
    current[:, 1:3] = np.arange(8)[:, None]
    return DatasetBundle(
        protocols,
        np.arange(4, dtype=np.float64),
        current,
        np.zeros((8, 4, 1)),
        1.0,
        1.0,
        3.0,
    )


def test_step_bundle_has_strict_five_two_one_splits_and_no_prmls() -> None:
    bundle = _bundle()

    validate_bundle(bundle)
    manifest = bundle.describe()

    assert manifest["split_counts"] == {"train": 5, "validation": 2, "test": 1}
    assert {item["family"] for item in manifest["protocols"]} == {"step"}
