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

"""Verify the local nonlinear protocol and 72-root geometry registration."""

import numpy as np
from types import SimpleNamespace
import brainstate
import brainunit as u
import jax

from validation.optim.nonlinear_pattern_separation.runners.braincell.model import (
    RA_BOUNDS_OHM_CM,
    LENGTH_BOUNDS_UM,
    build_strict_experiment,
    export_physical_parameters,
    load_reference,
    parameter_count,
    REFERENCE,
)


def test_reference_loads_from_this_workflow_after_changing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reference = load_reference()
    assert reference.__file__ == str(REFERENCE)
    assert reference.N_CV == 12
    assert reference.READOUT_INDEX == 120


def test_strict_builder_registers_72_roots_and_keeps_cm_fixed():
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        _reference, cell, parameters = build_strict_experiment(seed=0)
        physical = parameters.physical_values()
        assert set(physical) == {"na.g_max", "k.g_max", "leak.g_max", "length", "radius_scale", "Ra"}
        assert all(value.size == 12 for value in physical.values())
        assert parameter_count(parameters) == 72
        assert cell.n_cv == 12


def test_jaxley_style_geometry_initialization_is_bounded_and_per_field():
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        _reference, cell, parameters = build_strict_experiment(seed=0, randomize_geometry=True)
        exported = export_physical_parameters(SimpleNamespace(cell=cell, parameters=parameters))
        for name, lower, upper in (
            ("length", LENGTH_BOUNDS_UM[0], LENGTH_BOUNDS_UM[1]),
            ("Ra", RA_BOUNDS_OHM_CM[0], RA_BOUNDS_OHM_CM[1]),
        ):
            values = exported[name]
            assert np.all((values >= lower) & (values <= upper))
            assert np.allclose(values, values[0])
        radius = exported["radius"]
        assert np.all((radius >= 0.1) & (radius <= 5.0))
        assert np.allclose(radius, radius[0])
