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

"""Shared bounded 72-parameter BrainCell model."""

from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path
import sys
import json
import time
import numpy as np

import brainstate
import jax
import jax.numpy as jnp

from braincell import trainable
import brainunit as u

import braincell
from braincell.filter import AllRegion
from braincell import mech


WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = WORKFLOW_ROOT / "artifacts" / "nonlinear_pattern_separation_2026-09-15" / "raw" / "braincell"
REFERENCE = Path(__file__).resolve().parent / "task.py"
RADIUS_REFERENCE_UM = 2.55
RADIUS_SCALE_BOUNDS = (0.1 / RADIUS_REFERENCE_UM, 5.0 / RADIUS_REFERENCE_UM)
LENGTH_BOUNDS_UM = (1.0, 20.0)
RA_BOUNDS_OHM_CM = (500.0, 5500.0)


def load_reference():
    spec = importlib.util.spec_from_file_location("braincell_reference_nonlinear_pattern", REFERENCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference experiment from {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_strict_experiment(seed: int = 0):
    ref = load_reference()
    if ref.braincell is not braincell:
        raise RuntimeError(f"Reference imported a different braincell package: {ref.braincell.__file__}")
    random = brainstate.random.RandomState(seed + 1_000)
    cell = braincell.Cell(
        ref.build_morphology(),
        cv_policy=braincell.CVPerBranch(ref.N_CV_PER_BRANCH),
        pop_size=(1,),
        V_init=-70.0 * u.mV,
        solver="staggered",
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-70.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=3000.0 * u.ohm * u.cm,
        ),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", name="na"),
        mech.Channel("K_HH1952", name="k"),
        mech.Channel("IL", name="leak", E=-54.387 * u.mV),
    )
    channel_specs = []
    for channel_name in ("na", "k", "leak"):
        lower, upper = ref.PARAMETER_BOUNDS[f"{channel_name}.g_max"]
        sampled = random.uniform(lower, upper, dtype=jnp.float64)
        initial = jnp.full((ref.N_CV,), sampled, dtype=jnp.float64) * ref.CONDUCTANCE_UNIT
        channel_specs.append((channel_name, trainable.parameter(
                initial,
                group_by="cv",
                transform=brainstate.nn.SigmoidT(
                    lower * ref.CONDUCTANCE_UNIT,
                    upper * ref.CONDUCTANCE_UNIT,
                ),
                name=f"{channel_name}.g_max",
            )))

    # Channel registration follows the current channel lifecycle. Geometry is
    # a runtime-only registration and is added after the final discretization.
    cell.init_state()
    if not cell._initialized:
        raise RuntimeError("init_state returned without initializing the strict comparison cell")
    if cell.n_cv != ref.N_CV:
        raise RuntimeError(f"Expected {ref.N_CV} CVs, got {cell.n_cv}.")
    for channel_name, source in channel_specs:
        cell.channels[channel_name].trainable(g_max=source)

    # Jaxley strict comparison: three conductances plus three geometry fields
    # on each of 12 CVs. cm stays fixed at the reference value.
    # Jaxley bounds physical values.  BrainCell's optimizer roots may remain
    # transformed/unconstrained internally, but every materialized geometry
    # value must be in the same physical range.
    cell.geometry.length.trainable(trainable.parameter(
        cell.geometry.length.get(), group_by="cv",
        transform=brainstate.nn.SigmoidT(*[x * u.um for x in LENGTH_BOUNDS_UM]),
        name="length",
    ))
    cell.geometry.radius_scale.trainable(trainable.parameter(
        jnp.ones((ref.N_CV,), dtype=jnp.float64), group_by="cv",
        transform=brainstate.nn.SigmoidT(*RADIUS_SCALE_BOUNDS),
        name="radius_scale",
    ))
    cell.geometry.Ra.trainable(trainable.parameter(
        cell.geometry.Ra.get(), group_by="cv",
        transform=brainstate.nn.SigmoidT(*[x * u.ohm * u.cm for x in RA_BOUNDS_OHM_CM]),
        name="Ra",
    ))
    cell.reset_state()
    parameters = cell.trainables.parameters()
    return ref, cell, parameters


def parameter_count(parameters) -> int:
    return sum(int(value.size) for value in parameters.physical_values().values())


def export_physical_parameters(experiment):
    """Export the six comparison fields as physical per-CV arrays."""
    values = experiment.parameters.physical_values()
    exported = {
        "radius": np.asarray(experiment.cell.geometry.radius_mid.get().to_decimal(u.um)),
        "length": np.asarray(values["length"].to_decimal(u.um)),
        "Ra": np.asarray(values["Ra"].to_decimal(u.ohm * u.cm)),
        "gNa": np.asarray(values["na.g_max"].to_decimal(u.mS / u.cm**2)),
        "gK": np.asarray(values["k.g_max"].to_decimal(u.mS / u.cm**2)),
        "gLeak": np.asarray(values["leak.g_max"].to_decimal(u.mS / u.cm**2)),
    }
    return {name: np.asarray(value).reshape((-1,)) for name, value in exported.items()}
