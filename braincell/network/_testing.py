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

"""Shared fixtures for the :mod:`braincell.network` test modules.

The leading underscore keeps pytest from collecting this file as a test
module. Every builder here is used by more than one sibling ``*_test.py``.
The two toy solvers are deliberately trivial: network tests care about when
an event arrives, not about membrane dynamics, so a cell that walks its
voltage up or down by a fixed step per call makes spike timing exact.
"""

import brainunit as u

import braincell
from braincell import CVPerBranch, Cell, Morphology
from braincell.filter import at

__all__ = [
    "make_post_cell",
    "make_post_cell_with_synapse_pool",
    "make_probe_cell",
    "make_recording_post_cell",
    "make_runtime_network",
    "make_soma_tree",
    "make_spiking_cell",
    "make_threshold_cell",
    "make_two_point_tree",
    "step_down_solver",
    "step_up_solver",
]


def make_soma_tree() -> Morphology:
    """A single 20 um soma branch."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def make_two_point_tree() -> Morphology:
    """A soma with one tapering basal dendrite, giving two placement sites."""
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def step_up_solver(cell):
    """Drive the cell across threshold on every call."""
    cell.V.value = cell.V.value + 40.0 * u.mV


def step_down_solver(cell):
    """Drift the cell away from threshold on every call."""
    cell.V.value = cell.V.value - 1.0 * u.mV


def make_spiking_cell(size: int = 2) -> Cell:
    """A population that spikes on every step, with a voltage probe."""
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-10.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=step_up_solver,
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    return cell


def make_post_cell(size: int = 2) -> Cell:
    """A silent population carrying one named ``ExpSyn`` and a conductance probe."""
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=step_down_solver,
    )
    cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"))
    cell.place(
        at("soma", 0.5),
        braincell.mech.SynapseSpec(
            "ExpSyn",
            tau=2.0 * u.ms,
            e=0.0 * u.mV,
            weight=1.0 * u.uS,
            name="exp",
        ),
    )
    return cell


def make_post_cell_with_synapse_pool(size: int = 2) -> Cell:
    """Like :func:`make_post_cell`, but the synapse spans two placement sites."""
    cell = Cell(
        make_two_point_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=step_down_solver,
    )
    cell.place(
        at("soma", 0.5) | at(1, 0.5),
        braincell.mech.SynapseSpec(
            "ExpSyn",
            tau=2.0 * u.ms,
            e=0.0 * u.mV,
            weight=1.0 * u.uS,
            name="exp",
        ),
    )
    return cell


def make_threshold_cell(size: int = 2) -> Cell:
    """A population that crosses threshold on every step, with nothing placed.

    Unlike :func:`make_spiking_cell` this leaves the cell bare, so tests that
    only care about spike emission are not also exercising probe machinery.
    """
    return Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-10.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=step_up_solver,
    )


def make_recording_post_cell(size: int = 2) -> Cell:
    """A silent population with one ``ExpSyn`` observed through ``record``.

    This is the :class:`~braincell.mech.SynapseSpec` + ``observe`` counterpart
    to :func:`make_post_cell`, which predates that API and still declares its
    synapse with ``mech.SynapseSpec`` plus a ``MechanismProbe``.
    """
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=step_down_solver,
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
    )
    cell.soma.record("g", braincell.observe.synapse(name="exp").state("g"))
    return cell


def make_runtime_network(*, delay=0.0 * u.ms, weight=0.2 * u.uS):
    """A two-population network wired pre -> post through one ``ExpSyn``."""
    from braincell.network import Network

    network = Network("runtime")
    pre = network.add_population("pre", make_threshold_cell())
    post = network.add_population("post", make_recording_post_cell())
    network.connect(
        "drive",
        source=pre.event_outputs["spike"],
        synapse=post.synapses["exp"],
        weight=weight,
        delay=delay,
    )
    return network


def make_probe_cell(size: int = 2) -> Cell:
    """A plain population with a voltage probe and no solver override."""
    cell = Cell(
        make_soma_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    return cell
