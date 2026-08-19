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

"""Shared fixtures for the ``braincell._compute`` runtime test modules.

This module is deliberately not named ``*_test.py``: it holds helpers, not
tests, so pytest must not collect it. ``_RuntimeTestTwoOwnerChannel`` in
particular must be defined exactly once in the repository, because
``braincell.mech.register_channel`` raises ``ValueError`` when the same
channel name is registered twice.
"""

import brainstate
import brainunit as u
import numpy as np

from braincell import Branch, Morphology
from braincell._base import Channel, IonInfo
from braincell.ion import NonSpecific, Potassium
from braincell.mech import register_channel


@register_channel("_RuntimeTestTwoOwnerChannel")
class _RuntimeTestTwoOwnerChannel(Channel):
    """Small multi-owner channel used by runtime binding tests."""

    __module__ = "braincell._compute._testing"
    root_type = brainstate.mixin.JointTypes[Potassium, NonSpecific]
    current_owner_types = {"k": Potassium, "no": NonSpecific}

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)

    def current(self, V, K: IonInfo, No: IonInfo):
        parts = self.current_components(V, K, No)
        return parts["k"] + parts["no"]

    def current_components(self, V, K: IonInfo, No: IonInfo):
        _ = (K, No)
        return {
            "k": 2.0 * u.math.ones_like(V.to_decimal(u.mV)) * (u.nA / u.cm**2),
            "no": 3.0 * u.math.ones_like(V.to_decimal(u.mV)) * (u.nA / u.cm**2),
        }


def _build_tree() -> Morphology:
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def _quantity_set_at(value, index: int, replacement):
    decimal = np.array(value.to_decimal(value.unit), copy=True)
    decimal[..., int(index)] = replacement.to_decimal(value.unit)
    return u.Quantity(decimal, value.unit)
