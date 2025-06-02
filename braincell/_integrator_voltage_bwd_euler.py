# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""
Implementation of the backward Euler integrator for voltage dynamics in multicompartment models.
"""

from ._misc import set_module_as


@set_module_as('braincell')
def stone_voltage_step():
    """
    Implicit euler solver with the Stone's algorithm.
    """
    pass


@set_module_as('braincell')
def thomas_voltage_step():
    """
    Implicit euler solver with the Thomas's algorithm.
    """
    pass


@set_module_as('braincell')
def dhs_voltage_step():
    """
    Implicit euler solver with the `dendritic hierarchical scheduling` (DHS, Zhang et al., 2023).
    """
    pass


@set_module_as('braincell')
def dense_voltage_step():
    """
    Implicit euler solver implementation by solving the dense matrix system.
    """
    pass


@set_module_as('braincell')
def sparse_voltage_step():
    """
    Implicit euler solver implementation by solving the sparse matrix system.
    """
    pass
