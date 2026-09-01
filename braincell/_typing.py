# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import os
from typing import Union, Callable, Hashable, Tuple, Dict

import brainstate
import brainunit as u
import jax

ArrayLike = brainstate.typing.ArrayLike
Size = brainstate.typing.Size
PyTree = brainstate.typing.PyTree
Initializer = Union[ArrayLike, Callable]
SectionName = Hashable
T = u.Quantity[u.second]
DT = u.Quantity[u.second]
VectorField = Callable
Y0 = jax.Array
Y1 = jax.Array
Jacobian = jax.Array
Args = Tuple
Aux = Dict
Path = Tuple[str, ...]
# A filesystem location accepted by the readers, writers, and checkpoint I/O.
# Note this is distinct from ``Path`` above, which is a state path in a model
# tree, not a file on disk.
FilePath = Union[str, os.PathLike]
