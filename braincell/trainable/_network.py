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

"""Network aggregation preserving Cell-local optimizer root identity."""

from collections.abc import Mapping
from dataclasses import replace
import weakref

from braincell.trainable._parameters import ParameterSet


class _NetworkRoots(Mapping):
    def __init__(self, manager):
        self.manager = manager

    def __iter__(self):
        return iter(self.manager._collect()[0])

    def __len__(self):
        return len(self.manager._collect()[0])

    def __getitem__(self, key):
        return self.manager._collect()[0][key]


class NetworkTrainables:
    """Aggregate live Cell roots without introducing another parameter owner."""

    def __init__(self, network):
        self._network_ref = weakref.ref(network)
        self.roots = _NetworkRoots(self)

    def _cells(self):
        network = self._network_ref()
        if network is None:
            raise RuntimeError("Owning Network no longer exists.")
        return [
            (name, population.cell)
            for name, population in sorted(network.populations.items())
            if population.kind == "cell"
        ]

    def _collect(self):
        roots, names = {}, {}
        for population, cell in self._cells():
            for local_name, root in cell.trainables.roots.items():
                if id(root) not in names:
                    name = f"{population}.{local_name}"
                    if name in roots:
                        raise ValueError(f"Ambiguous qualified trainable name {name!r}.")
                    roots[name] = root
                    names[id(root)] = name
        return roots, names

    def parameters(self):
        """Return the optimizer-facing live parameter collection."""
        return ParameterSet(self.roots)

    def bindings(self):
        """Return population-qualified bindings to the original roots."""
        _, names = self._collect()
        result = []
        for population, cell in self._cells():
            for binding in cell.trainables.bindings():
                result.append(
                    replace(
                        binding,
                        name=f"{population}.{binding.name}",
                        target_owner=f"{population}.{binding.target_owner}",
                        root_names=tuple(names[id(cell.trainables.roots[name])] for name in binding.root_names),
                    )
                )
        return tuple(result)

    def materialize(self):
        """Refresh every initialized Cell's physical runtime parameters."""
        for _, cell in self._cells():
            if cell.trainables.bindings():
                cell.trainables.materialize()
