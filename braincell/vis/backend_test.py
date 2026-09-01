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

"""Tests for :mod:`braincell.vis.backend` — scene-kind dispatch guards."""

import unittest
from unittest import mock

from braincell.vis import plot2d, plot3d
from braincell.vis._testing import (
    VisDefaultsResetMixin,
    make_node_tree,
)
from braincell.vis.backend_pyvista import PyVistaBackend


class BackendSceneKindTest(VisDefaultsResetMixin, unittest.TestCase):
    def test_plot2d_rejects_pyvista_backend(self) -> None:
        tree = make_node_tree()

        # Force PyVista to report as available so the dispatch reaches the
        # scene-kind validation step even when pyvista isn't installed.
        with mock.patch.object(PyVistaBackend, "available", return_value=True):
            with self.assertRaisesRegex(ValueError, "only supports 3D scenes"):
                plot2d(tree, backend="pyvista")

    def test_plot3d_rejects_matplotlib_backend(self) -> None:
        tree = make_node_tree()

        with self.assertRaisesRegex(ValueError, "only supports 2D scenes"):
            plot3d(tree, backend="matplotlib")


if __name__ == "__main__":
    unittest.main()
