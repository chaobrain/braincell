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

"""Tests for :mod:`braincell.network.lowering`."""

import unittest

import brainunit as u
import numpy as np

from braincell.network import (
    Connection,
    Population,
    lower_connections,
)
from braincell.network._testing import (
    make_post_cell,
    make_spiking_cell,
)


class LoweringTest(unittest.TestCase):
    def test_lowering_validates_unknown_population(self) -> None:
        post = make_post_cell()
        post.init_state()
        populations = {"I": Population("I", post)}
        conn = Connection("E", "I", [0], [0], "exp")

        with self.assertRaisesRegex(KeyError, "Unknown pre_population"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_validates_index_range(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [2], [0], "exp")

        with self.assertRaisesRegex(IndexError, "pre_index out of range"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_requires_named_synapse_on_post_cell(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [0], "missing")

        with self.assertRaisesRegex(KeyError, "no placed synapse"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_converts_zero_delay_to_next_step(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.0 * u.ms)

        block = lower_connections(populations, (conn,), dt=0.1 * u.ms)[0]

        np.testing.assert_array_equal(block.delay_steps, [1])

    def test_lowering_quantizes_delay_with_ceil_floor_and_strict(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms)

        ceil_block = lower_connections(
            populations,
            (conn,),
            dt=0.1 * u.ms,
            delay_quantization="ceil",
        )[0]
        floor_block = lower_connections(
            populations,
            (conn,),
            dt=0.1 * u.ms,
            delay_quantization="floor",
        )[0]

        np.testing.assert_array_equal(ceil_block.delay_steps, [2])
        np.testing.assert_array_equal(floor_block.delay_steps, [1])
        strict_block = lower_connections(
            populations,
            (Connection("E", "I", [0], [1], "exp", delay=0.2 * u.ms),),
            dt=0.1 * u.ms,
            delay_quantization="strict",
        )[0]
        np.testing.assert_array_equal(strict_block.delay_steps, [2])
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            lower_connections(
                populations,
                (conn,),
                dt=0.1 * u.ms,
                delay_quantization="strict",
            )

    def test_lowering_zero_delay_is_next_step_for_all_quantization_modes(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.0 * u.ms)

        for mode in ("ceil", "floor", "strict"):
            with self.subTest(mode=mode):
                block = lower_connections(
                    populations,
                    (conn,),
                    dt=0.1 * u.ms,
                    delay_quantization=mode,
                )[0]
                np.testing.assert_array_equal(block.delay_steps, [1])

    def test_lowering_rejects_unknown_delay_quantization(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp")

        with self.assertRaisesRegex(ValueError, "delay_quantization"):
            lower_connections(
                populations,
                (conn,),
                dt=0.1 * u.ms,
                delay_quantization="nearest",
            )


if __name__ == "__main__":
    unittest.main()
