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

import ast
import pathlib
import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np

from braincell._random import draw_seed


def drawn_seeds(seed: int, count: int) -> list[int]:
    """The first ``count`` seeds ``seed=None`` must draw under ``seed_context(seed)``, by the definition."""
    with brainstate.random.seed_context(seed):
        return [int(jax.random.bits(brainstate.random.split_key(), (), jnp.uint32)) & 0x7FFF_FFFF for _ in range(count)]


def default_key_data() -> np.ndarray:
    """The raw key of the BrainState default stream, to show a call left it alone."""
    return np.asarray(jax.random.key_data(brainstate.random.get_key()))


class DrawSeedTest(unittest.TestCase):
    """CELL-11a: one integer seed per call from the BrainState default stream."""

    def test_draws_follow_the_definition_and_advance_the_default_stream(self) -> None:
        for seed in (0, 42):
            with self.subTest(seed=seed):
                with brainstate.random.seed_context(seed):
                    drawn = [draw_seed("test"), draw_seed("test")]
                self.assertEqual(drawn, drawn_seeds(seed, 2))
                self.assertNotEqual(drawn[0], drawn[1])
                for value in drawn:
                    self.assertIs(type(value), int)
                    self.assertTrue(0 <= value < 2**31)

    def test_the_seed_does_not_depend_on_the_float_precision(self) -> None:
        drawn = []
        for precision in (32, 64):
            with brainstate.environ.context(precision=precision), brainstate.random.seed_context(42):
                drawn.append(draw_seed("test"))
        self.assertEqual(drawn[0], drawn[1])

    def test_outside_a_context_it_draws_from_the_global_stream(self) -> None:
        brainstate.random.seed(4)
        first = draw_seed("test")
        with brainstate.random.seed_context(4):
            expected = draw_seed("test")
        self.assertEqual(first, expected)

    def test_a_traced_default_stream_is_refused_and_left_usable(self) -> None:
        def in_loop(fn):
            return lambda: brainstate.transform.for_loop(lambda i: fn(), np.arange(2))

        for name, transform in (("jit", brainstate.transform.jit), ("for_loop", in_loop)):
            with self.subTest(transform=name):
                brainstate.random.seed(1)
                before = default_key_data()
                with self.assertRaisesRegex(TypeError, r"Network\(seed=None\).*traced"):
                    transform(lambda: draw_seed("Network"))()
                np.testing.assert_array_equal(default_key_data(), before)
                brainstate.random.seed(1)
                expected = float(brainstate.random.uniform())
                brainstate.random.seed(1)
                with self.assertRaises(TypeError):
                    transform(lambda: draw_seed("Network"))()
                self.assertEqual(float(brainstate.random.uniform()), expected)

    def test_the_module_imports_only_brainstate_and_jax(self) -> None:
        # ``braincell.network`` imports it while ``braincell`` is initializing.
        tree = ast.parse(pathlib.Path(__file__).with_name("_random.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"brainstate", "jax"})


if __name__ == "__main__":
    unittest.main()
