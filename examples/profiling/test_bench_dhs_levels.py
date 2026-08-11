from __future__ import annotations

import unittest

from examples.profiling.bench_dhs_levels import (
    _complete_binary_widths,
    make_problem,
    make_solver,
)


class TestBenchDHSLevels(unittest.TestCase):
    """Tests for the toy DHS level benchmark helpers."""

    def test_complete_binary_widths_uses_largest_tree_under_n_cv(self) -> None:
        self.assertEqual(_complete_binary_widths(1), (1,))
        self.assertEqual(_complete_binary_widths(16), (1, 2, 4, 8))
        self.assertEqual(_complete_binary_widths(1024), (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))

    def test_solver_runs_small_cpu_problem(self) -> None:
        import jax

        problem = make_problem(n_cv=8, popsize=2)
        solver = jax.jit(make_solver(problem.widths, popsize=2))

        out = solver(
            problem.diag_levels,
            problem.rhs_levels,
            problem.child_to_parent_levels,
            problem.parent_to_child_levels,
            1.0,
        )

        self.assertEqual(out.shape, ())
        self.assertTrue(bool(jax.numpy.isfinite(out)))


if __name__ == "__main__":
    unittest.main()
