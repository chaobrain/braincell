"""Unit tests for helpers in :mod:`braincell._multi_compartment.run`.

The full :func:`run` loop is exercised end-to-end in
``runnable_test.py`` once ``Cell.build()`` lands.
"""

import unittest

import brainunit as u

from braincell._multi_compartment.run import (
    RunResult,
    _normalize_run_traces,
    _validate_time_quantity,
)


class TestValidateTimeQuantity(unittest.TestCase):
    def test_accepts_scalar_time(self):
        _validate_time_quantity(0.1 * u.ms, name="dt")

    def test_rejects_plain_float(self):
        with self.assertRaises(TypeError):
            _validate_time_quantity(0.1, name="dt")

    def test_rejects_zero(self):
        with self.assertRaises(ValueError):
            _validate_time_quantity(0.0 * u.ms, name="dt")

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            _validate_time_quantity(-0.1 * u.ms, name="dt")


class TestNormalizeRunTraces(unittest.TestCase):
    def test_single_trace_scalar_wraps(self):
        out = _normalize_run_traces("X", n_traces=1)
        self.assertEqual(out, ("X",))

    def test_single_trace_tuple_passes_through(self):
        out = _normalize_run_traces(("X",), n_traces=1)
        self.assertEqual(out, ("X",))

    def test_many_traces_tuple(self):
        out = _normalize_run_traces(("a", "b"), n_traces=2)
        self.assertEqual(out, ("a", "b"))

    def test_many_traces_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _normalize_run_traces(("a",), n_traces=2)

    def test_many_traces_non_tuple_raises(self):
        with self.assertRaises(TypeError):
            _normalize_run_traces("a", n_traces=2)


class TestRunResult(unittest.TestCase):
    def test_is_frozen_dataclass(self):
        result = RunResult(time=u.math.arange(0.0, 3.0, 1.0) * u.ms, traces={"v": 0})
        self.assertEqual(result.traces, {"v": 0})


if __name__ == "__main__":
    unittest.main()
