"""Integration tests for :class:`RunnableCell`."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell import Branch, CVPerBranch, Cell, Morphology
from braincell.filter import RootLocation
from braincell.mech import StateProbe


def _simple_rcell():
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = Cell(
        Morphology.from_root(soma, name="soma"),
        cv_policy=CVPerBranch(),
    )
    cell.place(RootLocation(0.0), StateProbe(field="v", name="V_root"))
    return cell.build()


class TestRunnableCellBasic(unittest.TestCase):
    def test_has_required_state(self):
        rcell = _simple_rcell()
        self.assertTrue(hasattr(rcell, "V"))
        self.assertTrue(hasattr(rcell, "spike"))
        self.assertTrue(hasattr(rcell, "C"))
        self.assertTrue(hasattr(rcell, "V_th"))

    def test_update_advances_without_I_ext(self):
        rcell = _simple_rcell()
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            rcell.update()


class TestBug1ExternalCurrentNotDropped(unittest.TestCase):
    """Regression test for bug #1: I_ext silently dropped when no
    registered current_inputs exist. ``update(I_ext=...)`` must
    visibly move V."""

    def test_i_ext_propagates_through_update(self):
        rcell = _simple_rcell()
        before = np.asarray(rcell.V.value.to_decimal(u.mV)).copy()
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            rcell.update(I_ext=0.5 * u.nA)
        after = np.asarray(rcell.V.value.to_decimal(u.mV))
        self.assertFalse(np.array_equal(before, after))


class TestBug2NPointTotalCurrentRejected(unittest.TestCase):
    """Regression test for bug #2: total current shaped (n_point,)
    is ambiguous and must raise ValueError instead of producing
    NaN via divide-by-midpoint-area."""

    def test_n_point_total_current_rejected(self):
        rcell = _simple_rcell()
        bad = jnp.ones((rcell.n_point,)) * u.nA
        with brainstate.environ.context(dt=0.1 * u.ms, t=0.0 * u.ms):
            with self.assertRaises(ValueError):
                rcell.update(I_ext=bad)


if __name__ == "__main__":
    unittest.main()
