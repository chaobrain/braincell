"""Unit tests for :mod:`braincell._multi_compartment.currents`."""

import unittest

import brainunit as u
import numpy as np


def _soma_cell(*, clamp=None):
    import braincell
    from braincell import Branch, CVPerBranch, Cell, Morphology
    from braincell.filter import at

    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
    if clamp is not None:
        cell.place(at("soma", 0.5), clamp)
    cell.init_state()
    return cell


class TotalMembraneCurrentClampTest(unittest.TestCase):
    def test_without_clamp_or_channels_returns_zero_density(self) -> None:
        import brainstate
        from braincell._multi_compartment.currents import total_membrane_current

        cell = _soma_cell()

        with brainstate.environ.context(t=0.0 * u.ms):
            current = total_membrane_current(
                cell,
                V_cv=cell.V.value,
                t=0.0 * u.ms,
            )

        np.testing.assert_allclose(
            np.asarray(current.to_decimal(u.nA / u.cm ** 2)),
            np.zeros(cell.runtime.n_cv),
        )

    def test_current_clamp_total_current_is_converted_to_density(self) -> None:
        import brainstate
        from braincell.mech import CurrentClamp
        from braincell._multi_compartment.currents import total_membrane_current

        cell = _soma_cell(
            clamp=CurrentClamp(delay=0.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.2 * u.nA)
        )

        with brainstate.environ.context(t=0.0 * u.ms):
            current = total_membrane_current(
                cell,
                V_cv=cell.V.value,
                t=0.0 * u.ms,
            )

        expected = np.asarray((0.2 * u.nA / cell.runtime.cv_area).to_decimal(u.nA / u.cm ** 2))
        np.testing.assert_allclose(
            np.asarray(current.to_decimal(u.nA / u.cm ** 2)),
            expected,
        )


class TotalMembraneCurrentNarrowExceptTest(unittest.TestCase):
    """MED-04: channel errors outside the numeric set must not be swallowed."""

    def test_non_numeric_exception_passes_through(self) -> None:
        import brainstate
        from braincell._base import IonChannel
        from braincell._multi_compartment.currents import total_membrane_current

        cell = _soma_cell()

        channels = cell.runtime_objects(IonChannel, allowed_hierarchy=(1, 1))
        self.assertGreater(len(channels), 0)

        first_channel = next(iter(channels.values()))

        def boom(V):
            raise AttributeError("lookup failed")

        first_channel.current = boom  # type: ignore[assignment]

        with brainstate.environ.context(t=0.0 * u.ms):
            with self.assertRaises(AttributeError):
                total_membrane_current(
                    cell,
                    V_cv=cell.V.value,
                    t=0.0 * u.ms,
                )


if __name__ == "__main__":
    unittest.main()
